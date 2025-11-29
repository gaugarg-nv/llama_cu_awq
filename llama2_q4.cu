/*
Inference for Llama-2 Transformer model in pure Cuda.

### INT4 - AWQ quantization version ###

1. First generate AWQ int-4 quantized weights following steps in https://github.com/mit-han-lab/llm-awq
 E.g:
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-chat-metadata.pt
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-chat-metadata.pt --q_backend real --dump_quant awq_weights/llama2-7b-awq.pt
 Note - AWQ scripts doesn't run on Windows. Use Linux or WSL.

2. Convert AWQ weights into individual weight binary files using convert_awq_to_bin.py

3. Convert/repack the weight binary files using the weight_packer.cpp utility.

4. Run this program pointing to the final weight file.
*/

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <thread>

#include "common.h"
#include "gpu_kernels.h"
#include "tokenizer.h"
#include "sampler.h"
#include "perplexity.h"

constexpr int group_size = 128; // hardcoded for this implementation
#define DUMP_PER_TOKEN_TIMINGS 0
#define USE_CUDA_GRAPHS 0

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

void malloc_run_state(RunState* s, Config* p, bool allocLogitsArray) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->seq_len * sizeof(half));
    cudaMalloc((void**)&s->logits, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, sizeof(half) * p->n_layers * p->seq_len * kv_dim);    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, sizeof(half) * p->n_layers * p->seq_len * kv_dim);

    cudaMalloc((void**)&s->pos, sizeof(int));
    cudaMallocHost((void**)&s->shared_data, sizeof(SharedData));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->pos || !s->hb || !s->q
        || !s->att || !s->logits || !s->key_cache
        || !s->value_cache || !s->shared_data) {
        printf("malloc failed for allocaing run state!\n");
        exit(EXIT_FAILURE);
    }

    if (allocLogitsArray) {
        cudaMalloc((void**)&s->logits_array, sizeof(float) * p->seq_len * p->vocab_size);
        if (!s->logits_array) {
            printf("malloc failed for allocaing logits_array!\n");
            exit(EXIT_FAILURE);
        }
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->pos);
    cudaFree(s->hb);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    cudaFreeHost(s->shared_data);
}

size_t getPackedWeightHeight(size_t height)
{
    // Each uint32 element in the packed weight matrix contain 8 elements from the original matrix.
    // Also we load 4 uint's (32 elements) in a single instruction for getting better memory efficiency
    // This requires us to align the "height" dimension to a multiple of 4 uint (or 32 elements)
    return divUp(height, 32) * 4;
}

void custom_malloc(void** ptr, size_t size, bool allocOnGpu) {
    if(allocOnGpu)
        cudaMalloc(ptr, size);
    else
        *ptr = malloc(size);
}

void allocQWeight(QWeight* pWeight, size_t height, size_t width, bool allocOnGpu) {
    size_t packed_wt_height = getPackedWeightHeight(height);
    size_t scales_height = divUp(height, group_size);
    size_t packed_zeros_height = divUp(scales_height, 8);

    custom_malloc((void**)&pWeight->weight, packed_wt_height * width * sizeof(uint32_t), allocOnGpu);
    custom_malloc((void**)&pWeight->zeros, packed_zeros_height * width * sizeof(uint32_t), allocOnGpu);
    custom_malloc((void**)&pWeight->scales, scales_height * width * sizeof(half), allocOnGpu);
}

void custom_free(void* ptr, bool allocOnGpu) {
    if (allocOnGpu)
        cudaFree(ptr);
    else
        free(ptr);
}

void freeQWeight(QWeight* pWeight, bool allocOnGpu) {
    custom_free(pWeight->weight, allocOnGpu);
    custom_free(pWeight->zeros, allocOnGpu);
    custom_free(pWeight->scales, allocOnGpu);
}

void malloc_weights(TransformerWeights* w, Config* p, int numGPUs, bool allocOnGpu) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    custom_malloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half), allocOnGpu);
    w->layers = (PerLayerWeight*)malloc(p->n_layers * sizeof(PerLayerWeight));
    w->num_layers = p->n_layers;
    for (int l = 0; l < p->n_layers; l++)
    {
        PerLayerWeight* layer = &(w->layers[l]);
        custom_malloc((void**)&layer->rms_att_weight,  p->dim * sizeof(half), allocOnGpu);
        custom_malloc((void**)&layer->rms_ffn_weight,  p->dim * sizeof(half), allocOnGpu);
        allocQWeight(&layer->wq_q, p->dim, p->dim / numGPUs, allocOnGpu);
        allocQWeight(&layer->wq_k, p->dim, kv_dim / numGPUs, allocOnGpu);
        allocQWeight(&layer->wq_v, p->dim, kv_dim / numGPUs, allocOnGpu);
        allocQWeight(&layer->wq_o, p->dim, p->dim / numGPUs, allocOnGpu);
        allocQWeight(&layer->wq_gate, p->dim, p->hidden_dim / numGPUs, allocOnGpu);
        allocQWeight(&layer->wq_up, p->dim, p->hidden_dim / numGPUs, allocOnGpu);
        //allocQWeight(&layer->wq_down, p->hidden_dim / numGPUs, p->dim, allocOnGpu);
        allocQWeight(&layer->wq_down, p->hidden_dim, p->dim / numGPUs, allocOnGpu);
    }

    custom_malloc((void**)&w->rms_final_weight, p->dim * sizeof(half), allocOnGpu);
    custom_malloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half) / numGPUs, allocOnGpu);

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->layers ||
        !w->rms_final_weight || !w->wcls) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_weights(TransformerWeights* w, bool allocOnGpu) {
    custom_free(w->token_embedding_table, allocOnGpu);
    custom_free(w->rms_final_weight, allocOnGpu);
    custom_free(w->wcls, allocOnGpu);
    for (int l = 0; l < w->num_layers; l++) {
        PerLayerWeight* layer = &(w->layers[l]);
        custom_free(layer->rms_att_weight, allocOnGpu);
        custom_free(layer->rms_ffn_weight, allocOnGpu);
        freeQWeight(&layer->wq_q, allocOnGpu);
        freeQWeight(&layer->wq_k, allocOnGpu);
        freeQWeight(&layer->wq_v, allocOnGpu);
        freeQWeight(&layer->wq_o, allocOnGpu);
        freeQWeight(&layer->wq_gate, allocOnGpu);
        freeQWeight(&layer->wq_up, allocOnGpu);
        freeQWeight(&layer->wq_down, allocOnGpu);
    }
    free(w->layers);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint
void readWeight(void* op, FILE* fp, size_t bytes) {
    if (fread(op, 1, bytes, fp) != bytes) { 
        printf("error reading weights");  
        exit(EXIT_FAILURE); 
    }
}

void uploadQWeight(QWeight& weight, FILE* fp, size_t height, size_t width) {
    int meta_height = divUp(height, group_size);
    int packed_wt_height = getPackedWeightHeight(height);
    int packed_zeros_height = divUp(meta_height, 8);

    readWeight(weight.weight, fp, packed_wt_height * width * sizeof(uint32_t));
    readWeight(weight.zeros,  fp, packed_zeros_height * width * sizeof(uint32_t));
    readWeight(weight.scales, fp, meta_height * width * sizeof(half));
}

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    printf("\nLoading Weights... ");

    readWeight(w->token_embedding_table, f, p->vocab_size * p->dim * sizeof(half));
    readWeight(w->wcls, f, p->vocab_size * p->dim * sizeof(half));
    readWeight(w->rms_final_weight, f, p->dim * sizeof(half));

    // upload decoder block weight for each layer
    for (int i = 0; i < p->n_layers; i++) {
        uploadQWeight(w->layers[i].wq_q, f, p->dim, p->dim);
        uploadQWeight(w->layers[i].wq_k, f, p->dim, kv_dim);
        uploadQWeight(w->layers[i].wq_v, f, p->dim, kv_dim);
        uploadQWeight(w->layers[i].wq_o, f, p->dim, p->dim);

        uploadQWeight(w->layers[i].wq_up  , f, p->dim, p->hidden_dim);
        uploadQWeight(w->layers[i].wq_gate, f, p->dim, p->hidden_dim);
        uploadQWeight(w->layers[i].wq_down, f, p->hidden_dim, p->dim);

        readWeight(w->layers[i].rms_att_weight, f, p->dim * sizeof(half));
        readWeight(w->layers[i].rms_ffn_weight, f, p->dim * sizeof(half));
    }

    printf("done!\n");
    return 0;
}

void readWeight(void* op, void* src, size_t bytes) {
    cudaMemcpyAsync(op, src, bytes, cudaMemcpyHostToDevice);
}

void uploadQWeight(QWeight& weight, QWeight& srcWeight, size_t srcheight, size_t dstheight, size_t width) {
    int src_meta_height = divUp(srcheight, group_size);
    int src_packed_wt_height = getPackedWeightHeight(srcheight);
    int src_packed_zeros_height = divUp(src_meta_height, 8);

    int dst_meta_height = divUp(dstheight, group_size);
    int dst_packed_wt_height = getPackedWeightHeight(dstheight);
    int dst_packed_zeros_height = divUp(dst_meta_height, 8);

    cudaMemcpy2DAsync(weight.weight, dst_packed_wt_height * sizeof(uint32_t),
                        srcWeight.weight, src_packed_wt_height * sizeof(uint32_t),
                        dst_packed_wt_height * sizeof(uint32_t), width, cudaMemcpyHostToDevice);
}

enum Split
{
    ROW_WISE,
    COLUMN_WISE,
    NONE
};

void uploadQWeight(QWeight& weight, QWeight& srcWeight, size_t height, size_t width, Split split, int rank, int gpu_count) {

    if(split == COLUMN_WISE)
    {
        int meta_height = divUp(height, group_size);
        int packed_wt_height = getPackedWeightHeight(height);
        int packed_zeros_height = divUp(meta_height, 8);

        readWeight(weight.weight, srcWeight.weight + rank * (packed_wt_height * width / gpu_count), packed_wt_height * width * sizeof(uint32_t) / gpu_count);
        readWeight(weight.zeros,  srcWeight.zeros + rank * (packed_zeros_height * width / gpu_count), packed_zeros_height * width * sizeof(uint32_t) / gpu_count);
        readWeight(weight.scales, srcWeight.scales + rank * (meta_height * width / gpu_count), meta_height * width * sizeof(half) / gpu_count);
    }
    else
    {
        int meta_height = divUp(height, group_size);
        int packed_wt_height = getPackedWeightHeight(height);
        int packed_zeros_height = divUp(meta_height, 8);

        readWeight(weight.weight, srcWeight.weight, packed_wt_height * width * sizeof(uint32_t));
        readWeight(weight.zeros,  srcWeight.zeros, packed_zeros_height * width * sizeof(uint32_t));
        readWeight(weight.scales, srcWeight.scales, meta_height * width * sizeof(half));
    }
}


int checkpoint_init_weights(TransformerWeights* w, Config* p, TransformerWeights* w_src, int rank, int gpu_count) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    printf("\nLoading Weights... ");

    readWeight(w->token_embedding_table, w_src->token_embedding_table, p->vocab_size * p->dim * sizeof(half));
    readWeight(w->wcls, w_src->wcls + rank * (p->vocab_size * p->dim / gpu_count), p->vocab_size * p->dim * sizeof(half)/ gpu_count);
    readWeight(w->rms_final_weight, w_src->rms_final_weight, p->dim * sizeof(half));

    // upload decoder block weight for each layer
    for (int i = 0; i < p->n_layers; i++) {
        uploadQWeight(w->layers[i].wq_q, w_src->layers[i].wq_q, p->dim, p->dim, COLUMN_WISE, rank, gpu_count);
        uploadQWeight(w->layers[i].wq_k, w_src->layers[i].wq_k, p->dim, kv_dim, COLUMN_WISE, rank, gpu_count);
        uploadQWeight(w->layers[i].wq_v, w_src->layers[i].wq_v, p->dim, kv_dim, COLUMN_WISE, rank, gpu_count);
        uploadQWeight(w->layers[i].wq_o, w_src->layers[i].wq_o, p->dim, p->dim, COLUMN_WISE, rank, gpu_count);

        uploadQWeight(w->layers[i].wq_up  , w_src->layers[i].wq_up, p->dim, p->hidden_dim, COLUMN_WISE, rank, gpu_count);
        uploadQWeight(w->layers[i].wq_gate, w_src->layers[i].wq_gate, p->dim, p->hidden_dim, COLUMN_WISE, rank, gpu_count);
        uploadQWeight(w->layers[i].wq_down, w_src->layers[i].wq_down, p->hidden_dim, p->dim, COLUMN_WISE, rank, gpu_count);

        readWeight(w->layers[i].rms_att_weight, w_src->layers[i].rms_att_weight, p->dim * sizeof(half));
        readWeight(w->layers[i].rms_ffn_weight, w_src->layers[i].rms_ffn_weight, p->dim * sizeof(half));
    }

    printf("done!\n");
    return 0;
}


// ----------------------------------------------------------------------------
// neural net blocks
constexpr int MAX_GRAPHS = 8;
struct PerGPUExecutionData
{
    cudaStream_t stream;
    cudaGraphExec_t cudaGraphInstance[MAX_GRAPHS];
    bool graphCaptured[MAX_GRAPHS] = { false };
};

PerGPUExecutionData* streams;

void rmsnorm(int rank, half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<< 1, 1024, 0, streams[rank].stream>>> (o, x, weight, size, elementsPerThread);
}

void skip_connection(int rank, half* xout, half* x, half* skip, int n) {
    dim3 grid_dim(divUp(n, 256));
    skip_connection_kernel <<< grid_dim, 256, 0, streams[rank].stream>>> (xout, x, skip, n);
}

void matmul(int rank, half* xout, half* x, half* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    if ((n & 7) || (d & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;
    mat_vec_kernel <<<grid_dim, block_dim, 0, streams[rank].stream >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha);
}

void matmul(int rank, half* xout, half* x, QWeight &w, int inpSize, int opSize, bool accum = false, int loff = -1, int *pPos = nullptr) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    mat_vec_kernel_int4 <<<grid_dim, block_dim, 0, streams[rank].stream >>> (xout, x, w.weight, w.zeros, w.scales, inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, accum, loff, pPos);
}

void qkv_matvec(int rank, half* q, half *key_cache, half *value_cache, half* x, QWeight& qw, QWeight& kw, QWeight& vw, int inpSize, int opSize, int loff, int* pPos) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 3);
    qkv_matvec_kernel <<<grid_dim, block_dim, 0, streams[rank].stream >>> (q, key_cache, value_cache, x,
                                                             qw.weight, qw.zeros, qw.scales, 
                                                             kw.weight, kw.zeros, kw.scales, 
                                                             vw.weight, vw.zeros, vw.scales, 
                                                             inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, loff, pPos);
}

void ffn_matvec_silu(int rank, half* xout, half* x, QWeight& gate_w, QWeight& up_w, int inpSize, int opSize) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    ffn_matvec_silu_kernel <<<grid_dim, block_dim, 0, streams[rank].stream >>> (xout, x, gate_w.weight, gate_w.zeros, gate_w.scales, 
                                                                  up_w.weight, up_w.zeros, up_w.scales, 
                                                                  inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height);
}

void RoPERotation(int rank, half *q, half *k, int num_heads, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
    RoPERotation_kernel <<<num_heads, head_size / 2, 0, streams[rank].stream >>> (q, k, num_kv_heads, head_size, pPos, loff, rope_theta);
}

void MultiHeadAttention(int rank, half *output, half *q, half *key_cache, half * value_cache, half *att, int num_heads, int head_size, int kv_mul, int max_seq_len, int *pPos) {
    int dim = head_size * num_heads;
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(max_seq_len, 32), num_heads);      // using max_seq_len instead of real seq_len here has measurable impact on perf (2%) :-/
    mat_vec_kernel_simple <<< grid_dim1, block_dim, 0, streams[rank].stream >>> (att, q, key_cache, head_size, serialElements, head_size, head_size, dim / kv_mul, 1.0 / sqrt(head_size), pPos, kv_mul);

    // 2. Run softmax kernel
    if (max_seq_len <= MAX_SEQ_LEN_SMEM_KERNEL)
        softmax_kernel <<< num_heads, 1024, 0, streams[rank].stream >>> (att, num_heads, pPos);
    else
        softmax_kernel_no_smem <<< num_heads, 1024, 0, streams[rank].stream >>> (att, num_heads, pPos);

    // 3. weighted sum of the values to get the final result
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim, 0, streams[rank].stream >>> (output, att, value_cache, head_size, pPos, head_size, head_size, dim / kv_mul, kv_mul);
}

void run_llama_network(int rank, int gpu_count, const ncclComm_t& comm, int *pPos, Config* p, RunState* s, TransformerWeights* w, int seq_len_bin) {
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery

    copy_embedding_kernel <<<divUp(dim, 256), 256, 0, streams[rank].stream >>> (x, w->token_embedding_table, dim, s->shared_data->tokens, pPos);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(rank, s->xb, x, w->layers[l].rms_att_weight, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim / gpu_count; // kv cache layer offset for convenience

        // qkv matmuls for this position (opt: can be done in single kernel as batch of 3 - but only when num_kv_heads == num_heads)
        if (dim == kv_dim) {
            qkv_matvec(rank, s->q, s->key_cache, s->value_cache, s->xb, w->layers[l].wq_q, w->layers[l].wq_k, w->layers[l].wq_v, dim, dim / gpu_count, loff, pPos);
        }
        else {
            matmul(rank, s->q, s->xb, w->layers[l].wq_q, dim, dim / gpu_count);
            matmul(rank, s->key_cache, s->xb, w->layers[l].wq_k, dim, kv_dim / gpu_count, false, loff, pPos);
            matmul(rank, s->value_cache, s->xb, w->layers[l].wq_v, dim, kv_dim / gpu_count, false, loff, pPos);
        }

        // apply RoPE rotation to the q and k vectors for each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(rank, s->q, s->key_cache, p->n_heads / gpu_count, p->n_kv_heads / gpu_count, head_size, pPos, loff, p->rope_theta);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(rank, s->xb + rank * (dim / gpu_count), s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads / gpu_count, head_size, kv_mul, seq_len_bin, pPos);

        if(gpu_count > 1)
            ncclAllGather(s->xb + rank * (dim / gpu_count), s->xb, dim / gpu_count, ncclHalf, comm, streams[rank].stream);

        // final matmul to get the output of the attention fused with residual connection back into x
        matmul(rank, s->x + rank * (dim / gpu_count), s->xb, w->layers[l].wq_o, dim, dim / gpu_count, true);
        if(gpu_count > 1)
            ncclAllGather(s->x + rank * (dim / gpu_count), s->x, dim / gpu_count, ncclHalf, comm, streams[rank].stream);
        // ffn rmsnorm
        rmsnorm(rank, s->xb, x, w->layers[l].rms_ffn_weight, dim);
        // apply gate proj and up proj and then the silu activation in a single fused kernel
        ffn_matvec_silu(rank, s->hb + rank * (hidden_dim / gpu_count), s->xb, w->layers[l].wq_gate, w->layers[l].wq_up, dim, hidden_dim / gpu_count);

        if(gpu_count > 1)
            ncclAllGather(s->hb + rank * (hidden_dim / gpu_count), s->hb, hidden_dim / gpu_count, ncclHalf, comm, streams[rank].stream);

        // final matmul (down proj) to get the output of the ffn fused with residual connection back into x
        //matmul(rank, s->x, s->hb, w->layers[l].wq_down, hidden_dim, dim, true);
        // matmul(rank, s->xb, s->hb, w->layers[l].wq_down, hidden_dim / gpu_count, dim, false);
        // ncclAllReduce(s->xb, s->xb, dim, ncclHalf, ncclSum, comm, streams[rank].stream);

        matmul(rank, s->x + rank * (dim / gpu_count), s->hb, w->layers[l].wq_down, hidden_dim, dim / gpu_count, true);
        if(gpu_count > 1)
            ncclAllGather(s->x + rank * (dim / gpu_count), s->x, dim / gpu_count, ncclHalf, comm, streams[rank].stream);

        //skip_connection(rank, x, s->xb, x, dim);
    }

    // final rmsnorm
    rmsnorm(rank, x, x, w->rms_final_weight, dim);
    // classifier into logits
    matmul(rank, s->logits + rank * (p->vocab_size / gpu_count), x, w->wcls, p->dim, p->vocab_size / gpu_count);

    if(gpu_count > 1)
        ncclAllGather(s->logits + rank * (p->vocab_size / gpu_count), s->logits, p->vocab_size / gpu_count, ncclHalf, comm, streams[rank].stream);
}

void run_transformer(int rank, int gpu_count, const ncclComm_t& comm, bool gen_token, Config* p, RunState* s, TransformerWeights* w, bool copyLogits, Sampler *pSampler) {
#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, streams[rank].stream);
#endif

    int seq_len = s->shared_data->pos + 1;
#if USE_CUDA_GRAPHS
    int graphIndex;
    int seq_len_bin = 128;
    for (graphIndex = 0; graphIndex < MAX_GRAPHS - 1; seq_len_bin *= 2, graphIndex++)
        if (seq_len <= seq_len_bin) break;
    if ((seq_len > seq_len_bin) || (graphIndex == MAX_GRAPHS - 1)) seq_len_bin = p->seq_len;    // last bin holds max seq len

    if (!streams[rank].graphCaptured[graphIndex])
    {
        cudaGraph_t graph = {};
        CUDA_CHECK(cudaStreamBeginCapture(streams[rank].stream, cudaStreamCaptureModeThreadLocal));
        run_llama_network(rank, gpu_count, comm, s->pos, p, s, w, seq_len_bin);
        CUDA_CHECK(cudaStreamEndCapture(streams[rank].stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&streams[rank].cudaGraphInstance[graphIndex], graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        streams[rank].graphCaptured[graphIndex] = true;
    }
    cudaGraphLaunch(streams[rank].cudaGraphInstance[graphIndex], streams[rank].stream);
#else
    run_llama_network(rank, gpu_count, comm, s->pos, p, s, w, seq_len);
#endif

    if (copyLogits) {
        // copy to the right slot in logits_array (and convert to FP32)
        // we compute perplexity on the CPU later.
        float* pOutput = s->logits_array + p->vocab_size * s->shared_data->pos;
        convert_fp16_to_fp32 << < divUp(p->vocab_size, 128), 128, 0, streams[rank].stream >> > (pOutput, s->logits, p->vocab_size);
    }

    sample(rank, pSampler, s, gen_token, streams[rank].stream);

#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEventRecord(stop, streams[rank].stream);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf(" t: %g ", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void init_transformer(Transformer* t, bool perplexity, int rank, int gpu_count)
{
    cudaSetDevice(rank);
    // read in the Transformer weights
    malloc_weights(&(t->pInfo[rank].d_weights), &t->config, gpu_count, true);
    if (checkpoint_init_weights(&(t->pInfo[rank].d_weights), &t->config, &t->h_weights, rank, gpu_count)) { exit(1); }

    malloc_run_state(&(t->pInfo[rank].state), &t->config, perplexity);

    cudaDeviceSynchronize();
}

void build_transformer(Transformer* t, int gpu_count, char* checkpoint_path, bool perplexity) {
    // read in the model.bin file
    FILE* file = nullptr;
    file = fopen(checkpoint_path, "rb");
    if (!file) { printf("Couldn't open file %s\n", checkpoint_path); exit(1); }
    // read in the config header
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { printf("Invalid header size\n");  exit(1); }
    // Dump model config
    printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\nrope_theta: %g\n",
        t->config.dim, t->config.hidden_dim, t->config.n_heads, t->config.n_kv_heads, t->config.n_layers, t->config.seq_len, t->config.vocab_size, t->config.rope_theta);

    t->gpu_count = gpu_count;
    t->pInfo = new PerThreadInfo[gpu_count];
    ncclGetUniqueId(&t->ncclId);

    // read in the Transformer weights
    malloc_weights(&(t->h_weights), &t->config, 1, false);
    if (checkpoint_init_weights(&(t->h_weights), &t->config, file)) { exit(1); }

    std::vector<std::thread> threads;
    for (int i = 0; i < gpu_count; i++) {
        threads.push_back(std::thread(init_transformer, t, perplexity, i, gpu_count));
    }

    for (auto& t : threads) {
        t.join();
    }
    
    free_weights(&(t->h_weights), false);
    fclose(file);
}

void free_transformer_thread(Transformer* t, int rank) {
    cudaSetDevice(rank);
    free_run_state(&(t->pInfo[rank].state));
    free_weights(&(t->pInfo[rank].d_weights), true);
    cudaDeviceSynchronize();
}

void free_transformer(Transformer* t) {
    // free the RunState buffers

    std::vector<std::thread> threads;
    for (int i = 0; i < t->gpu_count; i++) {
        threads.push_back(std::thread(free_transformer_thread, t, i));
    }
    for (auto& t : threads) {
        t.join();
    }
}

// ----------------------------------------------------------------------------
// generation loop

void generate_thread(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, int* prompt_tokens, int num_prompt_tokens, int steps, int rank, int gpu_count) {

    cudaSetDevice(rank);

    ncclComm_t comm;
    ncclCommInitRank(&comm, gpu_count, transformer->ncclId, rank);

    cudaStreamCreate(&streams[rank].stream);

    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence

    // copy the prompt tokens into shared list of tokens (so that GPU can access them).
    // init state
    cudaMemset(transformer->pInfo[rank].state.pos, 0, sizeof(int));
    transformer->pInfo[rank].state.shared_data->pos = 0;
    memcpy(&transformer->pInfo[rank].state.shared_data->tokens, prompt_tokens, sizeof(int) * num_prompt_tokens);

    // start the main loop
    long start = time_in_ms();    // used to time our code

    while (pos < steps) {
        // wait for GPU work for previous iteration to complete
        // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
        cudaStreamSynchronize(streams[rank].stream);
        // Perf note: don't put CPU work here "before" calling transformer as it won't overlap with GPU execution.
        run_transformer(rank, gpu_count, comm, pos >= num_prompt_tokens - 1, &transformer->config, &transformer->pInfo[rank].state, &transformer->pInfo[rank].d_weights, false, sampler); // forward the transformer to get next token

        if (pos > 0) {
            next = transformer->pInfo[rank].state.shared_data->tokens[pos];  // Note: this is output token from previous iteration
            if (next >= transformer->config.vocab_size) next = 0;   // skip garbage tokens (can happen with NANs)

            if(rank == 0)
            {
                char* piece = decode(tokenizer, token, next);
                safe_printf(piece);             // same as printf("%s", piece), but skips "unsafe" bytes
            }
            if (next == eos_token) break;   // break if EOS token is reached
            // advance forward
            token = next;
        }
        pos++;
    }

    cudaStreamSynchronize(streams[rank].stream);

    // report achieved tok/s
    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    int timed_tokens = pos - 1;

    printf("\n");
    printf("\n[GPU: %d] achieved tok/s: %f. Tokens: %d, seconds: %g\n", rank, timed_tokens / time, timed_tokens, time);

#if USE_CUDA_GRAPHS
    for (int i = 0; i < MAX_GRAPHS; i++)
        if (streams[rank].graphCaptured[i]) cudaGraphExecDestroy(streams[rank].cudaGraphInstance[i]);
#endif

    ncclCommDestroy(comm);
}

void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps, int gpu_count) {
    char empty_prompt[] = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    printf("\nEncoding Prompt... ");   // Encoding can take a long time, print a message to show progress
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("Done!\n");

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < gpu_count; i++) {
        threads.push_back(std::thread(generate_thread, transformer, tokenizer, sampler, prompt_tokens, num_prompt_tokens, steps, i, gpu_count));
    }

    for (auto& t : threads) {
        t.join();
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
    char* cli_user_prompt, char* cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence

    int rank = 0;
    int gpu_count = 1;
    ncclComm_t comm;

    // init GPU state
    cudaMemset(transformer->pInfo[rank].state.pos, pos, sizeof(int));
    transformer->pInfo[rank].state.shared_data->pos = pos;

    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                }
                else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            }
            else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            }
            else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }

            printf("\nRendered prompt: %s\n", rendered_prompt); // Ankan - test!

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");

            // copy encoded tokens to GPU
            memcpy(&transformer->pInfo[rank].state.shared_data->tokens[pos], prompt_tokens, sizeof(int) * num_prompt_tokens);
        }

        // wait for GPU work for previous iteration to complete
        // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
        cudaStreamSynchronize(streams[rank].stream);
        run_transformer(rank, gpu_count, comm, user_idx >= num_prompt_tokens - 1, &transformer->config, &transformer->pInfo[rank].state, &transformer->pInfo[rank].d_weights, false, sampler); // forward the transformer to get next token

        user_idx++;

        if (user_idx > 0) {
            next = transformer->pInfo[rank].state.shared_data->tokens[pos];  // Note: this is output token from previous iteration
            if (next == eos_token) { 
                user_turn = 1;  // EOS token ends the Assistant turn
                printf("\n");
            } 
            else if (user_idx > num_prompt_tokens) {
                char* piece = decode(tokenizer, token, next);
                safe_printf(piece);
            }

            // advance forward
            token = next;
        }
        pos++;
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
void error_usage(char *argv[]) {
    fprintf(stderr, "Usage:   %s <checkpoint> [options]\n", argv[0]);
    fprintf(stderr, "Example: %s model.bin -n 256 -i \"Write a poem on GPUs\"\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <int>    max number of steps to run for, default = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -f <string> path to file containing input prompt. Can be used with for multi-line prompts.\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.5\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|perplexity, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -q <string> dataset file for computing perplexity\n");
    exit(EXIT_FAILURE);
}

int get_and_print_gpu_info()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        exit(EXIT_FAILURE);
    }

    printf("Found %d CUDA devices\n", device_count);
    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);
        printf("Device %d: %s\n", dev, device_prop.name);
        printf("  Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("  Total global memory: %.2f GB\n", (float)device_prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Multiprocessors: %d\n", device_prop.multiProcessorCount);
        printf("\n");
    }
    return device_count;
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {

    int gpu_count = get_and_print_gpu_info();

    // default parameters
    char* checkpoint_path = NULL;  // e.g. out/model.bin
    char default_tokenizer_path[] = "tokenizer.bin";
    char* tokenizer_path = default_tokenizer_path;
    char* dataset_path = NULL;
    int steps = 0;              // number of steps to run for
    char* prompt = nullptr;     // prompt string
    bool perplexity = false;
    float temperature = 0.5f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.6f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    unsigned long long rng_seed = 0; // seed rng with time by default
    char default_mode[] = "generate";
    char* mode = default_mode;  // generate|chat
    char* system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(argv); }

    for (int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(argv); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(argv); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(argv); } // must be -x (one dash, one letter)
        // read in the args
        switch (argv[i][1]) {
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': rng_seed = atoi(argv[i + 1]); break;
            case 'm': mode = argv[i + 1]; break;
            case 'y': system_prompt = argv[i + 1]; break;
            case 'q': {
                dataset_path = argv[i + 1];
                break;
            }
            case 'f': {
                FILE* file = fopen(argv[i + 1], "r");
                if (!file) { printf("Couldn't open file %s\n", argv[i + 1]); exit(1); }
                fseek(file, 0, SEEK_END);
                long fsize = ftell(file);
                fseek(file, 0, SEEK_SET);
                if (prompt) { printf("Warning: -f overrides -i\n"); }
                prompt = (char*)malloc(fsize + 1);
                fread(prompt, fsize, 1, file);
                fclose(file);
                prompt[fsize] = 0;
                break;
            }
            default: error_usage(argv);
        }
    }

    if (strcmp(mode, "perplexity") == 0) perplexity = true;

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (!perplexity && dataset_path)
        printf("Warning: dataset path is ignored in non-perplexity mode\n");

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, gpu_count, checkpoint_path, perplexity);
    if (steps <= 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // create and init the tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed, gpu_count);

    streams = new PerGPUExecutionData[gpu_count];

    if (perplexity)
        parseDataSetAndComputePreplexity(dataset_path, &tokenizer, &transformer.config, &transformer.pInfo[0].state, &transformer.pInfo[0].d_weights, &sampler);
    else if (strcmp(mode, "generate") == 0)
        generate(&transformer, &tokenizer, &sampler, prompt, steps, gpu_count);
    else if (strcmp(mode, "chat") == 0)
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    else
        error_usage(argv);

    // memory cleanup
    free_transformer(&transformer);

    free_tokenizer(&tokenizer);
    return 0;
}