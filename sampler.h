#pragma once

struct PerGPUMemory
{
    int* indices = nullptr;
    void* tempStorage_scan = nullptr;
    void* tempStorage_sort = nullptr;
};

struct Sampler{
    int vocab_size = 0;
    size_t temp_storage_bytes_scan = 0;
    size_t temp_storage_bytes_sort = 0;
    float temperature = 0;
    float topp = 0;
    unsigned long long rng_state = 0;
    PerGPUMemory* pMem = nullptr;
    int gpu_count = 0;
};

void init_sampler(Sampler* sampler, int gpu_id) {
    cudaSetDevice(gpu_id);
    // buffer only used with nucleus sampling
    cudaMalloc((void**) & sampler->pMem[gpu_id].indices, sampler->vocab_size * sizeof(int));
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed, int gpu_count) {

    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->gpu_count = gpu_count;
    sampler->pMem = new PerGPUMemory[gpu_count];

    std::vector<std::thread> threads;
    for (int i = 0; i < gpu_count; i++) {
        threads.push_back(std::thread(init_sampler, sampler, i));
    }

    for (auto& t : threads) {
        t.join();
    }
}

void destroy_sampler_thread(Sampler* sampler, int gpu_id) {
    cudaSetDevice(gpu_id);
    cudaFree(sampler->pMem[gpu_id].indices);
    cudaFree(sampler->pMem[gpu_id].tempStorage_sort);
    cudaFree(sampler->pMem[gpu_id].tempStorage_scan);
}

void destroy_sampler(Sampler* sampler) {

    std::vector<std::thread> threads;
    for (int i = 0; i < sampler->gpu_count; i++) {
        threads.push_back(std::thread(destroy_sampler_thread, sampler, i));
    }

    for (auto& t : threads) {
        t.join();
    }
}

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

// sample the token given the logits and some hyperparameters
void sample(int rank, Sampler* sampler, RunState* s, bool gen_token, cudaStream_t stream) {
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);

    if (sampler->temperature == 0.0f || !gen_token) {
        // greedy argmax sampling: take the token with the highest probability
        argmax_kernel << <1, 1024, 0, stream >> > (s->logits, sampler->vocab_size, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos, gen_token);
    }
    else {
        // apply the temperature to the logits, and then perform softmax
        softmax_logits_kernel <<<1, 1024, 0, stream >>> (s->logits, sampler->vocab_size, sampler->temperature, sampler->pMem[rank].indices);

        float threshold = 0.0f;
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            threshold = coin;
        }
        else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            if (sampler->temp_storage_bytes_sort == 0) {
                cub::DeviceRadixSort::SortPairsDescending(sampler->pMem[rank].tempStorage_sort, sampler->temp_storage_bytes_sort, s->logits, s->logits, sampler->pMem[rank].indices, sampler->pMem[rank].indices,
                    sampler->vocab_size, 0, sizeof(half) * 8, stream);
                cudaMalloc(&sampler->pMem[rank].tempStorage_sort, sampler->temp_storage_bytes_sort);
            }

            cub::DeviceRadixSort::SortPairsDescending(sampler->pMem[rank].tempStorage_sort, sampler->temp_storage_bytes_sort, s->logits, s->logits, sampler->pMem[rank].indices, sampler->pMem[rank].indices, 
                sampler->vocab_size, 0, sizeof(half) * 8, stream);
            threshold = coin * sampler->topp;
        }

        // Sample from the predicted probability distribution
        if (sampler->temp_storage_bytes_scan == 0) {
            cub::DeviceScan::InclusiveSum(sampler->pMem[rank].tempStorage_scan, sampler->temp_storage_bytes_scan, s->logits, s->logits, sampler->vocab_size, stream);
            cudaMalloc(&sampler->pMem[rank].tempStorage_scan, sampler->temp_storage_bytes_scan);
        }
        cub::DeviceScan::InclusiveSum(sampler->pMem[rank].tempStorage_scan, sampler->temp_storage_bytes_scan, s->logits, s->logits, sampler->vocab_size, stream);

        sample_top_p_kernel << <1, 1024, 0, stream >> > (s->logits, sampler->pMem[rank].indices, sampler->vocab_size, threshold, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos);
    }
}