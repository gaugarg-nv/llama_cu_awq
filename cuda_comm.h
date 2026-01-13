#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <barrier>

// Native CUDA multi-GPU communication implementation
// Replaces NCCL functionality with peer-to-peer memory access and synchronization

constexpr int MAX_GPUS = 8;

// Simple C++ barrier implementation (C++20 std::barrier alternative)
class CudaBarrier {
private:
    std::barrier<>* barrier_;
public:
    explicit CudaBarrier(int count) {
            barrier_ = new std::barrier<>(count);
    }

    ~CudaBarrier() {
        delete barrier_;
    }
    
    void wait() {
        barrier_->arrive_and_wait();
    }
};

struct CudaCommId {
    int magic;  // Simple identifier
    CudaBarrier* barrier;  // For synchronization across threads
    void* shared_buffers[MAX_GPUS];  // Shared memory buffers for each GPU
    cudaEvent_t events[MAX_GPUS];  // CUDA events for async synchronization
};

struct CudaComm {
    int rank;
    int gpu_count;
    CudaCommId* comm_id;
    void* local_buffer;   // Local buffer for this GPU
    void* staging_buffer; // Staging buffer for copying peer data
    size_t buffer_size;
    cudaStream_t stream;
    bool p2p_access[MAX_GPUS]; // Track which GPUs have P2P access
};

// Error checking macro
#define CUDA_COMM_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Enum for data types
enum CudaDataType {
    CUDA_HALF,
    CUDA_FLOAT,
    CUDA_INT
};

// Enum for reduction operations
enum CudaReduceOp {
    CUDA_SUM,
    CUDA_MAX,
    CUDA_MIN
};

// Initialize unique ID for communication
inline void cudaCommGetUniqueId(CudaCommId* id) {
    static int magic_counter = 0;
    id->magic = ++magic_counter;
    id->barrier = new CudaBarrier(1);  // Will be reinitialized with correct count
    for (int i = 0; i < MAX_GPUS; i++) {
        id->shared_buffers[i] = nullptr;
        id->events[i] = nullptr;
    }
}

// Initialize communicator for a specific rank
inline void cudaCommInitRank(CudaComm* comm, int gpu_count, CudaCommId* id, int rank) {
    comm->rank = rank;
    comm->gpu_count = gpu_count;
    comm->comm_id = id;
    
    // Reinitialize barrier with correct GPU count (only once)
    if (rank == 0) {
        delete id->barrier;
        id->barrier = new CudaBarrier(gpu_count);
    }
    
    // Allocate communication buffer (128 MB should be enough for most ops)
    comm->buffer_size = 128 * 1024 * 1024;
    CUDA_COMM_CHECK(cudaMalloc(&comm->local_buffer, comm->buffer_size));
    CUDA_COMM_CHECK(cudaMalloc(&comm->staging_buffer, comm->buffer_size));
    id->shared_buffers[rank] = comm->local_buffer;
    
    // Create CUDA event for this rank
    CUDA_COMM_CHECK(cudaEventCreateWithFlags(&id->events[rank], cudaEventDisableTiming));
    
    // Enable peer access to all other GPUs
    for (int i = 0; i < gpu_count; i++) {
        comm->p2p_access[i] = false;
#if 1
        if (i != rank) {
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, rank, i);
            if (can_access) {
                cudaSetDevice(rank);
                cudaError_t err = cudaDeviceEnablePeerAccess(i, 0);
                // May fail if already enabled, which is fine
                if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                    comm->p2p_access[i] = true;
                    std::cout << "GPU " << rank << " can access GPU " << i << std::endl;
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError(); // Clear the error
                    }
                }
                else
                {
                    std::cout << "GPU " << rank << " cannot access GPU " << i << std::endl;
                }
            }
            else
            {
                std::cout << "GPU " << rank << " cannot access GPU " << i << std::endl;
            }
        }
#endif
    }
}

// Kernel for half-precision sum reduction
__global__ void reduce_sum_half_kernel(half* output, const half* input, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = __hadd(output[idx], input[idx]);
    }
}

// Kernel for copying data
__global__ void copy_kernel_half(half* dst, const half* src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

// AllReduce: Reduce data across all GPUs with specified operation
inline void cudaAllReduce(const void* sendbuff, void* recvbuff, size_t count, 
                         CudaDataType datatype, CudaReduceOp op, 
                         CudaComm* comm, cudaStream_t stream) {
    if (comm->gpu_count == 1) {
        // Single GPU case - just copy
        if (sendbuff != recvbuff) {
            size_t bytes = count * sizeof(half);
            CUDA_COMM_CHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }
    
    // Multi-GPU case
    size_t element_size = (datatype == CUDA_HALF) ? sizeof(half) : 
                          (datatype == CUDA_FLOAT) ? sizeof(float) : sizeof(int);
    size_t bytes = count * element_size;
    
    // Copy local data to communication buffer
    CUDA_COMM_CHECK(cudaMemcpyAsync(comm->local_buffer, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    
    // Record event after copy completes
    CUDA_COMM_CHECK(cudaEventRecord(comm->comm_id->events[comm->rank], stream));
    
    // Barrier to ensure all GPUs have recorded their events
    comm->comm_id->barrier->wait();
    
    // Wait for all other GPUs' data to be ready
    for (int i = 0; i < comm->gpu_count; i++) {
        if (i != comm->rank) {
            CUDA_COMM_CHECK(cudaStreamWaitEvent(stream, comm->comm_id->events[i], 0));
        }
    }
    
    // Copy result to output first
    if (sendbuff != recvbuff) {
        CUDA_COMM_CHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    }
    
    // Perform reduction by reading from all other GPUs
    for (int i = 0; i < comm->gpu_count; i++) {
        if (i != comm->rank) {
            void* peer_buffer = comm->comm_id->shared_buffers[i];
            
            if (datatype == CUDA_HALF && op == CUDA_SUM) {
                int blocks = (count + 255) / 256;
                
                if (comm->p2p_access[i]) {
                    // P2P access available - directly launch kernel with peer buffer
                    reduce_sum_half_kernel<<<blocks, 256, 0, stream>>>(
                        (half*)recvbuff, (const half*)peer_buffer, count);
                } else {
                    // No P2P - copy to staging buffer first
                    CUDA_COMM_CHECK(cudaMemcpyPeerAsync(comm->staging_buffer, comm->rank, 
                                                         peer_buffer, i, bytes, stream));
                    reduce_sum_half_kernel<<<blocks, 256, 0, stream>>>(
                        (half*)recvbuff, (const half*)comm->staging_buffer, count);
                }
            }
        }
    }
    
    // Barrier to ensure all threads reach here before starting next operation
    // Stream ordering guarantees GPU operations complete before next copy
    comm->comm_id->barrier->wait();
}

// AllGather: Gather data from all GPUs
inline void cudaAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                         CudaDataType datatype, CudaComm* comm, cudaStream_t stream) {
    if (comm->gpu_count == 1) {
        // Single GPU case - just copy
        if (sendbuff != recvbuff) {
            size_t bytes = sendcount * sizeof(half);
            CUDA_COMM_CHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }
    
    size_t element_size = (datatype == CUDA_HALF) ? sizeof(half) : 
                          (datatype == CUDA_FLOAT) ? sizeof(float) : sizeof(int);
    size_t bytes = sendcount * element_size;
    
    // Copy local data to communication buffer
    CUDA_COMM_CHECK(cudaMemcpyAsync(comm->local_buffer, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
    
    // Record event after copy completes
    CUDA_COMM_CHECK(cudaEventRecord(comm->comm_id->events[comm->rank], stream));
    
    // Barrier to ensure all GPUs have recorded their events
    comm->comm_id->barrier->wait();
    
    // Wait for all other GPUs' data to be ready
    for (int i = 0; i < comm->gpu_count; i++) {
        if (i != comm->rank) {
            CUDA_COMM_CHECK(cudaStreamWaitEvent(stream, comm->comm_id->events[i], 0));
        }
    }
    
    // Gather data from all GPUs
    for (int i = 0; i < comm->gpu_count; i++) {
        void* dst = (char*)recvbuff + i * bytes;
        
        if (i == comm->rank) {
            // Copy from local sendbuff to local recvbuff
            CUDA_COMM_CHECK(cudaMemcpyAsync(dst, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
        } else {
            // Copy from peer GPU buffer to local recvbuff
            void* src = comm->comm_id->shared_buffers[i];
            CUDA_COMM_CHECK(cudaMemcpyPeerAsync(dst, comm->rank, src, i, bytes, stream));
        }
    }
    
    // Barrier to ensure all threads reach here before starting next operation
    // Stream ordering guarantees GPU operations complete before next copy
    comm->comm_id->barrier->wait();
}

// Cleanup
inline void cudaCommDestroy(CudaComm* comm) {
    if (comm->local_buffer) {
        cudaFree(comm->local_buffer);
    }
    if (comm->staging_buffer) {
        cudaFree(comm->staging_buffer);
    }
}

// Cleanup communicator ID
inline void cudaCommIdDestroy(CudaCommId* id) {
    if (id->barrier) {
        delete id->barrier;
        id->barrier = nullptr;
    }
    // Destroy all CUDA events
    for (int i = 0; i < MAX_GPUS; i++) {
        if (id->events[i]) {
            cudaEventDestroy(id->events[i]);
            id->events[i] = nullptr;
        }
    }
}
