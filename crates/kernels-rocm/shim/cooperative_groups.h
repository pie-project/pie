#pragma once

// The cooperative-groups surface the carried units use, for AMD. A grid
// barrier is the device library's `__ockl_grid_sync`, which is what HIP's
// own cooperative groups call and which holds under
// `hipModuleLaunchCooperativeKernel` (jit/launch.rs takes that path for a
// `Launch::cooperative()`). AMD has no thread-block clusters: every block is
// a cluster of one, so the cluster calls answer for that one block.

extern "C" __device__ void __ockl_grid_sync(void);

namespace cooperative_groups {

class thread_block {
public:
    __device__ __forceinline__ void sync() const { __syncthreads(); }
    __device__ __forceinline__ unsigned int thread_rank() const {
        return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    }
    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
    __device__ __forceinline__ dim3 group_index() const { return blockIdx; }
    __device__ __forceinline__ dim3 thread_index() const { return threadIdx; }
};

__device__ __forceinline__ thread_block this_thread_block() { return thread_block{}; }

class grid_group {
public:
    __device__ __forceinline__ void sync() const { __ockl_grid_sync(); }
    __device__ __forceinline__ unsigned long long thread_rank() const {
        const unsigned long long block =
            blockIdx.x + (unsigned long long)gridDim.x * (blockIdx.y + (unsigned long long)gridDim.y * blockIdx.z);
        const unsigned long long within =
            threadIdx.x + (unsigned long long)blockDim.x * (threadIdx.y + (unsigned long long)blockDim.y * threadIdx.z);
        return block * (blockDim.x * (unsigned long long)blockDim.y * blockDim.z) + within;
    }
    __device__ __forceinline__ unsigned long long size() const {
        return (unsigned long long)gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y *
               blockDim.z;
    }
    __device__ __forceinline__ unsigned int cluster_rank() const {
        return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    }
    __device__ __forceinline__ unsigned int num_clusters() const {
        return gridDim.x * gridDim.y * gridDim.z;
    }
};

class cluster_group {
public:
    __device__ __forceinline__ unsigned int block_rank() const { return 0; }
    __device__ __forceinline__ unsigned int num_blocks() const { return 1; }
    __device__ __forceinline__ unsigned int thread_rank() const {
        return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    }
    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
    template <typename T>
    __device__ __forceinline__ T* map_shared_rank(T* addr, int) const {
        return addr;
    }
};

__device__ __forceinline__ cluster_group this_cluster() { return cluster_group{}; }
__device__ __forceinline__ grid_group this_grid() { return grid_group{}; }

}  // namespace cooperative_groups

namespace cg = cooperative_groups;
