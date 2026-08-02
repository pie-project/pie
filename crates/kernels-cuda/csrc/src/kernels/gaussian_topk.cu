#include "kernels/gaussian_topk.hpp"

#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace pie_cuda_driver::kernels {

namespace cg = cooperative_groups;

namespace {

// One block per row. Threads cooperatively reduce the mean and variance
// (single Welford-style two-pass: first sum then sum-of-squared-diffs)
// in fp32, then sweep the row again to apply the cutoff.
__global__ void gaussian_topk_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    int dim,
    float std_multiplier)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __nv_bfloat16* row_ptr = x + static_cast<long long>(row) * dim;

    // Pass 1: row mean.
    float local_sum = 0.f;
    for (int j = tid; j < dim; j += blockDim.x) {
        local_sum += __bfloat162float(row_ptr[j]);
    }
    // Block-wide sum: warp shfl + shared-mem combine.
    extern __shared__ float smem[];
    // First reduce within warp.
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += tile.shfl_down(local_sum, off);
    }
    if (tile.thread_rank() == 0) smem[tile.meta_group_rank()] = local_sum;
    __syncthreads();
    if (tile.meta_group_rank() == 0) {
        local_sum = (tid < tile.meta_group_size()) ? smem[tid] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += tile.shfl_down(local_sum, off);
        }
        if (tile.thread_rank() == 0) smem[0] = local_sum;
    }
    __syncthreads();
    const float mean = smem[0] / static_cast<float>(dim);

    // Pass 2: variance.
    float local_var = 0.f;
    for (int j = tid; j < dim; j += blockDim.x) {
        const float v = __bfloat162float(row_ptr[j]) - mean;
        local_var += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_var += tile.shfl_down(local_var, off);
    }
    if (tile.thread_rank() == 0) smem[tile.meta_group_rank()] = local_var;
    __syncthreads();
    if (tile.meta_group_rank() == 0) {
        local_var = (tid < tile.meta_group_size()) ? smem[tid] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            local_var += tile.shfl_down(local_var, off);
        }
        if (tile.thread_rank() == 0) smem[0] = local_var;
    }
    __syncthreads();
    const float var = smem[0] / static_cast<float>(dim);
    const float stddev = sqrtf(var);
    const float cutoff = mean + stddev * std_multiplier;

    // Pass 3: apply.
    for (int j = tid; j < dim; j += blockDim.x) {
        const float v = __bfloat162float(row_ptr[j]) - cutoff;
        row_ptr[j] = __float2bfloat16(v > 0.f ? v : 0.f);
    }
}

}  // namespace

void launch_gaussian_topk_bf16(
    void* x, int N, int dim,
    float std_multiplier, cudaStream_t stream)
{
    if (N <= 0 || dim <= 0) return;
    constexpr int BLOCK = 256;
    const int smem_bytes = (BLOCK / 32) * sizeof(float);
    gaussian_topk_bf16_kernel<<<N, BLOCK, smem_bytes, stream>>>(
        static_cast<__nv_bfloat16*>(x), dim, std_multiplier);
}

}  // namespace pie_cuda_driver::kernels
