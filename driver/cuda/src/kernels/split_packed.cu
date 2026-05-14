#include "kernels/split_packed.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// Vectorise copies as ushort4 = 8 bf16 values. The matmul output dims
// (Hq, Hk, intermediate) are all multiples of head_dim or fc width and
// in practice multiples of 8 for every model we ship. Fall back to a
// scalar tail just in case.
__global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int q_dim, int kv_dim)
{
    const int n = blockIdx.y;
    const int stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* src_row = src + static_cast<long long>(n) * stride;

    // Q block: cols [0, q_dim)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < q_dim;
         j += blockDim.x * gridDim.x) {
        q_out[static_cast<long long>(n) * q_dim + j] = src_row[j];
    }
    // K block: cols [q_dim, q_dim + kv_dim)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kv_dim;
         j += blockDim.x * gridDim.x) {
        k_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + j];
    }
    // V block: cols [q_dim + kv_dim, q_dim + 2*kv_dim)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kv_dim;
         j += blockDim.x * gridDim.x) {
        v_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + kv_dim + j];
    }
}

__global__ void split_gate_up_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    int inter)
{
    const int n = blockIdx.y;
    const int stride = 2 * inter;
    const __nv_bfloat16* src_row = src + static_cast<long long>(n) * stride;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inter;
         j += blockDim.x * gridDim.x) {
        gate_out[static_cast<long long>(n) * inter + j] = src_row[j];
    }
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inter;
         j += blockDim.x * gridDim.x) {
        up_out[static_cast<long long>(n) * inter + j] = src_row[inter + j];
    }
}

}  // namespace

void launch_split_qkv_bf16(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    int n_tokens, int q_dim, int kv_dim,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    constexpr int BLOCK = 256;
    const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
    const int xblocks = (max_dim + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_tokens);
    split_qkv_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_out),
        static_cast<__nv_bfloat16*>(v_out),
        q_dim, kv_dim);
}

void launch_split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    constexpr int BLOCK = 256;
    const int xblocks = (inter + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_tokens);
    split_gate_up_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(gate_out),
        static_cast<__nv_bfloat16*>(up_out),
        inter);
}

}  // namespace pie_cuda_driver::kernels
