#include "layout/split_gate_up.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels::layout {
namespace {


}  // namespace

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


void split_gate_up_bf16(
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


}  // namespace pie_cuda_driver::kernels::layout
