#include "add_bias.cuh"

#include <cstddef>

#include <cuda_bf16.h>

namespace pie_cuda_device::kernels {

namespace {

constexpr int BLOCK = 256;

// One thread per element of the [num_tokens, dim] tensor. The bias index is
// the channel (column), so it is the flat index modulo dim.
__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    std::size_t n,
    int dim)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (idx >= n) return;
    const int i = static_cast<int>(idx % static_cast<std::size_t>(dim));
    const float a = __bfloat162float(x[idx]);
    const float b = __bfloat162float(bias[i]);
    x[idx] = __float2bfloat16(a + b);
}

}  // namespace

void add_bias_bf16(void* x, const void* bias, int num_tokens, int dim, cudaStream_t stream) {
    if (num_tokens <= 0 || dim <= 0) return;
    const std::size_t n = static_cast<std::size_t>(num_tokens) * static_cast<std::size_t>(dim);
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    add_bias_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(bias),
        n,
        dim);
}

}  // namespace pie_cuda_device::kernels
