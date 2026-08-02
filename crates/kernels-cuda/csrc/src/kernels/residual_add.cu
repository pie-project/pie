#include "kernels/residual_add.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void residual_add_bf16_kernel(
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float a = __bfloat162float(y[i]);
    const float b = __bfloat162float(x[i]);
    y[i] = __float2bfloat16(a + b);
}

}  // namespace

void launch_residual_add_bf16(
    void* y, const void* x,
    std::size_t n,
    cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    residual_add_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(x),
        n);
}

}  // namespace pie_cuda_driver::kernels
