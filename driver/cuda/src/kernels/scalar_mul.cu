#include "kernels/scalar_mul.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void scalar_mul_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    float s,
    std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    // Round the scalar to bf16 first so the multiplication is
    // bf16(x) * bf16(s) → fp32 → bf16, matching how PyTorch evaluates
    // `tensor * bf16_scalar`. Gemma-4's `embed_normalizer` is stored
    // as bf16 in the reference adapter for exactly this reason: a
    // raw fp32 scalar produces a 1-ULP-per-element drift that
    // RMSNorm amplifies into multi-unit divergence by layer ~5.
    const float s_rounded = __bfloat162float(__float2bfloat16(s));
    x[i] = __float2bfloat16(__bfloat162float(x[i]) * s_rounded);
}

}  // namespace

void launch_scalar_mul_bf16(
    void* x, float s, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    scalar_mul_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x), s, n);
}

}  // namespace pie_cuda_driver::kernels
