#include "kernels/softcap.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void logit_softcap_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    float inv_cap,
    float cap,
    std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(cap * tanhf(v * inv_cap));
}

}  // namespace

void launch_logit_softcap_bf16(
    void* x, float cap, std::size_t n, cudaStream_t stream)
{
    if (n == 0 || !(cap > 0.f)) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    logit_softcap_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x), 1.f / cap, cap, n);
}

}  // namespace pie_cuda_driver::kernels
