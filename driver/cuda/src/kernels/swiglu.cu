#include "kernels/swiglu.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ y,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = __bfloat162float(gate[idx]);
    const float u = __bfloat162float(up[idx]);
    const float silu = g / (1.f + expf(-g));
    y[idx] = __float2bfloat16(silu * u);
}

__global__ void gpt_oss_glu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ y,
    int n,
    float limit,
    float alpha)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // Asymmetric clamp on gate (upper only) and symmetric on up — matches
    // HF's GptOssExperts._apply_gate.
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    // QuickGELU-style: x * sigmoid(alpha * x), alpha = 1.702.
    const float glu = g / (1.f + expf(-alpha * g));
    y[idx] = __float2bfloat16((u + 1.f) * glu);
}

}  // namespace

void launch_swiglu_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    swiglu_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up),
        static_cast<__nv_bfloat16*>(y),
        num_elements);
}

void launch_gpt_oss_glu_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, cudaStream_t stream,
    float limit, float alpha)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    gpt_oss_glu_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up),
        static_cast<__nv_bfloat16*>(y),
        num_elements, limit, alpha);
}

namespace {

// GeLU(tanh) gate. `c = √(2/π) ≈ 0.7978845608…`. The cubic term coefficient
// is the canonical 0.044715 used by `torch.nn.functional.gelu(approximate="tanh")`
// (matches HF's `gelu_pytorch_tanh`).
__global__ void geglu_tanh_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ y,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    constexpr float c = 0.7978845608028654f;
    const float g = __bfloat162float(gate[idx]);
    const float u = __bfloat162float(up[idx]);
    const float gelu = 0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g)));
    y[idx] = __float2bfloat16(gelu * u);
}

}  // namespace

void launch_geglu_tanh_bf16(
    const void* gate, const void* up, void* y,
    int num_elements, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (num_elements + BLOCK - 1) / BLOCK;
    geglu_tanh_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up),
        static_cast<__nv_bfloat16*>(y),
        num_elements);
}

}  // namespace pie_cuda_driver::kernels
