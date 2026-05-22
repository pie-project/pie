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

namespace {

__global__ void sigmoid_gate_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = __bfloat162float(x[i]);
    const float gv = __bfloat162float(gate[i]);
    const float s  = 1.f / (1.f + __expf(-gv));
    x[i] = __float2bfloat16(xv * s);
}

}  // namespace

void launch_sigmoid_gate_inplace_bf16(
    void* x, const void* gate, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (n + BLOCK - 1) / BLOCK;
    sigmoid_gate_inplace_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gate),
        n);
}

namespace {

__global__ void chunked_swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16*       __restrict__ y,
    int N, int I)
{
    const int n = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    const float g = __bfloat162float(packed[packed_row + i]);
    const float u = __bfloat162float(packed[packed_row + I + i]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = __float2bfloat16(silu * u);
}

__global__ void chunked_swiglu_bf16_vec2_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16*       __restrict__ y,
    int N, int I)
{
    const int n = blockIdx.x;
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
    if (n >= N || i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    if (((I & 1) == 0) && i + 1 < I) {
        const auto gate2 = *reinterpret_cast<const __nv_bfloat162*>(
            packed + packed_row + i);
        const auto up2 = *reinterpret_cast<const __nv_bfloat162*>(
            packed + packed_row + I + i);
        const float2 g = __bfloat1622float2(gate2);
        const float2 u = __bfloat1622float2(up2);
        const float y0 = (g.x / (1.f + __expf(-g.x))) * u.x;
        const float y1 = (g.y / (1.f + __expf(-g.y))) * u.y;
        *reinterpret_cast<__nv_bfloat162*>(y + row + i) =
            __floats2bfloat162_rn(y0, y1);
        return;
    }

    const float g = __bfloat162float(packed[packed_row + i]);
    const float u = __bfloat162float(packed[packed_row + I + i]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = __float2bfloat16(silu * u);
}

}  // namespace

void launch_chunked_swiglu_bf16(
    const void* packed, void* y, int N, int I, cudaStream_t stream)
{
    if (N <= 0 || I <= 0) return;
    constexpr int BLOCK = 128;
    if (I > 10000) {
        dim3 grid(N, (I + BLOCK - 1) / BLOCK);
        chunked_swiglu_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed),
            static_cast<__nv_bfloat16*>(y),
            N, I);
        return;
    }
    dim3 grid(N, ((I + 1) / 2 + BLOCK - 1) / BLOCK);
    chunked_swiglu_bf16_vec2_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(y),
        N, I);
}

namespace {

// Same chunk layout as `chunked_swiglu_bf16_kernel` but applies the
// GELU-tanh activation that Gemma-4 uses on both its dense MLP and
// (for 26B-A4B) its routed-expert block. `gelu_tanh(x) = 0.5 * x *
// (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
__global__ void chunked_geglu_tanh_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16*       __restrict__ y,
    int N, int I)
{
    const int n = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || i >= I) return;
    const float g = __bfloat162float(packed[(long long)n * 2 * I + i]);
    const float u = __bfloat162float(packed[(long long)n * 2 * I + I + i]);
    constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/π)
    constexpr float kBeta  = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu  = 0.5f * g * (1.f + tanhf(inner));
    y[(long long)n * I + i] = __float2bfloat16(gelu * u);
}

}  // namespace

void launch_chunked_geglu_tanh_bf16(
    const void* packed, void* y, int N, int I, cudaStream_t stream)
{
    if (N <= 0 || I <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(N, (I + BLOCK - 1) / BLOCK);
    chunked_geglu_tanh_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(y),
        N, I);
}

namespace {

__global__ void sigmoid_scalar_gate_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scalar_gate,
    int N, int H)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N || h >= H) return;
    const float gv = __bfloat162float(scalar_gate[n]);
    const float s  = 1.f / (1.f + __expf(-gv));
    const long long i = (long long)n * H + h;
    x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

}  // namespace

void launch_sigmoid_scalar_gate_inplace_bf16(
    void* x, const void* scalar_gate, int N, int H, cudaStream_t stream)
{
    if (N <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(N, (H + BLOCK - 1) / BLOCK);
    sigmoid_scalar_gate_inplace_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(scalar_gate),
        N, H);
}

}  // namespace pie_cuda_driver::kernels
