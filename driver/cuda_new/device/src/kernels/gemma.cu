#include "gemma.cuh"

#include <cuda_bf16.h>

// Gemma-specific kernels, lifted verbatim from the legacy driver. Each entry
// keeps its original kernel body; the only changes are the namespace
// (pie_cuda_driver::kernels -> pie_cuda_device::kernels) and dropping the
// `launch_` prefix on the entry names.
//
//   rmsnorm_gemma_bf16  <- driver/cuda/src/kernels/rmsnorm.cu
//   geglu_tanh_bf16     <- driver/cuda/src/kernels/swiglu.cu
//   logit_softcap_bf16  <- driver/cuda/src/kernels/softcap.cu

namespace pie_cuda_device::kernels {

namespace {

// One block per row. `BLOCK` threads cooperate on the L2-norm reduction;
// each thread handles `hidden / BLOCK` elements. We always launch with
// BLOCK == 256, so the per-thread chunk is small even for hidden=8192.
//
// `WEIGHT_PLUS_ONE = true` selects Gemma's `(1 + w) * x_hat` convention.
template <int BLOCK, bool WEIGHT_PLUS_ONE>
__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* xr = x + row * hidden;
    __nv_bfloat16* yr = y + row * hidden;

    // L2 norm across the row, accumulated in fp32.
    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    // Block-wide reduction via shared memory. CUB would be cleaner, but this
    // avoids a heavy header for the M1.2.2 build.
    __shared__ float buf[BLOCK];
    buf[tid] = local;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = __bfloat162float(xr[i]);
        float wv = __bfloat162float(weight[i]);
        if constexpr (WEIGHT_PLUS_ONE) wv += 1.f;
        yr[i] = __float2bfloat16(xv * inv_rms * wv);
    }
}

}  // namespace

void rmsnorm_gemma_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_bf16_kernel<BLOCK, /*WEIGHT_PLUS_ONE=*/true><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, eps);
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

void geglu_tanh_bf16(
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

void logit_softcap_bf16(
    void* x, float cap, std::size_t n, cudaStream_t stream)
{
    if (n == 0 || !(cap > 0.f)) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    logit_softcap_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x), 1.f / cap, cap, n);
}

}  // namespace pie_cuda_device::kernels
