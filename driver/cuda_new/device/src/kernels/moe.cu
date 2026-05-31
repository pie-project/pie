#include "moe.cuh"

#include <cuda_bf16.h>
#include <cfloat>
#include <stdexcept>

// Lifted from driver/cuda/src/kernels/topk_softmax.cu (topk_softmax_bf16) and
// driver/cuda/src/kernels/swiglu.cu (chunked_swiglu_bf16 + the scalar and
// vec2 kernels its launcher dispatches to). Kernel bodies are verbatim; the
// only change is the namespace and dropping the `launch_` prefix on the entry
// points. The per-expert-scale / sigmoid-bias router variants and the strided
// / geglu / sigmoid-gate activation variants are lifted separately.

namespace pie_cuda_device::kernels {

namespace {

constexpr int BLOCK = 64;
// Qwen3.6-35B-A3B uses 256 experts; Kimi K2.6 uses 384. Keep a single
// static shared-memory slab large enough for both. 512 floats == 2 KB.
constexpr int MAX_EXPERTS = 512;

// One block per token. Phase 1: thread-local max-reduce + exp+sum-reduce
// for softmax. Phase 2: K iterations of argmax-with-exclusion to pick the
// top-K probs. Phase 3: thread 0 renormalizes and writes back.
__global__ void topk_softmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float buf[BLOCK];

    // 1. Stage row into shared memory + find max.
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float v = __bfloat162float(row[j]);
        probs[j] = v;
        if (v > local_max) local_max = v;
    }
    buf[tid] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    const float row_max = buf[0];
    __syncthreads();

    // 2. exp + sum.
    float local_sum = 0.f;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float e = expf(probs[j] - row_max);
        probs[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_Z = 1.f / buf[0];
    __syncthreads();

    if (tid == 0) {
        // Normalize in shared mem, then K-argmax with exclusion.
        for (int j = 0; j < num_experts; ++j) probs[j] *= inv_Z;

        std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
        float*        out_w   = topk_w   + static_cast<long long>(n) * K;
        float w_sum = 0.f;
        for (int k = 0; k < K; ++k) {
            int   best_i = -1;
            float best_v = -1.f;
            for (int j = 0; j < num_experts; ++j) {
                if (probs[j] > best_v) { best_v = probs[j]; best_i = j; }
            }
            out_idx[k] = best_i;
            out_w[k]   = best_v;
            w_sum += best_v;
            probs[best_i] = -1.f;  // exclude on next pass
        }
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) out_w[k] *= inv_w;
    }
}

}  // namespace

void topk_softmax_bf16(
    const void* logits,
    std::int32_t* topk_idx, float* topk_w,
    int N, int num_experts, int K,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    if (num_experts > MAX_EXPERTS) {
        throw std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    topk_softmax_bf16_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        topk_idx, topk_w,
        num_experts, K);
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

void chunked_swiglu_bf16(
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

}  // namespace pie_cuda_device::kernels
