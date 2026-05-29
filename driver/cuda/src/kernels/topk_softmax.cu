#include "kernels/topk_softmax.hpp"

#include <cuda_bf16.h>
#include <cfloat>
#include <stdexcept>

namespace pie_cuda_driver::kernels {

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

void launch_topk_softmax_bf16(
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

__global__ void apply_per_expert_scale_kernel(
    const std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const __nv_bfloat16* __restrict__ per_expert_scale,
    int total)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = topk_idx[t];
    const float s = __bfloat162float(per_expert_scale[e]);
    topk_w[t] *= s;
}

}  // namespace

void launch_apply_per_expert_scale_bf16(
    const std::int32_t* topk_idx,
    float* topk_w,
    const void* per_expert_scale_bf16,
    int N, int K,
    cudaStream_t stream)
{
    const int total = N * K;
    if (total <= 0) return;
    constexpr int BLOCK_T = 256;
    const int grid = (total + BLOCK_T - 1) / BLOCK_T;
    apply_per_expert_scale_kernel<<<grid, BLOCK_T, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const __nv_bfloat16*>(per_expert_scale_bf16),
        total);
}

namespace {

__global__ void topk_sigmoid_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ correction_bias,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row =
        logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float choice[MAX_EXPERTS];

    for (int j = tid; j < num_experts; j += BLOCK) {
        const float z = __bfloat162float(row[j]);
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    if (tid == 0) {
        std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
        float* out_w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        for (int k = 0; k < K; ++k) {
            int best_i = -1;
            float best_v = -FLT_MAX;
            for (int j = 0; j < num_experts; ++j) {
                if (choice[j] > best_v) {
                    best_v = choice[j];
                    best_i = j;
                }
            }
            out_idx[k] = best_i;
            out_w[k] = probs[best_i];
            sum += out_w[k];
            choice[best_i] = -FLT_MAX;
        }
        const float scale =
            normalize ? (routed_scaling_factor / (sum + 1e-20f))
                      : routed_scaling_factor;
        for (int k = 0; k < K; ++k) {
            out_w[k] *= scale;
        }
    }
}

__global__ void topk_sigmoid_bias_fp32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float choice[MAX_EXPERTS];

    for (int j = tid; j < num_experts; j += BLOCK) {
        const float z = row[j];
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    if (tid == 0) {
        std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
        float* out_w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        for (int k = 0; k < K; ++k) {
            int best_i = -1;
            float best_v = -FLT_MAX;
            for (int j = 0; j < num_experts; ++j) {
                if (choice[j] > best_v) {
                    best_v = choice[j];
                    best_i = j;
                }
            }
            out_idx[k] = best_i;
            out_w[k] = probs[best_i];
            sum += out_w[k];
            choice[best_i] = -FLT_MAX;
        }
        const float scale =
            normalize ? (routed_scaling_factor / (sum + 1e-20f))
                      : routed_scaling_factor;
        for (int k = 0; k < K; ++k) {
            out_w[k] *= scale;
        }
    }
}

}  // namespace

void launch_topk_sigmoid_bias_bf16(
    const void* logits,
    const float* correction_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    topk_sigmoid_bias_bf16_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

void launch_topk_sigmoid_bias_fp32(
    const float* logits,
    const float* correction_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    topk_sigmoid_bias_fp32_kernel<<<N, BLOCK, 0, stream>>>(
        logits,
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

}  // namespace pie_cuda_driver::kernels
