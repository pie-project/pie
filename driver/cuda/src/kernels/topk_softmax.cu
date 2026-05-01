#include "kernels/topk_softmax.hpp"

#include <cuda_bf16.h>
#include <cfloat>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 64;
constexpr int MAX_EXPERTS = 64;

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
    topk_softmax_bf16_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        topk_idx, topk_w,
        num_experts, K);
}

}  // namespace pie_cuda_driver::kernels
