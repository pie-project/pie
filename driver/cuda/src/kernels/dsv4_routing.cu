#include "kernels/dsv4_routing.hpp"

#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int MAX_EXPERTS = 512;
constexpr int TOPK_BLOCK = 256;

__device__ __forceinline__ float sqrtsoftplus(float x) {
    // sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
    // Numerically stable: for large x, softplus(x) ≈ x
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

__global__ void topk_sqrtsoftplus_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const float* __restrict__ correction_bias,
    int E,
    int K,
    bool renormalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row = logits + static_cast<long long>(n) * E;
    __shared__ float scores[MAX_EXPERTS];
    __shared__ float orig_scores[MAX_EXPERTS];

    for (int e = tid; e < E; e += TOPK_BLOCK) {
        const float x = __bfloat162float(row[e]);
        const float s = sqrtsoftplus(x);
        orig_scores[e] = s;
        scores[e] = correction_bias != nullptr ? s + correction_bias[e] : s;
    }
    __syncthreads();

    if (tid == 0) {
        std::int32_t* idx = topk_idx + static_cast<long long>(n) * K;
        float* w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        for (int k = 0; k < K; ++k) {
            int best_i = -1;
            float best_v = -FLT_MAX;
            for (int e = 0; e < E; ++e) {
                const float v = scores[e];
                if (v > best_v) {
                    best_v = v;
                    best_i = e;
                }
            }
            idx[k] = best_i;
            w[k] = orig_scores[best_i];
            sum += orig_scores[best_i];
            scores[best_i] = -FLT_MAX;
        }
        const float scale = renormalize && sum > 0.f
            ? routed_scaling_factor / sum
            : routed_scaling_factor;
        for (int k = 0; k < K; ++k) w[k] *= scale;
    }
}

__global__ void hash_route_lookup_kernel(
    const std::int32_t* __restrict__ token_ids,
    const std::int64_t* __restrict__ tid2eid,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int vocab_size,
    int K,
    float weight_per_expert)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    // bounds check happens in launch
    const int tok = token_ids[n];
    const int clamped = (tok >= 0 && tok < vocab_size) ? tok : 0;

    const std::int64_t* row = tid2eid + static_cast<long long>(clamped) * K;
    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;

    for (int k = 0; k < K; ++k) {
        out_idx[k] = static_cast<std::int32_t>(row[k]);
        out_w[k] = weight_per_expert;
    }
}

}  // namespace

void launch_topk_sqrtsoftplus_bf16(
    const void* logits,
    std::int32_t* topk_idx,
    float* topk_w,
    const float* correction_bias,
    int tokens,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (tokens <= 0 || num_experts <= 0 || top_k <= 0) return;
    if (num_experts > MAX_EXPERTS) return;
    topk_sqrtsoftplus_kernel<<<tokens, TOPK_BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        topk_idx, topk_w, correction_bias, num_experts, top_k,
        renormalize, routed_scaling_factor);
}

void launch_hash_route_lookup(
    const std::int32_t* token_ids,
    const std::int64_t* tid2eid,
    std::int32_t* topk_idx,
    float* topk_w,
    int tokens,
    int vocab_size,
    int top_k,
    float weight_per_expert,
    cudaStream_t stream)
{
    if (tokens <= 0 || top_k <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (tokens + BLOCK - 1) / BLOCK;
    hash_route_lookup_kernel<<<grid, BLOCK, 0, stream>>>(
        token_ids, tid2eid, topk_idx, topk_w,
        vocab_size, top_k, weight_per_expert);
}

}  // namespace pie_cuda_driver::kernels
