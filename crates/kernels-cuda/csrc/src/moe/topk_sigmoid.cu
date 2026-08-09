#include "moe/topk_sigmoid.hpp"

#include <cfloat>

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels::moe {
namespace {

constexpr int MAX_EXPERTS = 512;
constexpr int TOPK_BLOCK = 128;

}  // namespace

__global__ void topk_sigmoid_kernel(
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
        const float s = 1.f / (1.f + expf(-x));
        orig_scores[e] = s;
        scores[e] = correction_bias != nullptr ? s + correction_bias[e] : s;
    }
    __syncthreads();

    // `taken` rather than poisoning `scores` with -FLT_MAX: the poison value is
    // indistinguishable from a genuine score, so a row containing NaN (every
    // comparison against which is false) or K > E would leave the scan with no
    // winner. That used to fall out as `best_i == -1`, which then wrote
    // `scores[-1]` -- an out-of-bounds shared write -- and published expert -1
    // into `topk_idx`, where the MoE pointer builder turned it into a negative
    // weight offset and the failure finally surfaced as an illegal address
    // inside a batched GEMM, far from its cause.
    __shared__ bool taken[MAX_EXPERTS];
    for (int e = tid; e < E; e += TOPK_BLOCK) taken[e] = false;
    __syncthreads();

    if (tid == 0) {
        std::int32_t* idx = topk_idx + static_cast<long long>(n) * K;
        float* w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        const int picks = K < E ? K : E;
        for (int k = 0; k < picks; ++k) {
            int best_i = -1;
            float best_v = -FLT_MAX;
            for (int e = 0; e < E; ++e) {
                if (taken[e]) continue;
                const float v = scores[e];
                // Seeding from the first untaken expert keeps a winner even
                // when every remaining score is NaN; for ordinary rows this is
                // the same first-maximum the strict `>` scan already produced.
                if (best_i < 0 || v > best_v) {
                    best_v = v;
                    best_i = e;
                }
            }
            idx[k] = best_i;
            w[k] = orig_scores[best_i];
            sum += orig_scores[best_i];
            taken[best_i] = true;
        }
        // Only reachable when a checkpoint asks for more routes than it has
        // experts. Repeating the last expert would double-count it in the
        // weighted sum, so these slots are parked on expert 0 with zero weight.
        for (int k = picks; k < K; ++k) {
            idx[k] = 0;
            w[k] = 0.f;
        }
        const float scale = renormalize && sum > 0.f
            ? routed_scaling_factor / sum
            : routed_scaling_factor;
        for (int k = 0; k < K; ++k) w[k] *= scale;
    }
}


void topk_sigmoid_bf16(
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
    topk_sigmoid_kernel<<<tokens, TOPK_BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        topk_idx, topk_w, correction_bias, num_experts, top_k,
        renormalize, routed_scaling_factor);
}


}  // namespace pie_cuda_driver::kernels::moe
