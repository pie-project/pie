#include "kernels/kimi_mla.hpp"

#include <cuda_bf16.h>
#include <cfloat>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;
constexpr int TOPK_BLOCK = 128;
constexpr int MAX_EXPERTS = 512;

__global__ void split_q_b_kernel(
    const __nv_bfloat16* __restrict__ q_b,
    __nv_bfloat16* __restrict__ q_nope,
    __nv_bfloat16* __restrict__ q_pe,
    int total,
    int heads,
    int nope,
    int rope)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = nope + rope;
    const int d = i % per;
    const int h = (i / per) % heads;
    const int n = i / (heads * per);
    const __nv_bfloat16 v = q_b[i];
    if (d < nope) {
        q_nope[(static_cast<long long>(n) * heads + h) * nope + d] = v;
    } else {
        q_pe[(static_cast<long long>(n) * heads + h) * rope + (d - nope)] = v;
    }
}

__global__ void split_kv_a_kernel(
    const __nv_bfloat16* __restrict__ kv_a,
    __nv_bfloat16* __restrict__ kv_c,
    __nv_bfloat16* __restrict__ k_pe,
    int total,
    int kv_lora,
    int rope)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = kv_lora + rope;
    const int d = i % per;
    const int n = i / per;
    const __nv_bfloat16 v = kv_a[i];
    if (d < kv_lora) {
        kv_c[static_cast<long long>(n) * kv_lora + d] = v;
    } else {
        k_pe[static_cast<long long>(n) * rope + (d - kv_lora)] = v;
    }
}

// Fused split_kv_a + rmsnorm(kv_c): splits [kv_lora+rope] → kv_c[kv_lora] (normalized) + k_pe[rope]
template <int BLOCK_DIM>
__global__ void split_kv_a_norm_kernel(
    const __nv_bfloat16* __restrict__ kv_a,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ kv_c,
    __nv_bfloat16* __restrict__ k_pe,
    int kv_lora, int rope, int src_row_stride, float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row = kv_a + static_cast<long long>(n) * src_row_stride;

    // Copy k_pe (no normalization)
    for (int d = tid; d < rope; d += BLOCK_DIM) {
        k_pe[static_cast<long long>(n) * rope + d] = row[kv_lora + d];
    }

    // RMSNorm on kv_c portion
    float local = 0.f;
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = __bfloat162float(row[d]);
        local += v * v;
    }
    __shared__ float buf[BLOCK_DIM];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = __bfloat162float(row[d]);
        const float w = __bfloat162float(norm_weight[d]);
        kv_c[static_cast<long long>(n) * kv_lora + d] = __float2bfloat16(v * inv_rms * w);
    }
}

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

}  // namespace

void launch_kimi_split_q_b_bf16(
    const void* q_b,
    void* q_nope,
    void* q_pe,
    int tokens,
    int heads,
    int qk_nope_dim,
    int qk_rope_dim,
    cudaStream_t stream)
{
    const int total = tokens * heads * (qk_nope_dim + qk_rope_dim);
    if (total <= 0) return;
    split_q_b_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_b),
        static_cast<__nv_bfloat16*>(q_nope),
        static_cast<__nv_bfloat16*>(q_pe),
        total, heads, qk_nope_dim, qk_rope_dim);
}

void launch_kimi_split_kv_a_bf16(
    const void* kv_a,
    void* kv_c,
    void* k_pe,
    int tokens,
    int kv_lora_rank,
    int qk_rope_dim,
    cudaStream_t stream)
{
    const int total = tokens * (kv_lora_rank + qk_rope_dim);
    if (total <= 0) return;
    split_kv_a_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv_a),
        static_cast<__nv_bfloat16*>(kv_c),
        static_cast<__nv_bfloat16*>(k_pe),
        total, kv_lora_rank, qk_rope_dim);
}

void launch_kimi_split_kv_a_norm_bf16(
    const void* kv_a,
    const void* norm_weight,
    void* kv_c,
    void* k_pe,
    int tokens,
    int kv_lora_rank,
    int qk_rope_dim,
    float eps,
    cudaStream_t stream,
    int src_row_stride)
{
    if (tokens <= 0) return;
    constexpr int BS = 256;
    const int stride =
        src_row_stride > 0 ? src_row_stride : kv_lora_rank + qk_rope_dim;
    split_kv_a_norm_kernel<BS><<<tokens, BS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv_a),
        static_cast<const __nv_bfloat16*>(norm_weight),
        static_cast<__nv_bfloat16*>(kv_c),
        static_cast<__nv_bfloat16*>(k_pe),
        kv_lora_rank, qk_rope_dim, stride, eps);
}

void launch_topk_sigmoid_bf16(
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

}  // namespace pie_cuda_driver::kernels
