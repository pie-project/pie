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

__global__ void q_nope_to_latent_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ kv_b,
    __nv_bfloat16* __restrict__ out,
    int tokens,
    int heads,
    int nope,
    int v_dim,
    int kv_lora)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * heads * kv_lora;
    if (idx >= total) return;
    const int l = idx % kv_lora;
    const int h = (idx / kv_lora) % heads;
    const int n = idx / (heads * kv_lora);
    const __nv_bfloat16* q =
        q_nope + (static_cast<long long>(n) * heads + h) * nope;
    const __nv_bfloat16* w =
        kv_b + static_cast<long long>(h) * (nope + v_dim) * kv_lora + l;
    float acc = 0.f;
    for (int d = 0; d < nope; ++d) {
        acc += __bfloat162float(q[d]) *
               __bfloat162float(w[static_cast<long long>(d) * kv_lora]);
    }
    out[idx] = __float2bfloat16(acc);
}

__global__ void latent_to_v_kernel(
    const __nv_bfloat16* __restrict__ latent,
    const __nv_bfloat16* __restrict__ kv_b,
    __nv_bfloat16* __restrict__ out,
    int tokens,
    int heads,
    int nope,
    int v_dim,
    int kv_lora)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * heads * v_dim;
    if (idx >= total) return;
    const int v = idx % v_dim;
    const int h = (idx / v_dim) % heads;
    const int n = idx / (heads * v_dim);
    const __nv_bfloat16* x =
        latent + (static_cast<long long>(n) * heads + h) * kv_lora;
    const __nv_bfloat16* w =
        kv_b + (static_cast<long long>(h) * (nope + v_dim) + nope + v) * kv_lora;
    float acc = 0.f;
    for (int l = 0; l < kv_lora; ++l) {
        acc += __bfloat162float(x[l]) * __bfloat162float(w[l]);
    }
    out[idx] = __float2bfloat16(acc);
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

void launch_kimi_q_nope_to_latent_bf16(
    const void* q_nope,
    const void* kv_b_proj,
    void* q_latent,
    int tokens,
    int heads,
    int qk_nope_dim,
    int v_head_dim,
    int kv_lora_rank,
    cudaStream_t stream)
{
    const int total = tokens * heads * kv_lora_rank;
    if (total <= 0) return;
    q_nope_to_latent_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(kv_b_proj),
        static_cast<__nv_bfloat16*>(q_latent),
        tokens, heads, qk_nope_dim, v_head_dim, kv_lora_rank);
}

void launch_kimi_latent_to_v_bf16(
    const void* attn_latent,
    const void* kv_b_proj,
    void* attn_v,
    int tokens,
    int heads,
    int qk_nope_dim,
    int v_head_dim,
    int kv_lora_rank,
    cudaStream_t stream)
{
    const int total = tokens * heads * v_head_dim;
    if (total <= 0) return;
    latent_to_v_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(attn_latent),
        static_cast<const __nv_bfloat16*>(kv_b_proj),
        static_cast<__nv_bfloat16*>(attn_v),
        tokens, heads, qk_nope_dim, v_head_dim, kv_lora_rank);
}

}  // namespace pie_cuda_driver::kernels
