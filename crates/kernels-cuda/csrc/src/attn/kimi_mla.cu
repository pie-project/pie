#include "attn/kimi_mla.hpp"

#include <cuda_bf16.h>
#include <cfloat>

namespace pie_cuda_driver::kernels::attn {

namespace {

constexpr int BLOCK = 256;

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

}  // namespace

void kimi_split_q_b_bf16(
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

void kimi_split_kv_a_norm_bf16(
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

}  // namespace pie_cuda_driver::kernels::attn
