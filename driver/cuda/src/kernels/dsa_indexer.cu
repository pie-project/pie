#include "kernels/dsa_indexer.hpp"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace pie_cuda_driver::kernels {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ void rope_interleave_inplace(
    float* v, int rope_dim, int pos, float theta)
{
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = powf(theta, -2.f * static_cast<float>(i) /
                                       static_cast<float>(rope_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = v[2 * i];
        const float b = v[2 * i + 1];
        v[2 * i]     = a * c - b * s;
        v[2 * i + 1] = b * c + a * s;
    }
}

// LayerNorm(head_dim) + interleaved RoPE on idx_k. One block per token.
__global__ void index_knorm_rope_kernel(
    __nv_bfloat16* __restrict__ idx_k,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
    const std::int32_t* __restrict__ positions,
    int head_dim, int rope_dim, float theta, float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16* row = idx_k + static_cast<long long>(n) * head_dim;

    __shared__ float red[kBlock];
    float s = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) s += __bfloat162float(row[d]);
    red[tid] = s; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float mean = red[0] / head_dim;
    __syncthreads();
    float vv = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) { float x = __bfloat162float(row[d]) - mean; vv += x * x; }
    red[tid] = vv; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / head_dim + eps);
    __syncthreads();
    for (int d = tid; d < head_dim; d += kBlock) {
        const float x = (__bfloat162float(row[d]) - mean) * inv;
        row[d] = __float2bfloat16(x * __bfloat162float(w[d]) + __bfloat162float(b[d]));
    }
    __syncthreads();
    if (tid == 0) {
        float buf[256];
        for (int d = 0; d < rope_dim; ++d) buf[d] = __bfloat162float(row[d]);
        rope_interleave_inplace(buf, rope_dim, positions[n], theta);
        for (int d = 0; d < rope_dim; ++d) row[d] = __float2bfloat16(buf[d]);
    }
}

// Interleaved RoPE on first rope_dim of each index head of idx_q. One block
// per token, one thread per head.
__global__ void index_q_rope_kernel(
    __nv_bfloat16* __restrict__ idx_q,
    const std::int32_t* __restrict__ positions,
    int n_heads, int head_dim, int rope_dim, float theta)
{
    const int n = blockIdx.x;
    const int h = threadIdx.x;
    if (h >= n_heads) return;
    __nv_bfloat16* row =
        idx_q + (static_cast<long long>(n) * n_heads + h) * head_dim;
    float buf[256];
    for (int d = 0; d < rope_dim; ++d) buf[d] = __bfloat162float(row[d]);
    rope_interleave_inplace(buf, rope_dim, positions[n], theta);
    for (int d = 0; d < rope_dim; ++d) row[d] = __float2bfloat16(buf[d]);
}

// One block per query token i. Builds causal top-k mask over keys j<=i.
//   logit[i,j] = sum_h relu(q[i,h].k[j]) * w[i,h]
// (the positive softmax/n_head scale is monotonic so it's irrelevant to the
// top-k ranking and omitted).
__global__ void index_topk_mask_kernel(
    const __nv_bfloat16* __restrict__ idx_q,
    const __nv_bfloat16* __restrict__ idx_k,
    const __nv_bfloat16* __restrict__ idx_w,
    std::uint8_t* __restrict__ mask,
    int N, int H, int D, int topk)
{
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    std::uint8_t* mrow = mask + static_cast<long long>(i) * N;
    const int nkeys = i + 1;  // causal

    for (int j = nkeys + tid; j < N; j += kBlock) mrow[j] = 0;

    if (nkeys <= topk) {
        for (int j = tid; j < nkeys; j += kBlock) mrow[j] = 1;
        return;
    }

    extern __shared__ float logit[];  // [nkeys]
    const __nv_bfloat16* qi = idx_q + static_cast<long long>(i) * H * D;
    const __nv_bfloat16* wi = idx_w + static_cast<long long>(i) * H;
    for (int j = tid; j < nkeys; j += kBlock) {
        const __nv_bfloat16* kj = idx_k + static_cast<long long>(j) * D;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const __nv_bfloat16* qh = qi + static_cast<long long>(h) * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += __bfloat162float(qh[d]) * __bfloat162float(kj[d]);
            acc += fmaxf(dot, 0.f) * __bfloat162float(wi[h]);
        }
        logit[j] = acc;
    }
    __syncthreads();

    __shared__ float lo_s, hi_s;
    if (tid == 0) {
        float lo = CUDART_INF_F, hi = -CUDART_INF_F;
        for (int j = 0; j < nkeys; ++j) { lo = fminf(lo, logit[j]); hi = fmaxf(hi, logit[j]); }
        lo_s = lo; hi_s = hi;
    }
    __syncthreads();
    float lo = lo_s, hi = hi_s;
    __shared__ int cnt_s;
    float thr = hi;
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (tid == 0) cnt_s = 0;
        __syncthreads();
        int c = 0;
        for (int j = tid; j < nkeys; j += kBlock) if (logit[j] >= mid) c++;
        atomicAdd(&cnt_s, c);
        __syncthreads();
        const int cnt = cnt_s;
        if (cnt > topk) lo = mid; else hi = mid;
        __syncthreads();
        thr = hi;
    }
    for (int j = tid; j < nkeys; j += kBlock) mrow[j] = (logit[j] >= thr) ? 1 : 0;
}

}  // namespace

void launch_dsa_index_knorm_rope_bf16(
    void* idx_k, const void* k_norm_weight, const void* k_norm_bias,
    const std::int32_t* positions, int tokens, int head_dim, int rope_dim,
    float theta, float eps, cudaStream_t stream)
{
    if (tokens <= 0) return;
    index_knorm_rope_kernel<<<tokens, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(idx_k),
        static_cast<const __nv_bfloat16*>(k_norm_weight),
        static_cast<const __nv_bfloat16*>(k_norm_bias),
        positions, head_dim, rope_dim, theta, eps);
}

void launch_dsa_index_q_rope_bf16(
    void* idx_q, const std::int32_t* positions, int tokens, int n_heads,
    int head_dim, int rope_dim, float theta, cudaStream_t stream)
{
    if (tokens <= 0) return;
    int block = ((n_heads + 31) / 32) * 32;
    if (block < 32) block = 32;
    index_q_rope_kernel<<<tokens, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(idx_q), positions,
        n_heads, head_dim, rope_dim, theta);
}

void launch_dsa_index_topk_mask(
    const void* idx_q, const void* idx_k, const void* idx_w,
    std::uint8_t* mask, int tokens, int n_heads, int head_dim, int topk,
    cudaStream_t stream)
{
    if (tokens <= 0) return;
    const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
    index_topk_mask_kernel<<<tokens, kBlock, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(idx_q),
        static_cast<const __nv_bfloat16*>(idx_k),
        static_cast<const __nv_bfloat16*>(idx_w),
        mask, tokens, n_heads, head_dim, topk);
}

}  // namespace pie_cuda_driver::kernels
