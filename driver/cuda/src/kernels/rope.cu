#include "kernels/rope.hpp"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "kernels/rope_device.cuh"

namespace pie_cuda_driver::kernels {

namespace {

// One block per token; threads cover the full QK head_dim grid:
// (head, dim_pair_idx). For Qwen the convention pairs index `i` with
// `i + head_dim/2`, with frequency theta^(-2*i / head_dim).
// `rotate_pair` / `rotate_pair_interleaved` live in rope_device.cuh so the
// fused MLA-prepare kernel rotates through byte-identical code.

__global__ void rope_standard_table_kernel(
    const std::int32_t* __restrict__ positions,
    float* __restrict__ table,
    int head_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int half = head_dim / 2;
    const int pos = positions[n];
    float* row = table + static_cast<long long>(n) * head_dim;
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += blockDim.x) {
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = cos_v;
        row[dim_pair + half] = sin_v;
    }
}

__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    bool interleaved,
    int cache_pairs,
    int heads_per_block)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[n];

    // The rotation angle depends only on (pos, dim_pair): every head of this
    // token shares it. Computing it inside the element loop ran a full-precision
    // `powf` plus a `__sincosf` once per (head, pair) -- for GLM's 65 QK heads
    // that is 65 evaluations of the same 32 transcendentals, and it made this
    // kernel cost more than the attention it feeds. Hoisting them into shared
    // memory keeps the arithmetic identical, so the outputs are bit-for-bit
    // what the per-element form produced.
    extern __shared__ float rope_cs[];
    const int cached = cache_pairs;
    for (int dp = threadIdx.x; dp < cached; dp += blockDim.x) {
        const float freq = powf(theta, -2.f * static_cast<float>(dp) /
                                       static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        rope_cs[dp] = c;
        rope_cs[cached + dp] = s;
    }
    if (cached > 0) __syncthreads();

    // Each thread handles one (head, dim_pair_idx).
    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;

        float cos_v, sin_v;
        if (dim_pair < cached) {
            cos_v = rope_cs[dim_pair];
            sin_v = rope_cs[cached + dim_pair];
        } else {
            const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                           static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cos_v, sin_v);
            else rotate_pair(qp, half, dim_pair, cos_v, sin_v);
            continue;
        }
        {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cos_v, sin_v);
            else rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    __nv_bfloat16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = __bfloat162float(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = __bfloat162float(row[dim_pair]) *
            inv_rms * __bfloat162float(weight[dim_pair]);
        const float b = __bfloat162float(row[dim_pair + half]) *
            inv_rms * __bfloat162float(weight[dim_pair + half]);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = __float2bfloat16(a * cos_v - b * sin_v);
        row[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rope_bf16_rounded_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    __nv_bfloat16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = __bfloat162float(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const __nv_bfloat16 norm_a = __float2bfloat16(
            __bfloat162float(row[dim_pair]) *
            inv_rms * __bfloat162float(weight[dim_pair]));
        const __nv_bfloat16 norm_b = __float2bfloat16(
            __bfloat162float(row[dim_pair + half]) *
            inv_rms * __bfloat162float(weight[dim_pair + half]));
        const float a = __bfloat162float(norm_a);
        const float b = __bfloat162float(norm_b);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = __float2bfloat16(a * cos_v - b * sin_v);
        row[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
    }
}

// Fused per-head Q/K RMSNorm + interleaved M-RoPE (Qwen3-VL text tower).
// Reads three position components per token (`positions[3n+axis]`, axis
// 0=t,1=h,2=w) and selects the rotary axis per frequency index using the
// interleaved layout (HF `apply_interleaved_mrope`):
//   freqs_t = freqs[T]; H overwrites idx slice(1, 3*s1, 3); W slice(2, 3*s2, 3)
// i.e. for dim_pair j: axis = H if (j%3==1 && j < 3*s1); W if (j%3==2 && j<3*s2);
// otherwise T. The rotation itself is the standard half/half rotate_half pairing
// (j, j+head_dim/2) with frequency theta^(-2j/head_dim). Preserves the
// bf16(rmsnorm(x)) materialization point (parity-sensitive, like Gemma-4).
template <int BLOCK>
__global__ void qk_rmsnorm_mrope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,  // [num_tokens, 3] (t,h,w)
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    int s0, int s1, int s2)  // mrope_section (t,h,w)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    __nv_bfloat16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = __bfloat162float(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos_t = positions[3 * n + 0];
    const int pos_h = positions[3 * n + 1];
    const int pos_w = positions[3 * n + 2];
    (void)s0;
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const __nv_bfloat16 norm_a = __float2bfloat16(
            __bfloat162float(row[dim_pair]) *
            inv_rms * __bfloat162float(weight[dim_pair]));
        const __nv_bfloat16 norm_b = __float2bfloat16(
            __bfloat162float(row[dim_pair + half]) *
            inv_rms * __bfloat162float(weight[dim_pair + half]));
        const float a = __bfloat162float(norm_a);
        const float b = __bfloat162float(norm_b);

        // Interleaved axis selection.
        int axis_pos;
        const int m = dim_pair % 3;
        if (m == 1 && dim_pair < 3 * s1)      axis_pos = pos_h;
        else if (m == 2 && dim_pair < 3 * s2) axis_pos = pos_w;
        else                                  axis_pos = pos_t;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(axis_pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = __float2bfloat16(a * cos_v - b * sin_v);
        row[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
    }
}

}  // namespace

void launch_qk_rmsnorm_mrope_bf16(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_q_heads + num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    qk_rmsnorm_mrope_bf16_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(q_weight),
        static_cast<const __nv_bfloat16*>(k_weight),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, eps,
        mrope_section_t, mrope_section_h, mrope_section_w);
}

void launch_rope_standard_table(
    const std::int32_t* positions,
    float* table,
    int num_tokens,
    int head_dim,
    float theta,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || table == nullptr) return;
    constexpr int BLOCK = 128;
    rope_standard_table_kernel<<<num_tokens, BLOCK, 0, stream>>>(
        positions, table, head_dim, theta);
}

void launch_rope_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream,
    bool interleaved)
{
    constexpr int BLOCK = 256;
    // 32 KB caps the table at head_dim 8192; past that the pairs are recomputed.
    constexpr int kMaxCachedPairs = 4096;
    const int half = head_dim / 2;
    if (half <= 0) return;
    const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    const std::size_t smem = static_cast<std::size_t>(cache_pairs) * 2 * sizeof(float);
    // Splitting the heads across blockIdx.y keeps every SM fed at decode, where
    // `num_tokens` is 1 and a 1-D grid would run a single block on 148 SMs.
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    dim3 grid(num_tokens, (total_heads + heads_per_block - 1) / heads_per_block);
    dim3 block(BLOCK);
    rope_bf16_kernel<<<grid, block, smem, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, interleaved, cache_pairs,
        heads_per_block);
}

void launch_qk_rmsnorm_rope_bf16(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 128;
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    qk_rmsnorm_rope_bf16_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(q_weight),
        static_cast<const __nv_bfloat16*>(k_weight),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, eps);
}

void launch_qk_rmsnorm_rope_bf16_rounded(
    void* q, void* k,
    const void* q_weight, const void* k_weight,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_q_heads + num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    qk_rmsnorm_rope_bf16_rounded_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(q_weight),
        static_cast<const __nv_bfloat16*>(k_weight),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, eps);
}

// ── YaRN variant ────────────────────────────────────────────────────────────

namespace {

// Piecewise-linear interp between full-scale (high-freq pairs, kept
// untouched) and `factor`-scaled (low-freq pairs); smooth band uses
// `(orig_max_pos / wavelen - low_freq_factor) / (high - low)` blended.
__device__ __forceinline__ float yarn_freq(
    float base_freq, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos)
{
    constexpr float TWO_PI = 6.2831853071795864769f;
    const float wavelen   = TWO_PI / base_freq;
    const float low_wave  = orig_max_pos / low_freq_factor;
    const float high_wave = orig_max_pos / high_freq_factor;
    if (wavelen < high_wave) return base_freq;            // high-freq: no scale
    if (wavelen > low_wave)  return base_freq / factor;   // low-freq: full scale
    const float smooth = (orig_max_pos / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor);
    return (1.f - smooth) * (base_freq / factor) + smooth * base_freq;
}

__global__ void rope_yarn_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos,
    int heads_per_block)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;

        const float base_freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float freq = yarn_freq(base_freq, factor,
                                     low_freq_factor, high_freq_factor,
                                     orig_max_pos);
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            rotate_pair(qp, half, dim_pair, cos_v, sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

}  // namespace

void launch_rope_yarn_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    int original_max_position,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int half = head_dim / 2;
    if (half <= 0) return;
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    const dim3 grid(num_tokens,
                    (total_heads + heads_per_block - 1) / heads_per_block);
    rope_yarn_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_freq_factor, high_freq_factor,
        static_cast<float>(original_max_position),
        heads_per_block);
}

// ── Original YaRN variant (OLMo-3, gpt-oss) ───────────────────────────────

namespace {

// `yarn_original_freq` lives in rope_device.cuh, shared with the fused
// MLA-prepare kernel.

__global__ void rope_yarn_original_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_dim, float high_dim,
    float mscale,
    bool interleaved,
    int heads_per_block,
    int cache_pairs)
{
    extern __shared__ float2 yarn_cs[];
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    // The rotation angle depends only on `dim_pair` and the token's position,
    // not on the head -- so a block that covers many heads was recomputing the
    // same `powf` and `__sincosf` once per head. Do it `half` times instead of
    // `heads_per_block * half` times and share the result.
    auto angle = [&](int d) -> float2 {
        const float base_freq = powf(theta,
            -2.f * static_cast<float>(d) / static_cast<float>(head_dim));
        const float freq = yarn_original_freq(base_freq, factor,
                                              low_dim, high_dim, d);
        float cos_v, sin_v;
        __sincosf(static_cast<float>(pos) * freq, &sin_v, &cos_v);
        return make_float2(cos_v * mscale, sin_v * mscale);
    };
    for (int d = threadIdx.x; d < cache_pairs; d += blockDim.x) {
        yarn_cs[d] = angle(d);
    }
    if (cache_pairs > 0) __syncthreads();

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;
        const float2 cs = dim_pair < cache_pairs ? yarn_cs[dim_pair]
                                                 : angle(dim_pair);

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cs.x, cs.y);
            else             rotate_pair(qp, half, dim_pair, cs.x, cs.y);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cs.x, cs.y);
            else             rotate_pair(kp, half, dim_pair, cs.x, cs.y);
        }
    }
}

}  // namespace

void launch_rope_yarn_original_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float beta_fast, float beta_slow,
    float attention_factor,
    int original_max_position,
    cudaStream_t stream,
    bool interleaved)
{
    // correction_dim(rot) = head_dim * ln(max_pos / (rot * 2π)) / (2 * ln(theta)).
    // beta_slow → "low rotation count" → larger correction_dim → upper bound on
    // the ramp (above this, fully interpolated). beta_fast → smaller
    // correction_dim → lower bound (below this, fully extrapolated). HF clamps
    // to [0, head_dim/2 - 1].
    float low_dim = 0.f, high_dim = 0.f;
    yarn_original_ramp_bounds(head_dim, theta, beta_fast, beta_slow,
                              original_max_position, low_dim, high_dim);

    constexpr int BLOCK = 256;
    // One block per token leaves 147 of the B200's 148 SMs idle during decode,
    // where `num_tokens` is 1. Give each block a slice of the heads instead, so
    // the grid grows with the head count rather than the batch, and each thread
    // owns exactly one element -- one load/store round trip rather than a chain
    // of them.
    constexpr int kMaxCachedPairs = 4096;   // 32 KB of float2
    const int half = head_dim / 2;
    if (half <= 0) return;
    const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    const int total_heads = num_q_heads + num_kv_heads;
    const int heads_per_block = half >= BLOCK ? 1 : (BLOCK / half);
    const dim3 grid(num_tokens,
                    (total_heads + heads_per_block - 1) / heads_per_block);
    const std::size_t shared =
        static_cast<std::size_t>(cache_pairs) * sizeof(float2);
    rope_yarn_original_bf16_kernel<<<grid, BLOCK, shared, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_dim, high_dim, attention_factor, interleaved,
        heads_per_block, cache_pairs);
}

// ── Partial rotary (Gemma-4 full-attention layers) ─────────────────────────

namespace {

// Proportional RoPE (Gemma-4 full-attention layers, HF reference).
//
// Partial rotary rotates ONLY the first `rotary_dim` channels and leaves
// `[rotary_dim, head_dim)` untouched. HF applies `rotate_half` to the
// slice `x[..., :rotary_dim]`, so the pair offset is `rotary_dim/2` and
// the frequency denominator is `rotary_dim`:
//
//     freq[j] = theta^(-2j/rotary_dim),  pair (j, j + rotary_dim/2)
//
// The comment that stood here previously asserted the opposite — a
// `head_dim` denominator with a `head_dim/2` pair offset, on the theory
// that HF pads the table with identity rotations — and called the correct
// form "the previous draft [that] got it wrong". That is backwards, and
// for `head_dim=256, rotary_dim=64` it is not a small error:
//
//   * dims 32..63 were left UNROTATED; they are the second half of each
//     pair and must rotate.
//   * dims 128..159 were OVERWRITTEN; they are pass-through.
//   * the frequency denominator was 4x too large, so the angle
//     progression was wrong by up to 1.2e5 at j=31.
//
// Three independent references agree on the form above. HF's
// `modeling_qwen3_5.py` rotate_half's the `rotary_dim` slice. THIS REPO's
// own Metal driver (`driver/metal/src/kernels/rope.metal`) uses
// `half = rope_dims/2` and documents "Channels [rope_dims, head_dim) are
// pass-through (untouched)", citing MLX `fast::rope(traditional=false,
// dims=rope_dims)`. And numerically, against a double-precision HF
// reference at head_dim=256/rotary_dim=64/theta=1e7, this form reproduces
// it to 4.4e-16 while the previous form was off by 3.1.
//
// Only partial-rotary models were affected: both errors vanish when
// `rotary_dim == head_dim`. Qwen3.6-27B sets partial_rotary_factor=0.25,
// and its 16 full-attention layers are the ones that carry position — the
// 48 GDN layers take no position embeddings at all. The observable was a
// systematic ~2.2 nat logit disagreement against vLLM that grows with
// relative distance and is exactly zero at distance 0.
__global__ void rope_partial_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int position_delta,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int rope_angles = rotary_dim / 2;
    const int pos = positions[n] + position_delta;

    // One thread per rotated pair. Channels [rotary_dim, head_dim) are never
    // visited, which is what "pass-through" means -- writing them back
    // unchanged would be equivalent but this cannot touch them by accident.
    for (int t = threadIdx.x; t < total_heads * rope_angles; t += blockDim.x) {
        const int head_idx = t / rope_angles;
        const int dim_pair = t % rope_angles;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = __bfloat162float(qp[dim_pair]);
            const float b = __bfloat162float(qp[dim_pair + rope_angles]);
            qp[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
            qp[dim_pair + rope_angles] = __float2bfloat16(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = __bfloat162float(kp[dim_pair]);
            const float b = __bfloat162float(kp[dim_pair + rope_angles]);
            kp[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
            kp[dim_pair + rope_angles] = __float2bfloat16(b * cos_v + a * sin_v);
        }
    }
}

// vLLM-shaped partial rotary. Same channel semantics as the kernel above --
// rotate [0, rotary_dim) with pair offset rotary_dim/2, leave
// [rotary_dim, head_dim) untouched -- and a different numeric pipeline:
// vLLM's inv_freq exponent form, accurate fp32 trig, cos/sin ROUNDED TO BF16
// before use, and a bf16 rotate. See the block comment in rope_device.cuh.
//
// Kept as a separate kernel rather than a branch inside
// `rope_partial_bf16_kernel` so the default path stays byte-for-byte the code
// that shipped; the knob is meant to be a measurable A/B, and a shared kernel
// body would make "unset is bit-unchanged" an argument instead of a fact.
//
// The per-token cos/sin row is built once in shared memory and indexed by
// every head, which is literally the "index the table" step: `rope_angles` is
// 32 for Qwen3.5-9B while `total_heads` is 10, so the naive form would
// evaluate the same 32 transcendentals ten times.
__global__ void rope_partial_vllm_table_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int position_delta,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    const __nv_bfloat16* __restrict__ table,
    int table_capacity,
    unsigned int* __restrict__ oob)
{
    extern __shared__ __nv_bfloat16 rope_tab[];
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int rope_angles = rotary_dim / 2;
    const int pos = positions[n] + position_delta;

    // Uniform across the block: `pos` does not depend on threadIdx.
    const bool in_table =
        table != nullptr && pos >= 0 && pos < table_capacity;

    for (int j = threadIdx.x; j < rope_angles; j += blockDim.x) {
        if (in_table) {
            const long long row = static_cast<long long>(pos) * rotary_dim;
            rope_tab[j] = table[row + j];
            rope_tab[rope_angles + j] = table[row + rope_angles + j];
        } else {
            // Past the table. Degrades to device trig -- i.e. to what this
            // path did before the table existed -- which is NOT bit-parity
            // with the reference. Counted so it cannot be silent.
            __nv_bfloat16 cos_b, sin_b;
            rope_cos_sin_vllm_table(theta, j, rotary_dim, pos, cos_b, sin_b);
            rope_tab[j] = cos_b;
            rope_tab[rope_angles + j] = sin_b;
        }
    }
    if (!in_table && threadIdx.x == 0 && oob != nullptr) atomicAdd(oob, 1u);
    __syncthreads();

    for (int t = threadIdx.x; t < total_heads * rope_angles; t += blockDim.x) {
        const int head_idx = t / rope_angles;
        const int dim_pair = t % rope_angles;
        const __nv_bfloat16 cos_b = rope_tab[dim_pair];
        const __nv_bfloat16 sin_b = rope_tab[rope_angles + dim_pair];

        __nv_bfloat16* base;
        if (head_idx < num_q_heads) {
            base = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
        } else {
            const int kv_h = head_idx - num_q_heads;
            base = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
        }
        rotate_pair_bf16(base, dim_pair, dim_pair + rope_angles, cos_b, sin_b);
    }
}

// ── The cos/sin table, built on the HOST ───────────────────────────────────
//
// This is the part that makes the path faithful rather than merely similar.
// vLLM evaluates cos/sin once at init, on the host, and indexes the result
// thereafter; no GPU kernel of its own ever computes trig. Pie evaluating the
// same formula on the device is NOT equivalent, and the difference was
// measured rather than argued: over 30001 positions x 64 lanes, a device-trig
// implementation of this path matched the reference on 1,920,060 of 1,920,064
// entries and differed on 4 --
//
//   pos=4498   cos[ 8]  +1 ulp    0.3451 fp32 ulp from a bf16 midpoint
//   pos=7460   sin[16]  +1 ulp    0.1156
//   pos=13848  cos[ 7]  +1 ulp    0.7032
//   pos=21467  cos[ 6]  -1 ulp    0.5967
//
// -- every one exactly one bf16 ulp, every one within 0.71 fp32 ulp of a
// rounding midpoint. That is the signature of two independent ~1-ulp trig
// implementations disagreeing about which side of a boundary a value falls on:
// 52 entries in that population lie within 1 fp32 ulp of a midpoint and 4 of
// them flipped, ~8%. It is NOT a rounding-rule difference -- CUDA's
// `__float2bfloat16` and torch's `.to(bfloat16)` are both round-to-nearest-
// even, so a tie breaks identically on both sides no matter how many ties
// there are.
//
// There is no device-side fix for that. Correct rounding would not help,
// because the reference is not correctly rounded either; the only way to get
// the reference's bits is to run the reference's evaluation. So the table is
// built here with the host libm, exactly as the fixture in
// tests/rope_vllm_cos_sin_golden.hpp is built: fp32 inv_freq in vLLM's form,
// fp32 angle, trig in double, rounded to fp32, then rounded once to bf16 (RNE,
// which is what `.to(torch.bfloat16)` does).
//
// Whether this host's libm agrees with the reference host's on every entry is
// an EMPIRICAL question, not something the construction guarantees. It is the
// right structure and the best available shot; it is not a proof of parity.
//
// CAPACITY. Indexed by absolute position, so the table must span the context.
// Default 262144 positions (33.5 MB at rotary_dim=64), overridable with
// PIE_ROPE_VLLM_TABLE_MAX_POS. Beyond it the kernel degrades to device trig --
// i.e. to the 4-in-1.9M behaviour above -- and bumps `oob` so that degradation
// cannot be silent. The right fix is to size this from
// `cfg.max_position_embeddings`, which is in scope at all eight call sites;
// that is deferred to the change that flips the default, because it means
// touching those call sites and re-measuring.

std::uint16_t bf16_rne_bits(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return static_cast<std::uint16_t>((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

// Which host trig builds the table. THIS CHOICE IS EMPIRICAL, NOT PRINCIPLED,
// and the honest version of the reasoning is uncomfortable enough to be worth
// writing down.
//
// The reference's trig is Intel MKL VML, not SLEEF and not glibc. Measured:
// `vmsCos`/`vmsSin` with torch's exact mode word
// `VML_HA|VML_FTZDAZ_OFF|VML_ERRMODE_IGNORE` (0x140102) reproduce torch
// bit-for-bit, 0 diffs over 960,032 angles. `ATen/cpu/vml.h` specialises
// vcos/vsin onto MKL whenever `AT_MKL_ENABLED() && !__APPLE__`, so
// `Vectorized<float>::cos()` is never reached on an x86 torch build -- which is
// what the reference runs.
//
// We cannot call MKL from here, so the question is only which available
// implementation lands on the reference's side of a bf16 midpoint most often.
// The direction is settled: correct rounding is NOT the goal. MKL VML HA is
// itself ~0.60 ulp and not correctly rounded, so being similarly-imperfect
// agrees with it more often than being perfect does. That is correlation of
// error, not correctness -- glibc is not right here, it is wrong in a way that
// sometimes resembles MKL. A glibc version bump could move it, and nothing
// would notice except the parity test, which is why that test pins measured
// reference bits rather than a formula.
//
// BUT THE DEFAULT IS THE CORRECTLY-ROUNDED BUILD ANYWAY, because the win is
// one entry and the cost is determinism.
//
// Measured over positions 0..262143 (16,777,216 angles) against the reference:
//
//   double -> round to fp32 (correctly rounded) : 19 mismatches
//   plain cosf/sinf                             : 18 mismatches
//
// The single entry is pos=70308 cos[5]. `cosf`/`sinf` are ALMOST correctly
// rounded here: they differ from the double-rounded value on 1.18% of those
// angles, always by exactly 1 fp32 ulp, and bf16 absorbs all but ONE. The two
// backends are very nearly the same function.
//
// `exact` wins on determinism. Correct rounding has exactly one answer, so the
// table it produces depends on nothing -- not the C library, its version, the
// compiler, or the CPU. `libm` is by definition whatever the linked libm does,
// which would make Pie's table a property of the machine it was built on. For
// a parity campaign that is a real liability, and it is not worth trading for
// one entry in 16.7 million.
//
// TWO CLAIMS ABOUT THIS THAT ARE NOT TRUE, recorded so they are not re-derived:
//
//   * "plain cosf/sinf scores 0 in the campaign window and 1 overall." Not
//     reproduced. On the deployment base it is 18, and 13852 -- the one
//     in-window entry -- is among the 18 it misses.
//   * "the error correlation moves with the glibc version, 2.35 vs 2.41."
//     It does not. Both were measured directly, in ubuntu:22.04 (glibc 2.35)
//     and debian:13 (glibc 2.41), and the per-entry output is BYTE-IDENTICAL:
//     18/19 either way, and the same 1.1794% fp32 divergence. Whatever produced
//     0/1 elsewhere, it was not the glibc version.
//
// The deployment target is glibc 2.35 regardless:
// `backend/docker/eval-worker-base.Dockerfile` pins
// PIE_CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04, so the
// decoder runs on Ubuntu 22.04.
//
// `PIE_ROPE_VLLM_TABLE_TRIG=libm` selects the error-correlated build, so the
// comparison stays reproducible and the choice revisitable rather than
// entombed. Do not chase the remaining 19: the reference's exact bits require
// vendoring closed-source, x86-only oneMKL into a CUDA driver's host-side table
// builder. The open question is whether any of this flips a token, which needs
// an end-to-end run, not more numerics.
enum class TableTrig { Libm, Exact };

TableTrig vllm_table_trig() {
    static const TableTrig mode = [] {
        const char* v = std::getenv("PIE_ROPE_VLLM_TABLE_TRIG");
        if (v != nullptr && std::strcmp(v, "libm") == 0) return TableTrig::Libm;
        return TableTrig::Exact;
    }();
    return mode;
}

int vllm_table_capacity() {
    static const int cap = [] {
        if (const char* v = std::getenv("PIE_ROPE_VLLM_TABLE_MAX_POS")) {
            const int n = std::atoi(v);
            if (n > 0) return n;
        }
        return 262144;
    }();
    return cap;
}

struct VllmCosSinTable {
    int device = -1;
    float theta = 0.f;
    int rotary_dim = 0;
    int capacity = 0;
    __nv_bfloat16* dev = nullptr;
    unsigned int* oob = nullptr;
};

// Keyed by device as well as by shape: tensor parallelism runs every rank in
// this one process, each bound to its own device, so a process-global table
// would hand rank 1 an allocation belonging to rank 0.
const VllmCosSinTable* vllm_table_for(float theta, int rotary_dim) {
    if (rotary_dim <= 0 || (rotary_dim % 2) != 0) return nullptr;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;

    static std::mutex mu;
    static std::vector<std::unique_ptr<VllmCosSinTable>> cache;
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& t : cache) {
        if (t->device == device && t->theta == theta &&
            t->rotary_dim == rotary_dim) {
            return t.get();
        }
    }

    const int angles = rotary_dim / 2;
    const int cap = vllm_table_capacity();

    // inv_freq in vLLM's form: exponent 2j/rotary_dim formed in fp32, the
    // power taken first and the reciprocal AFTER it.
    std::vector<float> inv(angles);
    for (int j = 0; j < angles; ++j) {
        const float expo = (2.f * static_cast<float>(j)) /
                           static_cast<float>(rotary_dim);
        const float p = static_cast<float>(
            std::pow(static_cast<double>(theta), static_cast<double>(expo)));
        inv[j] = 1.f / p;
    }

    const bool exact = vllm_table_trig() == TableTrig::Exact;
    std::vector<std::uint16_t> host(
        static_cast<std::size_t>(cap) * static_cast<std::size_t>(rotary_dim));
    for (int p = 0; p < cap; ++p) {
        std::uint16_t* row =
            host.data() + static_cast<std::size_t>(p) * rotary_dim;
        for (int j = 0; j < angles; ++j) {
            const float ang = static_cast<float>(p) * inv[j];  // fp32 product
            // `std::cos(float)` is the float overload, i.e. `cosf`. The double
            // form is the correctly-rounded one and is the WORSE match; see
            // `vllm_table_trig`.
            const float c = exact
                ? static_cast<float>(std::cos(static_cast<double>(ang)))
                : std::cos(ang);
            const float s = exact
                ? static_cast<float>(std::sin(static_cast<double>(ang)))
                : std::sin(ang);
            row[j] = bf16_rne_bits(c);
            row[angles + j] = bf16_rne_bits(s);
        }
    }

    auto entry = std::make_unique<VllmCosSinTable>();
    entry->device = device;
    entry->theta = theta;
    entry->rotary_dim = rotary_dim;
    entry->capacity = cap;

    const std::size_t bytes = host.size() * sizeof(std::uint16_t);
    if (cudaMalloc(&entry->dev, bytes) != cudaSuccess) return nullptr;
    if (cudaMalloc(&entry->oob, sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(entry->dev);
        return nullptr;
    }
    cudaMemset(entry->oob, 0, sizeof(unsigned int));

    // Uploaded on a private stream. The caller's stream may be under CUDA-graph
    // capture, where a copy is RECORDED rather than performed -- the table
    // would then be filled on replay instead of now, or not at all.
    cudaStream_t build = nullptr;
    if (cudaStreamCreateWithFlags(&build, cudaStreamNonBlocking) != cudaSuccess) {
        cudaFree(entry->dev);
        cudaFree(entry->oob);
        return nullptr;
    }
    cudaMemcpyAsync(entry->dev, host.data(), bytes, cudaMemcpyHostToDevice, build);
    cudaStreamSynchronize(build);
    cudaStreamDestroy(build);

    cache.push_back(std::move(entry));
    return cache.back().get();
}

// Opt-in, default OFF. A global numerics change would move every partial-rotary
// model -- Gemma-4 reaches this launcher too -- so the A/B gets measured before
// the default flips.
bool rope_vllm_table_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_ROPE_VLLM_TABLE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

void dispatch_rope_partial(
    void* q, void* k,
    const std::int32_t* positions,
    int position_delta,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream,
    bool vllm_table)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    if (vllm_table) {
        const std::size_t smem = static_cast<std::size_t>(rotary_dim / 2) * 2 *
                                 sizeof(__nv_bfloat16);
        const VllmCosSinTable* t = vllm_table_for(theta, rotary_dim);
        rope_partial_vllm_table_bf16_kernel<<<grid, block, smem, stream>>>(
            static_cast<__nv_bfloat16*>(q),
            static_cast<__nv_bfloat16*>(k),
            positions,
            position_delta,
            num_q_heads, num_kv_heads, head_dim, rotary_dim, theta,
            t != nullptr ? t->dev : nullptr,
            t != nullptr ? t->capacity : 0,
            t != nullptr ? t->oob : nullptr);
        return;
    }
    rope_partial_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        position_delta,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
}

}  // namespace

void launch_rope_partial_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream)
{
    dispatch_rope_partial(q, k, positions, 0, num_tokens, num_q_heads,
                          num_kv_heads, head_dim, rotary_dim, theta, stream,
                          rope_vllm_table_enabled());
}

void launch_rope_partial_bf16_position_delta(
    void* q, void* k,
    const std::int32_t* positions,
    int position_delta,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream)
{
    dispatch_rope_partial(q, k, positions, position_delta, num_tokens,
                          num_q_heads, num_kv_heads, head_dim, rotary_dim,
                          theta, stream, rope_vllm_table_enabled());
}

void launch_rope_partial_vllm_table_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream)
{
    dispatch_rope_partial(q, k, positions, 0, num_tokens, num_q_heads,
                          num_kv_heads, head_dim, rotary_dim, theta, stream,
                          /*vllm_table=*/true);
}

unsigned int rope_vllm_table_oob_blocks(float theta, int rotary_dim)
{
    const VllmCosSinTable* t = vllm_table_for(theta, rotary_dim);
    if (t == nullptr || t->oob == nullptr) return 0u;
    unsigned int n = 0u;
    cudaMemcpy(&n, t->oob, sizeof(n), cudaMemcpyDeviceToHost);
    return n;
}

int rope_vllm_table_capacity_for(float theta, int rotary_dim)
{
    const VllmCosSinTable* t = vllm_table_for(theta, rotary_dim);
    return t != nullptr ? t->capacity : 0;
}

const char* rope_vllm_table_trig_name()
{
    return vllm_table_trig() == TableTrig::Exact ? "exact" : "libm";
}

namespace {

__global__ void rope_partial_last_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int rope_half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * rope_half; t += blockDim.x) {
        const int head_idx = t / rope_half;
        const int dim_pair = t % rope_half;

        float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor,
                                      yarn_low_dim, yarn_high_dim, dim_pair);
        }
        const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        const bool is_q = (head_idx < num_q_heads);
        __nv_bfloat16* base = is_q
            ? q + static_cast<long long>(n * num_q_heads + head_idx) * head_dim
            : k + static_cast<long long>(n * num_kv_heads + (head_idx - num_q_heads)) * head_dim;

        // GPT-J pairing (adjacent dims) for DeepSeek-V4 (`is_neox_style=False`
        // in vLLM `build_deepseek_v4_rope`); NeoX half/half otherwise.
        const int i = interleaved ? offset + 2 * dim_pair : offset + dim_pair;
        const int j = interleaved ? offset + 2 * dim_pair + 1
                                  : offset + dim_pair + rope_half;
        const float a = __bfloat162float(base[i]);
        const float b = __bfloat162float(base[j]);
        base[i] = __float2bfloat16(a * cos_v - b * sin_v);
        base[j] = __float2bfloat16(b * cos_v + a * sin_v);
    }
}

}  // namespace

void launch_rope_partial_last_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_beta_fast,
    float yarn_beta_slow,
    int   yarn_original_max_position)
{
    // Same ramp as `launch_rope_yarn_original_bf16`, but the correction range
    // is over `rotary_dim` (the rotated slice), not the full head_dim.
    float low_dim = 0.f, high_dim = 0.f;
    if (yarn_factor > 1.f && yarn_original_max_position > 0) {
        constexpr float TWO_PI = 6.2831853071795864769f;
        const float ln_theta = logf(theta);
        auto corr_dim = [&](float rot) -> float {
            return rotary_dim * logf(static_cast<float>(yarn_original_max_position) /
                                     (rot * TWO_PI)) / (2.f * ln_theta);
        };
        low_dim  = floorf(corr_dim(yarn_beta_fast));
        high_dim = ceilf(corr_dim(yarn_beta_slow));
        if (low_dim < 0.f) low_dim = 0.f;
        const float max_pair = static_cast<float>(rotary_dim / 2) - 1.f;
        if (high_dim > max_pair) high_dim = max_pair;
        if (high_dim < low_dim)  high_dim = low_dim;
    }
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    rope_partial_last_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta, inverse,
        interleaved, yarn_factor, low_dim, high_dim);
}

}  // namespace pie_cuda_driver::kernels
