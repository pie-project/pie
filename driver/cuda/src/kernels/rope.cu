#include "kernels/rope.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// One block per token; threads cover the full QK head_dim grid:
// (head, dim_pair_idx). For Qwen the convention pairs index `i` with
// `i + head_dim/2`, with frequency theta^(-2*i / head_dim).
__device__ __forceinline__ void rotate_pair(
    __nv_bfloat16* h_ptr, int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(h_ptr[dim_pair]);
    const float b = __bfloat162float(h_ptr[dim_pair + half]);
    h_ptr[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
    h_ptr[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
}

__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[n];

    // Each thread handles one (head, dim_pair_idx).
    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                       static_cast<float>(head_dim));
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

}  // namespace

void launch_rope_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    rope_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta);
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
    float orig_max_pos)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
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
    rope_yarn_bf16_kernel<<<num_tokens, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_freq_factor, high_freq_factor,
        static_cast<float>(original_max_position));
}

// ── Original YaRN variant (OLMo-3, gpt-oss) ───────────────────────────────

namespace {

// Linear ramp over dim index: 0 below low_dim, 1 above high_dim. Used
// to blend between unscaled (high freq) and `1/factor`-scaled (low
// freq) inv_freq, in the dim-index domain rather than the wavelen
// domain that Llama-3 YaRN uses.
__device__ __forceinline__ float yarn_original_freq(
    float base_freq, float factor,
    float low_dim, float high_dim, int dim_pair)
{
    const float denom = (high_dim == low_dim) ? (high_dim + 1e-3f - low_dim)
                                              : (high_dim - low_dim);
    float ramp = (static_cast<float>(dim_pair) - low_dim) / denom;
    if (ramp < 0.f) ramp = 0.f;
    if (ramp > 1.f) ramp = 1.f;
    // Below low_dim (ramp=0): extrapolation = base. Above high_dim
    // (ramp=1): interpolation = base / factor. Linear blend between.
    return base_freq * ((1.f - ramp) + ramp / factor);
}

__global__ void rope_yarn_original_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_dim, float high_dim,
    float mscale)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        const float base_freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float freq = yarn_original_freq(base_freq, factor,
                                              low_dim, high_dim, dim_pair);
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        cos_v *= mscale;
        sin_v *= mscale;

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

void launch_rope_yarn_original_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float beta_fast, float beta_slow,
    float attention_factor,
    int original_max_position,
    cudaStream_t stream)
{
    constexpr float TWO_PI = 6.2831853071795864769f;
    // correction_dim(rot) = head_dim * ln(max_pos / (rot * 2π)) / (2 * ln(theta))
    const float ln_theta = logf(theta);
    auto corr_dim = [&](float rot) -> float {
        return head_dim * logf(static_cast<float>(original_max_position) /
                               (rot * TWO_PI)) / (2.f * ln_theta);
    };
    // beta_slow → "low rotation count" → larger correction_dim → upper
    // bound on the ramp (above this, fully interpolated). beta_fast →
    // smaller correction_dim → lower bound (below this, fully
    // extrapolated). HF clamps to [0, head_dim/2 - 1] (we ramp over
    // dim_pair which has range [0, head_dim/2)).
    float low_dim  = floorf(corr_dim(beta_fast));
    float high_dim = ceilf(corr_dim(beta_slow));
    if (low_dim < 0.f) low_dim = 0.f;
    const float max_pair = static_cast<float>(head_dim / 2) - 1.f;
    if (high_dim > max_pair) high_dim = max_pair;
    if (high_dim < low_dim)  high_dim = low_dim;

    constexpr int BLOCK = 256;
    rope_yarn_original_bf16_kernel<<<num_tokens, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_dim, high_dim, attention_factor);
}

// ── Partial rotary (Gemma-4 full-attention layers) ─────────────────────────

namespace {

// Proportional RoPE (Gemma-4 full-attention layers, HF reference).
//
// HF builds the frequency table as `freq[k] = 1 / theta^(2k/head_dim)`
// for k ∈ [0, rotary_dim/2), then pads the rest of the head's lower-
// half dim entries with `cos=1 / sin=0` (identity). The pair offset
// is the *full* `head_dim/2`, NOT `rotary_dim/2` — every dim in the
// lower half rotates with its mate in the upper half, but the
// rotation angle is zero for k ≥ rotary_dim/2 (so those pairs pass
// through unchanged).
//
// Two ways the previous draft of this kernel got it wrong:
//   1. used `rotary_dim` as the frequency denominator instead of
//      `head_dim` — wrong angle progression.
//   2. used `rotary_dim/2` as the pair offset instead of
//      `head_dim/2` — paired the wrong dims with each other.
__global__ void rope_partial_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        float cos_v = 1.f, sin_v = 0.f;
        if (dim_pair < rope_angles) {
            const float freq = powf(theta,
                -2.f * static_cast<float>(dim_pair) /
                       static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        // Skip identity rotations entirely — `dim_pair ≥ rope_angles`
        // multiplies the pair by [[1,0],[0,1]] which is a no-op.
        if (dim_pair >= rope_angles) continue;

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = __bfloat162float(qp[dim_pair]);
            const float b = __bfloat162float(qp[dim_pair + half]);
            qp[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
            qp[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = __bfloat162float(kp[dim_pair]);
            const float b = __bfloat162float(kp[dim_pair + half]);
            kp[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
            kp[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
        }
    }
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
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    rope_partial_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
}

}  // namespace pie_cuda_driver::kernels
