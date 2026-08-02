#pragma once

// Device-side RoPE primitives, shared so that a fused kernel and the standalone
// `rope_bf16_kernel` cannot drift apart numerically. Anything that rotates a
// bf16 pair in this driver must call through here; a second copy of these three
// lines is a bit-exactness bug waiting to happen.

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

// NeoX / half-and-half pairing: dim `i` rotates against `i + head_dim/2`.
__device__ __forceinline__ void rotate_pair(
    __nv_bfloat16* h_ptr, int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(h_ptr[dim_pair]);
    const float b = __bfloat162float(h_ptr[dim_pair + half]);
    h_ptr[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
    h_ptr[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
}

// GPT-J / interleaved pairing: adjacent dims (2i, 2i+1). Required by GLM
// (`rope_interleave=true`) and by DeepSeek-V2/V3/Kimi's q_pe/k_pe.
__device__ __forceinline__ void rotate_pair_interleaved(
    __nv_bfloat16* h_ptr, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(h_ptr[2 * dim_pair]);
    const float b = __bfloat162float(h_ptr[2 * dim_pair + 1]);
    h_ptr[2 * dim_pair]     = __float2bfloat16(a * cos_v - b * sin_v);
    h_ptr[2 * dim_pair + 1] = __float2bfloat16(b * cos_v + a * sin_v);
}

// Out-of-place interleaved rotation: `src` and `dst` may alias.
__device__ __forceinline__ void rotate_pair_interleaved_to(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(src[2 * dim_pair]);
    const float b = __bfloat162float(src[2 * dim_pair + 1]);
    dst[2 * dim_pair]     = __float2bfloat16(a * cos_v - b * sin_v);
    dst[2 * dim_pair + 1] = __float2bfloat16(b * cos_v + a * sin_v);
}

__device__ __forceinline__ void rotate_pair_to(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(src[dim_pair]);
    const float b = __bfloat162float(src[dim_pair + half]);
    dst[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
    dst[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
}

// The angle for one (position, dim_pair). Kept as one expression because the
// `powf`/`__sincosf` pair is what every rope variant has to reproduce exactly.
__device__ __forceinline__ void rope_cos_sin(
    float theta, int dim_pair, int head_dim, int pos,
    float& cos_v, float& sin_v)
{
    const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                   static_cast<float>(head_dim));
    const float ang = static_cast<float>(pos) * freq;
    __sincosf(ang, &sin_v, &cos_v);
}

// Linear ramp over dim index: 0 below low_dim, 1 above high_dim. Blends
// between unscaled (high freq) and `1/factor`-scaled (low freq) inv_freq, in
// the dim-index domain rather than the wavelen domain Llama-3 YaRN uses.
__device__ __forceinline__ float yarn_original_freq(
    float base_freq, float factor,
    float low_dim, float high_dim, int dim_pair)
{
    const float denom = (high_dim == low_dim) ? (high_dim + 1e-3f - low_dim)
                                              : (high_dim - low_dim);
    float ramp = (static_cast<float>(dim_pair) - low_dim) / denom;
    if (ramp < 0.f) ramp = 0.f;
    if (ramp > 1.f) ramp = 1.f;
    return base_freq * ((1.f - ramp) + ramp / factor);
}

__device__ __forceinline__ void rope_cos_sin_yarn_original(
    float theta, int dim_pair, int head_dim, int pos,
    float factor, float low_dim, float high_dim, float mscale,
    float& cos_v, float& sin_v)
{
    const float base_freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                        static_cast<float>(head_dim));
    const float freq =
        yarn_original_freq(base_freq, factor, low_dim, high_dim, dim_pair);
    __sincosf(static_cast<float>(pos) * freq, &sin_v, &cos_v);
    cos_v *= mscale;
    sin_v *= mscale;
}

// Host-side ramp bounds for the original-YaRN variant. Shared so a fused
// kernel and `launch_rope_yarn_original_bf16` cannot disagree about where the
// ramp starts, which would silently change every position > 0.
inline void yarn_original_ramp_bounds(
    int head_dim, float theta, float beta_fast, float beta_slow,
    int original_max_position, float& low_dim, float& high_dim)
{
    constexpr float TWO_PI = 6.2831853071795864769f;
    const float ln_theta = logf(theta);
    auto corr_dim = [&](float rot) -> float {
        return head_dim * logf(static_cast<float>(original_max_position) /
                               (rot * TWO_PI)) / (2.f * ln_theta);
    };
    low_dim = floorf(corr_dim(beta_fast));
    high_dim = ceilf(corr_dim(beta_slow));
    if (low_dim < 0.f) low_dim = 0.f;
    const float max_pair = static_cast<float>(head_dim / 2) - 1.f;
    if (high_dim > max_pair) high_dim = max_pair;
    if (high_dim < low_dim) high_dim = low_dim;
}

}  // namespace pie_cuda_driver::kernels
