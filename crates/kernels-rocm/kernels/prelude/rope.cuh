#pragma once

#include "prelude/device.cuh"

namespace pie {

__device__ __forceinline__ void rotate_pair(
    bf16* h_ptr, int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = bf16_to_f32(h_ptr[dim_pair]);
    const float b = bf16_to_f32(h_ptr[dim_pair + half]);
    h_ptr[dim_pair]        = f32_to_bf16(a * cos_v - b * sin_v);
    h_ptr[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
}

__device__ __forceinline__ void rotate_pair_interleaved(
    bf16* h_ptr, int dim_pair, float cos_v, float sin_v)
{
    const float a = bf16_to_f32(h_ptr[2 * dim_pair]);
    const float b = bf16_to_f32(h_ptr[2 * dim_pair + 1]);
    h_ptr[2 * dim_pair]     = f32_to_bf16(a * cos_v - b * sin_v);
    h_ptr[2 * dim_pair + 1] = f32_to_bf16(b * cos_v + a * sin_v);
}

__device__ __forceinline__ void rotate_pair_interleaved_to(
    const bf16* src, bf16* dst,
    int dim_pair, float cos_v, float sin_v)
{
    const float a = bf16_to_f32(src[2 * dim_pair]);
    const float b = bf16_to_f32(src[2 * dim_pair + 1]);
    dst[2 * dim_pair]     = f32_to_bf16(a * cos_v - b * sin_v);
    dst[2 * dim_pair + 1] = f32_to_bf16(b * cos_v + a * sin_v);
}

__device__ __forceinline__ void rotate_pair_to(
    const bf16* src, bf16* dst,
    int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = bf16_to_f32(src[dim_pair]);
    const float b = bf16_to_f32(src[dim_pair + half]);
    dst[dim_pair]        = f32_to_bf16(a * cos_v - b * sin_v);
    dst[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
}

__device__ __forceinline__ void rope_cos_sin(
    float theta, int dim_pair, int head_dim, int pos,
    float& cos_v, float& sin_v)
{
    const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                   static_cast<float>(head_dim));
    const float ang = static_cast<float>(pos) * freq;
    __sincosf(ang, &sin_v, &cos_v);
}

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

__host__ __device__ inline void yarn_original_ramp_bounds(
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

}
