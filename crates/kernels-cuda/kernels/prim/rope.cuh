#pragma once

#include "prim/device.cuh"

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

}
