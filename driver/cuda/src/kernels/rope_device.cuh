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

// ── vLLM cos/sin-table primitives (opt-in, PIE_DEBUG_ROPE_VLLM_TABLE=1) ──────────
//
// vLLM never evaluates trig on a GPU. `RotaryEmbedding.__init__` builds
// `cos_sin_cache` once on the host in fp32 -- inv_freq from a weak-scalar
// base, angle = fp32(pos) * inv_freq, eager `cos()`/`sin()` -- and then does
// `cache.to(dtype)`, which ROUNDS THE WHOLE TABLE TO BF16 AND STORES IT. The
// path our reference leg actually runs (`triton_mrope`, i.e. no
// `--language-model-only`) loads that bf16 row with no fp32 cast and casts q/k
// DOWN via `.to(sin_row.dtype)`, so the rotate is bf16 end to end.
//
// So three things have to change together, and "call accurate trig" is only
// one of them:
//   1. the exponent form -- `1 / theta^(2j/rotary_dim)`, not `theta^(-2j/d)`
//   2. cos/sin rounded to bf16 BEFORE they are used at all
//   3. the rotate done in bf16, not fp32
// Swapping `__sincosf` for `sincosf` alone would leave Pie strictly MORE
// accurate than the reference and still mismatch it.
//
// The table itself is a memoization, not a numerical device: entry (pos, j) is
// a pure function of (pos, j) with no accumulated state, so evaluating it per
// token here is bit-identical to indexing vLLM's preallocated array. What is
// reproduced is the arithmetic, not the allocation.

// inv_freq[j], vLLM form: `1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))`.
//
// Two details that are not interchangeable with `powf(theta, -2j/rotary_dim)`:
// the reciprocal is taken AFTER the power (a separate fp32 rounding), and the
// exponent is the positive `2j/rotary_dim`. The power itself is evaluated in
// double and rounded once because device `powf` carries up to 8 ulp, and at
// j=1 the angle reaches ~1.2e4 rad at position 20000 where 8 ulp is ~6e-3 rad
// -- a full bf16 step in cos. vLLM's host-side power is correctly rounded, so
// the double evaluation is the FAITHFUL one here, not the over-accurate one.
__device__ __forceinline__ float rope_inv_freq_vllm(
    float theta, int dim_pair, int rotary_dim)
{
    // Exponent formed in fp32, as vLLM's fp32 arange/divide does.
    const float exponent = (2.f * static_cast<float>(dim_pair)) /
                           static_cast<float>(rotary_dim);
    const float p = static_cast<float>(
        pow(static_cast<double>(theta), static_cast<double>(exponent)));
    return 1.f / p;
}

// One table entry computed ON THE DEVICE: fp32 trig, then rounded to bf16
// (RNE, which is what `.to(torch.bfloat16)` does).
//
// THIS IS THE FALLBACK, NOT THE PATH. The table is built on the host (see
// `vllm_table_for` in rope.cu) because device trig does not reproduce the
// reference's bits: measured over 30001 positions x 64 lanes, this function
// differs from the reference on 4 entries, each exactly one bf16 ulp, each
// within 0.71 fp32 ulp of a rounding midpoint. This is reached only past the
// end of the host table, where parity is explicitly not claimed and the
// occurrence is counted.
//
// `sincosf` rather than `__sincosf`, but NOT because the intrinsic falls apart.
// Measured on sm89 at lane 0 -- where inv_freq[0] is exactly 1.0, so the angle
// in radians IS the token position -- `__sincosf`'s absolute error is 1.78e-05
// at positions 0-200, 5.39e-04 at 1000-4000, 2.31e-03 at 13000-20000 and
// 4.24e-03 past 20000, against a bf16 step of 7.8e-03. It degrades
// monotonically; it does not collapse.
//
// The reason the swap is needed is not accuracy. vLLM's values are a specific
// bf16-rounded table, and only reproducing that table reproduces its bits: an
// error of half a bf16 step is exactly the size that flips table entries, and
// a flipped entry is a parity miss no matter how small it is.
__device__ __forceinline__ void rope_cos_sin_vllm_table(
    float theta, int dim_pair, int rotary_dim, int pos,
    __nv_bfloat16& cos_b, __nv_bfloat16& sin_b)
{
    const float ang = static_cast<float>(pos) *
                      rope_inv_freq_vllm(theta, dim_pair, rotary_dim);
    float sin_v, cos_v;
    sincosf(ang, &sin_v, &cos_v);
    cos_b = __float2bfloat16(cos_v);
    sin_b = __float2bfloat16(sin_v);
}

// bf16 arithmetic, one rounding per operation.
//
// The exact product of two bf16 values needs 16 significand bits and the exact
// sum needs at most 24 for any exponent gap a rotation produces, so fp32 holds
// both EXACTLY and the single `__float2bfloat16` is the only rounding. That
// makes these bit-identical to native `mul.rn.bf16` / `add.rn.bf16` without an
// `__CUDA_ARCH__ >= 800` guard. This is emphatically NOT `rotate_pair`'s fp32
// rotate, which rounds once at the end of the whole expression.
__device__ __forceinline__ __nv_bfloat16 bf16_mul(
    __nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

__device__ __forceinline__ __nv_bfloat16 bf16_add(
    __nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

__device__ __forceinline__ __nv_bfloat16 bf16_sub(
    __nv_bfloat16 a, __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
}

// `x1*cos - x2*sin` / `x2*cos + x1*sin` with every intermediate rounded to
// bf16 -- what an all-bf16 tensor expression evaluates to. Dims `i` and `j` are
// passed explicitly so the caller owns the pairing rule.
__device__ __forceinline__ void rotate_pair_bf16(
    __nv_bfloat16* base, int i, int j,
    __nv_bfloat16 cos_b, __nv_bfloat16 sin_b)
{
    const __nv_bfloat16 a = base[i];
    const __nv_bfloat16 b = base[j];
    base[i] = bf16_sub(bf16_mul(a, cos_b), bf16_mul(b, sin_b));
    base[j] = bf16_add(bf16_mul(b, cos_b), bf16_mul(a, sin_b));
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
