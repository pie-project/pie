#pragma once

// Activations quantized for a low-precision product, in the two formats a
// deployment states:
//
//   NVFP4  e2m1 under an e4m3 scale a 16, the scales under a static global
//          `gs = 1 / input_scale` (vLLM's and TensorRT-LLM's recipe, their
//          approximate reciprocals included): `sf = e4m3(gs * amax / 6)`,
//          `q = e2m1(x / (sf / gs))`, the value `q * sf / gs`.
//   E4M3   e4m3 under a ue8m0 scale a `kGran` (DeepSeek's `act_quant`): the
//          amax floored at 1e-4, `s = 2^ceil(log2(amax / 448))`,
//          `q = e4m3(clamp(x / s, ±448))`.
//
// Each lands either as the tcgen05 GEMM reads it (codes, and scales in its
// 512-byte atoms of 128 rows x 4 scale columns) or read back to bf16 without
// the global, `q * sf`, every such product exact in bf16, for a bf16 product
// that multiplies by the global once.
//
// Out row `r` reads `x` row `r / div`; with `routes`, its global is its
// route's expert's, and a dropped route (`routes[r] < 0`) lands zeros.

#include "linear/quant.cuh"

namespace pie::linear {

constexpr int kQuantCodes = 0;
constexpr int kQuantValues = 1;

// The byte of scale `(row, col)` in an atom-layout tensor of `cols` columns.
__device__ __forceinline__ long long atom_offset(int row, int col, int cols) {
    const int atoms_k = (cols + 3) / 4;
    return (static_cast<long long>(row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16
        + ((row / 32) % 4) * 4 + col % 4;
}

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

// Round to the nearest e2m1 code, ties to the even code, saturating at 6;
// the sign in bit 3.
__device__ __forceinline__ unsigned e2m1_code(float v) {
    const float a = fabsf(v);
    unsigned c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.f) c = 6;
    else c = 7;
    return (v < 0.f ? 8u : 0u) | c;
}

__device__ __forceinline__ float e4m3_value(u8 b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

__device__ __forceinline__ float global_of(const float* __restrict__ input_scales,
                                           const i32* __restrict__ routes, int r, bool* dropped) {
    int e = 0;
    if (routes != nullptr) {
        e = routes[r];
        if (e < 0) {
            *dropped = true;
            return 0.f;
        }
    }
    return __fdiv_rn(1.f, input_scales[e]);
}

// One thread a 16-wide block.
template <int kMode>
__global__ void act_quant_nvfp4(
    const bf16* __restrict__ x,
    const i32* __restrict__ routes,
    const float* __restrict__ input_scales,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    const int blocks = k / 16;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(rows) * blocks) return;
    const int r = static_cast<int>(at / blocks);
    const int b = static_cast<int>(at % blocks);
    bool dropped = false;
    const float gs = global_of(input_scales, routes, r, &dropped);
    const bf16* src = x + static_cast<long long>(r / div) * k + b * 16;
    float v[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = dropped ? 0.f : bf16_to_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float sf = gs * (amax * rcp_approx_ftz(6.f));
    const u8 sf_code = __nv_cvt_float_to_fp8(sf, __NV_SATFINITE, __NV_E4M3);
    const float sf_value = e4m3_value(sf_code);
    const float out_scale = sf_value != 0.f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(gs)) : 0.f;
    if constexpr (kMode == kQuantCodes) {
        u8* dst = codes + static_cast<long long>(r) * (k / 2) + b * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[i] = static_cast<u8>(e2m1_code(v[2 * i] * out_scale)
                                     | (e2m1_code(v[2 * i + 1] * out_scale) << 4));
        }
        atoms[atom_offset(r, b, blocks)] = sf_code;
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            dst[i] = f32_to_bf16(kFp4Lut[e2m1_code(v[i] * out_scale)] * sf_value);
        }
    }
}

// One warp a `kGran`-wide block, each lane `kGran / 32` elements.
template <int kMode, int kGran>
__global__ void act_quant_e4m3(
    const bf16* __restrict__ x,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    constexpr int kEach = kGran / 32;
    const int blocks = k / kGran;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= static_cast<long long>(rows) * blocks) return;
    const int r = static_cast<int>(warp / blocks);
    const int b = static_cast<int>(warp % blocks);
    const bf16* src = x + static_cast<long long>(r / div) * k + b * kGran + lane * kEach;
    float v[kEach];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < kEach; ++i) {
        v[i] = bf16_to_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    amax = fmaxf(amax, 1e-4f);
    // 2^ceil(log2(amax / 448)), from the bits of the quotient.
    const unsigned bits = __float_as_uint(amax * (1.f / 448.f));
    const int e = static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
    const float s = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
    if constexpr (kMode == kQuantCodes) {
        u8* dst = codes + static_cast<long long>(r) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) {
            dst[i] = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                           __NV_SATFINITE, __NV_E4M3);
        }
        if (lane == 0) atoms[atom_offset(r, b, blocks)] = static_cast<u8>(e + 127);
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) {
            const u8 q = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                               __NV_SATFINITE, __NV_E4M3);
            dst[i] = f32_to_bf16(e4m3_value(q) * s);
        }
    }
}

// A row-major plane of one-byte scales, `cols` a row, into the atom layout;
// atom row `r` reads plane row `r / div` (a 128-row block's one scale
// spread to each of its rows).
__global__ void scales_to_atoms(
    const u8* __restrict__ plane,
    u8* __restrict__ atoms,
    int rows,
    int cols,
    int div)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(rows) * cols) return;
    const int r = static_cast<int>(at / cols);
    const int c = static_cast<int>(at % cols);
    atoms[atom_offset(r, c, cols)] = plane[static_cast<long long>(r / div) * cols + c];
}

// Each expert's row tensor scales times its activations' static global:
// the one factor a bf16 product of read-back values multiplies its sum by.
__global__ void fold_input_scales(
    const float* __restrict__ tensor_scales,
    const float* __restrict__ input_scales,
    float* __restrict__ folded,
    int experts,
    int rows)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(experts) * rows) return;
    folded[at] = input_scales[at / rows] * tensor_scales[at];
}

}  // namespace pie::linear
