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
// route's expert's, and a dropped route (`routes[r] < 0`) lands zeros. With
// `pos`, its codes land at row `pos[r]` of a grouped layout (a dropped route
// nowhere), its scales at that row's atom row: the same row for the
// contiguous layout, expert `pos / m_max`'s whole row blocks for the masked
// one (`m_max > 0`).

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

// The atom row of grouped row `o`.
__device__ __forceinline__ int atom_row(int o, int m_max) {
    if (m_max == 0) return o;
    const int pad = (m_max + 127) / 128 * 128;
    return o / m_max * pad + o % m_max;
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
    const i32* __restrict__ pos,
    int m_max,
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
        const int o = pos != nullptr ? pos[r] : r;
        if (o < 0) return;
        u8* dst = codes + static_cast<long long>(o) * (k / 2) + b * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[i] = static_cast<u8>(e2m1_code(v[2 * i] * out_scale)
                                     | (e2m1_code(v[2 * i + 1] * out_scale) << 4));
        }
        atoms[atom_offset(atom_row(o, m_max), b, blocks)] = sf_code;
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
    const i32* __restrict__ pos,
    int m_max,
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
        const int o = pos != nullptr ? pos[r] : r;
        if (o < 0) return;
        u8* dst = codes + static_cast<long long>(o) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) {
            dst[i] = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                           __NV_SATFINITE, __NV_E4M3);
        }
        if (lane == 0) atoms[atom_offset(atom_row(o, m_max), b, blocks)] = static_cast<u8>(e + 127);
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

// Routes laid out by expert for the grouped GEMM: each routed expert's
// count, its rows' start (`align` rows apart, rounded up, for the contiguous
// layout, `m_max` apart for the masked one) and each route's row.
__global__ void route_count(const i32* __restrict__ routes, int count, i32* __restrict__ counts)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < count && routes[r] >= 0) atomicAdd(&counts[routes[r]], 1);
}

__global__ void route_offsets(
    const i32* __restrict__ counts,
    int experts,
    int m_max,
    int align,
    i32* __restrict__ offsets,
    i32* __restrict__ cursor)
{
    if (threadIdx.x != 0) return;
    int at = 0;
    for (int e = 0; e < experts; ++e) {
        offsets[e] = at;
        cursor[e] = at;
        at += m_max > 0 ? m_max : (counts[e] + align - 1) / align * align;
    }
    offsets[experts] = at;
}

__global__ void route_place(
    const i32* __restrict__ routes,
    int count,
    i32* __restrict__ cursor,
    i32* __restrict__ pos)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    pos[r] = routes[r] >= 0 ? atomicAdd(&cursor[routes[r]], 1) : -1;
}

// `y[r] = grouped[pos[r]]`, a dropped route zeros; a thread a pair.
__global__ void route_gather(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    bf16* __restrict__ y,
    int count,
    int n)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(count) * (n / 2)) return;
    const int r = static_cast<int>(at / (n / 2));
    const int c = static_cast<int>(at % (n / 2)) * 2;
    const int o = pos[r];
    bf16x2* dst = reinterpret_cast<bf16x2*>(y + static_cast<long long>(r) * n + c);
    if (o < 0) {
        *dst = f32_to_bf16x2(0.f, 0.f);
        return;
    }
    *dst = *reinterpret_cast<const bf16x2*>(grouped + static_cast<long long>(o) * n + c);
}

// `grouped[pos[r]] = x[r / div]`, 16 bytes a thread; a dropped route lands
// nowhere.
__global__ void route_scatter_rows(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    bf16* __restrict__ grouped,
    int count,
    int k,
    int div)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int runs = k / 8;
    if (at >= static_cast<long long>(count) * runs) return;
    const int r = static_cast<int>(at / runs);
    const int c = static_cast<int>(at % runs) * 8;
    const int o = pos[r];
    if (o < 0) return;
    *reinterpret_cast<uint4*>(grouped + static_cast<long long>(o) * k + c) =
        *reinterpret_cast<const uint4*>(x + static_cast<long long>(r / div) * k + c);
}

// Each expert's product scale: its activations' static global times its
// weight's tensor scale, the first row's standing for the expert's (vLLM
// reads the gate leg's for both of a gated bank's legs).
__global__ void expert_alpha(
    const float* __restrict__ input_scales,
    const float* __restrict__ tensor_scales,
    float* __restrict__ alpha,
    int experts,
    int rows)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts) return;
    alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * rows];
}

}  // namespace pie::linear
