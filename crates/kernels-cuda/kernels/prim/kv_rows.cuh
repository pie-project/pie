#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "prim/device.cuh"
#include "prim/rope.cuh"

namespace pie::kv_rows {

constexpr int kDsDim = 512;

constexpr int kDsNope = 448;

constexpr int kDsTile = 64;

constexpr int kDsScales = kDsNope / kDsTile;

// 7
constexpr int kDsData = kDsNope + (kDsDim - kDsNope) * 2;

// 576
constexpr int kDsScaleBytes = 8;

constexpr int kDsRow = kDsData + kDsScaleBytes;

// 584
constexpr float kFp8Max = 448.0f;

// A position's row: its 576 data bytes and its 8 scale bytes.
struct DsRow {
    std::uint8_t* data;
    std::uint8_t* scales;
};

__device__ __forceinline__ DsRow ds_row(std::uint8_t* plane, long long page, int off, int ps) {
    std::uint8_t* base = plane + page * static_cast<long long>(ps) * kDsRow;
    return {base + static_cast<long long>(off) * kDsData,
            base + static_cast<long long>(ps) * kDsData + static_cast<long long>(off) * kDsScaleBytes};
}

// The inverse rotation of an attention output's last `dim` dims (DeepSeek-V4
// turns its output back by the row's position), landed with the output:
// `dim` zero turns nothing. The angles are `rope_partial_last`'s, negated.
struct Turn {
    int dim;
    float theta;
    float factor;
    float low;
    float high;
};

// The turn's frequency of pair `pair`, as `rope_partial_last` works it.
__device__ __forceinline__ float turn_freq(int pair, const Turn& t) {
    float freq = powf(t.theta, -2.f * static_cast<float>(pair) / static_cast<float>(t.dim));
    if (t.factor > 1.f) freq = ::pie::yarn_original_freq(freq, t.factor, t.low, t.high, pair);
    return freq;
}

// Interleaved pair (a, b) at head dim `d` (even) of a `head_dim`-wide head,
// turned back by position `pos` where it is the tail's; `freq` the turn's
// frequencies (`turn_freq`), a pair each.
__device__ __forceinline__ void turn_pair(float& a, float& b, int pos, float freq) {
    float s, c;
    __sincosf(-static_cast<float>(pos) * freq, &s, &c);
    const float a2 = a * c - b * s;
    const float b2 = b * c + a * s;
    a = a2;
    b = b2;
}

__device__ __forceinline__ void turn_back(float& a, float& b, int d, int head_dim, int pos, const Turn& t,
                                          const float* freq) {
    const int at = d - (head_dim - t.dim);
    if (t.dim <= 0 || at < 0) return;
    turn_pair(a, b, pos, freq[at / 2]);
}

__device__ __forceinline__ float ue8m0(std::uint8_t e) {
    return __uint_as_float(static_cast<unsigned>(e) << 23);
}

// One row of 512 into `row`, by one warp: lane l holds dims 16l..16l+15,
// four lanes a 64-wide scale tile; lanes 28..31 hold the rotated dims, kept
// in bf16. As vLLM's insert: the exponent ceil(log2(amax / 448)), amax at
// least 1e-4, the codes saturated.
__device__ __forceinline__ void ds_store_warp(const bf16* __restrict__ src, DsRow row, int lane) {
    const int d0 = lane * 16;
    float x[16];
    {
        const uint4 a = *reinterpret_cast<const uint4*>(src + d0);
        const uint4 b = *reinterpret_cast<const uint4*>(src + d0 + 8);
        const bf16* pa = reinterpret_cast<const bf16*>(&a);
        const bf16* pb = reinterpret_cast<const bf16*>(&b);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = Elem<bf16>::to_f32(pa[i]);
            x[8 + i] = Elem<bf16>::to_f32(pb[i]);
        }
    }
    if (d0 >= kDsNope) {
        // The rotated dims, as they are.
        std::uint8_t* dst = row.data + kDsNope + (d0 - kDsNope) * 2;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src + d0);
        *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(src + d0 + 8);
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (d0 >= kDsNope) return;
    amax = fmaxf(amax, 1e-4f);
    const float e = ceilf(log2f(amax / kFp8Max));
    const float inv = exp2f(-e);
    std::uint8_t codes[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float v = fminf(fmaxf(x[i] * inv, -kFp8Max), kFp8Max);
        codes[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint4*>(row.data + d0) = *reinterpret_cast<const uint4*>(codes);
    if ((lane & 3) == 0) {
        row.scales[lane >> 2] = static_cast<std::uint8_t>(fmaxf(fminf(e + 127.f, 255.f), 0.f));
        if (lane == 0) row.scales[kDsScales] = 0;
    }
}

// 8 dims of a row from its data and scale bytes: `j` the first dim (a
// multiple of 8).
__device__ __forceinline__ uint4 ds_load8(const std::uint8_t* data, const std::uint8_t* scales, int j) {
    if (j >= kDsNope) return *reinterpret_cast<const uint4*>(data + kDsNope + (j - kDsNope) * 2);
    const float s = ue8m0(scales[j / kDsTile]);
    const uint2 codes = *reinterpret_cast<const uint2*>(data + j);
    const std::uint16_t* c2 = reinterpret_cast<const std::uint16_t*>(&codes);
    uint4 out;
    std::uint32_t* o32 = reinterpret_cast<std::uint32_t*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo, hi;
        ::pie::e4m3_pair_f32(c2[i], lo, hi);
        const __nv_bfloat162 b = __floats2bfloat162_rn(lo * s, hi * s);
        o32[i] = *reinterpret_cast<const std::uint32_t*>(&b);
    }
    return out;
}

// DeepSeek-V4.1's rows, as its reference quantizes its caches and as vLLM's
// sm100 kernels read them. A page of `ps` positions is `ps` data rows, then
// `ps` scale rows, as V4's.
//   kMx8: the window's plane, all 512 dims e4m3 under one ue8m0 exponent a
//         32 (the reference's `act_quant(kv, 32, ue8m0)`): 512 + 16 bytes.
//   kNv4: the compressed rows, 512 dims e2m1 (two a byte, the lower dim in
//         the low nibble) under one e4m3 scale a 16 (`fp4_act_quant(kv, 16,
//         e4m3)`): 256 + 32 bytes.
// kV4 names V4's 584-byte rows above.
constexpr int kV4 = 1;

constexpr int kMx8 = 2;

constexpr int kNv4 = 3;

template <int F>
struct Rows;

// bf16 planes: no quantized row.
template <>
struct Rows<0> {
    static constexpr int kData = 0;
    static constexpr int kScale = 0;
};

template <>
struct Rows<kV4> {
    static constexpr int kData = kDsData;
    static constexpr int kScale = kDsScaleBytes;
};

template <>
struct Rows<kMx8> {
    static constexpr int kData = kDsDim;
    static constexpr int kScale = kDsDim / 32;
};

template <>
struct Rows<kNv4> {
    static constexpr int kData = kDsDim / 2;
    static constexpr int kScale = kDsDim / 16;
};

template <int F>
__device__ __forceinline__ DsRow rows_at(std::uint8_t* plane, long long page, int off, int ps) {
    using T = Rows<F>;
    std::uint8_t* base = plane + page * static_cast<long long>(ps) * (T::kData + T::kScale);
    return {base + static_cast<long long>(off) * T::kData,
            base + static_cast<long long>(ps) * T::kData + static_cast<long long>(off) * T::kScale};
}

// The nearest e2m1 code of `v` (within ±6), ties to the even code and the
// sign kept on a zero, as the hardware's `cvt.rn` rounds.
__device__ __forceinline__ unsigned e2m1_code(float v) {
    const float a = fabsf(v);
    const unsigned m = static_cast<unsigned>(a > 0.25f) + static_cast<unsigned>(a >= 0.75f) +
                       static_cast<unsigned>(a > 1.25f) + static_cast<unsigned>(a >= 1.75f) +
                       static_cast<unsigned>(a > 2.5f) + static_cast<unsigned>(a >= 3.5f) +
                       static_cast<unsigned>(a > 5.f);
    return signbit(v) ? (m | 8u) : m;
}

// An e2m1 code's bf16 bits: 0.5 at magnitude 1, 2^(e - 1) (1 + m / 2) from
// 2 on, built from the code's bits with no conversion.
__device__ __forceinline__ unsigned e2m1_bf16(unsigned c) {
    const unsigned m = c & 7u;
    const unsigned mag = m < 2u ? m * 0x3F00u : (((m >> 1) + 126u) << 7) | ((m & 1u) << 6);
    return mag | ((c & 8u) << 12);
}

__device__ __forceinline__ float e2m1_value(unsigned c) {
    return __uint_as_float(e2m1_bf16(c) << 16);
}

__device__ __forceinline__ float e4m3_float(std::uint8_t b) {
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const float v = e != 0u ? __uint_as_float(((e + 120u) << 23) | (m << 20))
                            : static_cast<float>(m) * 0.001953125f;
    return (b & 0x80u) != 0u ? -v : v;
}

// 8 MXFP8 codes under a ue8m0 scale `s`, widened to bf16.
__device__ __forceinline__ uint4 mx8_widen8(uint2 codes, float s) {
    const std::uint16_t c2[4] = {static_cast<std::uint16_t>(codes.x), static_cast<std::uint16_t>(codes.x >> 16),
                                 static_cast<std::uint16_t>(codes.y), static_cast<std::uint16_t>(codes.y >> 16)};
    uint4 out;
    std::uint32_t* o32 = reinterpret_cast<std::uint32_t*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo, hi;
        ::pie::e4m3_pair_f32(c2[i], lo, hi);
        const __nv_bfloat162 b = __floats2bfloat162_rn(lo * s, hi * s);
        o32[i] = *reinterpret_cast<const std::uint32_t*>(&b);
    }
    return out;
}

// 8 e2m1 codes (two a byte, the lower dim in the low nibble) under an e4m3
// scale `s2` (as bf16, twice). An e2m1 value times an e4m3 scale has at most
// five significant bits, so the bf16 product is exact: the codes widen as
// bf16 bits and one bf16x2 multiply a pair lands what the f32 product rounds
// to.
__device__ __forceinline__ uint4 nv4_widen8(std::uint32_t nib, __nv_bfloat162 s2) {
    uint4 out;
    std::uint32_t* o32 = reinterpret_cast<std::uint32_t*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = (nib >> (8 * i)) & 0xffu;
        const std::uint32_t pair = e2m1_bf16(byte & 0xfu) | (e2m1_bf16(byte >> 4) << 16);
        const __nv_bfloat162 b = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&pair), s2);
        o32[i] = *reinterpret_cast<const std::uint32_t*>(&b);
    }
    return out;
}

// 8 dims of a row from its data and scale bytes: `j` the first dim (a
// multiple of 8).
template <int F>
__device__ __forceinline__ uint4 rows_load8(const std::uint8_t* data, const std::uint8_t* scales, int j) {
    if constexpr (F == kV4) {
        return ds_load8(data, scales, j);
    } else if constexpr (F == kMx8) {
        return mx8_widen8(*reinterpret_cast<const uint2*>(data + j), ue8m0(scales[j / 32]));
    } else {
        return nv4_widen8(*reinterpret_cast<const std::uint32_t*>(data + j / 2),
                          __float2bfloat162_rn(e4m3_float(scales[j / 16])));
    }
}

}
