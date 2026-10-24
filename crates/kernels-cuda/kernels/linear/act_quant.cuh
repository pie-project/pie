#pragma once

#include "linear/quant.cuh"

namespace pie::linear {

// The byte of scale `(row, col)` in an atom-layout tensor of `cols` columns.
__device__ __forceinline__ long long atom_offset(int row, int col, int cols) {
    const int atoms_k = (cols + 3) / 4;
    return (static_cast<long long>(row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16
        + ((row / 32) % 4) * 4 + col % 4;
}

// DeepSeek's ue8m0 scale exponent for a block whose largest magnitude is
// `amax`: 2^ceil(log2(max(amax, 1e-4) / 448)), read off the quotient's bits.
__device__ __forceinline__ int e4m3_block_exp(float amax) {
    amax = fmaxf(amax, 1e-4f);
    const unsigned bits = __float_as_uint(amax * (1.f / 448.f));
    return static_cast<int>((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
}

// The atom row of grouped row `o`.
__device__ __forceinline__ int atom_row(int o, int m_max) {
    if (m_max == 0) return o;
    const int pad = (m_max + 127) / 128 * 128;
    return o / m_max * pad + o % m_max;
}

// The e4m3 quantizers eight elements a lane: a `kGran`-wide block is
// kGran / 8 lanes (a warp 32 / that many blocks), read in one 16-byte load
// and landed in one 8-byte store a lane. The scale a power of two, the
// quotient `v / s` is `v * 2^-e` exactly, so the codes and values are the
// one-element-a-lane kernels' byte for byte.
template <int kGran>
struct E4m3Block {
    static constexpr int kLanes = kGran / 8;
    float v[8];
    int e;
    // Lane `lane`'s eight of the block at `src` (its kLanes lanes adjacent).
    __device__ __forceinline__ void load(const bf16* __restrict__ src, int lane) {
        const uint4 w = *reinterpret_cast<const uint4*>(src + (lane % kLanes) * 8);
        const unsigned words[4] = {w.x, w.y, w.z, w.w};
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = __uint_as_float(words[i] << 16);
            v[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000u);
            amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
        }
#pragma unroll
        for (int o = kLanes / 2; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        e = e4m3_block_exp(amax);
    }
    // The eight codes, packed low element first.
    __device__ __forceinline__ unsigned long long codes() const {
        const float inv = __uint_as_float(static_cast<unsigned>(127 - e) << 23);
        // Two codes a convert, the low element in the low byte.
        unsigned half[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = fminf(fmaxf(v[2 * i] * inv, -448.f), 448.f);
            const float hi = fminf(fmaxf(v[2 * i + 1] * inv, -448.f), 448.f);
            // The shim's convert: the instruction from sm_89, its portable
            // rounding below (sm_80 serves its W8A16 load through here).
            half[i] = static_cast<unsigned short>(
                __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
        }
        return static_cast<unsigned long long>(half[0] | (half[1] << 16)) |
               (static_cast<unsigned long long>(half[2] | (half[3] << 16)) << 32);
    }
    // A producer's eight of the block, held in registers: each rounded to
    // bf16 first, as a quantizer reading the producer's bf16 row back
    // would see it, so the codes and scale are `load`'s from that row.
    __device__ __forceinline__ void hold(const float (&x)[8]) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = bf16_to_f32(f32_to_bf16(x[i]));
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int o = kLanes / 2; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        }
        e = e4m3_block_exp(amax);
    }
    // Lands the block as row `o`'s block `b` of a GEMM's operand: lane
    // `lane`'s eight codes, and the block's ue8m0 scale in the atom layout.
    __device__ __forceinline__ void store(u8* __restrict__ to, u8* __restrict__ atoms, int o,
                                          int b, int k, int m_max, int lane) const {
        const int sub = lane % kLanes;
        *reinterpret_cast<unsigned long long*>(to + static_cast<long long>(o) * k + b * kGran +
                                               sub * 8) = codes();
        if (sub == 0) {
            atoms[atom_offset(atom_row(o, m_max), b, k / kGran)] = static_cast<u8>(e + 127);
        }
    }
};

}
