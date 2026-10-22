#pragma once

#include "linear/act_quant.cuh"
#include "prim/device.cuh"
#ifdef __CUDACC_RTC__

#include "prim/mma_sm80.cuh"
#else

#include <cuda_bf16.h>
#include <mma.h>
#endif
#include "linear/moe_w4.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

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

// Two values' e2m1 codes in a byte, `lo` in the low nibble: one conversion
// on a B200 (whose -0 keeps its sign bit, a zero all the same).
__device__ __forceinline__ u32 e2m1_pair(float lo, float hi) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
    u32 byte;
    asm("{\n\t.reg .b8 q;\n\tcvt.rn.satfinite.e2m1x2.f32 q, %1, %2;\n\tcvt.u32.u8 %0, q;\n\t}"
        : "=r"(byte)
        : "f"(hi), "f"(lo));
    return byte;
#else
    return e2m1_code(lo) | (e2m1_code(hi) << 4);
#endif
}

__device__ __forceinline__ float e4m3_value(u8 b) {
    __nv_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// A 16-wide bf16 block read to `v`; its largest magnitude.
__device__ __forceinline__ float load_block16(const bf16* __restrict__ x, float (&v)[16]) {
    uint4 raw[2];
    raw[0] = reinterpret_cast<const uint4*>(x)[0];
    raw[1] = reinterpret_cast<const uint4*>(x)[1];
    const bf16* h = reinterpret_cast<const bf16*>(raw);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = bf16_to_f32(h[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    return amax;
}

// A 16-wide block's e2m1 codes `w` under global `gs`; its scale's e4m3 code.
__device__ __forceinline__ u8 nvfp4_block16(const float (&v)[16], float amax, float gs, uint2& w) {
    const u8 sf_code = __nv_cvt_float_to_fp8(gs * (amax * rcp_approx_ftz(6.f)), __NV_SATFINITE, __NV_E4M3);
    const float sf_value = e4m3_value(sf_code);
    const float out_scale = sf_value != 0.f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(gs)) : 0.f;
    u32 c[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const u32 byte = e2m1_pair(v[2 * i] * out_scale, v[2 * i + 1] * out_scale);
        c[i / 4] |= byte << (8 * (i % 4));
    }
    w = make_uint2(c[0], c[1]);
    return sf_code;
}

using f32 = float;

namespace w4mma {

using ::pie::ld_stream;

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// a * b, bf16 x 2, subnormals kept.
__device__ __forceinline__ unsigned mul2(unsigned a, unsigned b) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}

/// The e2m1 codes at bits 12..15 and 28..31 of `q` as two bf16, each 2^-126
/// times its value (an exponent field of e, a code of exponent 0 subnormal).
__device__ __forceinline__ unsigned e2m1_pair(unsigned q) {
    return (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}

__device__ __forceinline__ unsigned ld_scale2(const u8* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned ld_scale(const u8* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

using ::pie::glu_fn::glu;

// A K-split CTA's later groups' sums (`part` past 0) folded through shared
// memory into its first group's, a lane its own: false for the threads that
// land nothing past it (the later groups, a warp past the bank's rows).
template <int kKSplit, int kTiles, int kNb>
__device__ __forceinline__ bool fold_ksplit(float (&acc)[kTiles][kNb][4], int part, int warp,
                                            int lane, bool rows_live) {
    if constexpr (kKSplit > 1) {
        constexpr int kAcc = kTiles * kNb * 4;
        __shared__ float fold[kKSplit - 1][4][kAcc][32];
        if (part > 0) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int b = 0; b < kNb; ++b)
#pragma unroll
                    for (int e = 0; e < 4; ++e) fold[part - 1][warp][(t * kNb + b) * 4 + e][lane] = acc[t][b][e];
        }
        __syncthreads();
        if (part > 0 || !rows_live) return false;
#pragma unroll
        for (int p = 0; p < kKSplit - 1; ++p)
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int b = 0; b < kNb; ++b)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[t][b][e] += fold[p][warp][(t * kNb + b) * 4 + e][lane];
    }
    return true;
}

__device__ __forceinline__ uint4 ld_act(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

/// An NVFP4 block scale (e4m3) as bf16 bits, times 2^118: an e2m1 code's
/// 2^-126 (`e2m1_pair`) times it is the product times 2^-8, exact and
/// normal across e4m3's range; the epilogue's factor restores the 2^8.
__device__ __forceinline__ unsigned nvfp4_scale_bf16(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0) {
        // m 2^-9 2^118, exact in f32.
        return __float_as_uint(static_cast<float>(m) * 6.4903710731685345e32f) >> 16;
    }
    return ((e + 238u) << 7) | (m << 4);
}

/// One 128-code chunk a lane holds: its rows' code groups and scales, and
/// its routes' activations over the group.
template <int kTiles, int kNb>
struct Chunk {
    uint4 w[kTiles][2];
    unsigned s[kTiles][2];
    uint4 x[kNb][4];
};

// 2^(s - 1) makes up a code's 2^-126 and the block's 2^(s - 127); past
// s = 128 that overflows bf16, and a warp with such a block takes it in
// two factors.
// NVFP4 (`kNv`): a lane's 32 codes are two 16-code blocks, words 0-1 and
// 2-3, the two e4m3 scales in the low and high byte of its scale word.
template <bool wide, bool kNv, int kTiles, int kNb>
__device__ __forceinline__ void compute(const Chunk<kTiles, kNb>& cc, float (&acc)[kTiles][kNb][4]) {
    unsigned s1[kTiles][2], s2[kTiles][2], nv[kTiles][2][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned s = cc.s[t][h];
            if constexpr (kNv) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const unsigned v = nvfp4_scale_bf16((s >> (8 * q)) & 0xFFu);
                    nv[t][h][q] = v | (v << 16);
                }
                s1[t][h] = s2[t][h] = 0u;
            } else {
                const unsigned a = !wide || s <= 128u ? (s + 126u) << 7 : 253u << 7;
                const unsigned b = s <= 128u ? 0x3F80u : s << 7;
                s1[t][h] = a | (a << 16);
                s2[t][h] = b | (b << 16);
            }
        }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (kNv) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) s1[t][h] = nv[t][h][w >> 1];
        }
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            unsigned b0[kNb], b1[kNb];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                const uint4 x = cc.x[b][w];
                const unsigned lo = jj == 0 ? x.y : x.x;
                const unsigned hi = jj == 0 ? x.w : x.z;
                b0[b] = __byte_perm(lo, hi, 0x7632);
                b1[b] = __byte_perm(lo, hi, 0x5410);
            }
            const int p0 = 2 * jj;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const unsigned w0 = (&cc.w[t][0].x)[w];
                const unsigned w1 = (&cc.w[t][1].x)[w];
                unsigned a0 = mul2(e2m1_pair(w0 << (4 * p0)), s1[t][0]);
                unsigned a1 = mul2(e2m1_pair(w1 << (4 * p0)), s1[t][1]);
                unsigned a2 = mul2(e2m1_pair(w0 << (4 * p0 + 4)), s1[t][0]);
                unsigned a3 = mul2(e2m1_pair(w1 << (4 * p0 + 4)), s1[t][1]);
                if constexpr (wide) {
                    a0 = mul2(a0, s2[t][0]);
                    a1 = mul2(a1, s2[t][1]);
                    a2 = mul2(a2, s2[t][0]);
                    a3 = mul2(a3, s2[t][1]);
                }
#pragma unroll
                for (int b = 0; b < kNb; ++b) mma(acc[t][b], a0, a1, a2, a3, b0[b], b1[b]);
            }
        }
    }
}

// A warp's sums landed at their routes' rows: each row's bias (`b_e`, or
// none) after its `factor`, the gate and up rows through the GLU into
// `inter` columns where `kGlu`, else `n` columns as they are; tile `t`'s
// rows `kTileRows t` past the warp's.
template <int kGlu, int kTiles, int kNb, int kTileRows = 16, class Factor>
__device__ __forceinline__ void w4_land(const float (&acc)[kTiles][kNb][4], Factor factor,
                                        const bf16* b_e, const i32* __restrict__ order, int begin,
                                        int count, bf16* __restrict__ out, int n, int inter,
                                        int row_base, int g, int c, float limit, float galpha) {
    if constexpr (kGlu != 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = row_base + 8 * h + g;
            if (col >= inter) continue;
            const float bg = b_e != nullptr ? bf16_to_f32(b_e[col]) : 0.f;
            const float bu = b_e != nullptr ? bf16_to_f32(b_e[inter + col]) : 0.f;
            const float fg = factor(col), fu = factor(inter + col);
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * inter + col] = f32_to_bf16(glu<kGlu>(
                            acc[0][b][2 * h + e] * fg + bg, acc[1][b][2 * h + e] * fu + bu, limit,
                            galpha));
                    }
                }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row_base + kTileRows * t + 8 * h + g;
            if (row >= n) continue;
            const float add = b_e != nullptr ? bf16_to_f32(b_e[row]) : 0.f;
            const float f = factor(row);
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * n + row] = f32_to_bf16(acc[t][b][2 * h + e] * f + add);
                    }
                }
        }
}

}

__device__ __forceinline__ float nvfp4_scale(u8 b) {
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const float v = e != 0u ? __int_as_float(static_cast<int>(((e + 120u) << 23) | (m << 20)))
                            : static_cast<float>(m) * 0.001953125f;
    return (b & 0x80u) != 0u ? -v : v;
}

__device__ __forceinline__ float mxfp4_block_scale(u8 b) {
    return b == 0 ? exp2f(-127.f)
                  : __int_as_float(static_cast<int>(b) << 23);
}

constexpr int kOffPost = 0;

constexpr int kOffPreInt = 1;

constexpr int kOffPreReal = 2;

}
