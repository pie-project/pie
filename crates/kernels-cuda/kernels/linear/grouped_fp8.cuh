#pragma once

#include "linear/act_quant.cuh"

namespace pie::linear {

namespace gfp8 {

using ::pie::ld_stream;

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// Two e4m3 codes (the low byte the low half) as floats, exactly.
__device__ __forceinline__ float2 e4m3x2_f32(unsigned short pair) {
    float2 r;
    ::pie::e4m3_pair_f32(pair, r.x, r.y);
    return r;
}

/// `v` as an e4m3 code, rounded to nearest even and saturated at 448 (as
/// sm_89's `cvt.rn.satfinite` lands it; NaN lands the NaN code).
__device__ __forceinline__ unsigned e4m3_of(float v) {
    const unsigned sign = (__float_as_uint(v) >> 24) & 0x80u;
    const float a = fabsf(v);
    if (!(a == a)) return 0x7Fu;
    if (a >= 448.f) return sign | 0x7Eu;
    // Below 2^-6 the codes step 2^-9 apart: a subnormal code is the count
    // (eight of them reaching the smallest normal, code 0x08).
    if (a < 0x1p-6f) return sign | static_cast<unsigned>(__float2int_rn(a * 512.f));
    // Else keep three mantissa bits, nearest even, a carry moving the exponent.
    unsigned bits = __float_as_uint(a);
    bits += 0x7FFFFu + ((bits >> 20) & 1u);
    const unsigned e = (bits >> 23) - 120u;  // bias 127 to bias 7
    const unsigned code = (e << 3) | ((bits >> 20) & 7u);
    return sign | (code > 0x7Eu ? 0x7Eu : code);
}

/// Two floats as e4m3 codes, `lo`'s the low byte.
__device__ __forceinline__ unsigned short e4m3x2_of(float hi, float lo) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    return static_cast<unsigned short>(e4m3_of(lo) | (e4m3_of(hi) << 8));
#else
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

/// 2^(e - 127) for a biased exponent `e` in 1 .. 254.
__device__ __forceinline__ float exp2_biased(int e) {
    return __uint_as_float(static_cast<unsigned>(e) << 23);
}

}

}
