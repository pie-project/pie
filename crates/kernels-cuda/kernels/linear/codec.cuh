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
