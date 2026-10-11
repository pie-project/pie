
#pragma once

#include "prelude/device.cuh"

#include "cuda_fp16.h"

// Every target is sm_80 or newer, which has the bf16 cvt, fma and max;
// sm_90 adds a bf16x2 multiply.
#if defined(PIE_HALFTYPE_FORCE_PORTABLE)
#define PIE_BF16_HAS_SM90 0
#else
#define PIE_BF16_HAS_SM90 (__CUDA_ARCH__ >= 900)
#endif


using __nv_bfloat16 = ::pie::bf16;

using nv_bfloat16 = __nv_bfloat16;

using __nv_bfloat162 = ::pie::bf16x2;

using nv_bfloat162 = __nv_bfloat162;

struct __align__(2) __nv_bfloat16_raw {
    unsigned short x;

    __nv_bfloat16_raw() = default;

    __device__ constexpr __nv_bfloat16_raw(unsigned short bits) : x(bits) {}
    __device__ __forceinline__ __nv_bfloat16_raw(const __nv_bfloat16 v) : x(v.raw) {}

    __device__ constexpr operator __nv_bfloat16() const { return __nv_bfloat16{x}; }
};

struct __align__(4) __nv_bfloat162_raw {
    unsigned short x;
    unsigned short y;

    __nv_bfloat162_raw() = default;
    __device__ __forceinline__ __nv_bfloat162_raw(const __nv_bfloat162 v)
        : x(v.x.raw), y(v.y.raw) {}
    __device__ __forceinline__ operator __nv_bfloat162() const {
        __nv_bfloat162 out;
        out.x.raw = x;
        out.y.raw = y;
        return out;
    }
};

#define __CUDA_BF16_TYPES_EXIST__

namespace pie_bf16_detail {

__device__ __forceinline__ unsigned int pack(__nv_bfloat162 v) {
    return (static_cast<unsigned int>(v.y.raw) << 16) | static_cast<unsigned int>(v.x.raw);
}

__device__ __forceinline__ __nv_bfloat162 unpack(unsigned int bits) {
    __nv_bfloat162 out;
    out.x.raw = static_cast<unsigned short>(bits & 0xffffu);
    out.y.raw = static_cast<unsigned short>(bits >> 16);
    return out;
}

}


__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 v) {
    return ::pie::bf16_to_f32(v);
}

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float f) {
    __nv_bfloat16 out;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out.raw) : "f"(f));
    return out;
}

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

__device__ __forceinline__ __nv_bfloat162 __float2bfloat162_rn(float f) {
    const __nv_bfloat16 t = __float2bfloat16(f);
    __nv_bfloat162 out;
    out.x = t;
    out.y = t;
    return out;
}

__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
    unsigned int bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return pie_bf16_detail::unpack(bits);
}

__device__ __forceinline__ __nv_bfloat162 __float22bfloat162_rn(float2 f) {
    return __floats2bfloat162_rn(f.x, f.y);
}

__device__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat162 v) {
    return ::pie::bf16x2_to_f32(v);
}

__device__ __forceinline__ __nv_bfloat162 make_bfloat162(__nv_bfloat16 x, __nv_bfloat16 y) {
    __nv_bfloat162 out;
    out.x = x;
    out.y = y;
    return out;
}


__device__ __forceinline__ __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
#if PIE_BF16_HAS_SM90
    unsigned int bits;
    asm("mul.rn.bf16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#else
    unsigned int bits;

    asm("{ .reg .b32 z;\n"
        "  mov.b32 z, 0x80008000;\n"
        "  fma.rn.bf16x2 %0, %1, %2, z; }"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#endif
}


namespace pie_bf16_detail {

__device__ __forceinline__ __nv_bfloat16 fma_once(
    __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
    const float p = __bfloat162float(a) * __bfloat162float(b);
    const float x = __bfloat162float(c);
    const float s = p + x;
    const float bp = s - x;
    const float err = (p - bp) + (x - (s - bp));
    unsigned int bits = __float_as_int(s);
    const unsigned int magnitude = bits & 0x7fffffffu;

    if (magnitude != 0u && magnitude < 0x7f800000u && (bits & 1u) == 0u) {
        const bool negative = (bits & 0x80000000u) != 0u;
        if (err > 0.0f) {
            bits += negative ? 0xffffffffu : 1u;
        } else if (err < 0.0f) {
            bits += negative ? 1u : 0xffffffffu;
        }
    }
    return __float2bfloat16(__uint_as_float(bits));
}

__device__ __forceinline__ __nv_bfloat16 negate(__nv_bfloat16 v) {
    __nv_bfloat16 out;
    out.raw = static_cast<unsigned short>(v.raw ^ 0x8000u);
    return out;
}

}

__device__ __forceinline__ __nv_bfloat162 __hfma2(
    __nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    unsigned int bits;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)),
          "r"(pie_bf16_detail::pack(c)));
    return pie_bf16_detail::unpack(bits);
}

__device__ __forceinline__ __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    unsigned int bits;

    asm("{ .reg .b32 o, m;\n"
        "  mov.b32 o, 0x3f803f80;\n"
        "  xor.b32 m, %2, 0x80008000;\n"
        "  fma.rn.bf16x2 %0, %1, o, m; }"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
}


__device__ __forceinline__ __nv_bfloat162 operator*(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __hmul2(a, b);
}

__device__ __forceinline__ __nv_bfloat162 __hadd2_rn(__nv_bfloat162 a, __nv_bfloat162 b) {
    const float2 x = __bfloat1622float2(a);
    const float2 y = __bfloat1622float2(b);
    float2 sum;
    sum.x = x.x + y.x;
    sum.y = x.y + y.y;
    return __float22bfloat162_rn(sum);
}

__device__ __forceinline__ __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 v) {
    return make_bfloat162(v, v);
}

__device__ __forceinline__ unsigned short __bfloat16_as_ushort(__nv_bfloat16 v) { return v.raw; }

__device__ __forceinline__ __nv_bfloat16 __habs(__nv_bfloat16 a) {
    __nv_bfloat16 out;
    out.raw = static_cast<unsigned short>(a.raw & 0x7fffu);
    return out;
}

__device__ __forceinline__ __nv_bfloat162 __habs2(__nv_bfloat162 a) {
    return make_bfloat162(__habs(a.x), __habs(a.y));
}

__device__ __forceinline__ __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b) {
    __nv_bfloat16 out;
    asm("max.bf16 %0, %1, %2;" : "=h"(out.raw) : "h"(a.raw), "h"(b.raw));
    return out;
}

__device__ __forceinline__ __nv_bfloat162 __hmax2(__nv_bfloat162 a, __nv_bfloat162 b) {
    unsigned int bits;
    asm("max.bf16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
}

__device__ __forceinline__ __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

#undef PIE_BF16_HAS_SM90
