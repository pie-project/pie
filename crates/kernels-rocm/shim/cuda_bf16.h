
#pragma once

#include "prelude/device.cuh"

#include "cuda_fp16.h"

// gfx94x has no bf16 arithmetic a unit can name, so every operation below
// is f32 work rounded once to bf16: the fma through `fma_once`'s round-to-odd,
// which is correctly rounded, and the rest through it as PTX did on sm_80.


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
    return ::pie::f32_to_bf16(f);
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
    return pie_bf16_detail::unpack(::pie::pack_bf16x2(lo, hi));
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
    return make_bfloat162(pie_bf16_detail::fma_once(a.x, b.x, c.x),
                          pie_bf16_detail::fma_once(a.y, b.y, c.y));
}

// PTX's sm_80 `__hmul2` was `fma(a, b, -0)`; the same here.
__device__ __forceinline__ __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    const __nv_bfloat16 z{static_cast<unsigned short>(0x8000u)};
    return make_bfloat162(pie_bf16_detail::fma_once(a.x, b.x, z),
                          pie_bf16_detail::fma_once(a.y, b.y, z));
}

// PTX's `__hsub2` was `fma(a, 1, -b)`; the same here.
__device__ __forceinline__ __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    const __nv_bfloat16 one{static_cast<unsigned short>(0x3f80u)};
    return make_bfloat162(
        pie_bf16_detail::fma_once(a.x, one, pie_bf16_detail::negate(b.x)),
        pie_bf16_detail::fma_once(a.y, one, pie_bf16_detail::negate(b.y)));
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

// PTX's `max.bf16`: the larger, and a NaN only when both are NaN. The f32
// widening is exact, so the f32 max picks the same value.
__device__ __forceinline__ __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
}

__device__ __forceinline__ __nv_bfloat162 __hmax2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return make_bfloat162(__hmax(a.x, b.x), __hmax(a.y, b.y));
}

__device__ __forceinline__ __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

