
#pragma once

#include "prelude/device.cuh"

// gfx9 and newer have every fp16 operation below in hardware; clang's
// `_Float16` lowers to them, rounding to nearest even like PTX's `.rn`.


using __half = ::pie::f16;

using half = __half;

struct __align__(4) __half2 {
    __half x;
    __half y;
};

using half2 = __half2;

struct __align__(2) __half_raw {
    unsigned short x;

    __half_raw() = default;

    __device__ constexpr __half_raw(unsigned short bits) : x(bits) {}
    __device__ __forceinline__ __half_raw(const __half h) : x(h.raw) {}

    __device__ constexpr operator __half() const { return __half{x}; }
};

struct __align__(4) __half2_raw {
    unsigned short x;
    unsigned short y;

    __half2_raw() = default;
    __device__ __forceinline__ __half2_raw(const __half2 v) : x(v.x.raw), y(v.y.raw) {}
    __device__ __forceinline__ operator __half2() const {
        __half2 out;
        out.x.raw = x;
        out.y.raw = y;
        return out;
    }
};

#define __CUDA_FP16_TYPES_EXIST__

using nv_half = __half;
using nv_half2 = __half2;
using __nv_half = __half;
using __nv_half2 = __half2;
using __nv_half_raw = __half_raw;
using __nv_half2_raw = __half2_raw;

namespace pie_fp16_detail {

__device__ __forceinline__ unsigned int pack(__half2 v) {
    return (static_cast<unsigned int>(v.y.raw) << 16) | static_cast<unsigned int>(v.x.raw);
}

__device__ __forceinline__ __half2 unpack(unsigned int bits) {
    __half2 out;
    out.x.raw = static_cast<unsigned short>(bits & 0xffffu);
    out.y.raw = static_cast<unsigned short>(bits >> 16);
    return out;
}

__device__ __forceinline__ _Float16 native(__half h) { return __builtin_bit_cast(_Float16, h.raw); }

__device__ __forceinline__ __half wrap(_Float16 h) {
    __half out;
    out.raw = __builtin_bit_cast(unsigned short, h);
    return out;
}

}


__device__ __forceinline__ __half2 make_half2(__half x, __half y);

__device__ __forceinline__ float __half2float(__half h) {
    return static_cast<float>(pie_fp16_detail::native(h));
}

__device__ __forceinline__ __half __float2half(float f) {
    return pie_fp16_detail::wrap(static_cast<_Float16>(f));
}

__device__ __forceinline__ __half __float2half_rn(float f) {
    return __float2half(f);
}

__device__ __forceinline__ __half2 __float2half2_rn(float f) {
    const __half t = __float2half(f);
    return make_half2(t, t);
}

__device__ __forceinline__ __half2 __floats2half2_rn(float lo, float hi) {
    return make_half2(__float2half(lo), __float2half(hi));
}

__device__ __forceinline__ __half2 __float22half2_rn(float2 f) {
    return __floats2half2_rn(f.x, f.y);
}

__device__ __forceinline__ __half2 __halves2half2(__half lo, __half hi) {
    __half2 out;
    out.x = lo;
    out.y = hi;
    return out;
}

__device__ __forceinline__ float2 __half22float2(__half2 v) {
    float2 out;
    out.x = __half2float(v.x);
    out.y = __half2float(v.y);
    return out;
}

__device__ __forceinline__ __half2 make_half2(__half x, __half y) {
    __half2 out;
    out.x = x;
    out.y = y;
    return out;
}


__device__ __forceinline__ unsigned short __half_as_ushort(__half h) {
    return h.raw;
}

__device__ __forceinline__ __half __ushort_as_half(unsigned short bits) {
    __half out;
    out.raw = bits;
    return out;
}


__device__ __forceinline__ __half __hmul(__half a, __half b) {
    return pie_fp16_detail::wrap(pie_fp16_detail::native(a) * pie_fp16_detail::native(b));
}

__device__ __forceinline__ __half __hsub(__half a, __half b) {
    return pie_fp16_detail::wrap(pie_fp16_detail::native(a) - pie_fp16_detail::native(b));
}

__device__ __forceinline__ __half __hadd(__half a, __half b) {
    return pie_fp16_detail::wrap(pie_fp16_detail::native(a) + pie_fp16_detail::native(b));
}

__device__ __forceinline__ __half __hmax(__half a, __half b) {
    return pie_fp16_detail::wrap(__builtin_fmaxf16(pie_fp16_detail::native(a), pie_fp16_detail::native(b)));
}

__device__ __forceinline__ __half2 __hmul2(__half2 a, __half2 b) {
    return make_half2(__hmul(a.x, b.x), __hmul(a.y, b.y));
}

__device__ __forceinline__ __half2 __hsub2(__half2 a, __half2 b) {
    return make_half2(__hsub(a.x, b.x), __hsub(a.y, b.y));
}

namespace pie_fp16_detail {

__device__ __forceinline__ __half fma_once(__half a, __half b, __half c) {
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    const float fc = __half2float(c);
    const float p = fa * fb;
    const float s = p + fc;
    const float t = s - p;
    const float err = (p - (s - t)) + (fc - t);

    unsigned int bits = __float_as_uint(s);
    const unsigned int magnitude = bits & 0x7fffffffu;

    if (magnitude != 0u && magnitude < 0x7f800000u && (bits & 1u) == 0u) {
        const bool negative = (bits & 0x80000000u) != 0u;
        if (err > 0.0f) {
            bits += negative ? 0xffffffffu : 1u;
        } else if (err < 0.0f) {
            bits += negative ? 1u : 0xffffffffu;
        }
    }
    return __float2half(__uint_as_float(bits));
}

}

__device__ __forceinline__ __half2 __hfma2(__half2 a, __half2 b, __half2 c) {
    using pie_fp16_detail::native;
    using pie_fp16_detail::wrap;
    return make_half2(wrap(__builtin_fmaf16(native(a.x), native(b.x), native(c.x))),
                      wrap(__builtin_fmaf16(native(a.y), native(b.y), native(c.y))));
}

__device__ __forceinline__ __half2 __hmax2(__half2 a, __half2 b) {
    return make_half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
}

__device__ __forceinline__ __half2 __half2half2(__half v) { return make_half2(v, v); }

__device__ __forceinline__ __half __habs(__half a) {
    __half out;
    out.raw = static_cast<unsigned short>(a.raw & 0x7fffu);
    return out;
}

__device__ __forceinline__ __half2 __habs2(__half2 a) {
    return make_half2(__habs(a.x), __habs(a.y));
}


__device__ __forceinline__ __half operator*(__half a, __half b) {
    return __hmul(a, b);
}

__device__ __forceinline__ __half operator-(__half a, __half b) {
    return __hsub(a, b);
}

__device__ __forceinline__ __half2 operator*(__half2 a, __half2 b) {
    return __hmul2(a, b);
}

__device__ __forceinline__ __half2 operator-(__half2 a, __half2 b) {
    return __hsub2(a, b);
}


__device__ __forceinline__ __half2 __shfl_xor_sync(unsigned int mask, __half2 var, int lane_mask,
                                                   int width = warpSize) {
    return pie_fp16_detail::unpack(
        __shfl_xor_sync(mask, pie_fp16_detail::pack(var), lane_mask, width));
}


