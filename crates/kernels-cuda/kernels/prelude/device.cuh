#pragma once

namespace pie {

struct bf16 {
    unsigned short raw;

    bf16() = default;

    explicit constexpr __device__ bf16(unsigned short bits) : raw(bits) {}

    explicit __device__ bf16(float f);

    __device__ bf16& operator=(float f);

    explicit __device__ operator float() const;
};

struct f16 {
    unsigned short raw;

    f16() = default;

    explicit constexpr __device__ f16(unsigned short bits) : raw(bits) {}

    explicit __device__ f16(float f);

    __device__ f16& operator=(float f);

    explicit __device__ operator float() const;
};

__device__ __forceinline__ float bf16_to_f32(bf16 v) {
    float f;
    asm("mov.b32 %0, {0, %1};" : "=f"(f) : "h"(v.raw));
    return f;
}

__device__ __forceinline__ bf16 f32_to_bf16(float f) {
    const unsigned int b = __float_as_int(f);

    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return bf16{static_cast<unsigned short>((b >> 16) | 0x0040u)};
    }
    const unsigned int rounding = 0x7fffu + ((b >> 16) & 1u);
    return bf16{static_cast<unsigned short>((b + rounding) >> 16)};
}

__device__ __forceinline__ unsigned short bf16_as_u16(bf16 v) { return v.raw; }
__device__ __forceinline__ bf16 u16_as_bf16(unsigned short v) { return bf16{v}; }

__device__ __forceinline__ bf16 f32_to_bf16_rd(float f) {
    const unsigned int b = __float_as_int(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return bf16{static_cast<unsigned short>((b >> 16) | 0x0040u)};
    }
    unsigned short hi = static_cast<unsigned short>(b >> 16);

    if ((b & 0xffffu) != 0u && (b >> 31) != 0u) {
        hi = static_cast<unsigned short>(hi + 1);
    }
    return bf16{hi};
}

__device__ __forceinline__ bf16 f32_to_bf16_ru(float f) {
    const unsigned int b = __float_as_int(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return bf16{static_cast<unsigned short>((b >> 16) | 0x0040u)};
    }
    unsigned short hi = static_cast<unsigned short>(b >> 16);
    if ((b & 0xffffu) != 0u && (b >> 31) == 0u) {
        hi = static_cast<unsigned short>(hi + 1);
    }
    return bf16{hi};
}

__device__ __forceinline__ float f16_to_f32(f16 v) {
    const unsigned int s = (static_cast<unsigned int>(v.raw) & 0x8000u) << 16;
    const unsigned int e = (static_cast<unsigned int>(v.raw) >> 10) & 0x1fu;
    const unsigned int m = static_cast<unsigned int>(v.raw) & 0x3ffu;
    if (e == 0) {

        if (m == 0) return __int_as_float(s);

        const float magnitude = __int_as_float(0x33800000u) * static_cast<float>(m);
        return __int_as_float(__float_as_int(magnitude) | s);
    }
    if (e == 31) return __int_as_float(s | 0x7f800000u | (m << 13));
    return __int_as_float(s | ((e + 112u) << 23) | (m << 13));
}

__device__ __forceinline__ f16 f32_to_f16(float f) {
    const unsigned int b = __float_as_int(f);
    const unsigned int s = (b >> 16) & 0x8000u;
    int e = static_cast<int>((b >> 23) & 0xffu) - 127 + 15;
    const unsigned int m = b & 0x7fffffu;
    if (e >= 31) {
        return f16{static_cast<unsigned short>(
            s | 0x7c00u | ((m && ((b >> 23 & 0xffu) == 0xffu)) ? 0x200u : 0u))};
    }
    if (e <= 0) return f16{static_cast<unsigned short>(s)};
    const unsigned int mm = m >> 13;
    const unsigned int round = ((m >> 12) & 1u) & (((m & 0xfffu) != 0u) | (mm & 1u));
    return f16{static_cast<unsigned short>(
        (s | (static_cast<unsigned int>(e) << 10) | mm) + round)};
}

__device__ __forceinline__ bf16::operator float() const { return bf16_to_f32(*this); }
__device__ __forceinline__ f16::operator float() const { return f16_to_f32(*this); }
__device__ __forceinline__ bf16::bf16(float f) : raw(f32_to_bf16(f).raw) {}
__device__ __forceinline__ f16::f16(float f) : raw(f32_to_f16(f).raw) {}
__device__ __forceinline__ bf16& bf16::operator=(float f) {
    raw = f32_to_bf16(f).raw;
    return *this;
}
__device__ __forceinline__ f16& f16::operator=(float f) {
    raw = f32_to_f16(f).raw;
    return *this;
}

template <class T>
struct Elem;

template <>
struct Elem<bf16> {
    static __device__ __forceinline__ float to_f32(bf16 v) { return bf16_to_f32(v); }
    static __device__ __forceinline__ bf16 from_f32(float v) { return f32_to_bf16(v); }
};

template <>
struct Elem<f16> {
    static __device__ __forceinline__ float to_f32(f16 v) { return f16_to_f32(v); }
    static __device__ __forceinline__ f16 from_f32(float v) { return f32_to_f16(v); }
};

__device__ __forceinline__ float block_sum(float local, float* smem) {
    const unsigned int active = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(active, local, off);
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warps = static_cast<int>((blockDim.x + 31) >> 5);
    if (lane == 0) smem[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float v = (static_cast<int>(threadIdx.x) < warps) ? smem[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(active, v, off);
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float flt_max() { return __int_as_float(0x7f7fffffu); }
__device__ __forceinline__ float pos_inf() { return __int_as_float(0x7f800000u); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000u); }

struct __align__(4) bf16x2 {
    bf16 x;
    bf16 y;
};

__device__ __forceinline__ float2 bf16x2_to_f32(bf16x2 v) {
    float2 out;
    out.x = bf16_to_f32(v.x);
    out.y = bf16_to_f32(v.y);
    return out;
}

__device__ __forceinline__ bf16x2 f32_to_bf16x2(float lo, float hi) {
    bf16x2 out;
    out.x = f32_to_bf16(lo);
    out.y = f32_to_bf16(hi);
    return out;
}

template <class T>
__device__ __forceinline__ T ldg(const T* p) {
    return __ldg(p);
}

template <>
__device__ __forceinline__ bf16 ldg<bf16>(const bf16* p) {
    return bf16{__ldg(reinterpret_cast<const unsigned short*>(p))};
}

template <>
__device__ __forceinline__ bf16x2 ldg<bf16x2>(const bf16x2* p) {
    const unsigned int raw = __ldg(reinterpret_cast<const unsigned int*>(p));
    bf16x2 out;
    out.x = bf16{static_cast<unsigned short>(raw & 0xffffu)};
    out.y = bf16{static_cast<unsigned short>(raw >> 16)};
    return out;
}

__device__ __forceinline__ float bf16x2_lo(bf16x2 v) { return bf16_to_f32(v.x); }
__device__ __forceinline__ float bf16x2_hi(bf16x2 v) { return bf16_to_f32(v.y); }

template <class A, class B>
struct is_same {
    static constexpr bool value = false;
};
template <class A>
struct is_same<A, A> {
    static constexpr bool value = true;
};

template <bool C, class T, class F>
struct conditional {
    using type = T;
};
template <class T, class F>
struct conditional<false, T, F> {
    using type = F;
};

using i8 = signed char;
using u8 = unsigned char;
using i16 = short;
using u16 = unsigned short;
using i32 = int;
using u32 = unsigned int;

using usize = decltype(sizeof(0));
using isize = decltype(static_cast<char*>(nullptr) - static_cast<char*>(nullptr));

using i64 = long long;
using u64 = unsigned long long;

static_assert(sizeof(i8) == 1 && sizeof(i16) == 2 && sizeof(i32) == 4 && sizeof(i64) == 8);
static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8);
static_assert(sizeof(usize) == sizeof(void*) && sizeof(isize) == sizeof(void*));

struct true_type {
    static constexpr bool value = true;
};
struct false_type {
    static constexpr bool value = false;
};

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ void warp_argmax(float& v, int& idx) {
    for (int off = 16; off > 0; off >>= 1) {
        const float other_v = __shfl_down_sync(0xffffffffu, v, off);
        const int other_i = __shfl_down_sync(0xffffffffu, idx, off);
        if (other_v > v || (other_v == v && other_i < idx)) {
            v = other_v;
            idx = other_i;
        }
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    idx = __shfl_sync(0xffffffffu, idx, 0);
}

}
