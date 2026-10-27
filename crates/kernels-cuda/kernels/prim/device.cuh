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

    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return bf16{r};
}

__device__ __forceinline__ unsigned int pack_bf16x2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// a * b + c a lane of each pair, each rounded once: one packed f32x2 FMA
// where the chip has it (sm_100 up), else two.
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    unsigned long long d;
    asm("{\n\t.reg .b64 a, b, c;\n\t"
        "mov.b64 a, {%1, %2};\n\tmov.b64 b, {%3, %4};\n\tmov.b64 c, {%5, %6};\n\t"
        "fma.rn.f32x2 %0, a, b, c;\n\t}"
        : "=l"(d)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
    return make_float2(__uint_as_float(static_cast<unsigned>(d)), __uint_as_float(static_cast<unsigned>(d >> 32)));
#else
    return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ bf16 u16_as_bf16(unsigned short v) { return bf16{v}; }
__device__ __forceinline__ unsigned int bf16x2_as_u32(bf16 lo, bf16 hi) {
    return (static_cast<unsigned int>(hi.raw) << 16) | lo.raw;
}

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

template <>
struct Elem<float> {
    static __device__ __forceinline__ float to_f32(float v) { return v; }
    static __device__ __forceinline__ float from_f32(float v) { return v; }
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

// Every lane's sum of `v` over its warp, offsets 16, 8, 4, 2, 1: the
// butterfly the kernels walked by hand, so the result is bit for bit theirs.
template <class T>
__device__ __forceinline__ T warp_sum(T v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
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

// An e4m3 code's value (its finite codes; exact in bf16).
__device__ __forceinline__ float e4m3_to_f32(unsigned b) {
    const unsigned e = (b >> 3) & 0xf;
    const unsigned m = b & 7;
    const float v = e != 0 ? __uint_as_float(((e + 120) << 23) | (m << 20))
                           : static_cast<float>(m) * 0.001953125f;
    return (b & 0x80) != 0 ? -v : v;
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

// Programmatic dependent launch (`kernel.pdl`). A kernel launched as the
// stream's previous kernel's dependent calls `pdl_wait` before it reads or
// writes anything that kernel may still touch: it returns once that grid
// has finished and flushed. `pdl_launch` lets this kernel's own dependents
// launch (they wait likewise). Without the launch attribute both return at
// once, and before sm_90 they are nothing.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// A count other blocks of the grid add to, read with acquire at gpu
// scope: what an adder wrote before its release is visible after.
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// The same count read relaxed, for a spin: an acquire load invalidates the
// SM's L1 (CCTL.IVALL) each time, which a spinning warp would do every few
// hundred cycles under its block's other warps. Spin relaxed, then acquire
// once.
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void red_add_release_gpu(unsigned* p, unsigned v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// 16 bytes streamed past L1 (read once: weights, codes).
__device__ __forceinline__ uint4 ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Two e4m3 codes (the low byte first) as f32, exactly: sm_89's convert, or
// below it each code laid into an f32 by integer ops (2^-120 times its
// value, an e4m3 subnormal an f32 one) and taken back up.
__device__ __forceinline__ void e4m3_pair_f32(unsigned short pair, float& lo, float& hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    const unsigned b0 = pair & 0xFFu, b1 = static_cast<unsigned>(pair) >> 8;
    lo = __uint_as_float(((b0 & 0x80u) << 24) | ((b0 & 0x7Fu) << 20)) * 0x1p120f;
    hi = __uint_as_float(((b1 & 0x80u) << 24) | ((b1 & 0x7Fu) << 20)) * 0x1p120f;
#else
    asm("{\n\t.reg .b32 h2;\n\t.reg .b16 l, h;\n\t"
        "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
        "mov.b32 {l, h}, h2;\n\t"
        "cvt.f32.f16 %0, l;\n\t"
        "cvt.f32.f16 %1, h;\n\t}"
        : "=f"(lo), "=f"(hi)
        : "h"(pair));
#endif
}

}
