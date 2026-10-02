
#pragma once

#if !defined(__CUDA_ARCH__)
#error "cuda_fp8.h (pie shim) is device text: it is compiled by NVRTC for one \
architecture. There is no host half, because the host half of this crate is Rust."
#endif

// The E4M3/E5M2 conversions are sm_89 instructions (cvt.rn.satfinite.e4m3x2.f32
// and cvt.rn.f16x2.e4m3x2). Below that -- an sm_80 A100 -- the same conversions
// are done in integer arithmetic. Every unit that touches fp8 includes this
// header, and most of them (attn/kv, the FlashInfer roots) also carry bf16
// kernels, so an #error here refused every model on Ampere rather than the
// fp8 ones; a portable path that lowers everywhere is the smaller failure
// surface. The portable path answers the hardware bit for bit: it was held
// against the cvt over every f32 bit pattern for both encodings and every
// code for both decodings on an sm_120. The one place the two could have
// parted is NaN, which the hardware lands as 0x7F with the sign dropped, for
// both encodings; the portable path does the same. Defining
// PIE_HALFTYPE_FORCE_PORTABLE selects the portable path on any architecture,
// which is how it is gated against the hardware on a box that has it.
#if defined(PIE_HALFTYPE_FORCE_PORTABLE)
#define PIE_FP8_HAS_SM89 0
#else
#define PIE_FP8_HAS_SM89 (__CUDA_ARCH__ >= 890)
#endif


typedef unsigned char __nv_fp8_storage_t;

typedef unsigned short __nv_fp8x2_storage_t;

typedef unsigned int __nv_fp8x4_storage_t;

typedef enum __nv_fp8_interpretation_t {
    __NV_E4M3 = 0,
    __NV_E5M2 = 1
} __nv_fp8_interpretation_t;

typedef enum __nv_saturation_t { __NV_SATFINITE = 1 } __nv_saturation_t;


// f32 -> fp8, round to nearest even, saturating to the largest finite value
// (cvt's .satfinite): E4M3 has no infinity and its 0x7F is NaN, E5M2 keeps
// infinity out of reach the same way. Subnormals are kept, not flushed.
__device__ __forceinline__ __nv_fp8_storage_t
__pie_fp8_from_f32_portable(const float f, const __nv_fp8_interpretation_t interp) {
    const unsigned int x = __float_as_uint(f);
    const unsigned int absx = x & 0x7FFFFFFFu;
    const unsigned int sign = (x >> 31) << 7;
    unsigned int sig, mask, min_denorm_o2, min_norm, overflow, maxnorm;
    int bias;
    if (interp == __NV_E5M2) {
        sig = 3u;
        mask = 0x3u;
        bias = 15;
        min_denorm_o2 = 0x37000000u;  // 2^-17, half the smallest subnormal
        min_norm = 0x38800000u;       // 2^-14
        overflow = 0x476EFFFFu;       // below 61440, the midpoint to infinity
        maxnorm = 0x7Bu;              // 57344
    } else {
        sig = 4u;
        mask = 0x7u;
        bias = 7;
        min_denorm_o2 = 0x3A800000u;  // 2^-10, half the smallest subnormal
        min_norm = 0x3C800000u;       // 2^-6
        overflow = 0x43E80000u;       // 464, the midpoint to the NaN code
        maxnorm = 0x7Eu;              // 448
    }
    const unsigned int half_ulp = 1u << (24u - sig - 1u);
    const int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + bias;
    unsigned int mantissa = (x >> (24u - sig)) & mask;
    unsigned int res;
    if (absx <= min_denorm_o2) {
        res = 0u;
    } else if (absx > 0x7F800000u) {
        return static_cast<__nv_fp8_storage_t>(0x7Fu);
    } else if (absx > overflow) {
        res = maxnorm;
    } else if (absx >= min_norm) {
        res = (static_cast<unsigned int>(exp) << (sig - 1u)) | mantissa;
        const unsigned int round = x & ((half_ulp << 1u) - 1u);
        if (round > half_ulp || (round == half_ulp && (mantissa & 1u))) res += 1u;
    } else {
        const unsigned int shift = static_cast<unsigned int>(1 - exp);
        mantissa |= 1u << (sig - 1u);
        res = mantissa >> shift;
        const unsigned int round = (x | (1u << 23)) & ((half_ulp << (shift + 1u)) - 1u);
        if (round > (half_ulp << shift) || (round == (half_ulp << shift) && (res & 1u))) res += 1u;
    }
    return static_cast<__nv_fp8_storage_t>(res | sign);
}

// fp8 -> f16 bits. Every fp8 value is exactly representable in f16, so this
// is a widening: E5M2 is the top byte of an f16 outright, E4M3 rebiases the
// exponent and normalises its subnormals. NaN lands as the canonical 0x7FFF.
__device__ __forceinline__ unsigned short
__pie_fp8_to_halfbits_portable(const __nv_fp8_storage_t x,
                               const __nv_fp8_interpretation_t interp) {
    unsigned short ur = static_cast<unsigned short>(static_cast<unsigned short>(x) << 8);
    if (interp == __NV_E5M2) {
        if ((ur & 0x7FFFu) > 0x7C00u) ur = 0x7FFFu;
        return ur;
    }
    if ((x & 0x7Fu) == 0x7Fu) return 0x7FFFu;
    const unsigned short sign = ur & 0x8000u;
    unsigned short exponent = static_cast<unsigned short>(((ur & 0x7800u) >> 1) + 0x2000u);
    unsigned short mantissa = (ur & 0x0700u) >> 1;
    if (exponent == 0x2000u) {
        if (mantissa != 0u) {
            mantissa = static_cast<unsigned short>(mantissa << 1);
            while ((mantissa & 0x0400u) == 0u) {
                mantissa = static_cast<unsigned short>(mantissa << 1);
                exponent = static_cast<unsigned short>(exponent - 0x0400u);
            }
            mantissa &= 0x03FFu;
        } else {
            exponent = 0u;
        }
    }
    return static_cast<unsigned short>(sign | exponent | mantissa);
}

// The two primitives everything below is spelled with: a float2 to a packed
// fp8 pair (x low, y high), and a packed fp8 pair to a packed f16 pair.
__device__ __forceinline__ __nv_fp8x2_storage_t
__pie_fp8x2_from_float2(const float2 x, const __nv_fp8_interpretation_t interp) {
#if PIE_FP8_HAS_SM89
    __nv_fp8x2_storage_t storage;
    if (interp == __NV_E5M2) {
        asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
            : "=h"(storage)
            : "f"(x.y), "f"(x.x));
    } else {
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(storage)
            : "f"(x.y), "f"(x.x));
    }
    return storage;
#else
    return static_cast<__nv_fp8x2_storage_t>(
        __pie_fp8_from_f32_portable(x.x, interp) |
        (static_cast<unsigned int>(__pie_fp8_from_f32_portable(x.y, interp)) << 8));
#endif
}

__device__ __forceinline__ unsigned int
__pie_fp8x2_to_halfbits2(const __nv_fp8x2_storage_t x, const __nv_fp8_interpretation_t interp) {
#if PIE_FP8_HAS_SM89
    unsigned int pair;
    if (interp == __NV_E5M2) {
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(pair) : "h"(x));
    } else {
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(x));
    }
    return pair;
#else
    return static_cast<unsigned int>(
               __pie_fp8_to_halfbits_portable(static_cast<__nv_fp8_storage_t>(x & 0xFFu), interp)) |
           (static_cast<unsigned int>(
                __pie_fp8_to_halfbits_portable(static_cast<__nv_fp8_storage_t>(x >> 8), interp))
            << 16);
#endif
}


__device__ __forceinline__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(
    const float2 x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation) {
    (void)saturate;
    return __pie_fp8x2_from_float2(x, fp8_interpretation);
}

__device__ __forceinline__ __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation) {
    const float2 pair = make_float2(x, 0.0f);
    return (__nv_fp8_storage_t)__nv_cvt_float2_to_fp8x2(pair, saturate,
                                                        fp8_interpretation);
}

#if defined(__CUDA_FP16_TYPES_EXIST__)

__device__ __forceinline__ __half_raw
__nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
                        const __nv_fp8_interpretation_t fp8_interpretation) {
    const unsigned int pair = __pie_fp8x2_to_halfbits2((unsigned short)x, fp8_interpretation);
    __half_raw res;
    res.x = (unsigned short)(pair & 0xFFFFu);
    return res;
}

#endif


__device__ __forceinline__ float __pie_fp8_halfbits_to_float(const unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

__device__ __forceinline__ __nv_fp8_storage_t
__pie_fp8_from_float(const float f, const __nv_fp8_interpretation_t interp) {
    return __nv_cvt_float_to_fp8(f, __NV_SATFINITE, interp);
}


struct __nv_fp8_e4m3 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e4m3() = default;

    explicit __device__ __forceinline__ __nv_fp8_e4m3(const float f) {
        __x = __pie_fp8_from_float(f, __NV_E4M3);
    }

#if defined(__CUDA_FP16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8_e4m3(const __half f) {
        __x = __pie_fp8_from_float(
            __pie_fp8_halfbits_to_float(static_cast<__half_raw>(f).x), __NV_E4M3);
    }

    explicit __device__ __forceinline__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E4M3));
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8_e4m3(const __nv_bfloat16 f) {
        const unsigned int bits = ((unsigned int)static_cast<__nv_bfloat16_raw>(f).x) << 16;
        __x = __pie_fp8_from_float(__int_as_float((int)bits), __NV_E4M3);
    }

    explicit __device__ __forceinline__ operator __nv_bfloat16() const {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(float(*this)));
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return static_cast<__nv_bfloat16>(raw);
    }
#endif

    explicit __device__ __forceinline__ operator float() const {
        const unsigned int pair = __pie_fp8x2_to_halfbits2((unsigned short)__x, __NV_E4M3);
        return __pie_fp8_halfbits_to_float((unsigned short)(pair & 0xFFFFu));
    }
};

struct __nv_fp8_e5m2 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e5m2() = default;

    explicit __device__ __forceinline__ __nv_fp8_e5m2(const float f) {
        __x = __pie_fp8_from_float(f, __NV_E5M2);
    }

#if defined(__CUDA_FP16_TYPES_EXIST__)
    explicit __device__ __forceinline__ __nv_fp8_e5m2(const __half f) {
        __x = __pie_fp8_from_float(
            __pie_fp8_halfbits_to_float(static_cast<__half_raw>(f).x), __NV_E5M2);
    }

    explicit __device__ __forceinline__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E5M2));
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
    explicit __device__ __forceinline__ __nv_fp8_e5m2(const __nv_bfloat16 f) {
        const unsigned int bits = ((unsigned int)static_cast<__nv_bfloat16_raw>(f).x) << 16;
        __x = __pie_fp8_from_float(__int_as_float((int)bits), __NV_E5M2);
    }

    explicit __device__ __forceinline__ operator __nv_bfloat16() const {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(float(*this)));
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return static_cast<__nv_bfloat16>(raw);
    }
#endif

    explicit __device__ __forceinline__ operator float() const {
        const unsigned int pair = __pie_fp8x2_to_halfbits2((unsigned short)__x, __NV_E5M2);
        return __pie_fp8_halfbits_to_float((unsigned short)(pair & 0xFFFFu));
    }
};


struct __nv_fp8x2_e4m3 {
    __nv_fp8x2_storage_t __x;

    __nv_fp8x2_e4m3() = default;

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(__nv_fp8x2_storage_t bits) : __x(bits) {}

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const float2 f)
        : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3)) {}

#if defined(__CUDA_FP16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const __half2 v) {
        const __half2_raw raw = static_cast<__half2_raw>(v);
        float2 f;
        f.x = __pie_fp8_halfbits_to_float(raw.x);
        f.y = __pie_fp8_halfbits_to_float(raw.y);
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }

    explicit __device__ __forceinline__ operator __half2() const {
        const unsigned int pair = __pie_fp8x2_to_halfbits2(__x, __NV_E4M3);
        __half2_raw raw;
        raw.x = (unsigned short)(pair & 0xFFFFu);
        raw.y = (unsigned short)(pair >> 16);
        return static_cast<__half2>(raw);
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const __nv_bfloat162 v) {
        const __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(v);
        float2 f;
        f.x = __int_as_float((int)(((unsigned int)raw.x) << 16));
        f.y = __int_as_float((int)(((unsigned int)raw.y) << 16));
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
#endif
};

struct __nv_fp8x2_e5m2 {
    __nv_fp8x2_storage_t __x;
};

struct __nv_fp8x4_e4m3 {
    __nv_fp8x4_storage_t __x;
};

struct __nv_fp8x4_e5m2 {
    __nv_fp8x4_storage_t __x;
};
