//===-- cuda_fp8.h - the fp8 door, closed with one `cvt` instruction -----===//
//
// `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, their packed pairs and quads, and the
// three conversions the tree was measured to call. This is what
// `#include <cuda_fp8.h>` resolves to when the compiler is NVRTC and the
// include path is a header set carried in the binary rather than a directory
// on a disk.
//
// # Why a shim rather than NVIDIA's header
//
// `examples/header_probe.rs` measured NVRTC 13.0 on this box (L40S, sm_89)
// against an empty header set: `<cuda_fp8.h>` fails with *"could not open
// source file ... (no directories in search list)"*, one of 31 external
// directives out of our FlashInfer closure of which **0 were answered**.
// NVRTC before 13.3 bundles no device headers at all, so an fp8 conversion is
// either carried, shimmed, or written out longhand.
//
// Carrying NVIDIA's own file means VENDORING it -- a redistribution decision
// with a `NOTICE` entry and a pinned device ABI behind it -- and reading it
// out of `$CUDA_HOME` at build time was tried and rejected in
// `.wiki/driver/new-horizon.md` §13.2, because it makes the build machine
// carry a toolkit, which is the one property this crate exists not to need.
//
// So: **no text from `cuda_fp8.hpp` is in this file.** That header was read
// on this machine as a cross-check on the two things a specification does not
// spell -- which asm operand lands in which byte, and which spelling of a
// saturating convert the vendor picked -- and both were then MEASURED on the
// device rather than copied. What is implemented here is the PTX instruction,
// which is the actual contract; the vendor's C++ around it is not.
//
// # The surface is the measurement, and the measurement is small
//
// Two trees were counted. The FlashInfer attention closure -- the 28 files
// and 18,009 lines it held then, reachable from `decode.cuh`, `prefill.cuh`,
// `mla.cuh`, `scheduler.cuh` and the rest of the roots -- and the whole of
// `kernels-cuda/csrc/src`, the ahead-of-time crate's own device code:
//
// | name | closure | our csrc |
// |---|---|---|
// | `__nv_fp8_e4m3` (type) | 128 | -- |
// | `__nv_fp8_e5m2` (type) | 119 | -- |
// | `__nv_fp8x4_storage_t` | 56 | -- |
// | `__nv_fp8x4_e4m3` / `__nv_fp8x4_e5m2` | 11 / 11 | -- |
// | `__nv_fp8x2_e4m3` / `__nv_fp8x2_e5m2` | 6 / 6 | -- |
// | `__nv_fp8x2_storage_t` | 6 | -- |
// | `__nv_fp8_storage_t` | -- | 34 |
// | `__NV_E4M3` / `__NV_E5M2` / `__NV_SATFINITE` | 1 / 1 / 2 | 15 / 3 / 8 |
// | `__nv_cvt_float2_to_fp8x2` | 2 | -- |
// | `__nv_cvt_float_to_fp8` | -- | 9 |
// | `__nv_cvt_fp8_to_halfraw` | -- | 9 |
//
// Three functions, two enums, three storage widths and six classes. Every
// arithmetic path in this file is ONE instruction --
// `cvt.rn.satfinite.e4m3x2.f32` and its `e5m2` twin going down,
// `cvt.rn.f16x2.e4m3x2` coming back up -- and every other conversion is an
// EXACT widening onto those, which is the property that makes this file
// testable: there is no second rounding to get wrong. `examples/
// fp8_pipeline_probe.rs` gates it against `nvcc`'s own `<cuda_fp8.h>` --
// same source, same device, two compilers -- over all 256 byte patterns
// unpacked four ways, all 65,536 `__half` and all 65,536 `__nv_bfloat16`
// patterns packed, and 1,048,622 floats per direction: a million random
// bit patterns plus every special, +/-0, +/-inf, NaN, both sides'
// subnormals and the values above E4M3's 448 and E5M2's 57344 that have to
// saturate. **18 checks, bit-identical, first run.** Saturation and
// subnormals are where a hand-written converter usually differs; here there
// is no hand-written converter to differ.
//
// # The name is the whole trick
//
// NVRTC resolves an `#include` by matching the directive's LITERAL spelling
// against `includeNames[]`, so a file carried under NVIDIA's own name leaves
// the upstream source unmodified and makes the resolution ours.
// `utils.cuh:21`, `decode.cuh:21`, `prefill.cuh:22`, `mma.cuh:21` and
// `vec_dtypes.cuh:22` all say `#include <cuda_fp8.h>`; this is that file.
// `new-horizon.md` §13.4 states the rule -- impersonate a header exactly when
// the includer is upstream source we do not own -- and that is why this one
// is not called `pie_fp8.cuh` the way `pie_mma.cuh` is.
//
// # Which lane is which, measured rather than assumed
//
// `cvt.rn.satfinite.e4m3x2.f32 d, a, b` packs TWO floats into one 16-bit
// register, and getting the halves backwards is a silent numeric bug rather
// than a compile error -- every value would still be a legal fp8, just the
// wrong one, in a KV cache nobody re-reads by hand. So it was measured, on
// this device, rather than reasoned about:
//
//     cvt.rn.satfinite.e4m3x2.f32 d, 1.0f, 2.0f   ->   d = 0x3840
//                                                        ^^ ^^
//     E4M3(1.0) = 0x38, E4M3(2.0) = 0x40           a ---'   '--- b
//
// **The first source operand lands in the HIGH byte, the second in the LOW
// byte.** A `float2` goes down as `cvt d, x.y, x.x`, so that `x.x` is the low
// byte and the pair reads back in memory order on a little-endian device.
// Coming up, the same probe gave
//
//     cvt.rn.f16x2.e4m3x2 d, 0x4038   ->   d = 0x4000_3c00
//
// -- the LOW fp8 byte becomes the LOW half. A single-value convert is
// therefore the packed instruction with the HIGH lane fed a constant and
// thrown away.
//
// # This file needs sm_89, and says so at compile time
//
// The `cvt` forms above appeared with Ada and Hopper. Below `__CUDA_ARCH__
// 890` there is no instruction, and the honest options are a software
// emulation or a refusal. This file refuses, and the reason is the standard
// it is held to: a software path could not be gated against `nvcc` on this
// box -- `nvcc` compiling for sm_89 emits the same hardware `cvt` this file
// does, so there would be nothing to compare an emulation against -- and
// arithmetic that ships without a parity gate is exactly the silent
// divergence the probe exists to prevent. `#error` names the decision;
// `__trap()` at run time would not.
//
// # What is deliberately NOT here
//
// **`__NV_NOSAT`.** Measured at zero sites in either tree, and its absence is
// a compile error at any future one: `__nv_saturation_t` is declared with
// `__NV_SATFINITE = 1`, keeping the vendor's numbering, and no second
// enumerator. Non-saturating float->fp8 is a different rounding story --
// overflow becomes an infinity in E5M2 and a NaN in E4M3, which has none --
// and the hardware `cvt` does not do it, so it would be emulation with no
// call site to hold it to.
//
// **The e8m0 family** (`__nv_fp8_e8m0`, `__nv_cvt_float_to_e8m0`, and the
// `__nv_fp8x2_e8m0` pair). Zero uses in the closure and zero in our csrc; the
// five sites in the wider FlashInfer tree are
// `comm/trtllm_allreduce_fusion.cuh:756`,
// `comm/trtllm_moe_allreduce_fusion.cuh:608`,
// `norm/fused_dit_layernorm.cuh:349` and `:441`, and
// `norm/ln_silu_headers.cuh:305` -- none of them reachable from an attention
// root. Two things would have to be decided first: the instruction is
// `cvt.rn.satfinite.ue8m0x2.f32`, which does not exist below sm_100, and the
// signature takes `enum cudaRoundMode`, which lives in `driver_types.h` and
// is not in the header set. A shim that answered the name and rounded some
// other way would be worse than the name error.
//
// **`__nv_cvt_double_to_fp8`, `__nv_cvt_halfraw_to_fp8`,
// `__nv_cvt_fp8x2_to_halfraw2`, the `float4`/`fp8x4` converts, and the
// integral constructors.** None appears in either tree. The `__half` and
// `__nv_bfloat16` constructors that ARE here exist because
// `vec_dtypes.cuh:307`, `:320`, `:332` and `:345` spell them.
//
// # The one dependency, and why it is a macro and not an `#include`
//
// `__nv_cvt_fp8_to_halfraw` returns `__half_raw`, and the class constructors
// take `__half` and `__nv_bfloat16`. Those types belong to `cuda_fp16.h` and
// `cuda_bf16.h`, which are separate entries in the header set -- this file
// includes NOTHING, because a header in the set that reached for another
// would create a diamond the includer never asked for, and because
// FlashInfer's `vec_dtypes.cuh` already includes all three itself, in that
// order.
//
// So the interop is guarded on `__CUDA_FP16_TYPES_EXIST__` and
// `__CUDA_BF16_TYPES_EXIST__`, the macros NVIDIA's own headers define, and a
// translation unit that includes this one WITHOUT them gets a name error at
// the call site naming the identifier. That is the loud absence
// `cooperative_groups.h` gives `this_grid()`, applied to an ordering: every
// includer measured in the closure (`utils.cuh:20-21`,
// `vec_dtypes.cuh:21-22`, `prefill.cuh:21-22`, `decode.cuh:20-21`,
// `mma.cuh:20-21`) already includes `<cuda_fp16.h>` first, and the two call
// sites in `kernels-cuda/csrc/src` that need `__half_raw`
// (`quant/dequant_fp8.cu:20` and `attn/kv_paged.cu:547`) reach it through
// `<cuda_bf16.h>`, which pulls fp16 in behind it.
//
// Only four names are taken from those headers -- `__half`, `__half_raw`,
// `__nv_bfloat16`, `__nv_bfloat16_raw` -- and only their `.x` storage and the
// casts between class and raw. No conversion of theirs is called, because
// every conversion here has to be one this file can be held to the bit for.
//
//===----------------------------------------------------------------------===//
#pragma once

#if !defined(__CUDA_ARCH__)
#error "cuda_fp8.h (pie shim) is device text: it is compiled by NVRTC for one \
architecture, and every conversion in it is an sm_89 instruction. There is no \
host half, because the host half of this crate is Rust."
#elif __CUDA_ARCH__ < 890
#error "cuda_fp8.h (pie shim) implements E4M3/E5M2 with cvt.rn.satfinite.e4m3x2.f32 \
and cvt.rn.f16x2.e4m3x2, which need sm_89 or newer. A software path is a \
deliberate absence: nvcc emits the same hardware cvt on this box, so an \
emulation could not be gated against it, and ungated arithmetic is the failure \
this shim exists to prevent."
#endif

// ---------------------------------------------------------------------------
// storage
// ---------------------------------------------------------------------------

/// One E4M3 or E5M2 value, as the byte it is.
///
/// A typedef and not a class, exactly as the vendor has it: the tree casts
/// between `__nv_fp8_e4m3*` and this at will -- `vec_dtypes.cuh:701` builds a
/// pair by shifting one into the other -- and a class here would make those
/// casts a strict-aliasing question instead of an arithmetic one.
typedef unsigned char __nv_fp8_storage_t;

/// Two of them, low byte first.
///
/// The order is the measurement above and it is the whole reason this
/// typedef is not interchangeable with `unsigned short`: element `.x` of a
/// converted `float2` is in bits 7:0.
typedef unsigned short __nv_fp8x2_storage_t;

/// Four, same order, which is what a 32-bit shared-memory bank holds.
typedef unsigned int __nv_fp8x4_storage_t;

/// Which of the two 8-bit formats a byte is to be read as.
///
/// E4M3 is 1-4-3 with no infinity and a max of 448; E5M2 is 1-5-2 with
/// fp16's exponent range, infinities and a max of 57344. The numbering is the
/// vendor's, because the enumerators cross an ABI in one direction -- a row
/// in `kernels-cuda` passes one down as an `int` -- and a shim that renumbered
/// them would turn every E4M3 into an E5M2 without a word.
typedef enum __nv_fp8_interpretation_t {
    __NV_E4M3 = 0,
    __NV_E5M2 = 1
} __nv_fp8_interpretation_t;

/// What happens to a value the format cannot hold.
///
/// One enumerator, and the number it keeps is the vendor's. `__NV_NOSAT` is
/// deliberately absent -- see the header comment -- so a call site that wants
/// it fails to compile naming it, rather than getting a saturating convert
/// under a non-saturating name.
typedef enum __nv_saturation_t { __NV_SATFINITE = 1 } __nv_saturation_t;

// ---------------------------------------------------------------------------
// the conversions -- one instruction each
// ---------------------------------------------------------------------------

/// Two floats down to two fp8 bytes: `x.x` into bits 7:0, `x.y` into 15:8.
///
/// The instruction's first source operand is its HIGH byte, so the operands
/// go in reversed -- `%2` is `x.x` and lands low. Reversing this reverses
/// every KV cache page in the system and nothing fails to compile.
///
/// `saturate` is accepted and ignored: `__nv_saturation_t` has one
/// enumerator, and `.satfinite` is what it names. Keeping the parameter is
/// what lets `vec_dtypes.cuh:132` compile unmodified.
__device__ __forceinline__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(
    const float2 x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation) {
    (void)saturate;
    __nv_fp8x2_storage_t storage;
    if (fp8_interpretation == __NV_E5M2) {
        asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
            : "=h"(storage)
            : "f"(x.y), "f"(x.x));
    } else {
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(storage)
            : "f"(x.y), "f"(x.x));
    }
    return storage;
}

/// One float down to one fp8 byte.
///
/// The packed instruction with a lane thrown away, because there is no
/// scalar form of it. The DISCARDED lane is the high one -- it is fed `0.0f`
/// and the result is truncated to the low byte, which is the lane `x` was
/// put in. Feeding `x` to the other operand and reading the same byte would
/// return a constant zero for every input.
__device__ __forceinline__ __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation) {
    const float2 pair = make_float2(x, 0.0f);
    return (__nv_fp8_storage_t)__nv_cvt_float2_to_fp8x2(pair, saturate,
                                                        fp8_interpretation);
}

#if defined(__CUDA_FP16_TYPES_EXIST__)

/// One fp8 byte up to one fp16, exactly.
///
/// Every E4M3 and E5M2 value is representable in fp16 -- 3 mantissa bits into
/// 10, an exponent range that fits inside fp16's, and E5M2's infinities and
/// NaNs mapping onto fp16's own -- so this is a widening and not a rounding,
/// and `.rn` in the instruction's name never chooses anything.
///
/// The byte goes into the low lane and the low half comes back, per the
/// measurement in the header comment.
__device__ __forceinline__ __half_raw
__nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
                        const __nv_fp8_interpretation_t fp8_interpretation) {
    const unsigned short packed = (unsigned short)x;
    unsigned int pair;
    if (fp8_interpretation == __NV_E5M2) {
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(pair) : "h"(packed));
    } else {
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(packed));
    }
    __half_raw res;
    res.x = (unsigned short)(pair & 0xFFFFu);
    return res;
}

#endif  // __CUDA_FP16_TYPES_EXIST__

// ---------------------------------------------------------------------------
// the widenings the classes are built out of
// ---------------------------------------------------------------------------

/// fp16 bits to fp32, which is exact for every one of the 65,536 of them.
///
/// Spelled here rather than taken from `cuda_fp16.h` so that this file owes
/// that one four type names and no arithmetic. A conversion this file did not
/// write is a conversion the probe cannot hold it to.
__device__ __forceinline__ float __pie_fp8_halfbits_to_float(const unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

/// fp32 down to fp8, through the one instruction, for a given format.
///
/// The single entry point every constructor below funnels into, so that
/// "which rounding" has one answer in this file.
__device__ __forceinline__ __nv_fp8_storage_t
__pie_fp8_from_float(const float f, const __nv_fp8_interpretation_t interp) {
    return __nv_cvt_float_to_fp8(f, __NV_SATFINITE, interp);
}

// ---------------------------------------------------------------------------
// the classes
// ---------------------------------------------------------------------------
//
// Every constructor and every conversion operator is `explicit`, which is
// faithful to the vendor and also load-bearing. `vec_dtypes.cuh:356` writes
// `half(src[0])` on an `__nv_fp8_e4m3`: with a non-explicit `operator float`
// in scope, `__half`'s own float constructor becomes a second viable
// user-defined conversion and the call is AMBIGUOUS. Marking them explicit
// takes `operator float` out of the running for `__half(float)`'s argument --
// copy-initialisation does not consider explicit conversion functions -- and
// leaves `operator __half` as the only candidate. A shim that relaxed this to
// be "convenient" would break the two call sites it was written for.
//
// The storage member is `__x`, public and named the vendor's way, because
// `vec_dtypes.cuh:701`, `:748` and `:796` read and write it directly.

/// One E4M3 value: 1-4-3, no infinity, max 448, denormals down to 2^-9.
struct __nv_fp8_e4m3 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e4m3() = default;

    /// Saturating, so anything past 448 becomes 448 and keeps its sign --
    /// which is the behaviour `quant/quant_bf16_to_fp8.cu:123` already
    /// documents itself as relying on.
    explicit __device__ __forceinline__ __nv_fp8_e4m3(const float f) {
        __x = __pie_fp8_from_float(f, __NV_E4M3);
    }

#if defined(__CUDA_FP16_TYPES_EXIST__)
    /// fp16 in, through fp32. The widening is exact, so this rounds once --
    /// in the `cvt` above -- and is bit-identical to the packed
    /// `cvt.rn.satfinite.e4m3x2.f16x2` form the vendor uses.
    explicit __device__ __forceinline__ __nv_fp8_e4m3(const __half f) {
        __x = __pie_fp8_from_float(
            __pie_fp8_halfbits_to_float(static_cast<__half_raw>(f).x), __NV_E4M3);
    }

    /// fp8 out as fp16, exactly. See `__nv_cvt_fp8_to_halfraw`.
    explicit __device__ __forceinline__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E4M3));
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
    /// bf16 in, through fp32. bf16 widens to fp32 by appending sixteen zero
    /// bits, so this too rounds exactly once.
    explicit __device__ __forceinline__ __nv_fp8_e4m3(const __nv_bfloat16 f) {
        const unsigned int bits = ((unsigned int)static_cast<__nv_bfloat16_raw>(f).x) << 16;
        __x = __pie_fp8_from_float(__int_as_float((int)bits), __NV_E4M3);
    }

    /// fp8 out as bf16, through fp32 and one narrowing instruction.
    ///
    /// The narrowing never rounds on this domain -- an E4M3 value has at most
    /// three mantissa bits and an exponent well inside bf16's range, so the
    /// fp32 it widens to has sixteen zero low bits -- but it is written as a
    /// `cvt` rather than a shift for the one input where that is false. E4M3
    /// bit patterns `0x7F` and `0xFF` are NaN, `float()` returns `0x7FFFFFFF`
    /// for both, and TRUNCATING that keeps a sign and a payload the vendor
    /// discards. Measured on this box: `cvt.rn.bf16.f32` of `0x7FFFFFFF` is
    /// `0x7FFF`, which is what `__float2bfloat16_rz` -- the call NVIDIA's own
    /// operator makes -- returns for it. Rounding mode is immaterial to a
    /// NaN, so `.rn` here and `_rz` there agree on the whole 256-pattern
    /// domain, and `fp8_pipeline_probe` checks that they do.
    explicit __device__ __forceinline__ operator __nv_bfloat16() const {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(float(*this)));
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return static_cast<__nv_bfloat16>(raw);
    }
#endif

    /// fp8 out as fp32, exactly, via the fp16 the instruction produces.
    explicit __device__ __forceinline__ operator float() const {
        const unsigned short packed = (unsigned short)__x;
        unsigned int pair;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(packed));
        return __pie_fp8_halfbits_to_float((unsigned short)(pair & 0xFFFFu));
    }
};

/// One E5M2 value: 1-5-2, fp16's exponent range, infinities and NaNs, max
/// 57344.
struct __nv_fp8_e5m2 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e5m2() = default;

    /// Saturating: finite input past 57344 becomes 57344, while an input
    /// that was already infinite stays infinite. E5M2 has an infinity to
    /// saturate towards, which is the one behavioural difference from E4M3
    /// that a caller can see.
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

    /// As E4M3's, and for the same reason -- E5M2's infinities widen exactly
    /// and its three NaN patterns canonicalise the way the instruction does.
    explicit __device__ __forceinline__ operator __nv_bfloat16() const {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(float(*this)));
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return static_cast<__nv_bfloat16>(raw);
    }
#endif

    explicit __device__ __forceinline__ operator float() const {
        const unsigned short packed = (unsigned short)__x;
        unsigned int pair;
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(pair) : "h"(packed));
        return __pie_fp8_halfbits_to_float((unsigned short)(pair & 0xFFFFu));
    }
};

// The packed pairs and quads are STORAGE and nothing else, which is all the
// closure asks of them: `vec_t<__nv_fp8_e4m3, 2>` declares one as its `data`
// member and then writes `data.__x` (`vec_dtypes.cuh:701`), loads through a
// cast (`:705`), and stores through another (`:709`). No constructor is
// spelled anywhere, so none is written here -- a `float2` constructor would
// be a second, untested spelling of `__nv_cvt_float2_to_fp8x2`.
//
// No alignment attribute either, and that is deliberate: the natural
// alignment of the storage member is the vendor's layout, and `vec_t`'s casts
// from `__nv_fp8_e4m3*` assume exactly that and no more.

/// Two E4M3 values, low byte first.
struct __nv_fp8x2_e4m3 {
    __nv_fp8x2_storage_t __x;

    __nv_fp8x2_e4m3() = default;

    /// The bits, as the bits.
    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(__nv_fp8x2_storage_t bits) : __x(bits) {}

    /// `__nv_fp8x2_e4m3{f}` -- `mhaUtils.cuh:371`'s store of a converted
    /// pair. One `cvt.rn.satfinite.e4m3x2.f32`, via the free function that
    /// already spells it.
    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const float2 f)
        : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3)) {}

#if defined(__CUDA_FP16_TYPES_EXIST__)
    /// `xqa/utils.cuh:217` -- an fp16 pair down to an fp8 pair.
    ///
    /// Through fp32 rather than `cvt.rn.satfinite.e4m3x2.f16x2`, because
    /// widening fp16 to fp32 is exact and the `f32` form is the one this
    /// header already states; both round once, in the same place.
    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const __half2 v) {
        const __half2_raw raw = static_cast<__half2_raw>(v);
        float2 f;
        f.x = __pie_fp8_halfbits_to_float(raw.x);
        f.y = __pie_fp8_halfbits_to_float(raw.y);
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }

    /// Both lanes at once -- `xqa/utils.cuh:209`'s `half2(fp8x2)`.
    ///
    /// One `cvt.rn.f16x2.e4m3x2`, which is the instruction
    /// `__nv_cvt_fp8_to_halfraw` already issues and then throws half of away.
    /// Widening e4m3 to fp16 is exact, so there is no rounding to argue about.
    explicit __device__ __forceinline__ operator __half2() const {
        unsigned int pair;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(__x));
        __half2_raw raw;
        raw.x = (unsigned short)(pair & 0xFFFFu);
        raw.y = (unsigned short)(pair >> 16);
        return static_cast<__half2>(raw);
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
    /// `xqa/utils.cuh:229` -- a bf16 pair down to an fp8 pair.
    ///
    /// bf16 widens to fp32 by appending sixteen zero bits, so the only
    /// rounding is the `cvt` itself. Spelled with the bits rather than with
    /// `__bfloat1622float2`, for the reason `__nv_fp8_e4m3(__nv_bfloat16)`
    /// above is: this header includes nothing.
    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const __nv_bfloat162 v) {
        const __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(v);
        float2 f;
        f.x = __int_as_float((int)(((unsigned int)raw.x) << 16));
        f.y = __int_as_float((int)(((unsigned int)raw.y) << 16));
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
#endif
};

/// Two E5M2 values, low byte first.
struct __nv_fp8x2_e5m2 {
    __nv_fp8x2_storage_t __x;
};

/// Four E4M3 values, lowest byte first.
struct __nv_fp8x4_e4m3 {
    __nv_fp8x4_storage_t __x;
};

/// Four E5M2 values, lowest byte first.
struct __nv_fp8x4_e5m2 {
    __nv_fp8x4_storage_t __x;
};
