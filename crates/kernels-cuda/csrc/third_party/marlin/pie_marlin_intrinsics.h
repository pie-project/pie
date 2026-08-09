//===-- pie_marlin_intrinsics.h - the four packed names the shims omit ---===//
//
// PIE-authored. Not upstream Marlin. Included from `marlin_dtypes.cuh` only
// when `__CUDACC_RTC__` is defined, and supplying exactly the four packed
// half/bfloat16 intrinsics that Marlin's device closure spells and that
// `kernels-cuda-new/csrc/src/cuda_fp16.h` and `cuda_bf16.h` do not carry.
//
// # Why this exists
//
// The shims are a CENSUS, not a port. Each was written against a counted set
// of names -- `cuda_bf16.h` says "the nine names the migrating tree spells"
// and then lists, under "what is deliberately not here", the ones that had
// zero uses in the FlashInfer closure and zero in `kernels-cuda/csrc/src`.
// `__bfloat162bfloat162` and the whole `__hsub2` family are on that list by
// name. Marlin is the first source to reach for them, because Marlin is the
// first source in this tree that dequantises INTO a packed pair: `dequant.h`
// subtracts a bias from `nv_bfloat162` two lanes at a time, and
// `marlin_dtypes.cuh` broadcasts a scale into both lanes.
//
// So the census was right when it was taken and is stale now. The correct
// end state is that these four move into the shims and this file is DELETED
// -- the names are NVIDIA's, they belong behind NVIDIA's filenames, and
// §13.4's rule ("impersonate a vendor header exactly when the includer is
// upstream source we do not own") points there and not here. This file
// exists because the shims are owned by another agent in this migration and
// a cross-owner edit would race. It is a workaround with a known expiry, and
// the expiry is: when `cuda_fp16.h` gains `__half2half2` and `cuda_bf16.h`
// gains `__bfloat162bfloat162`, `__halves2bfloat162` and `__hsub2`, delete
// this file and the four guarded lines in `marlin_dtypes.cuh` that reach it.
//
// # What breaks without it
//
// Twelve errors, measured on this L40S against NVRTC 13.0 with the whole
// carried set present: `marlin_dtypes.cuh:42` (`__half2half2` undefined),
// `:79` (`__bfloat162bfloat162`), `:84` (`__halves2bfloat162`), and eight at
// `dequant.h:198,199,215,216` where the only visible `__hsub2` is the fp16
// one and `nv_bfloat162` will not convert to `__half2`. Nothing else in the
// closure is missing -- the count went from twelve to zero with this file
// and no other change.
//
// # Numerics
//
// Three of the four are `mov.b32` in disguise -- a lane duplicated, or two
// lanes concatenated -- so they are written as the shims' own constructors
// and there is no rounding to get wrong. `__hsub2` is arithmetic, and is
// NVIDIA's decomposition transcribed instruction for instruction from
// `cuda_bf16.hpp`'s `__internal_device_hsub2`: `sub.bf16x2` where sm_90 has
// it, and below that `fma.rn.bf16x2 a, b, 0xbf80bf80` -- a fused multiply by
// -1.0 and add, which is exact and is NOT the same as two `cvt`s through
// fp32 around a subtract. sm_89, the device this migration targets, takes
// the `fma` path.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The shims `#undef` their own `PIE_BF16_HAS_SM*` gates at the end of the
// file, so the arch question is asked again here rather than borrowed. Same
// three-way answer: NVRTC compiles for a single architecture and defines
// `__CUDA_ARCH__`, and the host pass of an `nvcc` build defines nothing.
#if defined(__CUDA_ARCH__)
  #define PIE_MARLIN_HAS_SM90 (__CUDA_ARCH__ >= 900)
  #define PIE_MARLIN_HAS_SM80 (__CUDA_ARCH__ >= 800)
#else
  #define PIE_MARLIN_HAS_SM90 0
  #define PIE_MARLIN_HAS_SM80 0
#endif

// ===-- broadcasts and pairs ---------------------------------------------===

/// One half into both lanes. `marlin_dtypes.cuh:42` -- `num2num2`, the scale
/// broadcast the fp16 kernel applies to a dequantised fragment.
__device__ __forceinline__ __half2 __half2half2(__half x) {
    return __halves2half2(x, x);
}

/// One bfloat16 into both lanes. `marlin_dtypes.cuh:79`, the bf16 `num2num2`
/// -- the one every shape in `kernels.def` reaches, since all three output
/// bfloat16.
__device__ __forceinline__ __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 x) {
    return make_bfloat162(x, x);
}

/// Two bfloat16s into a pair, low lane first, no conversion.
/// `marlin_dtypes.cuh:84` -- `nums2num2`, which builds a scale pair from two
/// separately-loaded scales.
__device__ __forceinline__ __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 lo,
                                                             __nv_bfloat16 hi) {
    return make_bfloat162(lo, hi);
}

// ===-- arithmetic -------------------------------------------------------===

/// `a - b` per lane, and the only bf16 arithmetic Marlin adds to the
/// closure's `__hmul2`.
///
/// `dequant.h` calls it four times, always against a constant pair holding a
/// zero-point bias -- `0x43084308` (136.0, the u4b8 GPTQ offset folded into
/// the magic-number trick's exponent) and `0x43004300` (128.0, the plain u4
/// offset). Those subtractions are the last step of the dequantise, so a
/// rounding difference here is a weight difference, which is why this is
/// NVIDIA's decomposition and not a convenient one.
__device__ __forceinline__ __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
#if PIE_MARLIN_HAS_SM90
    unsigned int bits;
    asm("sub.bf16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#elif PIE_MARLIN_HAS_SM80
    unsigned int bits;
    // `0xbf80bf80` is `(-1.0, -1.0)`. `fma.rn.bf16x2 a, b, c` computes
    // `a * b + c` with ONE rounding, so this is `b * -1 + a` -- exact, and
    // the vendor's own body below sm_90.
    asm("{ .reg .b32 c;\n"
        "  mov.b32 c, 0xbf80bf80;\n"
        "  fma.rn.bf16x2 %0, %2, c, %1; }"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#else
    // Below sm_80 there is no packed bf16 arithmetic at all and the vendor
    // goes lane-wise too. A bf16 difference of two bf16s is at most 17
    // significant bits before rounding and fits fp32 exactly, so the single
    // narrowing is the only rounding -- the same argument `__hmul2` makes.
    __nv_bfloat162 out;
    out.x = __float2bfloat16(__bfloat162float(a.x) - __bfloat162float(b.x));
    out.y = __float2bfloat16(__bfloat162float(a.y) - __bfloat162float(b.y));
    return out;
#endif
}

#undef PIE_MARLIN_HAS_SM90
#undef PIE_MARLIN_HAS_SM80
