//===-- cuda_fp4.h - the fp4 door, opened only as far as it is walked ----===//
//
// `__nv_fp4_e2m1`, its packed pair and quad, and the storage typedefs and
// interpretation enum around them. This is what `#include <cuda_fp4.h>`
// resolves to when the compiler is NVRTC and the include path is a header set
// carried in the binary rather than a directory on a disk.
//
// **It carries no conversions.** That is the whole design decision in this
// file, and the rest of this banner is the evidence for it.
//
// # Why a shim rather than NVIDIA's header
//
// The same reason as `cuda_fp8.h`, in one line: `examples/header_probe.rs`
// measured NVRTC 13.0 on this box against an empty header set and **0 of the
// closure's 31 external includes were answered**, `<cuda_fp4.h>` among them,
// with *"could not open source file ... (no directories in search list)"*.
// Vendoring NVIDIA's file is a redistribution decision; reading `$CUDA_HOME`
// at build time was tried and rejected in `.wiki/driver/new-horizon.md`
// §13.2, because it makes the build machine carry a toolkit, which is the one
// property this crate exists not to need.
//
// **No text from `cuda_fp4.h` or `cuda_fp4.hpp` is in this file.** Both were
// read on this machine as a cross-check on two facts a specification does not
// state -- that `__nv_fp4_storage_t` and `__nv_fp4x2_storage_t` are the same
// one-byte type, and that `__NV_E2M1` is the enumeration's only member -- and
// nothing was copied.
//
// # What the closure actually reaches
//
// `vec_dtypes.cuh:24` writes `#include <cuda_fp4.h>` unconditionally, so the
// directive must resolve on every compile. What is reached BEHIND it is far
// smaller than the header it names. Counted over the 28-file, 18,009-line
// FlashInfer attention closure and over the whole of `kernels-cuda/csrc/src`
// -- the archive crate's tree, whose `.cuh` half is this crate's `csrc/src`
// now:
//
// | name | closure | our csrc |
// |---|---|---|
// | `__nv_fp4_e2m1` (type, and `.__x`) | 45 | -- |
// | `__nv_fp4x2_storage_t` | 14 | -- |
// | `__nv_fp4x2_e2m1` (type) | 5 | -- |
// | `__nv_cvt_float2_to_fp4x2` | 0 | 0 |
// | `__nv_cvt_float_to_e8m0` | 0 | 0 |
// | any `__nv_fp4*` conversion, any `__nv_fp4_e2m1` constructor | 0 | 0 |
//
// Forty-four of the forty-five `__nv_fp4_e2m1` mentions sit behind
// `#if defined(FLASHINFER_ENABLE_FP4_E2M1) && CUDA_VERSION >= 12080` --
// `vec_dtypes.cuh:417-546` and `1174-1370`, the `vec_t<__nv_fp4_e2m1, N>`
// specialisations and the two `vec_cast` ones -- and we do not define that
// macro. The forty-fifth is `attention/prefill.cuh:55-60`:
//
//     template <> struct is_fp4_type<__nv_fp4x2_e2m1> : std::true_type {};
//
// guarded by `CUDA_VERSION >= 12080` ALONE, so on NVRTC 13.0 it is compiled
// every time. It needs the name to be a type. It needs nothing else.
//
// Where the guarded code does convert, it converts WITHOUT us: the two
// `vec_cast` specialisations reach for `cvt.rn.f16x2.e2m1x2` and
// `cvt.rn.bf16x2.e2m1x2` in inline asm on the raw bytes, never for a
// `__nv_cvt_*` call. FlashInfer already decided not to trust the vendor's
// C++ here. So would we.
//
// # The conversions this file REFUSES, and why refusing is the safe answer
//
// The brief named `__nv_cvt_float2_to_fp4x2` (6 uses tree-wide) and
// `__nv_cvt_float_to_e8m0` (5). Both were traced: every one of the eleven is
// in `comm/trtllm_*` or `norm/*` -- **outside the closure**, in files we do
// not compile. Three facts make implementing them anyway the wrong call:
//
// 1. **No parity reference exists on this box.** `cvt.rn.satfinite.e2m1x2.f32`
//    and `cvt.rn.satfinite.ue8m0x2.f32` are `sm_100`+ instructions. Measured
//    here, through nvcc and through NVRTC alike: *"Instruction 'cvt with
//    .e2m1x2' not supported on .target 'sm_89'"*, and the same sentence for
//    `.ue8m0x2` -- ptxas refuses the text, so a shim that emitted it would
//    not merely miscompute on an L40S, it would fail to build on one.
//    `examples/fp8_pipeline_probe.rs` gated fp8 on bit-parity against nvcc
//    over a million inputs; fp4 could be gated on nothing at all, then or
//    now. A conversion nobody can test, in a file nobody in our closure
//    calls, is two liabilities and no asset.
//
// 2. **The signature needs a type we do not carry.** NVIDIA's
//    `__nv_cvt_float2_to_fp4x2` takes an `enum cudaRoundMode`, which lives in
//    `driver_types.h` -- a header with no shim, not in the set, and not on
//    the list of doors this crate has chosen to close. Declaring the function
//    would mean inventing its parameter type, and an invented `cudaRoundMode`
//    that a real `driver_types.h` later contradicts is an ODR violation with
//    a two-week debugging tail.
//
// 3. **A software path would be a different function wearing the same name.**
//    E2M1 has three mantissa bits' worth of magnitudes -- {0, .5, 1, 1.5, 2,
//    3, 4, 6} -- and round-nearest-even on that ladder is easy to write and
//    easy to write *subtly differently* from silicon at the ties and at the
//    saturation edge. `cuda_fp8.h` refuses a software path for exactly this
//    reason and `#error`s below sm_89 instead.
//
// So they are ABSENT, and absence is the loud kind: a call to
// `__nv_cvt_float2_to_fp4x2` fails to compile with *"identifier is
// undefined"*, naming the function and the line. That is the pattern
// `cooperative_groups.h` set with `this_grid()` -- a name error at the call
// site beats a wrong number in an output tensor, because one is found by the
// compiler and the other by a customer.
//
// The same goes for the class constructors and conversion operators. NVIDIA's
// `__nv_fp4_e2m1` converts from `float`, `double`, `__half`, `__nv_bfloat16`
// and the integer types; ours converts from nothing. If a Blackwell path we
// never instantiate is one day switched on and reaches `__nv_fp4_e2m1(x)`,
// the build stops on a missing constructor -- which is the correct moment to
// decide what that conversion should do, on hardware that can be measured.
//
// # Includes nothing
//
// Not even `cuda_fp8.h`, which NVIDIA's version does include -- its
// `__nv_fp4_storage_t` is a typedef of `__nv_fp8_storage_t`. Ours restates
// `unsigned char` in one word instead. An include is a second resolution that
// has to succeed, and coupling the fp4 door to the fp8 one buys nothing when
// the shared content is the width of a byte. The two headers name disjoint
// types, so a translation unit that pulls in both -- `vec_dtypes.cuh` does --
// sees no redefinition.
//
//===----------------------------------------------------------------------===//

#pragma once

/// Storage for one E2M1 value: four live bits in the low nibble of a byte.
///
/// One value per byte is the SCALAR layout; it wastes the high nibble and
/// FlashInfer relies on that waste -- `vec_t<__nv_fp4_e2m1, 2>::load` reads a
/// `uint8_t` through a `__nv_fp4_e2m1*` and `fill` splices the same nibble
/// into both halves. `sizeof(__nv_fp4_e2m1) == 1` is therefore load-bearing,
/// not incidental.
typedef unsigned char __nv_fp4_storage_t;

/// Two E2M1 values packed into one byte: element 0 in bits [3:0], element 1
/// in bits [7:4]. `vec_dtypes.cuh:447` states that order and `:1184` builds a
/// byte by `(x << 4) | x`, which agrees.
typedef unsigned char __nv_fp4x2_storage_t;

/// Four E2M1 values in two bytes, low pair first.
typedef unsigned short __nv_fp4x4_storage_t;

/// Which fp4 encoding a storage word carries. One member, because one
/// encoding exists: two exponent bits, one mantissa bit, a sign, no infinity
/// and no NaN. The vendor's numbering starts at zero and so does ours -- the
/// value is ABI wherever an `__nv_cvt_*` call crosses between a shimmed and
/// an unshimmed translation unit, so it is worth matching even though nothing
/// in the closure passes one.
typedef enum __nv_fp4_interpretation_t {
    __NV_E2M1 = 0,
} __nv_fp4_interpretation_t;

/// One E2M1 value, alone in a byte.
///
/// A storage type and nothing more. `__x` is public and is the only member
/// FlashInfer touches -- `vec_dtypes.cuh:1184`, `:1212`, `:1242`, `:1271`,
/// `:1305` all read it to splat a nibble across a vector. No constructor from
/// a real type is declared; see the banner.
struct __nv_fp4_e2m1 {
    __nv_fp4_storage_t __x;
};

/// Two E2M1 values in one byte.
///
/// Reached in the closure only as a NAME: `attention/prefill.cuh:59`
/// specialises `is_fp4_type<__nv_fp4x2_e2m1>` on it, which requires the type
/// to exist and requires nothing of its contents. The `vec_cast`
/// specialisations that do read `__x` are behind `FLASHINFER_ENABLE_FP4_E2M1`
/// and convert with their own inline `cvt.rn.f16x2.e2m1x2`.
struct __nv_fp4x2_e2m1 {
    __nv_fp4x2_storage_t __x;
};

/// Four E2M1 values in two bytes.
///
/// Unreferenced by both trees. It is here because the vendor's header defines
/// it and a shim that answers `<cuda_fp4.h>` for two of the three widths
/// invites a confusing error from the third; two bytes of type is a cheaper
/// answer than that confusion.
struct __nv_fp4x4_e2m1 {
    __nv_fp4x4_storage_t __x;
};
