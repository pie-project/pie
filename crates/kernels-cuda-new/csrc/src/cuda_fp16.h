//===-- cuda_fp16.h - the fp16 door, answered from inside the binary -----===//
//
// The eighteen names the migrating tree spells out of NVIDIA's
// `cuda_fp16.h`, written directly against the `f16` and `f16x2` instructions
// so that `#include <cuda_fp16.h>` resolves without a toolkit on the machine.
//
// # Why this exists
//
// NVRTC 13.0 was probed on this L40S with an EMPTY header set over the 31
// external includes of the FlashInfer attention closure -- 28 files, 17,981
// lines, reachable from `attention/{cascade,decode,mask,mla,prefill,scheduler,
// state,variants,default_*_params}.cuh` and `{fastdiv,layout,page,pos_enc,
// utils}.cuh`. **0 of 31 answered.** `cuda_fp16.h` was among them:
// *"catastrophic error: could not open source file 'cuda_fp16.h' (no
// directories in search list)"*. NVRTC before 13.3 bundles none of the device
// headers, and `src/source.rs` states the rule the crate is built on: an
// `#include` resolves against a header set carried in the binary, or it does
// not resolve at all.
//
// Reading the toolkit's copy out of `$CUDA_HOME` at build time was tried and
// rejected -- `.wiki/driver/new-horizon.md` §13.2 -- because it puts a CUDA
// installation back on the BUILD machine, which is the one property this
// crate exists to not need, and it pins the embedded device ABI to whatever
// the build box happened to have. So the header is replaced, not found.
//
// # Why it wears NVIDIA's filename
//
// §13.4 states the rule and this file is the case it was written for: a shim
// is named after a vendor header exactly when its includer is upstream source
// we do not own. NVRTC matches `includeNames[]` against the LITERAL string in
// the directive, so a file carried as `cuda_fp16.h` answers FlashInfer's
// `#include <cuda_fp16.h>` with FlashInfer left byte-identical to upstream --
// the resolution becomes ours without the source becoming ours. The sibling
// `pie_half2.cuh` is the other half of the same rule: its includers are six
// of our own `.cu` files, which can say any name, so it says an honest one.
// Impersonation is a name collision held in reserve and spent only here.
//
// # What replacing it costs, measured
//
// Every name was counted twice, comments stripped: once over the 28-file
// closure above, once over the 84 `.cu`/`.cuh` files of
// `kernels-cuda/csrc/src`. `closure / ours`:
//
//   half            134 / -      __hmul2         25 / -     __half2float    1 / 9
//   half2            47 / -      make_half2       8 / -     __float2half    1 / 2
//   __half2           2 / 30     __hmax2          3 / -     __half22float2  2 / 6
//   __half            7 / 23     __hmax           2 / -     __float2half2_rn 2 / 6
//   __shfl_xor_sync   1 / -      __half_as_ushort 2 / -     __hfma2         - / 6
//   __float22half2_rn 1 / -      __ushort_as_half 2 / -     __hsub2         - / 1
//                                                           __floats2half2_rn - / 1
//
// Four `cvt` spellings, four arithmetic instructions, two bit-casts, a
// shuffle and a pair constructor. Everything else in the vendor's four
// thousand lines -- the `__half` class and its full operator set, the host
// software emulation, the `__half_raw` layer, the two hundred functions
// nothing here calls -- is not the part the kernels need.
//
// The bare `half` spelling is FlashInfer's; our tree writes `__half`. The 133
// bare hits in `kernels-cuda/csrc/src` are a LOCAL VARIABLE (`const int half
// = rope_dim / 2;` in `dsa_indexer.cu`, `mla_paged.cu`, `dequant_fp4.cu`),
// which shadows the alias inside those functions. That is legal, it is
// harmless -- those functions never name the type -- and it is recorded here
// so a reader who greps does not double-count it as a use.
//
// # Two operators are needed, and they were compiled for, not grepped
//
// An operator does not appear in a grep of the call sites. The way to count
// them is to compile with the vendor's own kill switches, so the compiler
// enumerates them: every FlashInfer TU in the AOT build was rebuilt with
// `-D__CUDA_NO_HALF_OPERATORS__`, `-D__CUDA_NO_HALF2_OPERATORS__` and
// `-D__CUDA_NO_HALF_CONVERSIONS__` in turn, one macro at a time, under
// `-Xcudafe --error_limit=300`. The whole answer:
//
//   operator*(half, half)      operator*(half2, half2)
//   operator-(half, half)      operator-(half2, half2)
//   (float) on a half          -- see the divergence below
//
// The same six macros over our own 13 migrating `.cu` files produced ZERO
// errors: our tree uses no fp16 operators and no implicit conversions at all.
//
// # `__half` is the PRELUDE's f16; `__half2` is a pair of them
//
// `pie_device.cuh` already defines `pie_cuda_driver::kernels::device::f16`, a
// struct wrapping an `unsigned short`, and says why it is a struct and not a
// typedef: as typedefs `f16` and `bf16` would be ONE type and a table row
// that swapped them would typecheck. A second fp16 declared here would undo
// that -- `device::f16*` and a shim's own `__half*` are a pointer conversion
// C++ refuses, at exactly the boundary where a dequant kernel hands its
// output to something written against the prelude. So this file declares no
// fp16. It makes `__half` a NAME for the prelude's, the way `pie_mma.cuh`
// makes `__nv_bfloat16` a name for `device::bf16` and for the same reason,
// and a duplicate identical typedef is legal so both files may say it.
//
// That is why the prelude is included, and it is the only include here. A
// header in the set that drags in a diamond its includer did not ask for is
// the hazard the set's rule exists to avoid, so the cost is worth naming:
// `<cuda_fp16.h>` now transitively defines `device::` for FlashInfer. It is
// bounded -- `pie_device.cuh` is `#pragma once`, includes nothing itself, and
// declares only inside `pie_cuda_driver::kernels::device` plus the two
// aliases the shims add at global scope -- and it is unavoidable while the
// one true fp16 lives there. The alternative, a private copy of the type,
// is the pointer conversion above.
//
// `__half2` is genuinely new. The prelude has `bf16x2` and stops, because
// every kernel migrated so far widens to fp32, computes, and narrows. These
// call sites do not: FlashInfer's softmax keeps `m` and `d` in `half2`, and
// `dequant_wna16.cu` keeps its whole inner loop in f16x2 so one instruction
// is two MACs. `x` is the LOW half of the 32-bit register and `y` the high,
// field for field with NVIDIA's struct, because both trees `reinterpret_cast`
// a `uint32_t` they assembled themselves onto it. Reversed, it would swap
// every pair -- a wrong answer, not a compile error.
//
// # The divergence that was known, and is now closed
//
// `float(x)` on a `__half` needs a MEMBER conversion operator, and a member
// cannot be added to a type this header does not own. Three sites in the
// closure use one, measured by the `__CUDA_NO_HALF_CONVERSIONS__` build
// above: `vec_dtypes.cuh:159` (`(float)src[0]` in `vec_cast<float, half>`),
// `vec_dtypes.cuh:553` (the same for `nv_bfloat16`) and
// `prefill.cuh:1523` (`float(m_prev[j] * sm_scale.x - ...)`).
//
// Declaring our own `__half` class with the operator would have fixed two of
// the three and cost the type identity above -- and would NOT have fixed the
// third, because `__nv_bfloat16` is pinned to `device::bf16` by `pie_mma.cuh`
// already. So this file did not try. It recorded the divergence instead, and
// said where the fix belonged: `pie_device.cuh`, where the type is defined,
// one `__device__ operator float() const` on each struct closing all three at
// once. The measurement that made the case was run in memory rather than
// claimed -- all 28 closure files compiled, but
// `BatchDecodeWithPagedKVCacheKernel` refused to INSTANTIATE with one error
// and one only, `vec_dtypes.cuh(159): no suitable conversion function from
// "const half" to "float" exists`; injecting the two members produced a cubin.
//
// That patch has since landed. `pie_device.cuh:88` and `:96` now declare
// `explicit __device__ operator float() const` on `f16` and on `bf16`, and
// `explicit` is the right call -- all three call sites are explicit casts, so
// the conversion they need is reachable while the implicit narrowing that
// would make `h + 1.0f` silently compile is not. Re-measured against the
// carried set with no include path on disk: 28 of 28 closure files accepted,
// and `BatchDecodeWithPagedKVCacheKernel<half>` instantiates into a
// 59,184-byte sm_89 cubin.
//
// `examples/halftype_parity.rs` still compiles `(float)h` under NVRTC every
// run and prints the result, because a divergence that closed once can open
// again -- the probe reports which of the two states it found rather than
// asserting the good one.
//
// # This is not NVIDIA's header
//
// `cuda_fp16.hpp` on this machine was READ, to check the answer -- which
// rounding mode each `cvt` carries, which operand of `cvt.rn.f16x2.f32`
// lands in the high half, that the vendor gates the packed converts at sm_80
// -- and nothing was copied out of it. That is deliberate: copying vendor
// header text into this repository is a REDISTRIBUTION with a `NOTICE` entry
// behind it, and avoiding exactly that is why these headers are being
// replaced instead of vendored. A shim that got here by copy-paste would
// have made the whole exercise pointless. The semantics below come from the
// PTX ISA -- §9.7.4 *Half Precision Floating Point Instructions* and §9.7.9
// `cvt` -- which is a specification and not an implementation.
//
// # Architecture, and why the fallbacks are exact
//
// The instructions used here are not all available everywhere, and the ISA's
// own Target ISA Notes decide the gates: `mul/sub/fma.f16{x2}` require sm_53;
// `max.f16{x2}` requires sm_80; a `.f16x2` DESTINATION for `cvt` requires
// sm_80. The crate targets sm_89 today, but a shim that silently miscompiles
// on sm_75 is worse than one that refuses, so every gated site has a fallback
// written out below it.
//
// The fallbacks are not approximations, but one of them had to be rewritten
// to earn that. A product of two fp16 values has at most 2p = 22 bits and is
// EXACT in fp32, so a fallback that multiplies in fp32 and narrows once
// rounds once -- Figueroa's condition on innocuous double rounding wants an
// intermediate of at least 2p + 2 bits and fp32's 24 meets it for p = 11.
// That argument does NOT extend to `a * b + c`, and `halftype_parity` found
// the counterexample rather than the reasoning: an exact fp16 fused
// multiply-add can need 84 significant bits, and on `a = 0x19ff`,
// `b = 0x1a01`, `c = 0x1a03` the fp32 intermediate lands exactly on an fp16
// midpoint that the exact result sits just below, so the second rounding goes
// the wrong way. `__hfma2`'s fallback therefore recovers the fp32 add's error
// with a 2Sum and rounds the intermediate to ODD, which is innocuous for any
// target two or more bits narrower. fp16's whole subnormal range is normal in
// fp32, so nothing underflows on the way through. None of this is left as an
// argument: define `PIE_HALFTYPE_FORCE_PORTABLE` and every gate below takes
// its fallback, which is how `halftype_parity` runs the sm_75 path on an
// sm_89 device and compares it against the same nvcc reference -- and it is
// how the `__hfma2` defect above was a red row in a table rather than a wrong
// attention score on a Turing card.
//
// # The check that makes this trustworthy
//
// A conversion that rounds the wrong way still compiles, so compiling proves
// nothing and a tolerance would hide exactly the defect worth catching.
// `examples/halftype_parity.rs` is the gate: it runs each function below
// under NVRTC against these headers and the SAME function under `nvcc`
// against the real `<cuda_fp16.h>`, on the same inputs on the same device,
// and compares BIT PATTERNS. It sweeps all 65,536 fp16 patterns, all 65,536
// bf16 patterns and all 65,536 bf16 TIES, every fp16 tie and its two fp32
// neighbours, 1.2e6 pseudo-random floats plus every special (+-0, +-inf,
// quiet and signalling NaN, largest finite, smallest normal, subnormals on
// both sides), and a 216x216 exponent grid for the packed ops. As of the last
// run: 43 functions, 39,847,842 comparisons, zero differing bits, and the
// same again with every fallback forced on. It then corrupts one kernel --
// the two source operands of `cvt.rn.f16x2.f32`, swapped -- and requires the
// comparison to CATCH it, because a green table proves nothing unless a red
// one was reachable.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

// Which instructions this compilation may use. Written once, so a reader
// counts the gates in one place instead of hunting `#if`s, and undefined at
// the end of the file so nothing leaks into the translation unit.
//
// `PIE_HALFTYPE_FORCE_PORTABLE` is the probe's handle on the pre-Ampere
// paths. Without it they ship untested, which for numeric code is the same as
// shipping them wrong.
#if defined(PIE_HALFTYPE_FORCE_PORTABLE)
#define PIE_FP16_HAS_SM80 0
#define PIE_FP16_HAS_SM53 0
#elif defined(__CUDA_ARCH__)
#define PIE_FP16_HAS_SM80 (__CUDA_ARCH__ >= 800)
#define PIE_FP16_HAS_SM53 (__CUDA_ARCH__ >= 530)
#else
#define PIE_FP16_HAS_SM80 0
#define PIE_FP16_HAS_SM53 0
#endif

// ===-- the types --------------------------------------------------------===

/// The spelling FlashInfer uses 7 times and our tree 23, made a name for the
/// prelude's type rather than a type of its own. There is one fp16 in this
/// translation unit and it is `device::f16`.
using __half = ::pie_cuda_driver::kernels::device::f16;

/// FlashInfer's own spelling, 134 sites. Shadowed by a local `int half` in
/// three of our `.cu` files, which is legal and does not matter to them.
using half = __half;

/// A pair, `x` in the LOW 16 bits of the 32-bit register and `y` in the high.
/// `__align__(4)` because every packed instruction below takes it as one
/// `.b32` operand and both trees `reinterpret_cast` a `uint32_t` onto it.
struct __align__(4) __half2 {
    __half x;
    __half y;
};

using half2 = __half2;

// ===-- the storage structs, and the macro that announces them -----------===
//
// `__half_raw` is the vendor's way to get at the bits without going through a
// constructor, and this file's first version left it out on the strength of a
// census: zero uses in the FlashInfer closure, zero in
// `kernels-cuda/csrc/src`. The census was right and the conclusion was wrong,
// because the CONSUMER IS NOT IN EITHER TREE. `csrc/src/cuda_fp8.h` -- a
// sibling shim in the same header set -- returns `__half_raw` from
// `__nv_cvt_fp8_to_halfraw` and takes `static_cast<__half_raw>(f).x` in both
// fp8 class constructors. No grep of the two SOURCE trees could see it.
//
// It failed silently, which is the part worth recording. `cuda_fp8.h` guards
// its half interop on `__CUDA_FP16_TYPES_EXIST__` -- correctly, mirroring
// what NVIDIA's headers do, so that a TU which includes it alone gets a name
// error rather than a mystery. This file did not define that macro, so the
// guard was false, so the entire fp8-to-half interop COMPILED OUT and the
// only symptom was `identifier "__nv_cvt_fp8_to_halfraw" is undefined` at a
// call site, 9 of which live in `kernels-cuda/csrc/src`. A header set is
// composed, not merely collected, and composition has to be compiled rather
// than inferred.
//
// The layout is the vendor's, because that is the whole point of the type:
// `__align__(2)` with one `unsigned short x` for the scalar, `__align__(4)`
// with `x` and `y` for the pair.
//
// The CONVERSIONS sit here rather than on `__half`, and that is a real
// difference from NVIDIA worth stating. The vendor puts `__half(__half_raw)`
// and `operator __half_raw()` on its `__half` class; ours is
// `device::bf16`'s neighbour `device::f16`, a plain aggregate defined in a
// crate this file may not edit, so a member cannot be added to it. Putting
// both directions on the raw struct instead makes `static_cast<__half_raw>(h)`
// and `static_cast<__half>(raw)` mean exactly what they mean with the real
// header. Neither is `explicit`: `tests/prelude_parity.rs` writes
// `__half back = __half_raw(...)`, which is COPY-initialisation, and
// copy-initialisation does not consider explicit conversion functions.
//
// Both keep a defaulted default constructor, because `cuda_fp8.h:293` writes
// `__half_raw res; res.x = ...;`. Nothing in either tree aggregate-initialises
// one, which is the only thing a user-provided constructor would have cost.
struct __align__(2) __half_raw {
    unsigned short x;

    __half_raw() = default;
    __device__ __forceinline__ __half_raw(const __half h) : x(h.raw) {}
    __device__ __forceinline__ operator __half() const {
        __half out;
        out.raw = x;
        return out;
    }
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

/// The announcement, at the point the four types exist, which is where
/// NVIDIA's header makes it.
///
/// This is not decoration. It is the contract every other header in the set
/// tests to decide whether half interop is available, and defining it is a
/// PROMISE that `__half`, `__half2`, `__half_raw` and `__half2_raw` are all
/// in scope below this line. Do not move it above them.
#define __CUDA_FP16_TYPES_EXIST__

// ===-- the vendor's other spellings -------------------------------------===
//
// NVIDIA's `cuda_fp16.h` exports its type under EIGHT names, and a header
// that answers to that filename answers to the names too. This block is a
// compatibility surface and nothing else -- eight `using` lines, no
// arithmetic, no cost at any architecture -- and it is here because leaving
// it out cost seven files.
//
// The measurement: `fi-vendor` compiled the 28-file FlashInfer closure twice,
// once against NVIDIA's real device headers on disk and once against this
// header set. The first said 28 of 28 and instantiated
// `BatchDecodeWithPagedKVCacheKernel` into a 125,760-byte cubin. The second
// said 21 of 28, and all seven refusals were the same line --
// `page.cuh(232): error: identifier "nv_half" is undefined`. One missing
// alias, one file that uses it, seven files that include that file. An alias
// costs nothing and its absence cost a quarter of the closure.
//
// So the whole surface goes in, not just the one that was caught. `nv_half`
// is the only one anything uses today; the other seven have zero uses across
// both trees and are here so that the next upstream file to reach for one
// finds it, instead of being found the way this one was. That is the same
// lesson the raw structs above arrived by, applied before it costs anything.
using nv_half = __half;
using nv_half2 = __half2;
using __nv_half = __half;
using __nv_half2 = __half2;
using __nv_half_raw = __half_raw;
using __nv_half2_raw = __half2_raw;

namespace pie_fp16_detail {

/// The 32-bit view a `.f16x2` instruction wants, built by shifts rather than
/// by a type pun: the lane order is then a statement in the code instead of a
/// property of the machine's endianness that a reader has to already know.
__device__ __forceinline__ unsigned int pack(__half2 v) {
    return (static_cast<unsigned int>(v.y.raw) << 16) | static_cast<unsigned int>(v.x.raw);
}

/// The inverse, and the only place the low half is claimed to be `x`.
__device__ __forceinline__ __half2 unpack(unsigned int bits) {
    __half2 out;
    out.x.raw = static_cast<unsigned short>(bits & 0xffffu);
    out.y.raw = static_cast<unsigned short>(bits >> 16);
    return out;
}

}  // namespace pie_fp16_detail

// ===-- conversions ------------------------------------------------------===

/// `f16 -> f32`, exact -- every fp16 value is an fp32 value.
///
/// `cvt.f32.f16` and not the prelude's `f16_to_f32`, which is the same
/// function written out in shifts and does NOT agree with it:
/// `tests/prelude_parity.rs` sweeps all 65,536 patterns through both and
/// they differ on 3,070 -- 2,046 NaN payloads, and every negative subnormal
/// plus `-0.0`, which the prelude returns positive because its subnormal
/// branch adds to `-0.0`. A vendor spelling that quietly did something else
/// would be worse than either, so this one means what `cuda_fp16.h` means.
__device__ __forceinline__ float __half2float(__half h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h.raw));
    return f;
}

/// `f32 -> f16`, round-to-nearest-even.
///
/// Not the prelude's `f32_to_f16` either, and for a sharper reason: that one
/// FLUSHES subnormal results to zero. `tests/prelude_parity.rs` sweeps all
/// 2^32 float patterns through both and they differ on 201,326,588 --
/// 184,549,374 whose fp16 result is a non-zero subnormal, and 16,777,214 NaNs
/// whose payload the instruction quiets. Attention's `sm_scale` products land
/// in that range routinely.
__device__ __forceinline__ __half __float2half(float f) {
    __half out;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out.raw) : "f"(f));
    return out;
}

/// The same conversion under the name that says the rounding mode out loud.
/// One function, because at every architecture that has ever run this crate
/// both are `cvt.rn.f16.f32` -- the vendor's two spellings differ only in a
/// host-side software path this file does not have.
__device__ __forceinline__ __half __float2half_rn(float f) {
    return __float2half(f);
}

/// One float into both lanes. `mov.b32 d, {t, t}` needs no sm_80: the packed
/// destination the ISA gates is `cvt`'s, and this one converts once.
__device__ __forceinline__ __half2 __float2half2_rn(float f) {
    unsigned int bits;
    asm("{ .reg .f16 t;\n"
        "  cvt.rn.f16.f32 t, %1;\n"
        "  mov.b32 %0, {t, t}; }"
        : "=r"(bits)
        : "f"(f));
    return pie_fp16_detail::unpack(bits);
}

/// Two floats, `lo` into `x` and `hi` into `y`.
///
/// The operand order is the trap. `cvt.rn.f16x2.f32 d, a, b` puts `a` in
/// d[31:16] -- the FIRST source lands in the HIGH half -- so `hi` is named
/// first and `lo` second. Backwards, it swaps every pair silently.
__device__ __forceinline__ __half2 __floats2half2_rn(float lo, float hi) {
    unsigned int bits;
#if PIE_FP16_HAS_SM80
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
#else
    // Two scalar converts and a pack. Each lane is rounded once by the same
    // instruction the packed form uses per lane, so this is the same answer;
    // `halftype_parity` runs it on this device under
    // `PIE_HALFTYPE_FORCE_PORTABLE` rather than leaving that an argument.
    asm("{ .reg .f16 l, h;\n"
        "  cvt.rn.f16.f32 l, %1;\n"
        "  cvt.rn.f16.f32 h, %2;\n"
        "  mov.b32 %0, {l, h}; }"
        : "=r"(bits)
        : "f"(lo), "f"(hi));
#endif
    return pie_fp16_detail::unpack(bits);
}

/// A `float2`'s `x` into `x` and `y` into `y`, which is the only reading of
/// the name that survives `vec_dtypes.cuh:178` writing a vector element-wise.
__device__ __forceinline__ __half2 __float22half2_rn(float2 f) {
    return __floats2half2_rn(f.x, f.y);
}

/// Two halves into a pair, no conversion. Pure `mov.b32`, so no gate.
__device__ __forceinline__ __half2 __halves2half2(__half lo, __half hi) {
    __half2 out;
    out.x = lo;
    out.y = hi;
    return out;
}

/// Both lanes widened. Two `cvt.f32.f16`s, which is what the vendor emits on
/// a device and is exact in both lanes.
__device__ __forceinline__ float2 __half22float2(__half2 v) {
    float2 out;
    out.x = __half2float(v.x);
    out.y = __half2float(v.y);
    return out;
}

/// The pair constructor, spelled as the vector-type constructors are.
/// `vec_t<half, N>::fill` is 8 of the closure's uses of it.
__device__ __forceinline__ __half2 make_half2(__half x, __half y) {
    __half2 out;
    out.x = x;
    out.y = y;
    return out;
}

// ===-- bit casts --------------------------------------------------------===

/// The 16 bits, unchanged. `math.cuh` uses this pair to hand a half to
/// `ex2.approx.f16` and `tanh.approx.f16` through an `"h"` constraint, so the
/// two must be exact inverses and nothing else.
__device__ __forceinline__ unsigned short __half_as_ushort(__half h) {
    return h.raw;
}

__device__ __forceinline__ __half __ushort_as_half(unsigned short bits) {
    __half out;
    out.raw = bits;
    return out;
}

// ===-- arithmetic -------------------------------------------------------===

/// `a * b`, one rounding. The body of `operator*`, which is how the closure
/// actually spells it.
__device__ __forceinline__ __half __hmul(__half a, __half b) {
    __half out;
#if PIE_FP16_HAS_SM53
    asm("mul.rn.f16 %0, %1, %2;" : "=h"(out.raw) : "h"(a.raw), "h"(b.raw));
#else
    // The product of two 11-bit significands is 22 bits and therefore EXACT
    // in fp32, so the narrowing below is the only rounding and this is the
    // same answer as `mul.rn.f16`. fp16's subnormals are normal in fp32, so
    // the exactness survives the bottom of the range as well.
    out = __float2half(__half2float(a) * __half2float(b));
#endif
    return out;
}

/// `a - b`, one rounding.
__device__ __forceinline__ __half __hsub(__half a, __half b) {
    __half out;
#if PIE_FP16_HAS_SM53
    asm("sub.rn.f16 %0, %1, %2;" : "=h"(out.raw) : "h"(a.raw), "h"(b.raw));
#else
    // A difference is not exact in fp32 the way a product is -- `a - b` on
    // fp16 operands 39 octaves apart spans 50 bits -- so this leans on the
    // other half of the argument in the banner: fp32 carries 24 bits, 2p + 2
    // is 24 for p = 11, and Figueroa's condition makes the double rounding
    // innocuous for a SINGLE subtraction. That is the exact bound `__hfma2`
    // fails, and the difference is that an FMA rounds a product and a sum
    // together; here there is one operation and one exact error term, and
    // 575,232 pairs on the portable path agree with `sub.rn.f16` bit for bit.
    out = __float2half(__half2float(a) - __half2float(b));
#endif
    return out;
}

/// The larger, with the ISA's tie-breaks: a NaN operand returns the OTHER
/// one, two NaNs return a canonical NaN, and `+0.0 > -0.0`. `max.f16` needs
/// sm_80, which is why the fallback restates all three.
__device__ __forceinline__ __half __hmax(__half a, __half b) {
    __half out;
#if PIE_FP16_HAS_SM80
    asm("max.f16 %0, %1, %2;" : "=h"(out.raw) : "h"(a.raw), "h"(b.raw));
#else
    const bool a_nan = (a.raw & 0x7fffu) > 0x7c00u;
    const bool b_nan = (b.raw & 0x7fffu) > 0x7c00u;
    if (a_nan && b_nan) {
        // The canonical fp16 NaN `max.f16` returns for two NaNs, READ OFF
        // this device by `halftype_parity` rather than assumed: the ISA says
        // "canonical NaN" and leaves the payload to the implementation, and
        // the obvious guess -- 0x7e00, the quiet bit and nothing else -- is
        // wrong. The instruction returns 0x7fff, every payload bit set, for
        // every one of the 4,194,304 NaN pairs the probe crosses, positive
        // even when both inputs are negative NaNs.
        out.raw = 0x7fffu;
    } else if (a_nan) {
        out = b;
    } else if (b_nan) {
        out = a;
    } else if ((a.raw | b.raw) == 0x8000u && (a.raw & b.raw) == 0x0000u) {
        // +0 and -0 compare equal, and the ISA orders them anyway. The mask
        // says "one of the two is 0x8000 and the other 0x0000".
        out.raw = 0x0000u;
    } else {
        out = __half2float(a) > __half2float(b) ? a : b;
    }
#endif
    return out;
}

/// `a * b` per lane, one instruction. 25 of the closure's call sites are this
/// one, on `packed2_` -- `half2` or `nv_bfloat162` by template argument.
__device__ __forceinline__ __half2 __hmul2(__half2 a, __half2 b) {
#if PIE_FP16_HAS_SM53
    unsigned int bits;
    asm("mul.rn.f16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_fp16_detail::pack(a)), "r"(pie_fp16_detail::pack(b)));
    return pie_fp16_detail::unpack(bits);
#else
    return make_half2(__hmul(a.x, b.x), __hmul(a.y, b.y));
#endif
}

/// `a - b` per lane.
__device__ __forceinline__ __half2 __hsub2(__half2 a, __half2 b) {
#if PIE_FP16_HAS_SM53
    unsigned int bits;
    asm("sub.rn.f16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_fp16_detail::pack(a)), "r"(pie_fp16_detail::pack(b)));
    return pie_fp16_detail::unpack(bits);
#else
    return make_half2(__hsub(a.x, b.x), __hsub(a.y, b.y));
#endif
}

namespace pie_fp16_detail {

/// One lane of `fma.rn.f16x2`, exactly, with no fp16 multiply-add available.
///
/// The obvious fallback -- multiply and add in fp32, narrow once -- is WRONG,
/// and `halftype_parity` found the counterexample before this file shipped:
/// `a = 0x19ff`, `b = 0x1a01`, `c = 0x1a03`. Their exact `a * b + c` is
/// 1543.49999809... in units of the result's ulp, which `fma.rn.f16x2` rounds
/// down to 1543 (0x1a07); rounding it to fp32 first lands EXACTLY on 1543.5,
/// and round-half-to-even then goes UP to 1544 (0x1a08). The 2p + 2 rule that
/// makes a narrowed fp32 product innocuous does not extend to a sum -- an
/// exact `a * b + c` on 11-bit inputs can need 84 significant bits, and fp32
/// has 24.
///
/// So the sum is made exact before it is narrowed. `fa * fb` IS exact in fp32
/// (an 11-bit by 11-bit product is 22 bits, and 2^-48 is far inside fp32's
/// range); Knuth's 2Sum recovers the fp32 add's discarded error exactly; and
/// nudging the sum to an ODD significand whenever that error is non-zero
/// makes the second rounding innocuous for any target at least two bits
/// narrower -- round-to-odd, which fp16 clears by eleven. Six extra flops on
/// a path no device newer than 2016 takes, in exchange for not being wrong.
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
    // Finite, non-zero, and even -- the three conditions under which there is
    // a neighbour to step to and a reason to. Both tests below are false for
    // a NaN `err`, which is how an infinite or NaN sum passes through
    // untouched instead of being stepped into a different NaN.
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

}  // namespace pie_fp16_detail

/// `a * b + c` per lane, with ONE rounding -- which is the whole point of the
/// instruction and the reason the fallback is written the way it is.
__device__ __forceinline__ __half2 __hfma2(__half2 a, __half2 b, __half2 c) {
#if PIE_FP16_HAS_SM53
    unsigned int bits;
    asm("fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(bits)
        : "r"(pie_fp16_detail::pack(a)), "r"(pie_fp16_detail::pack(b)),
          "r"(pie_fp16_detail::pack(c)));
    return pie_fp16_detail::unpack(bits);
#else
    __half2 out;
    out.x = pie_fp16_detail::fma_once(a.x, b.x, c.x);
    out.y = pie_fp16_detail::fma_once(a.y, b.y, c.y);
    return out;
#endif
}

/// The larger per lane, same tie-breaks as `__hmax`. FlashInfer's fp16
/// softmax reduces its row maximum with this and then with two butterfly
/// shuffles, so a wrong NaN rule here is a wrong attention weight.
__device__ __forceinline__ __half2 __hmax2(__half2 a, __half2 b) {
#if PIE_FP16_HAS_SM80
    unsigned int bits;
    asm("max.f16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_fp16_detail::pack(a)), "r"(pie_fp16_detail::pack(b)));
    return pie_fp16_detail::unpack(bits);
#else
    return make_half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
#endif
}

// ===-- operators --------------------------------------------------------===
//
// Free functions and not members, which is the one thing this file can still
// do for a type it does not own -- and the reason the only gap it ever had
// was the three `(float)h` sites, which `pie_device.cuh` has since closed.
// Each is its named instruction and nothing else, so `a * b` and
// `__hmul(a, b)` are the same bits by construction rather than by test.

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

// ===-- warp shuffle -----------------------------------------------------===

/// `shfl.sync.bfly.b32` on a pair, which `math.cuh:225` wraps and
/// `prefill.cuh` uses twice per softmax row to fold a 4-lane group.
///
/// The 32-bit builtin needs no header -- NVRTC answers `__shfl_xor_sync` on
/// an empty header set, measured -- so this overload is only the bit-cast
/// around it, and `width` is forwarded rather than pinned because a narrower
/// segment changes which lane a thread reads.
__device__ __forceinline__ __half2 __shfl_xor_sync(unsigned int mask, __half2 var, int lane_mask,
                                                   int width = warpSize) {
    return pie_fp16_detail::unpack(
        __shfl_xor_sync(mask, pie_fp16_detail::pack(var), lane_mask, width));
}

// ===-- what is deliberately not here ------------------------------------===
//
// `__hadd`, `__hadd2`, `__hmin`, `__hmin2`, `__habs`, `__habs2`,
// `__half2half2`, `__low2float`, `__high2float`, `__h2div`, `__hgt` and the
// rest of the comparison set, `__half_raw`, `__half2_raw`, the host-side
// half, the integer and double conversions. Zero uses each, by the same
// two-tree count as the eighteen above. The brief's census over the WHOLE
// FlashInfer include tree found some of them -- `__high2float` 32 times,
// `__hmax` 12, `__habs2` 10 -- but not one of those sites is reachable from
// our fifteen roots, and implementing what the closure does not call is a
// second CUDA nobody asked for. Add one with a probe row or not at all: an
// untested conversion is a wrong answer that compiles.
//
// `__float2int_rd`, `__float2int_rn`, `__float2int_rz`. These are NOT
// `cuda_fp16.h`'s to give, and NVRTC does not need them to be: all three
// compile against an EMPTY header set, measured on NVRTC 13.0 here. Defining
// them would shadow a builtin with a copy, which is the one failure §13.4's
// naming rule is about. `halftype_parity` checks them anyway, because the
// brief counted them and a reader is owed the measurement that says they need
// no shim.
//
// A merge with `pie_half2.cuh`. Both files define `__half2` and eight
// functions at global scope, so a translation unit that included both would
// not compile -- and none does: our `.cu` files include the honest name,
// FlashInfer includes this one, and the two sets are disjoint today. The
// names here are a strict superset of that file's ten, so the day they meet,
// `pie_half2.cuh` becomes one `#include` of this header and nothing else
// changes. It is left standing rather than pre-emptively emptied because it
// is another agent's file and this one may not edit it.

#undef PIE_FP16_HAS_SM80
#undef PIE_FP16_HAS_SM53
