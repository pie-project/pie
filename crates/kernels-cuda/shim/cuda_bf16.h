//===-- cuda_bf16.h - the bf16 door, answered from inside the binary -----===//
//
// The nine names the migrating tree spells out of NVIDIA's `cuda_bf16.h`,
// written directly against the `bf16` and `bf16x2` instructions so that
// `#include <cuda_bf16.h>` resolves without a toolkit on the machine.
//
// # Why this exists
//
// NVRTC 13.0 was probed on this L40S with an EMPTY header set over the 31
// external includes of the FlashInfer attention closure -- 28 files, 17,981
// lines, reachable from `attention/{cascade,decode,mask,mla,prefill,scheduler,
// state,variants,default_*_params}.cuh` and `{fastdiv,layout,page,pos_enc,
// utils}.cuh`. **0 of 31 answered**, `cuda_bf16.h` among them. NVRTC before
// 13.3 bundles none of the device headers, and `src/source.rs` states the
// rule the crate is built on: an `#include` resolves against a header set
// carried in the binary, or it does not resolve at all. Reading the toolkit's
// copy at build time was tried and rejected -- `.wiki/driver/new-horizon.md`
// §13.2 -- because it puts a CUDA installation back on the BUILD machine,
// which is the one property this crate exists to not need.
//
// # Why it wears NVIDIA's filename
//
// §13.4's rule: impersonate a vendor header exactly when the includer is
// upstream source we do not own. NVRTC matches `includeNames[]` against the
// LITERAL string in the directive, so a file carried as `cuda_bf16.h` answers
// FlashInfer's `#include <cuda_bf16.h>` with FlashInfer left byte-identical
// to upstream -- the resolution becomes ours without the source becoming
// ours. `pie_mma.cuh` and `pie_half2.cuh` are the other side of the same
// rule: their includers are our own `.cu` files, so they use honest names.
//
// # What replacing it costs, measured
//
// Every name counted twice with comments stripped -- once over the 28-file
// closure, once over the 84 `.cu`/`.cuh` files of `kernels-cuda/csrc/src`:
// the ahead-of-time archive crate's tree, deleted at `85c6c674b`, which is
// what every `kernels-cuda/` path in this file names and never this crate's
// own `csrc/`. `closure / ours`:
//
//   nv_bfloat16    106 / 3      __hmul2               25 / -
//   __nv_bfloat162  18 / 3      make_bfloat162         8 / -
//   nv_bfloat162    18 / -      __bfloat162float       3 / 90
//   __nv_bfloat16    7 / 576    __float22bfloat162_rn  3 / -
//                               __floats2bfloat162_rn  2 / 3
//                               __float2bfloat16_rn    2 / -
//                               __bfloat1622float2     2 / 4
//                               __float2bfloat16       1 / 59
//                               __float2bfloat162_rn   1 / -
//
// Five `cvt` spellings, one arithmetic instruction, a widen and a pair
// constructor. The 25 `__hmul2` sites are one text: FlashInfer's `packed2_`
// is `half2` or `nv_bfloat162` by template argument, so the same 25 lines
// count once in this header's table and once in `cuda_fp16.h`'s.
//
// The counts also decide the SHAPE of the file. `__float2bfloat16` at 59 uses
// and `__bfloat162float` at 90 are our own tree's, from kernels that were
// written against the toolkit before the JIT existed; the closure's own
// traffic is packed, because FlashInfer moves bf16 two at a time. Both
// spellings therefore have to be first-class, and neither may be an alias
// that quietly rounds differently from the other.
//
// # No operator is needed, and that was compiled for, not grepped
//
// Operators do not show up in a grep of the call sites, so every FlashInfer
// TU in the AOT build was recompiled with the vendor's own kill switches --
// `-D__CUDA_NO_BFLOAT16_OPERATORS__` and `-D__CUDA_NO_BFLOAT162_OPERATORS__`,
// one at a time, under `-Xcudafe --error_limit=300`. **Zero errors, both
// macros.** The closure never adds, compares or multiplies a bf16 with an
// operator; it converts to fp32, or it uses `__hmul2`. So this file defines
// no operator overloads, and the absence is a measurement rather than an
// omission. `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` is the one that does bite --
// see the divergence below.
//
// # The bf16 type is the PRELUDE's, and `__nv_bfloat16` is a name for it
//
// `pie_mma.cuh` already declares `using __nv_bfloat16 = pie::bf16;` and
// argues the case at length: `pie_device.cuh` defines the type, every
// migrated kernel already speaks it, and a second bf16 declared in a shim
// would make `pie::bf16*` and `shim::bf16*` a pointer conversion C++ refuses
// at exactly the boundary where a staging buffer meets a fragment load. This file MATCHES that
// decision -- a duplicate identical typedef is legal, which is the invitation
// that file's comment leaves open -- so a translation unit that includes both
// has exactly one `__nv_bfloat16`, and it is `pie::bf16`.
//
// `__nv_bfloat162` is `pie::bf16x2`, which the prelude already defines as an
// `__align__(4)` pair with `x` in the low half. That is not a free choice
// either: `page.cuh` and `vec_dtypes.cuh` `reinterpret_cast` 32-bit words
// onto it, so a reversed pair would swap every other element -- a wrong
// answer, not a compile error.
//
// That is why `pie_device.cuh` is included, and it is the only include here.
// A header in the set that drags in a diamond its includer did not ask for is
// the hazard the set's rule exists to avoid, so the cost is named: a
// FlashInfer source that says `#include <cuda_bf16.h>` now also gets `pie`.
// It is bounded -- the prelude is `#pragma once`, includes nothing itself,
// and puts everything but the shims' aliases inside that namespace -- and it
// is what having ONE bf16 costs.
//
// # The divergence that was known, and is now closed
//
// `float(x)` on a `__nv_bfloat16` needs a MEMBER conversion operator, and a
// member cannot be added to a type this header does not own. Recompiling the
// closure with `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` named the sites:
// `vec_dtypes.cuh:553` (`(float)src[0]` in `vec_cast<float, nv_bfloat16>`),
// and its fp16 twins at `vec_dtypes.cuh:159` and `prefill.cuh:1523`.
//
// There was never a version of this file that closed it. Declaring our own
// `__nv_bfloat16` class with the operator would collide with `pie_mma.cuh`'s
// typedef in any TU that has both -- a hard error, not a divergence -- and it
// would reintroduce the two-pointer-types problem that typedef exists to
// prevent. So this file stated the divergence and named where the fix
// belonged: `pie_device.cuh`, where the type is defined, one
// `__device__ operator float() const` on `bf16` and on `f16` closing all
// three sites at once. The evidence offered was a measurement, not an
// argument -- all 28 closure files compiled, and
// `BatchDecodeWithPagedKVCacheKernel` refused to instantiate on that one
// conversion alone.
//
// That patch has landed. `pie_device.cuh:88` and `:96` now declare
// `explicit __device__ operator float() const` on both structs; `explicit` is
// the right call, because all three sites are explicit casts, so what they
// need is reachable while implicit narrowing stays refused. Re-measured
// against the carried set with no include path on disk: 28 of 28, and
// `BatchDecodeWithPagedKVCacheKernel<half>` instantiates into a 59,184-byte
// sm_89 cubin. `cuda_fp16.h` carries the same note; the measurement is in
// both because either type reaches it first depending on which `DTypeQ` a
// caller picks. `examples/halftype_parity.rs` compiled `(float)b` under NVRTC
// on every run and printed which of the two states it found, because a
// divergence that closed once can open again.
//
// # This is not NVIDIA's header
//
// `cuda_bf16.hpp` on this machine was READ, to check the answer -- which
// operand of `cvt.rn.bf16x2.f32` lands in the high half, that the vendor
// emulates the packed multiply with an FMA below sm_90, that its widen is a
// `mov`, not a `cvt`, below sm_90 -- and nothing was copied out of it.
// Copying vendor header text into this repository is a REDISTRIBUTION with a
// `NOTICE` entry behind it, and avoiding exactly that is why the header is
// being replaced instead of vendored; a shim that got here by copy-paste
// would have made the whole exercise pointless. The semantics below come from
// the PTX ISA -- §9.7.4 *Half Precision Floating Point Instructions* and
// §9.7.9 `cvt` -- which is a specification, not an implementation.
//
// # Architecture, which for bf16 is the whole difficulty
//
// bf16 is younger than fp16 and its ISA support is split three ways, which
// the Target ISA Notes state exactly: a `.bf16` or `.bf16x2` DESTINATION for
// `cvt` requires sm_80; `fma{.relu}.{bf16,bf16x2}` requires sm_80; and
// `mul{.rnd}.bf16` and `mul{.rnd}.bf16x2` require **sm_90**. There is no bf16
// multiply instruction on this L40S at all.
//
// So `__hmul2` has three bodies. On sm_90 it is `mul.rn.bf16x2`. On sm_80
// through sm_89 -- the crate's target today -- it is `fma.rn.bf16x2` against
// a constant `-0.0`, which is the vendor's own decomposition and has to be
// matched instruction for instruction or the bits will not agree. `-0.0` and
// not `+0.0`: `x + (-0.0)` is `x` under round-to-nearest for every `x`
// including both zeros, while `(-0.0) + (+0.0)` is `+0.0`, so an addend of
// `+0.0` would erase the sign of every negative-zero product. Below sm_80 it
// is an fp32 multiply and a software round, which is exact for a reason
// rather than by luck: a bf16 product has at most 16 significant bits and is
// therefore EXACT in fp32, so the narrowing is the only rounding.
//
// The exception to that exactness is stated because it is the kind of thing
// this file exists to not hide: when the product is smaller than fp32's own
// smallest normal, 2^-126, the fp32 result is itself rounded before the
// narrowing sees it. bf16's subnormals reach down to 2^-133, and fp32 still
// carries 16 bits of significand there, so the whole bf16 subnormal range is
// safe; only products BELOW it -- which round to zero in bf16 either way --
// pass through two roundings, and the one input where that could change an
// answer is a product within half an ulp of 2^-134. `halftype_parity` aimed a
// case list at exactly that window -- every pair of exponents summing to 132
// through 135, crossed with the mantissas that straddle the tie -- and the
// fallback came back bit-identical to `mul.rn.bf16x2` on all of them. The
// window is real; nothing reachable falls into it.
//
// The software round also does NOT quiet a NaN in place, which is what the
// obvious implementation does and what this one did until the sweep reached
// its first signalling NaN. `cvt.rn.bf16.f32` discards the payload AND the
// sign: every one of the 65,536 float patterns with an all-ones exponent and
// a non-zero mantissa comes back 0x7fff. The fallback says 0x7fff too.
//
// `PIE_HALFTYPE_FORCE_PORTABLE` makes every gate below take its fallback,
// which is how the sm_75 path was measured on an sm_89 device instead of
// shipping untested, and is still the switch to measure it with.
//
// # The check that made this trustworthy, and where it went
//
// A conversion that rounds the wrong way still compiles, so compiling proves
// nothing, and a tolerance would hide exactly the defect worth catching -- a
// rounding-mode error in attention is a silent wrong answer, not a crash.
// `examples/halftype_parity.rs` was the gate: each function below ran under
// NVRTC against these headers and the same function ran under `nvcc` against
// the real `<cuda_bf16.h>`, on the same inputs on the same device, compared
// as BIT PATTERNS. It swept all 65,536 bf16 patterns, all 65,536 bf16 TIES
// -- `bits | 0x8000`, the values a truncating converter gets wrong and a
// ties-away one gets wrong on the other parity -- 1.2e6 pseudo-random floats
// plus every special (+-0, +-inf, quiet and signalling NaN, largest finite,
// smallest normal, subnormals on both sides), a 216x216 exponent grid for the
// packed multiply, and the 2^-134 case list above. As of its last run: 43
// functions, 39,847,842 comparisons, zero differing bits, and the same again
// with every fallback forced on. It then corrupted one kernel and required
// the comparison to CATCH it, because a green table proves nothing unless a
// red one was reachable.
//
// **That probe is gone**, deleted with the whole of `kernels-cuda/examples/`,
// which is why every mention of `halftype_parity` below is in the past tense
// and names a measurement rather than a file. `cuda_fp16.h` says the rest of
// this in full and it applies here unchanged: the comparison is still the
// standard, and a function added or changed below is unchecked until someone
// rebuilds it.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

// The one edge this file adds, and it is NVIDIA's own: `cuda_bf16.h` line 133
// on this machine reads `#include "cuda_fp16.h"`, so a translation unit that
// names only `<cuda_bf16.h>` gets the whole fp16 environment with it. That is
// not a detail -- it is load-bearing in this tree. Four files in
// `kernels-cuda/csrc/src` (`attn/attention_naive_paged.cu`, `attn/kv_paged.cu`,
// `quant/transcode.cuh`, `quant/dequant_fp8.cu`) include `<cuda_bf16.h>` and
// `<cuda_fp8.h>` and NOT `<cuda_fp16.h>`, and then call
// `__nv_cvt_fp8_to_halfraw`, which `cuda_fp8.h` guards on
// `__CUDA_FP16_TYPES_EXIST__`. Measured without this line: `identifier
// "__half" is undefined`, `identifier "__nv_cvt_fp8_to_halfraw" is undefined`,
// `identifier "__half_raw" is undefined`. They compiled with nvcc only
// because NVIDIA's header pulled fp16 in for them.
//
// So this is not a convenience include and not a diamond the includer did not
// ask for -- it is the transitive surface of the filename this file wears. A
// shim that answers to `cuda_bf16.h` and does not bring fp16 is not a
// drop-in, and the way that failure presents is a name error at a call site
// hundreds of lines from any `#include`. The edge is one-directional and
// cannot cycle: NVIDIA's `cuda_fp16.h` names `cuda_bf16.h` only in a doc
// comment at line 4679, never in a directive, and neither does ours.
#include "cuda_fp16.h"

// Which instructions this compilation may use, written once so the gates are
// countable, and undefined at the end so nothing leaks into the TU.
#if defined(PIE_HALFTYPE_FORCE_PORTABLE)
#define PIE_BF16_HAS_SM90 0
#define PIE_BF16_HAS_SM80 0
#elif defined(__CUDA_ARCH__)
#define PIE_BF16_HAS_SM90 (__CUDA_ARCH__ >= 900)
#define PIE_BF16_HAS_SM80 (__CUDA_ARCH__ >= 800)
#else
#define PIE_BF16_HAS_SM90 0
#define PIE_BF16_HAS_SM80 0
#endif

// ===-- the types --------------------------------------------------------===

/// The prelude's bf16, under the name FlashInfer and our own kernels were
/// written against. Identical to `pie_mma.cuh`'s declaration on purpose: a
/// duplicate typedef is legal, and a TU with both files still has one bf16.
using __nv_bfloat16 = ::pie::bf16;

/// FlashInfer's shorter spelling, 106 sites -- and the whole alias surface
/// NVIDIA's `cuda_bf16.h` exports, which is two names to fp16's eight.
///
/// This one and `nv_bfloat162` below were already here, and that is the only
/// reason the bf16 half of the closure never produced the refusal the fp16
/// half did. `fi-vendor` compiled the 28 files against this header set and
/// got seven `identifier "nv_half" is undefined` and not one complaint about
/// bf16. An unprefixed alias is a compatibility surface, not arithmetic: it
/// costs nothing at any architecture and its absence costs whole files.
using nv_bfloat16 = __nv_bfloat16;

/// The prelude's pair, `x` in the low 16 bits of the 32-bit register.
using __nv_bfloat162 = ::pie::bf16x2;

using nv_bfloat162 = __nv_bfloat162;

// ===-- the storage structs, and the macro that announces them -----------===
//
// The same pair of names on the bf16 side, here for the same reason and by
// the same lesson: `csrc/shim/cuda_fp8.h` reads
// `static_cast<__nv_bfloat16_raw>(f).x` in both fp8 constructors and builds a
// `__nv_bfloat16_raw` to hand back from both `operator __nv_bfloat16()`, all
// of it behind `#if defined(__CUDA_BF16_TYPES_EXIST__)`. This file did not
// define that macro, so the guard was false, so the bf16 half of the fp8
// interop compiled out silently. A census over the FlashInfer closure and
// `kernels-cuda/csrc/src` counted zero uses and was right about both trees --
// and blind to a consumer that is itself a shim in this header set.
//
// Layout is the vendor's: `__align__(2)` with one `unsigned short x`, and the
// pair with `x` and `y`. The conversions live on the raw struct rather than
// on `__nv_bfloat16`, because that name is `bf16` -- pinned by
// `pie_mma.cuh` and defined in a crate this file may not edit -- so no member
// can be added to it. `static_cast` in both directions still means what it
// means with NVIDIA's header, which is the only property a caller can see.
// Neither conversion is `explicit`, so copy-initialisation reaches them.
//
// `__nv_bfloat162_raw` earns its place twice over: `vec_dtypes.cuh:103` and
// `:104` name it. Both sites are inside FlashInfer's dead
// `#if (__CUDACC_VER_MAJOR__ * 10000 + ... < 120200) && (__CUDA_ARCH__ < 800)`
// block, so nothing reaches them at CUDA 13.0 on sm_89 -- but a type that two
// upstream lines already spell is not a type to leave out on the strength of
// a preprocessor branch being false today.
struct __align__(2) __nv_bfloat16_raw {
    unsigned short x;

    __nv_bfloat16_raw() = default;
    /// From the bits, `constexpr`.
    ///
    /// Added 2026-08-16 for
    /// `flashinfer/comm/trtllm_allreduce_fusion.cuh:1212-1216`'s
    /// `neg_zero<nv_bfloat16>`, whose `static constexpr __nv_bfloat16 value =
    /// __nv_bfloat16_raw{0x8000U}` is the Lamport protocol's empty-slot
    /// sentinel. `csrc/shim/cuda_fp16.h`'s `__half_raw(unsigned short)` is the
    /// same addition for the same struct in the fp16 tree and carries the
    /// argument in full, including why `constexpr` reaches down into
    /// `pie_device.cuh`.
    ///
    /// It is the LIVE half of that pair: the bf16 sentinel is what
    /// `driver-cuda`'s `fire::all_reduce` writes into the Lamport buffer with
    /// `cuMemsetD16_v2`, and `clear_vec.fill(neg_zero_v<T>)` in the one-shot
    /// kernel is what reads it back.
    __device__ constexpr __nv_bfloat16_raw(unsigned short bits) : x(bits) {}
    __device__ __forceinline__ __nv_bfloat16_raw(const __nv_bfloat16 v) : x(v.raw) {}
    /// The bits as a `__nv_bfloat16`. See `__half_raw`'s counterpart for why
    /// this is the prelude's bit constructor rather than assign-to-member.
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

/// The announcement, at the point the four types exist.
///
/// Every other header in the set tests this to decide whether bf16 interop is
/// available; defining it promises that `__nv_bfloat16`, `__nv_bfloat162`,
/// `__nv_bfloat16_raw` and `__nv_bfloat162_raw` are all in scope below this
/// line. Do not move it above them.
#define __CUDA_BF16_TYPES_EXIST__

namespace pie_bf16_detail {

/// The 32-bit view a `.bf16x2` instruction takes, built by shifts rather than
/// by a type pun, so the lane order is a statement in the code instead of a
/// property of the machine a reader has to already know.
__device__ __forceinline__ unsigned int pack(__nv_bfloat162 v) {
    return (static_cast<unsigned int>(v.y.raw) << 16) | static_cast<unsigned int>(v.x.raw);
}

/// The inverse, and the only place the low half is claimed to be `x`.
__device__ __forceinline__ __nv_bfloat162 unpack(unsigned int bits) {
    __nv_bfloat162 out;
    out.x.raw = static_cast<unsigned short>(bits & 0xffffu);
    out.y.raw = static_cast<unsigned short>(bits >> 16);
    return out;
}

}  // namespace pie_bf16_detail

// ===-- conversions ------------------------------------------------------===

/// `bf16 -> f32`, exact and free: bfloat16 IS fp32 with the low 16 bits
/// dropped, so widening is a shift into the high half.
///
/// The prelude's `bf16_to_f32` and nothing else, which is the one case in
/// these two headers where the vendor spelling and the crate's own are the
/// same function -- `cvt.f32.bf16` does not exist below sm_90 and NVIDIA
/// emits `mov.b32 %0, {0, %1}` there, which is this shift. The probe checks
/// it over all 65,536 patterns, NaN payloads included, rather than taking the
/// argument's word for it.
__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 v) {
    return ::pie::bf16_to_f32(v);
}

/// `f32 -> bf16`, round-to-nearest-even.
///
/// The `_rn` suffix is not decoration and getting the two spellings apart is
/// a silent accuracy bug rather than a compile error, so both names are
/// defined and both are this function. At sm_80 and above they ARE one
/// instruction; the vendor's two host fallbacks differ, and its `_rn` one
/// adds a second increment for positive inexact results that is not
/// round-to-nearest-even at all. This file does not reproduce that.
__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float f) {
    __nv_bfloat16 out;
#if PIE_BF16_HAS_SM80
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out.raw) : "f"(f));
#else
    const unsigned int bits = __float_as_uint(f);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        // A NaN. The instruction does NOT keep the sign or the payload --
        // `halftype_parity` compared this branch against `cvt.rn.bf16.f32`
        // over all 65,536 patterns with an all-ones exponent and every one
        // came back 0x7fff, positive even for a negative NaN. Quieting the
        // payload in place, which is what fp16's `cvt` does and what this
        // branch used to do, was wrong here on the first signalling NaN the
        // sweep reached. Rounding it arithmetically would be wrong twice
        // over: the carry could turn a NaN into an infinity.
        out.raw = 0x7fffu;
    } else {
        // Round-to-nearest-even in one add: half an ulp, plus one more when
        // the surviving low bit is odd, so a tie goes to even. The carry out
        // of the mantissa walks into the exponent, which is what makes the
        // largest finite round to infinity without a branch.
        const unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
        out.raw = static_cast<unsigned short>(rounded >> 16);
    }
#endif
    return out;
}

/// The same conversion under the name that says the rounding mode out loud.
__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

/// One float into both lanes. The `mov.b32 d, {t, t}` half needs no sm_80 --
/// only the `cvt` with a `.bf16` destination does, and that one is scalar.
__device__ __forceinline__ __nv_bfloat162 __float2bfloat162_rn(float f) {
    const __nv_bfloat16 t = __float2bfloat16(f);
    __nv_bfloat162 out;
    out.x = t;
    out.y = t;
    return out;
}

/// Two floats, `lo` into `x` and `hi` into `y`.
///
/// The operand order is the trap. `cvt.rn.bf16x2.f32 d, a, b` puts `a` in
/// d[31:16] -- the FIRST source lands in the HIGH half -- so `hi` is named
/// first in the instruction and `lo` second. Backwards, it swaps every pair,
/// silently and everywhere.
__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
#if PIE_BF16_HAS_SM80
    unsigned int bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return pie_bf16_detail::unpack(bits);
#else
    __nv_bfloat162 out;
    out.x = __float2bfloat16(lo);
    out.y = __float2bfloat16(hi);
    return out;
#endif
}

/// A `float2`'s `x` into `x` and `y` into `y`, which is the only reading that
/// survives `vec_dtypes.cuh:572` writing a vector element-wise.
__device__ __forceinline__ __nv_bfloat162 __float22bfloat162_rn(float2 f) {
    return __floats2bfloat162_rn(f.x, f.y);
}

/// Both lanes widened, exactly. The prelude's `bf16x2_to_f32`, for the reason
/// `__bfloat162float` gives: it is the same two shifts.
__device__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat162 v) {
    return ::pie::bf16x2_to_f32(v);
}

/// The pair constructor, spelled as the vector-type constructors are.
/// `vec_t<nv_bfloat16, N>::fill` is 8 of the closure's uses.
__device__ __forceinline__ __nv_bfloat162 make_bfloat162(__nv_bfloat16 x, __nv_bfloat16 y) {
    __nv_bfloat162 out;
    out.x = x;
    out.y = y;
    return out;
}

// ===-- arithmetic -------------------------------------------------------===

/// `a * b` per lane, and the only bf16 arithmetic the closure asks for.
///
/// Three bodies, because bf16 multiply arrived in three stages -- see the
/// header comment. The sm_80 one is an FMA against `-0.0` and is the vendor's
/// decomposition, matched instruction for instruction because anything else
/// would round in a different place.
__device__ __forceinline__ __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
#if PIE_BF16_HAS_SM90
    unsigned int bits;
    asm("mul.rn.bf16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#elif PIE_BF16_HAS_SM80
    unsigned int bits;
    // `0x80008000` is `(-0.0, -0.0)`. The constant goes through a register
    // because `.bf16x2` operands are register-only.
    asm("{ .reg .b32 z;\n"
        "  mov.b32 z, 0x80008000;\n"
        "  fma.rn.bf16x2 %0, %1, %2, z; }"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#else
    // Exact: a bf16 product is 16 significant bits and fits fp32 without
    // rounding, so `__float2bfloat16` below is the only rounding -- except
    // beneath fp32's smallest normal, where the banner says what happens and
    // where `halftype_parity` aimed a case list and found no difference.
    __nv_bfloat162 out;
    out.x = __float2bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x));
    out.y = __float2bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y));
    return out;
#endif
}

// ===-- what is deliberately not here ------------------------------------===
//
// `__hmul`, `__hadd`, `__hsub`, `__hfma`, `__hmax`, `__hmin`, `__habs` and
// their packed forms; `__bfloat162bfloat162`, `__high2float`, `__low2float`,
// `__bfloat16_as_ushort`, `__ushort_as_bfloat16`, `__nv_bfloat16_raw`,
// `__nv_bfloat162_raw`, `__double2bfloat16`, the comparison set, the operator
// set, the host-side bfloat16. Zero uses each, by the same two-tree count as
// the nine above.
//
// The brief's census over the WHOLE FlashInfer include tree -- not the
// closure -- found `__high2float` 32 times, `__hmax2` 17, `__habs2` 10,
// `__bfloat16_as_ushort` 6. Not one of those sites is reachable from our
// fifteen roots. Narrowing the census to the closure is the entire difference
// between this file and a second CUDA nobody asked for; add a name with a
// probe row when something calls it, or not at all.
//
// The three `__hmul` hits the closure does contain are `vec_dtypes.cuh:74-86`,
// inside `#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 <
// 120200) && (__CUDA_ARCH__ < 800)` -- FlashInfer's own shim for toolkits
// older than 12.2, dead at CUDA 13 and dead again at sm_89. It defines
// `make_bfloat162`, `__hmul`, `__hmul2`, `__float2bfloat162_rn` and
// `__float22bfloat162_rn` itself, which is worth knowing for a different
// reason: on a pre-12.2 toolkit those five would be defined TWICE, once there
// and once here. NVRTC 13.0 is the only compiler this header set is fed to,
// so the collision is unreachable -- but it is the reason the block is
// mentioned rather than ignored.
//
// A widening `cvt.f32.bf16`. It exists from sm_80 in the ISA, and NVIDIA
// still emits `mov.b32 {0, h}` below sm_90; the shift is the same bits, and
// one statement of it -- the prelude's -- is better than two.

// `a * b` and `__hadd2_rn(a, b)` over a packed pair.
//
// XQA's `smemFp16ArraySum` and its X-row rescale write both spellings against
// `InputElem2`, which is `__nv_bfloat162` under `-DDTYPE=bf16`. The
// multiply is `__hmul2`, already here in its three arch bodies. The add goes
// through fp32 and back, which is not an approximation: a bf16 sum is
// correctly rounded exactly when it is computed wide and rounded once, and
// `__float22bfloat162_rn` is that one rounding -- the same answer
// `add.rn.bf16x2` gives, without needing sm_90 to say it.
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

/// `(v, v)`, one value splatted across both lanes.
///
/// Added 2026-08-16 for `flashinfer/comm/trtllm_allreduce_fusion.cuh:232-246`
/// -- `bf162bf162` and the `cuda_cast<__nv_bfloat162, __nv_bfloat16>` that
/// forwards to it. `make_bfloat162(v, v)`, and nothing else it could be.
__device__ __forceinline__ __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 v) {
    return make_bfloat162(v, v);
}

/// The bits of `v`, unchanged.
///
/// Added 2026-08-16 for `flashinfer/comm/trtllm_allreduce_fusion.cuh:1251-1254`'s
/// `is_negative_zero<__nv_bfloat16>`, which asks `__bfloat16_as_ushort(x) ==
/// 0x8000`. That predicate is LIVE: `has_neg_zero` and `remove_neg_zero` run
/// it over every element of every vector the one-shot fused all-reduce loads,
/// which is how the Lamport protocol tells an arrived value from an empty
/// slot. A reinterpretation, so there is nothing to round and nothing to
/// measure.
__device__ __forceinline__ unsigned short __bfloat16_as_ushort(__nv_bfloat16 v) { return v.raw; }

/// `|a|`, by clearing the sign bit.
///
/// Added 2026-08-16 for `flashinfer/comm/trtllm_allreduce_fusion.cuh:313-321`'s
/// `maths::cuda_abs`. `csrc/shim/cuda_fp16.h`'s `__habs` carries the argument:
/// magnitude is bits 14:0 for both formats, so the mask is exact for every
/// input including NaN and infinity.
__device__ __forceinline__ __nv_bfloat16 __habs(__nv_bfloat16 a) {
    __nv_bfloat16 out;
    out.raw = static_cast<unsigned short>(a.raw & 0x7fffu);
    return out;
}

/// `|a|` per lane.
__device__ __forceinline__ __nv_bfloat162 __habs2(__nv_bfloat162 a) {
    return make_bfloat162(__habs(a.x), __habs(a.y));
}

/// The larger, with the ISA's tie-breaks.
///
/// Added 2026-08-16 for `flashinfer/comm/trtllm_allreduce_fusion.cuh:340-349`'s
/// `maths::cuda_max<__nv_bfloat16>(__nv_bfloat162)`.
///
/// `max.bf16` is the instruction and its rule is *"if either operand is NaN,
/// the result is the other"* -- the same rule `csrc/shim/cuda_fp16.h`'s
/// `__hmax` states for `max.f16`, and the reason neither is a plain `>`.
///
/// **One difference from that file's `__hmax` is worth stating.** Its
/// pre-sm_80 fallback returns a canonical two-NaN payload that was READ OFF
/// this device by `halftype_parity`. The fallback below returns `0x7fc0` --
/// the quiet bit and nothing else, which is what the ISA calls canonical --
/// and that has NOT been read off anything: `max.bf16` needs sm_80, this box
/// is sm_89, so the fallback is unreachable here and there was nothing to
/// compare it against. On every architecture this crate targets the `asm`
/// arm is the one taken.
__device__ __forceinline__ __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b) {
    __nv_bfloat16 out;
#if PIE_BF16_HAS_SM80
    asm("max.bf16 %0, %1, %2;" : "=h"(out.raw) : "h"(a.raw), "h"(b.raw));
#else
    const bool a_nan = (a.raw & 0x7fffu) > 0x7f80u;
    const bool b_nan = (b.raw & 0x7fffu) > 0x7f80u;
    if (a_nan && b_nan) {
        out.raw = 0x7fc0u;
    } else if (a_nan) {
        out = b;
    } else if (b_nan) {
        out = a;
    } else {
        // `+0.0 > -0.0`, which a bit comparison of the two zeros gets wrong,
        // so the compare goes through fp32 where the sign is honoured.
        out = (__bfloat162float(a) >= __bfloat162float(b)) ? a : b;
    }
#endif
    return out;
}

/// The larger per lane, same tie-breaks as [`__hmax`].
///
/// Added 2026-08-16 for `flashinfer/comm/trtllm_allreduce_fusion.cuh:369-373`'s
/// binary `maths::cuda_max<__nv_bfloat162>`.
__device__ __forceinline__ __nv_bfloat162 __hmax2(__nv_bfloat162 a, __nv_bfloat162 b) {
#if PIE_BF16_HAS_SM80
    unsigned int bits;
    asm("max.bf16x2 %0, %1, %2;"
        : "=r"(bits)
        : "r"(pie_bf16_detail::pack(a)), "r"(pie_bf16_detail::pack(b)));
    return pie_bf16_detail::unpack(bits);
#else
    return make_bfloat162(__hmax(a.x, b.x), __hmax(a.y, b.y));
#endif
}

/// `a + b`, scalar.
///
/// Added 2026-08-16 for `flashinfer/comm/vllm_custom_all_reduce.cuh:115-118`
/// -- `DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b)`, the
/// scalar step of the vllm P2P all-reduce's `packed_assign_add`. The section
/// above lists `__hadd` among the names at zero uses; the comm closure is the
/// first root to reach it, and adding one when something calls it is the rule
/// that section states.
///
/// The body is `__hadd2_rn`'s on one lane, and so is the argument: a bf16 sum
/// is correctly rounded exactly when it is computed wide and rounded once,
/// and `__float2bfloat16` is that one rounding. No arch split, because there
/// is nothing for one to buy -- `add.rn.bf16` needs sm_90 and the fp32 route
/// gives the same bits below it.
__device__ __forceinline__ __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

#undef PIE_BF16_HAS_SM90
#undef PIE_BF16_HAS_SM80
