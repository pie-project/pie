//===-- pie_device.cuh - the prelude every JIT-compiled kernel gets ------===//
//
// The scalar layer the device families are written against: bf16 and f16 as
// storage, the conversions between them and `float`, the `Elem<T>` that
// selects a pair, packed pairs, float limits, and the reductions a row-wise
// kernel folds with.
//
// # Why this exists at all
//
// Every `.cu` in this tree opened with `#include <cuda_bf16.h>` and most also
// took `<cooperative_groups.h>`, `<cfloat>` and `<type_traits>`. None of those
// is available to NVRTC: the CUDA device headers are not bundled before 13.3,
// so including one means VENDORING it -- a redistribution decision with a
// `NOTICE` entry and a pinned device ABI behind it, which
// `.wiki/driver/new-horizon.md` §3.3 says to settle deliberately rather than
// on the way past.
//
// So this file is the answer to "what did those headers actually give us?",
// and the answer is small. `cuda_bf16.h` is 204 KB and these kernels use four
// of its conversions. `<cfloat>` is consulted for one constant. `<type_traits>`
// for two traits. `cooperative_groups` for a warp shuffle that
// `__shfl_down_sync` already is.
//
// Writing the subset out costs the lines below and buys the property the
// whole design is arranged around: **a toolkit-free RUN.** A machine with a
// GPU and no CUDA headers compiles every kernel in this tree, because every
// byte a kernel includes is carried in the Rust binary by
// `driver-cuda/src/bind/headers.rs` and handed to NVRTC as an in-memory
// virtual filesystem.
//
// # Why the namespace is family-neutral
//
// `pie_cuda_driver::kernels::device`, not `...::norm::device`. This was under
// `norm` while `norm` was the only family that had been migrated, and the
// second family to widen a bf16 is exactly the moment that becomes wrong --
// either it reaches into another family's namespace for a scalar type, or it
// restates the type and the two are no longer one instantiation.
//
// # What is NOT here
//
// Anything a kernel could compute for itself, and anything only one family
// needs. A prelude that grows a helper per kernel stops being a prelude; the
// test is whether a second family would want it.
//
//===----------------------------------------------------------------------===//
#pragma once

namespace pie_cuda_driver::kernels::device {

/// bfloat16, as storage.
///
/// A STRUCT and not a `using bf16 = unsigned short`, and the difference is
/// the whole reason a row can be checked. Both formats are sixteen bits, so
/// as typedefs they would be ONE type: `tanh_inplace<bf16>` and
/// `tanh_inplace<f16>` would be one instantiation, a row naming either would
/// get whichever was emitted, and the generated typecheck would accept a row
/// that swapped them because there would be nothing to swap.
///
/// Wrapping the bits makes them two types, which makes the two rows two
/// kernels and makes `const bf16*` where `const f16*` is meant a pointer
/// conversion C++ refuses. It costs nothing on the device: a struct of one
/// `unsigned short` is two bytes aligned to two, which is what the buffers
/// already are.
///
/// `__nv_bfloat16` would do the same job and costs an include -- see the
/// header comment. Nothing below does bf16 ARITHMETIC; every kernel widens,
/// computes in fp32 and narrows, which is what the originals did.
struct bf16 {
    unsigned short raw;

    /// Uninitialised, like the aggregate this used to be.
    ///
    /// Declaring any constructor costs aggregate-ness, and `bf16{bits}` is
    /// written about ten times in this file alone. `= default` plus the
    /// explicit constructor below keeps every one of those working — a braced
    /// initialiser is direct-list-initialisation, which considers explicit
    /// constructors — while `bf16 b = 5;` and `bf16 b = 1.0f;` stay refused.
    bf16() = default;

    /// The bits, as the bits. What `bf16{raw}` meant before there was a
    /// constructor, and what it still means.
    explicit __device__ bf16(unsigned short bits) : raw(bits) {}

    /// `bf16(f)`, and only in that spelling.
    ///
    /// The mirror of [`bf16::operator float`], and it exists for the mirror
    /// reason: FlashInfer's `vec_dtypes.cuh:568` writes `nv_bfloat16(src[0])`
    /// in `vec_cast<nv_bfloat16, float>::cast`, which is the STORE half of
    /// every vectorised conversion in the decode kernel. Without it
    /// `BatchDecodeWithPagedKVCacheKernel` refuses to instantiate with *"no
    /// suitable constructor exists to convert from `const float`"*.
    ///
    /// `explicit`, for [`bf16::operator float`]'s reason: these two structs
    /// are distinct types so that mixing them up is a conversion C++ refuses,
    /// and an implicit constructor from `float` would make `bf16 x = 1.0f;`
    /// compile — silently narrowing wherever a `float` met a `bf16` parameter.
    explicit __device__ bf16(float f);

    /// `(float)b`, and only in that spelling.
    ///
    /// **`explicit` is the whole point.** These two structs are distinct types
    /// so that handing a `bf16*` where an `f16*` is meant is a conversion C++
    /// refuses — `norm_device.rs` records the earlier version, where both were
    /// `unsigned short` and a row that swapped the two formats compiled. An
    /// implicit `operator float` would give that discipline back with one
    /// hand what it took with the other: `bf16 * 2.0f` would silently widen
    /// rather than fail.
    ///
    /// It exists because three sites in the vendored FlashInfer closure need
    /// it — `vec_dtypes.cuh:159`, `vec_dtypes.cuh:553`, `prefill.cuh:1523` —
    /// and every one of them spells an explicit cast, `(float)src[0]` or
    /// `float(...)`, which is exactly what an `explicit` conversion serves.
    /// Without it, `BatchDecodeWithPagedKVCacheKernel` refuses to instantiate
    /// with *"no suitable conversion function from `const half` to `float`"*;
    /// with it, measured, it produces a cubin.
    explicit __device__ operator float() const;
};

/// fp16, as storage. Distinct from [`bf16`] for the reason stated there.
struct f16 {
    unsigned short raw;

    /// See [`bf16::bf16()`].
    f16() = default;
    /// See [`bf16::bf16(unsigned short)`].
    explicit __device__ f16(unsigned short bits) : raw(bits) {}
    /// See [`bf16::bf16(float)`]. Symmetric, so that a template instantiated
    /// at either format finds the same surface.
    explicit __device__ f16(float f);
    /// `(float)h`. See [`bf16::operator float`] for why it is `explicit`.
    explicit __device__ operator float() const;
};

/// `bf16 -> f32`, exact: bfloat16 is fp32 with the low sixteen bits dropped,
/// so widening is a shift and cannot round.
///
/// # Why this is one `mov` and not a shift
///
/// The obvious spelling, `__int_as_float((unsigned)v.raw << 16)`, is what this
/// was and it costs two instructions: a `MOV` to widen the halfword into a
/// 32-bit register and a `SHF.L.U32` to place it. The PTX below is what
/// NVIDIA's own `__internal_device_bfloat162float` uses on this architecture —
/// `mov.b32 %0, {0, %1}` assembles a 32-bit value from two 16-bit halves, so
/// the halfword never leaves a 16-bit register and ptxas emits **one `PRMT`**.
///
/// **Bit-exact by construction**, which is why it is worth doing at all:
/// both forms put the sixteen bits in the high half and zeros in the low, and
/// there is no rounding to disagree about. It is still held to the same gate
/// as everything else here — `kernels-cuda-new/examples/halftype_parity.rs`
/// sweeps `__bfloat162float` over 1,150,464 inputs against nvcc, and
/// `tests/prelude_parity.rs` sweeps the fp16 twin exhaustively.
///
/// The measurement that prompted it: compiling FlashInfer's
/// `BatchDecodeWithPagedKVCacheKernel` against this prelude rather than
/// NVIDIA's headers produced **bit-identical output over 48 launches** and
/// 1,520 SASS instructions against 1,368 — a delta of **+178 `SHF.L.U32`,
/// −96 `PRMT`, +48 `MOV`**, all of it here, on the widening path of a kernel
/// that widens constantly.
__device__ __forceinline__ float bf16_to_f32(bf16 v) {
    float f;
    asm("mov.b32 %0, {0, %1};" : "=f"(f) : "h"(v.raw));
    return f;
}

/// `f32 -> bf16`, round-to-nearest-even -- what `__float2bfloat16` does.
///
/// The `+ 0x7fff + lsb` is the round: adding half an ulp, plus the low bit of
/// the result, breaks ties toward even. Truncating instead is a bias of half
/// an ulp per narrowing, which over a residual stream's worth of them is a
/// drift no test that reads one value would catch.
__device__ __forceinline__ bf16 f32_to_bf16(float f) {
    const unsigned int b = __float_as_int(f);
    // A NaN must stay a NaN. Rounding one can carry into the exponent and
    // produce an infinity, which is a different value with the same smell.
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return bf16{static_cast<unsigned short>((b >> 16) | 0x0040u)};
    }
    const unsigned int rounding = 0x7fffu + ((b >> 16) & 1u);
    return bf16{static_cast<unsigned short>((b + rounding) >> 16)};
}

/// The bits, and back. `__bfloat16_as_ushort` / `__ushort_as_bfloat16`.
///
/// A `bf16` IS its `raw` field, so these are `.raw` and a brace — named
/// anyway, because the call sites that want them are doing a CAS loop on a
/// `unsigned short*` view of a bf16 array, and `bf16_as_u16(x)` says that is
/// deliberate where `x.raw` reads like reaching into a struct.
__device__ __forceinline__ unsigned short bf16_as_u16(bf16 v) { return v.raw; }
__device__ __forceinline__ bf16 u16_as_bf16(unsigned short v) { return bf16{v}; }

/// `f32 -> bf16`, rounded toward -inf and toward +inf.
///
/// `__float2bfloat16_rd` / `_ru`, and the reason they are not a rounding-mode
/// parameter on [`f32_to_bf16`]: these are the two halves of an INTERVAL. The
/// envelope kernels keep a per-dimension `[min, max]` in bf16, and an envelope
/// is only sound if its low end never rounds up and its high end never rounds
/// down. Round-to-nearest on either bound can exclude a value the interval is
/// supposed to contain, and nothing downstream would report it — the envelope
/// simply stops covering a point it covered before.
///
/// Truncating the low sixteen bits moves a POSITIVE value toward -inf and a
/// NEGATIVE one toward +inf, because it always reduces magnitude. So each
/// direction adjusts exactly the sign that truncation carried the wrong way,
/// and only when bits were actually dropped.
__device__ __forceinline__ bf16 f32_to_bf16_rd(float f) {
    const unsigned int b = __float_as_int(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return bf16{static_cast<unsigned short>((b >> 16) | 0x0040u)};
    }
    unsigned short hi = static_cast<unsigned short>(b >> 16);
    // Inexact and negative: truncation reduced the magnitude, which for a
    // negative number is toward +inf. One ulp of magnitude puts it back.
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

/// `f16 -> f32`. Written out for the same reason as the bf16 pair: naming
/// `__half` costs an include, and this is six lines of shifts.
///
/// The sign is applied to the RESULT'S BITS and never by arithmetic. An
/// earlier version of this function returned `__int_as_float(s) + 2^-24 * m`
/// for the subnormal case, which is wrong in a way that reads as right:
/// `__int_as_float(s)` is `-0.0f` when the sign bit is set, and `-0.0 + x`
/// is `+x` for every positive `x`. So **all 1,024 negative fp16 subnormals
/// widened positive, and `-0.0` widened to `+0.0`** — a sign flip on the
/// smallest values, which no test that reads a normal number can see. It was
/// found by an exhaustive 65,536-pattern sweep against `__half2float`, which
/// is the only thing that finds a defect confined to one exponent.
__device__ __forceinline__ float f16_to_f32(f16 v) {
    const unsigned int s = (static_cast<unsigned int>(v.raw) & 0x8000u) << 16;
    const unsigned int e = (static_cast<unsigned int>(v.raw) >> 10) & 0x1fu;
    const unsigned int m = static_cast<unsigned int>(v.raw) & 0x3ffu;
    if (e == 0) {
        // Zero keeps its sign, which is `s` and nothing else.
        if (m == 0) return __int_as_float(s);
        // `2^-24 * m` for m in [1, 1023] is exact in fp32 and positive, so
        // the sign goes on afterwards as a bit rather than as an addition.
        const float magnitude = __int_as_float(0x33800000u) * static_cast<float>(m);
        return __int_as_float(__float_as_int(magnitude) | s);
    }
    if (e == 31) return __int_as_float(s | 0x7f800000u | (m << 13));
    return __int_as_float(s | ((e + 112u) << 23) | (m << 13));
}

/// `f32 -> f16`, round-to-nearest-even, flushing subnormals to zero.
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

/// The conversion operators and the float constructors, defined here because
/// they forward to the widening and narrowing functions above and a member
/// cannot call what has not been declared.
///
/// Out of line rather than in the struct, so the structs stay what they read
/// as — a `raw` and a handful of one-line declarations. `__forceinline__`
/// because a cast that became a call would appear in the middle of every
/// vectorised copy in FlashInfer's `vec_dtypes.cuh`, which is the only reason
/// these exist.
__device__ __forceinline__ bf16::operator float() const { return bf16_to_f32(*this); }
__device__ __forceinline__ f16::operator float() const { return f16_to_f32(*this); }
__device__ __forceinline__ bf16::bf16(float f) : raw(f32_to_bf16(f).raw) {}
__device__ __forceinline__ f16::f16(float f) : raw(f32_to_f16(f).raw) {}

/// How a kernel widens and narrows the format it was instantiated at.
///
/// Specialised on the STORAGE type, which the wrapper structs above are what
/// make possible: as typedefs both formats were `unsigned short` and there
/// would be one specialisation where a row means two. Specialised rather
/// than overloaded, because a set of overloads taking `float` would be
/// chosen by implicit conversion.
///
/// A row that names a format with no specialisation here does not compile,
/// which is the check that costs nothing to keep: adding fp8 means adding a
/// struct and four lines, and forgetting the four lines is a compile error
/// rather than a kernel that reads the wrong bits.
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

/// Block-wide reduction of `local` (one float per thread) to thread 0.
///
/// `__shfl_down_sync` and a shared-memory combine, which is what
/// `cg::tiled_partition<32>(...).shfl_down(...)` lowered to before. `smem`
/// must hold `blockDim.x / 32` floats.
///
/// The fold ORDER is part of the contract, not an implementation detail: a
/// different order sums the same values to a different last bit, and
/// `driver-pipeline`'s tolerance contract holds argmax indices to zero. So
/// this is the original's order, warp by warp, and not a tidier one.
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

// ---------------------------------------------------------------------------
// Beyond the scalar pair: what the rest of the families reach for.
// ---------------------------------------------------------------------------

/// `<cfloat>`'s `FLT_MAX`, and the infinity a max-reduction starts below.
///
/// Stated as bit patterns rather than as literals so the value is exactly the
/// one the C library names: `3.402823466e+38F` is a decimal that has to round
/// back to the same float, and this cannot.
__device__ __forceinline__ float flt_max() { return __int_as_float(0x7f7fffffu); }
__device__ __forceinline__ float pos_inf() { return __int_as_float(0x7f800000u); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000u); }

/// Two bf16 side by side, which is how a vectorised kernel reads a row.
///
/// `__nv_bfloat162` by another name, and deliberately the same SHAPE: two
/// members called `x` and `y`, `x` first. A kernel that reads a row four
/// bytes at a time indexes them by name, and a packed `unsigned int` would
/// have made every one of those call sites shift and mask -- which is both
/// noisier and a chance to get the endianness backwards once.
///
/// Four bytes aligned to four, like the type it replaces, so a `bf16x2*` view
/// of a bf16 array is the same reinterpretation it always was.
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

/// A read-only cached load. `__ldg`, which has overloads for the built-in
/// types and none for the prelude's.
///
/// The generic form forwards to the intrinsic, so `char4` and `float2` and
/// everything else a kernel already loads this way keep the instruction they
/// had. The overloads below are the types `__ldg` has never heard of; each
/// loads the bytes as the built-in of the same width -- one `ld.global.nc`,
/// which is the whole reason a call site reached for `__ldg` -- and then
/// splits.
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

/// The halves of a packed pair, on their own. `__low2float` / `__high2float`.
__device__ __forceinline__ float bf16x2_lo(bf16x2 v) { return bf16_to_f32(v.x); }
__device__ __forceinline__ float bf16x2_hi(bf16x2 v) { return bf16_to_f32(v.y); }

/// `<type_traits>`'s two, and only the two that are used.
///
/// A kernel templated over its element type asks `is_same` to pick a path and
/// `conditional` to name a type. Everything else `<type_traits>` offers has
/// no caller here, and a prelude that carried it would be carrying it for the
/// sake of the name on the file it came from.
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

/// `<cstdint>`'s fixed widths, spelled as the COMPILER's own types.
///
/// `__SIZE_TYPE__` and friends rather than `unsigned long long` and friends,
/// and the difference is an ABI rather than a preference. `std::size_t` is
/// `unsigned long` on LP64 and `unsigned long long` on LLP64; both are 64
/// bits, and C++ mangles them DIFFERENTLY. A launcher declared in a `.hpp`
/// with `std::size_t` and defined in a `.cu` with `unsigned long long` links
/// on neither -- which is exactly the failure this file caused once, as
/// `undefined symbol: ...scalar_mul_bf16(void*, float, unsigned long, ...)`.
///
/// Every one of these macros is predefined by both compilers that read this
/// header, so asking them is both shorter than guessing and correct on a
/// target nobody has tried yet.
using i8 = signed char;
using u8 = unsigned char;
using i16 = short;
using u16 = unsigned short;
using i32 = int;
using u32 = unsigned int;
// `decltype(sizeof(0))` IS `std::size_t` -- the standard says so, which makes
// this the one spelling that needs no macro and no target knowledge. NVRTC
// defines neither `__SIZE_TYPE__` nor `__INT64_TYPE__`; it does have
// `sizeof`.
using usize = decltype(sizeof(0));
using isize = decltype(static_cast<char*>(nullptr) - static_cast<char*>(nullptr));
// The 64-bit pair is SPELLED, not derived from the pointer-width pair.
//
// It used to be `using i64 = isize`, on the argument that `ptrdiff_t` resolves
// to "whichever one this target uses -- which is the same one `<cstdint>`
// picks for `int64_t`". That is false here and the device typecheck measured
// it: `csrc/shim/cstdint` picks `long long`, `ptrdiff_t` on this LP64 target
// is `long`, and the two are distinct types that mangle differently. A row
// declaring `*const i64` was refused against a `__global__` taking
// `const i64*` because the assertion spells the row's side `::std::int64_t*`.
// Same width, same ABI, different type -- which is exactly the class of
// mismatch this file exists to make impossible.
using i64 = long long;
using u64 = unsigned long long;

static_assert(sizeof(i8) == 1 && sizeof(i16) == 2 && sizeof(i32) == 4 && sizeof(i64) == 8);
static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8);
static_assert(sizeof(usize) == sizeof(void*) && sizeof(isize) == sizeof(void*));

/// The two tag types a dispatch is written against.
struct true_type {
    static constexpr bool value = true;
};
struct false_type {
    static constexpr bool value = false;
};

/// Warp-wide maximum of `v`, in every lane.
///
/// The fold ORDER is part of the contract for the same reason [`block_sum`]'s
/// is: a max carries an INDEX with it in the argmax kernels, and two orders
/// that agree on the value can disagree on which lane produced it.
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

/// Warp-wide argmax: the largest `v` and the index that carried it.
///
/// Ties go to the LOWER index, which is what a greedy decode has to promise:
/// `driver-pipeline`'s tolerance contract holds argmax indices to zero
/// difference, so "some lane that had the maximum" is not an answer.
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

}  // namespace pie_cuda_driver::kernels::device
