//===-- pie_mma.cuh - one WMMA shape, restated over mma.sync -------------===//
//
// `nvcuda::wmma`, narrowed to the single fragment shape this tree actually
// instantiates, written directly against the `mma.sync.aligned.m16n8k16`
// instruction so that no `#include <mma.h>` has to resolve.
//
// # Why this exists
//
// `examples/header_probe.rs` measured NVRTC 13.0 on an L40S refusing
// `mma.h` -- *"could not open source file 'mma.h' (no directories in search
// list)"* -- along with `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`,
// `cooperative_groups.h` and `cuda/std/limits`. NVRTC before 13.3 bundles
// none of the device headers, and `src/source.rs` states the rule the crate
// is built on: an `#include` resolves against a header set carried in the
// binary, or it does not resolve at all. Reading `mma.h` out of `$CUDA_HOME`
// at build time was considered and rejected -- it puts a toolkit back on the
// build machine, which is the exact property this crate exists to not need,
// and it pins the embedded device ABI to whatever the build box happened to
// have.
//
// So the header is replaced rather than found. Most of the refused headers
// are replaceable cheaply, because what the kernels want out of them is one
// instruction: `__nv_cvt_float_to_fp8` is a `cvt.rn.satfinite.e4m3x2.f32`,
// `__hfma2` is an `fma.rn.f16x2`, `block.sync()` is a `bar.sync`. This one is
// not cheap and the reason is worth stating, because it is what bounds the
// file:
//
// **`wmma::fragment` is an opaque type the compiler knows BY NAME.** Its
// per-lane register layout is not declared anywhere a header can reach --
// `mma_sync` is a builtin that recognises the type, not a function that reads
// its fields -- so a structurally identical struct is a *different* type and
// buys nothing. The layout has to be reproduced by hand, over the lower-level
// `mma.sync` instruction that does expose it.
//
// In general that would be prohibitive: `wmma` spans four shapes, six element
// types and two accumulate types. Here it is not, because the whole tree
// wants ONE instantiation set. Grepping every `.cu` in `kernels-cuda/csrc`
// finds exactly two users --
//
//   moe/moe_dispatch.cu:547-571      (`moe_decode_wmma_bf16_kernel`)
//   moe/moe_grouped_gemm.cu:69-84    (`kFrag` is `constexpr int kFrag = 16`,
//                                     so its `kFrag,kFrag,kFrag` is 16,16,16)
//
// -- and both ask for the same thing:
//
//   fragment<matrix_a,    16,16,16, bf16, row_major>
//   fragment<matrix_b,    16,16,16, bf16, col_major>
//   fragment<accumulator, 16,16,16, float>
//   fill_fragment / load_matrix_sync / mma_sync / store_matrix_sync(mem_row_major)
//
// One shape, one element type, one store layout. That is the entire surface
// below, and the `static_assert` on the primary template is there so that a
// third caller wanting a fifth shape is told what this file implements rather
// than left to discover it as a register-count diagnostic out of `ptxas`.
//
// # This is not NVIDIA's header
//
// `crt/mma.hpp` on this machine was READ, to check the answer -- the register
// counts, the `.row.col` requirement, the two-instruction decomposition -- and
// nothing was copied out of it. That is deliberate and it is the same
// decision the rest of the migration turns on: copying vendor header text
// into this repository is a REDISTRIBUTION with a `NOTICE` entry behind it,
// and avoiding exactly that is why `mma.h` is being replaced instead of
// vendored. A shim that got here by copy-paste would have made the whole
// exercise pointless.
//
// The lane maps below come from the PTX ISA, §9.7.15.5.8 *Matrix Fragments
// for mma.m16n8k16 with floating point type*, which is a specification and
// not an implementation. They are transcribed as the spec states them, in the
// spec's own `groupID` / `threadID_in_group` variables, so that a reader can
// diff a comment against a paragraph.
//
// # 16x16x16 is TWO instructions
//
// `wmma`'s 16x16x16 has no single `mma.sync` behind it. The hardware
// instruction is `m16n8k16`: M and K match, N does not. So B's 16 columns and
// the accumulator's 16 columns split into two 8-wide halves, and one
// `mma_sync` issues two `mma.sync`es sharing one A fragment. Per lane that is
// 4 x `.b32` of A (8 bf16), 2 x 2 x `.b32` of B (8 bf16), and 2 x 4 x `.f32`
// of accumulator (8 floats) -- which is, not by coincidence, the same 8/8/8
// per-lane element count the real fragments have, since both distribute the
// same 256 elements over 32 lanes.
//
// # The bf16 type is the PRELUDE's, and `__nv_bfloat16` is a name for it
//
// `pie_device.cuh` already defines `pie_cuda_driver::kernels::device::bf16`,
// a struct wrapping an `unsigned short`, and its header explains why it is a
// struct rather than a typedef: as typedefs, `bf16` and `f16` would be ONE
// type and a row that swapped them would typecheck. Every migrated kernel in
// this tree already speaks that type.
//
// Defining a second bf16 here would undo that -- `device::bf16*` and
// `mma::bf16*` would be a pointer conversion C++ refuses, at the boundary
// between a kernel's staging buffer and its fragment load, which is precisely
// where they meet. So this file defines NO bf16. It uses the prelude's, and
// adds the one thing the call sites need: a global-scope `__nv_bfloat16` that
// NAMES it. The two call sites spell `__nv_bfloat16` today because they were
// written against `cuda_bf16.h`; a typedef makes that spelling resolve to the
// prelude's type instead of a vendored one, so the migration touches neither
// `.cu` and there is still exactly one bf16 in the translation unit. A
// duplicate identical typedef is legal, so a sibling shim may declare it too.
//
// It belongs in `pie_device.cuh` rather than here -- it is a scalar-layer
// fact, not an MMA one -- and moves there the moment that file is willing to
// take it. Nothing else about this header changes when it does.
//
// The CONVERSIONS the same call sites use, `__float2bfloat16` and
// `__bfloat162float`, are that same scalar layer's business and are not
// restated here: the prelude spells them `f32_to_bf16` / `bf16_to_f32`.
//
// # What is deliberately not here
//
// `ldmatrix`. It is the fast way to fill an A or B fragment, and it is wrong
// for these two call sites: both hand a plain pointer with a caller-stated
// `ldm` -- one into shared memory, one straight into a global weight matrix
// at stride `K` -- and `ldmatrix` requires a shared-memory address and a
// per-lane row pointer the caller has not got. The loads below compute each
// lane's element indices from its own lane id and read them one at a time,
// which is what `wmma::load_matrix_sync` on an arbitrary `ldm` does anyway.
//
// A `.x[]` member. The real fragments have one; nothing in this tree touches
// it, and it cannot be provided honestly. For A and B the storage IS the
// register vector the instruction takes, so a bf16-typed `.x` would either be
// a second copy of the data or a repack on every `mma_sync`; for the
// accumulator the element ORDER within `.x` is unspecified by the vendor, so
// a caller indexing it is relying on something neither implementation
// promises. Leaving it out makes `frag.x[i]` a compile error pointing here
// instead of a silent disagreement between the two paths.
//
// `load_matrix_sync` for an accumulator, and `fill_fragment` for A or B.
// Unused. Add them with their own parity check or not at all -- an untested
// lane map is a wrong answer that compiles.
//
// # The check that makes this trustworthy
//
// A lane-mapping error here is a silent wrong answer, not a compile error, so
// compiling proves nothing. `examples/mma_probe.rs` is the gate: it runs this
// shim under NVRTC and the real `nvcuda::wmma` under `nvcc` on the same
// inputs on the same device and compares the two 16x16 results element by
// element with no tolerance. Measured on an L40S (sm_89), CUDA 13.0 /
// NVRTC 13.0: **bit-identical, max abs difference exactly 0, on all four
// cases** -- a pseudo-random tile of multiples of 1/8, a pseudo-random tile
// with full bf16 mantissa entropy over sixteen octaves of exponent, `A = I`,
// and a `B` with exactly one non-zero column. The probe then flips its own
// store to `mem_col_major` and requires the comparison to CATCH it, because
// four passes prove nothing unless a failure was reachable.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

/// The spelling `moe_dispatch.cu` and `moe_grouped_gemm.cu` already use, made
/// a name for the prelude's type rather than a type of its own. See the
/// header comment: there is one bf16 in this translation unit and it is
/// `pie_cuda_driver::kernels::device::bf16`.
using __nv_bfloat16 = ::pie_cuda_driver::kernels::device::bf16;

namespace nvcuda {
namespace wmma {

/// The fragment roles, as tag types. Incomplete on purpose -- nothing ever
/// makes one, they only select a specialization.
struct matrix_a;
struct matrix_b;
struct accumulator;

/// How a multiplicand's tile is laid out in the memory `load_matrix_sync`
/// reads. Also tags, for the same reason.
struct row_major;
struct col_major;

/// How an accumulator's tile is laid out in the memory `store_matrix_sync`
/// writes. A runtime value in the real API and kept one here, because the
/// call sites pass it as one.
enum layout_t { mem_row_major, mem_col_major, mem_undefined };

namespace detail {

/// A `false` the compiler cannot evaluate until the template is instantiated,
/// so the primary `fragment` below fires its `static_assert` on use and not
/// on definition.
template <typename...>
struct dependent_false {
    static constexpr bool value = false;
};

/// `pie_cuda_driver::kernels::device`, once, so the code below reads as
/// arithmetic rather than as qualification.
namespace pie = ::pie_cuda_driver::kernels::device;

/// The PTX ISA's `groupID` for this lane: which of the eight four-lane groups
/// it is in. Every fragment map below is written in terms of this and
/// [`lane_in_group`], because that is how §9.7.15.5.8 writes them.
///
/// `threadIdx.x % 32` and not `%laneid`, matching what `wmma` documents: a
/// fragment op is warp-collective over the calling warp, and a block that is
/// not a whole number of warps would make the two disagree.
__device__ __forceinline__ int lane_group() { return static_cast<int>(threadIdx.x % 32u) >> 2; }

/// The PTX ISA's `threadID_in_group`: this lane's position within its group
/// of four.
__device__ __forceinline__ int lane_in_group() { return static_cast<int>(threadIdx.x % 32u) & 3; }

/// Two bf16 into the `.b32` an `.f16x2` operand is, low half first.
///
/// The order is the spec's *"Elements (low to high): a0, a1, ..."* -- the
/// even-indexed element of each pair occupies bits 0..15. Written as a shift
/// rather than a pointer cast because the two elements come from memory
/// locations `ldm` apart as often as adjacent ones, and a cast that assumed
/// otherwise would be right in exactly the cases a probe with `ldm == 16`
/// tests.
__device__ __forceinline__ unsigned pack(pie::bf16 low, pie::bf16 high) {
    return (static_cast<unsigned>(high.raw) << 16) | static_cast<unsigned>(low.raw);
}

/// One `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`: `d = a * b + c`
/// over a 16x16 A, a 16x8 B and a 16x8 accumulator.
///
/// `.row.col` is not a choice -- it is the only layout combination the
/// instruction accepts for 16-bit operands, and it means A's fragment is
/// indexed `(m, k)` and B's `(k, n)`. That is what makes `wmma`'s `row_major`
/// A and `col_major` B the free case: both land in the fragment the hardware
/// already wants, and the loads below differ only in which index strides.
///
/// `d` and `c` are separate parameters and separate registers even though
/// every caller passes the same fragment for both. Writing through the output
/// operands directly would be correct for this single instruction, but the
/// aliasing would be an assumption about inline-asm operand ordering rather
/// than something the code says; the compiler coalesces the copy.
__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4],
    const unsigned (&a)[4],
    const unsigned (&b)[2],
    const float (&c)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    // Pre-Ampere has no bf16 `mma.sync`, so there is nothing to degrade to. A
    // trap rather than a silent zero: both call sites already guard their
    // bodies with `__CUDA_ARCH__ >= 800`, so reaching this means someone
    // removed a guard, and that should stop rather than return plausible
    // numbers.
    (void)a;
    (void)b;
    d[0] = c[0];
    d[1] = c[1];
    d[2] = c[2];
    d[3] = c[3];
    __trap();
#endif
}

}  // namespace detail

/// The primary template: everything this file does not implement.
///
/// Declared with a body that always fails rather than left undefined, because
/// an undefined primary reports "incomplete type" at the declaration -- true,
/// and useless. This says what is available instead.
template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
    static_assert(
        detail::dependent_false<Use, T>::value,
        "pie_mma.cuh implements exactly one WMMA instantiation set, because that is "
        "what this tree instantiates: fragment<matrix_a, 16,16,16, __nv_bfloat16, "
        "row_major>, fragment<matrix_b, 16,16,16, __nv_bfloat16, col_major> and "
        "fragment<accumulator, 16,16,16, float>. See moe/moe_dispatch.cu:547 and "
        "moe/moe_grouped_gemm.cu:69 for the two callers, and this file's header for "
        "why a shape is hand-written rather than resolved out of <mma.h>. A new "
        "shape needs a new lane map from PTX ISA 9.7.15.5 and a parity case in "
        "examples/mma_probe.rs -- an untested lane map compiles and answers wrong.");
};

/// A, as `mma.sync` wants it: four `.b32`, eight bf16, one 16x16 tile.
template <>
struct fragment<matrix_a, 16, 16, 16, ::pie_cuda_driver::kernels::device::bf16, row_major> {
    /// The `.f16x2` operand vector, in the spec's element order: `reg[i]`
    /// holds `a[2i]` low and `a[2i+1]` high. Shared unchanged by both halves
    /// of the 16x16x16, since only N splits.
    unsigned reg[4];
};

/// B, as `mma.sync` wants it, twice: `reg[0..1]` is the n in [0,8) half and
/// `reg[2..3]` the n in [8,16) half.
template <>
struct fragment<matrix_b, 16, 16, 16, ::pie_cuda_driver::kernels::device::bf16, col_major> {
    unsigned reg[4];
};

/// The accumulator, twice: `reg[0..3]` is the n in [0,8) half and `reg[4..7]`
/// the n in [8,16) half, each in the spec's `c0..c3` order.
template <>
struct fragment<accumulator, 16, 16, 16, float, void> {
    float reg[8];
};

/// Every accumulator element on this lane set to `value`.
///
/// Which is every element of the tile, across the warp, exactly once -- the
/// accumulator map is a bijection from (lane, index) to (row, col), so a fill
/// needs no knowledge of the map at all. That is only true for the
/// accumulator; A and B have no `fill_fragment` here for the reason in the
/// header.
__device__ __forceinline__ void fill_fragment(
    fragment<accumulator, 16, 16, 16, float>& frag,
    float value)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag.reg[i] = value;
    }
}

/// Load this lane's eight elements of a row-major 16x16 A tile at `ptr` with
/// leading dimension `ldm`: element `(row, col)` lives at `ptr[row * ldm +
/// col]`.
///
/// PTX ISA §9.7.15.5.8, matrix A, `.f16`/`.bf16`:
///
///     row = groupID              for a_i where 0 <= i < 2 || 4 <= i < 6
///           groupID + 8          otherwise
///     col = threadID_in_group * 2 + (i & 1)        for a_i where i < 4
///           threadID_in_group * 2 + (i & 1) + 8    for a_i where i >= 4
///
/// so a lane owns a 2x2 block at `(g, 2t)` and the three others reached by
/// stepping 8 down and 8 across -- the k-halves of the 16-deep A tile, which
/// is why they land in separate registers.
///
/// `ldm` is not required to be a multiple of eight. The real
/// `load_matrix_sync` requires it, because it issues vector loads; these are
/// scalar, so the constraint does not exist here. That makes this shim accept
/// inputs the real one rejects, which is worth knowing when reading a parity
/// result: the probe uses a legal `ldm` so that both paths are answering the
/// same question.
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, ::pie_cuda_driver::kernels::device::bf16, row_major>& frag,
    const ::pie_cuda_driver::kernels::device::bf16* ptr,
    unsigned ldm)
{
    const int g = detail::lane_group();
    const int t = detail::lane_in_group();
    const int stride = static_cast<int>(ldm);

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        // `half` is the k-half: a0..a3 read columns [0,8), a4..a7 read [8,16).
        const int col = 2 * t + 8 * half;
        const int top = g * stride + col;
        const int bottom = (g + 8) * stride + col;
        frag.reg[2 * half + 0] = detail::pack(ptr[top], ptr[top + 1]);
        frag.reg[2 * half + 1] = detail::pack(ptr[bottom], ptr[bottom + 1]);
    }
}

/// Load this lane's eight elements of a column-major 16x16 B tile at `ptr`
/// with leading dimension `ldm`: element `(k, n)` lives at `ptr[n * ldm + k]`.
///
/// That indexing is what makes `moe_grouped_gemm.cu` need no staging pass --
/// its weight is `[N, K]` row-major, and a column-major view of `W^T` with
/// `ldm = K` is the same bytes.
///
/// PTX ISA §9.7.15.5.8, matrix B, `.f16`/`.bf16`, per 16x8 half:
///
///     row = threadID_in_group * 2 + (i & 1)        for b_i where i < 2
///           threadID_in_group * 2 + (i & 1) + 8    for b_i where i >= 2
///     col = groupID
///
/// A lane therefore owns one column of the 16x8 half and four of its rows.
/// The half's column `groupID` is global column `8 * half + groupID`, which
/// is the only place the two `mma.sync`es differ in what they read.
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, ::pie_cuda_driver::kernels::device::bf16, col_major>& frag,
    const ::pie_cuda_driver::kernels::device::bf16* ptr,
    unsigned ldm)
{
    const int g = detail::lane_group();
    const int t = detail::lane_in_group();
    const int stride = static_cast<int>(ldm);

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        // `half` is the n-half: this lane's column of the 16x8 tile.
        const int base = (8 * half + g) * stride + 2 * t;
        frag.reg[2 * half + 0] = detail::pack(ptr[base], ptr[base + 1]);
        frag.reg[2 * half + 1] = detail::pack(ptr[base + 8], ptr[base + 9]);
    }
}

/// `d = a * b + c`, as the two `mma.sync`es the 16-wide N splits into.
///
/// Both read the same A registers. Nothing here is warp-synchronising beyond
/// what `mma.sync` itself is: the instruction is warp-collective and the
/// fragments are already in registers, so there is no barrier to add and the
/// real `mma_sync` adds none either.
__device__ __forceinline__ void mma_sync(
    fragment<accumulator, 16, 16, 16, float>& d,
    const fragment<matrix_a, 16, 16, 16, ::pie_cuda_driver::kernels::device::bf16, row_major>& a,
    const fragment<matrix_b, 16, 16, 16, ::pie_cuda_driver::kernels::device::bf16, col_major>& b,
    const fragment<accumulator, 16, 16, 16, float>& c)
{
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned b_half[2] = {b.reg[2 * half + 0], b.reg[2 * half + 1]};
        const float c_half[4] = {
            c.reg[4 * half + 0],
            c.reg[4 * half + 1],
            c.reg[4 * half + 2],
            c.reg[4 * half + 3],
        };
        float d_half[4];
        detail::mma_m16n8k16(d_half, a.reg, b_half, c_half);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            d.reg[4 * half + i] = d_half[i];
        }
    }
}

/// Write this lane's eight accumulator elements to the 16x16 tile at `ptr`
/// with leading dimension `ldm`.
///
/// PTX ISA §9.7.15.5.8, accumulator, per 16x8 half:
///
///     row = groupID          for c_i where i < 2
///           groupID + 8      for c_i where i >= 2
///     col = threadID_in_group * 2 + (i & 1)
///
/// with the half's column `col` being global column `8 * half + col` -- the
/// same 2x2-block-plus-offset shape as A, which is what the two share by
/// being the same M.
///
/// **The contract this pins down**, because it is the one a transposed store
/// would break silently: the accumulator is indexed `(m, n)` where `m` is A's
/// row and `n` is B's COLUMN in `(k, n)` terms. Under `mem_row_major` that
/// lands at `ptr[m * ldm + n]`. So with `A = I`, the result read back
/// row-major is B in `(k, n)` order -- which is the TRANSPOSE of B's own
/// memory read row-major, since B was loaded column-major. `mma_probe`'s
/// identity case asserts exactly that, and would fail if this stored `(n, m)`.
__device__ __forceinline__ void store_matrix_sync(
    float* ptr,
    const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm,
    layout_t layout)
{
    const int g = detail::lane_group();
    const int t = detail::lane_in_group();
    const int stride = static_cast<int>(ldm);

#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i >> 1);
            const int col = 8 * half + 2 * t + (i & 1);
            // `mem_col_major` costs one swap and is implemented rather than
            // refused, because the alternative -- accepting the enumerator
            // and ignoring it -- is a wrong answer with no diagnostic. Only
            // `mem_row_major` is exercised by the two call sites and by the
            // probe; the other is here so that the enum cannot lie.
            const int at = (layout == mem_col_major) ? (col * stride + row) : (row * stride + col);
            ptr[at] = frag.reg[4 * half + i];
        }
    }
}

}  // namespace wmma
}  // namespace nvcuda
