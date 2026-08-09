//===-- gather_rows.cuh - row gathers, the NLD/LND transpose, the ----===//
//===-- scaled embed-concat -------------------------------------------===//
//
// Four `__global__`s. `gather_rows.cu` includes this file and keeps its three
// launchers, so exactly ONE definition of each kernel exists in the tree -- a
// split and not a copy. Two copies agree the day they are written and each
// stays right for whichever half of the tests exercises it;
// `norm/altup_aux` shipped a release that way.
//
// # Two are rows, two cannot be
//
// `gather_rows<T>` is `LaunchRule::RouteRows` -- one block per destination
// row -- and `transpose_nld_to_lnd<T>` is `LaunchRule::Elementwise`.
//
// `transpose_nld_to_lnd_vec4` is not, and neither is `embed_scaled_concat`.
// The first is chosen on the HOST from `dim & 7`, and the extent it launches
// over is `dim >> 3` -- an element count that depends on the answer to a test
// no `Source` in `kernels/src/lib.rs` makes. The second has no ahead-of-time
// `KernelSig` at all: `vocab`, `scale` and `hidden_first` are three operands
// with no `Source` between them, and `new-horizon.md` §10.5 refuses an
// invented one. Both stay here because this file is the family's device text,
// not because a row will find them.
//
// # The stride that had to change
//
// `gather_rows` strode `j += BLOCK` against a file-scope `constexpr int BLOCK
// = 256`. A `LaunchRule` picks the block, so a kernel that hard-codes it is a
// kernel that silently drops elements the day a rule picks anything else --
// `RouteRows` picks `min(1024, ceil(width/32)*32)`, which is 256 only by
// coincidence of width. It strides `j += blockDim.x` now. The ahead-of-time
// launcher passes 256 and the two are identical there, so no numeric parity
// commit is owed.
//
// The `uint4` fast path was keyed on `vocab & 7` -- eight `u16` per 16-byte
// element, true only for a two-byte `T`. It is keyed on BYTES now:
// `width * sizeof(T)` divisible by 16. For `T = u16` that is the same
// predicate and the same `n4`, which is the arithmetic the ahead-of-time path
// still runs.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::layout::device {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, so a row may keep spelling its
// element type `device::bf16`, and so the launchers in the enclosing
// namespace -- which write `device::` meaning the prelude's -- go on
// resolving to the same types through these declarations.
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u16;
using ::pie_cuda_driver::kernels::device::usize;

/// One block per destination row -- `LaunchRule::RouteRows`. `row_indices` is
/// read on the device, so the gather is a pure copy the host never sees.
///
/// Vectorises through `uint4` when the row is a whole number of 16-byte
/// elements. The rows themselves are always 16-byte aligned: a contiguous
/// `cudaMalloc` satisfies 256, and every row starts at a multiple of the row
/// width in bytes, which the test just established.
///
/// Strides by `blockDim.x` and not by a file-scope constant. A `LaunchRule`
/// picks the block -- `RouteRows` picks `min(1024, ceil(width/32)*32)` -- and
/// a kernel that hard-codes 256 drops every element past the 256th the first
/// time a rule disagrees.
template <class T>
__global__ void gather_rows(
    const T* __restrict__ src,
    const i32* __restrict__ row_indices,
    T* __restrict__ dst,
    int width)
{
    const int slot = blockIdx.x;
    const int row = row_indices[slot];
    const T* src_row = src + static_cast<long long>(row) * width;
    T* dst_row = dst + static_cast<long long>(slot) * width;

    const long long bytes = static_cast<long long>(width) * sizeof(T);
    if ((bytes & 15) == 0) {
        const auto* src4 = reinterpret_cast<const uint4*>(src_row);
        auto*       dst4 = reinterpret_cast<uint4*>(dst_row);
        const int n4 = static_cast<int>(bytes >> 4);
        for (int j = threadIdx.x; j < n4; j += blockDim.x) {
            dst4[j] = src4[j];
        }
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            dst_row[j] = src_row[j];
        }
    }
}

/// The `uint4` form of the transpose, chosen on the HOST from `dim & 7`.
/// Not a row: the element count it is launched over is `dim >> 3`, so the
/// extent depends on the answer to an alignment test no `Source` makes.
__global__ void transpose_nld_to_lnd_vec4(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int n,
    int layers,
    int dim4,
    usize total4)
{
    const usize idx =
        static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total4) return;
    const int d4 = static_cast<int>(idx % dim4);
    const int row = static_cast<int>((idx / dim4) % n);
    const int layer = static_cast<int>(idx / (static_cast<usize>(dim4) * n));
    const usize src_idx =
        (static_cast<usize>(row) * layers + layer) * dim4 + d4;
    dst[idx] = src[src_idx];
}

/// `[n, layers, dim] -> [layers, n, dim]`, one thread per element --
/// `LaunchRule::Elementwise`. `total` survives as an operand because
/// `Elementwise` rounds the element count up to a whole block and the tail
/// threads have to be told to stop; an extent a rule RECOVERS is not an
/// operand, an extent a rule ROUNDS is.
template <class T>
__global__ void transpose_nld_to_lnd(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int n,
    int layers,
    int dim,
    usize total)
{
    const usize idx =
        static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int d = static_cast<int>(idx % dim);
    const int row = static_cast<int>((idx / dim) % n);
    const int layer = static_cast<int>(idx / (static_cast<usize>(dim) * n));
    const usize src_idx =
        (static_cast<usize>(row) * layers + layer) * dim + d;
    dst[idx] = src[src_idx];
}

/// Gemma-3n's per-layer-embedding concat: one half of every output row is
/// the hidden state, the other is a scaled embedding lookup, and
/// `hidden_first` says which. Not a row -- `vocab`, `scale` and
/// `hidden_first` have no `Source` between them, and the family refuses to
/// invent one.
///
/// `scale` is ROUNDED THROUGH bf16 before it multiplies, because the
/// reference implementation stores it as a bf16 tensor and reads it back; the
/// unrounded float differs in the last bit and the tolerance contract holds
/// argmax indices to zero.
__global__ void embed_scaled_concat(
    const i32* __restrict__ token_ids,
    const bf16* __restrict__ embed_weight,
    const bf16* __restrict__ hidden,
    bf16* __restrict__ dst,
    int hidden_cols,
    int vocab,
    float scale,
    bool hidden_first,
    usize total)
{
    const usize idx =
        static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int out_cols = hidden_cols * 2;
    const int row = static_cast<int>(idx / out_cols);
    const int col = static_cast<int>(idx % out_cols);
    const int logical_col =
        (col < hidden_cols) ? col : (col - hidden_cols);
    const bool write_hidden =
        hidden_first ? (col < hidden_cols) : (col >= hidden_cols);
    if (write_hidden) {
        dst[idx] =
            hidden[static_cast<usize>(row) * hidden_cols + logical_col];
        return;
    }

    const i32 tid_raw = token_ids[row];
    const int tid = (tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0;
    const float scale_rounded =
        bf16_to_f32(f32_to_bf16(scale));
    const bf16 v =
        embed_weight[static_cast<long long>(tid) * hidden_cols + logical_col];
    dst[idx] = f32_to_bf16(bf16_to_f32(v) * scale_rounded);
}

}  // namespace pie_cuda_driver::kernels::layout::device
