//===-- gather_rows.cu - the row-gather launchers --------------------===//
//
// Three host launchers and not one `__global__`: the device text is in
// `layout/gather_rows.cuh`, which this file includes so the archive and the
// JIT header set hold the SAME definition rather than two that drift.
//
// Two of the four kernels are also rows in
// `kernels-cuda-new/src/families/layout.rs`. The launchers stay -- this
// migration extracts device text and adds rows, it deletes nothing, and
// `new-horizon.md` §10.10 fixes that order so the two paths can be measured
// against each other.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "layout/gather_rows.hpp"

// The `__global__`s these launchers fire. ONE definition of each.
#include "layout/gather_rows.cuh"

namespace pie_cuda_driver::kernels::layout {

constexpr int BLOCK = 256;

void gather_bf16_rows(
    const device::u16* src,
    const device::i32*  row_indices,
    device::u16*       dst,
    int                  num_dst_rows,
    int                  vocab,
    cudaStream_t         stream)
{
    if (num_dst_rows <= 0) return;
    device::gather_rows<device::u16><<<num_dst_rows, BLOCK, 0, stream>>>(
        src, row_indices, dst, vocab);
}

void transpose_bf16_nld_to_lnd(
    const device::u16* src,
    device::u16*       dst,
    int                  n,
    int                  layers,
    int                  dim,
    cudaStream_t         stream)
{
    if (n <= 0 || layers <= 0 || dim <= 0) return;
    constexpr int BLOCK = 256;
    if ((dim & 7) == 0) {
        const int dim4 = dim >> 3;
        const device::usize total4 =
            static_cast<device::usize>(layers) *
            static_cast<device::usize>(n) *
            static_cast<device::usize>(dim4);
        const int grid = static_cast<int>((total4 + BLOCK - 1) / BLOCK);
        device::transpose_nld_to_lnd_vec4<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const uint4*>(src),
            reinterpret_cast<uint4*>(dst),
            n, layers, dim4, total4);
    } else {
        const device::usize total =
            static_cast<device::usize>(layers) *
            static_cast<device::usize>(n) *
            static_cast<device::usize>(dim);
        const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
        device::transpose_nld_to_lnd<device::u16><<<grid, BLOCK, 0, stream>>>(
            src, dst, n, layers, dim, total);
    }
}

void embed_scaled_concat_bf16(
    const device::i32* token_ids,
    const void*         embed_weight,
    const device::u16* hidden,
    device::u16*       dst,
    int                  rows,
    int                  hidden_cols,
    int                  vocab,
    float                scale,
    bool                 hidden_first,
    cudaStream_t         stream)
{
    if (rows <= 0 || hidden_cols <= 0 || vocab <= 0) return;
    const device::usize total =
        static_cast<device::usize>(rows) *
        static_cast<device::usize>(hidden_cols) * 2u;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    device::embed_scaled_concat<<<grid, BLOCK, 0, stream>>>(
        token_ids,
        static_cast<const device::bf16*>(embed_weight),
        reinterpret_cast<const device::bf16*>(hidden),
        reinterpret_cast<device::bf16*>(dst),
        hidden_cols, vocab, scale, hidden_first, total);
}

}  // namespace pie_cuda_driver::kernels::layout
