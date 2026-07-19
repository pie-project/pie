#pragma once

// Gather selected rows of a 2-D bf16 buffer into the compact PTIR logits
// matrix. Treating the payload as `std::uint16_t` keeps the ABI plain CUDA.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_gather_bf16_rows(
    const std::uint16_t* src,        // [num_src_rows, vocab] bf16
    const std::int32_t*  row_indices, // [num_dst_rows] device — source rows
    std::uint16_t*       dst,         // [num_dst_rows, vocab] bf16
    int                  num_dst_rows,
    int                  vocab,
    cudaStream_t         stream);

void launch_transpose_bf16_nld_to_lnd(
    const std::uint16_t* src,        // [N, L, D] bf16
    std::uint16_t*       dst,        // [L, N, D] bf16
    int                  n,
    int                  layers,
    int                  dim,
    cudaStream_t         stream);

void launch_embed_scaled_concat_bf16(
    const std::int32_t* token_ids,    // [rows]
    const void*         embed_weight, // [vocab, hidden] bf16
    const std::uint16_t* hidden,      // [rows, hidden] bf16
    std::uint16_t*       dst,         // [rows, 2 * hidden] bf16
    int                  rows,
    int                  hidden_cols,
    int                  vocab,
    float                scale,
    bool                 hidden_first,
    cudaStream_t         stream);

}  // namespace pie_cuda_driver::kernels
