#include "kernels/gather_rows.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void gather_bf16_rows_kernel(
    const std::uint16_t* __restrict__ src,
    const std::int32_t* __restrict__ row_indices,
    std::uint16_t* __restrict__ dst,
    int vocab)
{
    const int slot = blockIdx.x;
    const int row = row_indices[slot];
    const std::uint16_t* src_row = src + static_cast<long long>(row) * vocab;
    std::uint16_t* dst_row = dst + static_cast<long long>(slot) * vocab;

    // Vectorize via uint4 (8 × u16) when the vocab is 8-aligned and
    // both rows are 16-byte aligned (which they are here: contiguous
    // `cudaMalloc`'d allocations satisfy 256-byte alignment).
    if ((vocab & 7) == 0) {
        const auto* src4 = reinterpret_cast<const uint4*>(src_row);
        auto*       dst4 = reinterpret_cast<uint4*>(dst_row);
        const int n4 = vocab >> 3;
        for (int j = threadIdx.x; j < n4; j += BLOCK) {
            dst4[j] = src4[j];
        }
    } else {
        for (int j = threadIdx.x; j < vocab; j += BLOCK) {
            dst_row[j] = src_row[j];
        }
    }
}

__global__ void transpose_bf16_nld_to_lnd_vec4_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int n,
    int layers,
    int dim4,
    std::size_t total4)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total4) return;
    const int d4 = static_cast<int>(idx % dim4);
    const int row = static_cast<int>((idx / dim4) % n);
    const int layer = static_cast<int>(idx / (static_cast<std::size_t>(dim4) * n));
    const std::size_t src_idx =
        (static_cast<std::size_t>(row) * layers + layer) * dim4 + d4;
    dst[idx] = src[src_idx];
}

__global__ void transpose_bf16_nld_to_lnd_kernel(
    const std::uint16_t* __restrict__ src,
    std::uint16_t* __restrict__ dst,
    int n,
    int layers,
    int dim,
    std::size_t total)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int d = static_cast<int>(idx % dim);
    const int row = static_cast<int>((idx / dim) % n);
    const int layer = static_cast<int>(idx / (static_cast<std::size_t>(dim) * n));
    const std::size_t src_idx =
        (static_cast<std::size_t>(row) * layers + layer) * dim + d;
    dst[idx] = src[src_idx];
}

__global__ void embed_scaled_concat_bf16_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed_weight,
    const __nv_bfloat16* __restrict__ hidden,
    __nv_bfloat16* __restrict__ dst,
    int hidden_cols,
    int vocab,
    float scale,
    bool hidden_first,
    std::size_t total)
{
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
            hidden[static_cast<std::size_t>(row) * hidden_cols + logical_col];
        return;
    }

    const std::int32_t tid_raw = token_ids[row];
    const int tid = (tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0;
    const float scale_rounded =
        __bfloat162float(__float2bfloat16(scale));
    const __nv_bfloat16 v =
        embed_weight[static_cast<long long>(tid) * hidden_cols + logical_col];
    dst[idx] = __float2bfloat16(__bfloat162float(v) * scale_rounded);
}

}  // namespace

void launch_gather_bf16_rows(
    const std::uint16_t* src,
    const std::int32_t*  row_indices,
    std::uint16_t*       dst,
    int                  num_dst_rows,
    int                  vocab,
    cudaStream_t         stream)
{
    if (num_dst_rows <= 0) return;
    gather_bf16_rows_kernel<<<num_dst_rows, BLOCK, 0, stream>>>(
        src, row_indices, dst, vocab);
}

void launch_transpose_bf16_nld_to_lnd(
    const std::uint16_t* src,
    std::uint16_t*       dst,
    int                  n,
    int                  layers,
    int                  dim,
    cudaStream_t         stream)
{
    if (n <= 0 || layers <= 0 || dim <= 0) return;
    constexpr int BLOCK = 256;
    if ((dim & 7) == 0) {
        const int dim4 = dim >> 3;
        const std::size_t total4 =
            static_cast<std::size_t>(layers) *
            static_cast<std::size_t>(n) *
            static_cast<std::size_t>(dim4);
        const int grid = static_cast<int>((total4 + BLOCK - 1) / BLOCK);
        transpose_bf16_nld_to_lnd_vec4_kernel<<<grid, BLOCK, 0, stream>>>(
            reinterpret_cast<const uint4*>(src),
            reinterpret_cast<uint4*>(dst),
            n, layers, dim4, total4);
    } else {
        const std::size_t total =
            static_cast<std::size_t>(layers) *
            static_cast<std::size_t>(n) *
            static_cast<std::size_t>(dim);
        const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
        transpose_bf16_nld_to_lnd_kernel<<<grid, BLOCK, 0, stream>>>(
            src, dst, n, layers, dim, total);
    }
}

void launch_embed_scaled_concat_bf16(
    const std::int32_t* token_ids,
    const void*         embed_weight,
    const std::uint16_t* hidden,
    std::uint16_t*       dst,
    int                  rows,
    int                  hidden_cols,
    int                  vocab,
    float                scale,
    bool                 hidden_first,
    cudaStream_t         stream)
{
    if (rows <= 0 || hidden_cols <= 0 || vocab <= 0) return;
    const std::size_t total =
        static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(hidden_cols) * 2u;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    embed_scaled_concat_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        token_ids,
        static_cast<const __nv_bfloat16*>(embed_weight),
        reinterpret_cast<const __nv_bfloat16*>(hidden),
        reinterpret_cast<__nv_bfloat16*>(dst),
        hidden_cols, vocab, scale, hidden_first, total);
}

}  // namespace pie_cuda_driver::kernels
