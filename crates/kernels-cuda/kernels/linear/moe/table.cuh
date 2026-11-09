#pragma once

#include "prim/device.cuh"

// Where a routed bank's experts lie (the weight cache's expert table).
//
// A bank is one or more planes (codes, scales, a bias, ...), each `[experts,
// ...]` when contiguous. An expert table names, for every expert, the device
// address of each of its planes: expert `e`'s plane `p` at
// `table[e * planes + p]`, in the plane order of the bank's kind
// (`kernels_cuda::linear::moe::Planes`). A null table is the contiguous bank,
// expert `e`'s plane `p` at `base_p + e * stride_p`: the degenerate table,
// computed rather than read.
//
// A kernel resolves an expert's planes once (per tile or per route), before
// its reduction loop, and walks them with the strides it always did inside;
// the indirection costs one load an expert a tile.
//
// A row table is the same for a table read by token-id row gathers (PLE):
// row block `b` (rows `[b * block_rows, (b + 1) * block_rows)`) has plane `p`
// at `table[b * planes + p]`, row `r` of it at that plus `(r % block_rows) *
// row_stride_p`; null is the contiguous table.

namespace pie::linear {

template <class T>
__device__ __forceinline__ const T* expert_plane(
    const T* base,
    long long stride,
    const u64* __restrict__ table,
    int planes,
    int plane,
    int expert)
{
    return table != nullptr
        ? reinterpret_cast<const T*>(table[static_cast<long long>(expert) * planes + plane])
        : base + static_cast<long long>(expert) * stride;
}

// The first row of expert `e`'s plane in a TMA map whose rows are
// `row_bytes` wide from `window` on: `e * rows` under the contiguous bank
// (the map's base its own), else the table entry's row past the window.
__device__ __forceinline__ long long expert_row(
    const u64* __restrict__ table,
    int planes,
    int plane,
    int expert,
    long long rows,
    u64 window,
    long long row_bytes)
{
    return table != nullptr
        ? static_cast<long long>(
              (table[static_cast<long long>(expert) * planes + plane] - window)
              / static_cast<u64>(row_bytes))
        : static_cast<long long>(expert) * rows;
}

template <class T>
__device__ __forceinline__ const T* table_row(
    const T* base,
    long long row_stride,
    const u64* __restrict__ table,
    int planes,
    int plane,
    int block_rows,
    long long row)
{
    if (table == nullptr) return base + row * row_stride;
    const long long block = row / block_rows;
    return reinterpret_cast<const T*>(table[block * planes + plane])
        + (row - block * block_rows) * row_stride;
}

}
