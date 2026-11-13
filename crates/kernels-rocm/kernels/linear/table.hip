#pragma once

// Where the weight cache places a weight's pieces (the HIP twin of
// `linear/moe/table.cuh`).
//
// A routed bank is one or more planes (codes, scales, a bias, ...), each
// `[experts, ...]` when contiguous. An expert table names, for every expert,
// the address of each of its planes: expert `e`'s plane `p` at
// `table[e * planes + p]`, in the plane order of the bank's kind
// (`kernels_rocm::linear::moe::Planes`). A null table is the contiguous
// bank, expert `e`'s plane `p` at `base_p + e * stride_p`: the degenerate
// table, computed rather than read. A kernel resolves an expert's planes
// once (a route, an item or a tile) and walks them with the strides it
// always did.
//
// A row table names, for each block of `block_rows` rows and each of a
// weight's `planes`, the address the block starts at (a pool slot, or the
// mapped host copy). A null table is the contiguous weight.

#include "prim/device.hip"

namespace pie::linear {

// Expert `expert`'s plane `plane` of a bank `stride` elements an expert: at
// `base` laid contiguous where `table` is null, else at its entry.
template <class T>
__device__ __forceinline__ const T* expert_plane(const T* base, long long stride, const u64* __restrict__ table,
                                                 int planes, int plane, int expert) {
    return table != nullptr ? reinterpret_cast<const T*>(table[static_cast<long long>(expert) * planes + plane])
                            : base + static_cast<long long>(expert) * stride;
}

// Counts `reads` of expert `expert` where the table's counter is given.
__device__ __forceinline__ void expert_note(unsigned int* __restrict__ hits, int expert, unsigned int reads) {
    if (hits != nullptr && expert >= 0) atomicAdd(&hits[expert], reads);
}

// Row `row` of plane `plane` of a weight `row_stride` elements a row: at
// `base` laid contiguous where `table` is null, else in its block's place.
template <class T>
__device__ __forceinline__ const T* table_row(const T* base, long long row_stride, const u64* __restrict__ table,
                                              int planes, int plane, int block_rows, long long row) {
    if (table == nullptr) return base + row * row_stride;
    const long long block = row / block_rows;
    return reinterpret_cast<const T*>(table[block * planes + plane]) + (row - block * block_rows) * row_stride;
}

}  // namespace pie::linear
