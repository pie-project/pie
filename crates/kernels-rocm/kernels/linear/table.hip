#pragma once

// Where a weight's rows lie when the weight cache places them: a row table
// names, for each block of `block_rows` rows and each of a weight's
// `planes`, the address the block starts at (a pool slot, or the mapped host
// copy). A null table is the contiguous weight.

#include "prim/device.hip"

namespace pie::linear {

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
