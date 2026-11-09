#pragma once

#include "prim/device.cuh"
#include "prim/rope.cuh"

#include "prim/kv_rows.cuh"

namespace pie::attn::compressed {

__device__ __forceinline__ long long paged_slot(
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int req, int pos, int page_size) {
    const u32 page =
        kv_page_indices[kv_page_indptr[req] + pos / page_size];
    return static_cast<long long>(page) * page_size + (pos % page_size);
}

// Position `pos`'s `row`-wide row in a pool whose pages lie `page_stride`
// elements apart (`page_size * row` packed), as an element offset.
__device__ __forceinline__ long long paged_row(
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int req, int pos, int page_size, long long page_stride, int row) {
    const u32 page =
        kv_page_indices[kv_page_indptr[req] + pos / page_size];
    return static_cast<long long>(page) * page_stride +
           static_cast<long long>(pos % page_size) * row;
}

// Compressed entry `c` of a pool at `ratio`, as an element offset of its
// `row`-wide row in a pool whose pages lie `page_stride` elements apart:
// compact, one a row (a page of `page_size` positions holds `page_size /
// ratio` of them); else a row a position, at the last position of its group.
__device__ __forceinline__ long long entry_row(
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int req, int c, int ratio, int page_size, int compact, long long page_stride, int row) {
    if (compact) {
        return paged_row(kv_page_indices, kv_page_indptr, req, c, page_size / ratio, page_stride, row);
    }
    return paged_row(kv_page_indices, kv_page_indptr, req, (c + 1) * ratio - 1, page_size, page_stride,
                     row);
}

// The positions request `req`'s pages hold: a padded lane owns none, and no
// position at or past this is read through `paged_slot`.
__device__ __forceinline__ int owned_positions(
    const u32* __restrict__ kv_page_indptr, int req, int page_size) {
    const int pages = static_cast<int>(kv_page_indptr[req + 1]) -
                      static_cast<int>(kv_page_indptr[req]);
    return max(pages, 0) * page_size;
}

}
