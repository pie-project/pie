#pragma once

// Where a current-step token's K/V row lands in the paged cache.
//
// Used by the RoPE write-through path in rope.cu, which lands K in the cache
// directly instead of handing it to a second kernel that reads it straight
// back. `write_kv_kernel` in kv_paged.cu still carries its own copy of this
// arithmetic -- it is the reference, and deliberately left untouched, since a
// paged-addressing bug does not fail loudly: it writes correct-looking keys
// to the wrong slot and the model quietly degrades. What keeps the two
// honest is smallop_bench, which runs the fused path and the real
// rope-then-write_kv pair over the same paging inputs and requires the cache
// contents to come out bit-identical.

// The fixed-width integer names, out of the prelude: NVRTC has no
// `<cstdint>`, and this header addresses pages in `u32`.
#include "pie_device.cuh"

namespace pie_cuda_driver::kernels {

// Linear scan to find the request index — `R` is small (≤ batch_size, which
// is bounded by max_forward_requests, ≤ a few hundred).
__device__ __forceinline__ int kv_find_request(
    const device::u32* __restrict__ qo_indptr, int R, int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

struct KvSlot {
    int page;             // physical page index
    int offset_in_page;   // row within that page
};

// Resolve token `t` of the current step to its physical (page, offset).
__device__ __forceinline__ KvSlot kv_slot_for_token(
    const device::u32* __restrict__ qo_indptr,
    const device::u32* __restrict__ kv_page_indices,
    const device::u32* __restrict__ kv_page_indptr,
    const device::u32* __restrict__ kv_last_page_lens,
    int t, int R, int page_size) {
    const int r = kv_find_request(qo_indptr, R, t);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = t - qo_lo;

    const int pages_first = kv_page_indptr[r];
    const int pages_last = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after =
        (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;

    KvSlot s;
    s.page = static_cast<int>(
        kv_page_indices[pages_first + abs_kv_pos / page_size]);
    s.offset_in_page = abs_kv_pos % page_size;
    return s;
}

// Element `i` of the (h_kv * head_dim) row, in the cache's own layout.
template <bool HND_LAYOUT>
__device__ __forceinline__ long long kv_dst_index(
    const KvSlot& s, int i, int page_size, int h_kv, int d) {
    if constexpr (HND_LAYOUT) {
        const int h = i / d;
        const int j = i - h * d;
        return ((static_cast<long long>(s.page) * h_kv + h) * page_size +
                s.offset_in_page) * d + j;
    } else {
        return ((static_cast<long long>(s.page) * page_size) +
                s.offset_in_page) * (static_cast<long long>(h_kv) * d) + i;
    }
}

}  // namespace pie_cuda_driver::kernels
