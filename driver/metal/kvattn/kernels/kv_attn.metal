// PTIR M3 — Metal KV/attention data-movement kernels.
//
// Bit-exact Metal duals of the CUDA paged-KV movement kernels
// (driver/cuda/src/kernels/kv_paged.cu write_kv_kernel, gather_rows.cu
// gather_bf16_rows_kernel). K/V carried as opaque 16-bit words (bf16 bits) — a
// copy/gather is bit-exact regardless of numeric interpretation.
//
// Runtime-compiled (CLT-only box; no offline metal compiler).

#include <metal_stdlib>
using namespace metal;

// find_request: linear scan (matches kv_paged.cu:19). qo_indptr is [R+1].
static inline int find_request(device const uint* qo_indptr, int R, int token_idx) {
    for (int r = 0; r < R; ++r)
        if (token_idx < (int)qo_indptr[r + 1]) return r;
    return R - 1;
}

// write_kv: one thread per (token, element) — grid = total_tokens * (h_kv*d).
// Scatters each current-step token's K/V row into its paged slot. `hnd_layout`
// (0/1) selects head-major dst indexing, matching write_kv_kernel.
kernel void write_kv(
    device const ushort* k_curr           [[buffer(0)]],
    device const ushort* v_curr           [[buffer(1)]],
    device ushort*       k_pages          [[buffer(2)]],
    device ushort*       v_pages          [[buffer(3)]],
    device const uint*   qo_indptr        [[buffer(4)]],
    device const uint*   kv_page_indices  [[buffer(5)]],
    device const uint*   kv_page_indptr   [[buffer(6)]],
    device const uint*   kv_last_page_lens[[buffer(7)]],
    constant int&        R                [[buffer(8)]],
    constant int&        page_size        [[buffer(9)]],
    constant int&        h_kv             [[buffer(10)]],
    constant int&        d                [[buffer(11)]],
    constant int&        hnd_layout       [[buffer(12)]],
    constant uint&       total_elems      [[buffer(13)]],  // total_tokens * h_kv*d
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elems) return;
    const int row = h_kv * d;
    const int t = (int)(gid / (uint)row);
    const int i = (int)(gid % (uint)row);

    const int r = find_request(qo_indptr, R, t);
    const int qo_lo = (int)qo_indptr[r];
    const int new_tokens_r = (int)qo_indptr[r + 1] - qo_lo;
    const int offset_in_new = t - qo_lo;
    const int pages_first = (int)kv_page_indptr[r];
    const int num_pages_r = (int)kv_page_indptr[r + 1] - pages_first;
    const int total_kv_after = (num_pages_r - 1) * page_size + (int)kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;
    const int page_in_req = abs_kv_pos / page_size;
    const int offset_in_page = abs_kv_pos % page_size;
    const int actual_page = (int)kv_page_indices[pages_first + page_in_req];

    ulong dst;
    if (hnd_layout != 0) {
        const int h = i / d;
        const int j = i - h * d;
        dst = ((ulong)((ulong)actual_page * h_kv + h) * page_size + offset_in_page) * d + j;
    } else {
        dst = ((ulong)((ulong)actual_page * page_size) + offset_in_page) * row + i;
    }
    const ulong src = (ulong)t * row + i;
    k_pages[dst] = k_curr[src];
    v_pages[dst] = v_curr[src];
}

// gather_rows: dst[slot*vocab + j] = src[row_indices[slot]*vocab + j].
// grid = num_dst_rows * vocab.
kernel void gather_rows(
    device const ushort* src         [[buffer(0)]],
    device const int*    row_indices [[buffer(1)]],
    device ushort*       dst         [[buffer(2)]],
    constant int&        vocab       [[buffer(3)]],
    constant uint&       total       [[buffer(4)]],  // num_dst_rows * vocab
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const int slot = (int)(gid / (uint)vocab);
    const int j = (int)(gid % (uint)vocab);
    const int row = row_indices[slot];
    dst[gid] = src[(ulong)row * vocab + j];
}
