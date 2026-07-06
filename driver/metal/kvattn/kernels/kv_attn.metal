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

// ── paged decode attention (parity reference kernel) ─────────────────────────
// One thread per (query_row, q_head). Online (flash) softmax over kv positions
// [0, position_ids[row]] with the SAME NHD paged layout as write_kv/sdpa_paged.
// Uses PRECISE exp (fast-math disabled in the harness) for host parity. head_dim
// bounded by MAX_D. grid = N * n_q_heads.
constant constexpr int MAX_D = 128;

kernel void paged_attention_decode(
    device const float* queries          [[buffer(0)]],   // [N, n_q_heads, d]
    device const float* k_pages          [[buffer(1)]],   // [pages, page_size, n_kv_heads, d]
    device const float* v_pages          [[buffer(2)]],
    device float*       out              [[buffer(3)]],   // [N, n_q_heads, d]
    device const int*   position_ids     [[buffer(4)]],   // [N]
    device const int*   req_of_token     [[buffer(5)]],   // [N]
    device const uint*  kv_page_indices  [[buffer(6)]],
    device const uint*  kv_page_indptr   [[buffer(7)]],   // [R+1]
    constant int&       n_q_heads        [[buffer(8)]],
    constant int&       n_kv_heads       [[buffer(9)]],
    constant int&       d                [[buffer(10)]],
    constant int&       page_size        [[buffer(11)]],
    constant int&       gqa_factor       [[buffer(12)]],
    constant float&     scale            [[buffer(13)]],
    constant uint&      total            [[buffer(14)]],  // N * n_q_heads
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const int row = (int)(gid / (uint)n_q_heads);
    const int qh = (int)(gid % (uint)n_q_heads);
    const int kv_head = qh / gqa_factor;
    const int r = req_of_token[row];
    const int q_pos = position_ids[row];
    const int page_base = (int)kv_page_indptr[r];
    const ulong qbase = ((ulong)row * n_q_heads + qh) * d;

    float m = -3.0e38f, l = 0.0f;
    float acc[MAX_D];
    for (int i = 0; i < d; ++i) acc[i] = 0.0f;

    for (int kp = 0; kp <= q_pos; ++kp) {
        const int page = (int)kv_page_indices[page_base + kp / page_size];
        const ulong slot = (ulong)page * page_size + (kp % page_size);
        const ulong kb = (slot * n_kv_heads + kv_head) * d;
        float score = 0.0f;
        for (int i = 0; i < d; ++i) score += queries[qbase + i] * k_pages[kb + i];
        score *= scale;
        const float nm = max(m, score);
        const float factor = exp(m - nm);
        const float e = exp(score - nm);
        l = l * factor + e;
        for (int i = 0; i < d; ++i) acc[i] = acc[i] * factor + e * v_pages[kb + i];
        m = nm;
    }
    for (int i = 0; i < d; ++i) out[qbase + i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}
