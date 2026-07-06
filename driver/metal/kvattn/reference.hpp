#pragma once
//
// CPU reference for the Metal KV/attention parity harness (M3) — bit-exact ports
// of the CUDA driver's paged-KV data-movement kernels
// (driver/cuda/src/kernels/kv_paged.cu, gather_rows.cu). K/V are carried as
// opaque 16-bit words (bf16 bits) since write/gather are pure movement — the
// copy is bit-exact regardless of the numeric interpretation.
//
// Pure C++17, no Metal dependency.

#include <cstdint>
#include <vector>

namespace kvattn::ref {

// Linear request scan — matches write_kv_kernel's find_request (kv_paged.cu:19).
inline int find_request(const std::vector<std::uint32_t>& qo_indptr, int R, int token_idx) {
    for (int r = 0; r < R; ++r)
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    return R - 1;
}

struct KvGeometry {
    int R = 0, page_size = 0, h_kv = 0, d = 0;
    std::vector<std::uint32_t> qo_indptr;          // [R+1]
    std::vector<std::uint32_t> kv_page_indices;    // flat page ids (CSR by kv_page_indptr)
    std::vector<std::uint32_t> kv_page_indptr;     // [R+1]
    std::vector<std::uint32_t> kv_last_page_lens;  // [R]
    int total_pages = 0;                           // physical pages allocated

    int total_tokens() const { return static_cast<int>(qo_indptr[R]); }
    int row() const { return h_kv * d; }
    std::size_t pages_words() const {
        return static_cast<std::size_t>(total_pages) * h_kv * page_size * d;
    }
};

// write_kv: scatter each current-step token's [h_kv*d] K/V row into its paged
// slot. `hnd_layout` selects head-major dst indexing (write_kv_kernel, kv_paged.cu).
// Returns the full pages buffer (pre-zeroed) so a comparison covers untouched slots.
inline void write_kv(const KvGeometry& g,
                     const std::vector<std::uint16_t>& k_curr,
                     const std::vector<std::uint16_t>& v_curr,
                     bool hnd_layout,
                     std::vector<std::uint16_t>& k_pages,
                     std::vector<std::uint16_t>& v_pages) {
    const int row = g.row();
    k_pages.assign(g.pages_words(), 0);
    v_pages.assign(g.pages_words(), 0);
    for (int t = 0; t < g.total_tokens(); ++t) {
        const int r = find_request(g.qo_indptr, g.R, t);
        const int qo_lo = g.qo_indptr[r];
        const int new_tokens_r = static_cast<int>(g.qo_indptr[r + 1]) - qo_lo;
        const int offset_in_new = t - qo_lo;
        const int pages_first = g.kv_page_indptr[r];
        const int num_pages_r = static_cast<int>(g.kv_page_indptr[r + 1]) - pages_first;
        const int total_kv_after = (num_pages_r - 1) * g.page_size + g.kv_last_page_lens[r];
        const int pre_kv_len = total_kv_after - new_tokens_r;
        const int abs_kv_pos = pre_kv_len + offset_in_new;
        const int page_in_req = abs_kv_pos / g.page_size;
        const int offset_in_page = abs_kv_pos % g.page_size;
        const int actual_page = static_cast<int>(g.kv_page_indices[pages_first + page_in_req]);
        const std::size_t src = static_cast<std::size_t>(t) * row;
        for (int i = 0; i < row; ++i) {
            std::size_t dst;
            if (hnd_layout) {
                const int h = i / g.d;
                const int j = i - h * g.d;
                dst = ((static_cast<std::size_t>(actual_page) * g.h_kv + h) * g.page_size +
                       offset_in_page) * g.d + j;
            } else {
                dst = ((static_cast<std::size_t>(actual_page) * g.page_size) + offset_in_page) *
                          row + i;
            }
            k_pages[dst] = k_curr[src + i];
            v_pages[dst] = v_curr[src + i];
        }
    }
}

// gather_bf16_rows: dst[slot] = src[row_indices[slot]] over `vocab` u16 words
// (gather_bf16_rows_kernel, gather_rows.cu:11).
inline std::vector<std::uint16_t> gather_rows(const std::vector<std::uint16_t>& src,
                                              const std::vector<std::int32_t>& row_indices,
                                              int vocab) {
    std::vector<std::uint16_t> dst(row_indices.size() * vocab);
    for (std::size_t slot = 0; slot < row_indices.size(); ++slot) {
        const std::size_t s = static_cast<std::size_t>(row_indices[slot]) * vocab;
        const std::size_t dbase = slot * vocab;
        for (int j = 0; j < vocab; ++j) dst[dbase + j] = src[s + j];
    }
    return dst;
}

}  // namespace kvattn::ref
