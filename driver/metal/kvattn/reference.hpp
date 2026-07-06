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
#include <cmath>
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

// ── paged decode attention (f32 reference) ──────────────────────────────────
// Single-query-per-row causal attention over a paged NHD KV cache
// (k/v_pages = [num_pages, page_size, n_kv_heads, d], element =
// (slot*n_kv_heads + kv_head)*d + i, slot = page*page_size + kp%page_size,
// page = kv_page_indices[kv_page_indptr[r] + kp/page_size]) — the same layout as
// write_kv (NHD) and raw_metal sdpa_paged. Row `row` attends kv positions
// [0, position_ids[row]]. Online (flash) softmax, so the accumulation order
// matches the Metal kernel; only exp rounding can differ.
struct AttnConfig {
    int N = 0;            // query rows
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int d = 0;            // head_dim
    int page_size = 0;
    float scale = 1.0f;
    std::vector<std::int32_t> position_ids;   // [N] causal bound
    std::vector<std::int32_t> req_of_token;   // [N] owning request
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_page_indptr;  // [R+1]
    int gqa_factor() const { return n_q_heads / n_kv_heads; }
};

inline std::vector<float> paged_attention(const AttnConfig& c,
                                          const std::vector<float>& queries,  // [N,n_q_heads,d]
                                          const std::vector<float>& k_pages,
                                          const std::vector<float>& v_pages) {
    std::vector<float> out(static_cast<std::size_t>(c.N) * c.n_q_heads * c.d, 0.0f);
    for (int row = 0; row < c.N; ++row) {
        const int r = c.req_of_token[row];
        const int q_pos = c.position_ids[row];
        const int page_base = static_cast<int>(c.kv_page_indptr[r]);
        for (int qh = 0; qh < c.n_q_heads; ++qh) {
            const int kv_head = qh / c.gqa_factor();
            const std::size_t qbase = (static_cast<std::size_t>(row) * c.n_q_heads + qh) * c.d;
            float m = -3.0e38f, l = 0.0f;
            std::vector<float> acc(c.d, 0.0f);
            for (int kp = 0; kp <= q_pos; ++kp) {
                const int page = static_cast<int>(c.kv_page_indices[page_base + kp / c.page_size]);
                const std::size_t slot =
                    static_cast<std::size_t>(page) * c.page_size + (kp % c.page_size);
                const std::size_t kb = (slot * c.n_kv_heads + kv_head) * c.d;
                float score = 0.0f;
                for (int i = 0; i < c.d; ++i) score += queries[qbase + i] * k_pages[kb + i];
                score *= c.scale;
                const float nm = m > score ? m : score;
                const float factor = std::exp(m - nm);
                const float e = std::exp(score - nm);
                l = l * factor + e;
                for (int i = 0; i < c.d; ++i) acc[i] = acc[i] * factor + e * v_pages[kb + i];
                m = nm;
            }
            for (int i = 0; i < c.d; ++i) out[qbase + i] = l == 0.0f ? 0.0f : acc[i] / l;
        }
    }
    return out;
}

}  // namespace kvattn::ref
