#pragma once

// Key envelopes: per page and kv head, the elementwise min and max of the
// page's live keys, which bound q·k for every token on it. The bounds are
// rounded outward (min down, max up) so a bf16 envelope still contains every
// key it summarises.

#include "prelude/device.hip"

namespace pie::layout {

// f32 → bf16 rounding toward -inf (RD) or +inf (RU). A NaN stays a (quiet)
// NaN.
template <bool UP>
__device__ __forceinline__ bf16 bf16_directed(float f) {
    const u32 b = __builtin_bit_cast(u32, f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return __builtin_bit_cast(bf16, static_cast<u16>((b >> 16) | 0x40u));
    u16 hi = static_cast<u16>(b >> 16);
    // Truncation rounds toward zero: one step away from it for the side
    // that moves away from zero.
    const bool negative = (b >> 31) != 0;
    if ((b & 0xffffu) != 0u && negative != UP) ++hi;
    return __builtin_bit_cast(bf16, hi);
}

// The envelope of `live` tokens of page `page`, kv head `kh`, in an NHD
// pool ([page][token][head][dim]).
template <class T>
__device__ __forceinline__ void reduce_page(const T* __restrict__ k_pages, int page, int kh, int live,
                                            int page_size, int kv_heads, int head_dim, bf16* __restrict__ env_min,
                                            bf16* __restrict__ env_max) {
    const long long token_stride = static_cast<long long>(kv_heads) * head_dim;
    const T* base = k_pages + static_cast<long long>(page) * page_size * token_stride +
                    static_cast<long long>(kh) * head_dim;
    const long long env = (static_cast<long long>(page) * kv_heads + kh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float mn = __builtin_huge_valf();
        float mx = neg_inf();
        for (int t = 0; t < live; ++t) {
            const float v = to_f32(base[t * token_stride + d]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        env_min[env + d] = bf16_directed<false>(mn);
        env_max[env + d] = bf16_directed<true>(mx);
    }
}

// Recomputes the envelopes of every page an append touched. Grid: (slots,
// kv heads), a slot per touched page, slots numbered request by request in
// order: request r's appended tokens [total - qo_len, total) touch its pages
// first..last. A slot past every request's touched pages does nothing.
__global__ void update_appended(const bf16* __restrict__ k_pages, const u32* __restrict__ qo_indptr,
                                const u32* __restrict__ page_indices, const u32* __restrict__ page_indptr,
                                const u32* __restrict__ last_page_lens, bf16* __restrict__ env_min,
                                bf16* __restrict__ env_max, int requests, int page_size, int kv_heads,
                                int head_dim, const u32* __restrict__ win) {
    const Window window{win};
    requests = window.lanes(requests);
    const int slot = blockIdx.x;
    const int kh = blockIdx.y;

    int seen = 0;
    for (int r = 0; r < requests; ++r) {
        const int first_at = static_cast<int>(page_indptr[r]);
        const int pages = static_cast<int>(page_indptr[r + 1]) - first_at;
        const int qo_len = static_cast<int>(qo_indptr[r + 1]) - static_cast<int>(qo_indptr[r]);
        if (pages <= 0 || qo_len <= 0) continue;
        const int last_len = static_cast<int>(last_page_lens[r]);
        const int total = (pages - 1) * page_size + last_len;
        if (total <= 0) continue;
        const int first = (total - qo_len) / page_size;
        const int last = (total - 1) / page_size;
        const int touched = last - first + 1;
        if (slot < seen + touched) {
            const int page = first + (slot - seen);
            if (page >= pages) return;
            const int live = page == last ? last_len : page_size;
            if (live <= 0) return;
            reduce_page(k_pages, static_cast<int>(page_indices[first_at + page]), kh, live, page_size, kv_heads,
                        head_dim, env_min, env_max);
            return;
        }
        seen += touched;
    }
}

}  // namespace pie::layout
