#pragma once

// The paged KV cache's writes. A page holds `page_size` token slots of
// `h_kv` heads × `d` bf16, in one of two layouts:
//   NHD: page → slot → head → d   (a token's heads contiguous)
//   HND: page → head → slot → d   (a head's slots contiguous)

#include "prelude/device.hip"

namespace pie::attn {

// The element offset of (page, slot, head, j) in either layout.
template <bool HND>
__device__ __forceinline__ long long kv_at(long long page, int slot, int head, int j, int page_size,
                                           int h_kv, int d) {
    return HND ? ((page * h_kv + head) * page_size + slot) * d + j
               : ((page * page_size + slot) * h_kv + head) * d + j;
}

// Token row b's k and v (h_kv·d each) into page w_page[b], slot w_off[b].
// A row whose slot is out of range, or that `row_valid` marks dead, is
// skipped. One block per token row; 8 elements a thread when d allows.
template <bool HND>
__global__ void kv_append_explicit(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   bf16* __restrict__ k_pages, bf16* __restrict__ v_pages,
                                   const u32* __restrict__ w_page, const u32* __restrict__ w_off,
                                   const u8* __restrict__ row_valid, int B, int page_size,
                                   int h_kv, int d, const u32* __restrict__ win) {
    const Window window{win};
    if (static_cast<int>(blockIdx.x) >= B || !window.live(blockIdx.x)) return;
    const long long b = window.plane(blockIdx.x);
    if (row_valid != nullptr && row_valid[b] == 0) return;
    const long long page = w_page[b];
    const int slot = static_cast<int>(w_off[b]);
    if (slot < 0 || slot >= page_size) return;
    const int row = h_kv * d;
    const bf16* ks = k + b * row;
    const bf16* vs = v + b * row;
    if (d % 8 == 0 && aligned16(ks) && aligned16(vs) && aligned16(k_pages) && aligned16(v_pages)) {
        for (int i = threadIdx.x * 8; i < row; i += blockDim.x * 8) {
            const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
            *reinterpret_cast<b128*>(k_pages + dst) = *reinterpret_cast<const b128*>(ks + i);
            *reinterpret_cast<b128*>(v_pages + dst) = *reinterpret_cast<const b128*>(vs + i);
        }
        return;
    }
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
        k_pages[dst] = ks[i];
        v_pages[dst] = vs[i];
    }
}

}  // namespace pie::attn
