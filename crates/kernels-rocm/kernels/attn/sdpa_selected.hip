#pragma once

// Paged GQA attention over the key blocks a per-row selection names (DSA /
// QSA sparse attention), plus the open block that runs up to the query:
//
//   keys of row t = ⋃_{c ∈ selection[t], 0 ≤ c < ⌊j_end / ratio⌋} [c·ratio, (c+1)·ratio)
//                 ∪ [⌊j_end / ratio⌋·ratio, j_end)
//
// with j_end the row's causal end. A block below ⌊j_end / ratio⌋ holds only
// keys before j_end, so rejecting a block id is the whole causal test.
//
// One block per (token row, group of G query heads): 256 threads as eight
// 32-lane groups, K = 8 / G of them per head walking its keys round-robin
// with an online softmax, folded through LDS at the end. A 32-lane group
// (half a wavefront) dots one key, D / 32 elements a lane, so a wavefront
// reads two keys at once.

#include "prelude/device.hip"

namespace pie::attn::sdpa_selected {

constexpr int kBlock = 256;
constexpr int kLanes = 32;
constexpr int kGroups = kBlock / kLanes;
constexpr int kMaxPer = 16;

template <class T>
__global__ __launch_bounds__(kBlock) void sdpa_paged_selected_kernel(
    const T* __restrict__ queries, const T* __restrict__ k_pages, const T* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr, const u32* __restrict__ page_indices,
    const u32* __restrict__ page_indptr, const u32* __restrict__ last_page_lens,
    const i32* __restrict__ selection, T* __restrict__ o, i32 R, i32 H, i32 KVH, i32 D,
    i32 page_size, float sm_scale, i32 top_k, i32 ratio, i32 G, const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    R = window.lanes(R);

    const int sub = static_cast<int>(threadIdx.x) % kLanes;
    const int group = static_cast<int>(threadIdx.x) / kLanes;
    const int K = kGroups / G;
    const int g = group / K;
    const int s = group % K;
    const int h = static_cast<int>(blockIdx.y) * G + g;
    const int kv_head = h / (H / KVH);
    const int per = D / kLanes;

    // The request of launch row t.
    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid;
        else lo = mid + 1;
    }
    const int qo_lo = static_cast<int>(qo_indptr[lo]);
    const int fresh = static_cast<int>(qo_indptr[lo + 1]) - qo_lo;
    const int pages_first = static_cast<int>(page_indptr[lo]);
    const int pages = static_cast<int>(page_indptr[lo + 1]) - pages_first;
    const int kv_len = (pages - 1) * page_size + static_cast<int>(last_page_lens[lo]);
    const int j_end = kv_len - fresh + (t - qo_lo) + 1;

    const int nblocks = ratio > 0 ? j_end / ratio : 0;
    const int blocks = min(top_k, nblocks);
    const i32* srow = selection + t_row * top_k;

    extern __shared__ float smem[];
    float* gacc = smem;
    float* gm = gacc + kGroups * D;
    float* gl = gm + kGroups;

    const T* qrow = queries + (t_row * H + h) * D;
    float qreg[kMaxPer], acc[kMaxPer];
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        qreg[i] = i < per ? to_f32(qrow[sub + i * kLanes]) * sm_scale : 0.f;
        acc[i] = 0.f;
    }
    float m = neg_inf(), l = 0.f;

    const int steps = blocks * ratio + (ratio - 1);
    for (int n = s; n < steps; n += K) {
        int j;
        if (n < blocks * ratio) {
            const int c = srow[n / ratio];
            if (c < 0 || c >= nblocks) continue;
            j = c * ratio + n % ratio;
        } else {
            j = nblocks * ratio + (n - blocks * ratio);
            if (j >= j_end) continue;
        }
        const long long slot = static_cast<long long>(page_indices[pages_first + j / page_size]) * page_size + j % page_size;
        const T* kj = k_pages + (slot * KVH + kv_head) * D;
        const T* vj = v_pages + (slot * KVH + kv_head) * D;

        float pd = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i)
            if (i < per) pd += qreg[i] * to_f32(kj[sub + i * kLanes]);
#pragma unroll
        for (int off = kLanes / 2; off > 0; off >>= 1) pd += __shfl_xor(pd, off, kLanes);
        const float m_new = fmaxf(m, pd);
        const float corr = __expf(m - m_new);
        const float p = __expf(pd - m_new);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * to_f32(vj[sub + i * kLanes]);
        m = m_new;
    }

#pragma unroll
    for (int i = 0; i < kMaxPer; ++i)
        if (i < per) gacc[group * D + sub + i * kLanes] = acc[i];
    if (sub == 0) {
        gm[group] = m;
        gl[group] = l;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < G * D; idx += kBlock) {
        const int gg = idx / D;
        const int d = idx % D;
        const int first = gg * K;
        float m_all = neg_inf();
        for (int w = first; w < first + K; ++w) m_all = fmaxf(m_all, gm[w]);
        float l_all = 0.f, v = 0.f;
        for (int w = first; w < first + K; ++w)
            if (gm[w] > neg_inf()) {
                const float e = __expf(gm[w] - m_all);
                l_all += gl[w] * e;
                v += gacc[w * D + d] * e;
            }
        const float inv = l_all > 0.f ? 1.f / l_all : 0.f;
        o[(t_row * H + static_cast<int>(blockIdx.y) * G + gg) * D + d] = from_f32<T>(v * inv);
    }
}

}  // namespace pie::attn::sdpa_selected
