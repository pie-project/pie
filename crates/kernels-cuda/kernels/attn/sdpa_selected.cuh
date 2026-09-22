#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <math_constants.h>

#include "prelude/device.cuh"

namespace pie::attn::sdpa_selected {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kMaxPer = 16;

// Paged GQA attention over the blocks a selection names, plus the open block
// that runs to the query. `attention.decode`/`attention.prefill` are FlashInfer
// and take no per-row selection, so this walk is written out: the reduction is
// `mla_naive_paged_kernel`'s, and only the key loop and the K/V addressing
// differ (two planes at `(slot * n_kv_heads + kv_head) * D`, V accumulated
// rather than K).
template <class T>
__global__ void sdpa_paged_selected_kernel(
    const T* __restrict__ queries,
    const T* __restrict__ k_pages,
    const T* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const i32* __restrict__ selection,
    T* __restrict__ o,
    i32 R, i32 H, i32 KVH, i32 D, i32 page_size, float sm_scale,
    i32 top_k, i32 ratio, i32 G,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = kWarps / G;
    const int g = warp / K;
    const int s = warp % K;
    const int h = static_cast<int>(blockIdx.y) * G + g;
    const int kv_head = h / (H / KVH);
    const int per = D / 32;

    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid; else lo = mid + 1;
    }
    const int r = lo;
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int j_end = kv_len - new_tokens + (t - qo_lo) + 1;

    const int nblocks = (ratio > 0) ? j_end / ratio : 0;
    const int blocks = min(top_k, nblocks);
    const i32* srow = selection + static_cast<long long>(t_row) * top_k;

    extern __shared__ float smem[];
    float* wacc = smem;
    float* wm   = wacc + kWarps * D;
    float* wl   = wm + kWarps;

    const T* qrow = queries + (static_cast<long long>(t_row) * H + h) * D;
    float qreg[kMaxPer];
    for (int i = 0; i < per; ++i) {
        qreg[i] = Elem<T>::to_f32(qrow[lane + i * 32]) * sm_scale;
    }

    float acc[kMaxPer];
    for (int i = 0; i < per; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, lsum = 0.f;

    // A block below `nblocks` holds only cells below `nblocks * ratio <= j_end`,
    // so rejecting the block is the whole causal test.
    const int steps = blocks * ratio + (ratio - 1);
    for (int n = s; n < steps; n += K) {
        int j;
        if (n < blocks * ratio) {
            const int c = srow[n / ratio];
            if (c < 0 || c >= nblocks) continue;
            j = c * ratio + (n % ratio);
        } else {
            j = nblocks * ratio + (n - blocks * ratio);
            if (j >= j_end) continue;
        }
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const long long slot = static_cast<long long>(page) * page_size + off;
        const T* kj = k_pages + (slot * KVH + kv_head) * D;
        const T* vj = v_pages + (slot * KVH + kv_head) * D;

        float pd = 0.f;
        for (int i = 0; i < per; ++i) {
            pd += qreg[i] * Elem<T>::to_f32(kj[lane + i * 32]);
        }
        #pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) {
            pd += __shfl_xor_sync(0xffffffffu, pd, sh);
        }
        const float m_new = fmaxf(m, pd);
        const float corr = __expf(m - m_new);
        const float p = __expf(pd - m_new);
        lsum = lsum * corr + p;
        for (int i = 0; i < per; ++i) {
            acc[i] = acc[i] * corr + p * Elem<T>::to_f32(vj[lane + i * 32]);
        }
        m = m_new;
    }

    for (int i = 0; i < per; ++i) {
        wacc[warp * D + lane + i * 32] = acc[i];
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = lsum; }
    __syncthreads();

    const int total_out = G * D;
    for (int idx = tid; idx < total_out; idx += kBlock) {
        const int gg = idx / D;
        const int d = idx % D;
        const int w0 = gg * K;
        float m_all = -CUDART_INF_F;
        for (int w = w0; w < w0 + K; ++w) m_all = fmaxf(m_all, wm[w]);
        float l_all = 0.f, v = 0.f;
        for (int w = w0; w < w0 + K; ++w) {
            if (wm[w] > -CUDART_INF_F) {
                const float e = __expf(wm[w] - m_all);
                l_all += wl[w] * e;
                v += wacc[w * D + d] * e;
            }
        }
        const float inv = (l_all > 0.f) ? (1.f / l_all) : 0.f;
        o[(static_cast<long long>(t_row) * H + static_cast<int>(blockIdx.y) * G + gg) * D + d] =
            Elem<T>::from_f32(v * inv);
    }
}

}
