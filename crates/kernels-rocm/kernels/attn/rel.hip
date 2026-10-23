#pragma once

// Paged attention under a learned relative-position bias (Inkling): s =
// q·k·sm_scale + bias[t][h][p − j] (inside its extent), times 1 + log_alpha·
// ln((p + 1) / log_floor) past the floor, softmaxed over the keys the row
// sees. A wavefront a (row, head), keys 64 a tile: a lane scores a key, then
// each lane accumulates its columns with the key's weight broadcast.

#include "prim/device.hip"
#include "attn/kv.hip"
#include "attn/mla.hip"

namespace pie::attn {

constexpr int kRelWaves = 4;
constexpr int kRelMaxDim = 256;

template <bool HND>
__global__ __launch_bounds__(kRelWaves* kWave) void paged_rel(
    const bf16* __restrict__ queries, const bf16* __restrict__ k_pages, const bf16* __restrict__ v_pages,
    const i32* __restrict__ qo_indptr, const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, const float* __restrict__ bias, bf16* __restrict__ o, i32 requests,
    i32 H, i32 KVH, i32 D, i32 page_size, float sm_scale, i32 window_left, i32 extent, i32 log_floor,
    float log_alpha, const u32* __restrict__ win) {
    constexpr int PER = kRelMaxDim / kWave;
    __shared__ float qs[kRelWaves][kRelMaxDim];
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    requests = window.lanes(requests);
    const int wv = static_cast<int>(threadIdx.x) / kWave, l = lane();
    const int h = static_cast<int>(blockIdx.y) * kRelWaves + wv;
    if (h >= H) return;
    const int kv_head = h / (H / KVH);

    const int r = request_of(qo_indptr, requests, t);
    const int kv_len = paged_len(page_indptr, last_page_lens, r, page_size);
    const int pos = kv_len - (qo_indptr[r + 1] - qo_indptr[r]) + (t - qo_indptr[r]);
    const int hi = min(pos, kv_len - 1);
    const int lo = window_left >= 0 ? max(0, pos - window_left) : 0;
    const i32* pages = page_indices + page_indptr[r];

    const bf16* qrow = queries + (t_row * H + h) * D;
    for (int d = l; d < D; d += kWave) qs[wv][d] = to_f32(qrow[d]);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

    const float* brow = bias + (t_row * H + h) * static_cast<long long>(extent);
    const float grow = log_alpha != 0.f && log_floor != 0 && pos + 1 > log_floor
                           ? 1.f + log_alpha * logf(static_cast<float>(pos + 1) / static_cast<float>(log_floor))
                           : 1.f;

    float m = neg_inf(), lsum = 0.f, acc[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) acc[i] = 0.f;
    for (int j0 = lo; j0 <= hi; j0 += kWave) {
        const int j = j0 + l;
        float s = neg_inf();
        if (j <= hi) {
            const bf16* kr = k_pages + kv_at<HND>(pages[j / page_size], j % page_size, kv_head, 0, page_size, KVH, D);
            float dot = 0.f;
            for (int d = 0; d < D; d += 8) {
                const Pack8<bf16> kp = *reinterpret_cast<const Pack8<bf16>*>(kr + d);
#pragma unroll
                for (int e = 0; e < 8; ++e) dot += qs[wv][d + e] * to_f32(kp.v[e]);
            }
            s = dot * sm_scale;
            const int back = pos - j;
            if (back < extent) s += brow[back];
            s *= grow;
        }
        float mt = s;
#pragma unroll
        for (int off = kWave / 2; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor(mt, off, kWave));
        const float m_new = fmaxf(m, mt);
        const float corr = expf(m - m_new);
        const float p = j <= hi ? expf(s - m_new) : 0.f;
        float ps = p;
#pragma unroll
        for (int off = kWave / 2; off > 0; off >>= 1) ps += __shfl_xor(ps, off, kWave);
        lsum = lsum * corr + ps;
#pragma unroll
        for (int i = 0; i < PER; ++i) acc[i] *= corr;
        const int n = min(kWave, hi - j0 + 1);
        for (int jj = 0; jj < n; ++jj) {
            const float pj = __shfl(p, jj, kWave);
            const int key = j0 + jj;
            const bf16* vr = v_pages + kv_at<HND>(pages[key / page_size], key % page_size, kv_head, 0, page_size, KVH, D);
#pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int d = l + kWave * i;
                if (d < D) acc[i] += pj * to_f32(vr[d]);
            }
        }
        m = m_new;
    }
    const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
    bf16* orow = o + (t_row * H + h) * D;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int d = l + kWave * i;
        if (d < D) orow[d] = from_f32<bf16>(acc[i] * inv);
    }
}

}  // namespace pie::attn
