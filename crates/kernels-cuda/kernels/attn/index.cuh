#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

constexpr int kBlock = 256;

constexpr int kMaxRopeDim = 256;

__device__ __forceinline__ void rope_interleave_inplace(
    float* v, int rope_dim, int pos, float theta)
{
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = powf(theta, -2.f * static_cast<float>(i) /
                                       static_cast<float>(rope_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = v[2 * i];
        const float b = v[2 * i + 1];
        v[2 * i]     = a * c - b * s;
        v[2 * i + 1] = b * c + a * s;
    }
}

template <class T>
__global__ void index_knorm_rope(
    T* __restrict__ idx_k,
    const T* __restrict__ w,
    const T* __restrict__ b,
    const i32* __restrict__ positions,
    i32 head_dim, i32 rope_dim, float theta, float eps,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns — the key
    // row and the position that dates it are both read there.
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = static_cast<int>(threadIdx.x);
    T* row = idx_k + static_cast<long long>(n_row) * head_dim;

    __shared__ float red[kBlock];
    float s = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) s += Elem<T>::to_f32(row[d]);
    red[tid] = s; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float mean = red[0] / head_dim;
    __syncthreads();
    float vv = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) { float x = Elem<T>::to_f32(row[d]) - mean; vv += x * x; }
    red[tid] = vv; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / head_dim + eps);
    __syncthreads();
    for (int d = tid; d < head_dim; d += kBlock) {
        const float x = (Elem<T>::to_f32(row[d]) - mean) * inv;
        row[d] = Elem<T>::from_f32(x * Elem<T>::to_f32(w[d]) + Elem<T>::to_f32(b[d]));
    }
    __syncthreads();
    if (tid == 0) {
        float buf[kMaxRopeDim];
        for (int d = 0; d < rope_dim; ++d) buf[d] = Elem<T>::to_f32(row[d]);
        rope_interleave_inplace(buf, rope_dim, positions[n_row], theta);
        for (int d = 0; d < rope_dim; ++d) row[d] = Elem<T>::from_f32(buf[d]);
    }
}

template <class T>
__global__ void index_q_rope(
    T* __restrict__ idx_q,
    const i32* __restrict__ positions,
    i32 n_heads, i32 head_dim, i32 rope_dim, float theta,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns — the query
    // row and the position that dates it are both read there.
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = static_cast<int>(threadIdx.x);
    if (h >= n_heads) return;
    T* row = idx_q + (static_cast<long long>(n_row) * n_heads + h) * head_dim;
    float buf[kMaxRopeDim];
    for (int d = 0; d < rope_dim; ++d) buf[d] = Elem<T>::to_f32(row[d]);
    rope_interleave_inplace(buf, rope_dim, positions[n_row], theta);
    for (int d = 0; d < rope_dim; ++d) row[d] = Elem<T>::from_f32(buf[d]);
}

template <class T>
__global__ void index_topk_mask(
    const T* __restrict__ idx_q,
    const T* __restrict__ idx_k,
    const T* __restrict__ idx_w,
    u8* __restrict__ mask,
    i32 N, i32 H, i32 D, i32 topk)
{
    const int i = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    u8* mrow = mask + static_cast<long long>(i) * N;
    const int nkeys = i + 1;

    for (int j = nkeys + tid; j < N; j += kBlock) mrow[j] = 0;

    if (nkeys <= topk) {
        for (int j = tid; j < nkeys; j += kBlock) mrow[j] = 1;
        return;
    }

    extern __shared__ float logit[];
    const T* qi = idx_q + static_cast<long long>(i) * H * D;
    const T* wi = idx_w + static_cast<long long>(i) * H;
    for (int j = tid; j < nkeys; j += kBlock) {
        const T* kj = idx_k + static_cast<long long>(j) * D;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const T* qh = qi + static_cast<long long>(h) * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += Elem<T>::to_f32(qh[d]) * Elem<T>::to_f32(kj[d]);
            acc += fmaxf(dot, 0.f) * Elem<T>::to_f32(wi[h]);
        }
        logit[j] = acc;
    }
    __syncthreads();

    __shared__ float lo_s, hi_s;
    if (tid == 0) {
        float lo = pos_inf(), hi = -pos_inf();
        for (int j = 0; j < nkeys; ++j) { lo = fminf(lo, logit[j]); hi = fmaxf(hi, logit[j]); }
        lo_s = lo; hi_s = hi;
    }
    __syncthreads();
    float lo = lo_s, hi = hi_s;
    __shared__ int cnt_s;
    float thr = hi;
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (tid == 0) cnt_s = 0;
        __syncthreads();
        int c = 0;
        for (int j = tid; j < nkeys; j += kBlock) if (logit[j] >= mid) c++;
        atomicAdd(&cnt_s, c);
        __syncthreads();
        const int cnt = cnt_s;
        if (cnt > topk) lo = mid; else hi = mid;
        __syncthreads();
        thr = hi;
    }
    for (int j = tid; j < nkeys; j += kBlock) mrow[j] = (logit[j] >= thr) ? 1 : 0;
}

template <class T>
__global__ void index_topk_paged(
    const T* __restrict__ idx_q,
    const T* __restrict__ idx_w,
    const T* __restrict__ key_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    i32* __restrict__ selection,
    i32 R, i32 H, i32 D, i32 page_size, i32 score_stride, i32 topk,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Only the
    // PLANES move by it — the qo boundaries below are the window's own,
    // rebased to its zero, and the score scratch starts at its own too.
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    // `R` may be the key's lane ceiling; `win[2]` is the live request count.
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    i32* srow = selection + static_cast<long long>(t_row) * topk;

    int r = 0;
    for (; r < R - 1; ++r) {
        if (t < static_cast<int>(qo_indptr[r + 1])) break;
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    int nkeys = abs_q + 1;

    if (nkeys > score_stride) nkeys = score_stride;
    if (nkeys < 0) nkeys = 0;

    float* frow = scores + static_cast<long long>(t) * score_stride;
    const T* qi = idx_q + static_cast<long long>(t_row) * H * D;
    const T* wi = idx_w + static_cast<long long>(t_row) * H;
    for (int j = tid; j < nkeys; j += kBlock) {
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const T* kj =
            key_pages + (static_cast<long long>(page) * page_size + off) * D;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const T* qh = qi + static_cast<long long>(h) * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += Elem<T>::to_f32(qh[d]) * Elem<T>::to_f32(kj[d]);
            acc += fmaxf(dot, 0.f) * Elem<T>::to_f32(wi[h]);
        }
        frow[j] = acc;
    }
    __syncthreads();

    if (nkeys <= topk) {
        for (int n = tid; n < topk; n += kBlock) srow[n] = (n < nkeys) ? n : -1;
        return;
    }

    __shared__ float red[kBlock];
    float lo_l = pos_inf(), hi_l = -pos_inf();
    for (int j = tid; j < nkeys; j += kBlock) {
        lo_l = fminf(lo_l, frow[j]);
        hi_l = fmaxf(hi_l, frow[j]);
    }
    red[tid] = lo_l; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fminf(red[tid], red[tid + o]); __syncthreads(); }
    float lo = red[0];
    __syncthreads();
    red[tid] = hi_l; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    float hi = red[0];
    __syncthreads();

    __shared__ int cnt_s;
    float thr = hi;
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (tid == 0) cnt_s = 0;
        __syncthreads();
        int c = 0;
        for (int j = tid; j < nkeys; j += kBlock) if (frow[j] >= mid) c++;
        atomicAdd(&cnt_s, c);
        __syncthreads();
        const int cnt = cnt_s;
        if (cnt > topk) lo = mid; else hi = mid;
        __syncthreads();
        thr = hi;
    }

    if (tid == 0) {
        int n = 0;
        for (int j = 0; j < nkeys && n < topk; ++j) {
            if (frow[j] >= thr) srow[n++] = j;
        }
        for (; n < topk; ++n) srow[n] = -1;
    }
}

}
