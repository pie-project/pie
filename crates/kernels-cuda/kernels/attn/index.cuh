#pragma once

#include <cstdint>

#include "prelude/device.cuh"
#include "prelude/rope.cuh"

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

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

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

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

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

// The `topk` best of a row's `nkeys` scores, ascending, `-1` past the end:
// the threshold the counts bisect to, then the scores at or above it in order.
__device__ __forceinline__ void select_top(const float* __restrict__ frow, int nkeys, int topk,
                                           i32* __restrict__ srow, int tid)
{
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

// `select_top` by radix: the `topk` largest of `nkeys` scores, ascending by
// index, `-1` past the end. The k-th largest score's bits are found a byte a
// pass (a 256-bin histogram of the survivors, most significant byte first),
// then every score above it and the first ties at it (in index order) take
// their places by a block scan. Four passes over the scores, where the
// bisection walks forty.
__device__ __forceinline__ u32 order_bits(float v) {
    const u32 b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Block-wide exclusive prefix of `flag`, and the block's total.
__device__ __forceinline__ int block_prefix(bool flag, int tid, int* warp_sums, int& total) {
    const int lane = tid & 31, warp = tid >> 5;
    const unsigned ballot = __ballot_sync(0xffffffffu, flag);
    const int before = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) warp_sums[warp] = __popc(ballot);
    __syncthreads();
    int base = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < kBlock / 32; ++w) {
        const int c = warp_sums[w];
        if (w < warp) base += c;
        total += c;
    }
    __syncthreads();
    return base + before;
}

__device__ __forceinline__ void select_top_radix(const float* __restrict__ frow, int nkeys,
                                                 int topk, i32* __restrict__ srow, int tid)
{
    __shared__ int hist[256];
    __shared__ int s_digit, s_need;
    __shared__ int warp_sums[kBlock / 32];
    u32 prefix = 0, mask = 0;
    int need = topk;  // how many of the survivors (bits matching prefix under mask) to take
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += kBlock) hist[i] = 0;
        __syncthreads();
        // Scores crowd a few bins (the top byte is sign and exponent): a
        // warp adds each bin once, its lanes in that bin counted together.
        for (int j0 = 0; j0 < nkeys; j0 += kBlock) {
            const int j = j0 + tid;
            const u32 u = j < nkeys ? order_bits(frow[j]) : 0u;
            const bool in = j < nkeys && (u & mask) == prefix;
            const int bin = in ? static_cast<int>((u >> shift) & 255) : -1;
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
            if (in && (tid & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], __popc(peers));
        }
        __syncthreads();
        if (tid < 32) {
            // From the top bin down, the bin where the count reaches `need`.
            int run = 0;
            int found = -1, found_need = 0;
            // Each lane holds eight bins, highest first across the warp.
            int c[8];
            int lane_sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c[i] = hist[255 - (tid * 8 + i)];
                lane_sum += c[i];
            }
            int incl = lane_sum;
#pragma unroll
            for (int sh = 1; sh < 32; sh <<= 1) {
                const int x = __shfl_up_sync(0xffffffffu, incl, sh);
                if (tid >= sh) incl += x;
            }
            run = incl - lane_sum;
            if (run < need && incl >= need) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (found < 0 && run + c[i] >= need) {
                        found = 255 - (tid * 8 + i);
                        found_need = need - run;
                    }
                    run += c[i];
                }
                s_digit = found;
                s_need = found_need;
            }
        }
        __syncthreads();
        prefix |= static_cast<u32>(s_digit) << shift;
        mask |= 255u << shift;
        need = s_need;
        __syncthreads();
    }
    // `prefix` is the k-th largest score's bits; `need` of its ties are taken.
    int written = 0, ties_seen = 0;
#pragma unroll 1
    for (int j0 = 0; j0 < nkeys; j0 += kBlock) {
        const int j = j0 + tid;
        const u32 u = j < nkeys ? order_bits(frow[j]) : 0u;
        const bool tie = j < nkeys && u == prefix;
        int ties;
        const int tie_rank = ties_seen + block_prefix(tie, tid, warp_sums, ties);
        ties_seen += ties;
        const bool take = j < nkeys && (u > prefix || (tie && tie_rank < need));
        int took;
        const int at = written + block_prefix(take, tid, warp_sums, took);
        if (take) srow[at] = j;
        written += took;
    }
    for (int n = written + tid; n < topk; n += kBlock) srow[n] = -1;
}

// A pooled selection read back as tokens: each selected block `b` (of
// `ratio` rows, -1 none) its rows `b * ratio ..`, then the row's own open
// block's rows (past its last closed block, up to and with the row), the
// rest -1; `topk * ratio + ratio - 1` a row.
__global__ void index_pool_tokens(
    const i32* __restrict__ blocks,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    i32* __restrict__ tokens,
    i32 R, i32 page_size, i32 topk, i32 ratio,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    int r = 0;
    for (; r < R - 1; ++r) {
        if (t < static_cast<int>(qo_indptr[r + 1])) break;
    }
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - static_cast<int>(kv_page_indptr[r]);
    const int kv_len = (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    const int open_from = (abs_q + 1) / ratio * ratio;
    const i32* brow = blocks + static_cast<long long>(t_row) * topk;
    const int width = topk * ratio + ratio - 1;
    i32* trow = tokens + static_cast<long long>(t_row) * width;
    for (int n = static_cast<int>(threadIdx.x); n < width; n += static_cast<int>(blockDim.x)) {
        int j = -1;
        if (n < topk * ratio) {
            const int b = brow[n / ratio];
            if (b >= 0) j = b * ratio + n % ratio;
        } else {
            const int at = open_from + (n - topk * ratio);
            if (at <= abs_q) j = at;
        }
        trow[n] = j;
    }
}

template <class T, bool kRadix = false>
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
    i32 ratio,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

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

    const int stride = (ratio > 0) ? ratio : 1;
    const int total = abs_q + 1;
    int nkeys = (total > 0) ? total / stride : 0;
    if (nkeys > score_stride) nkeys = score_stride;

    // A row that sees no more blocks than the budget selects them all: its
    // scores would rank nothing, so none are taken.
    if (nkeys <= topk) {
        for (int n = tid; n < topk; n += kBlock) srow[n] = (n < nkeys) ? n : -1;
        return;
    }

    float* frow = scores + static_cast<long long>(t) * score_stride;
    const T* qi = idx_q + static_cast<long long>(t_row) * H * D;
    const T* wi = (idx_w != nullptr) ? idx_w + static_cast<long long>(t_row) * H : nullptr;
    // The row's query, read once into shared memory where every key's
    // thread reads it (the same values, in the same order).
    constexpr int kStagedQuery = 1024;
    __shared__ float q_s[kStagedQuery];
    const bool staged = H * D <= kStagedQuery;
    if (staged) {
        for (int i = tid; i < H * D; i += kBlock) q_s[i] = Elem<T>::to_f32(qi[i]);
    }
    __syncthreads();
    for (int j = tid; j < nkeys; j += kBlock) {
        const int cell = (j + 1) * stride - 1;
        const int page =
            static_cast<int>(kv_page_indices[pages_first + cell / page_size]);
        const int off = cell % page_size;
        const T* kj =
            key_pages + (static_cast<long long>(page) * page_size + off) * D;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const T* qh = qi + static_cast<long long>(h) * D;
            float dot = 0.f;
            if (staged) {
                for (int d = 0; d < D; ++d) dot += q_s[h * D + d] * Elem<T>::to_f32(kj[d]);
            } else {
                for (int d = 0; d < D; ++d) dot += Elem<T>::to_f32(qh[d]) * Elem<T>::to_f32(kj[d]);
            }
            acc += fmaxf(dot, 0.f) * (idx_w != nullptr ? Elem<T>::to_f32(wi[h]) : 1.f);
        }
        frow[j] = acc;
    }
    __syncthreads();

    if constexpr (kRadix) {
        select_top_radix(frow, nkeys, topk, srow, tid);
    } else {
        select_top(frow, nkeys, topk, srow, tid);
    }
}

// `index_topk_paged` with the scores on the tensor cores: the row's heads
// are an mma's rows, `kTiles` tiles of 16 (`H` <= 16 kTiles, the rest zero),
// and each warp scores its own
// 16-block chunks of the row's keys (`D` = 128), gathered into shared memory
// a chunk at a time. The ranking that follows is the same.
namespace index_mma {

constexpr int kD = 128;
constexpr int kLd = kD + 8;
constexpr int kWarps = kBlock / 32;
constexpr int kKeys = 16;

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                             const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace index_mma

template <class T, int kTiles = 1, bool kRadix = false>
__global__ void index_topk_paged_mma(
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
    i32 ratio,
    const i32* __restrict__ turn_pos,
    i32 turn_dim,
    float turn_theta,
    float turn_factor,
    float turn_low,
    float turn_high,
    const u32* __restrict__ win)
{
    using namespace index_mma;
    const int t = static_cast<int>(blockIdx.x);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
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

    const int stride = (ratio > 0) ? ratio : 1;
    const int total = abs_q + 1;
    int nkeys = (total > 0) ? total / stride : 0;
    if (nkeys > score_stride) nkeys = score_stride;

    if (nkeys <= topk) {
        for (int n = tid; n < topk; n += kBlock) srow[n] = (n < nkeys) ? n : -1;
        return;
    }

    float* frow = scores + static_cast<long long>(t) * score_stride;
    const T* qi = idx_q + static_cast<long long>(t_row) * H * kD;
    // The row's query (16 kTiles rows of kLd) and each warp's key chunk, in
    // dynamic shared memory.
    extern __shared__ __align__(16) unsigned char q_raw[];
    T* q_s = reinterpret_cast<T*>(q_raw);
    T* k_s0 = q_s + 16 * kTiles * kLd;
    __shared__ float w_s[16 * kTiles];
    __shared__ long long slot_s[kWarps][kKeys];
    for (int i = tid; i < 16 * kTiles * (kD / 8); i += kBlock) {
        const int row = i / (kD / 8);
        const int d = (i % (kD / 8)) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < H) v = *reinterpret_cast<const int4*>(qi + row * kD + d);
        // The query's tail turned as it lands (`rope_partial_last`, forward,
        // interleaved, at the row's position), where the ranking holds it.
        if (turn_dim > 0 && d >= kD - turn_dim) {
            T* e = reinterpret_cast<T*>(&v);
            const float pos = static_cast<float>(turn_pos[t_row]);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int pair = (d - (kD - turn_dim)) / 2 + u;
                float freq = powf(turn_theta, -2.f * static_cast<float>(pair) / static_cast<float>(turn_dim));
                if (turn_factor > 1.f) freq = ::pie::yarn_original_freq(freq, turn_factor, turn_low, turn_high, pair);
                float sn, cs;
                __sincosf(pos * freq, &sn, &cs);
                const float a = Elem<T>::to_f32(e[2 * u]);
                const float b = Elem<T>::to_f32(e[2 * u + 1]);
                e[2 * u] = Elem<T>::from_f32(a * cs - b * sn);
                e[2 * u + 1] = Elem<T>::from_f32(b * cs + a * sn);
            }
        }
        *reinterpret_cast<int4*>(q_s + row * kLd + d) = v;
    }
    for (int h = tid; h < 16 * kTiles; h += kBlock) {
        w_s[h] = h < H
            ? (idx_w != nullptr ? Elem<T>::to_f32(idx_w[static_cast<long long>(t_row) * H + h]) : 1.f)
            : 0.f;
    }
    __syncthreads();

    const int g = lane >> 2, p = (lane & 3) << 1;
    T* ks = k_s0 + warp * kKeys * kLd;
    for (int base = warp * kKeys; base < nkeys; base += kWarps * kKeys) {
        if (lane < kKeys) {
            const int j = base + lane;
            long long slot = -1;
            if (j < nkeys) {
                const int cell = (j + 1) * stride - 1;
                slot = static_cast<long long>(kv_page_indices[pages_first + cell / page_size]) * page_size
                    + cell % page_size;
            }
            slot_s[warp][lane] = slot;
        }
        __syncwarp();
        for (int c = lane; c < kKeys * (kD / 8); c += 32) {
            const int key = c / (kD / 8);
            const int d = (c % (kD / 8)) * 8;
            const long long slot = slot_s[warp][key];
            *reinterpret_cast<int4*>(ks + key * kLd + d) = slot >= 0
                ? *reinterpret_cast<const int4*>(key_pages + slot * kD + d)
                : make_int4(0, 0, 0, 0);
        }
        __syncwarp();
        // A key's score, this lane's share: each head tile's weighted
        // rectified dots, rows `g` and `g + 8` of the tile.
        float part[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
        for (int ht = 0; ht < kTiles; ++ht) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int k0 = 0; k0 < kD; k0 += 16) {
            std::uint32_t a[4];
            {
                const T* rp = q_s + (ht * 16 + (lane & 15)) * kLd + k0 + ((lane & 16) ? 8 : 0);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                             : "r"(smem_addr(rp)));
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                std::uint32_t b[2];
                const T* rp = ks + (nt * 8 + (lane & 7)) * kLd + k0 + ((lane & 8) ? 8 : 0);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(b[0]), "=r"(b[1])
                             : "r"(smem_addr(rp)));
                mma_m16n8k16(acc[nt], a, b);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                part[nt][i] += fmaxf(acc[nt][i], 0.f) * w_s[ht * 16 + g]
                    + fmaxf(acc[nt][2 + i], 0.f) * w_s[ht * 16 + g + 8];
        }
        // A key's score sums its heads' weighted rectified dots, across the
        // eight lane groups.
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float v = part[nt][i];
#pragma unroll
                for (int sh = 4; sh < 32; sh <<= 1) v += __shfl_xor_sync(0xffffffffu, v, sh);
                const int j = base + nt * 8 + p + i;
                if (g == 0 && j < nkeys) frow[j] = v;
            }
        }
        __syncwarp();
    }
    __syncthreads();

    if constexpr (kRadix) {
        select_top_radix(frow, nkeys, topk, srow, tid);
    } else {
        select_top(frow, nkeys, topk, srow, tid);
    }
}

constexpr int kMaxIndexHeads = 8;
constexpr int kMaxIndexPer = 8;

// A row's ranking of the blocks it sees, one per head: a block scores the
// best `q_h·k` among its keys at or before the row; the row's own block is
// always kept, the other `topk - 1` are the best of the rest, and the run is
// written ascending, `-1` past its end. Warps score the blocks (a lane holds
// `D / 32` of each head's query); warp `h` then ranks head `h`.
template <class T>
__global__ void index_topk_block_max(
    const T* __restrict__ idx_q,
    const T* __restrict__ key_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    i32* __restrict__ selection,
    i32 R, i32 H, i32 D, i32 page_size, i32 score_stride, i32 topk,
    i32 block,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kWarps = kBlock / 32;
    i32* srow = selection + static_cast<long long>(t_row) * H * topk;

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
    if (abs_q < 0) {
        for (int n = tid; n < H * topk; n += kBlock) srow[n] = -1;
        return;
    }

    int nb = abs_q / block + 1;
    if (nb > score_stride) nb = score_stride;
    if (nb <= topk) {
        for (int n = tid; n < H * topk; n += kBlock) {
            const int at = n % topk;
            srow[n] = (at < nb) ? at : -1;
        }
        return;
    }

    const int per = D / 32;
    float qreg[kMaxIndexHeads][kMaxIndexPer];
    const T* qi = idx_q + static_cast<long long>(t_row) * H * D;
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < per; ++i) {
            qreg[h][i] = Elem<T>::to_f32(qi[h * D + lane + i * 32]);
        }
    }

    float* frow = scores + static_cast<long long>(t) * H * score_stride;
    for (int b = warp; b < nb; b += kWarps) {
        float best[kMaxIndexHeads];
        for (int h = 0; h < H; ++h) best[h] = neg_inf();
        const int lo = b * block;
        const int hi = min(lo + block, abs_q + 1);
        for (int j = lo; j < hi; ++j) {
            const int page =
                static_cast<int>(kv_page_indices[pages_first + j / page_size]);
            const T* kj = key_pages +
                (static_cast<long long>(page) * page_size + j % page_size) * D;
            float kv[kMaxIndexPer];
            for (int i = 0; i < per; ++i) kv[i] = Elem<T>::to_f32(kj[lane + i * 32]);
            for (int h = 0; h < H; ++h) {
                float pd = 0.f;
                for (int i = 0; i < per; ++i) pd += qreg[h][i] * kv[i];
                #pragma unroll
                for (int sh = 16; sh > 0; sh >>= 1) {
                    pd += __shfl_xor_sync(0xffffffffu, pd, sh);
                }
                best[h] = fmaxf(best[h], pd);
            }
        }
        if (lane == 0) {
            for (int h = 0; h < H; ++h) frow[h * score_stride + b] = best[h];
        }
    }
    __syncthreads();

    for (int h = warp; h < H; h += kWarps) {
        float* hrow = frow + h * score_stride;
        i32* out = srow + h * topk;
        const int own = nb - 1;
        if (lane == 0) hrow[own] = neg_inf();
        __syncwarp();
        int picked = 0;
        for (int n = 0; n < topk - 1; ++n) {
            float v = neg_inf();
            int at = nb;
            for (int b = lane; b < nb; b += 32) {
                const float s = hrow[b];
                if (s > v || (s == v && b < at && s > neg_inf())) { v = s; at = b; }
            }
            #pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, sh);
                const int oa = __shfl_xor_sync(0xffffffffu, at, sh);
                if (ov > v || (ov == v && oa < at)) { v = ov; at = oa; }
            }
            if (at >= nb) break;
            if (lane == 0) {
                out[picked] = at;
                hrow[at] = neg_inf();
            }
            ++picked;
            __syncwarp();
        }
        if (lane == 0) {
            out[picked++] = own;
            for (int a = 1; a < picked; ++a) {
                const int key = out[a];
                int c = a - 1;
                while (c >= 0 && out[c] > key) { out[c + 1] = out[c]; --c; }
                out[c + 1] = key;
            }
            for (int n = picked; n < topk; ++n) out[n] = -1;
        }
    }
}

}
