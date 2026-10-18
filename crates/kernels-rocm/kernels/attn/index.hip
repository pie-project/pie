#pragma once

// The sparse-attention indexer (DSA / CSA): its query and key rotations, and
// the ranking that picks, per query row, the top-k cached keys it attends
// over. A key's score is Σ_h w_h · relu(q_h · k): every indexer head (one
// weight each) against the one shared key row, read from a paged key pool
// that keeps one row per `ratio` tokens (cell (j + 1)·ratio − 1 for key j).

#include "prelude/device.hip"
#include "prelude/mfma.hip"
#include "attn/mla.hip"

namespace pie::attn {

// Interleaved rotary pair i of a `rope_dim`-wide head at position `pos`:
// theta^(−2i/rope_dim) radians per position.
__device__ __forceinline__ void index_rotation(int pos, int i, int rope_dim, float theta, float& c,
                                               float& s) {
    const float freq = powf(theta, -2.f * static_cast<float>(i) / static_cast<float>(rope_dim));
    sincosf(static_cast<float>(pos) * freq, &s, &c);
}

// The first rope_dim elements of every head of q turn as interleaved pairs.
// One block per token row.
__global__ void index_q_rope(bf16* __restrict__ q, const i32* __restrict__ positions, i32 n_heads, i32 head_dim,
                             i32 rope_dim, float theta, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int half = rope_dim / 2;
    const int pos = positions[n];
    for (int i = threadIdx.x; i < n_heads * half; i += blockDim.x) {
        const int h = i / half, p = i % half;
        bf16* at = q + (n * n_heads + h) * head_dim + 2 * p;
        float c, s;
        index_rotation(pos, p, rope_dim, theta, c, s);
        const float a = to_f32(at[0]), b = to_f32(at[1]);
        at[0] = from_f32<bf16>(a * c - b * s);
        at[1] = from_f32<bf16>(b * c + a * s);
    }
}

// The index key row LayerNormed (weight, bias) in place, then its first
// rope_dim elements turned as interleaved pairs of the normed (rounded)
// values. One block per row, two elements a thread step.
__global__ __launch_bounds__(256) void index_knorm_rope(bf16* __restrict__ k, const bf16* __restrict__ w,
                                                        const bf16* __restrict__ b, const i32* __restrict__ positions,
                                                        i32 head_dim, i32 rope_dim, float theta, float eps,
                                                        const u32* __restrict__ win) {
    constexpr int BLOCK = 256;
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    bf16* row = k + n * head_dim;
    float sum = 0.f;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) sum += to_f32(row[d]);
    const float mean = block_sum(sum, smem) / static_cast<float>(head_dim);
    float var = 0.f;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
        const float x = to_f32(row[d]) - mean;
        var += x * x;
    }
    const float inv = rsqrtf(block_sum(var, smem) / static_cast<float>(head_dim) + eps);
    auto norm = [&](int d) {
        const float x = (to_f32(row[d]) - mean) * inv;
        const float y = x * (w != nullptr ? to_f32(w[d]) : 1.f) + (b != nullptr ? to_f32(b[d]) : 0.f);
        return to_f32(from_f32<bf16>(y));
    };
    const int pos = positions[n];
    for (int p = threadIdx.x; 2 * p < head_dim; p += BLOCK) {
        const int d = 2 * p;
        float y0 = norm(d);
        float y1 = d + 1 < head_dim ? norm(d + 1) : 0.f;
        if (d + 1 < rope_dim) {
            float c, s;
            index_rotation(pos, p, rope_dim, theta, c, s);
            const float a = y0;
            y0 = a * c - y1 * s;
            y1 = y1 * c + a * s;
        }
        row[d] = from_f32<bf16>(y0);
        if (d + 1 < head_dim) row[d + 1] = from_f32<bf16>(y1);
    }
}

// The keys token row t ranks: the visible compressed positions of its
// request, (abs_q + 1) / ratio of them, capped at the score row's width.
struct IndexRow {
    const i32* pages;
    int nkeys;
};

__device__ __forceinline__ IndexRow index_row(const i32* qo_indptr, const i32* page_indices,
                                              const i32* page_indptr, const i32* last_page_lens, int requests,
                                              int page_size, int score_stride, int ratio, int t) {
    const int r = request_of(qo_indptr, requests, t);
    const int kv_len = paged_len(page_indptr, last_page_lens, r, page_size);
    const int abs_q = kv_len - (qo_indptr[r + 1] - qo_indptr[r]) + (t - qo_indptr[r]);
    const int stride = ratio > 0 ? ratio : 1;
    const int total = abs_q + 1;
    return {page_indices + page_indptr[r], min(total > 0 ? total / stride : 0, score_stride)};
}

// ------------------------------------------------------------ scoring --

constexpr int kSummary = 16;  // a scoring block's kept top scores, four a wavefront (see selection)
//
// A block a (256 keys, token row); wavefront w scores keys 64w.. as four
// 16-key MFMA tiles (Sᵀ = K·Qᵀ, Qᵀ in registers), weighs relu(Sᵀ) by each
// head's weight and sums the heads across the lanes that hold them.

template <int D, int NT>
__global__ __launch_bounds__(256) void index_score_paged(
    const bf16* __restrict__ q, const bf16* __restrict__ wts, const bf16* __restrict__ key_pages,
    const i32* __restrict__ qo_indptr, const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, float* __restrict__ scores, i32 requests, i32 heads, i32 page_size,
    i32 score_stride, i32 ratio, i32 compact, float* __restrict__ summary, i32 summary_stride,
    const u32* __restrict__ win) {
    static_assert(D % 32 == 0, "a key row is whole 32-wide k pairs");
    constexpr int KP = D / 32, TILES = 4;
    __shared__ float block_scores[256];
    const Window window{win};
    const int t = blockIdx.y;
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const IndexRow row = index_row(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(requests),
                                   page_size, score_stride, ratio, t);
    const int w = threadIdx.x / kWave, l = lane(), c = l % 16, g = l / 16;
    if (static_cast<int>(blockIdx.x) * 256 >= row.nkeys) return;
    const int base = static_cast<int>(blockIdx.x) * 256 + 64 * w;

    // A compact pool holds its keys a row each, `page_size / ratio` a page;
    // else key j sits at its group's last position.
    const int kps = compact ? page_size / ratio : page_size;
    // The tiles' key rows first: their addresses wait on nothing else.
    mfma_ab2 kr[TILES][KP];
    {
        long long slot[TILES];
#pragma unroll
        for (int kt = 0; kt < TILES; ++kt) {
            const int j = min(base + 16 * kt + c, row.nkeys - 1);
            const int cell = compact ? j : (j + 1) * ratio - 1;
            slot[kt] = static_cast<long long>(row.pages[cell / kps]) * kps + cell % kps;
        }
#pragma unroll
        for (int kt = 0; kt < TILES; ++kt)
#pragma unroll
            for (int st = 0; st < KP; ++st) kr[kt][st] = load8(key_pages + slot[kt] * D + 32 * st + 8 * g);
    }
    mfma_ab2 qf[NT][KP];
    float wt[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int h = 16 * nt + c;
        const bf16* qh = q + (t_row * heads + min(h, heads - 1)) * D;
#pragma unroll
        for (int st = 0; st < KP; ++st) qf[nt][st] = load8(qh + 32 * st + 8 * g);
        wt[nt] = h < heads ? (wts != nullptr ? to_f32(wts[t_row * heads + h]) : 1.f) : 0.f;
    }

    float* frow = scores + static_cast<long long>(t) * score_stride;
#pragma unroll
    for (int kt = 0; kt < TILES; ++kt) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            mfma_cd s = mfma_zero();
#pragma unroll
            for (int st = 0; st < KP; ++st) s = mfma16x2(kr[kt][st], qf[nt][st], s);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += fmaxf(s[i], 0.f) * wt[nt];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) v[i] += __shfl_xor(v[i], off, kWave);
        const int j = base + 16 * kt + 4 * g + c;
        const float score = c == 0 ? v[0] : c == 1 ? v[1] : c == 2 ? v[2] : v[3];
        if (c < 4 && j < row.nkeys) frow[j] = score;
        if (c < 4) block_scores[64 * w + 16 * kt + 4 * g + c] = j < row.nkeys ? score : neg_inf();
    }
    if (summary == nullptr) return;

    // The wavefront's top four of its 64 scores (one a lane, through its own
    // stretch of LDS), four rounds of taking the largest left and striking it.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    float mine = block_scores[64 * w + l];
    float* out = summary + static_cast<long long>(t) * summary_stride +
                 static_cast<long long>(blockIdx.x) * kSummary + 4 * w;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float best = wave_max(mine);
        const u64 holders = __ballot(mine == best);
        if (l == __builtin_ctzll(holders)) mine = neg_inf();
        if (l == 0) out[r] = best;
    }
}

// The scoring for any head width or count: a thread per key.
__global__ __launch_bounds__(256) void index_score_any(
    const bf16* __restrict__ q, const bf16* __restrict__ wts, const bf16* __restrict__ key_pages,
    const i32* __restrict__ qo_indptr, const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, float* __restrict__ scores, i32 requests, i32 heads, i32 head_dim,
    i32 page_size, i32 score_stride, i32 ratio, i32 compact, const u32* __restrict__ win) {
    const Window window{win};
    const int t = blockIdx.y;
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const IndexRow row = index_row(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(requests),
                                   page_size, score_stride, ratio, t);
    const int j = static_cast<int>(blockIdx.x) * 256 + threadIdx.x;
    if (j >= row.nkeys) return;
    const int kps = compact ? page_size / ratio : page_size;
    const int cell = compact ? j : (j + 1) * ratio - 1;
    const bf16* kj = key_pages + (static_cast<long long>(row.pages[cell / kps]) * kps + cell % kps) * head_dim;
    const bf16* qi = q + t_row * heads * head_dim;
    float acc = 0.f;
    for (int h = 0; h < heads; ++h) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) dot += to_f32(qi[h * head_dim + d]) * to_f32(kj[d]);
        acc += fmaxf(dot, 0.f) * (wts != nullptr ? to_f32(wts[t_row * heads + h]) : 1.f);
    }
    scores[static_cast<long long>(t) * score_stride + j] = acc;
}

// ---------------------------------------------------------- selection --
//
// The exact top-k of a score row in ascending key order, ties at the cut to
// the lower keys: radix select on the order-preserving key (11, 11, 10-bit
// passes). A decode's few long rows go wide: each wavefront keeps its 64
// keys' top 4, their k-th largest bounds the cut, the scores past the bound
// are collected and selected among. A row whose candidates overflow the
// list takes the one-block walk over its whole row instead.

constexpr int kSelectThreads = 1024;
constexpr int kSelectWaves = kSelectThreads / kWave;
constexpr int kCandidates = 4096;     // a row's candidate list
constexpr int kBitmapKeys = 1 << 18;  // the widest row the bitmap covers
constexpr int kTies = 1024;           // ties at the cut the bitmap sorts

// Larger float, larger key; −0 below +0, NaN above +inf.
__device__ __forceinline__ u32 order_key(float f) {
    const u32 u = __builtin_bit_cast(u32, f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// An inclusive scan over the wavefront.
__device__ __forceinline__ int wave_inclusive(int v) {
    const int ln = lane();
#pragma unroll
    for (int off = 1; off < kWave; off <<= 1) {
        const int up = __shfl_up(v, off, kWave);
        if (ln >= off) v += up;
    }
    return v;
}

// An exclusive scan over the block of one int a thread; `total` gets the
// sum. `smem` holds one int per wavefront.
template <int THREADS>
__device__ __forceinline__ int block_exclusive(int v, int* smem, int& total) {
    const int wv = threadIdx.x / kWave;
    const int inc = wave_inclusive(v);
    if (lane() == kWave - 1) smem[wv] = inc;
    __syncthreads();
    int before = 0;
    total = 0;
    for (int i = 0; i < THREADS / kWave; ++i) {
        const int s = smem[i];
        if (i < wv) before += s;
        total += s;
    }
    __syncthreads();
    return before + inc - v;
}

typedef float f32x4 __attribute__((ext_vector_type(4)));

// Eight four-score reads, [4i, 4i + 4) for i = i0 + u·step (clamped to
// `last`; past the row's n they are junk the caller masks), all in flight
// before any is used: the empty asm takes them all, so the scheduler cannot
// trade the group for registers by issuing each read just before its use.
// A score row's pitch is a multiple of four, so each is one 16-byte read.
__device__ __forceinline__ void eight(const float* row, int i0, int step, int last, f32x4 (&v)[8]) {
#pragma unroll
    for (int u = 0; u < 8; ++u) v[u] = *reinterpret_cast<const f32x4*>(row + 4 * min(i0 + u * step, last));
    asm volatile("" ::"v"(v[0]), "v"(v[1]), "v"(v[2]), "v"(v[3]), "v"(v[4]), "v"(v[5]), "v"(v[6]), "v"(v[7]));
}

// The k-th largest of a set of order keys: its key `thr`, and how many keys
// equal to it the top k hold (`need`). The set comes in `groups` groups of
// PER keys a thread: `group(g, key, ok)` fills thread's keys of group g and
// whether each is in the set. `hist` holds 2048 + 64 ints.
template <int THREADS, int PER, class Group>
__device__ __forceinline__ void radix_select(int groups, int k, int* hist, int* scan, u32* found,
                                             Group&& group, u32& thr, int& need) {
    constexpr int BINS = 2048;
    u32 prefix = 0, mask = 0;
    int rank = k;
    // A set of one group is read once, for all three passes.
    u32 kept[PER];
    bool kept_ok[PER];
    if (groups == 1) group(0, kept, kept_ok);
#pragma unroll 1
    for (int pass = 0; pass < 3; ++pass) {
        const int sh = pass == 0 ? 21 : pass == 1 ? 10 : 0;
        const int bins = pass == 2 ? 1024 : 2048;
        for (int i = threadIdx.x; i < BINS + kWave; i += THREADS) hist[i] = 0;
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < groups; ++g) {
            u32 key[PER];
            bool ok[PER];
            if (groups == 1) {
#pragma unroll
                for (int u = 0; u < PER; ++u) key[u] = kept[u], ok[u] = kept_ok[u];
            } else {
                group(g, key, ok);
            }
            // Every bin first, then the adds. A key that does not count adds
            // nothing: a short row's padding (every slot past it reads the
            // same clamped score) and the keys a narrowed prefix leaves out
            // would all land on a few discard words, and LDS serializes the
            // atomics on one word, so a 500-key row cost 40 us a pass.
            int bin[PER];
#pragma unroll
            for (int u = 0; u < PER; ++u)
                bin[u] = ok[u] && (key[u] & mask) == prefix ? static_cast<int>((key[u] >> sh) & (bins - 1)) : -1;
#pragma unroll
            for (int u = 0; u < PER; ++u)
                if (bin[u] >= 0) __hip_atomic_fetch_add(&hist[bin[u]], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        __syncthreads();
        // Thread i owns `per` bins counted from the top.
        constexpr int per = BINS / THREADS > 0 ? BINS / THREADS : 1;
        int mine = 0;
#pragma unroll
        for (int j = 0; j < per; ++j) {
            const int bin = bins - 1 - per * static_cast<int>(threadIdx.x) - j;
            if (bin >= 0) mine += hist[bin];
        }
        int total;
        int above = block_exclusive<THREADS>(mine, scan, total);
        if (above < rank && rank <= above + mine) {
            for (int j = 0; j < per; ++j) {
                const int bin = bins - 1 - per * static_cast<int>(threadIdx.x) - j;
                const int h = hist[bin];
                if (rank <= above + h) {
                    found[0] = static_cast<u32>(bin);
                    found[1] = static_cast<u32>(rank - above);
                    break;
                }
                above += h;
            }
        }
        __syncthreads();
        prefix |= found[0] << sh;
        mask |= static_cast<u32>(bins - 1) << sh;
        rank = static_cast<int>(found[1]);
        __syncthreads();
    }
    thr = prefix;
    need = rank;
}

// The scoring's summaries and the candidate lists, per row of the launch.
struct TopkLists {
    float* summary;      // [row][blocks · kSummary]
    float* bound;        // [row]
    i32* count;          // [row]
    i32* cand_idx;       // [row][kCandidates]
    u32* cand_key;       // [row][kCandidates]
    i32 summary_stride;  // blocks · kSummary
};

// The bound of row t: the k-th largest of its blocks' summaries, or +inf
// (no candidates, so the select walks the row) when they are too few. It
// also clears the row's candidate count. One block of 1024 per row.
__global__ __launch_bounds__(kSelectThreads) void index_topk_bound(const TopkLists lists, const i32* __restrict__ qo_indptr,
                                                        const i32* __restrict__ page_indices,
                                                        const i32* __restrict__ page_indptr,
                                                        const i32* __restrict__ last_page_lens, i32 requests,
                                                        i32 page_size, i32 score_stride, i32 topk, i32 ratio,
                                                        const u32* __restrict__ win) {
    __shared__ int hist[2048 + kWave];
    __shared__ int scan[kSelectWaves];
    __shared__ u32 found[2];
    const Window window{win};
    const int t = blockIdx.x;
    if (!window.live(t)) return;
    const IndexRow row = index_row(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(requests),
                                   page_size, score_stride, ratio, t);
    const int s = (row.nkeys + 255) / 256 * kSummary;
    if (threadIdx.x == 0) lists.count[t] = 0;
    if (row.nkeys <= topk || s < topk) {
        if (threadIdx.x == 0) lists.bound[t] = __builtin_huge_valf();
        return;
    }
    const float* sum = lists.summary + static_cast<long long>(t) * lists.summary_stride;
    u32 thr;
    int need;
    radix_select<kSelectThreads, 8>(
        (s + 8 * kSelectThreads - 1) / (8 * kSelectThreads), topk, hist, scan, found,
        [&](int g, u32 (&key)[8], bool (&ok)[8]) {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int i = g * 8 * kSelectThreads + u * kSelectThreads + static_cast<int>(threadIdx.x);
                key[u] = order_key(sum[min(i, s - 1)]);
                ok[u] = i < s;
            }
        },
        thr, need);
    if (threadIdx.x == 0) {
        const u32 bits = (thr & 0x80000000u) ? (thr & 0x7fffffffu) : ~thr;
        lists.bound[t] = __builtin_bit_cast(float, bits);
    }
}

// Every score of row t at or above its bound into the row's candidate list
// (in no order; a list that would overflow keeps counting, so the select
// knows). One block per (2048 keys, row), eight keys a thread.
__global__ __launch_bounds__(256) void index_topk_collect(const TopkLists lists, const float* scores,
                                                          const i32* __restrict__ qo_indptr,
                                                          const i32* __restrict__ page_indices,
                                                          const i32* __restrict__ page_indptr,
                                                          const i32* __restrict__ last_page_lens, i32 requests,
                                                          i32 page_size, i32 score_stride, i32 topk, i32 ratio,
                                                          const u32* __restrict__ win) {
    const Window window{win};
    const int t = blockIdx.y;
    if (!window.live(t)) return;
    const IndexRow row = index_row(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(requests),
                                   page_size, score_stride, ratio, t);
    const int j0 = static_cast<int>(blockIdx.x) * 2048 + 8 * static_cast<int>(threadIdx.x);
    if (row.nkeys <= topk || static_cast<int>(blockIdx.x) * 2048 >= row.nkeys) return;
    const u32 bound = order_key(lists.bound[t]);
    const float* frow = scores + static_cast<long long>(t) * score_stride;
    float v[8];
    if (j0 + 8 <= score_stride) {
        const f32x4 a = *reinterpret_cast<const f32x4*>(frow + j0);
        const f32x4 b = *reinterpret_cast<const f32x4*>(frow + j0 + 4);
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    } else {
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = frow[min(j0 + u, score_stride - 1)];
    }
    u32 key[8];
    int mine = 0;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        key[u] = order_key(v[u]);
        mine += j0 + u < row.nkeys && key[u] >= bound;
    }
    // One add per wavefront for its slots.
    const int inc = wave_inclusive(mine);
    int base = 0;
    if (lane() == kWave - 1 && inc > 0) base = atomicAdd(&lists.count[t], inc);
    base = __shfl(base, kWave - 1, kWave) + inc - mine;
    i32* idx = lists.cand_idx + static_cast<long long>(t) * kCandidates;
    u32* keys = lists.cand_key + static_cast<long long>(t) * kCandidates;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        if (j0 + u < row.nkeys && key[u] >= bound) {
            if (base < kCandidates) {
                idx[base] = j0 + u;
                keys[base] = key[u];
            }
            ++base;
        }
    }
}

// Row t's selection: from its candidates when they fit, else by the walk
// over its whole row. One block of 1024 per row; a row no longer than k
// lists every key and pads with −1.
__global__ __launch_bounds__(kSelectThreads) void index_topk_select(
    const float* scores, i32* __restrict__ selection, const i32* __restrict__ qo_indptr,
    const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, i32 requests, i32 page_size, i32 score_stride, i32 topk, i32 ratio,
    const TopkLists lists, const u32* __restrict__ win) {
    constexpr int BINS = 2048;
    __shared__ int hist[BINS + kWave];
    __shared__ int scan[kSelectWaves];
    __shared__ int gts[kSelectWaves], eqs[kSelectWaves];
    __shared__ u32 found[2];
    __shared__ u32 bitmap[kBitmapKeys / 32];
    __shared__ i32 ties[kTies];
    __shared__ int n_ties;
    const Window window{win};
    const int t = blockIdx.x;
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const IndexRow row = index_row(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(requests),
                                   page_size, score_stride, ratio, t);
    const int n = row.nkeys;
    i32* srow = selection + t_row * topk;
    if (n <= topk) {
        for (int i = threadIdx.x; i < topk; i += kSelectThreads) srow[i] = i < n ? i : -1;
        return;
    }
    const float* frow = scores + static_cast<long long>(t) * score_stride;
    const int wv = threadIdx.x / kWave, ln = lane();

    // A set of keys ranked from a short list: the row's candidates, or the
    // whole row when it is no longer than a list (a decode's few hundred
    // pooled keys), which then skips the walk's 32-key-a-thread passes.
    // `key_at(i)` / `idx_at(i)` give entry i's order key and its key index;
    // false leaves the row to the walk (its ties overflowed).
    auto from_list = [&](int cand, auto&& key_at, auto&& idx_at) -> bool {
        u32 thr;
        int need;
        radix_select<kSelectThreads, 4>(
            1, topk, hist, scan, found,
            [&](int, u32 (&key)[4], bool (&ok)[4]) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int i = u * kSelectThreads + static_cast<int>(threadIdx.x);
                    key[u] = key_at(min(i, cand - 1));
                    ok[u] = i < cand;
                }
            },
            thr, need);
        // The bitmap of keys taken: every candidate above T, and of those
        // equal to it the `need` lowest keys.
        const int words = (n + 31) / 32;
        for (int i = threadIdx.x; i < words; i += kSelectThreads) bitmap[i] = 0;
        if (threadIdx.x == 0) n_ties = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < cand; i += kSelectThreads) {
            const u32 key = key_at(i);
            const int j = idx_at(i);
            if (key > thr) {
                atomicOr(&bitmap[j / 32], 1u << (j % 32));
            } else if (key == thr) {
                const int at = atomicAdd(&n_ties, 1);
                if (at < kTies) ties[at] = j;
            }
        }
        __syncthreads();
        // Of the ties, the `need` lowest keys: each ranked among the others
        // (they are few). A tie list that overflowed takes the walk below.
        const int nt = n_ties;
        if (nt <= kTies) {
            for (int i = threadIdx.x; i < nt; i += kSelectThreads) {
                int below = 0;
                if (nt != need)
                    for (int k2 = 0; k2 < nt; ++k2) below += ties[k2] < ties[i];
                if (below < need) atomicOr(&bitmap[ties[i] / 32], 1u << (ties[i] % 32));
            }
        }
        __syncthreads();
        if (nt <= kTies) {
            // The set bits in order: a thread a run of words.
            const int run = (words + kSelectThreads - 1) / kSelectThreads;
            const int w0 = min(words, static_cast<int>(threadIdx.x) * run), w1 = min(words, w0 + run);
            int mine = 0;
            for (int i = w0; i < w1; ++i) mine += __popc(bitmap[i]);
            int total;
            int at = block_exclusive<kSelectThreads>(mine, scan, total);
            for (int i = w0; i < w1; ++i) {
                u32 bits = bitmap[i];
                while (bits != 0) {
                    const int b = __builtin_ctz(bits);
                    bits &= bits - 1;
                    srow[at++] = 32 * i + b;
                }
            }
            return true;
        }
        return false;
    };
    const int cand = lists.count != nullptr ? lists.count[t] : 0;
    if (cand >= topk && cand <= kCandidates && n <= kBitmapKeys) {
        const i32* cidx = lists.cand_idx + static_cast<long long>(t) * kCandidates;
        const u32* ckey = lists.cand_key + static_cast<long long>(t) * kCandidates;
        if (from_list(cand, [&](int i) { return ckey[i]; }, [&](int i) { return cidx[i]; })) return;
    } else if (n <= kCandidates) {
        if (from_list(n, [&](int i) { return order_key(frow[i]); }, [&](int i) { return i; })) return;
    }

    // The walk over the whole row: eight four-score reads in flight a
    // thread, the row in groups of 8 · 1024 quads.
    const int quads = (n + 3) / 4;
    u32 thr;
    int need;
    radix_select<kSelectThreads, 32>(
        (quads + 8 * kSelectThreads - 1) / (8 * kSelectThreads), topk, hist, scan, found,
        [&](int g, u32 (&key)[32], bool (&ok)[32]) {
            const int i0 = g * 8 * kSelectThreads + static_cast<int>(threadIdx.x);
            f32x4 v[8];
            eight(frow, i0, kSelectThreads, quads - 1, v);
#pragma unroll
            for (int u = 0; u < 8; ++u)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j = 4 * (i0 + u * kSelectThreads) + e;
                    key[4 * u + e] = order_key(v[u][e]);
                    ok[4 * u + e] = j < n;
                }
        },
        thr, need);

    // The compaction: wavefront w owns a contiguous share of the row, eight
    // steps of 64 quads at a time; its counts above and at the cut place its
    // writes.
    const int share = (quads + kSelectWaves * kWave - 1) / (kSelectWaves * kWave) * kWave;
    const int lo = min(quads, wv * share), hi = min(quads, lo + share);
    int gt = 0, eq = 0;
    for (int g0 = lo; g0 < hi; g0 += 8 * kWave) {
        f32x4 v[8];
        eight(frow, g0 + ln, kWave, quads - 1, v);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = g0 + u * kWave + ln;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool in = i < hi && 4 * i + e < n;
                const u32 key = order_key(v[u][e]);
                gt += in && key > thr;
                eq += in && key == thr;
            }
        }
    }
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) {
        gt += __shfl_xor(gt, off, kWave);
        eq += __shfl_xor(eq, off, kWave);
    }
    if (ln == 0) {
        gts[wv] = gt;
        eqs[wv] = eq;
    }
    __syncthreads();
    int eq_before = 0, at = 0;
    for (int i = 0; i < wv; ++i) {
        at += gts[i] + min(max(need - eq_before, 0), eqs[i]);
        eq_before += eqs[i];
    }
    int eq_left = min(max(need - eq_before, 0), eq);
    for (int g0 = lo; g0 < hi; g0 += 8 * kWave) {
        f32x4 vs[8];
        eight(frow, g0 + ln, kWave, quads - 1, vs);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = g0 + u * kWave + ln;
            bool big[4], same[4];
            int n_same = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool in = i < hi && 4 * i + e < n;
                const u32 key = order_key(vs[u][e]);
                big[e] = in && key > thr;
                same[e] = in && key == thr;
                n_same += same[e];
            }
            const int same_inc = wave_inclusive(n_same);
            int same_rank = same_inc - n_same;
            bool take[4];
            int n_take = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                take[e] = big[e] || (same[e] && same_rank < eq_left);
                same_rank += same[e];
                n_take += take[e];
            }
            const int take_inc = wave_inclusive(n_take);
            int pos = at + take_inc - n_take;
#pragma unroll
            for (int e = 0; e < 4; ++e)
                if (take[e]) srow[pos++] = 4 * i + e;
            at += __shfl(take_inc, kWave - 1, kWave);
            eq_left -= min(eq_left, __shfl(same_inc, kWave - 1, kWave));
        }
    }
}

// ---------------------------------------------------------- block max --
//
// A row's ranking of the blocks of `block` keys it sees, per head: a block
// scores its best q·k at or before the row; the row's own block is always
// kept, then the best topk − 1 others (ties to the lower), written
// ascending, −1 past the end. Four wavefronts a row: a lane a key of a
// block for scoring, wavefront h ranking head h.

constexpr int kBlockMaxHeads = 8;
constexpr int kBlockMaxDim = 256;

__global__ __launch_bounds__(256) void index_topk_block_max(
    const bf16* __restrict__ idx_q, const bf16* __restrict__ key_pages, const i32* __restrict__ qo_indptr,
    const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, float* __restrict__ scores, i32* __restrict__ selection, i32 requests,
    i32 H, i32 D, i32 page_size, i32 score_stride, i32 topk, i32 block, const u32* __restrict__ win) {
    constexpr int kWaves = 256 / kWave;
    __shared__ float qs[kBlockMaxHeads * kBlockMaxDim];
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    requests = window.lanes(requests);
    const int tid = static_cast<int>(threadIdx.x), w = tid / kWave, l = lane();
    i32* srow = selection + t_row * H * topk;

    const int r = request_of(qo_indptr, requests, t);
    const int kv_len = paged_len(page_indptr, last_page_lens, r, page_size);
    const int abs_q = kv_len - (qo_indptr[r + 1] - qo_indptr[r]) + (t - qo_indptr[r]);
    if (abs_q < 0) {
        for (int n = tid; n < H * topk; n += 256) srow[n] = -1;
        return;
    }
    const int nb = min(abs_q / block + 1, score_stride);
    if (nb <= topk) {
        for (int n = tid; n < H * topk; n += 256) {
            const int at = n % topk;
            srow[n] = at < nb ? at : -1;
        }
        return;
    }

    const bf16* qi = idx_q + t_row * H * D;
    for (int i = tid; i < H * D; i += 256) qs[i] = to_f32(qi[i]);
    __syncthreads();

    const i32* pages = page_indices + page_indptr[r];
    float* frow = scores + static_cast<long long>(t) * H * score_stride;
    for (int b = w; b < nb; b += kWaves) {
        float best[kBlockMaxHeads];
#pragma unroll
        for (int h = 0; h < kBlockMaxHeads; ++h) best[h] = neg_inf();
        const int lo = b * block, hi = min(lo + block, abs_q + 1);
        for (int j = lo + l; j < hi; j += kWave) {
            const bf16* kj = key_pages + (static_cast<long long>(pages[j / page_size]) * page_size + j % page_size) * D;
            float dot[kBlockMaxHeads];
#pragma unroll
            for (int h = 0; h < kBlockMaxHeads; ++h) dot[h] = 0.f;
            for (int d = 0; d < D; d += 8) {
                const Pack8<bf16> pk = *reinterpret_cast<const Pack8<bf16>*>(kj + d);
                float kv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) kv[e] = to_f32(pk.v[e]);
#pragma unroll
                for (int h = 0; h < kBlockMaxHeads; ++h) {
                    if (h >= H) break;
                    const float* qh = qs + h * D + d;
#pragma unroll
                    for (int e = 0; e < 8; ++e) dot[h] += qh[e] * kv[e];
                }
            }
#pragma unroll
            for (int h = 0; h < kBlockMaxHeads; ++h) best[h] = fmaxf(best[h], dot[h]);
        }
#pragma unroll
        for (int h = 0; h < kBlockMaxHeads; ++h) {
            float v = best[h];
#pragma unroll
            for (int off = kWave / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, kWave));
            if (l == 0 && h < H) frow[h * score_stride + b] = v;
        }
    }
    __syncthreads();

    for (int h = w; h < H; h += kWaves) {
        float* hrow = frow + h * score_stride;
        i32* out = srow + h * topk;
        const int own = nb - 1;
        if (l == 0) hrow[own] = neg_inf();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        int picked = 0;
        for (int n = 0; n < topk - 1; ++n) {
            float v = neg_inf();
            int at = nb;
            for (int b = l; b < nb; b += kWave) {
                const float sc = hrow[b];
                if (sc > v) {
                    v = sc;
                    at = b;
                }
            }
#pragma unroll
            for (int off = kWave / 2; off > 0; off >>= 1) {
                const float ov = __shfl_xor(v, off, kWave);
                const int oa = __shfl_xor(at, off, kWave);
                if (ov > v || (ov == v && oa < at)) {
                    v = ov;
                    at = oa;
                }
            }
            if (at >= nb) break;
            if (l == 0) {
                out[picked] = at;
                hrow[at] = neg_inf();
            }
            ++picked;
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
        if (l == 0) {
            out[picked++] = own;
            for (int a = 1; a < picked; ++a) {
                const int key = out[a];
                int c = a - 1;
                while (c >= 0 && out[c] > key) {
                    out[c + 1] = out[c];
                    --c;
                }
                out[c + 1] = key;
            }
            for (int n = picked; n < topk; ++n) out[n] = -1;
        }
    }
}

// A pooled selection read back as tokens: each picked block `b` (`ratio`
// rows, -1 none) its rows `b * ratio ..`, then the row's own open block's
// rows up to and with the row, the rest -1; `topk * ratio + ratio - 1` a row.
__global__ __launch_bounds__(256) void index_pool_tokens(const i32* __restrict__ blocks,
                                                        const i32* __restrict__ qo_indptr,
                                                        const i32* __restrict__ page_indptr,
                                                        const i32* __restrict__ last_page_lens,
                                                        i32* __restrict__ tokens, i32 requests, i32 page_size,
                                                        i32 topk, i32 ratio, const u32* __restrict__ win) {
    const Window window{win};
    const int t = blockIdx.x;
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const int r = request_of(qo_indptr, window.lanes(requests), t);
    const int kv_len = paged_len(page_indptr, last_page_lens, r, page_size);
    const int abs_q = kv_len - (qo_indptr[r + 1] - qo_indptr[r]) + (t - qo_indptr[r]);
    const int open_from = (abs_q + 1) / ratio * ratio;
    const i32* brow = blocks + t_row * topk;
    const int width = topk * ratio + ratio - 1;
    i32* trow = tokens + t_row * width;
    for (int n = threadIdx.x; n < width; n += blockDim.x) {
        int j = -1;
        if (n < topk * ratio) {
            const int b = brow[n / ratio];
            if (b >= 0) j = b * ratio + n % ratio;
        } else if (open_from + (n - topk * ratio) <= abs_q) {
            j = open_from + (n - topk * ratio);
        }
        trow[n] = j;
    }
}

}  // namespace pie::attn
