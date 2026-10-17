#pragma once

#include "attn/index.cuh"
#include "attn/tc5_fmha.cuh"

// A pooled index's scores at prefill widths on tcgen05 (sm_100a): for each
// row and each block it sees, `sum_h w_h * relu(q_h . k)` over the row's
// heads, as `index_topk_paged_mma` scores it, landed in the scores scratch
// for `index_select_rows` to rank. A pass takes two rows of one request
// (one where a request's rows run out), their 2 x 64 heads the MMA's 128
// lanes, against tiles of 128 of the request's blocks:
//
//   - Q (128 lanes by 128 dims, bf16) sits in shared memory as two 128-byte
//     swizzled atoms, the queries' tails turned as they land, two buffers.
//   - A tile's blocks land by TMA a page at a time: a page's entries one
//     `ratio` apart are one box of `per` rows (`map`: [dims, ratio,
//     cells / ratio]), two atoms of 64 dims, three tiles a ring.
//   - S = Q K^T is one M = 128, N = 128 MMA chain into TMEM, two buffers.
//   - The epilogue's thread is a lane (row lane / 64, head lane % 64): each
//     32-block chunk of S rectified and weighted, then summed over the
//     warp's 32 heads by a halving exchange that leaves block L's sum in
//     lane L, the row's two warps' halves meeting in shared memory.
//
// Warps 0-7 run the epilogue (0-3 the even tiles, 4-7 the odd), warp 8
// issues the tiles' loads, warp 9 owns TMEM and issues the MMAs, warps 10-11
// load the queries.
namespace pie::attn::index_tc5 {

using tc5::u32;
using tc5::u64;

constexpr int kD = 128;
constexpr int kBN = 128;                 // blocks a tile
constexpr int kSlots = 3;
constexpr u32 kAtom = 128 * 128;         // 128 rows by 64 bf16
constexpr u32 kTile = 2 * kAtom;         // 32 KB
constexpr u32 kOffQ = 0;                 // two Q buffers
constexpr u32 kOffK = 2 * kTile;
constexpr u32 kOffRed = kOffK + kSlots * kTile;  // [8 warps][128] f32
constexpr u32 kOffW = kOffRed + 8 * 128 * 4;     // [2 buffers][128] f32
constexpr u32 kOffBar = kOffW + 2 * 128 * 4;
constexpr u32 kCols = 512;
constexpr int kSBufs = 4;  // S buffers in TMEM, a tile each

__host__ __device__ constexpr u32 smem_bytes() { return kOffBar + 256u + 1024u; }

struct Params {
    const unsigned short* q;  // [rows][heads * 128]
    const unsigned short* w;  // [rows][heads] or null
    const u32* qo_indptr;
    const u32* page_indices;
    const u32* page_indptr;
    const u32* last_page_lens;
    float* scores;           // [rows][score_stride]
    const i32* turn_pos;
    const u32* win;
    int requests;
    int heads;
    int page_size;
    int score_stride;
    int top_k;
    int ratio;
    int per;                 // a page's entries
    int rows;
    int turn_dim;
    float turn_theta;
    float turn_factor;
    float turn_low;
    float turn_high;
};

// One row's request, first page, page count and the blocks it sees.
struct RowInfo {
    int req, pages_first, num_pages, nkeys;
};

__device__ __forceinline__ RowInfo row_info(const Params& p, int t, int R) {
    RowInfo ri;
    // The request whose rows hold t, by halving (a prefill's 32 requests in
    // five reads, where every role asks for both rows of every pair).
    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(p.qo_indptr[mid + 1])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    const int r = lo;
    const int qo_lo = static_cast<int>(p.qo_indptr[r]);
    const int new_tokens = static_cast<int>(p.qo_indptr[r + 1]) - qo_lo;
    ri.pages_first = static_cast<int>(p.page_indptr[r]);
    ri.num_pages = static_cast<int>(p.page_indptr[r + 1]) - ri.pages_first;
    const int kv_len = (ri.num_pages - 1) * p.page_size + static_cast<int>(p.last_page_lens[r]);
    const int total = kv_len - new_tokens + (t - qo_lo) + 1;
    const int stride = p.ratio > 0 ? p.ratio : 1;
    ri.nkeys = total > 0 ? min(total / stride, p.score_stride) : 0;
    ri.req = r;
    return ri;
}

// A pass: rows `t[0]` and, where `two`, `t[1]` (local rows) of request
// `ri[0].req`, scored over `nkeys` blocks (the larger of theirs). Rows that
// see no more than `top_k` blocks are left to the ranking; a pass with no
// row to score is skipped. Every role walks the same passes.
struct Pass {
    int t[2];
    bool two;
    bool score[2];
    RowInfo ri[2];
    int nkeys;
};

// The pair `pair`'s passes (one or two), in order: `n` of them into `out`.
__device__ __forceinline__ int passes_of(const Params& p, int pair, int live, int R, Pass (&out)[2]) {
    const int t0 = 2 * pair;
    const int t1 = t0 + 1;
    RowInfo a = row_info(p, t0, R);
    const bool has1 = t1 < live;
    RowInfo b = has1 ? row_info(p, t1, R) : a;
    int n = 0;
    auto make = [&](int ta, const RowInfo& ia, bool two, int tb, const RowInfo& ib) {
        Pass ps;
        ps.t[0] = ta;
        ps.t[1] = tb;
        ps.two = two;
        ps.ri[0] = ia;
        ps.ri[1] = ib;
        ps.score[0] = ia.nkeys > p.top_k;
        ps.score[1] = two && ib.nkeys > p.top_k;
        ps.nkeys = max(ps.score[0] ? ia.nkeys : 0, ps.score[1] ? ib.nkeys : 0);
        if (ps.nkeys > 0) out[n++] = ps;
    };
    if (has1 && a.req == b.req) {
        make(t0, a, true, t1, b);
    } else {
        make(t0, a, false, t0, a);
        if (has1) make(t1, b, false, t1, b);
    }
    return n;
}

// Lane L of the warp gets the sum over the warp's lanes of v[L] (v the
// lane's 32 values): five halvings, 31 exchanges.
__device__ __forceinline__ float reduce_scatter32(float (&v)[32], int lane) {
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
        const bool upper = (lane & s) != 0;
#pragma unroll
        for (int i = 0; i < s; ++i) {
            const float send = upper ? v[i] : v[i + s];
            const float keep = upper ? v[i + s] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, s);
        }
    }
    return v[0];
}

__global__ __launch_bounds__(384, 1) void index_logits_tc5(const __grid_constant__ tc5::TmaMap map,
                                                           const __grid_constant__ tc5::TmaMap map8,
                                                           const Params p) {
    using namespace tc5;
    const int live = p.win != nullptr ? min(static_cast<int>(p.win[0]), p.rows) : p.rows;
    const int base_row = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    const int R = (p.win != nullptr && static_cast<int>(p.win[2]) < p.requests) ? static_cast<int>(p.win[2])
                                                                               : p.requests;
    const int pairs = (live + 1) / 2;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);
    float* red = reinterpret_cast<float*>(smem + kOffRed);
    float* wsm = reinterpret_cast<float*>(smem + kOffW);
    u64* k_full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* k_empty = k_full + kSlots;
    u64* q_full = k_empty + kSlots;
    u64* q_empty = q_full + 2;
    u64* s_full = q_empty + 2;
    u64* s_free = s_full + kSBufs;
    u32* tmem_slot = reinterpret_cast<u32*>(s_free + kSBufs);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
        for (int s = 0; s < kSlots; ++s) {
            bar_init(&k_full[s], 1);
            bar_init(&k_empty[s], 1);
        }
        for (int s = 0; s < 2; ++s) {
            bar_init(&q_full[s], 64);
            // The MMAs' commit and the epilogue's 256 (done with the weights).
            bar_init(&q_empty[s], 257);
        }
        for (int s = 0; s < kSBufs; ++s) {
            bar_init(&s_full[s], 1);
            bar_init(&s_free[s], 128);
        }
        bar_init_fence();
    }
    if (warp == 9) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    if (warp == 8) {
        // The tiles' loads: a page's `per` entries a box, two atoms.
        int J = 0;
        for (int pair = static_cast<int>(blockIdx.x); pair < pairs; pair += static_cast<int>(gridDim.x)) {
            Pass ps[2];
            const int np = passes_of(p, pair, live, R, ps);
            for (int k = 0; k < np; ++k) {
                const Pass& s = ps[k];
                const int tiles = (s.nkeys + kBN - 1) / kBN;
                for (int jt = 0; jt < tiles; ++jt, ++J) {
                    const int slot = J % kSlots;
                    // The tile's page ids a lane each, read before the slot frees.
                    const int pages = kBN / p.per;
                    const int pg = static_cast<int>(lane);
                    int id = 0;
                    if (pg < pages) {
                        const int page = min(jt * pages + pg, s.ri[0].num_pages - 1);
                        id = static_cast<int>(p.page_indices[s.ri[0].pages_first + page]);
                    }
                    bar_wait(&k_empty[slot], ((J / kSlots) & 1) ^ 1);
                    if (lane == 0) bar_arrive_expect_tx(&k_full[slot], kTile);
                    __syncwarp();
                    // Eight pages whose ids run on (a fresh prefill's, mostly) land
                    // as one box of `map8`: the lane of the run's first page
                    // issues it, the others nothing; else a box a page.
                    const unsigned fits = pages >= 8 ? 0xffffffffu : 0u;
                    const int prev = __shfl_up_sync(0xffffffffu, id, 1);
                    const bool follows = (pg % 8) != 0 && pg < pages && id == prev + 1;
                    const unsigned links = __ballot_sync(0xffffffffu, follows);
                    // Lane pg (a multiple of 8) leads a run where lanes pg+1..pg+7 follow.
                    const bool run = fits && (pg % 8) == 0 && pg < pages && ((links >> pg) & 0xfeu) == 0xfeu;
                    const bool in_run = fits && pg < pages && ((links >> (pg & ~7)) & 0xfeu) == 0xfeu;
                    unsigned char* dst = smem + kOffK + slot * kTile;
                    if (run) {
#pragma unroll
                        for (int a = 0; a < 2; ++a) {
                            tma_load_3d(dst + a * kAtom + pg * p.per * 128, &k_full[slot], &map8, a * 64, p.ratio - 1,
                                        id * p.per);
                        }
                    } else if (pg < pages && !in_run) {
#pragma unroll
                        for (int a = 0; a < 2; ++a) {
                            tma_load_3d(dst + a * kAtom + pg * p.per * 128, &k_full[slot], &map, a * 64, p.ratio - 1,
                                        id * p.per);
                        }
                    }
                    __syncwarp();
                }
            }
        }
    } else if (warp == 9) {
        const u32 idesc = idesc_dense(kF16Bf16, kF16Bf16, 128, kBN);
        int J = 0, P = 0;
        for (int pair = static_cast<int>(blockIdx.x); pair < pairs; pair += static_cast<int>(gridDim.x)) {
            Pass ps[2];
            const int np = passes_of(p, pair, live, R, ps);
            for (int k = 0; k < np; ++k, ++P) {
                const Pass& s = ps[k];
                const int qb = P & 1;
                bar_wait(&q_full[qb], (P >> 1) & 1);
                tc_fence_after();
                const int tiles = (s.nkeys + kBN - 1) / kBN;
                for (int jt = 0; jt < tiles; ++jt, ++J) {
                    const int slot = J % kSlots;
                    const int sb = J % kSBufs;
                    if (J >= kSBufs) bar_wait(&s_free[sb], ((J / kSBufs) - 1) & 1);
                    bar_wait(&k_full[slot], (J / kSlots) & 1);
                    tc_fence_after();
                    if (lane == 0) {
                        const u32 qa = sbase + kOffQ + qb * kTile;
                        const u32 ka = sbase + kOffK + slot * kTile;
#pragma unroll
                        for (int a = 0; a < 2; ++a) {
#pragma unroll
                            for (int st = 0; st < 4; ++st) {
                                mma_f16(tmem + sb * kBN, sdesc_sw128(qa + a * kAtom) + 2 * st,
                                        sdesc_sw128(ka + a * kAtom) + 2 * st, idesc, a > 0 || st > 0);
                            }
                        }
                        tc_commit(&k_empty[slot]);
                        tc_commit(&s_full[sb]);
                    }
                    __syncwarp();
                }
                if (lane == 0) tc_commit(&q_empty[qb]);
                __syncwarp();
            }
        }
    } else if (warp >= 10) {
        // The queries, 64 threads: lane l of the pass's 128 a row's head,
        // its 16 runs of 8 dims swizzled into the two atoms, the tail
        // turned as `index_topk_paged_mma` turns it; and the head weights.
        const int qt = static_cast<int>(threadIdx.x) - 320;
        __shared__ float turn_f[kD / 2];
        __shared__ float turn_c[2][32], turn_s[2][32];
        if (p.turn_dim > 0) {
            for (int pair = qt; pair < p.turn_dim / 2; pair += 64) {
                float freq = powf(p.turn_theta, -2.f * static_cast<float>(pair) / static_cast<float>(p.turn_dim));
                if (p.turn_factor > 1.f) {
                    freq = ::pie::yarn_original_freq(freq, p.turn_factor, p.turn_low, p.turn_high, pair);
                }
                turn_f[pair] = freq;
            }
        }
        asm volatile("bar.sync 1, 64;" ::: "memory");
        int P = 0;
        for (int pair = static_cast<int>(blockIdx.x); pair < pairs; pair += static_cast<int>(gridDim.x)) {
            Pass ps[2];
            const int np = passes_of(p, pair, live, R, ps);
            for (int k = 0; k < np; ++k, ++P) {
                const Pass& s = ps[k];
                const int qb = P & 1;
                // The pass's angles, a (row, pair) a thread.
                if (p.turn_dim > 0) {
                    const int rr = qt / 32;
                    const int pr = qt % 32;
                    if (pr < p.turn_dim / 2 && (rr == 0 || s.two)) {
                        const float pos = static_cast<float>(p.turn_pos[base_row + s.t[rr]]);
                        __sincosf(pos * turn_f[pr], &turn_s[rr][pr], &turn_c[rr][pr]);
                    }
                }
                bar_wait(&q_empty[qb], ((P >> 1) & 1) ^ 1);
                asm volatile("bar.sync 1, 64;" ::: "memory");
                unsigned char* qd = smem + kOffQ + qb * kTile;
                // A thread's 32 runs, eight loads in flight a batch.
                for (int b0 = qt; b0 < 128 * 16; b0 += 64 * 8) {
                    int4 v[8];
#pragma unroll
                    for (int u8 = 0; u8 < 8; ++u8) {
                        const int i = b0 + u8 * 64;
                        const int l = i / 16;
                        const int c = i % 16;
                        const int rr = l / 64;
                        const int h = l % 64;
                        v[u8] = make_int4(0, 0, 0, 0);
                        if ((rr == 0 || s.two) && h < p.heads) {
                            const int t_row = base_row + s.t[rr];
                            v[u8] = *reinterpret_cast<const int4*>(
                                p.q + (static_cast<long long>(t_row) * p.heads + h) * kD + c * 8);
                        }
                    }
#pragma unroll
                    for (int u8 = 0; u8 < 8; ++u8) {
                        const int i = b0 + u8 * 64;
                        const int l = i / 16;        // lane: row l / 64, head l % 64
                        const int c = i % 16;        // run of 8 dims
                        const int rr = l / 64;
                        if (p.turn_dim > 0 && c * 8 >= kD - p.turn_dim) {
                            unsigned short* e = reinterpret_cast<unsigned short*>(&v[u8]);
#pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const int pr = (c * 8 - (kD - p.turn_dim)) / 2 + u;
                                const float sn = turn_s[rr][pr];
                                const float cs = turn_c[rr][pr];
                                const float a = __uint_as_float(static_cast<u32>(e[2 * u]) << 16);
                                const float b = __uint_as_float(static_cast<u32>(e[2 * u + 1]) << 16);
                                const u32 packed = fmha::pack_bf16(a * cs - b * sn, b * cs + a * sn);
                                e[2 * u] = static_cast<unsigned short>(packed & 0xffffu);
                                e[2 * u + 1] = static_cast<unsigned short>(packed >> 16);
                            }
                        }
                        const int atom = c / 8;
                        const int ch = c % 8;
                        *reinterpret_cast<int4*>(qd + atom * kAtom + l * 128 + ((ch ^ (l & 7)) * 16)) = v[u8];
                    }
                }
                for (int l = qt; l < 128; l += 64) {
                    const int rr = l / 64;
                    const int h = l % 64;
                    float wv = 0.f;
                    if ((rr == 0 || s.two) && h < p.heads) {
                        const int t_row = base_row + s.t[rr];
                        wv = p.w != nullptr
                                 ? __uint_as_float(static_cast<u32>(p.w[static_cast<long long>(t_row) * p.heads + h]) << 16)
                                 : 1.f;
                    }
                    wsm[qb * 128 + l] = wv;
                }
                fence_async_shared();
                bar_arrive(&q_full[qb]);
                asm volatile("bar.sync 1, 64;" ::: "memory");
            }
        }
    } else {
        // The epilogue, two warpgroups taking even and odd tiles: thread l
        // of a group is TMEM lane l.
        const int l = static_cast<int>(threadIdx.x) % 128;
        const int wg = static_cast<int>(warp) / 4;
        const int rr = l / 64;
        const u32 lane_base = ((warp % 4) * 32) << 16;
        const int bar_id = 2 + 2 * wg + rr;
        int J = 0, P = 0;
        for (int pair = static_cast<int>(blockIdx.x); pair < pairs; pair += static_cast<int>(gridDim.x)) {
            Pass ps[2];
            const int np = passes_of(p, pair, live, R, ps);
            for (int k = 0; k < np; ++k, ++P) {
                const Pass& s = ps[k];
                const int qb = P & 1;
                bar_wait(&q_full[qb], (P >> 1) & 1);
                const float wv = wsm[qb * 128 + l];
                const bool mine = rr == 0 || s.two;
                const int nk = s.ri[rr].nkeys;
                const bool write = mine && s.score[rr];
                float* frow = p.scores + static_cast<long long>(s.t[rr]) * p.score_stride;
                const int tiles = (s.nkeys + kBN - 1) / kBN;
                for (int jt = 0; jt < tiles; ++jt, ++J) {
                    const int sb = J % kSBufs;
                    if ((J & 1) != wg) continue;
                    bar_wait(&s_full[sb], (J / kSBufs) & 1);
                    tc_fence_after();
#pragma unroll 1
                    for (int c = 0; c < kBN / 32; ++c) {
                        u32 raw32[32];
                        tmem_ld32(tmem + lane_base + sb * kBN + c * 32, raw32);
                        tmem_wait_ld();
                        float v[32];
#pragma unroll
                        for (int i = 0; i < 32; ++i) v[i] = fmaxf(__uint_as_float(raw32[i]), 0.f) * wv;
                        red[warp * 128 + c * 32 + lane] = reduce_scatter32(v, static_cast<int>(lane));
                    }
                    tc_fence_before();
                    bar_arrive(&s_free[sb]);
                    // The row's two warps (heads 0-31, 32-63) meet.
                    asm volatile("bar.sync %0, 64;" ::"r"(bar_id) : "memory");
                    if (write && (warp & 1) == 0) {
#pragma unroll
                        for (int c = 0; c < kBN / 32; ++c) {
                            const int j = jt * kBN + c * 32 + static_cast<int>(lane);
                            if (j < nk) frow[j] = red[warp * 128 + c * 32 + lane] + red[(warp + 1) * 128 + c * 32 + lane];
                        }
                    }
                    asm volatile("bar.sync %0, 64;" ::"r"(bar_id) : "memory");
                }
                bar_arrive(&q_empty[qb]);
            }
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 9) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

// The ranking of rows `index_logits_tc5` scored, a block a row; rows that
// see no more than `top_k` blocks take them all.
template <bool kRadix>
__global__ void __launch_bounds__(kBlock) index_select_rows(const Params p, i32* __restrict__ selection) {
    const int t = static_cast<int>(blockIdx.x);
    const int live = p.win != nullptr ? min(static_cast<int>(p.win[0]), p.rows) : p.rows;
    if (t >= live) return;
    const int base_row = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    const int R = (p.win != nullptr && static_cast<int>(p.win[2]) < p.requests) ? static_cast<int>(p.win[2])
                                                                               : p.requests;
    const RowInfo ri = row_info(p, t, R);
    const int tid = static_cast<int>(threadIdx.x);
    i32* srow = selection + static_cast<long long>(base_row + t) * p.top_k;
    if (ri.nkeys <= p.top_k) {
        for (int n = tid; n < p.top_k; n += kBlock) srow[n] = (n < ri.nkeys) ? n : -1;
        return;
    }
    const float* frow = p.scores + static_cast<long long>(t) * p.score_stride;
    if constexpr (kRadix) {
        select_top_radix(frow, ri.nkeys, p.top_k, srow, tid);
    } else {
        select_top(frow, ri.nkeys, p.top_k, srow, tid);
    }
}

}  // namespace pie::attn::index_tc5
