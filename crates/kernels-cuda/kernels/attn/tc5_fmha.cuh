#pragma once

#include "prelude/tc5.cuh"

// A causal (or full, or sliding) prefill over a paged bf16 KV cache on
// tcgen05.
//
// A block takes one key head and 128 packed query rows of one request:
// packed row p is token p / G of the request at query head h * G + p % G,
// so the G query heads that share a key head share its tiles. Q sits in
// TMEM, read in place as the A operand of S = Q K^T; S lands in TMEM in two
// buffers, so the next tile's product runs while this one's softmax does;
// the softmax writes P over its tile's S, the A operand of O += P V; O
// accumulates in TMEM. Shared memory holds only the K and V rings. V is
// read as stored, its head dimension contiguous: the B operand of P V is
// MN-major.
//
// Warps 0-3 load Q, run the softmax (one row a thread, the rows of their
// own TMEM lane quarter) and store O; warp 4 is the TMA producer of K and V,
// a 16-token box a lane; warp 5 owns TMEM and its lane 0 issues every MMA.
// The softmax keeps a row's running max in the log2 domain and only
// rescales O when the max outgrows the one P was formed against by more
// than 2^8, so most tiles never touch O.
//
// KV tiles are kBKV tokens. The pool is NHD, `page` tokens a page; a tile
// is whole pages, landed as a box a 16-token page slice and 64-wide atom of
// the head. TMA spends far longer on many small boxes than on the same
// bytes in a few large ones, so a tile every row reads whole (no mask, no
// key past the request's) whose pages lie in a run, rising or falling (the
// pool hands pages out from a stack), lands as one box an atom instead
// (`map_k_run`, `map_v_run`, kBKV rows a box): a falling run's slices land
// in reverse, which no unmasked product can tell. A slice past the
// request's pages lands as whatever the pool holds there (zeros past its
// end): the softmax masks its scores and zeroes its V rows.

namespace pie::attn::fmha {

using tc5::u32;
using tc5::u64;
typedef unsigned short u16;

struct Params {
    const u16* q;
    u16* o;
    const int* qo_indptr;
    const int* kv_indptr;
    const int* kv_indices;
    const int* last_page_len;
    // The fire's live lanes, [first lane, requests], as the plan stages them
    // each fire: every per-lane array is indexed by the absolute lane.
    const int* live;
    int q_stride;
    int o_stride;
    int kv_heads;
    int group;
    int page;
    float scale_log2;
    int causal;
    // Keys a row reads reach back this far behind its own position; -1 is
    // no window.
    int window_left;
};

__host__ __device__ constexpr u32 smem_bytes(int d, int bkv, int stages, int dv = 0) {
    // The K and V rings, then barriers; 1024-aligned.
    return u32(stages) * u32(d / 64 + (dv > 0 ? dv : d) / 64) * u32(bkv) * 128u + 512u + 1024u;
}

__device__ __forceinline__ u32 pack_bf16(float lo, float hi) {
    u32 bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return bits;
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// MN-major, 128B-swizzled operand: atoms of 64 MN elements by 8 K rows,
// `lbo` bytes from one 64-wide MN atom to the next, 1024 bytes from one 8-row
// K group to the next.
__device__ __forceinline__ u64 sdesc_sw128_mn(u32 saddr, u32 lbo) {
    u64 d = 0;
    d |= static_cast<u64>((saddr & 0x3FFFFu) >> 4);
    d |= static_cast<u64>(lbo >> 4) << 16;
    d |= static_cast<u64>(1024 >> 4) << 32;
    d |= static_cast<u64>(1) << 46;
    d |= static_cast<u64>(2) << 61;
    return d;
}

__device__ __forceinline__ void tmem_st32(u32 taddr, const u32 (&v)[32]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
        "%11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
        "%28, %29, %30, %31, %32};" ::"r"(taddr),
        "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]),
        "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]),
        "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]),
        "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]),
        "r"(v[29]), "r"(v[30]), "r"(v[31])
        : "memory");
}

__device__ __forceinline__ void tmem_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}


// The block's work: its request, its first packed row, the request's
// extents. Blocks walk the requests' tiles from the last request's last
// tile back, so the longest causal tiles start first.
struct Work {
    int r, row0, qo_start, qo_len, kv_len, kv_page0;
};

__device__ __forceinline__ bool work_of(const Params& p, int bound, int block, Work& w) {
    const int t = bound - 1 - block;
    int start = 0;
    const int lane0 = p.live[0];
    for (int r = lane0; r < lane0 + p.live[1]; ++r) {
        const int qs = p.qo_indptr[r];
        const int ql = p.qo_indptr[r + 1] - qs;
        const int tiles = (ql * p.group + 127) / 128;
        if (t < start + tiles) {
            w.r = r;
            w.row0 = (t - start) * 128;
            w.qo_start = qs;
            w.qo_len = ql;
            w.kv_page0 = p.kv_indptr[r];
            const int pages = p.kv_indptr[r + 1] - w.kv_page0;
            w.kv_len = pages > 0 ? (pages - 1) * p.page + p.last_page_len[r] : 0;
            return true;
        }
        start += tiles;
    }
    return false;
}

// `DV` the value head's width where it differs from the query and key
// head's `D` (an expanded latent attention's: 192 against 128), its rows
// read at the V map's own head stride.
template <int D, int BKV, int ST, int DV = D>
__global__ __launch_bounds__(192, 1) void tc5_fmha_prefill(
    const __grid_constant__ tc5::TmaMap map_k, const __grid_constant__ tc5::TmaMap map_v,
    const __grid_constant__ tc5::TmaMap map_k_run, const __grid_constant__ tc5::TmaMap map_v_run,
    const Params p, const int bound) {
    using namespace tc5;
    static_assert(D == 64 || D == 128 || D == 192 || D == 256, "a head of 64, 128, 192 or 256");
    static_assert(DV == 64 || DV == 128 || DV == 256, "a value head of 64, 128 or 256");
    constexpr int kBKV = BKV;
    static_assert(BKV == 64 || BKV == 128, "a key tile is one or two 64-wide P atoms");
    constexpr u32 kNA = D / 64;            // 64-wide atoms along the key head
    constexpr u32 kNAV = DV / 64;          // and along the value head
    constexpr u32 kKA = kBKV / 64;         // 64-wide atoms along a key tile
    constexpr u32 kAtomKV = kBKV * 128;    // one 64-wide atom of a K or V tile
    constexpr u32 kTileKV = kNA * kAtomKV;
    constexpr u32 kTileV = kNAV * kAtomKV;
    // Shared memory holds the K and V rings alone: Q and P live in TMEM.
    constexpr u32 kOffK = 0;
    constexpr u32 kOffV = kOffK + ST * kTileKV;
    constexpr u32 kOffBar = kOffV + ST * kTileV;
    // TMEM: two S buffers (P overwrites its tile's, two bf16 a column), O,
    // and Q (two bf16 a column), the A operands read in place.
    constexpr u32 kColS = 0;
    constexpr u32 kColO = 2 * kBKV;
    constexpr u32 kColQ = kColO + DV;
    constexpr u32 kCols = 512;
    static_assert(kColQ + D / 2 <= kCols, "S twice, O and Q fit TMEM");
    (void)kKA;

    Work w;
    if (!work_of(p, bound, static_cast<int>(blockIdx.x), w)) return;
    const int h = blockIdx.y;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);

    u64* k_full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* k_empty = k_full + ST;
    u64* v_full = k_empty + ST;
    u64* v_empty = v_full + ST;
    u64* s_full = v_empty + ST;
    u64* q_ready = s_full + 2;
    u64* p_full = q_ready + 1;
    // P V of tile i commits to pv_done[i & 1]: each fills once per two tiles,
    // so a wait for either of the last two products names its phase alone.
    u64* pv_done = p_full + 2;
    u64* v_fixed = pv_done + 2;
    u32* tmem_slot = reinterpret_cast<u32*>(v_fixed + 1);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    // Key tiles: from the first row's oldest key (a window's) through the
    // last key any row of the tile reads.
    auto tiles_of = [&](const Work& x, int& first) {
        const int last_row = min(x.row0 + 127, x.qo_len * p.group - 1);
        const int last_tok = last_row / p.group;
        const int reach = p.causal ? min(x.kv_len, x.kv_len - x.qo_len + last_tok + 1) : x.kv_len;
        const int first_lo = p.window_left < 0
            ? 0
            : max(0, x.kv_len - x.qo_len + x.row0 / p.group - p.window_left);
        first = first_lo / kBKV;
        return max(0, (reach + kBKV - 1) / kBKV - first);
    };
    int jb = 0;
    const int nt = tiles_of(w, jb);

    if (threadIdx.x == 0) {
        for (int s = 0; s < ST; ++s) {
            bar_init(&k_full[s], 1);
            bar_init(&k_empty[s], 1);
            bar_init(&v_full[s], 1);
            bar_init(&v_empty[s], 1);
        }
        for (int s = 0; s < 2; ++s) {
            bar_init(&s_full[s], 1);
        }
        bar_init(q_ready, 128);
        bar_init(&p_full[0], 128);
        bar_init(&p_full[1], 128);
        bar_init(&pv_done[0], 1);
        bar_init(&pv_done[1], 1);
        bar_init(v_fixed, 128);
        bar_init_fence();
    }
    if (warp == 5) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    if (warp == 4) {
        if (nt > 0) {
            if (lane == 0) {
                tma_prefetch(&map_k);
                tma_prefetch(&map_v);
                tma_prefetch(&map_k_run);
                tma_prefetch(&map_v_run);
            }
            const int pages = (w.kv_len + p.page - 1) / p.page;
            // A tile is 16-token boxes, one 64-wide atom each, spread over the
            // warp's lanes so their loads issue side by side; or, its pages
            // in a run, one box an atom.
            constexpr int kBoxes = (kBKV / 16) * kNA;
            static_assert(kBoxes <= 32, "a box a lane");
            // This lane's box: its 16 tokens and its atom of the head. Its pool
            // row for the next tile is read while this one's loads go out, so
            // the page table's latency stays off the loads' path. A slice
            // past the request's pages is -1: any row will do.
            const int b = static_cast<int>(lane);
            const int t16 = b / static_cast<int>(kNA);
            const u32 a = static_cast<u32>(b) % kNA;
            // A narrower value head's boxes: its own lanes' slices, each
            // slice's pool row the key box's of that slice.
            constexpr int kBoxesV = (kBKV / 16) * kNAV;
            const int t16v = b / static_cast<int>(kNAV);
            const u32 av = static_cast<u32>(b) % kNAV;
            const int src_v = min(t16v, kBKV / 16 - 1) * static_cast<int>(kNA);
            auto row_of = [&](int j) {
                const int tok = (jb + j) * kBKV + t16 * 16;
                const int at = tok / p.page;
                return at < pages ? p.kv_indices[w.kv_page0 + at] * p.page + tok % p.page : -1;
            };
            auto load = [&](const TmaMap* map, const TmaMap* run, unsigned char* dst, u64* bar,
                            int row, int row0, bool whole, u32 na, int boxes, int t, u32 at) {
                if (whole) {
                    if (b < static_cast<int>(na)) {
                        tma_load_3d(dst + at * kAtomKV, bar, run, static_cast<int>(at * 64), h, row0);
                    }
                } else if (b < boxes) {
                    tma_load_3d(dst + at * kAtomKV + t * 16 * 128, bar, map,
                                static_cast<int>(at * 64), h, row < 0 ? 0x7fffff00 : row);
                }
            };
            // Tiles every live row of the block reads whole: from the last
            // row's oldest key to the first row's newest.
            const int live_rows = min(128, w.qo_len * p.group - w.row0);
            const int tok_first = w.row0 / p.group;
            const int tok_last = (w.row0 + live_rows - 1) / p.group;
            const int whole_lo = p.window_left < 0
                ? 0
                : max(0, w.kv_len - w.qo_len + tok_last - p.window_left);
            const int whole_hi = p.causal ? min(w.kv_len, w.kv_len - w.qo_len + tok_first + 1)
                                          : w.kv_len;
            int row = b < kBoxes ? row_of(0) : -1;
            for (int j = 0; j < nt; ++j) {
                const u32 st = j % ST;
                const u32 ph = (j / ST) & 1;
                const int next = b < kBoxes && j + 1 < nt ? row_of(j + 1) : -1;
                const int k0 = (jb + j) * kBKV;
                int row0 = __shfl_sync(0xffffffffu, row, 0);
                const bool inside = k0 >= whole_lo && k0 + kBKV <= whole_hi;
                const bool up = __all_sync(0xffffffffu, b >= kBoxes || row == row0 + t16 * 16);
                const bool down = __all_sync(0xffffffffu, b >= kBoxes || row == row0 - t16 * 16);
                const bool whole = inside && (up || down);
                if (!up) row0 -= (kBKV / 16 - 1) * 16;
                bar_wait(&k_empty[st], ph ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&k_full[st], kTileKV);
                __syncwarp();
                load(&map_k, &map_k_run, smem + kOffK + st * kTileKV, &k_full[st], row, row0, whole,
                     kNA, kBoxes, t16, a);
                const int row_v = DV == D ? row : __shfl_sync(0xffffffffu, row, src_v);
                bar_wait(&v_empty[st], ph ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&v_full[st], kTileV);
                __syncwarp();
                if constexpr (DV == D) {
                    load(&map_v, &map_v_run, smem + kOffV + st * kTileV, &v_full[st], row_v, row0,
                         whole, kNA, kBoxes, t16, a);
                } else {
                    load(&map_v, &map_v_run, smem + kOffV + st * kTileV, &v_full[st], row_v, row0,
                         whole, kNAV, kBoxesV, t16v, av);
                }
                row = next;
            }
        }
    } else if (warp == 5) {
        if (lane == 0 && nt > 0) {
            const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 128, kBKV);
            const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 128, DV) | (1u << 16);
            bar_wait(q_ready, 0);
            auto pv = [&](int i) {
                const u32 st = i % ST;
                bar_wait(&v_full[st], (i / ST) & 1);
                if (i == nt - 1) bar_wait(v_fixed, 0);
                bar_wait(&p_full[i & 1], (i >> 1) & 1);
                tc_fence_after();
                const u32 vb = sbase + kOffV + st * kTileV;
                const u32 pa = tmem + kColS + (i & 1) * kBKV;
#pragma unroll
                for (u32 s = 0; s < kBKV / 16; ++s) {
                    const u64 b = sdesc_sw128_mn(vb + s * 2048, kAtomKV);
                    mma_f16_ts(tmem + kColO, pa + s * 8, b, idesc_o, i > 0 || s > 0);
                }
                tc_commit(&v_empty[st]);
                tc_commit(&pv_done[i & 1]);
            };
            for (int j = 0; j < nt; ++j) {
                const u32 st = j % ST;
                const u32 sb = j & 1;
                bar_wait(&k_full[st], (j / ST) & 1);
                // This S buffer held P two tiles back: free once that product
                // has read it. Only it can have landed on pv_done yet.
                if (j >= 2) bar_wait(&pv_done[j & 1], ((j - 2) >> 1) & 1);
                tc_fence_after();
                const u32 kb = sbase + kOffK + st * kTileKV;
#pragma unroll
                for (u32 s = 0; s < D / 16; ++s) {
                    const u64 b = sdesc_sw128(kb + (s / 4) * kAtomKV) + 2 * (s % 4);
                    mma_f16_ts(tmem + kColS + sb * kBKV, tmem + kColQ + s * 8, b, idesc_s, s > 0);
                }
                tc_commit(&s_full[sb]);
                tc_commit(&k_empty[st]);
                if (j > 0) pv(j - 1);
            }
            pv(nt - 1);
        }
    } else {
        // The softmax warpgroup: row `row` of the tile is this thread's.
        const u32 row = warp * 32 + lane;
        const int prow = w.row0 + static_cast<int>(row);
        const bool live = prow < w.qo_len * p.group;
        const int tok = prow / p.group;
        const int qh = h * p.group + prow % p.group;
        // Keys this row reads: [lo, limit).
        const int lo = p.window_left < 0
            ? 0
            : max(0, w.kv_len - w.qo_len + tok - p.window_left);
        const int limit = !live ? 0
                          : p.causal ? min(w.kv_len, w.kv_len - w.qo_len + tok + 1)
                                     : w.kv_len;
        const u32 lane_base = (warp * 32) << 16;
        {
            // Q's row into this thread's TMEM lane, 64 values a store.
            const uint4* src = reinterpret_cast<const uint4*>(
                p.q + static_cast<long long>(w.qo_start + (live ? tok : 0)) * p.q_stride +
                static_cast<long long>(qh) * D);
            // Every load in flight at once, then the stores.
            uint4 qv[D / 8];
#pragma unroll
            for (int i = 0; i < D / 8; ++i) qv[i] = live ? src[i] : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (u32 c = 0; c < D / 2; c += 32) {
                u32 v[32];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[4 * i] = qv[c / 4 + i].x;
                    v[4 * i + 1] = qv[c / 4 + i].y;
                    v[4 * i + 2] = qv[c / 4 + i].z;
                    v[4 * i + 3] = qv[c / 4 + i].w;
                }
                tmem_st32(tmem + lane_base + kColQ + c, v);
            }
            tmem_wait_st();
            tc_fence_before();
            bar_arrive(q_ready);
        }
        float m_used = neg_inf();
        float l = 0.f;
        for (int j = 0; j < nt; ++j) {
            const u32 st = j & 1;
            bar_wait(&s_full[st], (j >> 1) & 1);
            tc_fence_after();
            float s[kBKV];
#pragma unroll
            for (u32 c = 0; c < kBKV; c += 32) {
                u32 v[32];
                tmem_ld32(tmem + lane_base + kColS + st * kBKV + c, v);
                tmem_wait_ld();
#pragma unroll
                for (int i = 0; i < 32; ++i) s[c + i] = __uint_as_float(v[i]);
            }
            const int k0 = (jb + j) * kBKV;
            // A tile wholly inside the row's keys skips the mask; the scale
            // folds into the exponent's FMA, the max taken on raw scores.
            float m4[4] = {neg_inf(), neg_inf(), neg_inf(), neg_inf()};
            if (k0 >= lo && k0 + kBKV <= limit) {
#pragma unroll
                for (int c = 0; c < kBKV; ++c) m4[c & 3] = fmaxf(m4[c & 3], s[c]);
            } else {
#pragma unroll
                for (int c = 0; c < kBKV; ++c) {
                    s[c] = k0 + c < limit && k0 + c >= lo ? s[c] : neg_inf();
                    m4[c & 3] = fmaxf(m4[c & 3], s[c]);
                }
            }
            const float raw_mx = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
            const float mx = raw_mx == neg_inf() ? neg_inf() : raw_mx * p.scale_log2;
            // The max P is formed against: kept until the row's grows past it
            // by 2^8, then O and the sum rescale to the new one.
            const bool grow = mx > m_used + 8.f;
            const float m_new = grow ? mx : m_used;
            const float alpha = grow ? ex2(m_used - m_new) : 1.f;
            const float base = m_new == neg_inf() ? 0.f : m_new;
            float s4[4] = {0.f, 0.f, 0.f, 0.f};
            u32 pk[kBKV / 2];
#pragma unroll
            for (int c = 0; c < kBKV; c += 2) {
                const float e0 = ex2(fmaf(s[c], p.scale_log2, -base));
                const float e1 = ex2(fmaf(s[c + 1], p.scale_log2, -base));
                s4[(c >> 1) & 3] += e0 + e1;
                pk[c / 2] = pack_bf16(e0, e1);
            }
            const float sum = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            l = l * alpha + sum;
            m_used = m_new;
            // O rescales once the last product is in.
            if (j > 0 && __any_sync(0xffffffffu, grow)) {
                bar_wait(&pv_done[(j - 1) & 1], ((j - 1) >> 1) & 1);
                tc_fence_after();
#pragma unroll 1
                for (u32 c = 0; c < DV; c += 32) {
                    u32 v[32];
                    tmem_ld32(tmem + lane_base + kColO + c, v);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        v[i] = __float_as_uint(__uint_as_float(v[i]) * alpha);
                    }
                    tmem_st32(tmem + lane_base + kColO + c, v);
                }
            }
            // P over its S, two bf16 a column, the A operand of P V.
#pragma unroll
            for (u32 c = 0; c < kBKV / 2; c += 32) {
                u32 v[32];
#pragma unroll
                for (int i = 0; i < 32; ++i) v[i] = pk[c + i];
                tmem_st32(tmem + lane_base + kColS + st * kBKV + c, v);
            }
            tmem_wait_st();
            if (j == nt - 1) {
                // The last tile's V rows past the request's keys hold whatever
                // the pool's page held: zero them so P's zeros stay zeros.
                const u32 vst = j % ST;
                bar_wait(&v_full[vst], (j / ST) & 1);
                const int valid = w.kv_len - k0;
                if (static_cast<int>(row) < kBKV && static_cast<int>(row) >= valid) {
#pragma unroll
                    for (u32 a = 0; a < kNAV; ++a) {
#pragma unroll
                        for (u32 c = 0; c < 8; ++c) {
                            const u32 at = sbase + kOffV + vst * kTileV + a * kAtomKV + row * 128 + c * 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(at), "r"(0u)
                                         : "memory");
                        }
                    }
                }
                fence_async_shared();
                bar_arrive(v_fixed);
            }
            tc_fence_before();
            bar_arrive(&p_full[j & 1]);
        }
        if (nt > 0) {
            if (nt >= 2) bar_wait(&pv_done[(nt - 2) & 1], ((nt - 2) >> 1) & 1);
            bar_wait(&pv_done[(nt - 1) & 1], ((nt - 1) >> 1) & 1);
            tc_fence_after();
        }
        // Whole-warp TMEM loads: a dead row still joins its warp's, and
        // stores nothing.
        const float inv = l > 0.f ? 1.f / l : 0.f;
        u16* dst = p.o + static_cast<long long>(w.qo_start + (live ? tok : 0)) * p.o_stride +
                   static_cast<long long>(qh) * DV;
#pragma unroll 1
        for (u32 c = 0; c < DV; c += 32) {
            u32 v[32];
            if (nt > 0) {
                tmem_ld32(tmem + lane_base + kColO + c, v);
                tmem_wait_ld();
            } else {
#pragma unroll
                for (int i = 0; i < 32; ++i) v[i] = 0u;
            }
            if (live) {
                u32 out[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    out[i] = pack_bf16(__uint_as_float(v[2 * i]) * inv,
                                       __uint_as_float(v[2 * i + 1]) * inv);
                }
                uint4* o4 = reinterpret_cast<uint4*>(dst + c);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o4[i] = make_uint4(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
                }
            }
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}


// A 256- or 512-wide head over 128-key tiles. `tc5_fmha_prefill` keeps Q
// in TMEM, which leaves room for 64-key tiles only, and S at 64 columns a
// product runs at half the rate of 128. Here S takes two buffers of 128
// columns beside O's 256, and Q sits in shared memory (QT of its columns
// may sit in TMEM instead, when the tiles leave room), the A operand of S
// read from there. At 512 (Gemma-4's global layers) O for 128 rows would
// fill TMEM alone, so a block lands half of it, the 256 columns of its half
// of V, its pair beside it in y landing the other; both compute S whole.
//
// K and V stream through one ring of slices of a tile (BKV tokens by SA
// 64-wide atoms of the head) in the order the MMA warp takes them: tile j's
// K slices, then tile j - 1's V slices, so the product of a tile's S runs
// while the softmax of the one before it does. A slice is two atoms: each
// slice costs its commit and wait beside its products, and P V at 128
// columns a product runs at twice the rate of 64.
//
// Q lands by TMA when a block's 128 packed rows are whole tokens (G divides
// 128), one box an atom of [64 columns, G heads, 128 / G tokens]; a TMEM
// part lands first in the ring's slots, from which the softmax warps copy it
// into TMEM before the ring starts. Otherwise the softmax warps store it
// from global memory.
namespace wide {

constexpr u32 kAtomQ = 128 * 128;  // 128 rows by 64 bf16

__host__ __device__ constexpr u32 smem_bytes(int d, int bkv, int qt, int sa, int slots) {
    return u32((d - qt) / 64) * kAtomQ + u32(slots) * u32(sa) * u32(bkv) * 128u + 512u + 1024u;
}

// Where in the ring's sequence tile j's first K slice and first V slice
// fall, nt tiles in all, KS K slices and VS V slices a tile.
template <int KS, int VS>
__device__ __forceinline__ int seq_k(int j) {
    return j == 0 ? 0 : (KS + VS) * j - VS;
}
template <int KS, int VS>
__device__ __forceinline__ int seq_v(int j, int nt) {
    return j < nt - 1 ? seq_k<KS, VS>(j + 1) + KS : (KS + VS) * nt - VS;
}

}  // namespace wide

template <int D, int BKV, int QT, int SA, int SLOTS>
__global__ __launch_bounds__(192, 1) void tc5_fmha_prefill_wide(
    const __grid_constant__ tc5::TmaMap map_q,
    const __grid_constant__ tc5::TmaMap map_k, const __grid_constant__ tc5::TmaMap map_v,
    const __grid_constant__ tc5::TmaMap map_k_run, const __grid_constant__ tc5::TmaMap map_v_run,
    const Params p, const int bound) {
    using namespace tc5;
    using namespace wide;
    static_assert(D == 256 || D == 512, "a head of 256 or 512");
    // O's columns a block lands: a 512-wide head's two blocks a half each.
    constexpr int DV = 256;
    constexpr bool kSplit = D > DV;
    constexpr int kBKV = BKV;
    static_assert(BKV == 64 || BKV == 128, "a key tile of 64 or 128");
    static_assert(QT % 64 == 0 && QT < D, "Q's TMEM part is whole atoms");
    constexpr u32 kAtomKV = kBKV * 128;         // BKV tokens by 64 bf16
    constexpr u32 kSlice = SA * kAtomKV;
    constexpr int kKS = D / 64 / SA;            // K slices a tile
    constexpr int kVS = DV / 64 / SA;           // V slices a tile: this block's O
    static_assert(SA == 1 || SA == 2 || SA == 4, "a slice is whole atoms of V's half");
    constexpr int kQA = QT / 64;                // Q atoms in TMEM
    constexpr int kQS = (D - QT) / 64;          // Q atoms in shared memory
    constexpr int kBoxes = kBKV / 16;           // 16-token boxes an atom
    static_assert(SA * kBoxes <= 32, "a box a lane");
    static_assert(SLOTS * kSlice >= kQA * kAtomQ, "Q's TMEM part stages in the ring");
    static_assert(SLOTS >= kVS, "a tile's V fits the ring");
    constexpr u32 kOffQ = 0;
    constexpr u32 kOffRing = kOffQ + kQS * kAtomQ;
    constexpr u32 kOffBar = kOffRing + SLOTS * kSlice;
    // TMEM: two S buffers (P overwrites its tile's, two bf16 a column), O,
    // then Q's part.
    constexpr u32 kColS = 0;
    constexpr u32 kColO = 2 * kBKV;
    constexpr u32 kColQ = kColO + DV;
    constexpr u32 kCols = 512;
    static_assert(kColQ + QT / 2 <= kCols, "S twice, O and Q's part fit TMEM");

    Work w;
    if (!work_of(p, bound, static_cast<int>(blockIdx.x), w)) return;
    const int h = kSplit ? static_cast<int>(blockIdx.y) >> 1 : static_cast<int>(blockIdx.y);
    const int v0 = kSplit ? (static_cast<int>(blockIdx.y) & 1) * DV : 0;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);

    u64* full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* empty = full + SLOTS;
    u64* s_full = empty + SLOTS;
    u64* q_ready = s_full + 2;
    u64* q_held = q_ready + 1;
    u64* p_full = q_held + 1;
    u64* pv_done = p_full + 2;
    u64* v_fixed = pv_done + 2;
    u32* tmem_slot = reinterpret_cast<u32*>(v_fixed + 1);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;
    const bool q_boxed = 128 % p.group == 0;

    int jb = 0;
    int nt = 0;
    {
        const int last_row = min(w.row0 + 127, w.qo_len * p.group - 1);
        const int last_tok = last_row / p.group;
        const int reach =
            p.causal ? min(w.kv_len, w.kv_len - w.qo_len + last_tok + 1) : w.kv_len;
        const int first_lo = p.window_left < 0
            ? 0
            : max(0, w.kv_len - w.qo_len + w.row0 / p.group - p.window_left);
        jb = first_lo / kBKV;
        nt = max(0, (reach + kBKV - 1) / kBKV - jb);
    }

    if (threadIdx.x == 0) {
        for (int s = 0; s < SLOTS; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], 1);
        }
        bar_init(&s_full[0], 1);
        bar_init(&s_full[1], 1);
        bar_init(q_ready, q_boxed ? 1 : 128);
        bar_init(q_held, 128);
        bar_init(&p_full[0], 128);
        bar_init(&p_full[1], 128);
        bar_init(&pv_done[0], 1);
        bar_init(&pv_done[1], 1);
        bar_init(v_fixed, 128);
        bar_init_fence();
    }
    if (warp == 5) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    if (warp == 4) {
        if (nt > 0) {
            if (lane == 0) {
                if (q_boxed) {
                    // Q's TMEM part into the ring's first slots, the rest home.
                    bar_arrive_expect_tx(q_ready, (D / 64) * kAtomQ);
#pragma unroll
                    for (int a = 0; a < D / 64; ++a) {
                        unsigned char* dst = a < kQA ? smem + kOffRing + a * kAtomQ
                                                     : smem + kOffQ + (a - kQA) * kAtomQ;
                        tma_load_3d(dst, q_ready, &map_q, a * 64, h * p.group,
                                    w.qo_start + w.row0 / p.group);
                    }
                }
                tma_prefetch(&map_k);
                tma_prefetch(&map_v);
                tma_prefetch(&map_k_run);
                tma_prefetch(&map_v_run);
            }
            const int pages = (w.kv_len + p.page - 1) / p.page;
            // Lane b < kBoxes holds the pool row of a tile's tokens
            // [16 b, 16 b + 16).
            const int b = static_cast<int>(lane);
            auto row_of = [&](int j) {
                const int tok = (jb + j) * kBKV + b * 16;
                const int at = tok / p.page;
                return b < kBoxes && at < pages
                           ? p.kv_indices[w.kv_page0 + at] * p.page + tok % p.page
                           : -1;
            };
            const int live_rows = min(128, w.qo_len * p.group - w.row0);
            const int tok_first = w.row0 / p.group;
            const int tok_last = (w.row0 + live_rows - 1) / p.group;
            const int whole_lo = p.window_left < 0
                ? 0
                : max(0, w.kv_len - w.qo_len + tok_last - p.window_left);
            const int whole_hi =
                p.causal ? min(w.kv_len, w.kv_len - w.qo_len + tok_first + 1) : w.kv_len;
            // A tile's rows: this lane's, and the whole tile's first when its
            // pages lie in a run (one box a slice).
            struct Tile {
                int row, row0;
                bool whole;
            };
            auto tile_of = [&](int j, int row) {
                Tile t;
                t.row = row;
                t.row0 = __shfl_sync(0xffffffffu, row, 0);
                const int k0 = (jb + j) * kBKV;
                const bool inside = k0 >= whole_lo && k0 + kBKV <= whole_hi;
                const bool up = __all_sync(0xffffffffu, b >= kBoxes || row == t.row0 + b * 16);
                const bool down = __all_sync(0xffffffffu, b >= kBoxes || row == t.row0 - b * 16);
                t.whole = inside && (up || down);
                if (!up) t.row0 -= (kBoxes - 1) * 16;
                return t;
            };
            int seq = 0;
            auto slice = [&](const TmaMap* map, const TmaMap* run, const Tile& t, int col) {
                const int slot = seq % SLOTS;
                bar_wait(&empty[slot], ((seq / SLOTS) & 1) ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&full[slot], kSlice);
                __syncwarp();
                unsigned char* dst = smem + kOffRing + slot * kSlice;
                // Lane b: atom b / kBoxes, its 16 tokens b % kBoxes, whose row
                // lane b % kBoxes holds.
                const int r = __shfl_sync(0xffffffffu, t.row, b % kBoxes);
                if (t.whole) {
                    if (b < SA) tma_load_3d(dst + b * kAtomKV, &full[slot], run, col + b * 64, h, t.row0);
                } else if (b < SA * kBoxes) {
                    tma_load_3d(dst + (b / kBoxes) * kAtomKV + (b % kBoxes) * 16 * 128,
                                &full[slot], map, col + (b / kBoxes) * 64, h,
                                r < 0 ? 0x7fffff00 : r);
                }
                ++seq;
            };
            Tile prev{};
            int row = row_of(0);
            // The ring's first slots hold Q's TMEM part until it is copied.
            if (kQA > 0) bar_wait(q_held, 0);
            for (int j = 0; j < nt; ++j) {
                const Tile t = tile_of(j, row);
                row = j + 1 < nt ? row_of(j + 1) : -1;
#pragma unroll 1
                for (int a = 0; a < kKS; ++a) slice(&map_k, &map_k_run, t, a * 64 * SA);
                if (j > 0) {
#pragma unroll 1
                    for (int c = 0; c < kVS; ++c) {
                        slice(&map_v, &map_v_run, prev, v0 + c * 64 * SA);
                    }
                }
                prev = t;
            }
#pragma unroll 1
            for (int c = 0; c < kVS; ++c) slice(&map_v, &map_v_run, prev, v0 + c * 64 * SA);
        }
    } else if (warp == 5) {
        if (lane == 0 && nt > 0) {
            const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 128, kBKV);
            const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 128, 64 * SA) | (1u << 16);
            bar_wait(q_ready, 0);
            bar_wait(q_held, 0);
            tc_fence_after();
            // The ring slot the next slice lands in, and its phase.
            int slot = 0;
            u32 phase = 0;
            auto next = [&] {
                if (++slot == SLOTS) {
                    slot = 0;
                    phase ^= 1;
                }
            };
            auto pv = [&](int i) {
                bar_wait(&p_full[i & 1], (i >> 1) & 1);
                if (i == nt - 1) bar_wait(v_fixed, 0);
                tc_fence_after();
                const u32 pa = tmem + kColS + (i & 1) * kBKV;
#pragma unroll
                for (int c = 0; c < kVS; ++c) {
                    bar_wait(&full[slot], phase);
                    tc_fence_after();
                    const u32 vb = sbase + kOffRing + slot * kSlice;
#pragma unroll
                    for (u32 s = 0; s < kBKV / 16; ++s) {
                        mma_f16_ts(tmem + kColO + c * 64 * SA, pa + s * 8,
                                   sdesc_sw128_mn(vb + s * 2048, kAtomKV), idesc_o,
                                   i > 0 || s > 0);
                    }
                    tc_commit(&empty[slot]);
                    next();
                }
                tc_commit(&pv_done[i & 1]);
            };
            for (int j = 0; j < nt; ++j) {
                const u32 sb = j & 1;
                // This S buffer held P two tiles back: free once that product
                // has read it.
                if (j >= 2) bar_wait(&pv_done[sb], ((j - 2) >> 1) & 1);
#pragma unroll
                for (int a = 0; a < kKS; ++a) {
                    bar_wait(&full[slot], phase);
                    tc_fence_after();
#pragma unroll
                    for (int a2 = 0; a2 < SA; ++a2) {
                        const int ga = a * SA + a2;  // the atom of the head
                        const u32 kb = sbase + kOffRing + slot * kSlice + a2 * kAtomKV;
                        if (ga < kQA) {
#pragma unroll
                            for (u32 s = 0; s < 4; ++s) {
                                mma_f16_ts(tmem + kColS + sb * kBKV,
                                           tmem + kColQ + ga * 32 + s * 8, sdesc_sw128(kb) + 2 * s,
                                           idesc_s, ga > 0 || s > 0);
                            }
                        } else {
                            const u32 qa = sbase + kOffQ + (ga - kQA) * kAtomQ;
#pragma unroll
                            for (u32 s = 0; s < 4; ++s) {
                                mma_f16(tmem + kColS + sb * kBKV, sdesc_sw128(qa) + 2 * s,
                                        sdesc_sw128(kb) + 2 * s, idesc_s, ga > 0 || s > 0);
                            }
                        }
                    }
                    tc_commit(&empty[slot]);
                    next();
                }
                tc_commit(&s_full[sb]);
                if (j > 0) pv(j - 1);
            }
            pv(nt - 1);
        }
    } else {
        // The softmax warpgroup: row `row` of the tile is this thread's.
        const u32 row = warp * 32 + lane;
        const int prow = w.row0 + static_cast<int>(row);
        const bool live = prow < w.qo_len * p.group;
        const int tok = prow / p.group;
        const int qh = h * p.group + prow % p.group;
        const int lo = p.window_left < 0
            ? 0
            : max(0, w.kv_len - w.qo_len + tok - p.window_left);
        const int limit = !live ? 0
                          : p.causal ? min(w.kv_len, w.kv_len - w.qo_len + tok + 1)
                                     : w.kv_len;
        const u32 lane_base = (warp * 32) << 16;
        if (nt > 0) {
            // Q's TMEM part: from the ring's slots, where TMA landed it, or
            // from global memory; the rest into shared memory as TMA would
            // land it (128-byte rows, their 16-byte words swizzled by the
            // row) unless TMA did.
            const uint4* src = reinterpret_cast<const uint4*>(
                p.q + static_cast<long long>(w.qo_start + (live ? tok : 0)) * p.q_stride +
                static_cast<long long>(qh) * D);
            if (q_boxed) bar_wait(q_ready, 0);
#pragma unroll 1
            for (int a = 0; a < kQA; ++a) {
                u32 v[32];
#pragma unroll
                for (u32 g = 0; g < 8; ++g) {
                    uint4 x;
                    if (q_boxed) {
                        const u32 at = sbase + kOffRing + a * kAtomQ + row * 128 +
                                       ((g ^ (row % 8)) * 16);
                        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
                                     : "r"(at));
                    } else {
                        x = live ? src[a * 8 + g] : make_uint4(0u, 0u, 0u, 0u);
                    }
                    v[4 * g] = x.x;
                    v[4 * g + 1] = x.y;
                    v[4 * g + 2] = x.z;
                    v[4 * g + 3] = x.w;
                }
                tmem_st32(tmem + lane_base + kColQ + a * 32, v);
            }
            if (!q_boxed) {
                // Sixteen loads in flight a round, then their stores.
#pragma unroll 1
                for (u32 g0 = QT / 8; g0 < D / 8; g0 += 16) {
                    uint4 x[16];
#pragma unroll
                    for (u32 i = 0; i < 16; ++i) {
                        x[i] = live ? src[g0 + i] : make_uint4(0u, 0u, 0u, 0u);
                    }
#pragma unroll
                    for (u32 i = 0; i < 16; ++i) {
                        const u32 g = g0 + i;
                        const u32 at = sbase + kOffQ + (g / 8 - kQA) * kAtomQ + row * 128 +
                                       (((g % 8) ^ (row % 8)) * 16);
                        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(at),
                                     "r"(x[i].x), "r"(x[i].y), "r"(x[i].z), "r"(x[i].w)
                                     : "memory");
                    }
                }
                fence_async_shared();
                bar_arrive(q_ready);
            }
            tmem_wait_st();
            tc_fence_before();
            bar_arrive(q_held);
        }
        float m_used = neg_inf();
        float l = 0.f;
        for (int j = 0; j < nt; ++j) {
            const u32 st = j & 1;
            bar_wait(&s_full[st], (j >> 1) & 1);
            tc_fence_after();
            float s[kBKV];
#pragma unroll
            for (u32 c = 0; c < kBKV; c += 32) {
                u32 v[32];
                tmem_ld32(tmem + lane_base + kColS + st * kBKV + c, v);
#pragma unroll
                for (int i = 0; i < 32; ++i) s[c + i] = __uint_as_float(v[i]);
            }
            tmem_wait_ld();
            const int k0 = (jb + j) * kBKV;
            float m4[4] = {neg_inf(), neg_inf(), neg_inf(), neg_inf()};
            if (k0 >= lo && k0 + kBKV <= limit) {
#pragma unroll
                for (int c = 0; c < kBKV; ++c) m4[c & 3] = fmaxf(m4[c & 3], s[c]);
            } else {
#pragma unroll
                for (int c = 0; c < kBKV; ++c) {
                    s[c] = k0 + c < limit && k0 + c >= lo ? s[c] : neg_inf();
                    m4[c & 3] = fmaxf(m4[c & 3], s[c]);
                }
            }
            const float raw_mx = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
            const float mx = raw_mx == neg_inf() ? neg_inf() : raw_mx * p.scale_log2;
            const bool grow = mx > m_used + 8.f;
            const float m_new = grow ? mx : m_used;
            const float alpha = grow ? ex2(m_used - m_new) : 1.f;
            const float base = m_new == neg_inf() ? 0.f : m_new;
            float s4[4] = {0.f, 0.f, 0.f, 0.f};
            // P over its S, two bf16 a column, the A operand of P V: 64
            // columns of S make 32 of P, stored as they are formed.
#pragma unroll
            for (u32 c = 0; c < kBKV; c += 64) {
                u32 pk[32];
#pragma unroll
                for (int i = 0; i < 64; i += 2) {
                    const float e0 = ex2(fmaf(s[c + i], p.scale_log2, -base));
                    const float e1 = ex2(fmaf(s[c + i + 1], p.scale_log2, -base));
                    s4[(i >> 1) & 3] += e0 + e1;
                    pk[i / 2] = pack_bf16(e0, e1);
                }
                tmem_st32(tmem + lane_base + kColS + st * kBKV + c / 2, pk);
            }
            const float sum = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            l = l * alpha + sum;
            m_used = m_new;
            // O rescales once the last product is in.
            if (j > 0 && __any_sync(0xffffffffu, grow)) {
                bar_wait(&pv_done[(j - 1) & 1], ((j - 1) >> 1) & 1);
                tc_fence_after();
#pragma unroll 1
                for (u32 c = 0; c < DV; c += 32) {
                    u32 v[32];
                    tmem_ld32(tmem + lane_base + kColO + c, v);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        v[i] = __float_as_uint(__uint_as_float(v[i]) * alpha);
                    }
                    tmem_st32(tmem + lane_base + kColO + c, v);
                }
            }
            tmem_wait_st();
            if (j == nt - 1) {
                // The last tile's V rows past the request's keys hold whatever
                // the pool's page held: zero them so P's zeros stay zeros.
                const int valid = w.kv_len - k0;
                const int sv = seq_v<kKS, kVS>(j, nt);
#pragma unroll 1
                for (int c = 0; c < kVS; ++c) {
                    const int sq = sv + c;
                    const int slot = sq % SLOTS;
                    bar_wait(&full[slot], (sq / SLOTS) & 1);
                    if (static_cast<int>(row) < kBKV && static_cast<int>(row) >= valid) {
#pragma unroll
                        for (u32 x = 0; x < 8 * SA; ++x) {
                            const u32 at = sbase + kOffRing + slot * kSlice + (x / 8) * kAtomKV +
                                           row * 128 + (x % 8) * 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(at),
                                         "r"(0u)
                                         : "memory");
                        }
                    }
                }
                fence_async_shared();
                bar_arrive(v_fixed);
            }
            tc_fence_before();
            bar_arrive(&p_full[j & 1]);
        }
        if (nt > 0) {
            bar_wait(&pv_done[(nt - 1) & 1], ((nt - 1) >> 1) & 1);
            tc_fence_after();
        }
        const float inv = l > 0.f ? 1.f / l : 0.f;
        u16* dst = p.o + static_cast<long long>(w.qo_start + (live ? tok : 0)) * p.o_stride +
                   static_cast<long long>(qh) * D + v0;
#pragma unroll 1
        for (u32 c = 0; c < DV; c += 32) {
            u32 v[32];
            if (nt > 0) {
                tmem_ld32(tmem + lane_base + kColO + c, v);
                tmem_wait_ld();
            } else {
#pragma unroll
                for (int i = 0; i < 32; ++i) v[i] = 0u;
            }
            if (live) {
                u32 out[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    out[i] = pack_bf16(__uint_as_float(v[2 * i]) * inv,
                                       __uint_as_float(v[2 * i + 1]) * inv);
                }
                uint4* o4 = reinterpret_cast<uint4*>(dst + c);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o4[i] = make_uint4(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
                }
            }
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

}  // namespace pie::attn::fmha
