#pragma once

#include "prim/mma_sm100.cuh"

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
    // Where asked, each row's log2-sum-exp2 (FA2's, `[rows][q heads]`), and
    // each q head's bf16 sink the row is scaled down by.
    float* lse;
    const u16* sinks;
    // A head's stride in Q and O where wider than D and DV (heads padded
    // past the widths read); 0 where they are the widths.
    int head;
};

__host__ __device__ constexpr u32 smem_bytes(int d, int bkv, int stages, int dv = 0) {
    // The K and V rings, then barriers and O's scales; 1024-aligned.
    return u32(stages) * u32(d / 64 + (dv > 0 ? dv : d) / 64) * u32(bkv) * 128u + 1024u + 1024u;
}

__device__ __forceinline__ u32 pack_bf16(float lo, float hi) {
    u32 bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return bits;
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// Packed f32 pairs (sm_100's two-lane FMA and add).
__device__ __forceinline__ u64 f32x2(float lo, float hi) {
    u64 d;
    asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(lo), "f"(hi));
    return d;
}

__device__ __forceinline__ void f32x2_split(u64 v, float& lo, float& hi) {
    asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(v));
}

__device__ __forceinline__ u64 fma_f32x2(u64 a, u64 b, u64 c) {
    u64 d;
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    return d;
}

__device__ __forceinline__ u64 add_f32x2(u64 a, u64 b) {
    u64 d;
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
    return d;
}

// A row's log2-sum-exp2 landed where asked, and the share its head's sink
// leaves it: sigmoid(lse·ln2 − sink), 1 for an empty row or no sinks.
__device__ __forceinline__ float lse_sink(const Params& p, bool live, long long row, int qh, float m, float l) {
    const float lse = l > 0.f ? m + log2f(l) : neg_inf();
    if (live && p.lse != nullptr) p.lse[row * (p.kv_heads * p.group) + qh] = lse;
    if (p.sinks == nullptr || !(l > 0.f)) return 1.f;
    const float sink = __uint_as_float(static_cast<u32>(p.sinks[qh]) << 16);
    return 1.f / (1.f + __expf(sink - lse * 0.69314718055994530942f));
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 2^x on the FMA pipe, for x at most 2^7: a cubic on x's fraction (within
// 1e-4 of it, under bf16's step) shifted by its integer part; under 2^-126
// it lands 2^-126.
__device__ __forceinline__ float ex2_fma(float x) {
    x = fmaxf(x, -126.f);
    const float t = x + 12582912.f;  // 1.5 * 2^23: x's nearest integer in t's low bits
    const float f = x - (t - 12582912.f);
    const float q = fmaf(fmaf(fmaf(0.055008914f, f, 0.24221101f), f, 0.69328296f), f, 1.f);
    return __int_as_float(__float_as_int(q) + (__float_as_int(t) << 23));
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
// tile back, so the longest causal tiles start first. Each warp scans the
// live lanes 32 at a time (a captured fire's lanes are many, mostly empty,
// and its spare blocks find nothing): every thread of a full warp calls it.
struct Work {
    int r, row0, qo_start, qo_len, kv_len, kv_page0;
};

__device__ __forceinline__ bool work_of(const Params& p, int bound, int block, Work& w,
                                        int rows = 128) {
    const int t = bound - 1 - block;
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int lane0 = p.live[0];
    const int lanes = p.live[1];
    int start = 0;
    for (int base = 0; base < lanes; base += 32) {
        const bool in = base + lane < lanes;
        const int r = lane0 + base + lane;
        const int qs = in ? p.qo_indptr[r] : 0;
        const int ql = in ? p.qo_indptr[r + 1] - qs : 0;
        const int tiles = (ql * p.group + rows - 1) / rows;
        int incl = tiles;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int up = __shfl_up_sync(0xffffffffu, incl, d);
            incl += lane >= d ? up : 0;
        }
        const unsigned hit = __ballot_sync(0xffffffffu, in && t < start + incl);
        if (hit != 0u) {
            const int src = __ffs(hit) - 1;
            const int before = start + __shfl_sync(0xffffffffu, incl - tiles, src);
            w.r = lane0 + base + src;
            w.row0 = (t - before) * rows;
            w.qo_start = __shfl_sync(0xffffffffu, qs, src);
            w.qo_len = __shfl_sync(0xffffffffu, ql, src);
            w.kv_page0 = p.kv_indptr[w.r];
            const int pages = p.kv_indptr[w.r + 1] - w.kv_page0;
            w.kv_len = pages > 0 ? (pages - 1) * p.page + p.last_page_len[w.r] : 0;
            return true;
        }
        start += __shfl_sync(0xffffffffu, incl, 31);
    }
    return false;
}

// `DV` the value head's width where it differs from the query and key
// head's `D` (an expanded latent attention's: 192 against 128), its rows
// read at the V map's own head stride.
//
// With PAIR a block's item is two tiles of 128 rows that read one K and V
// stream, each its own softmax warpgroup (warps 0-3, 6-9), S buffers, O
// and Q: every key tile loads once for 256 rows. Without IO a block walks
// several items (the host's grid an SM a block), each softmax warpgroup
// taking its tile's Q in and O out.
template <int D, int BKV, int ST, int DV = D, bool IO = false, bool PAIR = false>
__global__ __launch_bounds__(IO || PAIR ? 320 : 192, 1) void tc5_fmha_prefill(
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
    // TMEM, a tile's each: two S buffers (P overwrites its tile's, two bf16
    // a column), O, and Q (two bf16 a column), the A operands read in place.
    constexpr u32 kT = PAIR ? 2 : 1;
    constexpr u32 kColS = 0;
    constexpr u32 kColO = kT * 2 * kBKV;
    constexpr u32 kColQ = kColO + kT * DV;
    constexpr u32 kCols = 512;
    static_assert(kColQ + kT * D / 2 <= kCols, "S twice, O and Q fit TMEM");
    static_assert(!(IO && PAIR), "a pair's warpgroups move their own Q and O");
    constexpr int kRows = 128 * kT;
    // Without IO the softmax warps move each item's Q and O themselves.
    constexpr bool kOwn = !IO;
    (void)kKA;

    // A block's items: a tile of a KV head's rows, numbered tile-major from
    // (x, y), the grid's size apart; `next_work` steps `i` past empty ones.
    // A block walks several (the host's grid an SM a block), the K and V
    // ring, the S buffers and the P V products counting across them.
    // Without IO `i` is instead the block's round: the requests' tiles ranked
    // from each one's last (the longest causal ones first), a rank's every
    // request and head in turn (past 16 KV heads, a head at a time, its keys
    // kept in L2 while the blocks share them), dealt to the blocks back and
    // forth so their sums even out. Up to 32 requests with rows a warp holds
    // them a lane each, so a captured fire's empty lanes cost no rounds.
    const int first = kOwn ? 0 : static_cast<int>(blockIdx.x * gridDim.y + blockIdx.y);
    const int step = kOwn ? 1 : static_cast<int>(gridDim.x * gridDim.y);
    int h = 0;
    int ranks = 0;
    // The live requests (rows past 0), the n-th at lane n: its lane, first
    // row, rows and tiles.
    int nlive = 0, my_r = 0, my_qs = 0, my_ql = 0, my_tiles = 0;
    if constexpr (kOwn) {
        const int lane = static_cast<int>(threadIdx.x & 31);
        for (int base = 0; base < p.live[1]; base += 32) {
            const int r = p.live[0] + base + lane;
            const bool in = base + lane < p.live[1];
            const int qs = in ? p.qo_indptr[r] : 0;
            const int ql = in ? p.qo_indptr[r + 1] - qs : 0;
            const int tiles = (ql * p.group + kRows - 1) / kRows;
            ranks = max(ranks, tiles);
            const unsigned on = __ballot_sync(0xffffffffu, tiles > 0);
            const int at = nlive + __popc(on & ((1u << lane) - 1u));
            if (tiles > 0 && at < 32) {
                my_r = r;
                my_qs = qs;
                my_ql = ql;
                my_tiles = tiles;
            }
            nlive += __popc(on);
        }
        ranks = __reduce_max_sync(0xffffffffu, ranks);
    }
    auto next_work = [&](int& i, Work& x) {
        if constexpr (kOwn) {
            const bool held = nlive <= 32;
            const int lanes = held ? nlive : p.live[1];
            const int nb = static_cast<int>(gridDim.x);
            const int b = static_cast<int>(blockIdx.x);
            for (;; ++i) {
                const int item = i * nb + ((i & 1) ? nb - 1 - b : b);
                if (item >= ranks * lanes * p.kv_heads) return false;
                const bool by_head = p.kv_heads > 16;
                h = by_head ? item / (ranks * lanes) : item % p.kv_heads;
                const int q = by_head ? item % (ranks * lanes) : item / p.kv_heads;
                int r, qs, ql, tiles;
                if (held) {
                    r = __shfl_sync(0xffffffffu, my_r, q % lanes);
                    qs = __shfl_sync(0xffffffffu, my_qs, q % lanes);
                    ql = __shfl_sync(0xffffffffu, my_ql, q % lanes);
                    tiles = __shfl_sync(0xffffffffu, my_tiles, q % lanes);
                } else {
                    r = p.live[0] + q % lanes;
                    qs = p.qo_indptr[r];
                    ql = p.qo_indptr[r + 1] - qs;
                    tiles = (ql * p.group + kRows - 1) / kRows;
                }
                if (q / lanes >= tiles) continue;
                x.r = r;
                x.row0 = (tiles - 1 - q / lanes) * kRows;
                x.qo_start = qs;
                x.qo_len = ql;
                x.kv_page0 = p.kv_indptr[r];
                const int pages = p.kv_indptr[r + 1] - x.kv_page0;
                x.kv_len = pages > 0 ? (pages - 1) * p.page + p.last_page_len[r] : 0;
                return true;
            }
        }
        for (; i < bound * p.kv_heads; i += step) {
            h = i % p.kv_heads;
            const int t = i / p.kv_heads;
            if (work_of(p, bound, t, x, kRows)) return true;
        }
        return false;
    };
    // The first item, found before the block sets up: none, nothing to do.
    Work w0;
    int t0 = first;
    if (!next_work(t0, w0)) return;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);

    u64* k_full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* k_empty = k_full + ST;
    u64* v_full = k_empty + ST;
    u64* v_empty = v_full + ST;
    // A row tile x's S buffer b: s_full, p_full and pv_done [2 x + b].
    u64* s_full = v_empty + ST;
    u64* q_ready = s_full + 2 * kT;
    u64* p_full = q_ready + kT;
    // P V of key tile i commits to pv_done[i & 1]: each fills once per two
    // tiles, so a wait for either of the last two products names its phase.
    u64* pv_done = p_full + 2 * kT;
    u64* v_fixed = pv_done + 2 * kT;
    // A tile's last S is in (its Q's columns free); its O is whole, at the
    // scale `o_inv` holds; its O is out.
    u64* q_free = v_fixed + 1;
    u64* o_ready = q_free + 1;
    u64* o_free = o_ready + 1;
    u32* tmem_slot = reinterpret_cast<u32*>(o_free + 1);
    float* o_inv = reinterpret_cast<float*>(o_free + 2);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    // Key tiles: from the first row's oldest key (a window's) through the
    // last key any row of the tile reads.
    auto tiles_of = [&](const Work& x, int& first) {
        const int last_row = min(x.row0 + kRows - 1, x.qo_len * p.group - 1);
        const int last_tok = last_row / p.group;
        const int reach = p.causal ? min(x.kv_len, x.kv_len - x.qo_len + last_tok + 1) : x.kv_len;
        const int first_lo = p.window_left < 0
            ? 0
            : max(0, x.kv_len - x.qo_len + x.row0 / p.group - p.window_left);
        first = first_lo / kBKV;
        return max(0, (reach + kBKV - 1) / kBKV - first);
    };
    if (threadIdx.x == 0) {
        for (int s = 0; s < ST; ++s) {
            bar_init(&k_full[s], 1);
            bar_init(&k_empty[s], 1);
            bar_init(&v_full[s], 1);
            bar_init(&v_empty[s], 1);
        }
        for (int s = 0; s < 2 * kT; ++s) {
            bar_init(&s_full[s], 1);
            bar_init(&p_full[s], 128);
            bar_init(&pv_done[s], 1);
        }
        for (int x = 0; x < kT; ++x) bar_init(&q_ready[x], 128);
        bar_init(v_fixed, 128);
        bar_init(q_free, 1);
        bar_init(o_ready, 128);
        bar_init(o_free, kRows);
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

    // A softmax or IO thread's row of the tile, its warp's TMEM lanes; with
    // PAIR its warpgroup's tile `xt`, its row `irow` of the item's.
    const u32 row = (warp % 4) * 32 + lane;
    const u32 lane_base = ((warp % 4) * 32) << 16;
    const u32 xt = PAIR && warp >= 6 ? 1 : 0;
    const int irow = static_cast<int>(xt * 128 + row);
    const u32 col_o = kColO + xt * DV;
    const u32 col_q = kColQ + xt * (D / 2);
    // A tile's Q rows into TMEM, 64 values a store.
    auto load_q = [&](const Work& x) {
        const int prow = x.row0 + irow;
        const bool live = prow < x.qo_len * p.group;
        const u16* src =
            p.q + static_cast<long long>(x.qo_start + (live ? prow / p.group : 0)) * p.q_stride +
            static_cast<long long>(h * p.group + prow % p.group) * (p.head > 0 ? p.head : D);
        // Every load in flight at once, 32 bytes each, then the stores.
        u32 qv[D / 2];
#pragma unroll
        for (int i = 0; i < D / 16; ++i) {
            u32* q8 = qv + 8 * i;
            if (live) {
                asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                             : "=r"(q8[0]), "=r"(q8[1]), "=r"(q8[2]), "=r"(q8[3]), "=r"(q8[4]),
                               "=r"(q8[5]), "=r"(q8[6]), "=r"(q8[7])
                             : "l"(src + 16 * i));
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) q8[k] = 0u;
            }
        }
#pragma unroll
        for (u32 c = 0; c < D / 2; c += 32) {
            u32 v[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = qv[c + i];
            tmem_st32(tmem + lane_base + col_q + c, v);
        }
        tmem_wait_st();
        tc_fence_before();
        bar_arrive(&q_ready[xt]);
    };
    // A tile's O out at `inv`, its rows' scale. Whole-warp TMEM loads: a
    // dead row still joins its warp's, and stores nothing.
    auto store_o = [&](const Work& w, int nt, float inv, int wh) {
        const int prow = w.row0 + irow;
        const bool live = prow < w.qo_len * p.group;
        const long long o_row = w.qo_start + (live ? prow / p.group : 0);
        const int qh = wh * p.group + prow % p.group;
        u16* dst = p.o + o_row * p.o_stride + static_cast<long long>(qh) * (p.head > 0 ? p.head : DV);
#pragma unroll 1
        for (u32 c = 0; c < DV; c += 32) {
            u32 v[32];
            if (nt > 0) {
                tmem_ld32(tmem + lane_base + col_o + c, v);
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
                // 32 bytes a store, a whole sector.
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(
                                     dst + c + 16 * i),
                                 "r"(out[8 * i]), "r"(out[8 * i + 1]), "r"(out[8 * i + 2]),
                                 "r"(out[8 * i + 3]), "r"(out[8 * i + 4]), "r"(out[8 * i + 5]),
                                 "r"(out[8 * i + 6]), "r"(out[8 * i + 7])
                                 : "memory");
                }
            }
        }
    };

    if (warp == 4) {
        if (lane == 0) {
            tma_prefetch(&map_k);
            tma_prefetch(&map_v);
            tma_prefetch(&map_k_run);
            tma_prefetch(&map_v_run);
        }
        int g = 0;
        Work w = w0;
        for (int t = t0, more = 1; more; t += step, more = next_work(t, w)) {
            int jb = 0;
            const int nt = tiles_of(w, jb);
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
            const int live_rows = min(kRows, w.qo_len * p.group - w.row0);
            const int tok_first = w.row0 / p.group;
            const int tok_last = (w.row0 + live_rows - 1) / p.group;
            const int whole_lo = p.window_left < 0
                ? 0
                : max(0, w.kv_len - w.qo_len + tok_last - p.window_left);
            const int whole_hi = p.causal ? min(w.kv_len, w.kv_len - w.qo_len + tok_first + 1)
                                          : w.kv_len;
            int row = b < kBoxes ? row_of(0) : -1;
            for (int j = 0; j < nt; ++j) {
                const u32 st = (g + j) % ST;
                const u32 ph = ((g + j) / ST) & 1;
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
            g += nt;
        }
    } else if (warp == 5) {
        const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 128, kBKV);
        const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 128, DV) | (1u << 16);
        int g = 0, tile = 0, keyed = 0;
        // The whole warp walks a tile's products; the elected lane issues.
        auto issue = [&](const Work& w) {
            int jb = 0;
            const int nt = tiles_of(w, jb);
#pragma unroll
            for (u32 x = 0; x < kT; ++x) bar_wait(&q_ready[x], tile & 1);
            // The last tile's O out before this one's first product lands.
            if (nt == 0) {
                if (tile > 0) bar_wait(o_free, (tile - 1) & 1);
                __syncwarp();
                if (elect_one()) tc_commit(q_free);
                ++tile;
                return;
            }
            auto pv = [&](int i) {
                const int gi = g + i;
                const u32 st = gi % ST;
                if (i == 0 && tile > 0) bar_wait(o_free, (tile - 1) & 1);
                bar_wait(&v_full[st], (gi / ST) & 1);
                if (i == nt - 1) bar_wait(v_fixed, keyed & 1);
                const u32 vb = sbase + kOffV + st * kTileV;
#pragma unroll
                for (u32 x = 0; x < kT; ++x) {
                    const u32 sb = 2 * x + (gi & 1);
                    bar_wait(&p_full[sb], (gi >> 1) & 1);
                    tc_fence_after();
                    const u32 pa = tmem + kColS + sb * kBKV;
                    __syncwarp();
                    if (elect_one()) {
#pragma unroll
                        for (u32 s = 0; s < kBKV / 16; ++s) {
                            const u64 b = sdesc_sw128_mn(vb + s * 2048, kAtomKV);
                            mma_f16_ts(tmem + kColO + x * DV, pa + s * 8, b, idesc_o,
                                       i > 0 || s > 0);
                        }
                        tc_commit(&pv_done[sb]);
                        if (x == kT - 1) tc_commit(&v_empty[st]);
                    }
                }
            };
            for (int j = 0; j < nt; ++j) {
                const int gj = g + j;
                const u32 st = gj % ST;
                bar_wait(&k_full[st], (gj / ST) & 1);
                const u32 kb = sbase + kOffK + st * kTileKV;
#pragma unroll
                for (u32 x = 0; x < kT; ++x) {
                    const u32 sb = 2 * x + (gj & 1);
                    // This S buffer held P two tiles back: free once that
                    // product has read it. Only it can have landed yet.
                    if (gj >= 2) bar_wait(&pv_done[sb], ((gj - 2) >> 1) & 1);
                    tc_fence_after();
                    __syncwarp();
                    if (elect_one()) {
#pragma unroll
                        for (u32 s = 0; s < D / 16; ++s) {
                            const u64 b = sdesc_sw128(kb + (s / 4) * kAtomKV) + 2 * (s % 4);
                            mma_f16_ts(tmem + kColS + sb * kBKV,
                                       tmem + kColQ + x * (D / 2) + s * 8, b, idesc_s, s > 0);
                        }
                        tc_commit(&s_full[sb]);
                        if (x == kT - 1) {
                            tc_commit(&k_empty[st]);
                            if (j == nt - 1) tc_commit(q_free);
                        }
                    }
                }
                if (j > 0) pv(j - 1);
            }
            pv(nt - 1);
            g += nt;
            ++keyed;
            ++tile;
        };
        // The whole warp finds each tile, its lanes scanning the requests.
        Work w = w0;
        for (int t = t0, more = 1; more; t += step, more = next_work(t, w)) {
            issue(w);
        }
    } else if (warp < 4 || (PAIR && warp >= 6)) {
        // The softmax warpgroup (with PAIR, tile xt's).
        int g = 0, tile = 0;
        Work w = w0;
        for (int t = t0, more = 1; more; t += step, more = next_work(t, w)) {
            int jb = 0;
            const int nt = tiles_of(w, jb);
            // Each item's Q after the first went in under the last one's end.
            if (kOwn && tile == 0) load_q(w);
            const int wh = h;
            const int prow = w.row0 + irow;
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
            float m_used = neg_inf();
            float l = 0.f;
            for (int j = 0; j < nt; ++j) {
                const int gj = g + j;
                const u32 st = 2 * xt + (gj & 1);
                bar_wait(&s_full[st], (gj >> 1) & 1);
                tc_fence_after();
                // Every load out at once, then one wait.
                u32 sv[kBKV / 32][32];
#pragma unroll
                for (u32 c = 0; c < kBKV / 32; ++c) {
                    tmem_ld32(tmem + lane_base + kColS + st * kBKV + c * 32, sv[c]);
                }
                tmem_wait_ld();
                float s[kBKV];
#pragma unroll
                for (u32 c = 0; c < kBKV; ++c) s[c] = __uint_as_float(sv[c / 32][c % 32]);
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
                // Two f32 lanes an instruction (the scale's FMA, the sums); an
                // eighth of the exponentials on the FMA pipe beside the rest's
                // (a pair's 64-wide heads, a quarter).
                u64 s2[4] = {0, 0, 0, 0};
                const u64 sc2 = f32x2(p.scale_log2, p.scale_log2);
                const u64 nb2 = f32x2(-base, -base);
                u32 pk[kBKV / 2];
#pragma unroll
                for (int c = 0; c < kBKV; c += 2) {
                    float x0, x1;
                    f32x2_split(fma_f32x2(f32x2(s[c], s[c + 1]), sc2, nb2), x0, x1);
                    const bool fma_pipe = PAIR ? (c & 7) == 6 : (c & 15) == 14;
                    const float e0 = fma_pipe ? ex2_fma(x0) : ex2(x0);
                    const float e1 = fma_pipe ? ex2_fma(x1) : ex2(x1);
                    s2[(c >> 1) & 3] = add_f32x2(s2[(c >> 1) & 3], f32x2(e0, e1));
                    pk[c / 2] = pack_bf16(e0, e1);
                }
                float sum = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float a0, a1;
                    f32x2_split(s2[i], a0, a1);
                    sum += a0 + a1;
                }
                l = l * alpha + sum;
                m_used = m_new;
                // O rescales once the last product is in.
                if (j > 0 && __any_sync(0xffffffffu, grow)) {
                    bar_wait(&pv_done[2 * xt + ((gj - 1) & 1)], ((gj - 1) >> 1) & 1);
                    tc_fence_after();
#pragma unroll 1
                    for (u32 c = 0; c < DV; c += 32) {
                        u32 v[32];
                        tmem_ld32(tmem + lane_base + col_o + c, v);
                        tmem_wait_ld();
#pragma unroll
                        for (int i = 0; i < 32; ++i) {
                            v[i] = __float_as_uint(__uint_as_float(v[i]) * alpha);
                        }
                        tmem_st32(tmem + lane_base + col_o + c, v);
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
                if (j == nt - 1 && xt == 0) {
                    // The last tile's V rows past the request's keys hold whatever
                    // the pool's page held: zero them so P's zeros stay zeros.
                    const u32 vst = gj % ST;
                    bar_wait(&v_full[vst], (gj / ST) & 1);
                    const int valid = w.kv_len - k0;
                    if (static_cast<int>(row) < kBKV && static_cast<int>(row) >= valid) {
#pragma unroll
                        for (u32 a = 0; a < kNAV; ++a) {
#pragma unroll
                            for (u32 c = 0; c < 8; ++c) {
                                const u32 at = sbase + kOffV + vst * kTileV + a * kAtomKV +
                                               row * 128 + c * 16;
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
                bar_arrive(&p_full[st]);
            }
            if (kOwn) {
                // The next item's Q, once this one's last S has read this Q,
                // under this one's last products.
                Work wn;
                int tn = t + step;
                if (next_work(tn, wn)) {
                    bar_wait(q_free, tile & 1);
                    tc_fence_after();
                    load_q(wn);
                }
            }
            if (nt > 0) {
                const int gl = g + nt - 1;
                if (nt >= 2) bar_wait(&pv_done[2 * xt + ((gl - 1) & 1)], ((gl - 1) >> 1) & 1);
                bar_wait(&pv_done[2 * xt + (gl & 1)], (gl >> 1) & 1);
            }
            // O goes out at the row's scale: with IO, once the last tile's is.
            const long long o_row = w.qo_start + (live ? tok : 0);
            const float inv = (l > 0.f ? 1.f / l : 0.f) * lse_sink(p, live, o_row, qh, m_used, l);
            if (IO) {
                if (tile > 0) bar_wait(o_free, (tile - 1) & 1);
                o_inv[row] = inv;
            } else {
                tc_fence_after();
                store_o(w, nt, inv, wh);
                tmem_wait_ld();
            }
            tc_fence_before();
            if (IO) bar_arrive(o_ready);
            if (kOwn) bar_arrive(o_free);
            ++tile;
            g += nt;
        }
    } else if (IO) {
        // With IO, its own warpgroup: the next tile's Q in once this one's
        // last S is in, then this one's O out, both under the other warps'
        // next tile.
        int tile = 0;
        Work w = w0;
        load_q(w);
        for (int t = t0, more = 1; more; t += step, more = next_work(t, w)) {
            int jb = 0;
            const int nt = tiles_of(w, jb);
            bar_wait(q_free, tile & 1);
            {
                Work wn;
                int tn = t + step;
                if (next_work(tn, wn)) load_q(wn);
                h = t % p.kv_heads;
            }
            bar_wait(o_ready, tile & 1);
            tc_fence_after();
            store_o(w, nt, o_inv[row], h);
            tmem_wait_ld();
            tc_fence_before();
            bar_arrive(o_free);
            ++tile;
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
        const long long o_row = w.qo_start + (live ? tok : 0);
        const float inv = (l > 0.f ? 1.f / l : 0.f) * lse_sink(p, live, o_row, qh, m_used, l);
        u16* dst = p.o + o_row * p.o_stride + static_cast<long long>(qh) * D + v0;
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
