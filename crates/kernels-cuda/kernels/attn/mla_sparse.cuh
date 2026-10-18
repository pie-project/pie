#pragma once

#include "attn/dsv4_sparse.cuh"

// A DeepSeek-sparse MLA's attention on tcgen05, a block a row (GLM-5.3's
// DSA: 64 absorbed heads over one 512-wide latent, plus a 64-wide rotated
// part or none, the keys a row's index selected): `attn/dsv4_sparse.cuh`'s
// pass with no window and no sinks, the selected token ids its stream (in
// any order, a negative one anywhere skipped).
//
//   - Q's latent sits in TMEM as there (lanes 0-63 a head's first 256 dims,
//     lanes 64-127 its last), and its rotated part beside it: lanes 0-63
//     hold dims 0-31 then 32 zeros, lanes 64-127 32 zeros then dims 32-63.
//   - A tile is 64 keys' latent (64 KB) and, where Q has a rotated part,
//     their 64 rotated dims twice (16 KB): rows 0-63 and 64-127 of one
//     operand, so the same weight-stationary MMA lands each half's share of
//     the rotated dot in the lanes that hold the latent's.
//   - S has one TMEM buffer (Q's rotated part takes the second's columns);
//     the softmax reads it out at once, so the next tile's product waits
//     only on that read.
//
// The e4m3 instance (`kE4`) reads vLLM's standard fp8 rows (512 latent codes
// and the rotated dims' 64 or none, one per-tensor scale of 1) and
// runs both products on e4m3: Q quantized as it lands in TMEM (lanes 0-63 a
// head's first 256 latent codes, lanes 64-127 its last, and each half's 32
// rotated codes), a tile's 64 keys as four 128-code atoms (atoms a and a + 2
// side by side, so lanes 0-63 meet the first half), the rotated dims twice
// (rows 0-63 the first 32, rows 64-127 the last 32, each at byte 64 of its
// row), P in e4m3.
//
// Warps 0-3 run the softmax and store O (and the base-2 log-sum-exp where
// asked); warp 11 lists a tile's keys; warps 4 and 6-10 issue its gathers;
// warp 5 owns TMEM and issues every MMA.
namespace pie::attn::mlas {

using tc5::u32;
using tc5::u64;
using fmha::u16;
using fmha::ex2;
using fmha::neg_inf;
using fmha::pack_bf16;
using fmha::sdesc_sw128_mn;
using dsv4s::atom_at;
using dsv4s::gather4;
using dsv4s::mma_ws;
using dsv4s::mma_ws_ts;
using dsv4s::pair_sync;
using dsv4s::row_in;
using dsv4s::tile_pv;
using dsv4s::tile_softmax;

struct Params {
    u16* o;
    float* lse;
    const u16* q;
    const u16* q_pe;
    const u32* qo_indptr;
    const u32* page_indptr;
    const u32* page_indices;
    const u32* last_page_lens;
    const int* selection;
    const u32* live;
    int requests;
    int causal;
    int runs;
    int page;
    int top_k;
    int rows;
    int q_stride;
    int qpe_stride;
    int o_stride;
    float scale_log2;
    // Split-K: each row's tiles dealt over `splits` units, each unit's
    // normalized output (bf16) and base-2 log-sum-exp a head into `part_o` /
    // `part_lse` at (row * splits + split), folded by `mla_sparse_fold`.
    // `splits` 1 writes the row's output straight.
    u16* part_o;
    float* part_lse;
    int splits;
    // Rows that see no more keys than `top_k` read every key they see, in
    // order, their selection unread.
    int dense;
};

constexpr int kD = 512;
constexpr int kBK = 64;
constexpr u32 kAtom = kBK * 128;          // 64 keys by 64 bf16
constexpr u32 kTile = (kD / 64) * kAtom;  // 64 KB
constexpr u32 kRopeBytes = 2 * kAtom;     // 64 keys' rotated dims, twice
constexpr u32 kPBytes = 64 * 128;
constexpr u32 kXBytes = 32 * 128 * 4;
constexpr int kLists = 6;
constexpr u32 kColQ = 0;
constexpr u32 kColQr = 128;
constexpr u32 kColS = 192;
constexpr u32 kColO = 256;
constexpr u32 kCols = 512;
constexpr u32 kSlotMask = (1u << 30) - 1;
constexpr u32 kValid = 1u << 31;
constexpr int kAbsentRow = 0x7fffff00;
constexpr int kIssuers = 6;
constexpr u32 kE4Latent = 4 * kAtom;  // 64 keys by 512 codes
constexpr u32 kColQr8 = 64;
constexpr u32 kColS8 = 128;

// Where an e4m3 tile's atom `a` (codes 128 a ..) sits: atoms a and a + 2
// side by side.
__device__ __forceinline__ u32 e4_at(int a) { return (a & 1) * 2 * kAtom + (a >> 1) * kAtom; }

template <int kRope, bool kE4>
struct Layout {
    static constexpr int kSlots = kE4 ? 3 : 2;
    static constexpr u32 kSlot = kE4 ? kE4Latent + (kRope > 0 ? 2 * kAtom : 0u) : kTile + (kRope > 0 ? kRopeBytes : 0u);
    static constexpr u32 kOffRing = 0;
    static constexpr u32 kOffP = kOffRing + kSlots * kSlot;
    static constexpr u32 kOffX = kOffP + 2 * kPBytes;
    static constexpr u32 kOffList = kOffX + kXBytes;
    static constexpr u32 kOffBar = kOffList + kLists * kBK * 4;
    static constexpr u32 kSmem = kOffBar + 1024u + 1024u;
};

__device__ __forceinline__ void issue_sync() { asm volatile("bar.sync 2, %0;" ::"n"(kIssuers * 32) : "memory"); }

__device__ __forceinline__ void mma_ws_ts_f8(u32 d, u32 a, u64 b, u32 idesc, bool acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n\t}" ::"r"(d),
        "r"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc))
        : "memory");
}

__device__ __forceinline__ void mma_ws_f8(u32 d, u64 a, u64 b, u32 idesc, bool acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc))
        : "memory");
}

// Two words of bf16 pairs (four values, the first low) as one word of e4m3
// codes, saturating.
__device__ __forceinline__ u32 bf16x4_e4m3(u32 x, u32 y) {
    return dsv4s::e4m3x4(__uint_as_float(x << 16), __uint_as_float(x & 0xffff0000u),
                         __uint_as_float(y << 16), __uint_as_float(y & 0xffff0000u));
}

template <int kRope, bool kE4 = false>
__global__ __launch_bounds__(384, 1) void mla_sparse(
    const __grid_constant__ tc5::TmaMap map_ckv, const __grid_constant__ tc5::TmaMap map_ckv_run,
    const __grid_constant__ tc5::TmaMap map_kpe, const Params p) {
    using namespace tc5;
    using L = Layout<kRope, kE4>;
    constexpr int kSlots = L::kSlots;
    static_assert(kRope == 0 || kRope == 64, "the rotated part is 64 wide or absent");
    constexpr u32 kColSx = kE4 ? kColS8 : kColS;
    const int live_rows = p.live != nullptr ? static_cast<int>(p.live[0]) : p.rows;
    const int splits = p.splits > 0 ? p.splits : 1;
    // The units the block walks: a row a split.
    const int live = live_rows * splits;
    const int base = p.live != nullptr ? static_cast<int>(p.live[1]) : 0;
    const int requests =
        p.live != nullptr ? min(p.requests, static_cast<int>(p.live[2])) : p.requests;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);
    float* xbuf = reinterpret_cast<float*>(smem + L::kOffX);
    u32* lists = reinterpret_cast<u32*>(smem + L::kOffList);
    u64* full = reinterpret_cast<u64*>(smem + L::kOffBar);
    u64* empty = full + kSlots;
    u64* s_full = empty + kSlots;
    u64* s_free = s_full + 1;
    u64* p_full = s_free + 1;
    u64* pv_done = p_full + 2;
    u64* q_ready = pv_done + 2;
    u64* o_free = q_ready + 1;
    u64* list_full = o_free + 1;
    u64* list_free = list_full + kLists;
    u32* tmem_slot = reinterpret_cast<u32*>(list_free + kLists);
    float* row_max = reinterpret_cast<float*>(tmem_slot + 4);  // [128]
    // A unit's row, found once by the lists warp and read by the others: two
    // in flight, each held until its eleven readers (the six issuers, the
    // MMA warp, the four softmax warps) have taken it.
    u64* row_full = reinterpret_cast<u64*>(row_max + 128);
    u64* row_free = row_full + 2;

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < kSlots; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], 1);
        }
        bar_init(s_full, 1);
        bar_init(s_free, 128);
        for (int s = 0; s < 2; ++s) {
            bar_init(&p_full[s], 128);
            bar_init(&pv_done[s], 1);
        }
        bar_init(q_ready, 128);
        bar_init(o_free, 128);
        for (int s = 0; s < kLists; ++s) {
            bar_init(&list_full[s], 1);
            bar_init(&list_free[s], 129);
        }
        for (int s = 0; s < 2; ++s) {
            bar_init(&row_full[s], 1);
            bar_init(&row_free[s], kIssuers + 1 + 4);
        }
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
    // The fold (a programmatic dependent under `kernel.pdl`) may launch
    // now: it waits for this grid before it reads.
    asm volatile("griddepcontrol.launch_dependents;");

    // Row r of the live ones: its request (by the query rows' indptr), its
    // position in that request's keys, the keys it may see, and its selected
    // ids up to the first negative one.
    struct Row {
        int row, pos, last, entries, nt, j0, split, dense;
        const int* sel;
        const u32* pages;
    };
    auto row_at = [&](int u) {
        Row w;
        const int r = u / splits;
        w.split = u % splits;
        w.row = base + r;
        // The first request whose rows end past r, 32 requests a round with
        // their loads in flight (a search's halvings are as many dependent
        // loads, each a cold read at the pass's start); then its bounds,
        // read together.
        int req = requests - 1;
        for (int b = 0; b < requests; b += 32) {
            const int i = b + static_cast<int>(lane);
            const bool past = i < requests && r < static_cast<int>(__ldg(&p.qo_indptr[i + 1]));
            const unsigned m = __ballot_sync(0xffffffffu, past);
            if (m != 0u) {
                req = b + __ffs(m) - 1;
                break;
            }
        }
        const int qo_lo = static_cast<int>(__ldg(&p.qo_indptr[req]));
        const int qo_hi = static_cast<int>(__ldg(&p.qo_indptr[req + 1]));
        const int pages_first = static_cast<int>(__ldg(&p.page_indptr[req]));
        const int pages_end = static_cast<int>(__ldg(&p.page_indptr[req + 1]));
        const int last_len = static_cast<int>(__ldg(&p.last_page_lens[req]));
        const int fresh = qo_hi - qo_lo;
        const int num_pages = pages_end - pages_first;
        const int kv_len = (num_pages - 1) * p.page + last_len;
        w.pos = kv_len - fresh + (r - qo_lo);
        // The last key the row sees: its own (causal), or the request's.
        w.last = p.causal ? w.pos : kv_len - 1;
        w.pages = p.page_indices + pages_first;
        w.sel = p.selection + static_cast<long long>(w.row) * p.top_k;
        // The selection may hold -1 anywhere (a pooled index's dropped
        // blocks): the stream runs to its last id, a -1 before it masked.
        // Found from the end, 512 ids a round with their loads all in flight
        // (a full selection's last id is in the first round).
        // A row that sees no more keys than its selection holds reads them
        // all, in order: its index ranked every one.
        w.dense = p.dense != 0 && w.last + 1 <= p.top_k;
        w.entries = w.dense ? max(w.last + 1, 0) : 0;
        for (int end = p.top_k; !w.dense && end > 0 && w.entries == 0; end -= 16 * 32) {
            const int n0 = end - 16 * 32;
            int v[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int n = n0 + u * 32 + static_cast<int>(lane);
                v[u] = n >= 0 ? __ldg(w.sel + n) : -1;
            }
#pragma unroll
            for (int u = 15; u >= 0; --u) {
                const unsigned held = __ballot_sync(0xffffffffu, v[u] >= 0);
                if (held != 0u && w.entries == 0) w.entries = n0 + u * 32 + 32 - __clz(held);
            }
        }
        // A row always walks one tile: its O and its log-sum-exp land. A
        // split walks its share of the row's tiles; one past them (a row of
        // fewer tiles than splits) walks none and lands only its empty
        // log-sum-exp, which the fold weighs at nothing.
        const int tiles = max((w.entries + kBK - 1) / kBK, 1);
        const int share = (tiles + splits - 1) / splits;
        w.j0 = w.split * share;
        w.nt = max(min(tiles, w.j0 + share) - w.j0, 0);
        return w;
    };
    Row* shared_rows = reinterpret_cast<Row*>(row_free + 2);
    // Unit `n` of this block's walk (its `n`th), as the lists warp found it.
    auto take_row = [&](int n) {
        bar_wait(&row_full[n % 2], (n / 2) & 1);
        const Row w = shared_rows[n % 2];
        __syncwarp();
        if (lane == 0) bar_arrive(&row_free[n % 2]);
        return w;
    };

    if (warp == 11) {
        int J = 0;
        int n = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x), ++n) {
            if (n >= 2) bar_wait(&row_free[n % 2], ((n / 2) - 1) & 1);
            const Row w = row_at(r);
            if (lane == 0) shared_rows[n % 2] = w;
            __syncwarp();
            if (lane == 0) bar_arrive(&row_full[n % 2]);
            // Four tiles' ids a round, then their pool rows, every load of a
            // round in flight at once: a tile's list costs a round's two
            // latencies over four, not two of its own.
            constexpr int kRound = 4;
            for (int j0 = 0; j0 < w.nt; j0 += kRound) {
                int id[kRound][kBK / 32];
#pragma unroll
                for (int b = 0; b < kRound; ++b) {
#pragma unroll
                    for (int c = 0; c < kBK / 32; ++c) {
                        const int k = (w.j0 + j0 + b) * kBK + c * 32 + static_cast<int>(lane);
                        id[b][c] = j0 + b < w.nt && k < w.entries ? (w.dense ? k : __ldg(w.sel + k)) : -1;
                    }
                }
                u32 e[kRound][kBK / 32];
#pragma unroll
                for (int b = 0; b < kRound; ++b) {
#pragma unroll
                    for (int c = 0; c < kBK / 32; ++c) {
                        e[b][c] = id[b][c] >= 0 && id[b][c] <= w.last
                            ? static_cast<u32>(row_in(w.pages, id[b][c], p.page)) | kValid
                            : kSlotMask;
                    }
                }
#pragma unroll
                for (int b = 0; b < kRound; ++b) {
                    if (j0 + b >= w.nt) break;
                    if (J >= kLists) bar_wait(&list_free[J % kLists], ((J / kLists) - 1) & 1);
                    u32* l = lists + (J % kLists) * kBK;
#pragma unroll
                    for (int c = 0; c < kBK / 32; ++c) l[c * 32 + static_cast<int>(lane)] = e[b][c];
                    __syncwarp();
                    if (lane == 0) bar_arrive(&list_full[J % kLists]);
                    ++J;
                }
            }
        }
    } else if (warp == 4 || warp >= 6) {
        const bool lead = warp == 4;
        const int share = lead ? 0 : static_cast<int>(warp) - 5;
        // The latent: 16 groups of four keys by its atoms (8 bf16, 4 e4m3);
        // the rotated dims: 16 groups twice.
        constexpr int kLatentOps = kE4 ? 16 * 4 : 16 * 8;
        constexpr int kOps = kLatentOps + (kRope > 0 ? 16 * 2 : 0);
        int J = 0;
        int n = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x), ++n) {
            const Row w = take_row(n);
            for (int j = 0; j < w.nt; ++j, ++J) {
                const int slot = J % kSlots;
                const u32* l = lists + (J % kLists) * kBK;
                bar_wait(&list_full[J % kLists], (J / kLists) & 1);
                if (lead) {
                    bar_wait(&empty[slot], ((J / kSlots) & 1) ^ 1);
                    if (lane == 0) bar_arrive_expect_tx(&full[slot], L::kSlot);
                }
                issue_sync();
                unsigned char* dst = smem + L::kOffRing + slot * L::kSlot;
                for (int op = static_cast<int>(lane) * kIssuers + share; op < kOps; op += 32 * kIssuers) {
                    const int g = op % 16;
                    const u32* e = l + 4 * g;
                    int rr[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const u32 sl = e[i] & kSlotMask;
                        rr[i] = sl == kSlotMask ? kAbsentRow : static_cast<int>(sl);
                    }
                    if (op < kLatentOps) {
                        // A run of 16 consecutive cells lands as one box.
                        const int a = op / 16;
                        const u32* run = l + 16 * (g / 4);
                        const u32 first = run[0] & kSlotMask;
                        bool whole = p.runs != 0 && first != kSlotMask;
#pragma unroll
                        for (int i = 1; i < 16; ++i) {
                            whole = whole && (run[i] & kSlotMask) == first + static_cast<u32>(i);
                        }
                        unsigned char* at = dst + (kE4 ? e4_at(a) : atom_at(a)) + g * 4 * 128;
                        const int col = a * (kE4 ? 128 : 64);
                        if (whole) {
                            if (g % 4 == 0) tma_load_2d(at, &full[slot], &map_ckv_run, col, static_cast<int>(first));
                        } else {
                            gather4(at, &full[slot], &map_ckv, col, rr[0], rr[1], rr[2], rr[3]);
                        }
                    } else {
                        const int copy = (op - kLatentOps) / 16;
                        if constexpr (kE4) {
                            // A row's bytes from 448 + 32 copy: its rotated
                            // codes 32 copy .. at byte 64 (past the row,
                            // zeros).
                            unsigned char* at = dst + kE4Latent + copy * kAtom + g * 4 * 128;
                            gather4(at, &full[slot], &map_ckv, 448 + 32 * copy, rr[0], rr[1], rr[2], rr[3]);
                        } else {
                            unsigned char* at = dst + kTile + copy * kAtom + g * 4 * 128;
                            gather4(at, &full[slot], &map_kpe, 0, rr[0], rr[1], rr[2], rr[3]);
                        }
                    }
                }
                issue_sync();
                if (lead && lane == 0) bar_arrive(&list_free[J % kLists]);
            }
        }
    } else if (warp == 5) {
        constexpr u32 kKind = kE4 ? kE4M3 : kF16Bf16;
        const u32 idesc_s = idesc_dense(kKind, kKind, 64, 128);
        const u32 idesc_o = idesc_dense(kKind, kKind, 64, 256) | (1u << 16);
        auto pv = [&](int i, int first) {
            const u32 tile = sbase + L::kOffRing + (i % kSlots) * L::kSlot;
            const u32 pa = sbase + L::kOffP + (i & 1) * kPBytes;
            if constexpr (!kE4) {
                tile_pv(lane, tmem, tile, pa, idesc_o, i > first, &p_full[i & 1], (i >> 1) & 1, &empty[i % kSlots],
                        &pv_done[i & 1]);
                return;
            }
            // O += P V on e4m3: per 256 dims (atoms 2v and 2v + 1, 16 KB
            // apart), 32 keys a step.
            bar_wait(&p_full[i & 1], (i >> 1) & 1);
            tc_fence_after();
            if (lane == 0) {
#pragma unroll
                for (int v = 0; v < 2; ++v) {
#pragma unroll
                    for (int s2 = 0; s2 < kBK / 32; ++s2) {
                        mma_ws_f8(tmem + kColO + v * 128, sdesc_sw128(pa) + 2 * s2,
                                  sdesc_sw128_mn(tile + v * kAtom + s2 * 4096, 2 * kAtom), idesc_o,
                                  i > first || s2 > 0);
                    }
                }
                tc_commit(&empty[i % kSlots]);
                tc_commit(&pv_done[i & 1]);
            }
            __syncwarp();
        };
        int J = 0, rr = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x), ++rr) {
            const Row w = take_row(rr);
            bar_wait(q_ready, rr & 1);
            if (rr > 0) bar_wait(o_free, (rr - 1) & 1);
            tc_fence_after();
            if (w.nt == 0) continue;
            const int first = J;
            for (int j = 0; j < w.nt; ++j, ++J) {
                const int slot = J % kSlots;
                if (J >= 1) bar_wait(s_free, (J - 1) & 1);
                bar_wait(&full[slot], (J / kSlots) & 1);
                tc_fence_after();
                if (lane == 0 && kE4) {
                    const u32 tile = sbase + L::kOffRing + slot * L::kSlot;
                    // The latent: atom pairs (0, 2) and (1, 3), 32 codes a
                    // step; the rotated codes at byte 64 of their rows.
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        mma_ws_ts_f8(tmem + kColS8, tmem + kColQ + s * 8,
                                     sdesc_sw128(tile + (s / 4) * 2 * kAtom) + 2 * (s % 4), idesc_s, s > 0);
                    }
                    if constexpr (kRope > 0) {
                        mma_ws_ts_f8(tmem + kColS8, tmem + kColQr8, sdesc_sw128(tile + kE4Latent) + 4, idesc_s, true);
                    }
                    tc_commit(s_full);
                } else if (lane == 0) {
                    const u32 tile = sbase + L::kOffRing + slot * L::kSlot;
#pragma unroll
                    for (int s = 0; s < 16; ++s) {
                        mma_ws_ts(tmem + kColS, tmem + kColQ + s * 8,
                                  sdesc_sw128(tile + (s / 4) * 2 * kAtom) + 2 * (s % 4), idesc_s, s > 0);
                    }
                    if constexpr (kRope > 0) {
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            mma_ws_ts(tmem + kColS, tmem + kColQr + s * 8, sdesc_sw128(tile + kTile) + 2 * s,
                                      idesc_s, true);
                        }
                    }
                    tc_commit(s_full);
                }
                __syncwarp();
                if (j > 0) pv(J - 1, first);
            }
            pv(J - 1, first);
        }
    } else if (warp < 4) {
        const int t = static_cast<int>(threadIdx.x);
        const int head = t % 64;
        const int half = t / 64;
        const u32 lane_base = (warp * 32) << 16;
        auto load_q = [&](int qrow) {
            const uint4* src = reinterpret_cast<const uint4*>(
                p.q + static_cast<long long>(qrow) * p.q_stride + head * kD + half * 256);
            if constexpr (kE4) {
                // e4m3 codes a word of four: this half's 256 latent dims in
                // two loads of 32 columns, its 32 rotated dims in eight.
                // Every load in flight before any converts.
                const uint4* pe = reinterpret_cast<const uint4*>(
                    p.q_pe + static_cast<long long>(qrow) * p.qpe_stride + head * kRope + half * 32);
                constexpr int kC = kRope > 0 ? 3 : 2;
                uint4 x[kRope > 0 ? 36 : 32];
#pragma unroll
                for (int g = 0; g < 32; ++g) x[g] = src[g];
                if constexpr (kRope > 0) {
#pragma unroll
                    for (int g = 0; g < 4; ++g) x[32 + g] = pe[g];
                }
#pragma unroll
                for (int c = 0; c < kC; ++c) {
                    u32 v[32];
#pragma unroll
                    for (int g = 0; g < 16; ++g) {
                        const bool in = c < 2 || g < 4;
                        v[2 * g] = in ? bf16x4_e4m3(x[c * 16 + g].x, x[c * 16 + g].y) : 0u;
                        v[2 * g + 1] = in ? bf16x4_e4m3(x[c * 16 + g].z, x[c * 16 + g].w) : 0u;
                    }
                    tmem_st32(tmem + lane_base + (c < 2 ? kColQ + c * 32 : kColQr8), v);
                }
                tmem_wait_st();
                tc_fence_before();
                bar_arrive(q_ready);
                return;
            }
#pragma unroll 1
            for (int c = 0; c < 4; ++c) {
                u32 v[32];
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const uint4 x = src[c * 8 + g];
                    v[4 * g] = x.x;
                    v[4 * g + 1] = x.y;
                    v[4 * g + 2] = x.z;
                    v[4 * g + 3] = x.w;
                }
                tmem_st32(tmem + lane_base + kColQ + c * 32, v);
            }
            if constexpr (kRope > 0) {
                // This half's 32 rotated dims at its own place, zeros at the
                // other half's.
                const uint4* pe = reinterpret_cast<const uint4*>(
                    p.q_pe + static_cast<long long>(qrow) * p.qpe_stride + head * kRope + half * 32);
                u32 v[32];
#pragma unroll
                for (int i = 0; i < 32; ++i) v[i] = 0u;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const uint4 x = pe[g];
                    v[half * 16 + 4 * g] = x.x;
                    v[half * 16 + 4 * g + 1] = x.y;
                    v[half * 16 + 4 * g + 2] = x.z;
                    v[half * 16 + 4 * g + 3] = x.w;
                }
                tmem_st32(tmem + lane_base + kColQr, v);
            }
            tmem_wait_st();
            tc_fence_before();
            bar_arrive(q_ready);
        };
        int J = 0;
        int n = 0;
        // The first unit's Q goes in while the lists warp finds its row.
        if (static_cast<int>(blockIdx.x) < live) {
            load_q(base + static_cast<int>(blockIdx.x) / splits);
        }
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x), ++n) {
            const Row w = take_row(n);
            const int row = w.row;
            if (w.nt == 0) {
                if (r + static_cast<int>(gridDim.x) < live) {
                    load_q(base + (r + static_cast<int>(gridDim.x)) / splits);
                }
                if (half == 0) {
                    p.part_lse[(static_cast<long long>(row) * splits + w.split) * 64 + head] = neg_inf();
                }
                tc_fence_before();
                bar_arrive(o_free);
                continue;
            }

            float m_used = neg_inf();
            float l = 0.f;
            for (int j = 0; j < w.nt; ++j, ++J) {
                const u32 sb = J & 1;
                bar_wait(s_full, J & 1);
                tc_fence_after();
                const int mine0 = half * 32;
                const int theirs0 = (1 - half) * 32;
                u32 vm[32], vt[32];
                tmem_ld32(tmem + lane_base + kColSx + mine0, vm);
                tmem_ld32(tmem + lane_base + kColSx + theirs0, vt);
                tmem_wait_ld();
                tc_fence_before();
                bar_arrive(s_free);
                tile_softmax<kE4>(vm, vt, warp, t, lane_base, tmem, xbuf, row_max, lists + (J % kLists) * kBK,
                             &list_full[J % kLists], (J / kLists) & 1, &list_free[J % kLists], p.scale_log2,
                             J, j > 0, pv_done, sbase + L::kOffP + sb * kPBytes, p_full, m_used, l);
            }
            if (r + static_cast<int>(gridDim.x) < live) load_q(base + (r + static_cast<int>(gridDim.x)) / splits);
            bar_wait(&pv_done[(J - 1) & 1], ((J - 1) >> 1) & 1);
            tc_fence_after();
            pair_sync(warp);
            row_max[t] = l;
            pair_sync(warp);
            const float l_all = l + row_max[t ^ 64];
            const float scale = l_all > 0.f ? 1.f / l_all : 0.f;
            if (splits > 1) {
                const long long unit = static_cast<long long>(row) * splits + w.split;
                if (half == 0) {
                    p.part_lse[unit * 64 + head] = l_all > 0.f ? m_used + __log2f(l_all) : neg_inf();
                }
                // Through the score exchange's buffer, so a store is a
                // head's 128 adjacent bytes from eight adjacent lanes: each
                // warp pair (a head's two halves) stages its 64 runs of 64
                // dims (16-byte words XOR-swizzled by run), then stores them a
                // word a lane.
                uint4* stage = reinterpret_cast<uint4*>(xbuf) + (warp & 1) * 64 * 8;
                const int run = half * 32 + static_cast<int>(lane);
#pragma unroll 1
                for (u32 c2 = 0; c2 < 256; c2 += 64) {
                    u32 vv[2][32];
                    tmem_ld32(tmem + lane_base + kColO + c2, vv[0]);
                    tmem_ld32(tmem + lane_base + kColO + c2 + 32, vv[1]);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const u32 (&v)[32] = vv[i / 4];
                        const int j = (i % 4) * 8;
                        stage[run * 8 + (i ^ (run & 7))] = make_uint4(
                            pack_bf16(__uint_as_float(v[j]) * scale, __uint_as_float(v[j + 1]) * scale),
                            pack_bf16(__uint_as_float(v[j + 2]) * scale, __uint_as_float(v[j + 3]) * scale),
                            pack_bf16(__uint_as_float(v[j + 4]) * scale, __uint_as_float(v[j + 5]) * scale),
                            pack_bf16(__uint_as_float(v[j + 6]) * scale, __uint_as_float(v[j + 7]) * scale));
                    }
                    pair_sync(warp);
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int idx = run + 64 * k;
                        const int en = idx >> 3;
                        const int f = idx & 7;
                        const int hd = static_cast<int>(warp & 1) * 32 + (en & 31);
                        const int d0 = static_cast<int>(c2 / 128) * 256 + (en >> 5) * 128 + static_cast<int>(c2 % 128);
                        *reinterpret_cast<uint4*>(p.part_o + (unit * 64 + hd) * kD + d0 + 8 * f) =
                            stage[en * 8 + (f ^ (en & 7))];
                    }
                    pair_sync(warp);
                }
                tc_fence_before();
                bar_arrive(o_free);
                continue;
            }
            if (p.lse != nullptr && half == 0) {
                p.lse[static_cast<long long>(row) * 64 + head] =
                    l_all > 0.f ? m_used + __log2f(l_all) : neg_inf();
            }
            u16* dst = p.o + static_cast<long long>(row) * p.o_stride + static_cast<long long>(head) * kD;
#pragma unroll 1
            for (u32 c = 0; c < 256; c += 32) {
                u32 v[32];
                tmem_ld32(tmem + lane_base + kColO + c, v);
                tmem_wait_ld();
                const int d0 = static_cast<int>(c / 128) * 256 + half * 128 + static_cast<int>(c % 128);
                u32 out[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    out[i] = pack_bf16(__uint_as_float(v[2 * i]) * scale,
                                       __uint_as_float(v[2 * i + 1]) * scale);
                }
                uint4* o4 = reinterpret_cast<uint4*>(dst + d0);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o4[i] = make_uint4(out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]);
                }
            }
            tc_fence_before();
            bar_arrive(o_free);
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

// The split units of each row folded: a block a (row, head), the units'
// outputs weighted by 2^(lse - max) over their sum, the row's output (bf16)
// and its base-2 log-sum-exp (where asked) landed.
__global__ __launch_bounds__(128) void mla_sparse_fold(
    const u16* __restrict__ part_o, const float* __restrict__ part_lse, u16* __restrict__ o,
    float* __restrict__ lse, int splits, int o_stride, const u32* __restrict__ live) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int rb = static_cast<int>(blockIdx.x);
    if (live != nullptr && rb >= static_cast<int>(live[0])) return;
    const int row = live != nullptr ? rb + static_cast<int>(live[1]) : rb;
    const int head = static_cast<int>(blockIdx.y);
    const float* l = part_lse + static_cast<long long>(row) * splits * 64 + head;
    // The splits' weights a lane each (at most 32 splits), every warp
    // working them alike: the largest log-sum-exp, the sum over it, and
    // lane s's share.
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const float mine = lane < splits ? l[lane * 64] : neg_inf();
    float top = mine;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o));
    const float e = mine > neg_inf() ? ex2(mine - top) : 0.f;
    float sum = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float share = sum > 0.f ? e / sum : 0.f;
    u16* dst = o + static_cast<long long>(row) * o_stride + head * kD;
    const u16* base_o = part_o + (static_cast<long long>(row) * splits * 64 + head) * kD;
    for (int d = static_cast<int>(threadIdx.x) * 4; d < kD; d += 128 * 4) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
        for (int s2 = 0; s2 < splits; ++s2) {
            const float wgt = __shfl_sync(0xffffffffu, share, s2);
            if (wgt == 0.f) continue;
            const uint2 b = *reinterpret_cast<const uint2*>(base_o + static_cast<long long>(s2) * 64 * kD + d);
            const float4 v = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                         __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            acc.x = fmaf(wgt, v.x, acc.x);
            acc.y = fmaf(wgt, v.y, acc.y);
            acc.z = fmaf(wgt, v.z, acc.z);
            acc.w = fmaf(wgt, v.w, acc.w);
        }
        uint2 packed;
        packed.x = pack_bf16(acc.x, acc.y);
        packed.y = pack_bf16(acc.z, acc.w);
        *reinterpret_cast<uint2*>(dst + d) = packed;
    }
    if (lse != nullptr && threadIdx.x == 0) {
        lse[static_cast<long long>(row) * 64 + head] = sum > 0.f ? top + __log2f(sum) : neg_inf();
    }
}

}  // namespace pie::attn::mlas
