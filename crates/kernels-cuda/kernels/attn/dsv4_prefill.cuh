#pragma once

#include "attn/tc5_fmha.cuh"

// DeepSeek-V4's attention at prefill widths on tcgen05, whole: the window
// of keys behind each row, the compressed-pool entries it reads (every
// visible one, or those its index selected), one softmax over both, and the
// sinks over the result. What the trace runs as a windowed log-sum-exp
// attention, a pool attention, their merge and the sinks lands here in one
// pass, the sinked output written once.
//
// The shape is `tc5_fmha_prefill_wide`'s at a 512-wide head: a block takes
// 128 packed rows, S = Q K^T in two TMEM buffers of 128 keys, P over S, and
// half of O (256 columns), its twin beside it in y landing the other half;
// Q whole in shared memory, K and V through a ring of three two-atom slices.
// The 64 query heads of one token fill half a tile, so a block takes two
// consecutive rows of the fire, A and B: rows 0-63 are A's heads, 64-127
// B's. Their keys differ, so the block walks a stream that holds both rows'
// keys once, each key marked with the rows that read it:
//
//   - the window: every position either row's window holds (for two rows
//     of one request, one run of at most 129 positions), padded to four;
//   - the pool: the union of the two rows' entries. Selected entries come
//     in each row's ascending list; the producer merges the two lists a
//     tile at a time, an entry both rows read landing once.
//
// Rows of two different requests read disjoint streams, one after the
// other. Keys land by TMA gather (four rows of one plane a box), so a key
// past a stream's end or a window's padding lands as zeros.
//
// Warps 0-3 load nothing, run the softmax (one packed row a thread) and
// store O; warp 4 merges the key lists and issues the gathers; warp 5 owns
// TMEM and its lane 0 issues every MMA.
namespace pie::attn::dsv4p {

using tc5::u32;
using tc5::u64;
using fmha::u16;
using fmha::ex2;
using fmha::neg_inf;
using fmha::pack_bf16;
using fmha::sdesc_sw128_mn;

struct Params {
    u16* o;
    const int* positions;
    const int* requests;
    const u32* win_indptr;
    const u32* win_indices;
    const u32* ent_indptr;
    const u32* ent_indices;
    const int* selection;
    const void* sinks;
    const u32* live;
    int sinks_f32;
    // Whether `map_ent_run` lands sixteen entries of a page (`ratio` rows
    // apart) in a box: the page holds whole strides of them.
    int ent_runs;
    int win_page;
    int ent_page;
    int top_k;
    int ratio;
    int rows;
    int o_stride;
    float scale_log2;
};

constexpr int kD = 512;
constexpr int kDV = 256;
constexpr int kBKV = 128;
constexpr int kSA = 2;
constexpr int kSlots = 3;
constexpr int kWindow = 128;
constexpr u32 kAtomQ = 128 * 128;
constexpr u32 kAtomKV = kBKV * 128;
constexpr u32 kSlice = kSA * kAtomKV;
constexpr int kKS = kD / 64 / kSA;
constexpr int kVS = kDV / 64 / kSA;
constexpr u32 kOffQ = 0;
constexpr u32 kOffRing = kOffQ + (kD / 64) * kAtomQ;
constexpr u32 kOffList = kOffRing + kSlots * kSlice;
constexpr u32 kOffBar = kOffList + 3 * kBKV * 4;
constexpr u32 kColS = 0;
constexpr u32 kColO = 2 * kBKV;
constexpr u32 kCols = 512;

// A stream entry: the key's row in its plane, its plane, and the rows that
// read it. A padding entry reads as neither row's.
constexpr u32 kSlotMask = (1u << 29) - 1;
constexpr u32 kSeesA = 1u << 29;
constexpr u32 kSeesB = 1u << 30;
constexpr u32 kPool = 1u << 31;
constexpr int kAbsentRow = 0x7fffff00;

__host__ __device__ constexpr u32 smem_bytes() { return kOffBar + 512u + 1024u; }

__device__ __forceinline__ void gather4(void* dst, u64* bar, const tc5::TmaMap* map, int col,
                                        int r0, int r1, int r2, int r3) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(tc5::smem_addr(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tc5::smem_addr(bar))
        : "memory");
}

// gather4 landing at the same offset in both CTAs of the pair, each
// completing its own barrier there.
__device__ __forceinline__ void gather4_pair(void* dst, u64* bar, const tc5::TmaMap* map, int col,
                                             int r0, int r1, int r2, int r3) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
        ".multicast::cluster [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(tc5::smem_addr(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tc5::smem_addr(bar)),
        "h"(static_cast<unsigned short>(3))
        : "memory");
}

__device__ __forceinline__ void tma_load_3d_both(void* dst, u64* bar, const tc5::TmaMap* map,
                                                 int x, int y, int z) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(tc5::smem_addr(dst)),
        "l"(map), "r"(tc5::smem_addr(bar)), "r"(x), "r"(y), "r"(z),
        "h"(static_cast<unsigned short>(3))
        : "memory");
}

// One arrival, once this thread's tcgen05 work is done, on `bar`'s offset
// in both CTAs of the pair.
__device__ __forceinline__ void tc_commit_both(u64* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;" ::"r"(tc5::smem_addr(bar)),
        "h"(static_cast<unsigned short>(3))
        : "memory");
}

// The row a position lands at, through its request's pages.
__device__ __forceinline__ int row_in(const u32* pages, int pos, int page) {
    return static_cast<int>(pages[pos / page]) * page + pos % page;
}

__device__ __forceinline__ int owned(const u32* indptr, int req, int page) {
    return max(static_cast<int>(indptr[req + 1]) - static_cast<int>(indptr[req]), 0) * page;
}

template <bool kSelected>
__global__ __launch_bounds__(288, 1) void dsv4_prefill(
    const __grid_constant__ tc5::TmaMap map_q, const __grid_constant__ tc5::TmaMap map_win,
    const __grid_constant__ tc5::TmaMap map_ent, const __grid_constant__ tc5::TmaMap map_win_run,
    const __grid_constant__ tc5::TmaMap map_ent_run, const Params p) {
    using namespace tc5;
    const int live = p.live != nullptr ? static_cast<int>(p.live[0]) : p.rows;
    const int base = p.live != nullptr ? static_cast<int>(p.live[1]) : 0;
    // A cluster of two CTAs a pair of rows: each lands half of O, and the K
    // both read lands in both, each CTA gathering half of every slice.
    const int pair = static_cast<int>(blockIdx.x) / 2;
    if (2 * pair >= live) return;
    const u32 rank = cluster_rank();
    const int v0 = static_cast<int>(rank) * kDV;
    const int row_a = base + 2 * pair;
    const bool has_b = 2 * pair + 1 < live;
    const int row_b = row_a + 1;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);
    u32* lists = reinterpret_cast<u32*>(smem + kOffList);
    u64* full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* empty = full + kSlots;
    u64* s_full = empty + kSlots;
    u64* q_ready = s_full + 2;
    u64* p_full = q_ready + 1;
    u64* pv_done = p_full + 2;
    u64* list_full = pv_done + 2;
    u64* list_free = list_full + 3;
    u32* tmem_slot = reinterpret_cast<u32*>(list_free + 3);
    // [end_a, end_b, the entries in lists 0, 1 and 2]
    int* shared_ints = reinterpret_cast<int*>(tmem_slot + 1);
    int* counts = shared_ints + 2;

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    // The two rows, their requests and positions, and what each reads.
    const int req_a = p.requests[row_a];
    const int pos_a = p.positions[row_a];
    const int req_b = has_b ? p.requests[row_b] : -1;
    const int pos_b = has_b ? p.positions[row_b] : 0;
    const bool same = has_b && req_b == req_a;
    const int vis_a = min((pos_a + 1) / p.ratio, owned(p.ent_indptr, req_a, p.ent_page) / p.ratio);
    const int vis_b =
        has_b ? min((pos_b + 1) / p.ratio, owned(p.ent_indptr, req_b, p.ent_page) / p.ratio) : 0;
    const int* sel_a = kSelected ? p.selection + static_cast<long long>(row_a) * p.top_k : nullptr;
    const int* sel_b = kSelected ? p.selection + static_cast<long long>(row_b) * p.top_k : nullptr;
    // Each row's first page in either plane's table.
    const int wb_a = static_cast<int>(p.win_indptr[req_a]);
    const int wb_b = has_b ? static_cast<int>(p.win_indptr[req_b]) : 0;
    const int eb_a = static_cast<int>(p.ent_indptr[req_a]);
    const int eb_b = has_b ? static_cast<int>(p.ent_indptr[req_b]) : 0;

    if (threadIdx.x == 0) {
        shared_ints[0] = kSelected ? p.top_k : 0;
        shared_ints[1] = kSelected && has_b ? p.top_k : 0;
        for (int s = 0; s < kSlots; ++s) {
            bar_init(&full[s], 1);
            // Both CTAs' MMAs free a slot: K lands in both.
            bar_init(&empty[s], 2);
        }
        bar_init(&s_full[0], 1);
        bar_init(&s_full[1], 1);
        bar_init(q_ready, 1);
        bar_init(&p_full[0], 128);
        bar_init(&p_full[1], 128);
        bar_init(&pv_done[0], 1);
        bar_init(&pv_done[1], 1);
        for (int s = 0; s < 3; ++s) {
            bar_init(&list_full[s], 1);
            bar_init(&list_free[s], 129);
        }
        bar_init_fence();
    }
    if (warp == 5) {
        tmem_alloc(tmem_slot, kCols);
        tmem_relinquish();
    }
    tc_fence_before();
    // The peer's barriers are initialised before anything lands in them.
    cluster_sync();
    tc_fence_after();
    if (kSelected) {
        // Each list ends at its first negative id.
        for (int n = threadIdx.x; n < p.top_k; n += blockDim.x) {
            if (sel_a[n] < 0) atomicMin(&shared_ints[0], n);
            if (has_b && sel_b[n] < 0) atomicMin(&shared_ints[1], n);
        }
        __syncthreads();
    }
    const u32 tmem = *tmem_slot;
    const int end_a = kSelected ? shared_ints[0] : vis_a;
    const int end_b = kSelected ? shared_ints[1] : vis_b;
    // The stream: the window, padded to whole gathers, then the pool.
    const int win_lo = same ? max(0, min(pos_a, pos_b) - (kWindow - 1)) : 0;
    const int win_a = same ? max(pos_a, pos_b) - win_lo + 1 : min(pos_a, kWindow - 1) + 1;
    const int win_b = same || !has_b ? 0 : min(pos_b, kWindow - 1) + 1;
    const int nw = win_a + win_b;
    const int nwp = (nw + 3) & ~3;
    // The stream's length, but for two selections of one request, whose
    // union the producer finds as it merges them.
    const bool merged = kSelected && same;
    const int total = nwp + (same && !kSelected ? max(end_a, end_b) : end_a + end_b);

    if (warp == 4 || warp >= 6) {
        // The producer group: warp 6 merges the lists, tiles ahead; warps 4,
        // 7 and 8 issue the gathers, warp 4 leading (it waits for the ring and
        // arms its barriers), the three splitting each slice's boxes (one
        // warp's TMA issue serialises across its lanes).
        const bool builder = warp == 6;
        const bool lead = warp == 4;
        const int share = lead ? 0 : static_cast<int>(warp) - 6;
        auto group_sync = [] { asm volatile("bar.sync 1, 96;" ::: "memory"); };
        if (lead && lane == 0) {
            bar_arrive_expect_tx(q_ready, (kD / 64) * kAtomQ);
#pragma unroll
            for (int a = static_cast<int>(rank) * (kD / 128); a < (static_cast<int>(rank) + 1) * (kD / 128); ++a) {
                tma_load_3d_both(smem + kOffQ + a * kAtomQ, q_ready, &map_q, a * 64, 0, row_a);
            }
            tma_prefetch(&map_win);
            tma_prefetch(&map_ent);
        }
        // The merge's cursors into the two selections.
        int ia = 0, ib = 0;
        // Fill list `l` with stream entries [k0, k0 + 128).
        auto build = [&](u32* l, int k0) -> int {
            // The window's and the plainly indexed pool's entries.
#pragma unroll 4
            for (int c = static_cast<int>(lane); c < kBKV; c += 32) {
                const int k = k0 + c;
                u32 e = 0;
                if (k < nwp) {
                    if (k < nw) {
                        int req, pos;
                        u32 sees;
                        if (same) {
                            pos = win_lo + k;
                            req = req_a;
                            sees = (pos <= pos_a && pos > pos_a - kWindow ? kSeesA : 0u) |
                                   (pos <= pos_b && pos > pos_b - kWindow ? kSeesB : 0u);
                        } else if (k < win_a) {
                            pos = pos_a - win_a + 1 + k;
                            req = req_a;
                            sees = kSeesA;
                        } else {
                            pos = pos_b - win_b + 1 + (k - win_a);
                            req = req_b;
                            sees = kSeesB;
                        }
                        e = static_cast<u32>(row_in(p.win_indices + (req == req_a ? wb_a : wb_b), pos,
                                                       p.win_page)) | sees;
                    } else {
                        e = kSlotMask;  // padding: no plane row, read by neither
                    }
                } else if (k < total && !merged) {
                    const int u = k - nwp;
                    int id, req;
                    u32 sees;
                    if (same) {
                        id = u;
                        req = req_a;
                        sees = (id < vis_a ? kSeesA : 0u) | (id < vis_b ? kSeesB : 0u);
                    } else if (u < end_a) {
                        id = kSelected ? sel_a[u] : u;
                        req = req_a;
                        sees = id < vis_a ? kSeesA : 0u;
                    } else {
                        id = kSelected ? sel_b[u - end_a] : u - end_a;
                        req = req_b;
                        sees = id < vis_b ? kSeesB : 0u;
                    }
                    e = static_cast<u32>(row_in(p.ent_indices + (req == req_a ? eb_a : eb_b),
                                                   (id + 1) * p.ratio - 1, p.ent_page)) |
                        sees | kPool;
                } else if (k >= total) {
                    e = kSlotMask | kPool;
                }
                if (!merged || k < nwp) l[c] = e;
            }
            int count = max(0, min(total, k0 + kBKV) - k0);
            if (merged) {
                // The union's next entries, ranked: an entry's place is how
                // many entries of both lists precede it, less those both hold.
                const int first = max(k0, nwp);
                const int want = k0 + kBKV - first;
                const int na = min(end_a - ia, kBKV);
                const int nb = min(end_b - ib, kBKV);
                int made = 0;
                if (want > 0 && na + nb > 0) {
                    int va[4], vb[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int t = 4 * static_cast<int>(lane) + q;
                        va[q] = t < na ? sel_a[ia + t] : 0x7fffffff;
                        vb[q] = t < nb ? sel_b[ib + t] : 0x7fffffff;
                    }
                    // Each candidate's count of the other window's ids below it,
                    // and whether that window holds it: the lanes whose first id
                    // lies at or below it (a search over the 32 lane heads), then the
                    // last such lane's four ids. The eight searches step together.
                    int la[4], lb[4];
                    bool da[4], db[4];
                    {
                        int pa[4] = {0, 0, 0, 0}, pb[4] = {0, 0, 0, 0};
#pragma unroll
                        for (int step = 32; step > 0; step >>= 1) {
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int ca = pa[q] + step, cb = pb[q] + step;
                                const int hb = __shfl_sync(0xffffffffu, vb[0], min(ca, 32) - 1);
                                const int ha = __shfl_sync(0xffffffffu, va[0], min(cb, 32) - 1);
                                if (ca <= 32 && hb <= va[q]) pa[q] = ca;
                                if (cb <= 32 && ha <= vb[q]) pb[q] = cb;
                            }
                        }
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int sa = max(pa[q], 1) - 1, sb = max(pb[q], 1) - 1;
                            int cnt_a = 0, cnt_b = 0;
                            bool eq_a = false, eq_b = false;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int xb = __shfl_sync(0xffffffffu, vb[i], sa);
                                const int xa = __shfl_sync(0xffffffffu, va[i], sb);
                                cnt_a += xb < va[q];
                                cnt_b += xa < vb[q];
                                eq_a = eq_a || xb == va[q];
                                eq_b = eq_b || xa == vb[q];
                            }
                            la[q] = pa[q] > 0 ? 4 * sa + cnt_a : 0;
                            lb[q] = pb[q] > 0 ? 4 * sb + cnt_b : 0;
                            da[q] = pa[q] > 0 && eq_a;
                            db[q] = pb[q] > 0 && eq_b;
                        }
                    }
                    int dup_a = 0, dup_b = 0;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int t = 4 * static_cast<int>(lane) + q;
                        da[q] = da[q] && t < na;
                        db[q] = db[q] && t < nb;
                        dup_a += da[q];
                        dup_b += db[q];
                    }
                    // Exclusive prefix of each list's shared entries.
                    int pre_a = dup_a, pre_b = dup_b;
#pragma unroll
                    for (int sh = 1; sh < 32; sh <<= 1) {
                        const int xa = __shfl_up_sync(0xffffffffu, pre_a, sh);
                        const int xb = __shfl_up_sync(0xffffffffu, pre_b, sh);
                        if (static_cast<int>(lane) >= sh) {
                            pre_a += xa;
                            pre_b += xb;
                        }
                    }
                    pre_a -= dup_a;
                    pre_b -= dup_b;
                    int took_a = 0, took_b = 0, put = 0;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int t = 4 * static_cast<int>(lane) + q;
                        const int ra = t + la[q] - pre_a;
                        if (t < na && ra < want) {
                            const u32 sees = (va[q] < vis_a ? kSeesA : 0u) |
                                             (da[q] && va[q] < vis_b ? kSeesB : 0u);
                            l[first - k0 + ra] =
                                static_cast<u32>(row_in(p.ent_indices + eb_a,
                                                           (va[q] + 1) * p.ratio - 1,
                                                           p.ent_page)) |
                                sees | kPool;
                            ++took_a;
                            ++put;
                        }
                        pre_a += da[q];
                        const int rb = t + lb[q] - pre_b;
                        if (t < nb && rb < want) {
                            if (!db[q]) {
                                l[first - k0 + rb] =
                                    static_cast<u32>(row_in(p.ent_indices + eb_a,
                                                               (vb[q] + 1) * p.ratio - 1,
                                                               p.ent_page)) |
                                    (vb[q] < vis_b ? kSeesB : 0u) | kPool;
                                ++put;
                            }
                            ++took_b;
                        }
                        pre_b += db[q];
                    }
#pragma unroll
                    for (int sh = 16; sh > 0; sh >>= 1) {
                        took_a += __shfl_xor_sync(0xffffffffu, took_a, sh);
                        took_b += __shfl_xor_sync(0xffffffffu, took_b, sh);
                        put += __shfl_xor_sync(0xffffffffu, put, sh);
                    }
                    ia += took_a;
                    ib += took_b;
                    made = put;
                }
                // The tile's slots past the union read as nothing.
                for (int c = first - k0 + made + static_cast<int>(lane); c < kBKV; c += 32) {
                    l[c] = kSlotMask | kPool;
                }
                count = max(0, min(nwp, k0 + kBKV) - k0) + made;
            }
            __syncwarp();
            return count;
        };
        int seq = 0;
        // A slice: atoms [col / 64, col / 64 + kSA) of the tile listed in
        // `l`, a gather of four keys a lane.
        // A slice: atoms [col / 64, col / 64 + kSA) of the tile listed in
        // `l`. A K slice lands in both CTAs, each gathering half its keys; a
        // V slice (this CTA's half of the head) only here, all its keys.
        auto slice = [&](const u32* l, int col, bool both) {
            const int slot = seq % kSlots;
            if (lead) {
                bar_wait(&empty[slot], ((seq / kSlots) & 1) ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&full[slot], kSlice);
            }
            group_sync();
            // One box of four keys (or the first of a run) an op, ops dealt
            // round the three warps; a K slice's groups split between the
            // pair.
            const int groups = both ? 16 : 32;
            const int op = static_cast<int>(lane) * 3 + share;
            if (op < groups * kSA) {
                const int g = (both ? static_cast<int>(rank) * 16 : 0) + op % groups;
                const int a = op / groups;
                unsigned char* dst = smem + kOffRing + slot * kSlice;
                const u32* e = l + 4 * g;
                const bool pool = (e[0] & kPool) != 0;
                // Sixteen keys a run of rows (the window's positions, or a
                // page's entries one stride apart) land as one box, from the
                // run's first group; else four a gather.
                const u32* run = l + 16 * (g / 4);
                const int stride = pool ? p.ratio : 1;
                const u32 first = run[0] & kSlotMask;
                bool whole = (!pool || p.ent_runs) && first != kSlotMask;
#pragma unroll
                for (int i = 1; i < 16; ++i) {
                    whole = whole && (run[i] & (kSlotMask | kPool)) ==
                                         ((first + static_cast<u32>(i * stride)) | (run[0] & kPool));
                }
                unsigned char* at = dst + a * kAtomKV + g * 4 * 128;
                if (whole) {
                    if (g % 4 == 0) {
                        if (pool) {
                            if (both) {
                                tma_load_3d_both(at, &full[slot], &map_ent_run, col + a * 64,
                                                 p.ratio - 1, static_cast<int>(first) / p.ratio);
                            } else {
                                tma_load_3d(at, &full[slot], &map_ent_run, col + a * 64,
                                            p.ratio - 1, static_cast<int>(first) / p.ratio);
                            }
                        } else if (both) {
                            tma_load_2d_multicast(at, &full[slot], &map_win_run, col + a * 64,
                                                  static_cast<int>(first), 3);
                        } else {
                            tma_load_2d(at, &full[slot], &map_win_run, col + a * 64,
                                        static_cast<int>(first));
                        }
                    }
                } else {
                    int r[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const u32 sl = e[i] & kSlotMask;
                        r[i] = sl == kSlotMask ? kAbsentRow : static_cast<int>(sl);
                    }
                    const tc5::TmaMap* map = pool ? &map_ent : &map_win;
                    if (both) {
                        gather4_pair(at, &full[slot], map, col + a * 64, r[0], r[1], r[2], r[3]);
                    } else {
                        gather4(at, &full[slot], map, col + a * 64, r[0], r[1], r[2], r[3]);
                    }
                }
            }
            ++seq;
        };
        // Tile j's list sits in slot j % 3, listed up to three tiles ahead of
        // the gathers. A slot is rebuilt once the softmax has read its marks
        // and the issuers have sent its V slices (129 arrivals).
        if (builder) {
            for (int t = 0;; ++t) {
                if (t >= 3) bar_wait(&list_free[t % 3], ((t - 3) / 3) & 1);
                const int built = build(lists + (t % 3) * kBKV, t * kBKV);
                if (lane == 0) {
                    counts[t % 3] = built;
                    bar_arrive(&list_full[t % 3]);
                }
                if (built == 0) break;
            }
        } else {
            int nt = 0;
            for (int j = 0;; ++j) {
                bar_wait(&list_full[j % 3], (j / 3) & 1);
                if (counts[j % 3] == 0) break;
                nt = j + 1;
                const u32* l = lists + (j % 3) * kBKV;
#pragma unroll 1
                for (int a = 0; a < kKS; ++a) slice(l, a * 64 * kSA, true);
                if (j > 0) {
                    const u32* prev = lists + ((j - 1) % 3) * kBKV;
#pragma unroll 1
                    for (int c = 0; c < kVS; ++c) slice(prev, v0 + c * 64 * kSA, false);
                    group_sync();
                    if (lead && lane == 0) bar_arrive(&list_free[(j - 1) % 3]);
                }
            }
            const u32* last = lists + ((nt - 1) % 3) * kBKV;
#pragma unroll 1
            for (int c = 0; c < kVS; ++c) slice(last, v0 + c * 64 * kSA, false);
        }
    } else if (warp == 5) {
        if (lane == 0) {
            const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 128, kBKV);
            const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 128, 64 * kSA) | (1u << 16);
            bar_wait(q_ready, 0);
            tc_fence_after();
            int slot = 0;
            u32 phase = 0;
            auto next = [&] {
                if (++slot == kSlots) {
                    slot = 0;
                    phase ^= 1;
                }
            };
            auto pv = [&](int i) {
                bar_wait(&p_full[i & 1], (i >> 1) & 1);
                tc_fence_after();
                const u32 pa = tmem + kColS + (i & 1) * kBKV;
#pragma unroll
                for (int c = 0; c < kVS; ++c) {
                    bar_wait(&full[slot], phase);
                    tc_fence_after();
                    const u32 vb = sbase + kOffRing + slot * kSlice;
#pragma unroll
                    for (u32 s = 0; s < kBKV / 16; ++s) {
                        mma_f16_ts(tmem + kColO + c * 64 * kSA, pa + s * 8,
                                   sdesc_sw128_mn(vb + s * 2048, kAtomKV), idesc_o,
                                   i > 0 || s > 0);
                    }
                    tc_commit_both(&empty[slot]);
                    next();
                }
                tc_commit(&pv_done[i & 1]);
            };
            int nt = 0;
            for (int j = 0;; ++j) {
                const u32 sb = j & 1;
                bar_wait(&list_full[j % 3], (j / 3) & 1);
                if (counts[j % 3] == 0) break;
                nt = j + 1;
                if (j >= 2) bar_wait(&pv_done[sb], ((j - 2) >> 1) & 1);
#pragma unroll
                for (int a = 0; a < kKS; ++a) {
                    bar_wait(&full[slot], phase);
                    tc_fence_after();
#pragma unroll
                    for (int a2 = 0; a2 < kSA; ++a2) {
                        const int ga = a * kSA + a2;
                        const u32 kb = sbase + kOffRing + slot * kSlice + a2 * kAtomKV;
                        const u32 qa = sbase + kOffQ + ga * kAtomQ;
#pragma unroll
                        for (u32 s = 0; s < 4; ++s) {
                            mma_f16(tmem + kColS + sb * kBKV, sdesc_sw128(qa) + 2 * s,
                                    sdesc_sw128(kb) + 2 * s, idesc_s, ga > 0 || s > 0);
                        }
                    }
                    tc_commit_both(&empty[slot]);
                    next();
                }
                tc_commit(&s_full[sb]);
                if (j > 0) pv(j - 1);
            }
            pv(nt - 1);
        }
    } else {
        // The softmax warpgroup: packed row `row` is this thread's, row A's
        // head `row` or row B's head `row - 64`.
        const u32 row = warp * 32 + lane;
        const bool is_b = row >= 64;
        const bool live_row = !is_b || has_b;
        const u32 mine = is_b ? kSeesB : kSeesA;
        const u32 lane_base = (warp * 32) << 16;
        float m_used = neg_inf();
        float l = 0.f;
        int nt = 0;
        for (int j = 0;; ++j) {
            const u32 st = j & 1;
            const u32* marks = lists + (j % 3) * kBKV;
            bar_wait(&list_full[j % 3], (j / 3) & 1);
            if (counts[j % 3] == 0) break;
            nt = j + 1;
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
            float m4[4] = {neg_inf(), neg_inf(), neg_inf(), neg_inf()};
#pragma unroll
            for (int c = 0; c < kBKV; c += 4) {
                const uint4 e = *reinterpret_cast<const uint4*>(marks + c);
                s[c] = (e.x & mine) ? s[c] : neg_inf();
                s[c + 1] = (e.y & mine) ? s[c + 1] : neg_inf();
                s[c + 2] = (e.z & mine) ? s[c + 2] : neg_inf();
                s[c + 3] = (e.w & mine) ? s[c + 3] : neg_inf();
#pragma unroll
                for (int i = 0; i < 4; ++i) m4[i] = fmaxf(m4[i], s[c + i]);
            }
            bar_arrive(&list_free[j % 3]);
            const float raw_mx = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
            const float mx = raw_mx == neg_inf() ? neg_inf() : raw_mx * p.scale_log2;
            const bool grow = mx > m_used + 8.f;
            const float m_new = grow ? mx : m_used;
            const float alpha = grow ? ex2(m_used - m_new) : 1.f;
            const float bse = m_new == neg_inf() ? 0.f : m_new;
            float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (u32 c = 0; c < kBKV; c += 64) {
                u32 pk[32];
#pragma unroll
                for (int i = 0; i < 64; i += 2) {
                    const float e0 = ex2(fmaf(s[c + i], p.scale_log2, -bse));
                    const float e1 = ex2(fmaf(s[c + i + 1], p.scale_log2, -bse));
                    s4[(i >> 1) & 3] += e0 + e1;
                    pk[i / 2] = pack_bf16(e0, e1);
                }
                tmem_st32(tmem + lane_base + kColS + st * kBKV + c / 2, pk);
            }
            const float sum = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            l = l * alpha + sum;
            m_used = m_new;
            if (j > 0 && __any_sync(0xffffffffu, grow)) {
                bar_wait(&pv_done[(j - 1) & 1], ((j - 1) >> 1) & 1);
                tc_fence_after();
#pragma unroll 1
                for (u32 c = 0; c < kDV; c += 32) {
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
            tc_fence_before();
            bar_arrive(&p_full[j & 1]);
        }
        bar_wait(&pv_done[(nt - 1) & 1], ((nt - 1) >> 1) & 1);
        tc_fence_after();
        // The sink scales a head's output by sigmoid(lse - sink), lse in
        // natural log as the merge left it.
        const int head = static_cast<int>(row % 64);
        float scale = 0.f;
        if (l > 0.f) {
            constexpr float kLn2 = 0.69314718055994530942f;
            const float lse = (m_used + __log2f(l)) * kLn2;
            const float sink =
                p.sinks_f32 ? static_cast<const float*>(p.sinks)[head]
                            : __uint_as_float(static_cast<u32>(static_cast<const u16*>(p.sinks)[head])
                                              << 16);
            scale = 1.f / (l * (1.f + __expf(sink - lse)));
        }
        u16* dst = p.o + static_cast<long long>(is_b ? row_b : row_a) * p.o_stride +
                   static_cast<long long>(head) * kD + v0;
#pragma unroll 1
        for (u32 c = 0; c < kDV; c += 32) {
            u32 v[32];
            tmem_ld32(tmem + lane_base + kColO + c, v);
            tmem_wait_ld();
            if (live_row) {
                u32 out[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    out[i] = pack_bf16(__uint_as_float(v[2 * i]) * scale,
                                       __uint_as_float(v[2 * i + 1]) * scale);
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
    // Nothing the peer issued still lands here once both are past this.
    cluster_sync();
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc(tmem, kCols);
    }
}

}  // namespace pie::attn::dsv4p
