#pragma once

#include "attn/dsv4_kv.cuh"
#include "attn/tc5_fmha.cuh"

// DeepSeek-V4's attention at prefill widths on tcgen05, a block a row: the
// row's window of keys and its compressed-pool entries (every visible one,
// or those its index selected) in one softmax, the sinks over the result,
// the sinked output written once. The shape is FlashMLA's sparse prefill at
// 64 heads (`sm100/prefill/sparse/fwd/head64`), in this codebase's
// primitives:
//
//   - Q (64 heads by 512) sits in TMEM, heads a lane: lanes 0-63 hold the
//     first 256 of each head, lanes 64-127 the last, 128 columns in all.
//   - A tile is 64 keys whole (64 KB) in shared memory, three of them a
//     ring, laid out so a K step reads 64 keys' first-half dims as rows
//     0-63 and their second-half dims as rows 64-127 of one 128-row operand:
//     atom g (dims 64 g ..) and atom g + 4 sit side by side.
//   - S = Q K^T is the weight-stationary MMA at M = 64, N = 128, A from
//     TMEM: lanes 0-63 land Q's first halves against the keys' first halves,
//     lanes 64-127 the second against the second. The two partial scores of
//     a head meet in shared memory, each thread of the pair keeping 32 keys.
//   - O += P V is the same MMA at N = 256 from shared memory, V the tile K
//     already brought (no second gather): lanes 0-63 land a head's dims
//     [0, 128) and [256, 384), lanes 64-127 [128, 256) and [384, 512).
//
// Warps 0-3 run the softmax (thread t: head t % 64, keys or dims half
// t / 64) and store O; warp 4 lists a tile's keys, waits for the ring and
// arms it; warps 4, 6 and 7 issue its gathers; warp 5 owns TMEM and issues
// every MMA. A row's stream is its window (padded to whole gathers) then its
// entries; a key past it lands as zeros and is masked.
namespace pie::attn::dsv4s {

using tc5::u32;
using tc5::u64;
using fmha::u16;
using fmha::ex2;
using fmha::neg_inf;
using fmha::pack_bf16;
using fmha::sdesc_sw128_mn;

struct Params {
    u16* o;
    const u16* q;
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
    int ent_runs;
    // DeepSeek's fp8 rows (`attn/dsv4_kv.cuh`): the planes' bytes.
    const unsigned char* win_fp8;
    const unsigned char* ent_fp8;
    int win_page;
    int ent_page;
    int top_k;
    int ratio;
    int rows;
    int q_stride;
    int o_stride;
    float scale_log2;
};

constexpr int kD = 512;
constexpr int kBK = 64;
constexpr int kSlots = 3;
constexpr int kWindow = 128;
constexpr u32 kAtom = kBK * 128;          // 64 keys by 64 bf16
constexpr u32 kTile = (kD / 64) * kAtom;  // 64 KB
constexpr u32 kOffRing = 0;
constexpr u32 kOffP = kOffRing + kSlots * kTile;
constexpr u32 kPBytes = 64 * 128;  // 64 heads by 64 keys
constexpr u32 kOffX = kOffP + 2 * kPBytes;
constexpr u32 kXBytes = 32 * 128 * 4;
constexpr u32 kOffList = kOffX + kXBytes;
constexpr int kLists = 6;  // tile lists, built ahead of the ring
constexpr u32 kOffBar = kOffList + kLists * kBK * 4;
constexpr u32 kColQ = 0;
constexpr u32 kColS = 128;
constexpr u32 kColO = 256;
constexpr u32 kCols = 512;

constexpr u32 kSlotMask = (1u << 30) - 1;
constexpr u32 kPool = 1u << 30;
constexpr u32 kValid = 1u << 31;
constexpr int kAbsentRow = 0x7fffff00;

__host__ __device__ constexpr u32 smem_bytes() { return kOffBar + 512u + 1024u; }

// Where atom `a` (dims 64 a ..) of a tile sits: atoms a and a + 4 side by
// side, 64 rows each.
__device__ __forceinline__ u32 atom_at(int a) { return (a & 3) * 2 * kAtom + (a >> 2) * kAtom; }

__device__ __forceinline__ void mma_ws_ts(u32 d, u32 a, u64 b, u32 idesc, bool acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}" ::"r"(d),
        "r"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc))
        : "memory");
}

__device__ __forceinline__ void mma_ws(u32 d, u64 a, u64 b, u32 idesc, bool acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(static_cast<u32>(acc)));
}

__device__ __forceinline__ void gather4(void* dst, u64* bar, const tc5::TmaMap* map, int col,
                                        int r0, int r1, int r2, int r3) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(tc5::smem_addr(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tc5::smem_addr(bar))
        : "memory");
}

__device__ __forceinline__ int row_in(const u32* pages, int pos, int page) {
    return static_cast<int>(pages[pos / page]) * page + pos % page;
}

__device__ __forceinline__ int owned(const u32* indptr, int req, int page) {
    return max(static_cast<int>(indptr[req + 1]) - static_cast<int>(indptr[req]), 0) * page;
}

// The two warps whose lanes hold a head's two halves (warps w and w + 2) meet.
__device__ __forceinline__ void pair_sync(u32 warp) {
    if (warp & 1) {
        asm volatile("bar.sync 4, 64;" ::: "memory");
    } else {
        asm volatile("bar.sync 3, 64;" ::: "memory");
    }
}
constexpr int kIssuers = 6;  // warp 4 and warps 6-10; warp 11 lists
__device__ __forceinline__ void issue_sync() { asm volatile("bar.sync 2, %0;" ::"n"(kIssuers * 32) : "memory"); }

// kFp8: both planes hold DeepSeek's fp8 rows; the issuing warps read them
// and land them in bf16 themselves (TMA cannot widen e4m3).
template <bool kSelected, bool kFp8 = false>
__global__ __launch_bounds__(384, 1) void dsv4_sparse(
    const __grid_constant__ tc5::TmaMap map_win, const __grid_constant__ tc5::TmaMap map_ent,
    const __grid_constant__ tc5::TmaMap map_win_run, const __grid_constant__ tc5::TmaMap map_ent_run,
    const Params p) {
    using namespace tc5;
    const int live = p.live != nullptr ? static_cast<int>(p.live[0]) : p.rows;
    const int base = p.live != nullptr ? static_cast<int>(p.live[1]) : 0;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);
    float* xbuf = reinterpret_cast<float*>(smem + kOffX);
    u32* lists = reinterpret_cast<u32*>(smem + kOffList);
    u64* full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* empty = full + kSlots;
    u64* s_full = empty + kSlots;
    u64* s_free = s_full + 2;
    u64* p_full = s_free + 2;
    u64* pv_done = p_full + 2;
    u64* q_ready = pv_done + 2;
    u64* o_free = q_ready + 1;
    u64* list_full = o_free + 1;
    u64* list_free = list_full + kLists;
    u32* tmem_slot = reinterpret_cast<u32*>(list_free + kLists);
    int* shared_ints = reinterpret_cast<int*>(tmem_slot + 1);  // [end of the selection]
    float* row_max = reinterpret_cast<float*>(shared_ints + 4);  // [128]

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < kSlots; ++s) {
            // bf16: the lead's armed arrival (TMA counts the bytes); fp8:
            // every issuing thread once its stores are visible.
            bar_init(&full[s], kFp8 ? kIssuers * 32 : 1);
            bar_init(&empty[s], 1);
        }
        for (int s = 0; s < 2; ++s) {
            bar_init(&s_full[s], 1);
            bar_init(&s_free[s], 128);
            bar_init(&p_full[s], 128);
            bar_init(&pv_done[s], 1);
        }
        bar_init(q_ready, 128);
        bar_init(o_free, 128);
        for (int s = 0; s < kLists; ++s) {
            bar_init(&list_full[s], 1);
            // The softmax read its marks; the issuers sent its gathers.
            bar_init(&list_free[s], 129);
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

    // The block walks rows blockIdx.x, + gridDim.x, ...: a row's request,
    // position, what of the pool it sees, and its stream of keys. Every
    // lane of a warp asks (the selection's end is found a warp at a time).
    struct Row {
        int row, pos, vis, entries, nw, nwp, total, nt;
        const int* sel;
        const u32* win_pages;
        const u32* ent_pages;
    };
    auto row_at = [&](int r) {
        Row w;
        w.row = base + r;
        const int req = p.requests[w.row];
        w.pos = p.positions[w.row];
        w.vis = min((w.pos + 1) / p.ratio, owned(p.ent_indptr, req, p.ent_page) / p.ratio);
        w.sel = kSelected ? p.selection + static_cast<long long>(w.row) * p.top_k : nullptr;
        w.win_pages = p.win_indices + p.win_indptr[req];
        w.ent_pages = p.ent_indices + p.ent_indptr[req];
        w.entries = w.vis;
        if (kSelected) {
            w.entries = p.top_k;
            for (int n0 = 0; n0 < p.top_k; n0 += 32) {
                const int n = n0 + static_cast<int>(lane);
                const unsigned ended = __ballot_sync(0xffffffffu, n < p.top_k && w.sel[n] < 0);
                if (ended != 0u) {
                    w.entries = n0 + __ffs(ended) - 1;
                    break;
                }
            }
        }
        w.nw = min(w.pos, kWindow - 1) + 1;
        w.nwp = (w.nw + 3) & ~3;
        w.total = w.nwp + w.entries;
        w.nt = (w.total + kBK - 1) / kBK;
        return w;
    };

    if (warp == 11) {
        // The lister: tile J's keys into list J % kLists, ahead of the ring.
        int J = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x)) {
        const Row w = row_at(r);
        const int pos = w.pos, nw = w.nw, nwp = w.nwp, total = w.total, vis = w.vis;
        const int* sel = w.sel;
        const u32* win_pages = w.win_pages;
        const u32* ent_pages = w.ent_pages;
        for (int j = 0; j < w.nt; ++j, ++J) {
            if (J >= kLists) bar_wait(&list_free[J % kLists], ((J / kLists) - 1) & 1);
            u32* l = lists + (J % kLists) * kBK;
            // The tile's keys: their rows, planes and whether they count.
            for (int c = static_cast<int>(lane); c < kBK; c += 32) {
                const int k = j * kBK + c;
                u32 e;
                if (k < nwp) {
                    e = k < nw ? static_cast<u32>(row_in(win_pages, pos - nw + 1 + k,
                                                         p.win_page)) |
                                     kValid
                               : kSlotMask;
                } else if (k < total) {
                    const int id = kSelected ? sel[k - nwp] : k - nwp;
                    e = static_cast<u32>(row_in(ent_pages, (id + 1) * p.ratio - 1, p.ent_page)) |
                        kPool | (id < vis ? kValid : 0u);
                } else {
                    e = kSlotMask | kPool;
                }
                l[c] = e;
            }
            __syncwarp();
            if (lane == 0) bar_arrive(&list_full[J % kLists]);
        }
        }
    } else if (warp == 4 || warp >= 6) {
        const bool lead = warp == 4;
        const int share = lead ? 0 : static_cast<int>(warp) - 5;
        int J = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x)) {
        const Row w = row_at(r);
        for (int j = 0; j < w.nt; ++j, ++J) {
            const int slot = J % kSlots;
            const u32* l = lists + (J % kLists) * kBK;
            bar_wait(&list_full[J % kLists], (J / kLists) & 1);
            if (lead) {
                bar_wait(&empty[slot], ((J / kSlots) & 1) ^ 1);
                if (!kFp8 && lane == 0) bar_arrive_expect_tx(&full[slot], kTile);
            }
            issue_sync();
            if constexpr (kFp8) {
                // A key's 8-dim run an item, 64 keys by 64 runs, dealt round
                // the issuing threads: widened from its row's codes and scale
                // and stored where TMA would have landed it (128-byte rows,
                // their 16-byte words swizzled by the row).
                const int me = share * 32 + static_cast<int>(lane);
                const u32 tile = sbase + kOffRing + slot * kTile;
#pragma unroll 4
                for (int it = me; it < kBK * (kD / 8); it += kIssuers * 32) {
                    const int k = it / (kD / 8);
                    const int c = it % (kD / 8);
                    const u32 e = l[k];
                    const u32 cell = e & kSlotMask;
                    uint4 v = make_uint4(0u, 0u, 0u, 0u);
                    if (cell != kSlotMask) {
                        const bool pool = (e & kPool) != 0;
                        const int page = pool ? p.ent_page : p.win_page;
                        const dsv4::DsRow row = dsv4::ds_row(
                            const_cast<std::uint8_t*>(pool ? p.ent_fp8 : p.win_fp8),
                            static_cast<long long>(cell / static_cast<u32>(page)),
                            static_cast<int>(cell % static_cast<u32>(page)), page);
                        v = dsv4::ds_load8(row.data, row.scales, c * 8);
                    }
                    const int a = c / 8, q = c % 8;
                    const u32 at = tile + atom_at(a) + k * 128 + ((q ^ (k & 7)) * 16);
                    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(at), "r"(v.x),
                                 "r"(v.y), "r"(v.z), "r"(v.w)
                                 : "memory");
                }
                fence_async_shared();
                bar_arrive(&full[slot]);
                issue_sync();
                if (lead && lane == 0) bar_arrive(&list_free[J % kLists]);
                continue;
            }
            // 16 groups of four keys by 8 atoms, dealt round the three warps;
            // a run of 16 rows (window positions, or a page's entries one
            // stride apart) lands as one box from its first group.
            unsigned char* dst = smem + kOffRing + slot * kTile;
            for (int op = static_cast<int>(lane) * kIssuers + share; op < 16 * 8; op += 32 * kIssuers) {
                const int g = op % 16;
                const int a = op / 16;
                const u32* e = l + 4 * g;
                const bool pool = (e[0] & kPool) != 0;
                const u32* run = l + 16 * (g / 4);
                const int stride = pool ? p.ratio : 1;
                const u32 first = run[0] & kSlotMask;
                bool whole = (!pool || p.ent_runs) && first != kSlotMask;
#pragma unroll
                for (int i = 1; i < 16; ++i) {
                    whole = whole && (run[i] & (kSlotMask | kPool)) ==
                                         ((first + static_cast<u32>(i * stride)) | (run[0] & kPool));
                }
                unsigned char* at = dst + atom_at(a) + g * 4 * 128;
                if (whole) {
                    if (g % 4 == 0) {
                        if (pool) {
                            tma_load_3d(at, &full[slot], &map_ent_run, a * 64, p.ratio - 1,
                                        static_cast<int>(first) / p.ratio);
                        } else {
                            tma_load_2d(at, &full[slot], &map_win_run, a * 64,
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
                    gather4(at, &full[slot], pool ? &map_ent : &map_win, a * 64, r[0], r[1], r[2],
                            r[3]);
                }
            }
            issue_sync();
            if (lead && lane == 0) bar_arrive(&list_free[J % kLists]);
        }
        }
    } else if (warp == 5) {
        const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 64, 128);
        const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 64, 256) | (1u << 16);
        // O += P V of tile i (the row's first is `first`), then the tile's
        // slot and P's buffer are free.
        auto pv = [&](int i, int first) {
            bar_wait(&p_full[i & 1], (i >> 1) & 1);
            tc_fence_after();
            if (lane == 0) {
                const u32 tile = sbase + kOffRing + (i % kSlots) * kTile;
                const u32 pa = sbase + kOffP + (i & 1) * kPBytes;
#pragma unroll
                for (int v = 0; v < 2; ++v) {
#pragma unroll
                    for (int s = 0; s < kBK / 16; ++s) {
                        mma_ws(tmem + kColO + v * 128, sdesc_sw128(pa) + 2 * s,
                               sdesc_sw128_mn(tile + v * kAtom + s * 2048, 2 * kAtom), idesc_o,
                               i > first || s > 0);
                    }
                }
                tc_commit(&empty[i % kSlots]);
                tc_commit(&pv_done[i & 1]);
            }
            __syncwarp();
        };
        int J = 0, rr = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x), ++rr) {
            const Row w = row_at(r);
            // This row's Q is in TMEM, and the last row's O was read out.
            bar_wait(q_ready, rr & 1);
            if (rr > 0) bar_wait(o_free, (rr - 1) & 1);
            tc_fence_after();
            const int first = J;
            for (int j = 0; j < w.nt; ++j, ++J) {
                const int slot = J % kSlots;
                if (J >= 2) bar_wait(&s_free[J & 1], ((J - 2) >> 1) & 1);
                bar_wait(&full[slot], (J / kSlots) & 1);
                tc_fence_after();
                if (lane == 0) {
                    const u32 tile = sbase + kOffRing + slot * kTile;
#pragma unroll
                    for (int s = 0; s < 16; ++s) {
                        mma_ws_ts(tmem + kColS + (J & 1) * 64, tmem + kColQ + s * 8,
                                  sdesc_sw128(tile + (s / 4) * 2 * kAtom) + 2 * (s % 4), idesc_s,
                                  s > 0);
                    }
                    tc_commit(&s_full[J & 1]);
                }
                __syncwarp();
                if (j > 0) pv(J - 1, first);
            }
            pv(J - 1, first);
        }
    } else if (warp < 4) {
        // Thread t: head t % 64; its TMEM lane holds the head's first or
        // second half of Q and of O, and it keeps keys [32 half, + 32) of each
        // tile's scores.
        const int t = static_cast<int>(threadIdx.x);
        const int head = t % 64;
        const int half = t / 64;
        const u32 lane_base = (warp * 32) << 16;
        // A row's Q into TMEM, once the S products of the row before it are in.
        auto load_q = [&](int qrow) {
            const uint4* src = reinterpret_cast<const uint4*>(
                p.q + static_cast<long long>(qrow) * p.q_stride + head * kD + half * 256);
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
            tmem_wait_st();
            tc_fence_before();
            bar_arrive(q_ready);
        };
        int J = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x)) {
        const Row w = row_at(r);
        const int row = w.row;
        if (r == static_cast<int>(blockIdx.x)) load_q(row);
        float m_used = neg_inf();
        float l = 0.f;
        for (int j = 0; j < w.nt; ++j, ++J) {
            const int slot = J % kSlots;
            const u32 sb = J & 1;
            bar_wait(&s_full[sb], (J >> 1) & 1);
            tc_fence_after();
            // This thread's 32 keys' partial scores and the partner's 32,
            // each read straight into registers indexed at compile time.
            const int mine0 = half * 32;
            const int theirs0 = (1 - half) * 32;
            u32 vm[32], vt[32];
            tmem_ld32(tmem + lane_base + kColS + sb * 64 + mine0, vm);
            tmem_ld32(tmem + lane_base + kColS + sb * 64 + theirs0, vt);
            tmem_wait_ld();
            tc_fence_before();
            bar_arrive(&s_free[sb]);
            // Four floats a word, word w of thread t at (w * 128 + t): a
            // warp's words land on distinct banks.
            float4* x4 = reinterpret_cast<float4*>(xbuf);
#pragma unroll
            for (int w4 = 0; w4 < 8; ++w4) {
                x4[w4 * 128 + t] =
                    make_float4(__uint_as_float(vt[4 * w4]), __uint_as_float(vt[4 * w4 + 1]),
                                __uint_as_float(vt[4 * w4 + 2]), __uint_as_float(vt[4 * w4 + 3]));
            }
            pair_sync(warp);
            // The tile's marks: the keys that count (listed ahead of the ring).
            bar_wait(&list_full[J % kLists], (J / kLists) & 1);
            const uint4* marks4 = reinterpret_cast<const uint4*>(lists + (J % kLists) * kBK + mine0);
            float s[32];
            float mx = neg_inf();
#pragma unroll
            for (int w4 = 0; w4 < 8; ++w4) {
                const float4 o = x4[w4 * 128 + (t ^ 64)];
                const uint4 m = marks4[w4];
                const float ov[4] = {o.x, o.y, o.z, o.w};
                const u32 mv[4] = {m.x, m.y, m.z, m.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int k = 4 * w4 + i;
                    const float v = __uint_as_float(vm[k]) + ov[i];
                    s[k] = (mv[i] & kValid) ? v * p.scale_log2 : neg_inf();
                    mx = fmaxf(mx, s[k]);
                }
            }
            bar_arrive(&list_free[J % kLists]);
            row_max[t] = mx;
            pair_sync(warp);
            mx = fmaxf(mx, row_max[t ^ 64]);
            const bool grow = mx > m_used + 8.f;
            const float m_new = grow ? mx : m_used;
            const float alpha = grow ? ex2(m_used - m_new) : 1.f;
            const float bse = m_new == neg_inf() ? 0.f : m_new;
            u32 pk[16];
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 32; i += 2) {
                const float e0 = ex2(s[i] - bse);
                const float e1 = ex2(s[i + 1] - bse);
                sum += e0 + e1;
                pk[i / 2] = pack_bf16(e0, e1);
            }
            l = l * alpha + sum;
            m_used = m_new;
            // P's buffer two tiles back is free once its product is in; O
            // rescales once the last product is in.
            if (J >= 2) bar_wait(&pv_done[sb], ((J - 2) >> 1) & 1);
            if (j > 0 && __any_sync(0xffffffffu, grow)) {
                bar_wait(&pv_done[(J - 1) & 1], ((J - 1) >> 1) & 1);
                tc_fence_after();
#pragma unroll 1
                for (u32 c = 0; c < 256; c += 32) {
                    u32 v[32];
                    tmem_ld32(tmem + lane_base + kColO + c, v);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        v[i] = __float_as_uint(__uint_as_float(v[i]) * alpha);
                    }
                    tmem_st32(tmem + lane_base + kColO + c, v);
                }
                tmem_wait_st();
            }
            // P: head `head`'s keys [mine0, + 32) as four 16-byte words of its
            // 128-byte row, swizzled as the MMA reads it.
            const u32 prow = sbase + kOffP + sb * kPBytes + (head / 8) * 1024 + (head % 8) * 128;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const int chunk = half * 4 + w;
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(
                                 prow + ((chunk ^ (head % 8)) * 16)),
                             "r"(pk[4 * w]), "r"(pk[4 * w + 1]), "r"(pk[4 * w + 2]),
                             "r"(pk[4 * w + 3])
                             : "memory");
            }
            fence_async_shared();
            tc_fence_before();
            bar_arrive(&p_full[sb]);
        }
        // The next row's Q lands now, its first products running while this
        // row's O goes out.
        if (r + static_cast<int>(gridDim.x) < live) load_q(base + r + static_cast<int>(gridDim.x));
        bar_wait(&pv_done[(J - 1) & 1], ((J - 1) >> 1) & 1);
        tc_fence_after();
        // The head's sum over both halves of its keys, its sink, and O.
        pair_sync(warp);
        row_max[t] = l;
        pair_sync(warp);
        const float l_all = l + row_max[t ^ 64];
        float scale = 0.f;
        if (l_all > 0.f) {
            constexpr float kLn2 = 0.69314718055994530942f;
            const float lse = (m_used + __log2f(l_all)) * kLn2;
            const float sink =
                p.sinks_f32 ? static_cast<const float*>(p.sinks)[head]
                            : __uint_as_float(static_cast<u32>(static_cast<const u16*>(p.sinks)[head])
                                              << 16);
            scale = 1.f / (l_all * (1.f + __expf(sink - lse)));
        }
        u16* dst = p.o + static_cast<long long>(row) * p.o_stride + static_cast<long long>(head) * kD;
#pragma unroll 1
        for (u32 c = 0; c < 256; c += 32) {
            u32 v[32];
            tmem_ld32(tmem + lane_base + kColO + c, v);
            tmem_wait_ld();
            // Columns [128 v', + 128) are dims [256 v' + 128 half, + 128).
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

}  // namespace pie::attn::dsv4s
