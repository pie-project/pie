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
    // normalized output (f32) and base-2 log-sum-exp a head into `part_o` /
    // `part_lse` at (row * splits + split), folded by `mla_sparse_fold`.
    // `splits` 1 writes the row's output straight.
    float* part_o;
    float* part_lse;
    int splits;
    // fp8_ds_mla rows (`attn/mla.cuh`'s `mla_kv_append_fp8`), 656 bytes a
    // cell: the instance that reads them widens a tile's latent into the
    // ring itself and gathers the rotated dims from byte 528.
    const unsigned char* fp8_rows;
};

constexpr int kFp8Row = 656;
constexpr int kFp8Scales = 512;
// The rotated dims' column in a bf16 view of a row (byte 528).
constexpr int kFp8RopeCol = 264;

constexpr int kD = 512;
constexpr int kBK = 64;
constexpr int kSlots = 2;
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

template <int kRope>
struct Layout {
    static constexpr u32 kSlot = kTile + (kRope > 0 ? kRopeBytes : 0u);
    static constexpr u32 kOffRing = 0;
    static constexpr u32 kOffP = kOffRing + kSlots * kSlot;
    static constexpr u32 kOffX = kOffP + 2 * kPBytes;
    static constexpr u32 kOffList = kOffX + kXBytes;
    static constexpr u32 kOffBar = kOffList + kLists * kBK * 4;
    static constexpr u32 kSmem = kOffBar + 1024u + 1024u;
};

__device__ __forceinline__ void issue_sync() { asm volatile("bar.sync 2, %0;" ::"n"(kIssuers * 32) : "memory"); }

// Sixteen e4m3 codes under `scale`, as eight packed bf16 pairs.
__device__ __forceinline__ void widen16(uint4 codes, float scale, u32* out) {
    const u32 w[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned short pair = static_cast<unsigned short>(w[i] >> (16 * h));
            float a, b;
            asm("{\n\t.reg .b32 h2;\n\t.reg .b16 l, hh;\n\t"
                "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
                "mov.b32 {l, hh}, h2;\n\t"
                "cvt.f32.f16 %0, l;\n\t"
                "cvt.f32.f16 %1, hh;\n\t}"
                : "=f"(a), "=f"(b)
                : "h"(pair));
            out[2 * i + h] = pack_bf16(a * scale, b * scale);
        }
    }
}

template <int kRope, bool kFp8 = false>
__global__ __launch_bounds__(384, 1) void mla_sparse(
    const __grid_constant__ tc5::TmaMap map_ckv, const __grid_constant__ tc5::TmaMap map_ckv_run,
    const __grid_constant__ tc5::TmaMap map_kpe, const Params p) {
    using namespace tc5;
    using L = Layout<kRope>;
    static_assert(kRope == 0 || kRope == 64, "the rotated part is 64 wide or absent");
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

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        for (int s = 0; s < kSlots; ++s) {
            // An fp8 tile is landed by its issuers' stores as well as the
            // rotated dims' copies: each of them arrives.
            bar_init(&full[s], kFp8 ? 1 + kIssuers * 32 : 1);
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

    // Row r of the live ones: its request (by the query rows' indptr), its
    // position in that request's keys, the keys it may see, and its selected
    // ids up to the first negative one.
    struct Row {
        int row, pos, last, entries, nt, j0, split;
        const int* sel;
        const u32* pages;
    };
    auto row_at = [&](int u) {
        Row w;
        const int r = u / splits;
        w.split = u % splits;
        w.row = base + r;
        int lo = 0, hi = requests - 1;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (r < static_cast<int>(p.qo_indptr[mid + 1])) hi = mid; else lo = mid + 1;
        }
        const int req = lo;
        const int qo_lo = static_cast<int>(p.qo_indptr[req]);
        const int fresh = static_cast<int>(p.qo_indptr[req + 1]) - qo_lo;
        const int pages_first = static_cast<int>(p.page_indptr[req]);
        const int num_pages = static_cast<int>(p.page_indptr[req + 1]) - pages_first;
        const int kv_len = (num_pages - 1) * p.page + static_cast<int>(p.last_page_lens[req]);
        w.pos = kv_len - fresh + (r - qo_lo);
        // The last key the row sees: its own (causal), or the request's.
        w.last = p.causal ? w.pos : kv_len - 1;
        w.pages = p.page_indices + pages_first;
        w.sel = p.selection + static_cast<long long>(w.row) * p.top_k;
        // The selection may hold -1 anywhere (a pooled index's dropped
        // blocks): the stream runs to its last id, a -1 before it masked.
        // Found from the end, 512 ids a round with their loads all in flight
        // (a full selection's last id is in the first round).
        w.entries = 0;
        for (int end = p.top_k; end > 0 && w.entries == 0; end -= 16 * 32) {
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
        // split walks its share of the row's tiles, one past the row's keys
        // (every key masked) where the row has fewer tiles than splits.
        const int tiles = max((w.entries + kBK - 1) / kBK, 1);
        const int share = (tiles + splits - 1) / splits;
        w.j0 = w.split * share;
        w.nt = min(tiles, w.j0 + share) - w.j0;
        if (w.nt <= 0) {
            w.j0 = tiles;
            w.nt = 1;
        }
        return w;
    };

    if (warp == 11) {
        int J = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x)) {
            const Row w = row_at(r);
            for (int j = 0; j < w.nt; ++j, ++J) {
                if (J >= kLists) bar_wait(&list_free[J % kLists], ((J / kLists) - 1) & 1);
                u32* l = lists + (J % kLists) * kBK;
                for (int c = static_cast<int>(lane); c < kBK; c += 32) {
                    const int k = (w.j0 + j) * kBK + c;
                    u32 e = kSlotMask;
                    if (k < w.entries) {
                        const int id = w.sel[k];
                        if (id >= 0 && id <= w.last) {
                            e = static_cast<u32>(row_in(w.pages, id, p.page)) | kValid;
                        }
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
        constexpr int kOps = kFp8 ? 0 : 16 * 8 + (kRope > 0 ? 16 * 2 : 0);
        int J = 0;
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x)) {
            const Row w = row_at(r);
            for (int j = 0; j < w.nt; ++j, ++J) {
                const int slot = J % kSlots;
                const u32* l = lists + (J % kLists) * kBK;
                bar_wait(&list_full[J % kLists], (J / kLists) & 1);
                if (lead) {
                    bar_wait(&empty[slot], ((J / kSlots) & 1) ^ 1);
                    if (lane == 0) {
                        if constexpr (!kFp8) {
                            bar_arrive_expect_tx(&full[slot], L::kSlot);
                        } else if constexpr (kRope > 0) {
                            bar_arrive_expect_tx(&full[slot], kRopeBytes);
                        } else {
                            bar_arrive(&full[slot]);
                        }
                    }
                }
                issue_sync();
                unsigned char* dst = smem + L::kOffRing + slot * L::kSlot;
                if constexpr (kFp8) {
                    if constexpr (kRope > 0) {
                        // The rotated dims: 16 groups of four keys, twice,
                        // from each row's bf16 tail.
                        for (int op = static_cast<int>(lane) * kIssuers + share; op < 32;
                             op += 32 * kIssuers) {
                            const int g = op % 16;
                            const int copy = op / 16;
                            const u32* e = l + 4 * g;
                            int rr[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const u32 sl = e[i] & kSlotMask;
                                rr[i] = sl == kSlotMask ? kAbsentRow : static_cast<int>(sl);
                            }
                            unsigned char* at = dst + kTile + copy * kAtom + g * 4 * 128;
                            gather4(at, &full[slot], &map_kpe, kFp8RopeCol, rr[0], rr[1], rr[2], rr[3]);
                        }
                    }
                    // The latent: a key's 16-code run a step, widened under
                    // its tile's scale into the two 16-byte chunks it covers
                    // (128-byte swizzle: chunk c of row r lands at c ^ r % 8).
                    // Every load of the tile in flight before the first
                    // widening: a thread's runs are 192 apart, a warp's one
                    // key's 512 contiguous bytes.
                    const int tid = share * 32 + static_cast<int>(lane);
                    constexpr int kPer = (kBK * 32 + kIssuers * 32 - 1) / (kIssuers * 32);
                    uint4 codes[kPer];
                    float scales[kPer];
#pragma unroll
                    for (int k = 0; k < kPer; ++k) {
                        const int it = tid + k * kIssuers * 32;
                        codes[k] = make_uint4(0u, 0u, 0u, 0u);
                        scales[k] = 0.f;
                        if (it < kBK * 32) {
                            const u32 sl = l[it / 32] & kSlotMask;
                            if (sl != kSlotMask) {
                                const unsigned char* row = p.fp8_rows + static_cast<long long>(sl) * kFp8Row;
                                codes[k] = __ldg(reinterpret_cast<const uint4*>(row) + it % 32);
                                scales[k] = __ldg(reinterpret_cast<const float*>(row + kFp8Scales) + (it % 32) / 8);
                            }
                        }
                    }
#pragma unroll
                    for (int k = 0; k < kPer; ++k) {
                        const int it = tid + k * kIssuers * 32;
                        if (it >= kBK * 32) break;
                        const int key = it / 32;
                        const int run = it % 32;
                        u32 out[8];
                        widen16(codes[k], scales[k], out);
                        const int d0 = run * 16;
                        const int c = (d0 % 64) / 8;
                        const u32 at = sbase + L::kOffRing + slot * L::kSlot + atom_at(d0 / 64) + key * 128;
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(
                                             at + (((c + h) ^ (key % 8)) * 16)),
                                         "r"(out[4 * h]), "r"(out[4 * h + 1]), "r"(out[4 * h + 2]),
                                         "r"(out[4 * h + 3])
                                         : "memory");
                        }
                    }
                    fence_async_shared();
                    bar_arrive(&full[slot]);
                }
                for (int op = static_cast<int>(lane) * kIssuers + share; op < kOps; op += 32 * kIssuers) {
                    if (op < 16 * 8) {
                        // The latent: 16 groups of four keys by 8 atoms; a run
                        // of 16 consecutive cells lands as one box.
                        const int g = op % 16;
                        const int a = op / 16;
                        const u32* e = l + 4 * g;
                        const u32* run = l + 16 * (g / 4);
                        const u32 first = run[0] & kSlotMask;
                        bool whole = p.runs != 0 && first != kSlotMask;
#pragma unroll
                        for (int i = 1; i < 16; ++i) {
                            whole = whole && (run[i] & kSlotMask) == first + static_cast<u32>(i);
                        }
                        unsigned char* at = dst + atom_at(a) + g * 4 * 128;
                        if (whole) {
                            if (g % 4 == 0) {
                                tma_load_2d(at, &full[slot], &map_ckv_run, a * 64, static_cast<int>(first));
                            }
                        } else {
                            int rr[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const u32 sl = e[i] & kSlotMask;
                                rr[i] = sl == kSlotMask ? kAbsentRow : static_cast<int>(sl);
                            }
                            gather4(at, &full[slot], &map_ckv, a * 64, rr[0], rr[1], rr[2], rr[3]);
                        }
                    } else {
                        // The rotated dims: 16 groups of four keys, twice.
                        const int o2 = op - 16 * 8;
                        const int g = o2 % 16;
                        const int copy = o2 / 16;
                        const u32* e = l + 4 * g;
                        int rr[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const u32 sl = e[i] & kSlotMask;
                            rr[i] = sl == kSlotMask ? kAbsentRow : static_cast<int>(sl);
                        }
                        unsigned char* at = dst + kTile + copy * kAtom + g * 4 * 128;
                        gather4(at, &full[slot], &map_kpe, 0, rr[0], rr[1], rr[2], rr[3]);
                    }
                }
                issue_sync();
                if (lead && lane == 0) bar_arrive(&list_free[J % kLists]);
            }
        }
    } else if (warp == 5) {
        const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 64, 128);
        const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 64, 256) | (1u << 16);
        auto pv = [&](int i, int first) {
            bar_wait(&p_full[i & 1], (i >> 1) & 1);
            tc_fence_after();
            if (lane == 0) {
                const u32 tile = sbase + L::kOffRing + (i % kSlots) * L::kSlot;
                const u32 pa = sbase + L::kOffP + (i & 1) * kPBytes;
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
            bar_wait(q_ready, rr & 1);
            if (rr > 0) bar_wait(o_free, (rr - 1) & 1);
            tc_fence_after();
            const int first = J;
            for (int j = 0; j < w.nt; ++j, ++J) {
                const int slot = J % kSlots;
                if (J >= 1) bar_wait(s_free, (J - 1) & 1);
                bar_wait(&full[slot], (J / kSlots) & 1);
                tc_fence_after();
                if (lane == 0) {
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
        for (int r = static_cast<int>(blockIdx.x); r < live; r += static_cast<int>(gridDim.x)) {
            const Row w = row_at(r);
            const int row = w.row;
            if (r == static_cast<int>(blockIdx.x)) load_q(row);
            float m_used = neg_inf();
            float l = 0.f;
            for (int j = 0; j < w.nt; ++j, ++J) {
                const u32 sb = J & 1;
                bar_wait(s_full, J & 1);
                tc_fence_after();
                const int mine0 = half * 32;
                const int theirs0 = (1 - half) * 32;
                u32 vm[32], vt[32];
                tmem_ld32(tmem + lane_base + kColS + mine0, vm);
                tmem_ld32(tmem + lane_base + kColS + theirs0, vt);
                tmem_wait_ld();
                tc_fence_before();
                bar_arrive(s_free);
                float4* x4 = reinterpret_cast<float4*>(xbuf);
#pragma unroll
                for (int w4 = 0; w4 < 8; ++w4) {
                    x4[w4 * 128 + t] =
                        make_float4(__uint_as_float(vt[4 * w4]), __uint_as_float(vt[4 * w4 + 1]),
                                    __uint_as_float(vt[4 * w4 + 2]), __uint_as_float(vt[4 * w4 + 3]));
                }
                pair_sync(warp);
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
                const u32 prow = sbase + L::kOffP + sb * kPBytes + (head / 8) * 1024 + (head % 8) * 128;
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
                float* dst = p.part_o + (unit * 64 + head) * kD;
#pragma unroll 1
                for (u32 c = 0; c < 256; c += 32) {
                    u32 v[32];
                    tmem_ld32(tmem + lane_base + kColO + c, v);
                    tmem_wait_ld();
                    const int d0 = static_cast<int>(c / 128) * 256 + half * 128 + static_cast<int>(c % 128);
                    float4* o4 = reinterpret_cast<float4*>(dst + d0);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o4[i] = make_float4(__uint_as_float(v[4 * i]) * scale, __uint_as_float(v[4 * i + 1]) * scale,
                                            __uint_as_float(v[4 * i + 2]) * scale,
                                            __uint_as_float(v[4 * i + 3]) * scale);
                    }
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
    const float* __restrict__ part_o, const float* __restrict__ part_lse, u16* __restrict__ o,
    float* __restrict__ lse, int splits, int o_stride, const u32* __restrict__ live) {
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
    const float* base_o = part_o + (static_cast<long long>(row) * splits * 64 + head) * kD;
    for (int d = static_cast<int>(threadIdx.x) * 4; d < kD; d += 128 * 4) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
        for (int s2 = 0; s2 < splits; ++s2) {
            const float wgt = __shfl_sync(0xffffffffu, share, s2);
            if (wgt == 0.f) continue;
            const float4 v = *reinterpret_cast<const float4*>(base_o + static_cast<long long>(s2) * 64 * kD + d);
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
