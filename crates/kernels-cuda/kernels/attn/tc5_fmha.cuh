#pragma once

#include "prelude/tc5.cuh"

// A causal (or full) prefill over a paged bf16 KV cache on tcgen05.
//
// A block takes one key head and 128 packed query rows of one request:
// packed row p is token p / G of the request at query head h * G + p % G,
// so the G query heads that share a key head share its tiles. The scores
// S = Q K^T land in TMEM (two buffers, so the next tile's product runs
// while this one's softmax does), the probabilities P go to shared memory
// as the A operand of O += P V, and O accumulates in TMEM. V is read as
// stored, its head dimension contiguous: the B operand of P V is MN-major.
//
// Warps 0-3 load Q, run the softmax (one row a thread, the rows of their
// own TMEM lane quarter) and store O; warp 4's lane 0 is the TMA producer
// of K and V, page by page; warp 5 owns TMEM and its lane 0 issues every
// MMA. The softmax keeps a row's running max in the log2 domain and only
// rescales O when the max outgrows the one P was formed against by more
// than 2^8, so most tiles never touch O.
//
// KV tiles are kBKV tokens: 128, or 64 at a 256-wide head (so two stages
// of K and V fit beside Q and P). The pool is NHD, `page` tokens a page;
// a tile is whole pages, and a page past the request's lands as zeros.

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

__host__ __device__ constexpr u32 smem_bytes(int d, int bkv, int stages) {
    // Q, the K and V rings, P, then barriers; 1024-aligned.
    return u32(d / 64) * 16384u                                 // Q
           + 2u * u32(stages) * u32(d / 64) * u32(bkv) * 128u   // K, V rings
           + u32(bkv / 64) * 16384u                             // P
           + 512u + 1024u;
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

__device__ __forceinline__ void cp16(u32 dst, const void* src, bool live) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(live ? 16 : 0)
                 : "memory");
}

// Where 16-byte chunk `c` of row `r` of a 128B-swizzled atom lies.
__device__ __forceinline__ u32 sw128(u32 r, u32 c) { return r * 128u + ((c ^ (r & 7u)) << 4); }

// The block's work: its request, its first packed row, the request's
// extents. Blocks walk the requests' tiles from the last request's last
// tile back, so the longest causal tiles start first.
struct Work {
    int r, row0, qo_start, qo_len, kv_len, kv_page0;
};

__device__ __forceinline__ bool work_of(const Params& p, int bound, Work& w) {
    const int t = bound - 1 - static_cast<int>(blockIdx.x);
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

template <int D, int BKV, int ST>
__global__ __launch_bounds__(192, 1) void tc5_fmha_prefill(
    const __grid_constant__ tc5::TmaMap map_k, const __grid_constant__ tc5::TmaMap map_v,
    const Params p, const int bound) {
    using namespace tc5;
    static_assert(D == 64 || D == 128 || D == 256, "a head of 64, 128 or 256");
    constexpr int kBKV = BKV;
    static_assert(BKV == 64 || BKV == 128, "a key tile is one or two 64-wide P atoms");
    constexpr u32 kNA = D / 64;            // 64-wide atoms along the head
    constexpr u32 kKA = kBKV / 64;         // 64-wide atoms along a key tile
    constexpr u32 kAtomKV = kBKV * 128;    // one 64-wide atom of a K or V tile
    constexpr u32 kTileKV = kNA * kAtomKV;
    constexpr u32 kOffK = kNA * 16384;
    constexpr u32 kOffV = kOffK + ST * kTileKV;
    constexpr u32 kOffP = kOffV + ST * kTileKV;
    constexpr u32 kOffBar = kOffP + kKA * 16384;
    constexpr u32 kColS = 0;
    constexpr u32 kColO = 2 * kBKV;
    constexpr u32 kCols = 512;
    static_assert(kColO + D <= kCols, "S twice and O fit TMEM");

    Work w;
    if (!work_of(p, bound, w)) return;
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
    u64* s_free = s_full + 2;
    u64* q_ready = s_free + 2;
    u64* p_full = q_ready + 1;
    u64* pv_done = p_full + 1;
    u64* v_fixed = pv_done + 1;
    u32* tmem_slot = reinterpret_cast<u32*>(v_fixed + 1);

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;

    // Key tiles: through the last key any row of the tile reads.
    const int last_row = min(w.row0 + 127, w.qo_len * p.group - 1);
    const int last_tok = last_row / p.group;
    const int reach = p.causal ? min(w.kv_len, w.kv_len - w.qo_len + last_tok + 1) : w.kv_len;
    // A window starts the walk at the first row's oldest key.
    const int first_lo = p.window_left < 0
        ? 0
        : max(0, w.kv_len - w.qo_len + w.row0 / p.group - p.window_left);
    const int jb = first_lo / kBKV;
    const int nt = max(0, (reach + kBKV - 1) / kBKV - jb);

    if (threadIdx.x == 0) {
        for (int s = 0; s < ST; ++s) {
            bar_init(&k_full[s], 1);
            bar_init(&k_empty[s], 1);
            bar_init(&v_full[s], 1);
            bar_init(&v_empty[s], 1);
        }
        for (int s = 0; s < 2; ++s) {
            bar_init(&s_full[s], 1);
            bar_init(&s_free[s], 128);
        }
        bar_init(q_ready, 128);
        bar_init(p_full, 128);
        bar_init(pv_done, 1);
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
            }
            const int pages = (w.kv_len + p.page - 1) / p.page;
            // A tile is 16-token boxes, one 64-wide atom each, spread over the
            // warp's lanes so their loads issue side by side. A box past the
            // request's pages reads a row no tensor holds, which TMA lands as
            // zeros.
            constexpr int kBoxes = (kBKV / 16) * kNA;
            auto load = [&](const TmaMap* map, unsigned char* dst, u64* bar, int j) {
                for (int b = static_cast<int>(lane); b < kBoxes; b += 32) {
                    const int t16 = b / static_cast<int>(kNA);
                    const u32 a = static_cast<u32>(b) % kNA;
                    const int tok = (jb + j) * kBKV + t16 * 16;
                    const int at = tok / p.page;
                    const int row = at < pages ? p.kv_indices[w.kv_page0 + at] * p.page + tok % p.page
                                               : 0x7fffff00;
                    tma_load_3d(dst + a * kAtomKV + t16 * 16 * 128, bar, map, static_cast<int>(a * 64), h,
                                row);
                }
            };
            for (int j = 0; j < nt; ++j) {
                const u32 st = j % ST;
                const u32 ph = (j / ST) & 1;
                bar_wait(&k_empty[st], ph ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&k_full[st], kTileKV);
                __syncwarp();
                load(&map_k, smem + kOffK + st * kTileKV, &k_full[st], j);
                bar_wait(&v_empty[st], ph ^ 1);
                if (lane == 0) bar_arrive_expect_tx(&v_full[st], kTileKV);
                __syncwarp();
                load(&map_v, smem + kOffV + st * kTileKV, &v_full[st], j);
            }
        }
    } else if (warp == 5) {
        if (lane == 0 && nt > 0) {
            const u32 idesc_s = idesc_dense(kF16Bf16, kF16Bf16, 128, kBKV);
            const u32 idesc_o = idesc_dense(kF16Bf16, kF16Bf16, 128, D) | (1u << 16);
            bar_wait(q_ready, 0);
            auto pv = [&](int i) {
                const u32 st = i % ST;
                bar_wait(&v_full[st], (i / ST) & 1);
                if (i == nt - 1) bar_wait(v_fixed, 0);
                bar_wait(p_full, i & 1);
                tc_fence_after();
                const u32 vb = sbase + kOffV + st * kTileKV;
#pragma unroll
                for (u32 s = 0; s < kBKV / 16; ++s) {
                    const u64 a = sdesc_sw128(sbase + kOffP + (s / 4) * 16384) + 2 * (s % 4);
                    const u64 b = sdesc_sw128_mn(vb + s * 2048, kAtomKV);
                    mma_f16(tmem + kColO, a, b, idesc_o, i > 0 || s > 0);
                }
                tc_commit(&v_empty[st]);
                tc_commit(pv_done);
            };
            for (int j = 0; j < nt; ++j) {
                const u32 st = j % ST;
                const u32 sb = j & 1;
                bar_wait(&k_full[st], (j / ST) & 1);
                bar_wait(&s_free[sb], ((j >> 1) & 1) ^ 1);
                tc_fence_after();
                const u32 kb = sbase + kOffK + st * kTileKV;
#pragma unroll
                for (u32 s = 0; s < D / 16; ++s) {
                    const u64 a = sdesc_sw128(sbase + (s / 4) * 16384) + 2 * (s % 4);
                    const u64 b = sdesc_sw128(kb + (s / 4) * kAtomKV) + 2 * (s % 4);
                    mma_f16(tmem + kColS + sb * kBKV, a, b, idesc_s, s > 0);
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
        {
            const unsigned char* src = reinterpret_cast<const unsigned char*>(
                p.q + static_cast<long long>(w.qo_start + (live ? tok : 0)) * p.q_stride +
                static_cast<long long>(qh) * D);
#pragma unroll
            for (u32 a = 0; a < kNA; ++a) {
#pragma unroll
                for (u32 c = 0; c < 8; ++c) {
                    cp16(sbase + a * 16384 + sw128(row, c), src + a * 128 + c * 16, live);
                }
            }
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            fence_async_shared();
            bar_arrive(q_ready);
        }
        float m_used = neg_inf();
        float l = 0.f;
        const u32 lane_base = (warp * 32) << 16;
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
            tc_fence_before();
            bar_arrive(&s_free[st]);
            const int k0 = (jb + j) * kBKV;
            float mx = neg_inf();
#pragma unroll
            for (int c = 0; c < kBKV; ++c) {
                s[c] = k0 + c < limit && k0 + c >= lo ? s[c] * p.scale_log2 : neg_inf();
                mx = fmaxf(mx, s[c]);
            }
            // The max P is formed against: kept until the row's grows past it
            // by 2^8, then O and the sum rescale to the new one.
            const bool grow = mx > m_used + 8.f;
            const float m_new = grow ? mx : m_used;
            const float alpha = grow ? ex2(m_used - m_new) : 1.f;
            const float base = m_new == neg_inf() ? 0.f : m_new;
            float sum = 0.f;
            u32 pk[kBKV / 2];
#pragma unroll
            for (int c = 0; c < kBKV; c += 2) {
                const float e0 = ex2(s[c] - base);
                const float e1 = ex2(s[c + 1] - base);
                sum += e0 + e1;
                pk[c / 2] = pack_bf16(e0, e1);
            }
            l = l * alpha + sum;
            m_used = m_new;
            if (j > 0) {
                bar_wait(pv_done, (j - 1) & 1);
                tc_fence_after();
                if (__any_sync(0xffffffffu, grow)) {
#pragma unroll 1
                    for (u32 c = 0; c < D; c += 32) {
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
            }
            // P's row, K-major and swizzled as the MMA reads it.
#pragma unroll
            for (u32 c = 0; c < kBKV / 8; ++c) {
                const u32 at = sbase + kOffP + (c / 8) * 16384 + sw128(row, c % 8);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(at),
                             "r"(pk[4 * c]), "r"(pk[4 * c + 1]), "r"(pk[4 * c + 2]),
                             "r"(pk[4 * c + 3])
                             : "memory");
            }
            if (j == nt - 1) {
                // The last tile's V rows past the request's keys hold whatever
                // the pool's page held: zero them so P's zeros stay zeros.
                const u32 vst = j % ST;
                bar_wait(&v_full[vst], (j / ST) & 1);
                const int valid = w.kv_len - k0;
                if (static_cast<int>(row) < kBKV && static_cast<int>(row) >= valid) {
#pragma unroll
                    for (u32 a = 0; a < kNA; ++a) {
#pragma unroll
                        for (u32 c = 0; c < 8; ++c) {
                            const u32 at = sbase + kOffV + vst * kTileKV + a * kAtomKV + row * 128 + c * 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" ::"r"(at), "r"(0u)
                                         : "memory");
                        }
                    }
                }
                fence_async_shared();
                bar_arrive(v_fixed);
            }
            fence_async_shared();
            tc_fence_before();
            bar_arrive(p_full);
        }
        if (nt > 0) {
            bar_wait(pv_done, (nt - 1) & 1);
            tc_fence_after();
        }
        // Whole-warp TMEM loads: a dead row still joins its warp's, and
        // stores nothing.
        const float inv = l > 0.f ? 1.f / l : 0.f;
        u16* dst = p.o + static_cast<long long>(w.qo_start + (live ? tok : 0)) * p.o_stride +
                   static_cast<long long>(qh) * D;
#pragma unroll 1
        for (u32 c = 0; c < D; c += 32) {
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
