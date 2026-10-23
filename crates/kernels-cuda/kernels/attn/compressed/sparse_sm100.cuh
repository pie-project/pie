#pragma once

#include <cuda_fp16.h>

#include "prim/kv_rows.cuh"
#include "attn/fmha_sm100.cuh"

namespace pie::attn::compressed::sm100 {

using tc5::u32;

using tc5::u64;

using fmha::u16;

using fmha::ex2;

using fmha::neg_inf;

using fmha::sdesc_sw128_mn;

struct Params {
    u16* o;
    const u16* q;
    const int* positions;
    const int* requests;
    // Each row's replay floor (null: none): its window reads no key below.
    const int* floors;
    const u32* win_indptr;
    const u32* win_indices;
    const u32* ent_indptr;
    const u32* ent_indices;
    const int* selection;
    const void* sinks;
    const u32* live;
    // DeepSeek's quantized planes (formats kWF, kPF), read where the pass
    // widens their rows itself.
    const std::uint8_t* win_raw;
    const std::uint8_t* ent_raw;
    int sinks_f32;
    int ent_runs;
    int win_page;
    int ent_page;
    // A compact pool: its entries a row each, `ent_page` rows a page.
    int ent_compact;
    int top_k;
    int ratio;
    int rows;
    int q_stride;
    int o_stride;
    float scale_log2;
    kv_rows::Turn turn;
};

constexpr int kD = 512;

constexpr int kBK = 64;

constexpr int kSlots = 3;

constexpr u32 kAtom = kBK * 128;

// 64 keys by 64 bf16
constexpr u32 kTile = (kD / 64) * kAtom;

// 64 KB
constexpr u32 kOffRing = 0;

constexpr u32 kOffP = kOffRing + kSlots * kTile;

constexpr u32 kPBytes = 64 * 128;

// 64 heads by 64 keys
constexpr u32 kOffX = kOffP + 2 * kPBytes;

constexpr u32 kXBytes = 32 * 128 * 4;

constexpr u32 kOffList = kOffX + kXBytes;

constexpr int kLists = 6;

// tile lists, built ahead of the ring
constexpr u32 kOffBar = kOffList + kLists * kBK * 4;

constexpr u32 kColQ = 0;

constexpr u32 kColS = 128;

constexpr u32 kColO = 256;

constexpr u32 kCols = 512;

constexpr u32 kSlotMask = (1u << 30) - 1;

constexpr u32 kValid = 1u << 31;

constexpr int kAbsentRow = 0x7fffff00;

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
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(pie::smem_addr(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(pie::smem_addr(bar))
        : "memory");
}

__device__ __forceinline__ int row_in(const u32* pages, int pos, int page) {
    return static_cast<int>(pages[pos / page]) * page + pos % page;
}

// The two warps whose lanes hold a head's two halves (warps w and w + 2) meet.
__device__ __forceinline__ void pair_sync(u32 warp) {
    if (warp & 1) {
        asm volatile("bar.sync 4, 64;" ::: "memory");
    } else {
        asm volatile("bar.sync 3, 64;" ::: "memory");
    }
}

constexpr int kIssuers = 6;

// warp 4 and warps 6-10; warp 11 lists
__device__ __forceinline__ void issue_sync() { asm volatile("bar.sync 2, %0;" ::"n"(kIssuers * 32) : "memory"); }

// O += P V of one tile once its P is in (`full`'s phase `phase`): V the
// tile's keys read N-major, `acc` false for a row's first product; then the
// tile's slot (`empty`) and P's buffer (`done`) are free.
__device__ __forceinline__ void tile_pv(u32 lane, u32 tmem, u32 tile, u32 pa, u32 idesc_o, bool acc, u64* full,
                                        u32 phase, u64* empty, u64* done) {
    using namespace tc5;
    bar_wait(full, phase);
    tc_fence_after();
    if (lane == 0) {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
#pragma unroll
            for (int s = 0; s < kBK / 16; ++s) {
                mma_ws(tmem + kColO + v * 128, sdesc_sw128(pa) + 2 * s,
                       fmha::sdesc_sw128_mn(tile + v * kAtom + s * 2048, 2 * kAtom), idesc_o, acc || s > 0);
            }
        }
        tc_commit(empty);
        tc_commit(done);
    }
    __syncwarp();
}

// Four values as e4m3 codes, the first in the low byte, saturating.
__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d) {
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
    return static_cast<u32>(lo) | (static_cast<u32>(hi) << 16);
}

// One tile's softmax, thread t (head t % 64, its keys [32 half, + 32)):
// its keys' partial scores `vm` and its partner's `vt` met through `xbuf`,
// masked by the tile's list, the running max moved once a score passes it
// by 2^8 (O rescaled then, its last product in), and P landed at `p_base`
// in bf16, or in e4m3 (`kE4`, its 64 keys the first half of a 128-byte
// row), 128-byte swizzled as the MMA reads it.
template <bool kE4 = false>
__device__ __forceinline__ void tile_softmax(const u32 (&vm)[32], const u32 (&vt)[32], u32 warp, int t,
                                             u32 lane_base, u32 tmem, float* xbuf, float* row_max,
                                             const u32* list, u64* list_full, u32 list_phase, u64* list_free,
                                             float scale_log2, int J, bool rescale, u64* pv_done, u32 p_base,
                                             u64* p_full, float& m_used, float& l) {
    using namespace tc5;
    const int head = t % 64;
    const int half = t / 64;
    const u32 sb = J & 1;
    // Four floats a word, word w of thread t at (w * 128 + t): a warp's
    // words land on distinct banks.
    float4* x4 = reinterpret_cast<float4*>(xbuf);
#pragma unroll
    for (int w4 = 0; w4 < 8; ++w4) {
        x4[w4 * 128 + t] = make_float4(__uint_as_float(vt[4 * w4]), __uint_as_float(vt[4 * w4 + 1]),
                                       __uint_as_float(vt[4 * w4 + 2]), __uint_as_float(vt[4 * w4 + 3]));
    }
    pair_sync(warp);
    // The tile's marks: the keys that count (listed ahead of the ring).
    bar_wait(list_full, list_phase);
    const uint4* marks4 = reinterpret_cast<const uint4*>(list + half * 32);
    float s[32];
    float mx = fmha::neg_inf();
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
            s[k] = (mv[i] & kValid) ? v * scale_log2 : fmha::neg_inf();
            mx = fmaxf(mx, s[k]);
        }
    }
    bar_arrive(list_free);
    row_max[t] = mx;
    pair_sync(warp);
    mx = fmaxf(mx, row_max[t ^ 64]);
    const bool grow = mx > m_used + 8.f;
    const float m_new = grow ? mx : m_used;
    const float alpha = grow ? fmha::ex2(m_used - m_new) : 1.f;
    const float bse = m_new == fmha::neg_inf() ? 0.f : m_new;
    u32 pk[16];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
        const float e0 = fmha::ex2(s[i] - bse);
        const float e1 = fmha::ex2(s[i + 1] - bse);
        sum += e0 + e1;
        if constexpr (kE4) {
            s[i] = e0;
            s[i + 1] = e1;
        } else {
            pk[i / 2] = fmha::pack_bf16x2(e0, e1);
        }
    }
    if constexpr (kE4) {
#pragma unroll
        for (int i = 0; i < 8; ++i) pk[i] = e4m3x4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
    }
    l = l * alpha + sum;
    m_used = m_new;
    // P's buffer two tiles back is free once its product is in; O rescales
    // once the last product is in.
    if (J >= 2) bar_wait(&pv_done[sb], ((J - 2) >> 1) & 1);
    if (rescale && __any_sync(0xffffffffu, grow)) {
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
            fmha::tmem_st32(tmem + lane_base + kColO + c, v);
        }
        fmha::tmem_wait_st();
    }
    // P: head `head`'s keys [32 half, + 32) as four 16-byte words of its
    // 128-byte row.
    const u32 prow = p_base + (head / 8) * 1024 + (head % 8) * 128;
    constexpr int kWords = kE4 ? 2 : 4;
#pragma unroll
    for (int w = 0; w < kWords; ++w) {
        const int chunk = half * kWords + w;
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(prow + ((chunk ^ (head % 8)) * 16)),
                     "r"(pk[4 * w]), "r"(pk[4 * w + 1]), "r"(pk[4 * w + 2]), "r"(pk[4 * w + 3])
                     : "memory");
    }
    fence_async_shared();
    tc_fence_before();
    bar_arrive(&p_full[sb]);
}

}
