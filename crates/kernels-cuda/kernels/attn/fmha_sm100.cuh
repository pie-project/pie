#pragma once

#include "prim/mma_sm100.cuh"

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

}

}
