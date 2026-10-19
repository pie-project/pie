#pragma once

// A routed mxfp4 select at decode widths: each expert's weight rows stream
// once into registers, are dequantized there and meet the expert's few routes
// on mma.sync, with no shared-memory or TMEM round trip between.
//
// A warp holds two m16 tiles of weight rows (32 rows, or a GLU's 16 gate
// rows and their up rows) and walks K 128 codes at a time.
// In each 128-code chunk lane (g, c) owns the 32-code group c of rows g and
// g + 8: one 16-byte load a row. The MMA's K order is free as long as both
// operands agree, so k-step j reads word j / 2 of the group, and the pair
// whose low half is code 3 - p and high half code 7 - p of that word (p the
// pair's place, 0..3) is what one shift lays at bits 12..15 and 28..31: the
// e2m1 code there becomes a bf16 by two masks and a shift. The activations
// are permuted to match, two byte-perms a register.

#include "prelude/device.cuh"
#include "prelude/glu.cuh"

namespace pie::linear {

namespace w4mma {

using ::pie::ld_stream;

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// a * b, bf16 x 2, subnormals kept.
__device__ __forceinline__ unsigned mul2(unsigned a, unsigned b) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}

/// The e2m1 codes at bits 12..15 and 28..31 of `q` as two bf16, each 2^-126
/// times its value (an exponent field of e, a code of exponent 0 subnormal).
__device__ __forceinline__ unsigned e2m1_pair(unsigned q) {
    return (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}

__device__ __forceinline__ unsigned ld_scale2(const u8* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned ld_scale(const u8* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

using ::pie::glu_fn::glu;

__device__ __forceinline__ uint4 ld_act(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

/// An NVFP4 block scale (e4m3) as bf16 bits, times 2^118: an e2m1 code's
/// 2^-126 (`e2m1_pair`) times it is the product times 2^-8, exact and
/// normal across e4m3's range; the epilogue's factor restores the 2^8.
__device__ __forceinline__ unsigned nvfp4_scale_bf16(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0) {
        // m 2^-9 2^118, exact in f32.
        return __float_as_uint(static_cast<float>(m) * 6.4903710731685345e32f) >> 16;
    }
    return ((e + 238u) << 7) | (m << 4);
}

/// One 128-code chunk a lane holds: its rows' code groups and scales, and
/// its routes' activations over the group.
template <int kTiles, int kNb>
struct Chunk {
    uint4 w[kTiles][2];
    unsigned s[kTiles][2];
    uint4 x[kNb][4];
};

// 2^(s - 1) makes up a code's 2^-126 and the block's 2^(s - 127); past
// s = 128 that overflows bf16, and a warp with such a block takes it in
// two factors.
// NVFP4 (`kNv`): a lane's 32 codes are two 16-code blocks, words 0-1 and
// 2-3, the two e4m3 scales in the low and high byte of its scale word.
template <bool wide, bool kNv, int kTiles, int kNb>
__device__ __forceinline__ void compute(const Chunk<kTiles, kNb>& cc, float (&acc)[kTiles][kNb][4]) {
    unsigned s1[kTiles][2], s2[kTiles][2], nv[kTiles][2][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned s = cc.s[t][h];
            if constexpr (kNv) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const unsigned v = nvfp4_scale_bf16((s >> (8 * q)) & 0xFFu);
                    nv[t][h][q] = v | (v << 16);
                }
                s1[t][h] = s2[t][h] = 0u;
            } else {
                const unsigned a = !wide || s <= 128u ? (s + 126u) << 7 : 253u << 7;
                const unsigned b = s <= 128u ? 0x3F80u : s << 7;
                s1[t][h] = a | (a << 16);
                s2[t][h] = b | (b << 16);
            }
        }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (kNv) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) s1[t][h] = nv[t][h][w >> 1];
        }
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            unsigned b0[kNb], b1[kNb];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                const uint4 x = cc.x[b][w];
                const unsigned lo = jj == 0 ? x.y : x.x;
                const unsigned hi = jj == 0 ? x.w : x.z;
                b0[b] = __byte_perm(lo, hi, 0x7632);
                b1[b] = __byte_perm(lo, hi, 0x5410);
            }
            const int p0 = 2 * jj;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const unsigned w0 = (&cc.w[t][0].x)[w];
                const unsigned w1 = (&cc.w[t][1].x)[w];
                unsigned a0 = mul2(e2m1_pair(w0 << (4 * p0)), s1[t][0]);
                unsigned a1 = mul2(e2m1_pair(w1 << (4 * p0)), s1[t][1]);
                unsigned a2 = mul2(e2m1_pair(w0 << (4 * p0 + 4)), s1[t][0]);
                unsigned a3 = mul2(e2m1_pair(w1 << (4 * p0 + 4)), s1[t][1]);
                if constexpr (wide) {
                    a0 = mul2(a0, s2[t][0]);
                    a1 = mul2(a1, s2[t][1]);
                    a2 = mul2(a2, s2[t][0]);
                    a3 = mul2(a3, s2[t][1]);
                }
#pragma unroll
                for (int b = 0; b < kNb; ++b) mma(acc[t][b], a0, a1, a2, a3, b0[b], b1[b]);
            }
        }
    }
}

// A warp's sums landed at their routes' rows: each row's bias (`b_e`, or
// none) after its `factor`, the gate and up rows through the GLU into
// `inter` columns where `kGlu`, else `n` columns as they are.
template <int kGlu, int kTiles, int kNb, class Factor>
__device__ __forceinline__ void w4_land(const float (&acc)[kTiles][kNb][4], Factor factor,
                                        const bf16* b_e, const i32* __restrict__ order, int begin,
                                        int count, bf16* __restrict__ out, int n, int inter,
                                        int row_base, int g, int c, float limit, float galpha) {
    if constexpr (kGlu != 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = row_base + 8 * h + g;
            if (col >= inter) continue;
            const float bg = b_e != nullptr ? bf16_to_f32(b_e[col]) : 0.f;
            const float bu = b_e != nullptr ? bf16_to_f32(b_e[inter + col]) : 0.f;
            const float fg = factor(col), fu = factor(inter + col);
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * inter + col] = f32_to_bf16(glu<kGlu>(
                            acc[0][b][2 * h + e] * fg + bg, acc[1][b][2 * h + e] * fu + bu, limit,
                            galpha));
                    }
                }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row_base + 16 * t + 8 * h + g;
            if (row >= n) continue;
            const float add = b_e != nullptr ? bf16_to_f32(b_e[row]) : 0.f;
            const float f = factor(row);
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * n + row] = f32_to_bf16(acc[t][b][2 * h + e] * f + add);
                    }
                }
        }
}

}  // namespace w4mma

/// `y[route] = x[route / act_div] . W[expert]^T (+ bias[expert])` for the
/// routes `order` lists by expert, `work` naming (expert, slice of
/// `slice_routes` routes) per CTA (from `moe_route_order`). A CTA covers 128
/// weight rows; `kNb` 8-route blocks a slice (`slice_routes` = 8 kNb).
/// With `kGlu`, the bank's `n` rows are `inter` gate rows and as many up
/// rows: a warp's two tiles are 16 gate rows and their up rows, and `out`
/// is the GLU, `inter` wide (a CTA 64 of it).
/// A block-wide exclusive scan of one value a thread (blockDim.x a whole
/// number of warps, at most 1024); the total lands in `*total`.
__device__ __forceinline__ int w4_block_scan(int v, int* warp_sums, int* total) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warps = static_cast<int>(blockDim.x) >> 5;
    int x = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < warps ? warp_sums[lane] : 0;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += y;
        }
        if (lane < warps) warp_sums[lane] = w;
        if (lane == 31) *total = w;
    }
    __syncthreads();
    const int before = warp > 0 ? warp_sums[warp - 1] : 0;
    return before + x - v;
}

/// `moe_route_order`'s product (routes sorted by expert, each expert's
/// offset, the work list of (expert, slice) items in expert order) with the
/// two serial walks over the experts done as block scans; `offsets` holds
/// `num_experts + 3`: the experts' starts and the dropped routes' (those
/// naming no expert), the work items' count, and the live routes' count.
__global__ void moe_route_order_scan(
    const i32* __restrict__ routes,
    i32* __restrict__ order,
    i32* __restrict__ offsets,
    i32* __restrict__ work,
    int work_cap,
    int group_routes,
    int top_k,
    int route_count,
    int num_experts,
    const u32* __restrict__ win)
{
    extern __shared__ i32 order_smem[];
    i32* counts = order_smem;
    i32* fill = counts + num_experts + 1;
    __shared__ int warp_sums[32];
    __shared__ int total;
    const int live = win != nullptr
        ? min(static_cast<int>(win[0]) * top_k, route_count)
        : route_count;
    const int base = win != nullptr ? static_cast<int>(win[1]) * top_k : 0;
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    for (int e = tid; e <= num_experts; e += threads) {
        counts[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    for (int r = tid; r < live; r += threads) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        atomicAdd(counts + e, 1);
    }
    __syncthreads();
    // Each thread a contiguous run of experts: their routes and work items.
    const int per = (num_experts + 1 + threads - 1) / threads;
    const int e0 = min(tid * per, num_experts + 1);
    const int e1 = min(e0 + per, num_experts + 1);
    int mine = 0, items = 0;
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        mine += c;
        if (e < num_experts) items += (c + group_routes - 1) / group_routes;
    }
    int route_at = w4_block_scan(mine, warp_sums, &total);
    __syncthreads();
    int item_at = w4_block_scan(items, warp_sums, &total);
    const int items_total = total;
    __syncthreads();
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        counts[e] = route_at;
        offsets[e] = route_at;
        route_at += c;
        if (e < num_experts) {
            for (int g = 0; g * group_routes < c; ++g, ++item_at)
                if (item_at < work_cap) work[item_at] = e * 65536 + g;
        }
    }
    if (tid == 0) {
        offsets[num_experts + 1] = min(items_total, work_cap);
        // The dropped routes' bucket ends at the live routes' count.
        offsets[num_experts + 2] = live;
    }
    __syncthreads();
    for (int r = tid; r < live; r += threads) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        const int pos = counts[e] + atomicAdd(fill + e, 1);
        order[pos] = base + r;
    }
}

// `kNv`: an NVFP4 bank (an e4m3 scale a 16 codes, `tscale` an f32 tensor
// scale a weight row) in place of mxfp4's e8m0 a 32.
// `kKSplit`: K split across that many groups of the CTA's four warps, each
// walking every `kKSplit`-th chunk, their sums folded through shared memory
// before the first group lands them: a decode's few CTAs keep the card's
// memory busy with more warps, each a share of K.
template <int kNb, int kDepth, int kGlu, bool kNv = false, int kKSplit = 1>
__global__ void __launch_bounds__(128 * kKSplit) moe_select_mxfp4_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha,
    const float* __restrict__ tscale)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;
    constexpr int kSlice = 8 * kNb;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int warp = (static_cast<int>(threadIdx.x) >> 5) & 3;
    const int part = static_cast<int>(threadIdx.x) >> 7;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    // A tile's first row: the next 16 rows, or a GLU's up rows `inter` on.
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    // With K split, a warp past the bank's rows still meets the fold.
    if (kKSplit == 1 && row_base >= span) return;
    const bool rows_live = row_base < span;

    const int groups = k >> 5;
    const int chunks = (groups + 3) >> 2;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    // A row's scales: one a 32-code group (mxfp4), two (NVFP4).
    const int sf_row = kNv ? 2 * groups : groups;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    const u8* wrow[kTiles][2];
    const u8* srow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + r * row_bytes + c * 16;
            srow[t][h] = s_e + static_cast<long long>(r) * sf_row + (kNv ? 2 * c : c);
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 32
            : nullptr;
    }

    using Chunk = w4mma::Chunk<kTiles, kNb>;
    auto load = [&](int ch, Chunk& cc) {
        const bool live = ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cc.w[t][h] = live ? w4mma::ld_stream(wrow[t][h] + ch * 64) : make_uint4(0, 0, 0, 0);
                if constexpr (kNv) {
                    cc.s[t][h] = live ? w4mma::ld_scale2(srow[t][h] + ch * 8) : 0u;
                } else {
                    cc.s[t][h] = live ? w4mma::ld_scale(srow[t][h] + ch * 4) : 127u;
                }
            }
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                cc.x[b][q] = live && xrow[b] != nullptr
                    ? w4mma::ld_act(xrow[b] + ch * 128 + q * 8)
                    : make_uint4(0, 0, 0, 0);
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    // This group's chunks: part, part + kKSplit, ...
    const int mine = rows_live && part < chunks ? (chunks - part + kKSplit - 1) / kKSplit : 0;
    auto chunk_of = [&](int i) { return part + i * kKSplit; };
    Chunk ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < mine) load(chunk_of(d), ring[d]);
    for (int i0 = 0; i0 < mine; i0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int i = i0 + d;
            if (i < mine) {
                bool wide = false;
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
                    wide |= ring[d].s[t][0] > 128u || ring[d].s[t][1] > 128u;
                if constexpr (kNv) {
                    w4mma::compute<false, true>(ring[d], acc);
                } else if (__any_sync(0xffffffffu, wide)) {
                    w4mma::compute<true, false>(ring[d], acc);
                } else {
                    w4mma::compute<false, false>(ring[d], acc);
                }
                if (i + kDepth < mine) load(chunk_of(i + kDepth), ring[d]);
            }
        }
    }
    if constexpr (kKSplit > 1) {
        // The later groups' sums folded into the first's, a lane its own.
        constexpr int kAcc = kTiles * kNb * 4;
        __shared__ float fold[kKSplit - 1][4][kAcc][32];
        if (part > 0) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int b = 0; b < kNb; ++b)
#pragma unroll
                    for (int e = 0; e < 4; ++e) fold[part - 1][warp][(t * kNb + b) * 4 + e][lane] = acc[t][b][e];
        }
        __syncthreads();
        if (part > 0 || !rows_live) return;
#pragma unroll
        for (int p = 0; p < kKSplit - 1; ++p)
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int b = 0; b < kNb; ++b)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[t][b][e] += fold[p][warp][(t * kNb + b) * 4 + e][lane];
    }

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    // NVFP4: a weight row's tensor scale, and the 2^8 the scales held back.
    const float* t_e = kNv ? tscale + static_cast<long long>(expert) * n : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, inter,
                                      row_base, g, c, limit, galpha);
}


namespace w4mma {

// A staged activation chunk's 16-byte run `q` (0..15, codes 8q..8q+7 of
// the chunk's 128) of route row `row`, where the stage keeps it: a lane's
// four runs (4c..4c+3), and its quarter-warp's two rows, on eight bank
// quads.
__device__ __forceinline__ int x_slot(int row, int q) {
    return row * 16 + (q ^ (((q >> 3) & 1) << 1) ^ (row & 1));
}

// `compute` with the routes' activations read from the stage (`xs`, the
// chunk's runs) rather than held a chunk in registers.
template <bool wide, bool kNv, int kTiles, int kNb>
__device__ __forceinline__ void compute_staged(const uint4 (&cw)[kTiles][2], const unsigned (&cs)[kTiles][2],
                                               const uint4* __restrict__ xs, int g, int c,
                                               float (&acc)[kTiles][kNb][4]) {
    unsigned s1[kTiles][2], s2[kTiles][2], nv[kTiles][2][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned s = cs[t][h];
            if constexpr (kNv) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const unsigned v = nvfp4_scale_bf16((s >> (8 * q)) & 0xFFu);
                    nv[t][h][q] = v | (v << 16);
                }
                s1[t][h] = s2[t][h] = 0u;
            } else {
                const unsigned a = !wide || s <= 128u ? (s + 126u) << 7 : 253u << 7;
                const unsigned b = s <= 128u ? 0x3F80u : s << 7;
                s1[t][h] = a | (a << 16);
                s2[t][h] = b | (b << 16);
            }
        }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (kNv) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) s1[t][h] = nv[t][h][w >> 1];
        }
        uint4 xv[kNb];
#pragma unroll
        for (int b = 0; b < kNb; ++b) xv[b] = xs[x_slot(8 * b + g, 4 * c + w)];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            unsigned b0[kNb], b1[kNb];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                const unsigned lo = jj == 0 ? xv[b].y : xv[b].x;
                const unsigned hi = jj == 0 ? xv[b].w : xv[b].z;
                b0[b] = __byte_perm(lo, hi, 0x7632);
                b1[b] = __byte_perm(lo, hi, 0x5410);
            }
            const int p0 = 2 * jj;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const unsigned w0 = (&cw[t][0].x)[w];
                const unsigned w1 = (&cw[t][1].x)[w];
                unsigned a0 = mul2(e2m1_pair(w0 << (4 * p0)), s1[t][0]);
                unsigned a1 = mul2(e2m1_pair(w1 << (4 * p0)), s1[t][1]);
                unsigned a2 = mul2(e2m1_pair(w0 << (4 * p0 + 4)), s1[t][0]);
                unsigned a3 = mul2(e2m1_pair(w1 << (4 * p0 + 4)), s1[t][1]);
                if constexpr (wide) {
                    a0 = mul2(a0, s2[t][0]);
                    a1 = mul2(a1, s2[t][1]);
                    a2 = mul2(a2, s2[t][0]);
                    a3 = mul2(a3, s2[t][1]);
                }
#pragma unroll
                for (int b = 0; b < kNb; ++b) mma(acc[t][b], a0, a1, a2, a3, b0[b], b1[b]);
            }
        }
    }
}

}  // namespace w4mma

/// `moe_select_mxfp4_mma` for a slice of `8 kNb` routes an expert, up to
/// 64: the slice's activations staged a 128-code chunk at a time in shared
/// memory through a `kStages` cp.async ring the CTA's warps share, so a
/// wide fire streams each expert's codes once a 64 routes, not a 32.
// `kTiles` m16 tiles of weight rows a warp: 1 holds a slice of 128 routes
// (`kNb` 16) in the registers 2 holds 64 in.
template <int kNb, int kStages, int kGlu, bool kNv = false, int kTiles = 2>
__global__ void __launch_bounds__(128) moe_select_mxfp4_mma_staged(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha,
    const float* __restrict__ tscale)
{
    static_assert(kGlu == 0 || kTiles == 2, "a GLU warp is a gate tile and its up tile");
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;
    constexpr int kSlice = 8 * kNb;
    constexpr int kStageRuns = kSlice * 16;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    // A warp past the bank's rows still stages its share of the routes.
    const bool rows_live = row_base < span;

    const int groups = k >> 5;
    const int chunks = (groups + 3) >> 2;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    const int sf_row = kNv ? 2 * groups : groups;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    const u8* wrow[kTiles][2];
    const u8* srow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(max(tile_row(t), 0) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + r * row_bytes + c * 16;
            srow[t][h] = s_e + static_cast<long long>(r) * sf_row + (kNv ? 2 * c : c);
        }

    // The runs this thread stages a chunk: rows tid / 16 + 8 j, run tid % 16.
    const int run = tid & 15;
    const bf16* src[kNb];
#pragma unroll
    for (int j = 0; j < kNb; ++j) {
        const int tok = (tid >> 4) + 8 * j;
        src[j] = tok < count ? act + static_cast<long long>(order[begin + tok] / act_div) * k + run * 8
                             : nullptr;
    }
    extern __shared__ __align__(16) uint4 staged_x[];
    auto stage = [&](int ch) {
        if (ch < chunks) {
            uint4* dst = staged_x + (ch % kStages) * kStageRuns;
            const int col = ch * 128 + run * 8;
#pragma unroll
            for (int j = 0; j < kNb; ++j) {
                const int row = (tid >> 4) + 8 * j;
                const bool live = src[j] != nullptr && col < k;
                const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(dst + w4mma::x_slot(row, run)));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr),
                             "l"(live ? src[j] + ch * 128 : act), "r"(live ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto load_w = [&](int ch, uint4 (&cw)[kTiles][2], unsigned (&cs)[kTiles][2]) {
        const bool live = rows_live && ch < chunks && ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cw[t][h] = live ? w4mma::ld_stream(wrow[t][h] + ch * 64) : make_uint4(0, 0, 0, 0);
                if constexpr (kNv) {
                    cs[t][h] = live ? w4mma::ld_scale2(srow[t][h] + ch * 8) : 0u;
                } else {
                    cs[t][h] = live ? w4mma::ld_scale(srow[t][h] + ch * 4) : 127u;
                }
            }
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) stage(s);
    uint4 cw[kTiles][2], nw[kTiles][2];
    unsigned cs[kTiles][2], ns[kTiles][2];
    load_w(0, cw, cs);

    for (int ch = 0; ch < chunks; ++ch) {
        load_w(ch + 1, nw, ns);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        if (rows_live) {
            const uint4* xs = staged_x + (ch % kStages) * kStageRuns;
            bool wide = false;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) wide |= cs[t][0] > 128u || cs[t][1] > 128u;
            if constexpr (kNv) {
                w4mma::compute_staged<false, true>(cw, cs, xs, g, c, acc);
            } else if (__any_sync(0xffffffffu, wide)) {
                w4mma::compute_staged<true, false>(cw, cs, xs, g, c, acc);
            } else {
                w4mma::compute_staged<false, false>(cw, cs, xs, g, c, acc);
            }
        }
        stage(ch + kStages - 1);
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cw[t][h] = nw[t][h];
                cs[t][h] = ns[t][h];
            }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!rows_live) return;

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    const float* t_e = kNv ? tscale + static_cast<long long>(expert) * n : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, inter,
                                      row_base, g, c, limit, galpha);
}

}  // namespace pie::linear
