#pragma once

// A routed select over a block-FP8 bank (e4m3 codes `[experts, n, k]`, one
// ue8m0 exponent a weight row a `kGran` columns laid in the tcgen05 GEMM's
// atoms, a row an expert's weight row) against bf16 activations (W8A16), for
// the GPUs with no FP8 tensor cores: `moe_select_mxfp4_mma`'s walk
// (`linear/moe_w4.cuh`) over one-byte codes.
//
// A warp holds two m16 tiles of weight rows (or a GLU's 16 gate rows and
// their up rows) as the mma's A operand, its routes' activations as B, and
// walks K 64 codes at a time: lane (g, c) owns codes 16c..16c+15 of rows g
// and g + 8, one 16-byte load a row. The K order inside the 64 is permuted
// alike for both operands (`linear/w8a16.cuh`): the mma's step q, lane c,
// element j reads column 16c + 4q + j.

#include "linear/moe_w4.cuh"

namespace pie::linear {

namespace w8mma {

__device__ __forceinline__ long long atom_offset(int row, int col, int cols) {
    const int atoms_k = (cols + 3) / 4;
    return (static_cast<long long>(row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16
        + ((row / 32) % 4) * 4 + col % 4;
}

// 2^(e - 7): the block's 2^(e - 127) and the widening's 2^120.
__device__ __forceinline__ unsigned factor2(unsigned e) {
    const unsigned f = min(e + 120u, 254u) << 7;
    return f | (f << 16);
}

// `linear/w8a16.cuh`'s widening: `sel` 0x9180 codes 0 and 1, 0xB3A2 2 and 3.
__device__ __forceinline__ unsigned widen2(unsigned w, unsigned sel, unsigned f2) {
    unsigned p;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(p) : "r"(w), "r"(sel));
    return w4mma::mul2((p << 4) & 0x87F087F0u, f2);
}

template <int kTiles, int kNb>
struct Chunk {
    uint4 w[kTiles][2];
    unsigned f[kTiles][2];
    uint4 x[kNb][2];
};

__device__ __forceinline__ unsigned word(const uint4& v, int q) {
    return q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
}

template <int kTiles, int kNb>
__device__ __forceinline__ void compute(const Chunk<kTiles, kNb>& cc, float (&acc)[kTiles][kNb][4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        unsigned a[kTiles][4];
#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
            const unsigned lo = word(cc.w[t][0], q);
            const unsigned hi = word(cc.w[t][1], q);
            a[t][0] = widen2(lo, 0x9180u, cc.f[t][0]);
            a[t][1] = widen2(hi, 0x9180u, cc.f[t][1]);
            a[t][2] = widen2(lo, 0xB3A2u, cc.f[t][0]);
            a[t][3] = widen2(hi, 0xB3A2u, cc.f[t][1]);
        }
#pragma unroll
        for (int b = 0; b < kNb; ++b) {
            const uint4& xr = cc.x[b][q >> 1];
            const unsigned b0 = (q & 1) ? xr.z : xr.x;
            const unsigned b1 = (q & 1) ? xr.w : xr.y;
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
                w4mma::mma(acc[t][b], a[t][0], a[t][1], a[t][2], a[t][3], b0, b1);
        }
    }
}

}  // namespace w8mma

/// `y[route] = x[route / act_div] . W[expert]^T` for the routes `order`
/// lists by expert, `work` naming (expert, slice of `8 kNb` routes) per CTA
/// (`moe_route_order`), as `moe_select_mxfp4_mma` lands them; with `kGlu`
/// the bank's `n` rows are `inter` gate rows and as many up rows and `out`
/// the GLU.
template <int kNb, int kDepth, int kGlu, int kGran>
__global__ void __launch_bounds__(128) moe_select_fp8_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ atoms,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha)
{
    static_assert(kGran == 32 || kGran == 128, "a block is 32 or 128 columns of K");
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

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    if (row_base >= span) return;

    const int chunks = k >> 6;
    const int kblocks = k / kGran;
    const u8* w_e = codes + static_cast<long long>(expert) * n * k;

    const u8* wrow[kTiles][2];
    int arow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + static_cast<long long>(r) * k + c * 16;
            arow[t][h] = expert * n + r;
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 16
            : nullptr;
    }

    using Chunk = w8mma::Chunk<kTiles, kNb>;
    auto load = [&](int ch, Chunk& cc) {
        const int kb = kGran == 128 ? (ch >> 1) : ((ch * 64 + 16 * c) >> 5);
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cc.w[t][h] = ld_stream(wrow[t][h] + ch * 64);
                cc.f[t][h] = w8mma::factor2(atoms[w8mma::atom_offset(arow[t][h], kb, kblocks)]);
            }
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                cc.x[b][q] = xrow[b] != nullptr ? w4mma::ld_act(xrow[b] + ch * 64 + q * 8)
                                                : make_uint4(0, 0, 0, 0);
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    Chunk ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < chunks) load(d, ring[d]);
    for (int ch0 = 0; ch0 < chunks; ch0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int ch = ch0 + d;
            if (ch < chunks) {
                w8mma::compute(ring[d], acc);
                if (ch + kDepth < chunks) load(ch + kDepth, ring[d]);
            }
        }
    }

    w4mma::w4_land<kGlu, kTiles, kNb>(acc, [](int) { return 1.f; }, nullptr, order, begin, count,
                                      out, n, inter, row_base, g, c, limit, galpha);
}


/// The same walk over a bf16 bank (`[experts, n, k]` bf16): no widening, 32
/// columns of K a lane's 16-byte load a row, the mma's step q (of two) lane
/// c element j reading column 8c + 4q + j.
// `kTiles` m16 tiles a warp (1 halves a CTA's rows, doubling a decode's
// CTAs; a GLU's warp is always its 16 gate rows and their up rows), and K
// split across `kKSplit` groups of four warps, folded through shared memory
// before the first lands them.
template <int kNb, int kDepth, int kGlu, int kTiles = 2, int kKSplit = 1>
__global__ void __launch_bounds__(128 * kKSplit) moe_select_bf16_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const bf16* __restrict__ codes,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha)
{
    static_assert(kGlu == 0 || kTiles == 2, "a GLU warp is a gate tile and its up tile");
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
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    if (kKSplit == 1 && row_base >= span) return;
    const bool rows_live = row_base < span;

    const int chunks = k >> 5;
    const bf16* w_e = codes + static_cast<long long>(expert) * n * k;
    const bf16* wrow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + static_cast<long long>(r) * k + c * 8;
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 8
            : nullptr;
    }

    struct Chunk {
        uint4 w[kTiles][2];
        uint4 x[kNb];
    };
    auto load = [&](int ch, Chunk& cc) {
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) cc.w[t][h] = w4mma::ld_act(wrow[t][h] + ch * 32);
#pragma unroll
        for (int b = 0; b < kNb; ++b)
            cc.x[b] = xrow[b] != nullptr ? w4mma::ld_act(xrow[b] + ch * 32) : make_uint4(0, 0, 0, 0);
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
    Chunk ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < mine) load(part + d * kKSplit, ring[d]);
    for (int i0 = 0; i0 < mine; i0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int i = i0 + d;
            if (i < mine) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
#pragma unroll
                    for (int b = 0; b < kNb; ++b) {
                        const unsigned b0 = q ? ring[d].x[b].z : ring[d].x[b].x;
                        const unsigned b1 = q ? ring[d].x[b].w : ring[d].x[b].y;
#pragma unroll
                        for (int t = 0; t < kTiles; ++t) {
                            const uint4& lo = ring[d].w[t][0];
                            const uint4& hi = ring[d].w[t][1];
                            w4mma::mma(acc[t][b], q ? lo.z : lo.x, q ? hi.z : hi.x, q ? lo.w : lo.y,
                                       q ? hi.w : hi.y, b0, b1);
                        }
                    }
                }
                if (i + kDepth < mine) load(part + (i + kDepth) * kKSplit, ring[d]);
            }
        }
    }
    if constexpr (kKSplit > 1) {
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

    w4mma::w4_land<kGlu, kTiles, kNb>(acc, [](int) { return 1.f; }, nullptr, order, begin, count,
                                      out, n, inter, row_base, g, c, limit, galpha);
}

}  // namespace pie::linear
