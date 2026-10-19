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
// `kTiles` and `kKSplit` as `moe_select_bf16_mma` takes them.
template <int kNb, int kDepth, int kGlu, int kGran, int kTiles = 2, int kKSplit = 1>
__global__ void __launch_bounds__(128 * kKSplit) moe_select_fp8_mma(
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
                w8mma::compute(ring[d], acc);
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

/// The routed select over a bf16 bank as a tiled GEMM, for fires whose
/// experts take tens of routes: a CTA a slice of 64 of one expert's routes
/// by 128 of its weight rows (with `kGlu`, 64 gate rows and their up rows),
/// its eight warps 2 x 4 over the tile (a warp 32 routes by 32 rows: two
/// m16 tiles of routes, four n8 tiles of rows -- a GLU warp's first two its
/// gate rows, the last two their up rows), both operands staged through a
/// `kStages` cp.async ring 64 of K at a time, a row's eight 16-byte chunks
/// turned by its low three bits so a quarter-warp's reads span the banks.
/// Each lane reads K 16c..16c+15 of its rows a stage, the order inside the
/// 64 permuted alike for both operands.
template <int kStages, int kGlu>
__global__ void __launch_bounds__(256) moe_select_bf16_gemm(
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
    constexpr int kBM = 64;
    constexpr int kBN = 128;
    constexpr int kABytes = kBM * 128;
    constexpr int kStageBytes = kABytes + kBN * 128;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kBM;
    const int count = min(offsets[expert + 1] - begin, kBM);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int g = lane >> 2;
    const int t = lane & 3;
    // The CTA's first output column: 64 GLU columns, else 128 rows.
    const int c0 = static_cast<int>(blockIdx.y) * (kGlu != 0 ? 64 : kBN);
    // Local B row -> bank row: a GLU warp's 16 gate columns, then their up.
    auto bank_row = [&](int rr) {
        if constexpr (kGlu != 0) {
            const int w = rr & 31;
            const int col = min(c0 + 16 * (rr >> 5) + (w & 15), inter - 1);
            return w < 16 ? col : inter + col;
        } else {
            return min(c0 + rr, n - 1);
        }
    };

    const bf16* w_e = codes + static_cast<long long>(expert) * n * k;
    const int steps = k >> 6;

    extern __shared__ __align__(16) unsigned char bf16_gemm_smem[];
    const unsigned smem0 = static_cast<unsigned>(__cvta_generic_to_shared(bf16_gemm_smem));

    // Copies a stage: two A chunks and four B chunks a thread.
    const bf16* a_src[2];
    unsigned a_dst[2];
    int a_bytes[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int i = tid + 256 * j;
        const int r = i >> 3;
        const int c = i & 7;
        const int tok = min(r, count - 1);
        a_src[j] = act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 8;
        a_dst[j] = (r * 64 + ((c ^ (r & 7)) * 8)) * 2;
        a_bytes[j] = r < count ? 16 : 0;
    }
    const bf16* b_src[4];
    unsigned b_dst[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = tid + 256 * j;
        const int r = i >> 3;
        const int c = i & 7;
        b_src[j] = w_e + static_cast<long long>(bank_row(r)) * k + c * 8;
        b_dst[j] = kABytes + (r * 64 + ((c ^ (r & 7)) * 8)) * 2;
    }
    int issued = 0;
    auto issue = [&]() {
        if (issued < steps) {
            const unsigned base = smem0 + (issued % kStages) * kStageBytes;
            const long long ka = static_cast<long long>(issued) * 64;
#pragma unroll
            for (int j = 0; j < 2; ++j)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(base + a_dst[j]),
                             "l"(a_src[j] + ka), "r"(a_bytes[j]));
#pragma unroll
            for (int j = 0; j < 4; ++j)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(base + b_dst[j]),
                             "l"(b_src[j] + ka), "r"(16));
        }
        ++issued;
        asm volatile("cp.async.commit_group;\n" ::);
    };

    // A lane's reads a stage, as byte offsets in it: rows g and g + 8 of
    // each m16 tile, row g of each n8 tile, chunks 2t and 2t + 1.
    unsigned a_at[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int ra = wm * 32 + mt * 16 + g;
        const int rb = ra + 8;
        a_at[mt][0] = (ra * 64 + (((2 * t) ^ (ra & 7)) * 8)) * 2;
        a_at[mt][1] = (ra * 64 + (((2 * t + 1) ^ (ra & 7)) * 8)) * 2;
        a_at[mt][2] = (rb * 64 + (((2 * t) ^ (rb & 7)) * 8)) * 2;
        a_at[mt][3] = (rb * 64 + (((2 * t + 1) ^ (rb & 7)) * 8)) * 2;
    }
    unsigned b_at[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = wn * 32 + 8 * j + g;
        b_at[j][0] = kABytes + (r * 64 + (((2 * t) ^ (r & 7)) * 8)) * 2;
        b_at[j][1] = kABytes + (r * 64 + (((2 * t + 1) ^ (r & 7)) * 8)) * 2;
    }

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) issue();

    int slot = 0;
    for (int step = 0; step < steps; ++step) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        // The slot this refills was read last step, by every warp now past
        // the barrier above.
        issue();
        const unsigned char* stage = bf16_gemm_smem + slot * kStageBytes;
        uint4 bw[4][2];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) bw[j][h] = *reinterpret_cast<const uint4*>(stage + b_at[j][h]);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const uint4 a_lo = *reinterpret_cast<const uint4*>(stage + a_at[mt][0]);
            const uint4 a_hi = *reinterpret_cast<const uint4*>(stage + a_at[mt][1]);
            const uint4 b_lo = *reinterpret_cast<const uint4*>(stage + a_at[mt][2]);
            const uint4 b_hi = *reinterpret_cast<const uint4*>(stage + a_at[mt][3]);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4& ar = c < 2 ? a_lo : a_hi;
                const uint4& br = c < 2 ? b_lo : b_hi;
                const unsigned a0 = (c & 1) ? ar.z : ar.x;
                const unsigned a1 = (c & 1) ? br.z : br.x;
                const unsigned a2 = (c & 1) ? ar.w : ar.y;
                const unsigned a3 = (c & 1) ? br.w : br.y;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint4& wr = bw[j][c < 2 ? 0 : 1];
                    w4mma::mma(acc[mt][j], a0, a1, a2, a3, (c & 1) ? wr.z : wr.x, (c & 1) ? wr.w : wr.y);
                }
            }
        }
        slot = slot + 1 == kStages ? 0 : slot + 1;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // acc[mt][j]: routes 32 wm + 16 mt + g (+ 8 for the last two), rows
    // 8 j + 2 t, + 1 of the warp's 32.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int tok = wm * 32 + mt * 16 + g + 8 * hh;
            if (tok >= count) continue;
            const long long route = order[begin + tok];
            if constexpr (kGlu != 0) {
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int col = c0 + 16 * wn + 8 * j + 2 * t + e;
                        if (col < inter)
                            out[route * inter + col] = f32_to_bf16(
                                glu_fn::glu<kGlu>(acc[mt][j][2 * hh + e], acc[mt][j + 2][2 * hh + e], limit, galpha));
                    }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int row = c0 + 32 * wn + 8 * j + 2 * t + e;
                        if (row < n) out[route * n + row] = f32_to_bf16(acc[mt][j][2 * hh + e]);
                    }
            }
        }
}

}  // namespace pie::linear
