#include "linear/codec.cuh"
#include "linear/grouped_fp8.cuh"
#include "prim/device.cuh"
#ifdef __CUDACC_RTC__

#include "prim/mma_sm80.cuh"
#else

#include <cuda_bf16.h>
#include <mma.h>
#endif
#include "linear/moe_w4.cuh"
#include "linear/w8a16.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

/// `y[r, g*n + c] = q(x[r, g*k ..]) . (w[g*n + c, ..] * 2^(s - 127))`, `s`
/// the ue8m0 exponent of the 128x128 block (`scales` spread to the block's
/// rows in tcgen05's atoms, as the load lays an fp8b128 plane's), `q` the
/// e4m3 1x128 quantization. `kTiles` 16-row tiles a CTA, `kSplit` warps a
/// tile (a share of K each); each warp's first `kDepth` steps of codes are
/// in flight before the CTA quantizes the rows. `rows` at most 8, `k`
/// whole `128 kSplit`s, `n` whole `16 kTiles`. Grid `[n / (16 kTiles),
/// groups]`, `32 kSplit kTiles` threads, `rows * (k + 16) + rows * k / 32`
/// bytes of dynamic shared memory (at least `kTiles (kSplit - 1) 512`).
template <int kTiles, int kSplit, int kDepth>
__global__ void __launch_bounds__(32 * kSplit * kTiles) grouped_mma_fp8b128(
    const u8* __restrict__ w, const u8* __restrict__ scales, const bf16* __restrict__ x,
    bf16* __restrict__ y, int rows, int n, int k, int groups) {
    extern __shared__ uint4 gfp8_smem[];
    const int stride = k + 16;  // a row's codes, padded off the banks
    u8* xs = reinterpret_cast<u8*>(gfp8_smem);
    int* xe = reinterpret_cast<int*>(xs + rows * stride);
    const int g = blockIdx.y;
    const int warp_all = static_cast<int>(threadIdx.x) >> 5;
    const int warp = warp_all % kSplit;  // this warp's share of K
    const int tile = (blockIdx.x * kTiles + warp_all / kSplit) * 16;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int blocks = k / 128;

    const int gq = lane >> 2;
    const int c = lane & 3;
    const long long row_a = static_cast<long long>(g) * n + tile + gq;
    const u8* wa = w + row_a * k + 16 * c;
    const u8* wb = wa + 8LL * k;
    // This warp's share of K, in 128-wide blocks.
    const int per_warp = blocks / kSplit;
    const int kb0 = warp * per_warp;

    // A 128-wide block's codes a lane: rows g and g + 8, two 64-wide steps.
    struct Step {
        uint4 a[2][2];
    };
    auto load = [&](int kb, Step& st) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int at = kb * 128 + h * 64;
            st.a[h][0] = gfp8::ld_stream(wa + at);
            st.a[h][1] = gfp8::ld_stream(wb + at);
        }
    };
    // The codes start streaming before the rows are quantized.
    Step ring[kDepth];
#pragma unroll
    for (int d = 0; d < kDepth; ++d)
        if (d < per_warp) load(kb0 + d, ring[d]);

    // The group's slice of each row, quantized: 16 lanes a 128-wide block,
    // a batch of passes' loads in flight before the first reduces.
    {
        using Block = E4m3Block<128>;
        constexpr int kMaxPasses = 8;
        const int per_pass = static_cast<int>(blockDim.x) / Block::kLanes;
        const int total = rows * blocks;
        const int passes = (total + per_pass - 1) / per_pass;
        const long long xw = static_cast<long long>(groups) * k;
        const int sub = lane % Block::kLanes;
        for (int p0 = 0; p0 < passes; p0 += kMaxPasses) {
            uint4 raw[kMaxPasses];
#pragma unroll
            for (int p = 0; p < kMaxPasses; ++p) {
                if (p0 + p >= passes) break;
                const int slot = (p0 + p) * per_pass + static_cast<int>(threadIdx.x) / Block::kLanes;
                const int sl = slot < total ? slot : 0;
                const int r = sl / blocks, b = sl % blocks;
                raw[p] = *reinterpret_cast<const uint4*>(x + r * xw + static_cast<long long>(g) * k +
                                                         b * 128 + sub * 8);
            }
#pragma unroll
            for (int p = 0; p < kMaxPasses; ++p) {
                if (p0 + p >= passes) break;
                const int slot = (p0 + p) * per_pass + static_cast<int>(threadIdx.x) / Block::kLanes;
                const int sl = slot < total ? slot : 0;
                const int r = sl / blocks, b = sl % blocks;
                const unsigned ws[4] = {raw[p].x, raw[p].y, raw[p].z, raw[p].w};
                float f[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    f[2 * i] = __uint_as_float(ws[i] << 16);
                    f[2 * i + 1] = __uint_as_float(ws[i] & 0xFFFF0000u);
                }
                Block q;
                q.hold(f);
                if (slot < total) {
                    *reinterpret_cast<unsigned long long*>(xs + r * stride + b * 128 + sub * 8) =
                        q.codes();
                    if (sub == 0) xe[r * blocks + b] = q.e + 127;
                }
            }
        }
    }
    __syncthreads();

    // The activation row this lane's B fragment reads (column gq).
    const bool col_live = gq < rows;
    const u8* xb = xs + (col_live ? gq : 0) * stride + 16 * c;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i0 = 0; i0 < per_warp; i0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int i = i0 + d;
            if (i >= per_warp) break;
            const int kb = kb0 + i;
            float blk[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint4 xv = make_uint4(0, 0, 0, 0);
                if (col_live) xv = *reinterpret_cast<const uint4*>(xb + kb * 128 + h * 64);
                const uint4 a0 = ring[d].a[h][0], a1 = ring[d].a[h][1];
                gfp8::mma(blk, a0.x, a1.x, a0.y, a1.y, xv.x, xv.y);
                gfp8::mma(blk, a0.z, a1.z, a0.w, a1.w, xv.z, xv.w);
            }
            if (i + kDepth < per_warp) load(kb + kDepth, ring[d]);
            // The tile's 16 rows sit in one 128-row block: one weight
            // exponent a step, spread to each row in tcgen05's atoms.
            // C: (row gq, col 2c), (gq, 2c + 1), (gq + 8, 2c), (gq + 8, 2c + 1).
            const float sw = gfp8::exp2_biased(scales[atom_offset(row_a, kb, blocks)]);
            const int c0 = 2 * c, c1 = 2 * c + 1;
            const float s0 = c0 < rows ? sw * gfp8::exp2_biased(xe[c0 * blocks + kb]) : 0.f;
            const float s1 = c1 < rows ? sw * gfp8::exp2_biased(xe[c1 * blocks + kb]) : 0.f;
            acc[0] = fmaf(blk[0], s0, acc[0]);
            acc[1] = fmaf(blk[1], s1, acc[1]);
            acc[2] = fmaf(blk[2], s0, acc[2]);
            acc[3] = fmaf(blk[3], s1, acc[3]);
        }
    }

    // A tile's shares of K summed: its warps 1.. hand theirs to 0.
    __syncthreads();
    float* red = reinterpret_cast<float*>(gfp8_smem) + (warp_all / kSplit) * (kSplit - 1) * 4 * 32;
    if (warp != 0) {
#pragma unroll
        for (int e = 0; e < 4; ++e) red[((warp - 1) * 4 + e) * 32 + lane] = acc[e];
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int e = 0; e < 4; ++e)
#pragma unroll
        for (int q = 0; q < kSplit - 1; ++q) acc[e] += red[(q * 4 + e) * 32 + lane];
    const long long ys = static_cast<long long>(groups) * n;
    const long long at = static_cast<long long>(g) * n + tile + gq;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int col = 2 * c + (e & 1);
        const int row = e >= 2 ? 8 : 0;
        if (col < rows) y[col * ys + at + row] = f32_to_bf16(acc[e]);
    }
}

namespace wmma = ::nvcuda::wmma;

// A routed bf16 select at decode widths: the bf16 analog of
// `moe_select_mxfp4_mma`, without its GLU. Each expert's weight rows and the
// routes' rows are staged a K-tile at a time into shared memory with
// `cp.async`, read to the mma fragments by `ldmatrix`, and met on mma.sync.
// The routing (order/offsets/work) and the epilogue (`w4mma::w4_land`) are
// shared with the mxfp4 kernel; the fragments are the ones `attn/mla.cuh`
// loads, so the output fragment is the layout `w4_land` reads.
// The bf16 select's K step and its pipeline's stages (`moe_select_bf16_mma`).
constexpr int kBf16MmaBK = 64;

constexpr int kBf16MmaStages = 3;

template <int kNb>
__global__ void __launch_bounds__(128) moe_select_bf16_mma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const bf16* __restrict__ weight,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div, int n, int k, int num_experts)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;   // 128 weight rows a CTA
    constexpr int kSlice = 8 * kNb;           // routes a slice (B of the GEMM)
    constexpr int kBK = kBf16MmaBK;           // K staged a step (four mma k)
    constexpr int kPad = kBK + 8;             // smem row stride (ldmatrix)
    constexpr int kStages = kBf16MmaStages;
    constexpr int kChunks = kBK / 8;          // 16-byte chunks a row a step
    constexpr int kStage = (kRowsCta + kSlice) * kPad;

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
    // The CTA's 128 rows and this warp's 32 of them. A warp whose rows run
    // past the bank still joins the cooperative load and the barriers;
    // `w4_land` drops its out-of-range rows.
    const int cta_row0 = static_cast<int>(blockIdx.y) * kRowsCta;
    const int row_base = cta_row0 + warp * kRowsWarp;

    const bf16* w_e = weight + static_cast<long long>(expert) * n * k;
    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;

    // `kStages` stages, each 128 weight rows then the slice's route rows.
    extern __shared__ bf16 smem[];

    // Stage `st` takes K-step `step`: 16 bytes a cp.async. Rows past the
    // bank or routes past the slice are zeroed so the mma reads defined
    // smem (their outputs are not stored).
    auto load = [&](int step, int st) {
        bf16* sW = smem + st * kStage;
        bf16* sX = sW + kRowsCta * kPad;
        const int k0 = step * kBK;
#pragma unroll
        for (int i = tid; i < kRowsCta * kChunks; i += 128) {
            const int row = i / kChunks;
            const int ch = (i % kChunks) * 8;
            bf16* dst = sW + row * kPad + ch;
            if (cta_row0 + row < n) {
                __pipeline_memcpy_async(
                    dst, w_e + static_cast<long long>(cta_row0 + row) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < kSlice * kChunks; i += 128) {
            const int tok = i / kChunks;
            const int ch = (i % kChunks) * 8;
            bf16* dst = sX + tok * kPad + ch;
            if (tok < count) {
                const long long r = static_cast<long long>(order[begin + tok] / act_div);
                __pipeline_memcpy_async(dst, act + r * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
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

    // Every step commits a group (empty past the end), so the wait for
    // step `s` is always the same count behind.
    const int steps = k / kBK;
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) load(s, s);
        __pipeline_commit();
    }
    for (int s = 0; s < steps; ++s) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        // The stage refilled here was read at step s - 1, behind the barrier.
        const int next = s + kStages - 1;
        if (next < steps) load(next, next % kStages);
        __pipeline_commit();

        const bf16* sW = smem + (s % kStages) * kStage;
        const bf16* sX = sW + kRowsCta * kPad;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 16) {
            unsigned bk[kNb][2];
#pragma unroll
            for (int b = 0; b < kNb; ++b) ld_b(bk[b], sX + 8 * b * kPad + kk, kPad, lane);
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                unsigned a[4];
                ld_a(a, sW + (warp * kRowsWarp + 16 * t) * kPad + kk, kPad, lane);
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    mma_m16n8k16(acc[t][b], a, bk[b]);
                }
            }
        }
    }

    auto factor = [&](int) { return 1.f; };
    w4mma::w4_land<0, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, n, row_base,
                                   g, c, 0.f, 1.f);
}

namespace w8mma {

// The codes' scales and widening are the dense W8A16 projection's.
using w8a16::atom_offset;
using w8a16::factor2;
using w8a16::widen2;
using w8a16::word_of;

template <int kTiles, int kNb>
struct Chunk {
    uint4 w[kTiles][2];
    unsigned f[kTiles][2];
    uint4 x[kNb][2];
};

template <int kTiles, int kNb>
__device__ __forceinline__ void compute(const Chunk<kTiles, kNb>& cc, float (&acc)[kTiles][kNb][4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        unsigned a[kTiles][4];
#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
            const unsigned lo = word_of(cc.w[t][0], q);
            const unsigned hi = word_of(cc.w[t][1], q);
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
                mma_m16n8k16(acc[t][b], a[t], {b0, b1});
        }
    }
}

}

/// `y[route] = x[route / act_div] . W[expert]^T` for the routes `order`
/// lists by expert, `work` naming (expert, slice of `8 kNb` routes) per CTA
/// (`moe_route_order`), as `moe_select_mxfp4_mma` lands them; with `kGlu`
/// the bank's `n` rows are `inter` gate rows and as many up rows and `out`
/// the GLU.
// `kTiles` and `kKSplit` as `moe_select_bf16_held` takes them.
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
    if (!w4mma::fold_ksplit<kKSplit>(acc, part, warp, lane, rows_live)) return;

    w4mma::w4_land<kGlu, kTiles, kNb>(acc, [](int) { return 1.f; }, nullptr, order, begin, count,
                                      out, n, inter, row_base, g, c, limit, galpha);
}

/// The routed select over a bf16 bank as a tiled GEMM, for fires whose
/// experts take tens of routes: a CTA a slice of 64 of one expert's routes
/// by 128 of its weight rows (with `kGlu`, 64 gate rows and their up rows),
/// its warps 2 x `kWN` over the tile (a warp 32 routes by `128 / kWN` rows:
/// two m16 tiles of routes, `16 / kWN` n8 tiles of rows -- a GLU warp's
/// first half its gate rows, the second half their up rows), both operands
/// staged through a `kStages` cp.async ring 64 of K at a time, a row's eight
/// 16-byte chunks turned by its low three bits so a quarter-warp's reads
/// span the banks. Each lane reads K 16c..16c+15 of its rows a stage, the
/// order inside the 64 permuted alike for both operands.
template <int kStages, int kGlu, int kWN>
__global__ void __launch_bounds__(64 * kWN) moe_select_bf16_gemm(
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
    // `kWN` warps along the rows, a warp's rows `kNT` n8 tiles.
    constexpr int kThreads = 64 * kWN;
    constexpr int kNT = kBN / (8 * kWN);
    constexpr int kWRows = 8 * kNT;
    constexpr int kACopies = kBM * 8 / kThreads;
    constexpr int kBCopies = kBN * 8 / kThreads;

    const int cta = static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x);
    const int bx = cta / static_cast<int>(gridDim.y);
    if (bx >= offsets[num_experts + 1]) return;
    const int item = work[bx];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kBM;
    const int count = min(offsets[expert + 1] - begin, kBM);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / kWN;
    const int wn = warp % kWN;
    const int g = lane >> 2;
    const int t = lane & 3;
    // The CTA's first output column: 64 GLU columns, else 128 rows.
    const int c0 = cta % static_cast<int>(gridDim.y) * (kGlu != 0 ? 64 : kBN);
    // Local B row -> bank row: a GLU warp's half-tile of gate columns, then
    // their up.
    auto bank_row = [&](int rr) {
        if constexpr (kGlu != 0) {
            const int w = rr % kWRows;
            const int col = min(c0 + (kWRows / 2) * (rr / kWRows) + (w % (kWRows / 2)), inter - 1);
            return w < kWRows / 2 ? col : inter + col;
        } else {
            return min(c0 + rr, n - 1);
        }
    };

    const bf16* w_e = codes + static_cast<long long>(expert) * n * k;
    const int steps = k >> 6;

    extern __shared__ __align__(16) unsigned char bf16_gemm_smem[];
    const unsigned smem0 = static_cast<unsigned>(__cvta_generic_to_shared(bf16_gemm_smem));

    // Copies a stage: two A chunks and four B chunks a thread.
    const bf16* a_src[kACopies];
    unsigned a_dst[kACopies];
    int a_bytes[kACopies];
#pragma unroll
    for (int j = 0; j < kACopies; ++j) {
        const int i = tid + kThreads * j;
        const int r = i >> 3;
        const int c = i & 7;
        const int tok = min(r, count - 1);
        a_src[j] = act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 8;
        a_dst[j] = (r * 64 + ((c ^ (r & 7)) * 8)) * 2;
        a_bytes[j] = r < count ? 16 : 0;
    }
    const bf16* b_src[kBCopies];
    unsigned b_dst[kBCopies];
#pragma unroll
    for (int j = 0; j < kBCopies; ++j) {
        const int i = tid + kThreads * j;
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
            for (int j = 0; j < kACopies; ++j)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(base + a_dst[j]),
                             "l"(a_src[j] + ka), "r"(a_bytes[j]));
#pragma unroll
            for (int j = 0; j < kBCopies; ++j)
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
    unsigned b_at[kNT][2];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        const int r = wn * kWRows + 8 * j + g;
        b_at[j][0] = kABytes + (r * 64 + (((2 * t) ^ (r & 7)) * 8)) * 2;
        b_at[j][1] = kABytes + (r * 64 + (((2 * t + 1) ^ (r & 7)) * 8)) * 2;
    }

    float acc[2][kNT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
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
        if (wm * 32 >= count) {
            slot = slot + 1 == kStages ? 0 : slot + 1;
            continue;
        }
        uint4 bw[kNT][2];
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) bw[j][h] = *reinterpret_cast<const uint4*>(stage + b_at[j][h]);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            // An m16 tile past the slice's routes multiplies zeros.
            if (wm * 32 + mt * 16 >= count) continue;
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
                for (int j = 0; j < kNT; ++j) {
                    const uint4& wr = bw[j][c < 2 ? 0 : 1];
                    mma_m16n8k16(acc[mt][j], {a0, a1, a2, a3}, {(c & 1) ? wr.z : wr.x, (c & 1) ? wr.w : wr.y});
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
                for (int j = 0; j < kNT / 2; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int col = c0 + (kWRows / 2) * wn + 8 * j + 2 * t + e;
                        if (col < inter)
                            out[route * inter + col] = f32_to_bf16(act::glu<kGlu>(
                                acc[mt][j][2 * hh + e], acc[mt][j + kNT / 2][2 * hh + e], limit, galpha));
                    }
            } else {
#pragma unroll
                for (int j = 0; j < kNT; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int row = c0 + kWRows * wn + 8 * j + 2 * t + e;
                        if (row < n) out[route * n + row] = f32_to_bf16(acc[mt][j][2 * hh + e]);
                    }
            }
        }
}

/// `moe_select_bf16_gemm` over a block-FP8 bank (W8A16), for a prefill's
/// thousands of routes: each stage's codes (a weight row's 64 bytes a stage
/// row) land beside the activations, and the CTA widens them once -- each
/// thread four rows' 16-code runs, by their block's exponent -- into a bf16
/// tile laid as the bf16 GEMM lays its weights, which every warp's mma then
/// reads. The widening is paid once a 64 routes, not once a route tile.
template <int kStages, int kGlu, int kGran, int kWN>
__global__ void __launch_bounds__(64 * kWN) moe_select_fp8_gemm(
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
    constexpr int kBM = 64;
    constexpr int kBN = 128;
    constexpr int kABytes = kBM * 128;
    constexpr int kStageBytes = kABytes + kBN * 64;
    constexpr int kThreads = 64 * kWN;
    constexpr int kNT = kBN / (8 * kWN);
    constexpr int kWRows = 8 * kNT;
    constexpr int kACopies = kBM * 8 / kThreads;
    constexpr int kBCopies = kBN * 4 / kThreads;

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
    const int wm = warp / kWN;
    const int wn = warp % kWN;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c0 = static_cast<int>(blockIdx.y) * (kGlu != 0 ? 64 : kBN);
    auto bank_row = [&](int rr) {
        if constexpr (kGlu != 0) {
            const int w = rr % kWRows;
            const int col = min(c0 + (kWRows / 2) * (rr / kWRows) + (w % (kWRows / 2)), inter - 1);
            return w < kWRows / 2 ? col : inter + col;
        } else {
            return min(c0 + rr, n - 1);
        }
    };

    const u8* w_e = codes + static_cast<long long>(expert) * n * k;
    const int steps = k >> 6;
    const int kblocks = k / kGran;

    // The stages, then the widened tile (`kBN` rows of 64 bf16, swizzled).
    extern __shared__ __align__(16) unsigned char fp8_gemm_smem[];
    unsigned char* wide = fp8_gemm_smem + kStages * kStageBytes;
    const unsigned smem0 = static_cast<unsigned>(__cvta_generic_to_shared(fp8_gemm_smem));

    const bf16* a_src[kACopies];
    unsigned a_dst[kACopies];
    int a_bytes[kACopies];
#pragma unroll
    for (int j = 0; j < kACopies; ++j) {
        const int i = tid + kThreads * j;
        const int r = i >> 3;
        const int c = i & 7;
        const int tok = min(r, count - 1);
        a_src[j] = act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 8;
        a_dst[j] = (r * 64 + ((c ^ (r & 7)) * 8)) * 2;
        a_bytes[j] = r < count ? 16 : 0;
    }
    // This thread's code runs: rows i / 4, run i % 4 (16 codes), copied a
    // stage and widened from it; the row's exponents' row in the atoms.
    const u8* b_src[kBCopies];
    unsigned b_dst[kBCopies];
    int b_atoms_row[kBCopies];
    unsigned b_wide[kBCopies];
#pragma unroll
    for (int j = 0; j < kBCopies; ++j) {
        const int i = tid + kThreads * j;
        const int r = i >> 2;
        const int c = i & 3;
        b_src[j] = w_e + static_cast<long long>(bank_row(r)) * k + c * 16;
        b_dst[j] = kABytes + r * 64 + c * 16;
        b_atoms_row[j] = expert * n + bank_row(r);
        b_wide[j] = (r * 64 + (((2 * c) ^ (r & 7)) * 8)) * 2;
    }
    int issued = 0;
    auto issue = [&]() {
        if (issued < steps) {
            const unsigned base = smem0 + (issued % kStages) * kStageBytes;
            const long long ka = static_cast<long long>(issued) * 64;
#pragma unroll
            for (int j = 0; j < kACopies; ++j)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(base + a_dst[j]),
                             "l"(a_src[j] + ka), "r"(a_bytes[j]));
#pragma unroll
            for (int j = 0; j < kBCopies; ++j)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(base + b_dst[j]),
                             "l"(b_src[j] + ka), "r"(16));
        }
        ++issued;
        asm volatile("cp.async.commit_group;\n" ::);
    };

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
    unsigned b_at[kNT][2];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        const int r = wn * kWRows + 8 * j + g;
        b_at[j][0] = (r * 64 + (((2 * t) ^ (r & 7)) * 8)) * 2;
        b_at[j][1] = (r * 64 + (((2 * t + 1) ^ (r & 7)) * 8)) * 2;
    }

    float acc[2][kNT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < kNT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) issue();

    int slot = 0;
    for (int step = 0; step < steps; ++step) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        const unsigned char* stage = fp8_gemm_smem + slot * kStageBytes;
        // Widen this stage's codes into the tile (every warp is past last
        // step's mma: the barrier above).
#pragma unroll
        for (int j = 0; j < kBCopies; ++j) {
            const int c = (tid + kThreads * j) & 3;
            const int kb = kGran == 128 ? (step >> 1) : ((step * 64 + 16 * c) >> 5);
            const unsigned f = w8mma::factor2(__ldg(atoms + w8mma::atom_offset(b_atoms_row[j], kb, kblocks)));
            const uint4 w = *reinterpret_cast<const uint4*>(stage + b_dst[j]);
            uint4 lo, hi;
            lo.x = w8mma::widen2(w.x, 0x9180u, f);
            lo.y = w8mma::widen2(w.x, 0xB3A2u, f);
            lo.z = w8mma::widen2(w.y, 0x9180u, f);
            lo.w = w8mma::widen2(w.y, 0xB3A2u, f);
            hi.x = w8mma::widen2(w.z, 0x9180u, f);
            hi.y = w8mma::widen2(w.z, 0xB3A2u, f);
            hi.z = w8mma::widen2(w.w, 0x9180u, f);
            hi.w = w8mma::widen2(w.w, 0xB3A2u, f);
            const int r = (tid + kThreads * j) >> 2;
            *reinterpret_cast<uint4*>(wide + b_wide[j]) = lo;
            *reinterpret_cast<uint4*>(wide + (r * 64 + (((2 * c + 1) ^ (r & 7)) * 8)) * 2) = hi;
        }
        __syncthreads();
        issue();
        if (wm * 32 < count) {
            uint4 bw[kNT][2];
#pragma unroll
            for (int j = 0; j < kNT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) bw[j][h] = *reinterpret_cast<const uint4*>(wide + b_at[j][h]);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                if (wm * 32 + mt * 16 >= count) continue;
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
                    for (int j = 0; j < kNT; ++j) {
                        const uint4& wr = bw[j][c < 2 ? 0 : 1];
                        mma_m16n8k16(acc[mt][j], {a0, a1, a2, a3}, {(c & 1) ? wr.z : wr.x, (c & 1) ? wr.w : wr.y});
                    }
                }
            }
        }
        slot = slot + 1 == kStages ? 0 : slot + 1;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int tok = wm * 32 + mt * 16 + g + 8 * hh;
            if (tok >= count) continue;
            const long long route = order[begin + tok];
            if constexpr (kGlu != 0) {
#pragma unroll
                for (int j = 0; j < kNT / 2; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int col = c0 + (kWRows / 2) * wn + 8 * j + 2 * t + e;
                        if (col < inter)
                            out[route * inter + col] = f32_to_bf16(act::glu<kGlu>(
                                acc[mt][j][2 * hh + e], acc[mt][j + kNT / 2][2 * hh + e], limit, galpha));
                    }
            } else {
#pragma unroll
                for (int j = 0; j < kNT; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int row = c0 + kWRows * wn + 8 * j + 2 * t + e;
                        if (row < n) out[route * n + row] = f32_to_bf16(acc[mt][j][2 * hh + e]);
                    }
            }
        }
}

// The fp8 select's pipeline: K steps of one 128-wide block, three stages.
constexpr int kFp8a8Stages = 3;

/// `y[route] = q(x[route / act_div]) . (W[expert] * 2^(s - 127))^T` for a
/// 128x128-block FP8 bank (e4m3 codes, each block's ue8m0 exponent spread to
/// its rows in tcgen05's atoms, a weight row an atom row) against the rows
/// quantized by `quantize_rows_e4m3b128` (W8A8): `moe_select_bf16_mma`'s
/// staging over e4m3 x e4m3 mma.sync, each 128-wide block's sums scaled by
/// its two exponents. `kDense`: one plane, rows in order, no routing (`order`,
/// `offsets`, `work` unread, a CTA the `kSlice` rows at `blockIdx.x`), K
/// split `gridDim.z` ways: each split's f32 sums into `part`
/// (`[splits][rows][n]`) for `fp8a8_fold`, or one split's straight out.
template <int kNb, bool kDense, int kWN = 1>
__global__ void __launch_bounds__(128 * kWN) moe_select_fp8a8_mma(
    const u8* __restrict__ xq,
    const u8* __restrict__ xe,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ weight,
    const u8* __restrict__ atoms,
    bf16* __restrict__ out,
    float* __restrict__ part,
    int act_div, int n, int k, int num_experts, int rows)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;   // 128 weight rows a CTA: one block's
    constexpr int kSlice = 8 * kNb;           // routes a slice (B of the GEMM)
    constexpr int kBK = 128;                  // K a step: one block's codes
    constexpr int kPad = kBK + 16;            // smem row stride in bytes (ldmatrix)
    constexpr int kStages = kFp8a8Stages;
    constexpr int kChunks = kBK / 16;
    constexpr int kStage = (kRowsCta + kSlice) * kPad;

    int expert = 0, begin = 0, count = 0;
    if constexpr (kDense) {
        begin = static_cast<int>(blockIdx.x) * kSlice;
        count = min(rows - begin, kSlice);
    } else {
        if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
        const int item = work[blockIdx.x];
        expert = item >> 16;
        begin = offsets[expert] + (item & 0xFFFF) * kSlice;
        count = min(offsets[expert + 1] - begin, kSlice);
    }
    if (count <= 0) return;
    auto act_row = [&](int tok) {
        if constexpr (kDense) return begin + tok;
        else return order[begin + tok] / act_div;
    };

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = (tid >> 5) & 3;
    const int wb = tid >> 7;
    constexpr int kNbW = kNb / kWN;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int cta_row0 = static_cast<int>(blockIdx.y) * kRowsCta;
    const int row_base = cta_row0 + warp * kRowsWarp;
    const int blocks = k / kBK;
    // This CTA's share of K, in blocks: all of it but a split dense fire.
    const int per = (blocks + static_cast<int>(gridDim.z) - 1) / static_cast<int>(gridDim.z);
    const int s_lo = min(blocks, static_cast<int>(blockIdx.z) * per);
    const int s_hi = min(blocks, s_lo + per);

    const u8* w_e = weight + static_cast<long long>(expert) * n * k;
    const u8* atoms_at = atoms + atom_offset(expert * n + cta_row0, 0, blocks);

    extern __shared__ uint4 fp8a8_smem[];
    u8* smem = reinterpret_cast<u8*>(fp8a8_smem);

    auto load = [&](int step, int st) {
        u8* sW = smem + st * kStage;
        u8* sX = sW + kRowsCta * kPad;
        const int k0 = step * kBK;
#pragma unroll
        for (int i = tid; i < kRowsCta * kChunks; i += 128 * kWN) {
            const int row = i / kChunks;
            const int ch = (i % kChunks) * 16;
            u8* dst = sW + row * kPad + ch;
            if (cta_row0 + row < n) {
                __pipeline_memcpy_async(
                    dst, w_e + static_cast<long long>(cta_row0 + row) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < kSlice * kChunks; i += 128 * kWN) {
            const int tok = i / kChunks;
            const int ch = (i % kChunks) * 16;
            u8* dst = sX + tok * kPad + ch;
            if (tok < count) {
                __pipeline_memcpy_async(
                    dst, xq + static_cast<long long>(act_row(tok)) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // The activation rows of this lane's two columns in each 8-route block
    // (C's columns 2c and 2c + 1), -1 past the slice.
    int xrow[kNbW][2];
#pragma unroll
    for (int b = 0; b < kNbW; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tok = 8 * (b + wb * kNbW) + 2 * c + h;
            xrow[b][h] = tok < count ? act_row(tok) : -1;
        }

    float acc[kTiles][kNbW][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNbW; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (s_lo + i < s_hi) load(s_lo + i, i);
        __pipeline_commit();
    }
    for (int s = s_lo; s < s_hi; ++s) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        const int next = s + kStages - 1;
        if (next < s_hi) load(next, (next - s_lo) % kStages);
        __pipeline_commit();

        const u8* sW = smem + ((s - s_lo) % kStages) * kStage;
        const u8* sX = sW + kRowsCta * kPad;
        float blk[kTiles][kNbW][4];
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNbW; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) blk[t][b][e] = 0.f;
        // The bf16 ldmatrix loads read e4m3 fragments: a 16-byte row of an
        // 8x8 b16 matrix is a 16-code row of K, 4 codes a lane, the order
        // m16n8k32's A and B fragments take.
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 32) {
            unsigned bk[kNbW][2];
#pragma unroll
            for (int b = 0; b < kNbW; ++b) {
                ld_b(bk[b], reinterpret_cast<const bf16*>(sX + 8 * (b + wb * kNbW) * kPad + kk), kPad / 2, lane);
            }
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                unsigned a[4];
                ld_a(a, reinterpret_cast<const bf16*>(sW + (warp * kRowsWarp + 16 * t) * kPad + kk),
                     kPad / 2, lane);
#pragma unroll
                for (int b = 0; b < kNbW; ++b) {
                    gfp8::mma(blk[t][b], a[0], a[1], a[2], a[3], bk[b][0], bk[b][1]);
                }
            }
        }
        // The CTA's 128 rows are one block's: one weight exponent a step.
        const float sw = gfp8::exp2_biased(atoms_at[(s / 4) * 512 + s % 4]);
#pragma unroll
        for (int b = 0; b < kNbW; ++b) {
            const float s0 = xrow[b][0] >= 0 ? sw * gfp8::exp2_biased(xe[xrow[b][0] * blocks + s]) : 0.f;
            const float s1 = xrow[b][1] >= 0 ? sw * gfp8::exp2_biased(xe[xrow[b][1] * blocks + s]) : 0.f;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                acc[t][b][0] = fmaf(blk[t][b][0], s0, acc[t][b][0]);
                acc[t][b][1] = fmaf(blk[t][b][1], s1, acc[t][b][1]);
                acc[t][b][2] = fmaf(blk[t][b][2], s0, acc[t][b][2]);
                acc[t][b][3] = fmaf(blk[t][b][3], s1, acc[t][b][3]);
            }
        }
    }

    if constexpr (kDense) {
        // Rows in order: output row `begin + col`.
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNbW; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int tok = 8 * (b + wb * kNbW) + 2 * c + (e & 1);
                    const int row = row_base + 16 * t + g + (e >= 2 ? 8 : 0);
                    if (tok < count && row < n) {
                        const long long at = static_cast<long long>(begin + tok) * n + row;
                        if (gridDim.z > 1) {
                            part[blockIdx.z * static_cast<long long>(rows) * n + at] = acc[t][b][e];
                        } else {
                            out[at] = f32_to_bf16(acc[t][b][e]);
                        }
                    }
                }
    } else {
        auto factor = [&](int) { return 1.f; };
        w4mma::w4_land<0, kTiles, kNbW>(acc, factor, nullptr, order, begin + 8 * kNbW * wb, count - 8 * kNbW * wb, out, n, n,
                                       row_base, g, c, 0.f, 1.f);
    }
}

/// A split `moe_select_fp8a8_mma` dense fire's `splits` f32 partials
/// (`[splits][count]`) summed in split order into bf16 `out`. Four a thread.
__global__ void __launch_bounds__(256) fp8a8_fold(
    const float* __restrict__ part, bf16* __restrict__ out, long long count, int splits)
{
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (at >= count) return;
    float4 sum = *reinterpret_cast<const float4*>(part + at);
    for (int z = 1; z < splits; ++z) {
        const float4 v = *reinterpret_cast<const float4*>(part + z * count + at);
        sum.x += v.x;
        sum.y += v.y;
        sum.z += v.z;
        sum.w += v.w;
    }
    uint2 packed;
    packed.x = pack_bf16x2(sum.x, sum.y);
    packed.y = pack_bf16x2(sum.z, sum.w);
    *reinterpret_cast<uint2*>(out + at) = packed;
}

__device__ __forceinline__ unsigned short pie_ld_line_u16(const unsigned short* p) {
    unsigned short v;
    asm volatile("ld.global.nc.L2::128B.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

__global__ void moe_matmul_select_mxfp4_wmma(
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
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    namespace wmma = ::nvcuda::wmma;
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kBatches = 2;
    constexpr int kRoutes = kBatch * kBatches;
    constexpr int kLd = kTileK + 8;
    constexpr int kWarps = kTileN / 16;

    constexpr int kPerWord = 8;
    constexpr int kGroup = 32;
    static_assert(kRoutes * kLd * 2 >= kWarps * 16 * 16 * 4, "the C tiles fit where the activations were");

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item / 65536;
    const int slice = item % 65536;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert] + slice * kRoutes;
    const int end = min(offsets[expert + 1], begin + kRoutes);
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int groups_per_row = k / kGroup;
    const int words_per_row = k / kPerWord;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
    }
    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * (k / 2));
    const u8* s8 = scales_at + static_cast<long long>(expert) * n * groups_per_row;

    const bf16* b_row =
        bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;

    __shared__ __align__(32) bf16 w_tile[kTileN][kLd];
    __shared__ __align__(32) bf16 x_tile[kRoutes][kLd];
    float* c_tile = reinterpret_cast<float*>(&x_tile[0][0]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const bf16 zero = f32_to_bf16(0.f);

    constexpr int kQuadWords = 4;
    constexpr int kQuadCodes = kQuadWords * kPerWord;
    constexpr int kQuadsPerTileK = kTileK / kQuadCodes;
    constexpr int kQuadsPerThread = kTileN * kQuadsPerTileK / 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerTileK = kTileK / kVec;
    constexpr int kVecsPerThread = kRoutes * kVecsPerTileK / 256;
    static_assert(kGroup % kQuadCodes == 0, "a quad lies within one group");
    static_assert(kTileN * kQuadsPerTileK % 256 == 0 && kRoutes * kVecsPerTileK % 256 == 0, "even split");

    struct Staged {
        uint4 quad[kQuadsPerThread];
        unsigned char scale_byte[kQuadsPerThread];
        bool quad_live[kQuadsPerThread];
        uint4 vec[kVecsPerThread];
    };

    {
        const int g0 = begin;
        const int group = end - g0;
        const int batches = (group + kBatch - 1) / kBatch;

        float acc[kBatches][2][4];
#pragma unroll
        for (int b = 0; b < kBatches; ++b)
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[b][t][e] = 0.f;

        const bf16* vec_src[kVecsPerThread];
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int t = (tid + i * 256) / kVecsPerTileK;
            vec_src[i] = nullptr;
            if (t < group) {
                const int plane_route = order[g0 + t];
                vec_src[i] = act + static_cast<long long>(plane_route / act_div) * k;
            }
        }

        auto fetch = [&](int k0, Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                const int kk = k0 + q * kQuadCodes;
                const int rr = row0 + r;
                st.quad_live[i] = kk + kQuadCodes <= k && rr < n;
                if (st.quad_live[i]) {
                    const long long fx = static_cast<long long>(rr) * groups_per_row + kk / kGroup;
                    st.scale_byte[i] = s8[fx];
                    st.quad[i] = pie_ld_line(reinterpret_cast<const uint4*>(
                        w32 + static_cast<long long>(rr) * words_per_row + kk / kPerWord));
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int v = (tid + i * 256) % kVecsPerTileK;
                const int kk = k0 + v * kVec;
                st.vec[i] = make_uint4(0u, 0u, 0u, 0u);
                if (vec_src[i] != nullptr && kk + kVec <= k) {
                    st.vec[i] = *reinterpret_cast<const uint4*>(vec_src[i] + kk);
                }
            }
        };

        auto store = [&](int k0, const Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                unsigned* dst = reinterpret_cast<unsigned*>(&w_tile[r][q * kQuadCodes]);
                if (st.quad_live[i]) {
                    const float sv = mxfp4_block_scale(st.scale_byte[i]);
                    const unsigned words[kQuadWords] = {st.quad[i].x, st.quad[i].y, st.quad[i].z, st.quad[i].w};
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
                        __half2 qd[4];
                        mxfp4_unpack8(words[w], qd);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f = __half22float2(qd[j]);
                            dst[w * 4 + j] = pack_bf16x2(f.x * sv, f.y * sv);
                        }
                    }
                } else {

                    const int kk = k0 + q * kQuadCodes;
                    const int rr = row0 + r;
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
                        const int kw = kk + w * kPerWord;
                        if (kw < k && rr < n) {
                            const long long fx = static_cast<long long>(rr) * groups_per_row + kw / kGroup;
                            const float sv = mxfp4_block_scale(s8[fx]);
                            __half2 qd[4];
                            mxfp4_unpack8(
                                w32[static_cast<long long>(rr) * words_per_row + kw / kPerWord], qd);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float2 f = __half22float2(qd[j]);
                                w_tile[r][q * kQuadCodes + w * kPerWord + 2 * j] = f32_to_bf16(f.x * sv);
                                w_tile[r][q * kQuadCodes + w * kPerWord + 2 * j + 1] = f32_to_bf16(f.y * sv);
                            }
                        } else {
#pragma unroll
                            for (int j = 0; j < kPerWord; ++j)
                                w_tile[r][q * kQuadCodes + w * kPerWord + j] = zero;
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int t = idx / kVecsPerTileK;
                const int v = idx % kVecsPerTileK;
                *reinterpret_cast<uint4*>(&x_tile[t][v * kVec]) = st.vec[i];
            }

            const int tail = k - k0;
            if (tail < kTileK && (tail % kVec) != 0) {
                const int from = (tail / kVec) * kVec;
                for (int idx = tid; idx < kRoutes * (kTileK - from); idx += blockDim.x) {
                    const int t = idx / (kTileK - from);
                    const int kk = from + idx % (kTileK - from);
                    bf16 vv = zero;
                    if (t < group && k0 + kk < k) {
                        const int plane_route = order[g0 + t];
                        vv = act[static_cast<long long>(plane_route / act_div) * k + k0 + kk];
                    }
                    x_tile[t][kk] = vv;
                }
            }
        };

        Staged staged;
        fetch(0, staged);
        for (int k0 = 0; k0 < k; k0 += kTileK) {
            store(k0, staged);
            __syncthreads();
            if (k0 + kTileK < k) fetch(k0 + kTileK, staged);
#pragma unroll
            for (int kk = 0; kk < kTileK; kk += 16) {

                unsigned bfrag[4];
                ldmatrix_x4(bfrag, &w_tile[warp * 16 + (lane & 7) + ((lane >> 4) << 3)][kk + (((lane >> 3) & 1) << 3)]);

#pragma unroll
                for (int bt = 0; bt < kBatches; ++bt) {

                    unsigned afrag[4];
                    ldmatrix_x4(afrag, &x_tile[bt * kBatch + (lane & 15)][kk + ((lane >> 4) << 3)]);
                    mma_m16n8k16(acc[bt][0], afrag, {bfrag[0], bfrag[1]});
                    mma_m16n8k16(acc[bt][1], afrag, {bfrag[2], bfrag[3]});
                }
            }
            __syncthreads();
        }

        for (int bt = 0; bt < batches; ++bt) {

            {
                float* mine = c_tile + warp * 16 * 16;
                const int m = lane >> 2;
                const int col = (lane & 3) << 1;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    mine[m * 16 + t * 8 + col] = acc[bt][t][0];
                    mine[m * 16 + t * 8 + col + 1] = acc[bt][t][1];
                    mine[(m + 8) * 16 + t * 8 + col] = acc[bt][t][2];
                    mine[(m + 8) * 16 + t * 8 + col + 1] = acc[bt][t][3];
                }
            }
            __syncthreads();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int m = idx / 16;
                const int nn = idx % 16;
                const int t = bt * kBatch + m;
                const int row = row0 + warp * 16 + nn;
                if (t < group && row < n) {
                    const int plane_route = order[g0 + t];
                    float v = c_tile[warp * 16 * 16 + idx];
                    if (b_row != nullptr) v += Elem<bf16>::to_f32(b_row[row]);
                    out[static_cast<long long>(plane_route) * n + row] = f32_to_bf16(v);
                }
            }
            __syncthreads();
        }
    }
}

template <int kBits, int kGroup>
__global__ void moe_matmul_select_mlxu4_wmma(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits)
{
    namespace wmma = ::nvcuda::wmma;
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kBatch = 16;
    constexpr int kBatches = 2;
    constexpr int kRoutes = kBatch * kBatches;
    constexpr int kLd = kTileK + 8;
    constexpr int kWarps = kTileN / 16;
    constexpr int kPerWord = 32 / kBits;
    constexpr unsigned kMask = (1u << kBits) - 1u;
    static_assert(kRoutes * kLd * 2 >= kWarps * 16 * 16 * 4, "the C tiles fit where the activations were");

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item / 65536;
    const int slice = item % 65536;
    const int row0 = blockIdx.y * kTileN;
    if (row0 >= n) return;
    const int begin = offsets[expert] + slice * kRoutes;
    const int end = min(offsets[expert + 1], begin + kRoutes);
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));

    const int groups_per_row = k / kGroup;
    const int words_per_row = k / kPerWord;
    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }
    const unsigned* w32 = reinterpret_cast<const unsigned*>(
        codes_at + static_cast<long long>(expert) * n * words_per_row * 4);
    const bf16* s16 = reinterpret_cast<const bf16*>(
        scales_at + static_cast<long long>(expert) * n * groups_per_row * 2);
    const bf16* b16 = reinterpret_cast<const bf16*>(
        biases_at + static_cast<long long>(expert) * n * groups_per_row * 2);

    __shared__ __align__(32) bf16 w_tile[kTileN][kLd];
    __shared__ __align__(32) bf16 x_tile[kRoutes][kLd];
    float* c_tile = reinterpret_cast<float*>(&x_tile[0][0]);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const bf16 zero = f32_to_bf16(0.f);

    constexpr int kQuadWords = 4;
    constexpr int kQuadCodes = kQuadWords * kPerWord;
    constexpr int kQuadsPerTileK = kTileK / kQuadCodes;
    constexpr int kQuadsPerThread = kTileN * kQuadsPerTileK / 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerTileK = kTileK / kVec;
    constexpr int kVecsPerThread = kRoutes * kVecsPerTileK / 256;
    static_assert(kGroup % kQuadCodes == 0, "a quad lies within one group");
    static_assert(kTileN * kQuadsPerTileK % 256 == 0 && kRoutes * kVecsPerTileK % 256 == 0, "even split");

    struct Staged {
        uint4 quad[kQuadsPerThread];
        unsigned short scale_bits[kQuadsPerThread];
        unsigned short zero_bits[kQuadsPerThread];
        bool quad_live[kQuadsPerThread];
        uint4 vec[kVecsPerThread];
    };
    const unsigned short* s_bits = reinterpret_cast<const unsigned short*>(s16);
    const unsigned short* b_bits = reinterpret_cast<const unsigned short*>(b16);

    {
        const int g0 = begin;
        const int group = end - g0;
        const int batches = (group + kBatch - 1) / kBatch;

        float acc[kBatches][2][4];
#pragma unroll
        for (int b = 0; b < kBatches; ++b)
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[b][t][e] = 0.f;

        const bf16* vec_src[kVecsPerThread];
#pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int t = (tid + i * 256) / kVecsPerTileK;
            vec_src[i] = nullptr;
            if (t < group) {
                const int plane_route = order[g0 + t];
                vec_src[i] = act + static_cast<long long>(plane_route / act_div) * k;
            }
        }

        auto fetch = [&](int k0, Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                const int kk = k0 + q * kQuadCodes;
                const int rr = row0 + r;
                st.quad_live[i] = kk + kQuadCodes <= k && rr < n;
                if (st.quad_live[i]) {
                    const long long fx = static_cast<long long>(rr) * groups_per_row + kk / kGroup;
                    st.scale_bits[i] = pie_ld_line_u16(s_bits + fx);
                    st.zero_bits[i] = pie_ld_line_u16(b_bits + fx);
                    st.quad[i] = pie_ld_line(reinterpret_cast<const uint4*>(
                        w32 + static_cast<long long>(rr) * words_per_row + kk / kPerWord));
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int v = (tid + i * 256) % kVecsPerTileK;
                const int kk = k0 + v * kVec;
                st.vec[i] = make_uint4(0u, 0u, 0u, 0u);
                if (vec_src[i] != nullptr && kk + kVec <= k) {
                    st.vec[i] = *reinterpret_cast<const uint4*>(vec_src[i] + kk);
                }
            }
        };

        auto store = [&](int k0, const Staged& st) {
#pragma unroll
            for (int i = 0; i < kQuadsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int r = idx / kQuadsPerTileK;
                const int q = idx % kQuadsPerTileK;
                unsigned* dst = reinterpret_cast<unsigned*>(&w_tile[r][q * kQuadCodes]);
                if (st.quad_live[i]) {
                    const float sv = __uint_as_float(static_cast<unsigned>(st.scale_bits[i]) << 16);
                    const float zv = __uint_as_float(static_cast<unsigned>(st.zero_bits[i]) << 16);
                    const unsigned words[kQuadWords] = {st.quad[i].x, st.quad[i].y, st.quad[i].z, st.quad[i].w};
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
#pragma unroll
                        for (int j = 0; j < kPerWord; j += 2) {
                            const float c0 = static_cast<float>((words[w] >> (kBits * j)) & kMask);
                            const float c1 = static_cast<float>((words[w] >> (kBits * (j + 1))) & kMask);
                            dst[(w * kPerWord + j) / 2] = pack_bf16x2(fmaf(c0, sv, zv), fmaf(c1, sv, zv));
                        }
                    }
                } else {

                    const int kk = k0 + q * kQuadCodes;
                    const int rr = row0 + r;
#pragma unroll
                    for (int w = 0; w < kQuadWords; ++w) {
                        const int kw = kk + w * kPerWord;
                        if (kw < k && rr < n) {
                            const long long fx = static_cast<long long>(rr) * groups_per_row + kw / kGroup;
                            const float sv = Elem<bf16>::to_f32(s16[fx]);
                            const float bv = Elem<bf16>::to_f32(b16[fx]);
                            const unsigned word =
                                w32[static_cast<long long>(rr) * words_per_row + kw / kPerWord];
#pragma unroll
                            for (int j = 0; j < kPerWord; ++j) {
                                const float code = static_cast<float>((word >> (kBits * j)) & kMask);
                                w_tile[r][q * kQuadCodes + w * kPerWord + j] = f32_to_bf16(fmaf(code, sv, bv));
                            }
                        } else {
#pragma unroll
                            for (int j = 0; j < kPerWord; ++j)
                                w_tile[r][q * kQuadCodes + w * kPerWord + j] = zero;
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
                const int idx = tid + i * 256;
                const int t = idx / kVecsPerTileK;
                const int v = idx % kVecsPerTileK;
                *reinterpret_cast<uint4*>(&x_tile[t][v * kVec]) = st.vec[i];
            }

            const int tail = k - k0;
            if (tail < kTileK && (tail % kVec) != 0) {
                const int from = (tail / kVec) * kVec;
                for (int idx = tid; idx < kRoutes * (kTileK - from); idx += blockDim.x) {
                    const int t = idx / (kTileK - from);
                    const int kk = from + idx % (kTileK - from);
                    bf16 vv = zero;
                    if (t < group && k0 + kk < k) {
                        const int plane_route = order[g0 + t];
                        vv = act[static_cast<long long>(plane_route / act_div) * k + k0 + kk];
                    }
                    x_tile[t][kk] = vv;
                }
            }
        };

        Staged staged;
        fetch(0, staged);
        for (int k0 = 0; k0 < k; k0 += kTileK) {
            store(k0, staged);
            __syncthreads();
            if (k0 + kTileK < k) fetch(k0 + kTileK, staged);
#pragma unroll
            for (int kk = 0; kk < kTileK; kk += 16) {

                unsigned bfrag[4];
                ldmatrix_x4(bfrag, &w_tile[warp * 16 + (lane & 7) + ((lane >> 4) << 3)][kk + (((lane >> 3) & 1) << 3)]);

#pragma unroll
                for (int bt = 0; bt < kBatches; ++bt) {

                    unsigned afrag[4];
                    ldmatrix_x4(afrag, &x_tile[bt * kBatch + (lane & 15)][kk + ((lane >> 4) << 3)]);
                    mma_m16n8k16(acc[bt][0], afrag, {bfrag[0], bfrag[1]});
                    mma_m16n8k16(acc[bt][1], afrag, {bfrag[2], bfrag[3]});
                }
            }
            __syncthreads();
        }

        for (int bt = 0; bt < batches; ++bt) {

            {
                float* mine = c_tile + warp * 16 * 16;
                const int m = lane >> 2;
                const int col = (lane & 3) << 1;
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    mine[m * 16 + t * 8 + col] = acc[bt][t][0];
                    mine[m * 16 + t * 8 + col + 1] = acc[bt][t][1];
                    mine[(m + 8) * 16 + t * 8 + col] = acc[bt][t][2];
                    mine[(m + 8) * 16 + t * 8 + col + 1] = acc[bt][t][3];
                }
            }
            __syncthreads();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int m = idx / 16;
                const int nn = idx % 16;
                const int t = bt * kBatch + m;
                const int row = row0 + warp * 16 + nn;
                if (t < group && row < n) {
                    const int plane_route = order[g0 + t];
                    out[static_cast<long long>(plane_route) * n + row] =
                        f32_to_bf16(c_tile[warp * 16 * 16 + idx]);
                }
            }
            __syncthreads();
        }
    }
}

}
