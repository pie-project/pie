#pragma once

// Block-FP8 projections against e4m3 activations (W8A8) on mma.sync where
// there is an FP8 MMA (sm_89 on) but no tcgen05: the rows quantized in 1x128
// blocks once a fire (`quantize_rows_e4m3b128`, as the tcgen05 path does),
// e4m3 x e4m3 m16n8k32 over a cp.async ring that `moe_select_bf16_mma`'s
// staging shares, each 128-wide block's sums scaled by its two exponents.
// A routed select, and a dense plane's rows in order with K split where its
// row blocks leave the multiprocessors idle.

#include "linear/moe_w4.cuh"
#include "linear/grouped_fp8.cuh"

namespace pie::linear {

/// Rows `[rows, k]` of bf16 as e4m3 in 1x128 blocks, as the tcgen05 path
/// quantizes them (`E4m3Block`): codes row-major into `codes`, each block's
/// biased exponent a byte into `exps` (`[rows, k / 128]`). 16 lanes a
/// block; 128 threads, `ceil(rows k / 1024)` blocks.
__global__ void __launch_bounds__(128) quantize_rows_e4m3b128(
    const bf16* __restrict__ x, u8* __restrict__ codes, u8* __restrict__ exps, int rows, int k)
{
    using Block = E4m3Block<128>;
    const int blocks = k / 128;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    // Every lane joins the block's shuffles; a slot past the rows reads row 0.
    const bool live = slot < rows * blocks;
    const int sl = live ? slot : 0;
    const long long at = static_cast<long long>(sl / blocks) * k + (sl % blocks) * 128;
    Block q;
    q.load(x + at, lane);
    if (!live) return;
    *reinterpret_cast<unsigned long long*>(codes + at + (lane % Block::kLanes) * 8) = q.codes();
    if (lane % Block::kLanes == 0) exps[sl] = static_cast<u8>(q.e + 127);
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
template <int kNb, bool kDense>
__global__ void __launch_bounds__(128) moe_select_fp8a8_mma(
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
    const int warp = tid >> 5;
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
        for (int i = tid; i < kRowsCta * kChunks; i += 128) {
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
        for (int i = tid; i < kSlice * kChunks; i += 128) {
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
    int xrow[kNb][2];
#pragma unroll
    for (int b = 0; b < kNb; ++b)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tok = 8 * b + 2 * c + h;
            xrow[b][h] = tok < count ? act_row(tok) : -1;
        }

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
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
        float blk[kTiles][kNb][4];
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) blk[t][b][e] = 0.f;
        // The bf16 ldmatrix loads read e4m3 fragments: a 16-byte row of an
        // 8x8 b16 matrix is a 16-code row of K, 4 codes a lane, the order
        // m16n8k32's A and B fragments take.
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 32) {
            unsigned bk[kNb][2];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                bf16mma::ld_b(bk[b], reinterpret_cast<const bf16*>(sX + 8 * b * kPad + kk),
                              kPad / 2, lane);
            }
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                unsigned a[4];
                bf16mma::ld_a(a,
                              reinterpret_cast<const bf16*>(sW + (warp * kRowsWarp + 16 * t) * kPad + kk),
                              kPad / 2, lane);
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    gfp8::mma(blk[t][b], a[0], a[1], a[2], a[3], bk[b][0], bk[b][1]);
                }
            }
        }
        // The CTA's 128 rows are one block's: one weight exponent a step.
        const float sw = gfp8::exp2_biased(atoms_at[(s / 4) * 512 + s % 4]);
#pragma unroll
        for (int b = 0; b < kNb; ++b) {
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
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int tok = 8 * b + 2 * c + (e & 1);
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
        w4mma::w4_land<0, kTiles, kNb>(acc, factor, nullptr, order, begin, count, out, n, n,
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

/// Rows `[rows, k]` of bf16 as e4m3 in 1x128 blocks for cuBLASLt's
/// block-scaled FP8 GEMM: `quantize_rows_e4m3b128`'s codes (rows past `rows`
/// up to `mpad` zero), each block's factor an f32 at `scales[kb * mpad + r]`
/// (cuBLASLt's `VEC128_32F` B scales). 16 lanes a block, `mpad k / 1024`
/// blocks of 128 threads.
__global__ void __launch_bounds__(128) quantize_rows_e4m3b128_lt(
    const bf16* __restrict__ x, u8* __restrict__ codes, float* __restrict__ scales, int rows,
    int mpad, int k)
{
    using Block = E4m3Block<128>;
    const int blocks = k / 128;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    // Every lane joins the block's shuffles; a slot past the rows reads row 0.
    const bool live = slot < mpad * blocks;
    const int sl = live ? slot : 0;
    const int r = sl / blocks, kb = sl % blocks;
    const long long at = static_cast<long long>(r) * k + kb * 128;
    Block q;
    q.load(x + (r < rows ? at : kb * 128), lane);
    if (!live) return;
    u8* dst = codes + at + (lane % Block::kLanes) * 8;
    if (r < rows) {
        *reinterpret_cast<unsigned long long*>(dst) = q.codes();
        if (lane % Block::kLanes == 0) scales[kb * mpad + r] = gfp8::exp2_biased(q.e + 127);
    } else {
        *reinterpret_cast<unsigned long long*>(dst) = 0ull;
        if (lane % Block::kLanes == 0) scales[kb * mpad + r] = 0.f;
    }
}

/// A 128x128-block FP8 plane's exponents (in tcgen05's atoms, a weight row an
/// atom row) as cuBLASLt's `BLK128x128_32F` A scales: an f32 a block at
/// `out[nb * kbp + kb]`, `kbp` the K blocks rounded up to four (the padding
/// zero). One thread a slot of `n / 128` x `kbp`.
__global__ void __launch_bounds__(256) fp8b128_atoms_to_lt(
    const u8* __restrict__ atoms, float* __restrict__ out, int n, int k)
{
    const int blocks = k / 128;
    const int kbp = (blocks + 3) / 4 * 4;
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= (n / 128) * kbp) return;
    const int nb = i / kbp, kb = i % kbp;
    out[i] = kb < blocks ? gfp8::exp2_biased(atoms[atom_offset(nb * 128, kb, blocks)]) : 0.f;
}

}  // namespace pie::linear
