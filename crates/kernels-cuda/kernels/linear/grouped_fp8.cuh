#pragma once

// DeepSeek-V4's grouped output projection (`wo_a`) at decode widths on fp8
// planes, as vLLM runs it: each group's plane `[n, k]` of e4m3 codes under
// one ue8m0 exponent a 128x128 block (the checkpoint's own), against the
// group's slice of each row quantized to e4m3 in 1x128 blocks (DeepSeek's
// `act_quant`), e4m3 x e4m3 on mma.sync with f32 sums, each 128-wide
// block's sum scaled by its two exponents.
//
// A CTA takes 16-row tiles of a plane, `kSplit` warps a tile and a share
// of K a warp; the group's slice of the activations (at most 8 rows, the
// MMA's columns) is quantized once a CTA into shared memory, while the
// warps' first codes stream in. The MMA's K order is free as long as both
// operands agree: lane (g, c) loads 16 contiguous codes of rows g and g + 8
// at byte 16 c of a 64-wide step, and the activations' fragment reads the
// same 16 bytes of its row, so every weight load is one 16-byte stream.

#include "linear/act_quant.cuh"

namespace pie::linear {

namespace gfp8 {

using ::pie::ld_stream;

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// Two e4m3 codes (the low byte the low half) as floats, exactly.
__device__ __forceinline__ float2 e4m3x2_f32(unsigned short pair) {
    float lo, hi;
    asm("{ .reg .b32 h; .reg .b16 l, u;\n"
        "cvt.rn.f16x2.e4m3x2 h, %2;\n"
        "mov.b32 {l, u}, h;\n"
        "cvt.f32.f16 %0, l;\n"
        "cvt.f32.f16 %1, u; }\n"
        : "=f"(lo), "=f"(hi)
        : "h"(pair));
    float2 r;
    r.x = lo;
    r.y = hi;
    return r;
}

/// 2^(e - 127) for a biased exponent `e` in 1 .. 254.
__device__ __forceinline__ float exp2_biased(int e) {
    return __uint_as_float(static_cast<unsigned>(e) << 23);
}

}  // namespace gfp8

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

/// A bf16 plane `[rows, k]` (whole 128x128 blocks) as fp8b128: each block's
/// ue8m0 exponent `e` the least with `amax <= 448 * 2^e` (DeepSeek's rule,
/// found exactly rather than through a rounded quotient), spread to each
/// of its rows in tcgen05's atoms (`atoms`, zeroed by the caller), and each
/// value's code `e4m3(v * 2^-e)`. A plane whose values are already some
/// fp8b128 plane's (DeepSeek-V4's `wo_a`, dequantized at import) lands
/// codes of the same values: `e` is that plane's exponent or below it, and
/// each code its code times `2^(s - e)`, which e4m3 holds. Grid `[k / 128,
/// rows / 128]`, 256 threads.
__global__ void __launch_bounds__(256) quantize_fp8b128(const bf16* __restrict__ src,
                                                        u8* __restrict__ codes,
                                                        u8* __restrict__ atoms, int rows,
                                                        int k) {
    __shared__ float part[8];
    __shared__ int block_e;
    const int kb = blockIdx.x;
    const int rb = blockIdx.y;
    const int blocks = k / 128;
    const int tid = static_cast<int>(threadIdx.x);
    // A thread's 64 values: half of one row of the block.
    const int r = rb * 128 + tid / 2;
    const long long at = static_cast<long long>(r) * k + kb * 128 + (tid % 2) * 64;
    float v[64];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const uint4 w = *reinterpret_cast<const uint4*>(src + at + q * 8);
        const unsigned ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[q * 8 + 2 * i] = __uint_as_float(ws[i] << 16);
            v[q * 8 + 2 * i + 1] = __uint_as_float(ws[i] & 0xFFFF0000u);
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    if ((tid & 31) == 0) part[tid >> 5] = amax;
    __syncthreads();
    if (tid == 0) {
        float a = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) a = fmaxf(a, part[w]);
        // 448 * 2^e, exact for e in -126 .. 119.
        auto bound = [](int e) { return 448.f * __uint_as_float(static_cast<unsigned>(e + 127) << 23); };
        int e = -126;
        if (a > 0.f) {
            const unsigned bits = __float_as_uint(a / 448.f);
            e = static_cast<int>((bits >> 23) & 0xFFu) - 127;
            e = max(min(e, 119), -126);
            while (e < 119 && a > bound(e)) ++e;
            while (e > -126 && a <= bound(e - 1)) --e;
        }
        block_e = e;
    }
    __syncthreads();
    const int e = block_e;
    const float inv = __uint_as_float(static_cast<unsigned>(127 - e) << 23);
    unsigned packed[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        unsigned short lo, hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(v[4 * i + 1] * inv), "f"(v[4 * i] * inv));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(v[4 * i + 3] * inv), "f"(v[4 * i + 2] * inv));
        packed[i] = static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
        *reinterpret_cast<uint4*>(codes + at + q * 16) =
            make_uint4(packed[4 * q], packed[4 * q + 1], packed[4 * q + 2], packed[4 * q + 3]);
    if (tid % 2 == 0) atoms[atom_offset(r, kb, blocks)] = static_cast<u8>(e + 127);
}

/// An fp8b128 plane (codes `[rows, k]`, each 128x128 block's exponent in
/// tcgen05's atoms) as bf16, each value exactly: what `quantize_fp8b128`
/// read. 16 codes a thread; `k` whole 128s.
__global__ void __launch_bounds__(256) dequantize_fp8b128(const u8* __restrict__ codes,
                                                          const u8* __restrict__ atoms,
                                                          bf16* __restrict__ out, int rows,
                                                          int k) {
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (at >= static_cast<long long>(rows) * k) return;
    const int r = static_cast<int>(at / k);
    const int c = static_cast<int>(at % k);
    const float s = __uint_as_float(static_cast<unsigned>(atoms[atom_offset(r, c / 128, k / 128)]) << 23);
    const uint4 q = *reinterpret_cast<const uint4*>(codes + at);
    const unsigned qs[4] = {q.x, q.y, q.z, q.w};
    unsigned w[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 lo = gfp8::e4m3x2_f32(static_cast<unsigned short>(qs[i] & 0xFFFFu));
        const float2 hi = gfp8::e4m3x2_f32(static_cast<unsigned short>(qs[i] >> 16));
        w[2 * i] = pack_bf16x2(lo.x * s, lo.y * s);
        w[2 * i + 1] = pack_bf16x2(hi.x * s, hi.y * s);
    }
    *reinterpret_cast<uint4*>(out + at) = make_uint4(w[0], w[1], w[2], w[3]);
    *reinterpret_cast<uint4*>(out + at + 8) = make_uint4(w[4], w[5], w[6], w[7]);
}

}  // namespace pie::linear
