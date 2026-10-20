#pragma once

// A 128x128-block FP8 projection at a decode's few rows as a GEMV:
// `y[m] = x[m] . W^T`, W's e4m3 codes under one ue8m0 exponent a row a 128
// columns (the tcgen05 GEMM's atoms, read where they lie), the activations
// as they are (bf16, not quantized: the W8A16 product of the same codes).
// A warp `kRows` weight rows; a lane sixteen codes (one 16-byte load) of
// each a pass, the warp's lanes 512 adjacent codes; `kM` activation rows
// read through L1 beside them. `k` a whole number of 512s.

#include "linear/act_quant.cuh"

namespace pie::linear {

namespace fp8gemv {

using ::pie::ld_stream;

// The power of two a code's f32 falls short of its value by: none where the
// e4m3 convert is (sm_89); below it the codes are laid into bf16 by integer
// ops (an e4m3 exponent field in bf16's, bias 127 for 7: 2^-120), the
// activations taken 2^100 up (kXScale) to keep the products normal, and the
// block's exponent taking back the 2^20 left.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
constexpr unsigned kCodeBias = 20;
constexpr float kXScale = 0x1p100f;
#else
constexpr unsigned kCodeBias = 0;
constexpr float kXScale = 1.f;
#endif

// Four e4m3 codes of a word as f32: two codes a pair.
__device__ __forceinline__ void codes4(unsigned w, float (&out)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    // A code's sign replicated through its half's high byte, then shifted
    // to bf16's places: sign at 15, exponent field low four bits, mantissa
    // the top three; an e4m3 subnormal an f32 subnormal, exact. A bf16's
    // f32 is a shift, not a convert.
    unsigned lo, hi;
    asm("prmt.b32 %0, %1, 0, 0x9180;" : "=r"(lo) : "r"(w));
    asm("prmt.b32 %0, %1, 0, 0xB3A2;" : "=r"(hi) : "r"(w));
    lo = (lo << 4) & 0x87F087F0u;
    hi = (hi << 4) & 0x87F087F0u;
    out[0] = __uint_as_float(lo << 16);
    out[1] = __uint_as_float(lo & 0xFFFF0000u);
    out[2] = __uint_as_float(hi << 16);
    out[3] = __uint_as_float(hi & 0xFFFF0000u);
#else
    unsigned lo, hi;
    asm("{ .reg .b16 a, b;\n"
        "mov.b32 {a, b}, %2;\n"
        "cvt.rn.f16x2.e4m3x2 %0, a;\n"
        "cvt.rn.f16x2.e4m3x2 %1, b; }\n"
        : "=r"(lo), "=r"(hi)
        : "r"(w));
    float f0, f1, f2, f3;
    asm("{ .reg .f16 l, u;\n mov.b32 {l, u}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, u; }\n"
        : "=f"(f0), "=f"(f1) : "r"(lo));
    asm("{ .reg .f16 l, u;\n mov.b32 {l, u}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, u; }\n"
        : "=f"(f2), "=f"(f3) : "r"(hi));
    out[0] = f0;
    out[1] = f1;
    out[2] = f2;
    out[3] = f3;
#endif
}

}  // namespace fp8gemv

template <int kWarps, int kRows, int kM>
__global__ void __launch_bounds__(kWarps * 32) fp8_block_gemv(
    const bf16* __restrict__ x,
    int x_stride,
    const u8* __restrict__ codes,
    const u8* __restrict__ atoms,
    bf16* __restrict__ y,
    int y_stride,
    int m,
    int n,
    int k,
    int splits)
{
    using namespace fp8gemv;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // A block `kRows` weight rows, its warps each a `kWarps`-th of K: a
    // projection of few rows over a long K fills the SMs, and no warp
    // walks K alone.
    // `splits` warps share a row group's K (a power of two up to kWarps):
    // a long K over few rows spreads over the SMs, a short one keeps each
    // warp on rows of its own.
    const int group = warp / splits;
    const int part_k = warp % splits;
    const int row0 = (blockIdx.x * (kWarps / splits) + group) * kRows;
    const int blocks = k / 128;
    float acc[kRows][kM];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
#pragma unroll
        for (int j = 0; j < kM; ++j) acc[i][j] = 0.f;
    }
    for (int c0 = part_k * 512 + lane * 16; c0 < k; c0 += 512 * splits) {
        uint4 w[kRows];
        float s[kRows];
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            const int row = min(row0 + i, n - 1);
            w[i] = ld_stream(codes + static_cast<long long>(row) * k + c0);
            s[i] = __uint_as_float((static_cast<unsigned>(atoms[atom_offset(row, c0 / 128, blocks)]) + kCodeBias) << 23);
        }
        // The activations' sixteen columns of each row, read through L1
        // (every warp of the block reads the same).
        uint4 xr[kM][2];
        // The pass's dot unscaled, its block's exponent applied once.
        float dot[kRows][kM];
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kM; ++j) dot[i][j] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < kM; ++j) {
            const bf16* xp = x + static_cast<long long>(min(j, m - 1)) * x_stride + c0;
            xr[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
            xr[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float xv[kM][4];
#pragma unroll
            for (int j = 0; j < kM; ++j) {
                const uint4 h = xr[j][q / 2];
                const unsigned a = (q & 1) ? h.z : h.x;
                const unsigned b = (q & 1) ? h.w : h.y;
                xv[j][0] = __uint_as_float(a << 16) * kXScale;
                xv[j][1] = __uint_as_float(a & 0xFFFF0000u) * kXScale;
                xv[j][2] = __uint_as_float(b << 16) * kXScale;
                xv[j][3] = __uint_as_float(b & 0xFFFF0000u) * kXScale;
            }
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                const unsigned word = q == 0 ? w[i].x : q == 1 ? w[i].y : q == 2 ? w[i].z : w[i].w;
                float cv[4];
                codes4(word, cv);
#pragma unroll
                for (int j = 0; j < kM; ++j) {
                    // Fused by hand (the JIT compiles under --fmad=false).
                    dot[i][j] = fmaf(cv[3], xv[j][3],
                                     fmaf(cv[2], xv[j][2], fmaf(cv[1], xv[j][1], fmaf(cv[0], xv[j][0], dot[i][j]))));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kM; ++j) acc[i][j] = fmaf(dot[i][j], s[i], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
#pragma unroll
        for (int j = 0; j < kM; ++j) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[i][j] += __shfl_xor_sync(0xffffffffu, acc[i][j], off);
        }
    }
    __shared__ float part[kWarps][kRows * kM];
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
            for (int j = 0; j < kM; ++j) part[warp][i * kM + j] = acc[i][j];
        }
    }
    __syncthreads();
    if (threadIdx.x < kWarps * kRows * kM / splits) {
        const int g = threadIdx.x / (kRows * kM);
        const int at = threadIdx.x % (kRows * kM);
        const int i = at / kM;
        const int j = at % kM;
        float sum = 0.f;
        for (int w = 0; w < splits; ++w) sum += part[g * splits + w][at];
        const int row = (blockIdx.x * (kWarps / splits) + g) * kRows + i;
        if (row < n && j < m) y[static_cast<long long>(j) * y_stride + row] = f32_to_bf16(sum);
    }
}

}  // namespace pie::linear
