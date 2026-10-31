#include "linear/act_quant.cuh"
#include "prim/device.cuh"
#include "prim/smem.cuh"
#include <cuda_pipeline.h>

namespace pie::linear {

/// `d += (a * 2^(sa - 127)) . (b * 2^(sb - 127))^T` over 32 e4m3 codes of K:
/// sm_120a's block-scaled mma.sync, one ue8m0 exponent a row of A (the lane's
/// `(lane >> 2) + 8 (lane & 1)`) and a column of B (the lane's `lane >> 2`).
__device__ __forceinline__ void mma_e4m3_scaled(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                                unsigned sa, unsigned sb) {
    const unsigned short zero = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "h"(zero), "h"(zero),
          "r"(sb), "h"(zero), "h"(zero));
}

/// `out = q(x) . (W 2^(s - 127))^T` for a 128x128-block FP8 plane (`n x k`
/// e4m3 codes, each block's exponent in tcgen05's atoms) against rows
/// quantized by `quantize_rows_e4m3b128` (e4m3 codes, an exponent a 1x128
/// block): W8A8 on sm_120a's block-scaled mma.sync, both exponents applied
/// in the MMA. A CTA is `kWM x kWN` warps over `16 kTM` rows by `8 kTN`
/// columns each, K in 128-code steps, `kStages` of them in swizzled shared memory.
template <int kWM, int kWN, int kTM, int kTN, int kStages>
__global__ void __launch_bounds__(32 * kWM * kWN) fp8_dense_scaled(
    const u8* __restrict__ xq, const u8* __restrict__ xe, const u8* __restrict__ weight,
    const u8* __restrict__ atoms, bf16* __restrict__ out, int m, int n, int k)
{
    constexpr int kCM = kWM * 16 * kTM;
    constexpr int kCN = kWN * 8 * kTN;
    constexpr int kBK = 128;
    constexpr int kThreads = 32 * kWM * kWN;
    constexpr int kStage = (kCM + kCN) * kBK;
    extern __shared__ uint4 fp8_dense_smem[];
    u8* smem = reinterpret_cast<u8*>(fp8_dense_smem);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp % kWM;
    const int wn = warp / kWM;
    const int m0 = static_cast<int>(blockIdx.y) * kCM;
    const int n0 = static_cast<int>(blockIdx.x) * kCN;
    const int blocks = k / kBK;

    auto load = [&](int step, int st) {
        u8* sa = smem + st * kStage;
        u8* sb = sa + kCM * kBK;
        const int k0 = step * kBK;
#pragma unroll
        for (int i = tid; i < kCM * 8; i += kThreads) {
            const int r = i >> 3;
            const int ch = (i & 7) << 4;
            u8* dst = sa + r * kBK + (ch ^ ((r & 7) << 4));
            if (m0 + r < m) {
                __pipeline_memcpy_async(dst, xq + static_cast<long long>(m0 + r) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = tid; i < kCN * 8; i += kThreads) {
            const int r = i >> 3;
            const int ch = (i & 7) << 4;
            u8* dst = sb + r * kBK + (ch ^ ((r & 7) << 4));
            if (n0 + r < n) {
                __pipeline_memcpy_async(dst, weight + static_cast<long long>(n0 + r) * k + k0 + ch, 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // The rows whose exponents this lane hands the MMA, a tile each.
    int srow[kTM];
#pragma unroll
    for (int t = 0; t < kTM; ++t)
        srow[t] = min(m0 + wm * 16 * kTM + 16 * t + (lane >> 2) + 8 * (lane & 1), m - 1);
    float acc[kTM][kTN][4];
#pragma unroll
    for (int t = 0; t < kTM; ++t)
#pragma unroll
        for (int j = 0; j < kTN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][j][e] = 0.f;

#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (i < blocks) load(i, i);
        __pipeline_commit();
    }
    for (int s = 0; s < blocks; ++s) {
        unsigned sa[kTM], sb[kTN];
#pragma unroll
        for (int t = 0; t < kTM; ++t) sa[t] = xe[static_cast<long long>(srow[t]) * blocks + s];
#pragma unroll
        for (int j = 0; j < kTN; ++j)
            sb[j] = atoms[atom_offset(min(n0 + wn * 8 * kTN + 8 * j, n - 1), s, blocks)];
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        const int next = s + kStages - 1;
        if (next < blocks) load(next, next % kStages);
        __pipeline_commit();

        const u8* sA = smem + (s % kStages) * kStage;
        const u8* sB = sA + kCM * kBK;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 32) {
            unsigned a[kTM][4];
            unsigned b[kTN][2];
#pragma unroll
            for (int t = 0; t < kTM; ++t) {
                const int r = wm * 16 * kTM + 16 * t + (lane & 15);
                ldmatrix_x4(a[t], sA + r * kBK + ((kk + (lane & 16)) ^ ((r & 7) << 4)));
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                const int r = wn * 8 * kTN + 8 * j + (lane & 7);
                ldmatrix_x2(b[j], sB + r * kBK + ((kk + ((lane & 8) << 1)) ^ ((r & 7) << 4)));
            }
#pragma unroll
            for (int t = 0; t < kTM; ++t)
#pragma unroll
                for (int j = 0; j < kTN; ++j) mma_e4m3_scaled(acc[t][j], a[t], b[j], sa[t], sb[j]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int t = 0; t < kTM; ++t)
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int col = n0 + wn * 8 * kTN + 8 * j + 2 * (lane & 3);
            if (col >= n) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 16 * kTM + 16 * t + (lane >> 2) + 8 * h;
                if (row >= m) continue;
                const unsigned lo = f32_to_bf16(acc[t][j][2 * h]).raw;
                const unsigned hi = f32_to_bf16(acc[t][j][2 * h + 1]).raw;
                *reinterpret_cast<unsigned*>(out + static_cast<long long>(row) * n + col) = lo | (hi << 16);
            }
        }
}

}  // namespace pie::linear
