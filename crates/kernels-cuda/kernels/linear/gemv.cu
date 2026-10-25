#include "linear/contract.cuh"
#include "prim/device.cuh"
#include <cuda_bf16.h>

namespace pie::linear {

// `gemv_frame`'s dot over a grouped plane at one row: weight row `row` is
// group `row / group_n`'s, against that group's K-wide slice of the row.
template <int kWarps>
__global__ void gemv_bf16_grouped_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int N, int K, int group_n)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float acc = gemv_bf16_row<4>(weight, act + (long long)(row / group_n) * K, row, K);
    if (threadIdx.x == 0) out[row] = f32_to_bf16(acc);
}

// `out[row][n] = act[row] . weight[n]` for a product a few columns wide
// (`N` <= 8): a block a row, its threads striding the contraction eight
// elements at a time with every column's sum in registers, folded across
// the block. The bias, if any, lands as the gemv lands it.
template <int N>
__global__ void __launch_bounds__(256) gemm_narrow_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int k)
{
    constexpr int kThreads = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const bf16* x = act + static_cast<long long>(row) * k;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    for (int i = tid * 8; i < k; i += kThreads * 8) {
        const uint4 xv = *reinterpret_cast<const uint4*>(x + i);
        const bf16* x8 = reinterpret_cast<const bf16*>(&xv);
        float xf[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) xf[e] = bf16_to_f32(x8[e]);
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const uint4 wv = *reinterpret_cast<const uint4*>(weight + static_cast<long long>(n) * k + i);
            const bf16* w8 = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[n] += xf[e] * bf16_to_f32(w8[e]);
        }
    }
    __shared__ float part[kThreads / 32][N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
        v = ::pie::warp_sum(v);
        if ((tid & 31) == 0) part[tid >> 5][n] = v;
    }
    __syncthreads();
    if (tid < N) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) v += part[w][tid];
        bf16 o = f32_to_bf16(v);
        if (bias != nullptr) o = f32_to_bf16(bf16_to_f32(o) + bf16_to_f32(bias[tid]));
        out[static_cast<long long>(row) * N + tid] = o;
    }
}

}
