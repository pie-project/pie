#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

template <class TA, class TW, class TY>
__global__ void lane_gemm(
    const TA* __restrict__ act,
    const TW* __restrict__ w,
    TY* __restrict__ y,
    int rows,
    int n,
    int k)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int c = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) + warp;
    const int r = static_cast<int>(blockIdx.y);
    if (c >= n || r >= rows) return;
    const TA* a = act + static_cast<long long>(r) * k;
    const TW* ww = w + static_cast<long long>(c) * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) {
        acc = fmaf(Elem<TA>::to_f32(a[i]), Elem<TW>::to_f32(ww[i]), acc);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[static_cast<long long>(r) * n + c] = Elem<TY>::from_f32(acc);
    }
}

}
