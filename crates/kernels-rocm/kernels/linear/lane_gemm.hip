#pragma once

// y[r, c] = act[r] · w[c] over any mix of f32, bf16 and f16, f32
// accumulation: one wavefront an output, its lanes striding K. The
// projection of an f32 activation, which no matrix-core arm takes.

#include "prelude/device.hip"

namespace pie::linear {

// Grid (⌈n / waves⌉, rows); a block's wavefronts take adjacent columns.
template <class TA, class TW, class TY>
__global__ void lane_gemm(const TA* __restrict__ act, const TW* __restrict__ w, TY* __restrict__ y, int rows, int n,
                          int k) {
    const int c = blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
    const int r = blockIdx.y;
    if (c >= n || r >= rows) return;
    const TA* a = act + static_cast<long long>(r) * k;
    const TW* ww = w + static_cast<long long>(c) * k;
    float acc = 0.f;
    for (int i = lane(); i < k; i += kWave) acc = fmaf(to_f32(a[i]), to_f32(ww[i]), acc);
    acc = wave_sum(acc);
    if (lane() == 0) y[static_cast<long long>(r) * n + c] = from_f32<TY>(acc);
}

}  // namespace pie::linear
