#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

__global__ void sinusoid(
    const float* __restrict__ t,
    float* __restrict__ o,
    int dim,
    float max_period,
    int flip_sin_cos,
    float scale,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int half = dim / 2;
    const float tv = t[row];
    const float log_period = logf(max_period);
    float* orow = o + static_cast<long long>(row) * dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq =
            expf(-log_period * static_cast<float>(i) / static_cast<float>(half));
        float sin_v, cos_v;

        sincosf(scale * (tv * freq), &sin_v, &cos_v);
        orow[i] = flip_sin_cos ? cos_v : sin_v;
        orow[i + half] = flip_sin_cos ? sin_v : cos_v;
    }

    if ((dim & 1) != 0 && threadIdx.x == 0) orow[dim - 1] = 0.f;
}

}
