#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

__device__ __forceinline__ void grid_axis_taps(
    int index, int size, int side, int* taps, float* w)
{
    const float src = static_cast<float>(index) * static_cast<float>(side - 1) /
                      static_cast<float>(max(size - 1, 1));
    const float fl = floorf(src);
    for (int t = 0; t < 2; ++t) {
        taps[t] = min(max(static_cast<int>(fl) + t, 0), side - 1);
        w[t] = fmaxf(1.f - fabsf(src - fl - static_cast<float>(t)), 0.f);
    }
}

// Position-table taps for each patch: bilinear (kind 0) over its image's
// grid stretched onto a side x side table, or axes (kind 1), one unit tap
// on the column into the first `side` rows and one on the row into the
// next `side`.
__global__ void grid_taps(
    const i32* __restrict__ positions,
    const i32* __restrict__ grids,
    const i32* __restrict__ segments,
    i32* __restrict__ ids,
    float* __restrict__ weights,
    int kind,
    int side,
    int images,
    int patches)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= patches) return;

    int lo = 0, hi = images;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (segments[mid] <= n) lo = mid; else hi = mid;
    }
    const int row = positions[n * 3 + 1];
    const int col = positions[n * 3 + 2];
    if (kind == 1) {
        ids[n * 2] = min(col, side - 1);
        ids[n * 2 + 1] = side + min(row, side - 1);
        weights[n * 2] = 1.f;
        weights[n * 2 + 1] = 1.f;
        return;
    }
    const int gh = grids[lo * 3 + 1];
    const int gw = grids[lo * 3 + 2];
    int h_taps[2], w_taps[2];
    float h_w[2], w_w[2];
    grid_axis_taps(row, gh, side, h_taps, h_w);
    grid_axis_taps(col, gw, side, w_taps, w_w);
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            ids[n * 4 + a * 2 + b] = h_taps[a] * side + w_taps[b];
            weights[n * 4 + a * 2 + b] = h_w[a] * w_w[b];
        }
    }
}

}
