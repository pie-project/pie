#pragma once

// The relative-position bias: bias[row, h, d] = Σ_j r[row, h, j] ·
// proj[j, d], f32, over each head's `d_rel` relative features and a
// [d_rel, extent] profile bank. Grid (⌈extent / block⌉, heads, rows).

#include "prim/device.hip"

namespace pie::linear {

template <class T>
__global__ void rel_bias(const T* __restrict__ r, const T* __restrict__ proj, float* __restrict__ bias, int rows,
                         int heads, int d_rel, int extent, const u32* __restrict__ win) {
    const Window window{win};
    const int row = blockIdx.z;
    if (row >= rows || !window.live(row)) return;
    const long long at = window.plane(row);
    const int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads || d >= extent) return;
    const T* rr = r + (at * heads + h) * d_rel;
    float acc = 0.f;
    for (int j = 0; j < d_rel; ++j) acc += to_f32(rr[j]) * to_f32(proj[static_cast<long long>(j) * extent + d]);
    bias[(at * heads + h) * extent + d] = acc;
}

}  // namespace pie::linear
