#pragma once

// Gated activations. A packed row is [gate | up], each `I` wide; a launch
// over a window covers `fan` launch rows per stated row.

#include "prelude/device.hip"

namespace pie::linear {

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

// x *= sigmoid(gate), elementwise over a `width`-wide plane.
template <class T>
__global__ void gate_sigmoid_mul(T* __restrict__ x, const T* __restrict__ gate, i32 n, i32 width,
                                 i32 fan, const u32* __restrict__ win) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Window window{win};
    const long long row = i / width;
    if (!window.live(row, fan)) return;
    const long long at = window.plane(row, fan) * width + i % width;
    x[at] = from_f32<T>(to_f32(x[at]) * sigmoid(to_f32(gate[at])));
}

// y = silu(gate) · up. Grid: (rows, ⌈I / block⌉).
template <class T>
__global__ void mlp_swiglu(const T* __restrict__ packed, T* __restrict__ y, i32 I, i32 fan,
                           const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x, fan)) return;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;
    const long long row = window.plane(blockIdx.x, fan);
    const float g = to_f32(packed[row * 2 * I + i]);
    const float u = to_f32(packed[row * 2 * I + I + i]);
    y[row * I + i] = from_f32<T>(g * sigmoid(g) * u);
}

}  // namespace pie::linear
