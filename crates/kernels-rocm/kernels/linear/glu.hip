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

// 0.5·g·(1 + tanh(√(2/π)·(g + 0.044715·g³))), the tanh form of GELU.
__device__ __forceinline__ float gelu_tanh(float g) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    return 0.5f * g * (1.f + tanhf(kAlpha * (g + kBeta * g * g * g)));
}

// The flat element `i` of a `width`-wide plane a windowed launch covers:
// its plane offset, or -1 past the window's rows.
__device__ __forceinline__ long long flat_at(i32 i, i32 width, i32 fan, const u32* win) {
    const Window window{win};
    const long long row = i / width;
    if (!window.live(row, fan)) return -1;
    return window.plane(row, fan) * width + i % width;
}

// y = gelu_tanh(gate) · up, over separate gate and up planes.
template <class T>
__global__ void mlp_geglu_tanh(const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ y, i32 n,
                               i32 width, i32 fan, const u32* __restrict__ win) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = flat_at(i, width, fan, win);
    if (at < 0) return;
    y[at] = from_f32<T>(gelu_tanh(to_f32(gate[at])) * to_f32(up[at]));
}

// y = gelu_tanh(x).
template <class T>
__global__ void mlp_gelu_tanh(const T* __restrict__ x, T* __restrict__ y, i32 n, i32 width, i32 fan,
                              const u32* __restrict__ win) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = flat_at(i, width, fan, win);
    if (at < 0) return;
    y[at] = from_f32<T>(gelu_tanh(to_f32(x[at])));
}

// The packed launches below share mlp_swiglu's grid, (rows, ⌈I / block⌉):
// `row` is the plane row of the launch's row, -1 past the window, and `i`
// the element.
template <class F>
__device__ __forceinline__ void packed_each(i32 I, i32 fan, const u32* win, F&& f) {
    const Window window{win};
    if (!window.live(blockIdx.x, fan)) return;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;
    f(window.plane(blockIdx.x, fan), i);
}

// y = gelu_tanh(gate) · up over a packed [gate | up] row.
template <class T>
__global__ void mlp_geglu_tanh_packed(const T* __restrict__ packed, T* __restrict__ y, i32 I, i32 fan,
                                      const u32* __restrict__ win) {
    packed_each(I, fan, win, [&](long long row, i32 i) {
        const float g = to_f32(packed[row * 2 * I + i]);
        const float u = to_f32(packed[row * 2 * I + I + i]);
        y[row * I + i] = from_f32<T>(gelu_tanh(g) * u);
    });
}

// y = β·tanh(g/β)·sigmoid(g) · u, the up half capped to β'·tanh(u/β') when
// β' (`linear_beta`) is positive.
template <class T>
__global__ void mlp_situ(const T* __restrict__ packed, T* __restrict__ y, i32 I, float beta, float linear_beta,
                         i32 fan, const u32* __restrict__ win) {
    packed_each(I, fan, win, [&](long long row, i32 i) {
        const float g = to_f32(packed[row * 2 * I + i]);
        float u = to_f32(packed[row * 2 * I + I + i]);
        const float s = beta * tanhf(g / beta) / (1.f + __expf(-g));
        if (linear_beta > 0.f) u = linear_beta * tanhf(u / linear_beta);
        y[row * I + i] = from_f32<T>(s * u);
    });
}

// The clamped swiglu: the gate capped above at `limit`, the up half to
// [−limit, limit], then silu(g) · u.
template <class T>
__global__ void mlp_swiglu_clamp(const T* __restrict__ packed, T* __restrict__ y, i32 I, float limit, i32 fan,
                                 const u32* __restrict__ win) {
    packed_each(I, fan, win, [&](long long row, i32 i) {
        const float g = fminf(to_f32(packed[row * 2 * I + i]), limit);
        const float u = fminf(fmaxf(to_f32(packed[row * 2 * I + I + i]), -limit), limit);
        y[row * I + i] = from_f32<T>(g / (1.f + expf(-g)) * u);
    });
}

// As mlp_swiglu_clamp over separate gate and up planes (the combine behind
// expert banks the matmul lands as two halves).
template <class T>
__global__ void mlp_swiglu_clamp_split(const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ y,
                                       i32 n, i32 width, float limit, i32 fan, const u32* __restrict__ win) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = flat_at(i, width, fan, win);
    if (at < 0) return;
    const float g = fminf(to_f32(gate[at]), limit);
    const float u = fminf(fmaxf(to_f32(up[at]), -limit), limit);
    y[at] = from_f32<T>(g / (1.f + expf(-g)) * u);
}

// As mlp_swiglu_clamp, with the gate's sigmoid sharpened by `alpha` and the
// up half shifted by one: (u + 1) · g·sigmoid(α·g).
template <class T>
__global__ void mlp_swiglu_clamp_alpha(const T* __restrict__ packed, T* __restrict__ y, i32 I, float limit,
                                       float alpha, i32 fan, const u32* __restrict__ win) {
    packed_each(I, fan, win, [&](long long row, i32 i) {
        const float g = fminf(to_f32(packed[row * 2 * I + i]), limit);
        const float u = fminf(fmaxf(to_f32(packed[row * 2 * I + I + i]), -limit), limit);
        y[row * I + i] = from_f32<T>((u + 1.f) * (g / (1.f + expf(-alpha * g))));
    });
}

// ---------------------------------------------------------------- MoE --

// y = routed + shared · sigmoid(gate[row · stride]): the shared expert
// folded in behind its scalar gate. Grid (rows, ⌈H / block⌉).
template <class T>
__global__ void moe_sigmoid_gate_add(T* __restrict__ out, const T* __restrict__ sum, const T* __restrict__ x,
                                     const T* __restrict__ scalar_gate, i32 H, i32 stride,
                                     const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const i32 h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const float s = sigmoid(to_f32(scalar_gate[row * stride]));
    const long long i = row * H + h;
    out[i] = from_f32<T>(to_f32(sum[i]) + to_f32(x[i]) * s);
}

}  // namespace pie::linear

// ---------------------------------------------------------------------------
// Per-head sigmoid gate.
// ---------------------------------------------------------------------------

namespace pie::linear {

// x *= scale · sigmoid(gate[head]): one logit per `head_dim`-wide head, the
// gate plane `heads` wide. One block per row.
template <class T>
__global__ void gate_sigmoid_mul_heads(T* __restrict__ x, const T* __restrict__ gate, i32 heads,
                                       i32 head_dim, float scale, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const i32 width = heads * head_dim;
    T* xr = x + row * width;
    const T* gr = gate + row * heads;
    for (i32 i = threadIdx.x; i < width; i += blockDim.x) {
        const float s = scale / (1.f + __expf(-to_f32(gr[i / head_dim])));
        xr[i] = from_f32<T>(to_f32(xr[i]) * s);
    }
}

}  // namespace pie::linear
