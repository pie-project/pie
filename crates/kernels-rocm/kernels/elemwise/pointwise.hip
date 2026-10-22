#pragma once

// Flat pointwise ops: one thread per element of a `width`-wide plane, f32
// arithmetic, one rounding on the store.

#include "prim/device.hip"

namespace pie::elemwise {

// Where flat element `i` of a `width`-wide launch lands on the plane, or -1
// when its row is past the window. No division: the window compares and
// shifts in whole rows of elements.
__device__ __forceinline__ long long flat_at(const u32* __restrict__ win, long long i, int width) {
    if (win == nullptr) return i;
    if (i >= static_cast<long long>(win[0]) * width) return -1;
    return i + static_cast<long long>(win[1]) * width;
}

constexpr int kBinAdd = 0;
constexpr int kBinMul = 1;

// o = x + y or x · y, the planes the same shape.
template <class T, int OP>
__global__ void binary(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ o, i32 n,
                       i32 width, const u32* __restrict__ win) {
    const i32 i = static_cast<i32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const long long at = flat_at(win, i, width);
    if (at < 0) return;
    const float a = to_f32(x[at]);
    const float b = to_f32(y[at]);
    o[at] = from_f32<T>(OP == kBinAdd ? a + b : a * b);
}

constexpr int kActSilu = 0;
constexpr int kActTanh = 1;

// o = silu(x) or tanh(x).
template <class T, int ACT>
__global__ void activation(const T* __restrict__ x, T* __restrict__ o, i32 n, i32 width,
                           const u32* __restrict__ win) {
    const i32 i = static_cast<i32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const long long at = flat_at(win, i, width);
    if (at < 0) return;
    const float v = to_f32(x[at]);
    o[at] = from_f32<T>(ACT == kActSilu ? v / (1.f + __expf(-v)) : tanhf(v));
}

}  // namespace pie::elemwise
