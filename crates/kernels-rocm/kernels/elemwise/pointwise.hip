#pragma once

// Flat pointwise ops over the `n` elements of a `width`-wide plane, a
// grid-stride sweep (the grid capped at the device's x axis), f32
// arithmetic, one rounding on the store. `I` indexes the sweep: unsigned
// 32-bit while the plane holds at most 2^31 - 1 elements (one thread's
// next index then stays under 2^32), 64-bit past it.

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
constexpr int kBinDiv = 2;
constexpr int kBinMod = 3;

// o = x + y, x · y, x / y or x mod y (floored), the planes the same shape.
template <class T, int OP, class I>
__global__ void binary(const T* __restrict__ x, const T* __restrict__ y, T* __restrict__ o, I n,
                       i32 width, const u32* __restrict__ win) {
    sweep(n, [&](I i) {
        const long long at = flat_at(win, i, width);
        if (at < 0) return false;
        const float a = to_f32(x[at]);
        const float b = to_f32(y[at]);
        float r;
        if (OP == kBinAdd) r = a + b;
        else if (OP == kBinMul) r = a * b;
        else if (OP == kBinDiv) r = a / b;
        else r = a - b * floorf(a / b);
        o[at] = from_f32<T>(r);
        return true;
    });
}

constexpr int kActSilu = 0;
constexpr int kActTanh = 1;
constexpr int kActSqrt = 2;

template <class T, int ACT, class I>
__global__ void activation(const T* __restrict__ x, T* __restrict__ o, I n, i32 width,
                           const u32* __restrict__ win) {
    sweep(n, [&](I i) {
        const long long at = flat_at(win, i, width);
        if (at < 0) return false;
        const float v = to_f32(x[at]);
        o[at] = from_f32<T>(ACT == kActSilu   ? v / (1.f + __expf(-v))
                            : ACT == kActTanh ? tanhf(v)
                                              : sqrtf(v));
        return true;
    });
}

}  // namespace pie::elemwise
