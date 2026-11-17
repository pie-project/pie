#pragma once

// Clamps in place over the first `n` elements of a `width`-wide plane.

#include "prim/device.hip"
#include "elemwise/pointwise.hip"

namespace pie::elemwise {

// x = min(max(x, lo), hi), the bounds rounded to T first as the reference
// does: a bound between two representable values clamps to the stored one.
template <class T, class I>
__global__ void clamp(T* __restrict__ x, float lo, float hi, I n, int width,
                      const u32* __restrict__ win) {
    sweep(n, [&](I i) {
        const long long at = flat_at(win, i, width);
        if (at < 0) return false;
        const float l = to_f32(from_f32<T>(lo));
        const float h = to_f32(from_f32<T>(hi));
        x[at] = from_f32<T>(fminf(fmaxf(to_f32(x[at]), l), h));
        return true;
    });
}

// The same with the bounds read from one-element device planes (learned
// clip ranges).
template <class T, class I>
__global__ void clamp_learned(T* __restrict__ x, const T* __restrict__ lo, const T* __restrict__ hi,
                              I n, int width, const u32* __restrict__ win) {
    sweep(n, [&](I i) {
        const long long at = flat_at(win, i, width);
        if (at < 0) return false;
        x[at] = from_f32<T>(fminf(fmaxf(to_f32(x[at]), to_f32(lo[0])), to_f32(hi[0])));
        return true;
    });
}

}  // namespace pie::elemwise
