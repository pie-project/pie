#pragma once

// Layer norm. One block of `BLOCK` threads (whole wavefronts) per row: the
// mean, then the variance about it (two passes, as the reference: no
// E[x²] - E[x]² cancellation), f32 throughout, one rounding on the store.
// A row whose width is a multiple of 8 and whose planes are 16-byte aligned
// moves in 16-byte packs.

#include "prelude/device.hip"

namespace pie::elemwise {

// Calls `f(i, x[i])` for this thread's share of the row's `n` elements.
template <class T, int BLOCK, class F>
__device__ __forceinline__ void each_of_row(const T* __restrict__ x, int n, bool packed, F&& f) {
    if (packed) {
        for (int i = threadIdx.x; i < n / 8; i += BLOCK) {
            const Pack8<T> p = reinterpret_cast<const Pack8<T>*>(x)[i];
#pragma unroll
            for (int j = 0; j < 8; ++j) f(8 * i + j, to_f32(p.v[j]));
        }
    } else {
        for (int i = threadIdx.x; i < n; i += BLOCK) f(i, to_f32(x[i]));
    }
}

// y = (x - mean) · rsqrt(var + eps) [· w + b].
template <class T, int BLOCK, bool AFFINE>
__device__ __forceinline__ void layernorm_row(const T* __restrict__ x, const T* __restrict__ w,
                                              const T* __restrict__ b, T* __restrict__ y,
                                              int hidden, float eps, const u32* __restrict__ win) {
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x) * hidden;
    const T* xr = x + row;
    T* yr = y + row;
    const bool packed = hidden % 8 == 0 && aligned16(xr) && aligned16(yr) &&
                        (!AFFINE || (aligned16(w) && aligned16(b)));

    float local = 0.f;
    each_of_row<T, BLOCK>(xr, hidden, packed, [&](int, float v) { local += v; });
    const float mean = block_sum(local, smem) / static_cast<float>(hidden);

    float spread = 0.f;
    each_of_row<T, BLOCK>(xr, hidden, packed, [&](int, float v) {
        const float c = v - mean;
        spread += c * c;
    });
    const float inv = rsqrtf(block_sum(spread, smem) / static_cast<float>(hidden) + eps);

    auto normed = [&](int i, float v) {
        const float c = (v - mean) * inv;
        if constexpr (AFFINE) return fmaf(c, to_f32(w[i]), to_f32(b[i]));
        return c;
    };
    if (packed) {
        for (int i = threadIdx.x; i < hidden / 8; i += BLOCK) {
            const Pack8<T> p = reinterpret_cast<const Pack8<T>*>(xr)[i];
            Pack8<T> q;
#pragma unroll
            for (int j = 0; j < 8; ++j) q.v[j] = from_f32<T>(normed(8 * i + j, to_f32(p.v[j])));
            reinterpret_cast<Pack8<T>*>(yr)[i] = q;
        }
    } else {
        for (int i = threadIdx.x; i < hidden; i += BLOCK) yr[i] = from_f32<T>(normed(i, to_f32(xr[i])));
    }
}

template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void layernorm_no_scale(const T* __restrict__ x,
                                                            T* __restrict__ y, int hidden,
                                                            float eps,
                                                            const u32* __restrict__ win) {
    layernorm_row<T, BLOCK, false>(x, nullptr, nullptr, y, hidden, eps, win);
}

template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void layernorm(const T* __restrict__ x, const T* __restrict__ w,
                                                   const T* __restrict__ b, T* __restrict__ y,
                                                   int hidden, float eps,
                                                   const u32* __restrict__ win) {
    layernorm_row<T, BLOCK, true>(x, w, b, y, hidden, eps, win);
}

}  // namespace pie::elemwise
