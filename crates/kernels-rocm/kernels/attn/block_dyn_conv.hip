#pragma once

// A causal depthwise conv inside each request's block of rows whose taps
// are a shared base plus a per-row, per-group correction:
//
//   y[t][c] = Σ_{k < taps, k ≤ t} (base[side·taps + k][c]
//                                  + coeff[t][(side·taps + k)·groups + c / group])
//             · x[t − k][c]
//
// with t counted from the request's first row (the conv sees nothing before
// it). One thread per (channel, request), walking the request's rows.

#include "prelude/device.hip"

namespace pie::attn {

template <class T>
__global__ void block_dyn_conv(const T* __restrict__ x, const i32* __restrict__ indptr,
                               const T* __restrict__ coeff, const T* __restrict__ base,
                               T* __restrict__ y, int channels, int side, int taps, int group,
                               const u32* __restrict__ win) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    if (c >= channels) return;
    if (win != nullptr && blockIdx.y >= win[2]) return;
    const int begin = indptr[r];
    const int span = indptr[r + 1] - begin;

    const int groups = channels / group;
    const int g = c / group;
    const long long pitch = 2LL * taps * groups;
    for (int t = 0; t < span; ++t) {
        const long long row = begin + t;
        float acc = 0.f;
        for (int k = 0; k < taps && k <= t; ++k) {
            const int at = side * taps + k;
            const float coef = to_f32(base[static_cast<long long>(at) * channels + c]) +
                               to_f32(coeff[row * pitch + static_cast<long long>(at) * groups + g]);
            acc += coef * to_f32(x[(row - k) * channels + c]);
        }
        y[row * channels + c] = from_f32<T>(acc);
    }
}

}  // namespace pie::attn
