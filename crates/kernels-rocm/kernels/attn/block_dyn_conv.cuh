#pragma once


#include "prelude/device.cuh"

namespace pie::attn {

template <class T>
using Elem = ::pie::Elem<T>;

template <class T>
__global__ void block_dyn_conv(
    const T* __restrict__ x,
    const i32* __restrict__ indptr,
    const T* __restrict__ coeff,
    const T* __restrict__ base,
    T* __restrict__ y,
    int channels,
    int side,
    int taps,
    int group,
    const u32* __restrict__ win)
{
    const int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int r = static_cast<int>(blockIdx.y);
    if (c >= channels) return;
    if (win != nullptr && blockIdx.y >= win[2]) return;
    const int begin = indptr[r];
    const int end = indptr[r + 1];
    if (end <= begin) return;

    const int span = end - begin;
    const int groups = channels / group;
    const int g = c / group;
    const size_t chans = static_cast<size_t>(channels);
    const size_t pitch = static_cast<size_t>(2 * taps) * static_cast<size_t>(groups);

    for (int t = 0; t < span; ++t) {
        const size_t row = static_cast<size_t>(begin + t);
        float acc = 0.0f;
        for (int k = 0; k < taps; ++k) {
            const int src = t - k;
            if (src < 0) break;
            const int at = side * taps + k;
            const float coef =
                Elem<T>::to_f32(base[static_cast<size_t>(at) * chans + static_cast<size_t>(c)])
                + Elem<T>::to_f32(
                    coeff[row * pitch + static_cast<size_t>(at) * static_cast<size_t>(groups)
                          + static_cast<size_t>(g)]);
            acc += coef
                   * Elem<T>::to_f32(x[static_cast<size_t>(begin + src) * chans + static_cast<size_t>(c)]);
        }
        y[row * chans + static_cast<size_t>(c)] = Elem<T>::from_f32(acc);
    }
}

}
