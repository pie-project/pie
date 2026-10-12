#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

template <class T>
__global__ void embed_weighted(
    const i32* __restrict__ ids,
    const float* __restrict__ weights,
    const T* __restrict__ table,
    T* __restrict__ y,
    int hidden,
    int vocab,
    int taps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32* row_ids = ids + static_cast<long long>(row) * taps;
    const float* row_w = weights + static_cast<long long>(row) * taps;
    T* out = y + static_cast<long long>(row) * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < taps; ++t) {
            const i32 raw = row_ids[t];
            const int at = (raw >= 0 && raw < vocab) ? raw : 0;
            acc += row_w[t] *
                   Elem<T>::to_f32(table[static_cast<long long>(at) * hidden + i]);
        }
        out[i] = Elem<T>::from_f32(acc);
    }
}

}
