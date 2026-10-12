#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

template <class T>
__global__ void rel_bias(
    const T* __restrict__ r,
    const T* __restrict__ proj,
    float* __restrict__ bias,
    int rows, int heads, int d_rel, int extent,
    const u32* __restrict__ win)
{
    const int row = blockIdx.z;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    if (row >= rows) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;
    const int h = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads || d >= extent) return;
    const T* rr = r + (static_cast<long long>(plane_row) * heads + h) * d_rel;
    float acc = 0.f;
    for (int j = 0; j < d_rel; ++j) {
        acc += Elem<T>::to_f32(rr[j]) *
               Elem<T>::to_f32(proj[static_cast<long long>(j) * extent + d]);
    }
    bias[(static_cast<long long>(plane_row) * heads + h) * extent + d] = acc;
}

}
