#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

template <class T>
__global__ void pool_rows(
    const T* __restrict__ x,
    T* __restrict__ y,
    int width,
    int block)
{
    const int out = blockIdx.x;
    const long long base = static_cast<long long>(out) * block * width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < block; ++r) {
            acc += Elem<T>::to_f32(x[base + static_cast<long long>(r) * width + i]);
        }
        y[static_cast<long long>(out) * width + i] =
            Elem<T>::from_f32(acc / static_cast<float>(block));
    }
}


template <class T>
__global__ void merge_rows(
    const T* __restrict__ x,
    T* __restrict__ y,
    int width,
    int block)
{
    const int out = blockIdx.x;
    const long long units = static_cast<long long>(block) * width;
    const T* src = x + static_cast<long long>(out) * units;
    T* dst = y + static_cast<long long>(out) * units;

    for (long long i = threadIdx.x; i < units; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}
