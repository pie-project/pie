#pragma once

// Folds of consecutive row blocks (a vision tower's side x side patch
// squares, laid out block by block): averaged into one row, or set side by
// side into one wide row. One block per folded row.

#include "prim/device.hip"

namespace pie::layout {

// y[o] = the mean of x[o·block .. (o+1)·block].
template <class T>
__global__ void pool_rows(const T* __restrict__ x, T* __restrict__ y, int width, int block) {
    const long long out = blockIdx.x;
    const T* src = x + out * block * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < block; ++r) acc += to_f32(src[static_cast<long long>(r) * width + i]);
        y[out * width + i] = from_f32<T>(acc / static_cast<float>(block));
    }
}

// y[o] = [x[o·block] | x[o·block + 1] | ...]: the rows are contiguous, so a
// copy of block·width elements.
template <class T>
__global__ void merge_rows(const T* __restrict__ x, T* __restrict__ y, int width, int block) {
    const long long units = static_cast<long long>(block) * width;
    const T* src = x + blockIdx.x * units;
    T* dst = y + blockIdx.x * units;
    for (long long i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
}

}  // namespace pie::layout
