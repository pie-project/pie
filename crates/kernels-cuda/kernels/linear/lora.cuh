#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

[[maybe_unused]] constexpr int kLoraBlock = 256;

template <class T>
__global__ void lora_combine(
    const i32* __restrict__ routes,
    const T* __restrict__ t,
    const T* __restrict__ bank_b,
    T* __restrict__ y,
    const i32* __restrict__ segments,
    int segs,
    int rank, int out_width, long long adapter_stride,
    const u32* __restrict__ win)
{
    int row = blockIdx.y;
    if (segments != nullptr) {
        const int seg = blockIdx.z;
        if (seg >= segs) return;
        if ((int)blockIdx.y >= segments[2 * seg + 1]) return;
        row = segments[2 * seg] + (int)blockIdx.y;
    }

    if (win != nullptr) {
        if (row >= (int)win[0]) return;
        row += (int)win[1];
    }
    const int adapter = routes[row];
    if (adapter < 0) return;

    const T* b = bank_b + (long long)adapter * adapter_stride;
    const T* tv = t + (long long)row * rank;
    T* out = y + (long long)row * out_width;

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < out_width;
         n += gridDim.x * blockDim.x) {
        const T* brow = b + (long long)n * rank;
        float acc = 0.f;
        for (int r = 0; r < rank; ++r) {
            acc += Elem<T>::to_f32(brow[r]) * Elem<T>::to_f32(tv[r]);
        }
        out[n] = Elem<T>::from_f32(Elem<T>::to_f32(out[n]) + acc);
    }
}

}
