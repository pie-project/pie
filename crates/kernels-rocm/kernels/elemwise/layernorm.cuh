#pragma once

#include "prelude/device.cuh"
#include "elemwise/norm.cuh"

namespace pie::elemwise {

template <class T, int BLOCK, bool AFFINE>
__device__ __forceinline__ void layernorm_row(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;

    if (win != nullptr && row >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    __shared__ float buf[BLOCK];

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        local += Elem<T>::to_f32(xr[i]);
    }
    const float mean = block_reduce_sum_fast<BLOCK>(local, buf) /
                       static_cast<float>(hidden);

    __syncthreads();

    float spread = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = Elem<T>::to_f32(xr[i]) - mean;
        spread += c * c;
    }
    const float inv = rsqrtf(block_reduce_sum_fast<BLOCK>(spread, buf) /
                                 static_cast<float>(hidden) +
                             eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = (Elem<T>::to_f32(xr[i]) - mean) * inv;
        if constexpr (AFFINE) {
            yr[i] = Elem<T>::from_f32(
                fmaf(c, Elem<T>::to_f32(weight[i]), Elem<T>::to_f32(bias[i])));
        } else {
            yr[i] = Elem<T>::from_f32(c);
        }
    }
}

template <class T, int BLOCK = 256>
__global__ void layernorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    layernorm_row<T, BLOCK, false>(x, nullptr, nullptr, y, hidden, eps, win);
}

template <class T, int BLOCK = 256>
__global__ void layernorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    layernorm_row<T, BLOCK, true>(x, weight, bias, y, hidden, eps, win);
}

}
