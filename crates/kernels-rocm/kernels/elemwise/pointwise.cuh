#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

constexpr int kBinAdd = 0;
constexpr int kBinMul = 1;

template <class T, int OP>
__global__ void binary(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ o,
    i32 n,
    i32 width,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (win != nullptr &&
        static_cast<long long>(idx) >= static_cast<long long>(win[0]) * width) return;
    const long long at =
        win != nullptr ? idx + static_cast<long long>(win[1]) * width : idx;

    const float a = Elem<T>::to_f32(x[at]);
    const float b = Elem<T>::to_f32(y[at]);
    o[at] = Elem<T>::from_f32(OP == kBinAdd ? a + b : a * b);
}

constexpr int kActSilu = 0;
constexpr int kActTanh = 1;

template <class T, int ACT>
__global__ void activation(
    const T* __restrict__ x,
    T* __restrict__ o,
    i32 n,
    i32 width,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (win != nullptr &&
        static_cast<long long>(idx) >= static_cast<long long>(win[0]) * width) return;
    const long long at =
        win != nullptr ? idx + static_cast<long long>(win[1]) * width : idx;

    const float v = Elem<T>::to_f32(x[at]);

    o[at] = Elem<T>::from_f32(ACT == kActSilu ? v / (1.f + __expf(-v)) : tanhf(v));
}

}
