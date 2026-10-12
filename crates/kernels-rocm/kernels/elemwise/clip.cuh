#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

template <class T>
__global__ void clamp(T* __restrict__ x, float lo, float hi, usize n,
                      int width, const u32* __restrict__ win)
{
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float lo_r = Elem<T>::to_f32(Elem<T>::from_f32(lo));
    const float hi_r = Elem<T>::to_f32(Elem<T>::from_f32(hi));
    const float v = Elem<T>::to_f32(x[at]);
    x[at] = Elem<T>::from_f32(fminf(fmaxf(v, lo_r), hi_r));
}


template <class T>
__global__ void clamp_learned(
    T* __restrict__ x,
    const T* __restrict__ lo,
    const T* __restrict__ hi,
    usize n,
    int width,
    const u32* __restrict__ win)
{
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float lo_v = Elem<T>::to_f32(lo[0]);
    const float hi_v = Elem<T>::to_f32(hi[0]);
    const float v = Elem<T>::to_f32(x[at]);
    x[at] = Elem<T>::from_f32(fminf(fmaxf(v, lo_v), hi_v));
}

}
