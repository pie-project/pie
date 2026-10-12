#pragma once

#include "vision/tower_naive_kernels.cuh"

namespace pie::vision {

template <class T>
__global__ void k_clamp(const T* x, T* o, const T* lo, const T* hi, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i >= t) return;
    float v = F(x[i]);
    float l = lo ? F(*lo) : ::pie::neg_inf();
    float h = hi ? F(*hi) : ::pie::pos_inf();
    o[i] = Bf<T>(v < l ? l : (v > h ? h : v));
}

}
