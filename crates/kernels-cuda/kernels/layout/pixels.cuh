#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

// Patch rows of raw RGB bytes (patch x patch pixels, HWC) to the tower's
// rows: (v / 255 - mean[c]) / std[c], columns channel-major (order 0) or
// pixel-major (order 1), the frame repeated width / (3 patch^2) times.
template <class T>
__global__ void pixels(
    const u8* __restrict__ x,
    T* __restrict__ y,
    int patch,
    int width,
    int order,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2)
{
    const long long n = blockIdx.x;
    const int pixels = patch * patch;
    const int in_width = 3 * pixels;
    const int temporal = width / in_width;
    const u8* src = x + n * in_width;
    T* dst = y + n * width;

    for (int c_out = threadIdx.x; c_out < width; c_out += blockDim.x) {
        int ch, at;
        if (order == 0) {
            ch = c_out / (temporal * pixels);
            at = (c_out % pixels) * 3 + ch;
        } else {
            const int k = c_out % in_width;
            ch = k % 3;
            at = k;
        }
        const float mean = ch == 0 ? mean0 : (ch == 1 ? mean1 : mean2);
        const float sd = ch == 0 ? std0 : (ch == 1 ? std1 : std2);
        const float v = static_cast<float>(src[at]) / 255.f;
        dst[c_out] = Elem<T>::from_f32((v - mean) / sd);
    }
}

}
