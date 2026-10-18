#pragma once

// Patch rows of raw RGB bytes (patch x patch pixels, HWC) to a vision
// tower's input rows: (v / 255 - mean[c]) / std[c], the columns
// channel-major (order 0) or pixel-major (order 1), the frame repeated
// width / (3·patch²) times. One block per patch.

#include "prelude/device.hip"

namespace pie::layout {

template <class T>
__global__ void pixels(const u8* __restrict__ x, T* __restrict__ y, int patch, int width, int order,
                       float mean0, float mean1, float mean2, float std0, float std1, float std2) {
    const long long n = blockIdx.x;
    const int area = patch * patch;
    const int in_width = 3 * area;
    const int temporal = width / in_width;
    const u8* src = x + n * in_width;
    T* dst = y + n * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        int ch, at;
        if (order == 0) {
            ch = c / (temporal * area);
            at = (c % area) * 3 + ch;
        } else {
            at = c % in_width;
            ch = at % 3;
        }
        const float mean = ch == 0 ? mean0 : (ch == 1 ? mean1 : mean2);
        const float sd = ch == 0 ? std0 : (ch == 1 ? std1 : std2);
        dst[c] = from_f32<T>((static_cast<float>(src[at]) / 255.f - mean) / sd);
    }
}

}  // namespace pie::layout
