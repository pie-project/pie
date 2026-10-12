#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

constexpr int kRopeInterleaved = 0;
constexpr int kRopeNeox = 1;
constexpr int kRopeSplit = 2;
constexpr int kRopeSplitLadder = 3;

template <class T, int FORM>
__global__ void rope_axes(
    const T* __restrict__ x,
    const float* __restrict__ positions,
    T* __restrict__ o,
    int axes,
    int d0, int d1, int d2, int d3,
    float t0, float t1, float t2, float t3,
    int rotary_dim,
    int head_dim,
    int heads,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int dims[4] = {d0, d1, d2, d3};
    const float thetas[4] = {t0, t1, t2, t3};
    const int angles = rotary_dim / 2;
    const int width = heads * head_dim;

    const float* pos = positions + static_cast<long long>(row) * axes;
    const T* xr = x + static_cast<long long>(row) * width;
    T* orow = o + static_cast<long long>(row) * width;

    if constexpr (FORM == kRopeSplitLadder) {

        int span = 0;
        for (int a = 0; a < axes; ++a) span += dims[a];
        const int pad = (heads * rotary_dim - span) / 2;
        for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
            const int head = idx / angles;
            const int angle = idx % angles;
            float cos_v = 1.f;
            float sin_v = 0.f;
            if (idx >= pad) {
                const int slot = idx - pad;
                const int axis = slot % axes;
                const int f = slot / axes;
                const int ladder = dims[axis] / 2;
                const float exponent =
                    ladder > 1 ? static_cast<float>(f) / static_cast<float>(ladder - 1) : 0.f;
                sincosf(pos[axis] * powf(thetas[axis], exponent), &sin_v, &cos_v);
            }
            const int lo = angle;
            const int hi = angle + angles;
            const T* xh = xr + static_cast<long long>(head) * head_dim;
            T* oh = orow + static_cast<long long>(head) * head_dim;
            const float a = Elem<T>::to_f32(xh[lo]);
            const float b = Elem<T>::to_f32(xh[hi]);
            oh[lo] = Elem<T>::from_f32(a * cos_v - b * sin_v);
            oh[hi] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
        return;
    }

    for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
        const int head = idx / angles;
        const int angle = idx % angles;

        int axis = 0;
        int first_angle = 0;
        int first_channel = 0;
        while (axis < axes && angle >= first_angle + dims[axis] / 2) {
            first_angle += dims[axis] / 2;
            first_channel += dims[axis];
            ++axis;
        }

        if (axis >= axes) continue;
        const int within = angle - first_angle;

        const float freq = powf(thetas[axis],
            -2.f * static_cast<float>(within) / static_cast<float>(dims[axis]));
        float cos_v, sin_v;
        __sincosf(pos[axis] * freq, &sin_v, &cos_v);

        int lo;
        int hi;
        if constexpr (FORM == kRopeInterleaved) {
            lo = first_channel + 2 * within;
            hi = lo + 1;
        } else if constexpr (FORM == kRopeNeox) {
            lo = angle;
            hi = angle + angles;
        } else {
            lo = first_channel + within;
            hi = first_channel + dims[axis] / 2 + within;
        }

        const T* xh = xr + static_cast<long long>(head) * head_dim;
        T* oh = orow + static_cast<long long>(head) * head_dim;
        const float a = Elem<T>::to_f32(xh[lo]);
        const float b = Elem<T>::to_f32(xh[hi]);
        oh[lo] = Elem<T>::from_f32(a * cos_v - b * sin_v);
        oh[hi] = Elem<T>::from_f32(b * cos_v + a * sin_v);
    }

    if (o != x) {
        const int tail = head_dim - rotary_dim;
        for (int idx = threadIdx.x; idx < heads * tail; idx += blockDim.x) {
            const int head = idx / tail;
            const int i = rotary_dim + idx % tail;
            orow[head * head_dim + i] = xr[head * head_dim + i];
        }
    }
}

}
