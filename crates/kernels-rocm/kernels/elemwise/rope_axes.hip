#pragma once

// Rotary over up to four position axes (f32 positions, `axes` a row): angle
// pairs split among the axes by `dims` (each axis turning dims[a]/2 pairs at
// its own theta), laid out interleaved (2i, 2i + 1 in the axis's block), neox
// (i, i + rotary_dim/2 across the head), split (i, i + dims[a]/2 in the axis's
// block) or split ladder (angles dealt round-robin to the axes on a ladder
// theta^(f/(ladder-1)), the first `pad` left unturned). A block a row, x into
// o; the unrotated tail is copied when o is not x.

#include "elemwise/rope.hip"

namespace pie::elemwise {

constexpr int kRopeInterleaved = 0;
constexpr int kRopeNeox = 1;
constexpr int kRopeSplit = 2;
constexpr int kRopeSplitLadder = 3;

template <class T, int FORM>
__global__ void rope_axes(const T* __restrict__ x, const float* __restrict__ positions,
                          T* __restrict__ o, int axes, int d0, int d1, int d2, int d3, float t0,
                          float t1, float t2, float t3, int rotary_dim, int head_dim, int heads,
                          const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int dims[4] = {d0, d1, d2, d3};
    const float thetas[4] = {t0, t1, t2, t3};
    const int angles = rotary_dim / 2;
    const long long width = static_cast<long long>(heads) * head_dim;
    const float* pos = positions + row * axes;
    const T* xr = x + row * width;
    T* orow = o + row * width;

    for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
        const int head = idx / angles;
        const int angle = idx % angles;
        float c = 1.f;
        float s = 0.f;
        int lo;
        int hi;
        if constexpr (FORM == kRopeSplitLadder) {
            int span = 0;
            for (int a = 0; a < axes; ++a) span += dims[a];
            const int pad = (heads * rotary_dim - span) / 2;
            if (idx >= pad) {
                const int slot = idx - pad;
                const int axis = slot % axes;
                const int ladder = dims[axis] / 2;
                const float exponent = ladder > 1
                    ? static_cast<float>(slot / axes) / static_cast<float>(ladder - 1)
                    : 0.f;
                sincosf(pos[axis] * powf(thetas[axis], exponent), &s, &c);
            }
            lo = angle;
            hi = angle + angles;
        } else {
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
            const float freq = powf(thetas[axis], -2.f * static_cast<float>(within) /
                                                      static_cast<float>(dims[axis]));
            sincosf(pos[axis] * freq, &s, &c);
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
        }
        const T* xh = xr + static_cast<long long>(head) * head_dim;
        T* oh = orow + static_cast<long long>(head) * head_dim;
        const float a = to_f32(xh[lo]);
        const float b = to_f32(xh[hi]);
        oh[lo] = from_f32<T>(a * c - b * s);
        oh[hi] = from_f32<T>(b * c + a * s);
    }

    if (FORM != kRopeSplitLadder && o != x) {
        const int tail = head_dim - rotary_dim;
        for (int idx = threadIdx.x; idx < heads * tail; idx += blockDim.x) {
            const int head = idx / tail;
            const int i = rotary_dim + idx % tail;
            orow[head * head_dim + i] = xr[head * head_dim + i];
        }
    }
}

}  // namespace pie::elemwise
