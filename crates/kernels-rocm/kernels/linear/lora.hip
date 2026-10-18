#pragma once

// The LoRA correction's second half: y[row] += B[adapter] · t[row], t the
// row's rank-wide down projection, the up bank [adapter][out][rank]. A row
// with adapter < 0 takes none. Grid (⌈out / block⌉, rows) — or, given
// `segments` ({first row, rows} pairs), (⌈out / block⌉, the most rows a
// segment, segments) — f32 accumulation in rank order.

#include "prelude/device.hip"

namespace pie::linear {

template <class T>
__global__ void lora_combine(const i32* __restrict__ routes, const T* __restrict__ t, const T* __restrict__ bank_b,
                             T* __restrict__ y, const i32* __restrict__ segments, int segs, int rank, int out_width,
                             long long adapter_stride, const u32* __restrict__ win) {
    int row = blockIdx.y;
    if (segments != nullptr) {
        const int seg = blockIdx.z;
        if (seg >= segs || static_cast<int>(blockIdx.y) >= segments[2 * seg + 1]) return;
        row = segments[2 * seg] + blockIdx.y;
    }
    const Window window{win};
    if (!window.live(row)) return;
    const long long at = window.plane(row);
    const int adapter = routes[at];
    if (adapter < 0) return;

    const T* b = bank_b + static_cast<long long>(adapter) * adapter_stride;
    const T* tv = t + at * rank;
    T* out = y + at * out_width;
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < out_width; n += gridDim.x * blockDim.x) {
        const T* brow = b + static_cast<long long>(n) * rank;
        float acc = 0.f;
        for (int r = 0; r < rank; ++r) acc += to_f32(brow[r]) * to_f32(tv[r]);
        out[n] = from_f32<T>(to_f32(out[n]) + acc);
    }
}

}  // namespace pie::linear
