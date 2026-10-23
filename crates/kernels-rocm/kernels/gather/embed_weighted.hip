#pragma once

// y[n] = Σₜ weights[n][t] · table[ids[n][t]]: a row interpolated from a few
// table rows (a vision tower's position table resampled to an image's
// grid). An id outside the table reads row 0. One block per row.

#include "prim/device.hip"

namespace pie::layout {

template <class T>
__global__ void embed_weighted(const i32* __restrict__ ids, const float* __restrict__ weights,
                               const T* __restrict__ table, T* __restrict__ y, int hidden, int vocab, int taps,
                               const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const i32* row_ids = ids + row * taps;
    const float* row_w = weights + row * taps;
    T* out = y + row * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < taps; ++t) {
            const i32 id = row_ids[t];
            const long long at = static_cast<long long>(id >= 0 && id < vocab ? id : 0) * hidden + i;
            acc += row_w[t] * to_f32(table[at]);
        }
        out[i] = from_f32<T>(acc);
    }
}

}  // namespace pie::layout
