#pragma once

// wide[index[n]] = tight[n] for the live rows whose index is not negative:
// a tower's rows placed into the token rows they stand for. One block per
// row.

#include "prim/device.hip"

namespace pie::layout {

template <class U>
__global__ void scatter_live_rows(const U* __restrict__ tight, U* __restrict__ wide,
                                  const i32* __restrict__ index, int units, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const i32 at = index[row];
    if (at < 0) return;
    const U* src = tight + row * units;
    U* dst = wide + static_cast<long long>(at) * units;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
}

}  // namespace pie::layout
