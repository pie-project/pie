#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

template <class U>
__global__ void scatter_live_rows(
    const U* __restrict__ tight,
    U* __restrict__ wide,
    const i32* __restrict__ index,
    int units,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 at = index[row];
    if (at < 0) return;

    const U* src = tight + static_cast<long long>(row) * units;
    U* dst = wide + static_cast<long long>(at) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

}
