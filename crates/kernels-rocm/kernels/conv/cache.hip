#pragma once

// A causal conv3d's frame cache: the last `frames` frames of each lane, kept
// in a slot of a slab between fires. `cache_gather` lays a fire's slots out
// as `[Σ frames·plane, C]` rows, lane after lane; `cache_store` writes the
// frames the next fire reads back (this fire's last ones, topped up from the
// cache it read when the fire held fewer).

#include "prim/device.hip"
#include "prim/voxels.hip"

namespace pie::conv {

using namespace ::pie::voxels;

__device__ __forceinline__ int cache_lane_of(const int* __restrict__ grid, int lanes, int frames, int row, int& base,
                                             Lane& out) {
    int at = 0;
    for (int l = 0; l < lanes; ++l) {
        const Lane g = lane_at(grid, l);
        const int rows = frames * g.plane();
        if (row < at + rows) {
            base = at;
            out = g;
            return l;
        }
        at += rows;
    }
    return -1;
}

template <class T>
__global__ __launch_bounds__(256) void cache_gather(const T* __restrict__ slab, const int* __restrict__ slot_ids,
                                                    const int* __restrict__ grid, T* __restrict__ cache, int lanes,
                                                    int frames, int c, long long slot_stride, long long total) {
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    int base;
    Lane g;
    const int l = cache_lane_of(grid, lanes, frames, row, base, g);
    const long long local = static_cast<long long>(row - base) * c + col;
    if (l < 0 || local >= slot_stride) {
        cache[e] = from_f32<T>(0.f);
        return;
    }
    cache[e] = slab[static_cast<long long>(slot_ids[l]) * slot_stride + local];
}

template <class T>
__global__ __launch_bounds__(256) void cache_store(const T* __restrict__ x, const T* __restrict__ cache,
                                                   const int* __restrict__ slot_ids, const int* __restrict__ grid,
                                                   T* __restrict__ slab, int lanes, int frames, int c,
                                                   long long slot_stride, long long total) {
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    int base;
    Lane g;
    const int l = cache_lane_of(grid, lanes, frames, row, base, g);
    if (l < 0) return;
    const int local_row = row - base;
    const long long local = static_cast<long long>(local_row) * c + col;
    if (local >= slot_stride) return;
    const int plane = g.plane();
    const int f = local_row / plane;
    const int hw = local_row - f * plane;

    // Frame f of the stored window is the fire's frame t - frames + f, or,
    // before the fire's first, the cached frame it continues.
    const int src_t = g.t - frames + f;
    T value;
    if (src_t >= 0) {
        value = x[(static_cast<long long>(g.off) + static_cast<long long>(src_t) * plane + hw) * c + col];
    } else {
        value = cache[(static_cast<long long>(base) + static_cast<long long>(f + g.t) * plane + hw) * c + col];
    }
    slab[static_cast<long long>(slot_ids[l]) * slot_stride + local] = value;
}

}  // namespace pie::conv
