#pragma once

// A voxel grid's lane table: one `{t, h, w, row_offset}` i32 row a lane, the
// lane's voxels laid out frame by frame, row by row, from `row_offset`.

#include "prim/device.hip"

namespace pie::voxels {

struct Lane {
    int t;
    int h;
    int w;
    int off;

    __device__ __forceinline__ int voxels() const { return t * h * w; }
    __device__ __forceinline__ int plane() const { return h * w; }
};

__device__ __forceinline__ Lane lane_at(const int* __restrict__ grid, int l) {
    const int* row = grid + 4 * l;
    return Lane{row[0], row[1], row[2], row[3]};
}

// The lane holding `row`, or -1 for a row past every lane.
__device__ __forceinline__ int lane_of(const int* __restrict__ grid, int lanes, int row, Lane& out) {
    for (int l = 0; l < lanes; ++l) {
        const Lane g = lane_at(grid, l);
        const int local = row - g.off;
        if (local >= 0 && local < g.voxels()) {
            out = g;
            return l;
        }
    }
    return -1;
}

// The rows a voxel attends over: its whole lane, or its block of
// `seg_frames` frames.
__device__ __forceinline__ void segment_of(const Lane& g, int local, int seg_frames, int& begin, int& end) {
    if (seg_frames <= 0) {
        begin = g.off;
        end = g.off + g.voxels();
        return;
    }
    const int plane = g.plane();
    const int frame = plane > 0 ? local / plane : 0;
    const int first = frame / seg_frames * seg_frames;
    const int last = min(first + seg_frames, g.t);
    begin = g.off + first * plane;
    end = g.off + last * plane;
}

struct Voxel {
    int t;
    int h;
    int w;
};

__device__ __forceinline__ Voxel unravel(const Lane& g, int local) {
    const int w = local % g.w;
    const int rest = local / g.w;
    return Voxel{rest / g.h, rest % g.h, w};
}

__device__ __forceinline__ int ravel(const Lane& g, int t, int h, int w) {
    return g.off + (t * g.h + h) * g.w + w;
}

}  // namespace pie::voxels
