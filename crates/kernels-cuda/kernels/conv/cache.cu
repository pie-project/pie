#include "prim/device.cuh"
#include "prim/voxels.cuh"
#include "prim/state_rows.cuh"

namespace pie::conv {

using namespace ::pie::voxels;

// Where element `local` (row `local_row`, column `col`) of the frame slab of
// the lane at row `l` of the pool's lane table lives: the slab cuts along its frame-position rows, `rows.atoms` of
// them a piece.
template <class T>
__device__ __forceinline__ T* cache_cell(
    const StateRows& rows, int l, int local_row, int c, int col)
{
    const int p = local_row / rows.atoms;
    return rows.piece<T>(rows.row(l), p)
        + static_cast<long long>(local_row - p * rows.atoms) * c + col;
}

__device__ __forceinline__ int cache_lane_of(
    const int* __restrict__ grid, int lanes, int frames, int row, int& base, Lane& out)
{
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
__global__ __launch_bounds__(256) void cache_gather(
    StateRows rows,
    const int* __restrict__ grid,
    const int* __restrict__ clip_lanes,
    int lane_base,
    T* __restrict__ cache,
    int lanes,
    int frames,
    int c,
    long long row_elems,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    int base;
    Lane g;
    const int l = cache_lane_of(grid, lanes, frames, row, base, g);
    if (l < 0) {
        cache[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const long long local = static_cast<long long>(row - base) * c + col;
    if (local >= row_elems) {
        cache[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    cache[e] = *cache_cell<const T>(rows, clip_lanes[l] - lane_base, static_cast<int>(row - base), c, col);
}

template <class T>
__global__ __launch_bounds__(256) void cache_store(
    const T* __restrict__ x,
    const T* __restrict__ cache,
    StateRows rows,
    const int* __restrict__ grid,
    const int* __restrict__ clip_lanes,
    int lane_base,
    int lanes,
    int frames,
    int c,
    long long row_elems,
    int x_base,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    int base;
    Lane g;
    const int l = cache_lane_of(grid, lanes, frames, row, base, g);
    if (l < 0) return;
    // `x_base`: the row `grid` addresses `x`'s row 0 at.
    g.off -= x_base;
    const int local_row = row - base;
    const long long local = static_cast<long long>(local_row) * c + col;
    if (local >= row_elems) return;
    const int plane = g.plane();
    const int f = local_row / plane;
    const int hw = local_row - f * plane;

    const int src_t = g.t - frames + f;
    T value;
    if (src_t >= 0) {
        value = x[(static_cast<long long>(g.off) + static_cast<long long>(src_t) * plane + hw) * c + col];
    } else {
        value = cache[(static_cast<long long>(base) + static_cast<long long>(f + g.t) * plane + hw) * c + col];
    }
    *cache_cell<T>(rows, clip_lanes[l] - lane_base, local_row, c, col) = value;
}

}
