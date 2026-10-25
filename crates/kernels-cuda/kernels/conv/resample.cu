#include "prim/device.cuh"
#include "prim/voxels.cuh"

namespace pie::conv {

using namespace ::pie::voxels;

// `x_base` / `y_base`: the rows the fire-wide grids address `x`'s and `y`'s row 0 at.

template <class T>
__global__ __launch_bounds__(256) void upsample_nearest(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int ft,
    int fh,
    int fw,
    int keep_first,
    int x_base,
    int y_base,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c) + y_base;
    const int col = static_cast<int>(e % c);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int ti = keep_first ? (o.t == 0 ? 0 : (o.t - 1) / ft + 1) : o.t / ft;
    const int src = ravel(ig, ti, o.h / fh, o.w / fw) - x_base;
    y[e] = x[static_cast<long long>(src) * c + col];
}

template <class T>
__global__ __launch_bounds__(256) void pixel_shuffle(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int r1,
    int r2,
    int r3,
    int trim_t,
    int x_base,
    int y_base,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c) + y_base;
    const int col = static_cast<int>(e % c);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int ot = o.t + trim_t;
    const int src = ravel(ig, ot / r1, o.h / r2, o.w / r3) - x_base;
    const int block = ((ot % r1) * r2 + (o.h % r2)) * r3 + (o.w % r3);
    const int c_in = c * (r1 * r2 * r3);
    y[e] = x[static_cast<long long>(src) * c_in + col * (r1 * r2 * r3) + block];
}

template <class T>
__global__ __launch_bounds__(256) void pixel_unshuffle(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int r1,
    int r2,
    int r3,
    int x_base,
    int y_base,
    long long total)
{
    const int r = r1 * r2 * r3;
    const int c_out = c * r;
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c_out) + y_base;
    const int col = static_cast<int>(e % c_out);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int cin = col / r;
    const int block = col - cin * r;
    const int i1 = block / (r2 * r3);
    const int i2 = (block / r3) % r2;
    const int i3 = block % r3;
    const int src = ravel(ig, o.t * r1 + i1, o.h * r2 + i2, o.w * r3 + i3) - x_base;
    y[e] = x[static_cast<long long>(src) * c + cin];
}

template <class T>
__global__ __launch_bounds__(256) void avg_down(
    const T* __restrict__ x,
    const int* __restrict__ grid,
    T* __restrict__ y,
    const int* __restrict__ o_grid,
    int c,
    int lanes,
    int r1,
    int r2,
    int r3,
    int group,
    int x_base,
    int y_base,
    long long total)
{
    const int r = r1 * r2 * r3;
    const int c_out = c * r / group;
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c_out) + y_base;
    const int n = static_cast<int>(e % c_out);
    Lane og;
    const int l = lane_of(o_grid, lanes, row, og);
    if (l < 0) {
        y[e] = T{static_cast<unsigned short>(0)};
        return;
    }
    const Lane ig = lane_at(grid, l);
    const Voxel o = unravel(og, row - og.off);
    const int pad_t = (r1 - ig.t % r1) % r1;
    float acc = 0.f;
    for (int j = 0; j < group; ++j) {
        const int q = n * group + j;
        const int cin = q / r;
        const int block = q - cin * r;
        const int i1 = block / (r2 * r3);
        const int i2 = (block / r3) % r2;
        const int i3 = block % r3;
        const int ti = o.t * r1 + i1 - pad_t;
        if (ti < 0) continue;
        const int src = ravel(ig, ti, o.h * r2 + i2, o.w * r3 + i3) - x_base;
        acc += Elem<T>::to_f32(x[static_cast<long long>(src) * c + cin]);
    }
    y[e] = Elem<T>::from_f32(acc / static_cast<float>(group));
}

}
