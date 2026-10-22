

#include <metal_stdlib>
#include "../prim/register_table.h"

using namespace metal;

// Row `row`'s output at channel `c`: its lane's span [indptr[lo], indptr[lo + 1]) holds the row,
// and each tap reaches back no further than the lane's first row.
template <typename T>
inline float block_dyn_conv_at(const device T* x, const device int* indptr, const device T* coeff,
                               const device T* base, int channels, int side, int taps, int group,
                               int lanes, int row, int c) {
  int lo = 0, hi = lanes;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (indptr[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const int begin = indptr[lo];
  const int t = row - begin;
  const int groups = channels / group;
  const int g = c / group;
  const size_t chans = size_t(channels);
  const size_t pitch = size_t(2 * taps) * size_t(groups);

  float acc = 0.0f;
  for (int k = 0; k < taps; ++k) {
    const int src = t - k;
    if (src < 0) {
      break;
    }
    const int at = side * taps + k;
    const float coef = float(base[size_t(at) * chans + size_t(c)])
                     + float(coeff[size_t(row) * pitch + size_t(at) * size_t(groups) + size_t(g)]);
    acc += coef * float(x[size_t(begin + src) * chans + size_t(c)]);
  }
  return acc;
}

// A thread per channel and row. A row past every lane is left alone.
template <typename T>
[[kernel]] void block_dyn_conv(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* coeff          [[buffer(2)]],
    const device T* base           [[buffer(3)]],
    device T* y                    [[buffer(4)]],
    const constant int& channels   [[buffer(5)]],
    const constant int& side       [[buffer(6)]],
    const constant int& taps       [[buffer(7)]],
    const constant int& group      [[buffer(8)]],
    const constant int& lanes      [[buffer(9)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int row = int(pos.y);
  if (c >= channels || row < indptr[0] || row >= indptr[lanes]) {
    return;
  }
  y[size_t(row) * size_t(channels) + size_t(c)] =
      T(block_dyn_conv_at(x, indptr, coeff, base, channels, side, taps, group, lanes, row, c));
}

// [`block_dyn_conv`] over at most eight rows, a threadgroup a row of the register tile's input
// table (prim/register_table.h) for the projections of `y`, which it lays as it writes
// them: a row past `rows` as zeros, one inside them past every lane as `y` holds it.
[[kernel]] void block_dyn_conv_table(
    const device bfloat* x         [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device bfloat* coeff     [[buffer(2)]],
    const device bfloat* base      [[buffer(3)]],
    device bfloat* y               [[buffer(4)]],
    const constant int& channels   [[buffer(5)]],
    const constant int& side       [[buffer(6)]],
    const constant int& taps       [[buffer(7)]],
    const constant int& group      [[buffer(8)]],
    const constant int& lanes      [[buffer(9)]],
    device bfloat* table           [[buffer(10)]],
    const constant uint& rows      [[buffer(11)]],
    const constant uint& counters  [[buffer(12)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 threads2 [[threads_per_threadgroup]]) {
  threadgroup bfloat values[8192];
  const uint lid = lid2.x, threads = threads2.x, width = uint(channels);
  const int row = int(tile.y);
  const bool live = uint(row) < rows;
  const bool walked = live && row >= indptr[0] && row < indptr[lanes];
  const size_t at = size_t(row) * size_t(width);
  for (uint k = lid; k < width; k += threads) {
    bfloat v = bfloat(0.0f);
    if (walked) {
      v = bfloat(block_dyn_conv_at(x, indptr, coeff, base, channels, side, taps, group, lanes, row,
                                   int(k)));
      y[at + k] = v;
    } else if (live) {
      v = y[at + k];
    }
    table[register_table::slot(k, uint(row))] = v;
    values[k] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup bfloat* laid = values;
  register_table::lay_sums(laid, table, width, uint(row), lid, threads);
  if (row == 0)
    register_table::lay_rest(table, width, 8, counters, lid, threads);
}

#define instantiate_block_dyn_conv(name, itype)                         \
  template [[host_name("block_dyn_conv_" #name)]]                       \
  [[kernel]] void block_dyn_conv<itype>(                                \
      const device itype*, const device int*, const device itype*,      \
      const device itype*, device itype*, const constant int&,          \
      const constant int&, const constant int&, const constant int&,    \
      const constant int&, uint2);

instantiate_block_dyn_conv(bfloat16, bfloat)
