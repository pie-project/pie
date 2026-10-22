#include <metal_stdlib>
#include "../prim/register_table.h"
using namespace metal;

template <typename T, int N_READS>
METAL_FUNC float rms_lane_square_sum_at(
    const device T* x, uint axis_size, uint start) {
  float acc = 0.0f;
  if (start + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; ++i) {
      const float xi = float(x[i]);
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; ++i) {
      if (start + uint(i) < axis_size) {
        const float xi = float(x[i]);
        acc += xi * xi;
      }
    }
  }
  return acc;
}

METAL_FUNC float rms_inv_from_lane_sum(
    float acc, uint axis_size, float eps,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  acc = simd_sum(acc);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    acc = simd_sum(partials[simd_lane]);
    if (simd_lane == 0)
      inv_rms[0] = precise::rsqrt(acc / float(axis_size) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return inv_rms[0];
}

template <typename T, int N_READS>
METAL_FUNC void rms_row_body(
    const device T* x, const device T* w, device T* out,
    float eps, uint axis_size, uint w_stride, uint plus_one, float gain,
    size_t row_base,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint lid, uint simd_lane, uint simd_group, uint tg_size) {
  const uint span = tg_size * uint(N_READS);

  const device T* xr = x + row_base;
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    acc += rms_lane_square_sum_at<T, N_READS>(xr + start, axis_size, start);
  }
  const float inv = rms_inv_from_lane_sum(
      acc, axis_size, eps, inv_rms, partials, simd_lane, simd_group);

  device T* outr = out + row_base;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    const device T* xs = xr + start;
    const device T* ws = w + w_stride * start;
    device T* os = outr + start;
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        T wv = T(gain * (plus_one ? (1.0f + float(ws[w_stride * i]))
                                  : float(ws[w_stride * i])));
        os[i] = wv * static_cast<T>(xs[i] * inv);
      }
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void rms_single_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid) * axis_size,
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
}

template <typename T, int N_READS>
METAL_FUNC void residual_add_rms_row(
    const device T* x, device T* y, const device T* w, device T* out,
    float eps, uint axis_size, uint w_stride, uint plus_one, float gain, uint gid,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint lid, uint simd_lane, uint simd_group, uint tg_size) {
  const size_t row_base = size_t(gid) * axis_size;
  const uint span = tg_size * uint(N_READS);
  const device T* xr = x + row_base;
  device T* yr = y + row_base;
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    for (int i = 0; i < N_READS; i++) {
      const uint at = start + uint(i);
      if (at < axis_size) {
        const T folded = T(float(xr[at]) + float(yr[at]));
        yr[at] = folded;
        const float f = float(folded);
        acc += f * f;
      }
    }
  }
  const float inv = rms_inv_from_lane_sum(
      acc, axis_size, eps, inv_rms, partials, simd_lane, simd_group);
  device T* outr = out + row_base;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    const device T* ys = yr + start;
    const device T* ws = w + w_stride * start;
    device T* os = outr + start;
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        T wv = T(gain * (plus_one ? (1.0f + float(ws[w_stride * i]))
                                  : float(ws[w_stride * i])));
        os[i] = wv * static_cast<T>(ys[i] * inv);
      }
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void residual_add_rms_single_row(
    const device T* x          [[buffer(0)]],
    device T* y                [[buffer(1)]],
    const device T* w          [[buffer(2)]],
    device T* out              [[buffer(3)]],
    const constant float& eps  [[buffer(4)]],
    const constant uint& axis_size [[buffer(5)]],
    const constant uint& w_stride  [[buffer(6)]],
    const constant uint& plus_one  [[buffer(7)]],
    const constant float& gain     [[buffer(8)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const size_t row_base = size_t(gid) * axis_size;
  const uint span = tg_size * uint(N_READS);
  const device T* xr = x + row_base;
  device T* yr = y + row_base;
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    for (int i = 0; i < N_READS; i++) {
      const uint at = start + uint(i);
      if (at < axis_size) {
        const T folded = T(float(xr[at]) + float(yr[at]));
        yr[at] = folded;
        const float f = float(folded);
        acc += f * f;
      }
    }
  }
  const float inv = rms_inv_from_lane_sum(
      acc, axis_size, eps, inv_rms, partials, simd_lane, simd_group);
  device T* outr = out + row_base;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    const device T* ys = yr + start;
    const device T* ws = w + w_stride * start;
    device T* os = outr + start;
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        T wv = T(gain * (plus_one ? (1.0f + float(ws[w_stride * i]))
                                  : float(ws[w_stride * i])));
        os[i] = wv * static_cast<T>(ys[i] * inv);
      }
    }
  }
}

// Lays row `row` of the register tile's input table from the normed row this
// threadgroup just wrote at `outr`; the first row's threadgroup lays the rows
// past `rows` as zeros and zeroes the split tile's `counters` arrival counters.
METAL_FUNC void rms_lay_register_table(
    const device bfloat* outr, device bfloat* table, uint width, uint row,
    uint rows, uint counters, uint lid, uint tg_size) {
  // The row's writes, seen by every thread of its threadgroup.
  threadgroup_barrier(mem_flags::mem_device);
  for (uint k = lid; k < width; k += tg_size)
    table[register_table::slot(k, row)] = outr[k];
  register_table::lay_sums(outr, table, width, row, lid, tg_size);
  if (row == 0)
    register_table::lay_rest(table, width, rows, counters, lid, tg_size);
}

// `residual_add_rms_single_row` that also lays the register tile's input table
// of the normed rows (at most eight), so the projections that read them run
// no table pass of their own.
[[kernel]] void residual_add_rms_table_bfloat16(
    const device bfloat* x     [[buffer(0)]],
    device bfloat* y           [[buffer(1)]],
    const device bfloat* w     [[buffer(2)]],
    device bfloat* out         [[buffer(3)]],
    const constant float& eps  [[buffer(4)]],
    const constant uint& axis_size [[buffer(5)]],
    const constant uint& w_stride  [[buffer(6)]],
    const constant uint& plus_one  [[buffer(7)]],
    const constant float& gain     [[buffer(8)]],
    device bfloat* table       [[buffer(9)]],
    const constant uint& rows  [[buffer(10)]],
    const constant uint& counters [[buffer(11)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  residual_add_rms_row<bfloat, 4>(
      x, y, w, out, eps, axis_size, w_stride, plus_one, gain, gid,
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
  rms_lay_register_table(
      out + size_t(gid) * axis_size, table, axis_size, gid, rows, counters, lid, tg_size);
}

#define instantiate_residual_add_rms_single_row(name, itype, n_reads)  \
  template [[host_name("residual_add_rms_single_row_" #name)]]          \
  [[kernel]] void residual_add_rms_single_row<itype, n_reads>(          \
      const device itype*, device itype*, const device itype*,          \
      device itype*, const constant float&, const constant uint&,       \
      const constant uint&, const constant uint&, const constant float&, \
      uint, uint, uint, uint, uint);

instantiate_residual_add_rms_single_row(bfloat16, bfloat, 4)

#define instantiate_rms_single_row(name, itype, n_reads)               \
  template [[host_name("rms_single_row_" #name)]] [[kernel]] void       \
  rms_single_row<itype, n_reads>(                                       \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&,                       \
      uint, uint, uint, uint, uint);

instantiate_rms_single_row(bfloat16, bfloat, 4)

instantiate_rms_single_row(float32, float, 4)

template <typename T, int N_READS>
[[kernel]] void rms_grouped_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const constant uint& groups    [[buffer(8)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w + size_t(gid % groups) * size_t(axis_size) * size_t(w_stride), out,
      eps, axis_size, w_stride, plus_one, gain,
      size_t(gid) * size_t(axis_size),
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
}

// Two `rms_single_row` launches in one: the `first` threadgroups norm x into out by w, the rest
// x2 into out2 by w2 (a layer's query and key heads), each row as its own launch norms it.
template <typename T, int N_READS>
[[kernel]] void rms_single_row_pair(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const device T* x2         [[buffer(3)]],
    const device T* w2         [[buffer(4)]],
    device T* out2             [[buffer(5)]],
    const constant uint& first [[buffer(6)]],
    const constant float& eps  [[buffer(7)]],
    const constant uint& axis_size [[buffer(8)]],
    const constant uint& w_stride  [[buffer(9)]],
    const constant uint& plus_one  [[buffer(10)]],
    const constant float& gain     [[buffer(11)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const bool second = gid >= first;
  rms_row_body<T, N_READS>(
      second ? x2 : x, second ? w2 : w, second ? out2 : out, eps, axis_size, w_stride, plus_one,
      gain, size_t(second ? gid - first : gid) * axis_size,
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
}

template [[host_name("rms_single_row_pair_bfloat16")]] [[kernel]] void rms_single_row_pair<bfloat, 4>(
    const device bfloat*, const device bfloat*, device bfloat*, const device bfloat*,
    const device bfloat*, device bfloat*, const constant uint&, const constant float&,
    const constant uint&, const constant uint&, const constant uint&, const constant float&,
    uint, uint, uint, uint, uint);

#define instantiate_rms_grouped_row(name, itype, n_reads)              \
  template [[host_name("rms_grouped_row_" #name)]] [[kernel]] void      \
  rms_grouped_row<itype, n_reads>(                                      \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant uint&, \
      uint, uint, uint, uint, uint);

instantiate_rms_grouped_row(bfloat16, bfloat, 4)

// `rms_single_row` over whole rows that also lays the register tile's input
// table of the normed rows (at most eight), so the projections that read them
// run no table pass of their own.
[[kernel]] void rms_table_bfloat16(
    const device bfloat* x     [[buffer(0)]],
    const device bfloat* w     [[buffer(1)]],
    device bfloat* out         [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    device bfloat* table       [[buffer(8)]],
    const constant uint& rows  [[buffer(9)]],
    const constant uint& counters [[buffer(10)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<bfloat, 4>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid) * axis_size,
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
  rms_lay_register_table(
      out + size_t(gid) * axis_size, table, axis_size, gid, rows, counters, lid, tg_size);
}
