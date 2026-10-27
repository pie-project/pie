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

template <typename X, typename T, bool SILU, bool SCALED = true>
METAL_FUNC void gated_rms_body(
    const device X* x, const device T* z, const device X* w, device T* out,
    float eps, uint vd, size_t idx, uint lid,
    threadgroup float* inv_rms, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  const float xi = float(x[idx]);
  const float inv = rms_inv_from_lane_sum(
      xi * xi, vd, eps, inv_rms, partials, simd_lane, simd_group);
  const float zr = float(z[idx]);
  const float y = 1.0f / (1.0f + metal::exp(-metal::fabs(zr)));
  const float sig = zr < 0.0f ? 1.0f - y : y;
  const float gate = SILU ? zr * sig : sig;
  const float outhat = xi * inv;
  out[idx] = T((outhat * (SCALED ? float(w[lid]) : 1.0f)) * gate);
}

template <typename X, typename T, bool SILU, bool SCALED = true>
[[kernel]] void gated_rms(
    const device X* x        [[buffer(0)]],
    const device T* z        [[buffer(1)]],
    const device X* w        [[buffer(2)]],
    device T* out            [[buffer(3)]],
    const constant float& eps  [[buffer(4)]],
    const constant uint& vd    [[buffer(5)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z * tpg.y + tgpos.y) * vd + lid;
  gated_rms_body<X, T, SILU, SCALED>(
      x, z, w, out, eps, vd, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

#define instantiate_gated_rms(name, xtype, itype)                 \
  template [[host_name("gated_rms_" #name)]]                      \
  [[kernel]] void gated_rms<xtype, itype, true>(                  \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

#define instantiate_gated_rms_sigmoid(name, xtype, itype)         \
  template [[host_name("gated_rms_sigmoid_" #name)]]              \
  [[kernel]] void gated_rms<xtype, itype, false>(                 \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

instantiate_gated_rms(f32_bfloat16, float, bfloat)

instantiate_gated_rms_sigmoid(f32_bfloat16, float, bfloat)

// The scaleless gated norm (no weight read), over an f32 accumulator or
// bf16 rows (Muse Spark's).
#define instantiate_gated_rms_unscaled(name, xtype, itype, silu)  \
  template [[host_name(name)]]                                    \
  [[kernel]] void gated_rms<xtype, itype, silu, false>(           \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

instantiate_gated_rms_unscaled("gated_rms_unscaled_f32_bfloat16", float, bfloat, true)
instantiate_gated_rms_unscaled("gated_rms_unscaled_bfloat16_bfloat16", bfloat, bfloat, true)
instantiate_gated_rms_unscaled("gated_rms_sigmoid_unscaled_f32_bfloat16", float, bfloat, false)
instantiate_gated_rms_unscaled("gated_rms_sigmoid_unscaled_bfloat16_bfloat16", bfloat, bfloat, false)

// gated_rms (f32 rows, bf16 gate and output, SiLU gate) that also lays the
// register tile's input table of the rows it writes, so the out projection
// runs no table pass: a threadgroup per head and row lays its head's columns
// once the row's values are written, the first row's threadgroups lay the
// rows past `rows` as zeros, and the first threadgroup zeroes the split
// tile's `counters` arrival counters. Bit for bit `qmm_sg_table`'s table.
[[kernel]] void gated_rms_table_f32_bfloat16(
    const device float* x      [[buffer(0)]],
    const device bfloat* z     [[buffer(1)]],
    const device float* w      [[buffer(2)]],
    device bfloat* out         [[buffer(3)]],
    const constant float& eps  [[buffer(4)]],
    const constant uint& vd    [[buffer(5)]],
    device bfloat* table       [[buffer(6)]],
    const constant uint& rows  [[buffer(7)]],
    const constant uint& counters [[buffer(8)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z * tpg.y + tgpos.y) * vd + lid;
  gated_rms_body<float, bfloat, true>(
      x, z, w, out, eps, vd, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
  // The head's writes, seen by every thread of its threadgroup.
  threadgroup_barrier(mem_flags::mem_device);
  const uint width = tpg.y * vd, row = tgpos.z, from = tgpos.y * vd;
  const device bfloat* src = out + size_t(row) * width;
  table[register_table::slot(from + lid, row)] = src[from + lid];
  device float* sums = register_table::sums(table, width);
  if (lid < vd / 32) {
    const uint n = from / 32 + lid;
    sums[ulong(n) * 8 + row] = register_table::half_sum(src, n >> 1, n & 1);
  }
  if (row != 0)
    return;
  for (uint gone = rows; gone < 8; ++gone) {
    table[register_table::slot(from + lid, gone)] = bfloat(0.0f);
    if (lid < vd / 32)
      sums[ulong(from / 32 + lid) * 8 + gone] = 0.0f;
  }
  if (tgpos.y == 0) {
    device uint* count = reinterpret_cast<device uint*>(sums) + width / 4;
    for (uint n = lid; n < counters; n += vd)
      count[n] = 0;
  }
}

template <typename T, int N_READS>
[[kernel]] void vnorm_single_row(
    const device T* x        [[buffer(0)]],
    device T* out            [[buffer(1)]],
    const constant float& eps      [[buffer(2)]],
    const constant uint& axis_size [[buffer(3)]],
    uint gid                 [[threadgroup_position_in_grid]],
    uint lid                 [[thread_position_in_threadgroup]],
    uint simd_lane_id        [[thread_index_in_simdgroup]],
    uint simd_group_id       [[simdgroup_index_in_threadgroup]],
    uint tg_size             [[threads_per_threadgroup]]) {
  const uint span = tg_size * uint(N_READS);

  threadgroup float local_inv_rms[1];
  threadgroup float local_sums[32];

  const device T* xr = x + gid * size_t(axis_size);
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    acc += rms_lane_square_sum_at<T, N_READS>(xr + start, axis_size, start);
  }
  const float inv_rms = rms_inv_from_lane_sum(
      acc, axis_size, eps, local_inv_rms, local_sums,
      simd_lane_id, simd_group_id);

  device T* outr = out + gid * size_t(axis_size);
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        outr[start + uint(i)] = static_cast<T>(float(xr[start + uint(i)]) * inv_rms);
      }
    }
  }
}

#define instantiate_vnorm(name, itype, nreads)                         \
  template [[host_name("vnorm_single_row_" #name)]]                    \
  [[kernel]] void vnorm_single_row<itype, nreads>(                     \
      const device itype*, device itype*, const constant float&,       \
      const constant uint&, uint, uint, uint, uint, uint);

instantiate_vnorm(bfloat16, bfloat, 4)

METAL_FUNC float layernorm_group_sum(
    float acc, threadgroup float* out, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  acc = simd_sum(acc);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    acc = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) out[0] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out[0];
}

template <typename T>
[[kernel]] void layernorm(
    const device T* x              [[buffer(0)]],
    const device T* w              [[buffer(1)]],
    const device T* b              [[buffer(2)]],
    device T* out                  [[buffer(3)]],
    const constant float& eps      [[buffer(4)]],
    const constant uint& axis_size [[buffer(5)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
  threadgroup float reduced[1], partials[32];

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  device T* outr = out + row;

  float acc = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    acc += float(xr[i]);
  }
  const float mean =
      layernorm_group_sum(acc, reduced, partials, simd_lane, simd_group) /
      float(axis_size);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float spread = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = float(xr[i]) - mean;
    spread += c * c;
  }
  const float inv = precise::rsqrt(
      layernorm_group_sum(spread, reduced, partials, simd_lane, simd_group) /
          float(axis_size) +
      eps);

  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = (float(xr[i]) - mean) * inv;
    outr[i] = static_cast<T>(fma(c, float(w[i]), float(b[i])));
  }
}

#define instantiate_layernorm(name, itype)                            \
  template [[host_name("layernorm_" #name)]]                          \
  [[kernel]] void layernorm<itype>(                                   \
      const device itype*, const device itype*, const device itype*,  \
      device itype*, const constant float&, const constant uint&,     \
      uint, uint, uint, uint, uint);

instantiate_layernorm(bfloat16, bfloat)

template <typename T>
[[kernel]] void layernorm_no_scale(
    const device T* x              [[buffer(0)]],
    device T* out                  [[buffer(1)]],
    const constant float& eps      [[buffer(2)]],
    const constant uint& axis_size [[buffer(3)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
  threadgroup float reduced[1], partials[32];

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  device T* outr = out + row;

  float acc = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    acc += float(xr[i]);
  }
  const float mean =
      layernorm_group_sum(acc, reduced, partials, simd_lane, simd_group) /
      float(axis_size);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float spread = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = float(xr[i]) - mean;
    spread += c * c;
  }
  const float inv = precise::rsqrt(
      layernorm_group_sum(spread, reduced, partials, simd_lane, simd_group) /
          float(axis_size) +
      eps);

  for (uint i = lid; i < axis_size; i += tg_size) {
    outr[i] = static_cast<T>((float(xr[i]) - mean) * inv);
  }
}

#define instantiate_layernorm_no_scale(name, itype)                   \
  template [[host_name("layernorm_no_scale_" #name)]]                 \
  [[kernel]] void layernorm_no_scale<itype>(                          \
      const device itype*, device itype*,                             \
      const constant float&, const constant uint&,                    \
      uint, uint, uint, uint, uint);

instantiate_layernorm_no_scale(bfloat16, bfloat)
instantiate_layernorm_no_scale(float32, float)

template <typename T>
[[kernel]] void standardize(
    device T* out             [[buffer(0)]],
    const device T* bias      [[buffer(1)]],
    const device T* scale     [[buffer(2)]],
    const constant int& width [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(width) + size_t(tid.x);
  out[i] = T((float(out[i]) - float(bias[tid.x])) * float(scale[tid.x]));
}

#define instantiate_standardize(name, itype)                          \
  template [[host_name("standardize_" #name)]]                        \
  [[kernel]] void standardize<itype>(                                 \
      device itype*, const device itype*, const device itype*,        \
      const constant int&, uint2);

instantiate_standardize(bfloat16, bfloat)

// Muse Spark's post-norm and residual gate (`elemwise.norm_gate`), a
// threadgroup a row: `u = x * scale` rounded to T (`x` with no scale), `n =
// u / rms(u)` rounded to T, and with a residual `y = alpha * res + beta * n`
// in f32 (`w = gate / temperature`, `beta = sigmoid(w)`, `alpha =
// sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`), else `y = n`. Under `NEXT`
// the next sublayer's input norm over `y` too: `z = rmsnorm(y) * w` (`1 + w`
// where `plus_one`). CUDA's `norm_gate`, value for value.
template <typename T>
METAL_FUNC float norm_gate_u(
    const device T* x, const device T* scale, uint has_scale, size_t at, uint i) {
  const float v = float(x[at]);
  return has_scale != 0 ? float(T(v * float(scale[i]))) : v;
}

template <typename T, bool NEXT>
[[kernel]] void norm_gate(
    const device T* x              [[buffer(0)]],
    const device T* scale          [[buffer(1)]],
    const device T* res            [[buffer(2)]],
    const device float* gate       [[buffer(3)]],
    device T* y                    [[buffer(4)]],
    const constant float& eps      [[buffer(5)]],
    const constant float& temperature [[buffer(6)]],
    const constant uint& width     [[buffer(7)]],
    const constant uint& has_scale [[buffer(8)]],
    const constant uint& has_res   [[buffer(9)]],
    const device T* next_w         [[buffer(10)]],
    const constant float& next_eps [[buffer(11)]],
    const constant uint& plus_one  [[buffer(12)]],
    device T* z                    [[buffer(13)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const size_t base = size_t(gid) * width;
  float acc = 0.0f;
  for (uint i = lid; i < width; i += tg_size) {
    const float u = norm_gate_u(x, scale, has_scale, base + i, i);
    acc = fma(u, u, acc);
  }
  const float inv = rms_inv_from_lane_sum(
      acc, width, eps, inv_rms, partials, simd_lane, simd_group);
  float sq = 0.0f;
  for (uint i = lid; i < width; i += tg_size) {
    const float u = norm_gate_u(x, scale, has_scale, base + i, i);
    T o;
    if (has_res == 0) {
      o = T(u * inv);
    } else {
      const float n = float(T(u * inv));
      const float w = gate[i] / temperature;
      const float beta = 1.0f / (1.0f + metal::exp(-w));
      const float a2 = metal::max((1.0f / (1.0f + metal::exp(w))) * (1.0f + beta), 1e-3f);
      const float alpha = a2 * metal::rsqrt(a2);
      o = T(alpha * float(res[base + i]) + beta * n);
    }
    y[base + i] = o;
    if (NEXT) {
      const float of = float(o);
      sq += of * of;
    }
  }
  if (!NEXT)
    return;
  const float inv2 = rms_inv_from_lane_sum(
      sq, width, next_eps, inv_rms, partials, simd_lane, simd_group);
  for (uint i = lid; i < width; i += tg_size) {
    float wv = float(next_w[i]);
    if (plus_one != 0) wv += 1.0f;
    z[base + i] = T(float(y[base + i]) * inv2 * wv);
  }
}

#define instantiate_norm_gate(name, itype, next)                          \
  template [[host_name(name)]] [[kernel]] void norm_gate<itype, next>(     \
      const device itype*, const device itype*, const device itype*,       \
      const device float*, device itype*, const constant float&,           \
      const constant float&, const constant uint&, const constant uint&,   \
      const constant uint&, const device itype*, const constant float&,    \
      const constant uint&, device itype*, uint, uint, uint, uint, uint);

instantiate_norm_gate("norm_gate_bfloat16", bfloat, false)
instantiate_norm_gate("norm_gate_next_bfloat16", bfloat, true)
instantiate_norm_gate("norm_gate_float16", half, false)
instantiate_norm_gate("norm_gate_next_float16", half, true)
