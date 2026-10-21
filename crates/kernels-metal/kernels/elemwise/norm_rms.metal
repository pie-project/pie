#include <metal_stdlib>
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

// The register tile's input table (`qmm_sg_table` in linear/quant_qmm_sg.metal)
// of the rows this threadgroup's row is one of, laid from the normed row it
// just wrote: each of the row's inputs at its lane slot, each 32 inputs' sum
// added in the very order `qmm_sg_table` adds them, the rows past `rows`
// zeroed by the first row's threadgroup, as are the split tile's `counters`
// arrival counters. Bit for bit the table `qmm_sg_table` lays over `out`.
METAL_FUNC void rms_lay_register_table(
    const device bfloat* outr, device bfloat* table, uint width, uint row,
    uint rows, uint counters, uint lid, uint tg_size) {
  // The sums add in `qmm_sg_table`'s order, not one the compiler picks.
#pragma clang fp reassociate(off)
  // The row's writes, seen by every thread of its threadgroup.
  threadgroup_barrier(mem_flags::mem_device);
  device float* sums = reinterpret_cast<device float*>(table + 8 * ulong(width));
  const uint c = row >> 1, r = row & 1;
  // A thread per 32 inputs: group h = at / 2, half q = at % 2.
  for (uint at = lid; at < width / 32; at += tg_size) {
    const uint h = at >> 1, q = at & 1;
    float part[2][2];
    for (uint cl = 0; cl < 2; ++cl) {
      const uint cp = 2 * q + cl;
      for (uint i = 0; i < 2; ++i) {
        const uint fm = 2 * cp + i;
        float sum = 0.0f;
        for (uint j = 0; j < 2; ++j)
          for (uint p = 0; p < 4; ++p) {
            const uint k = 64 * h + 8 * (2 * cp + j) + p + 4 * i;
            const bfloat v = outr[k];
            table[(ulong(h) * 32 + fm * 4 + c) * 16 + 8 * j + 2 * p + r] = v;
            sum += float(v);
          }
        part[cl][i] = sum;
      }
    }
    sums[(2 * ulong(h) + q) * 8 + row] = (part[0][0] + part[0][1]) + (part[1][0] + part[1][1]);
  }
  if (row != 0)
    return;
  for (uint gone = rows; gone < 8; ++gone) {
    const uint gc = gone >> 1, gr = gone & 1;
    for (uint at = lid; at < width / 32; at += tg_size) {
      const uint h = at >> 1, q = at & 1;
      for (uint cl = 0; cl < 2; ++cl)
        for (uint i = 0; i < 2; ++i)
          for (uint j = 0; j < 2; ++j)
            for (uint p = 0; p < 4; ++p)
              table[(ulong(h) * 32 + (2 * (2 * q + cl) + i) * 4 + gc) * 16 + 8 * j + 2 * p + gr] =
                  bfloat(0.0f);
      sums[(2 * ulong(h) + q) * 8 + gone] = 0.0f;
    }
  }
  device uint* count = reinterpret_cast<device uint*>(sums) + width / 4;
  for (uint at = lid; at < counters; at += tg_size)
    count[at] = 0;
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

template <typename T, int N_READS>
[[kernel]] void rms_strided_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const constant int& row_pitch  [[buffer(8)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid) * size_t(row_pitch),
      inv_rms, partials, lid, simd_lane, simd_group, tg_size);
}

#define instantiate_rms_strided_row(name, itype, n_reads)              \
  template [[host_name("rms_strided_row_" #name)]] [[kernel]] void      \
  rms_strided_row<itype, n_reads>(                                      \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant int&,  \
      uint, uint, uint, uint, uint);

instantiate_rms_strided_row(bfloat16, bfloat, 4)

template <typename T, int N_READS>
[[kernel]] void rms_strided_head_row(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const constant int& row_pitch  [[buffer(8)]],

    uint3 gid                  [[threadgroup_position_in_grid]],
    uint3 lid                  [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]],
    uint3 tg_size              [[threads_per_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  rms_row_body<T, N_READS>(
      x, w, out, eps, axis_size, w_stride, plus_one, gain,
      size_t(gid.z) * size_t(row_pitch) + size_t(gid.y) * axis_size,
      inv_rms, partials, lid.x, simd_lane, simd_group, tg_size.x);
}

#define instantiate_rms_strided_head_row(name, itype, n_reads)         \
  template [[host_name("rms_strided_head_row_" #name)]] [[kernel]] void \
  rms_strided_head_row<itype, n_reads>(                                 \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant int&,  \
      uint3, uint3, uint, uint, uint3);

instantiate_rms_strided_head_row(bfloat16, bfloat, 4)

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

#define instantiate_rms_grouped_row(name, itype, n_reads)              \
  template [[host_name("rms_grouped_row_" #name)]] [[kernel]] void      \
  rms_grouped_row<itype, n_reads>(                                      \
      const device itype*, const device itype*, device itype*,          \
      const constant float&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant uint&, \
      uint, uint, uint, uint, uint);

instantiate_rms_grouped_row(bfloat16, bfloat, 4)

template <typename T, int N_READS, bool SCALED>
METAL_FUNC void rms_residual_impl(
    const device T* x,
    const device T* w,
    const device T* r,
    const device T* s,
    device T* out,
    float eps, uint axis_size, uint w_stride, uint plus_one, float gain,
    threadgroup float* local_inv_mean,
    threadgroup float* local_sums,
    uint gid,
    uint lid,
    uint simd_lane_id,
    uint simd_group_id,
    uint tg_size) {
  const uint span = tg_size * uint(N_READS);

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  float acc = 0.0f;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    acc += rms_lane_square_sum_at<T, N_READS>(xr + start, axis_size, start);
  }
  const float inv = rms_inv_from_lane_sum(
      acc, axis_size, eps, local_inv_mean, local_sums,
      simd_lane_id, simd_group_id);

  const float scale = SCALED ? float(s[0]) : 1.0f;
  const device T* rr = r + row;
  device T* outr = out + row;
  for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
    for (int i = 0; i < N_READS; i++) {
      if (start + uint(i) < axis_size) {
        const float wv = gain * (plus_one ? (1.0f + float(w[w_stride * (start + uint(i))]))
                                          : float(w[w_stride * (start + uint(i))]));
        const float normed = wv * (float(xr[start + uint(i)]) * inv);
        outr[start + uint(i)] = static_cast<T>((normed + float(rr[start + uint(i)])) * scale);
      }
    }
  }
}

template <typename T, int N_READS>
[[kernel]] void rms_residual(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const device T* r          [[buffer(8)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  rms_residual_impl<T, N_READS, false>(x, w, r, nullptr, out,
                                       eps, axis_size, w_stride, plus_one, gain,
                                       local_inv_mean, local_sums,
                                       gid, lid, simd_lane_id, simd_group_id, tg_size);
}

template <typename T, int N_READS>
[[kernel]] void rms_residual_scaled(
    const device T* x          [[buffer(0)]],
    const device T* w          [[buffer(1)]],
    device T* out              [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    const constant uint& axis_size [[buffer(4)]],
    const constant uint& w_stride  [[buffer(5)]],
    const constant uint& plus_one  [[buffer(6)]],
    const constant float& gain     [[buffer(7)]],
    const device T* r          [[buffer(8)]],
    const device T* s          [[buffer(9)]],
    uint gid                   [[threadgroup_position_in_grid]],
    uint lid                   [[thread_position_in_threadgroup]],
    uint simd_lane_id          [[thread_index_in_simdgroup]],
    uint simd_group_id         [[simdgroup_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  rms_residual_impl<T, N_READS, true>(x, w, r, s, out,
                                      eps, axis_size, w_stride, plus_one, gain,
                                      local_inv_mean, local_sums,
                                      gid, lid, simd_lane_id, simd_group_id, tg_size);
}

#define instantiate_rms_residual(name, itype, nreads)                    \
  template [[host_name("rms_residual_" #name)]]                          \
  [[kernel]] void rms_residual<itype, nreads>(                           \
      const device itype*, const device itype*, device itype*,           \
      const constant float&, const constant uint&, const constant uint&,  \
      const constant uint&, const constant float&, const device itype*,   \
      uint, uint, uint, uint, uint);                                           \
  template [[host_name("rms_residual_scaled_" #name)]]                   \
  [[kernel]] void rms_residual_scaled<itype, nreads>(                    \
      const device itype*, const device itype*, device itype*,           \
      const constant float&, const constant uint&, const constant uint&,  \
      const constant uint&, const constant float&, const device itype*,   \
      const device itype*, uint, uint, uint, uint, uint);

instantiate_rms_residual(bfloat16, bfloat, 4)
