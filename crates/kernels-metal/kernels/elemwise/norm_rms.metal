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
