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

template <typename X, typename T, bool SILU>
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
  out[idx] = T((outhat * float(w[lid])) * gate);
}

template <typename X, typename T, bool SILU>
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
  gated_rms_body<X, T, SILU>(
      x, z, w, out, eps, vd, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

template <typename X, typename T>
[[kernel]] void gated_rms_strided(
    const device X* x        [[buffer(0)]],
    const device T* z        [[buffer(1)]],
    const device X* w        [[buffer(2)]],
    device T* out            [[buffer(3)]],
    const constant float& eps  [[buffer(4)]],
    const constant uint& vd    [[buffer(5)]],
    const constant int& row_pitch [[buffer(6)]],
    uint3 tgpos       [[threadgroup_position_in_grid]],
    uint3 tpg         [[threadgroups_per_grid]],
    uint3 lid3        [[thread_position_in_threadgroup]],
    uint  simd_lane   [[thread_index_in_simdgroup]],
    uint  simd_group  [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv_rms[1], partials[32];
  const uint lid = lid3.x;
  const size_t idx =
      size_t(tgpos.z) * size_t(row_pitch) + size_t(tgpos.y) * vd + lid;
  (void)tpg;
  gated_rms_body<X, T, true>(
      x, z, w, out, eps, vd, idx, lid,
      inv_rms, partials, simd_lane, simd_group);
}

#define instantiate_gated_rms_strided(name, xtype, itype)         \
  template [[host_name("gated_rms_strided_" #name)]] [[kernel]] void \
  gated_rms_strided<xtype, itype>(                                \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      const constant int&, uint3, uint3, uint3, uint, uint);

instantiate_gated_rms_strided(bfloat16, bfloat, bfloat)

#define instantiate_gated_rms(name, xtype, itype)                 \
  template [[host_name("gated_rms_" #name)]]                      \
  [[kernel]] void gated_rms<xtype, itype, true>(                  \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

#define instantiate_gated_rms_by(name, xtype, itype)              \
  template [[host_name("gated_rms_by_" #name)]]                   \
  [[kernel]] void gated_rms<xtype, itype, false>(                 \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

instantiate_gated_rms(bfloat16, bfloat, bfloat)

instantiate_gated_rms(f32_bfloat16, float, bfloat)

instantiate_gated_rms_by(f32_bfloat16, float, bfloat)
