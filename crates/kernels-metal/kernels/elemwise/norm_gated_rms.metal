#include <metal_stdlib>
#include "../linear/register_table.h"
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
