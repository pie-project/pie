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

#define instantiate_gated_rms_sigmoid(name, xtype, itype)         \
  template [[host_name("gated_rms_sigmoid_" #name)]]              \
  [[kernel]] void gated_rms<xtype, itype, false>(                 \
      const device xtype*, const device itype*, const device xtype*, \
      device itype*, const constant float&, const constant uint&,    \
      uint3, uint3, uint3, uint, uint);

instantiate_gated_rms(bfloat16, bfloat, bfloat)

instantiate_gated_rms(f32_bfloat16, float, bfloat)

instantiate_gated_rms_sigmoid(f32_bfloat16, float, bfloat)

// One 32-input half (`q` of 64-input group `h`) of row `row` of the register
// tile's input table (`qmm_sg_table` in linear/quant_qmm_sg.metal), from that
// row's inputs at `src` (indexed by input): each input at its lane slot, the
// 32 summed in the order `qmm_sg_table` adds them. A row past the rows the
// matmul reads is laid as zeros (`live` false).
METAL_FUNC void lay_register_half(
    const device bfloat* src, bool live, device bfloat* table, uint width, uint row,
    uint h, uint q) {
  // The sums add in `qmm_sg_table`'s order, not one the compiler picks.
#pragma clang fp reassociate(off)
  const uint c = row >> 1, r = row & 1;
  float part[2][2];
  for (uint cl = 0; cl < 2; ++cl) {
    const uint cp = 2 * q + cl;
    for (uint i = 0; i < 2; ++i) {
      const uint fm = 2 * cp + i;
      float sum = 0.0f;
      for (uint j = 0; j < 2; ++j)
        for (uint p = 0; p < 4; ++p) {
          const uint k = 64 * h + 8 * (2 * cp + j) + p + 4 * i;
          const bfloat v = live ? src[k] : bfloat(0.0f);
          table[(ulong(h) * 32 + fm * 4 + c) * 16 + 8 * j + 2 * p + r] = v;
          sum += float(v);
        }
      part[cl][i] = sum;
    }
  }
  device float* sums = reinterpret_cast<device float*>(table + 8 * ulong(width));
  sums[(2 * ulong(h) + q) * 8 + row] = (part[0][0] + part[0][1]) + (part[1][0] + part[1][1]);
}

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
  const uint width = tpg.y * vd, row = tgpos.z;
  const device bfloat* src = out + size_t(row) * width;
  for (uint t = lid; t < vd / 32; t += vd) {
    const uint at = tgpos.y * vd + 32 * t;
    lay_register_half(src, true, table, width, row, at / 64, (at / 32) & 1);
    if (row == 0)
      for (uint gone = rows; gone < 8; ++gone)
        lay_register_half(src, false, table, width, gone, at / 64, (at / 32) & 1);
  }
  if (row == 0 && tgpos.y == 0) {
    device uint* count = reinterpret_cast<device uint*>(table + 8 * ulong(width)) + width / 4;
    for (uint n = lid; n < counters; n += vd)
      count[n] = 0;
  }
}
