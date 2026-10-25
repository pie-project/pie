#include <metal_simdgroup>
#include "../conv/grid.h"

template <typename T, int C_MAX, int SIMDS>
[[kernel]] void voxel_attention(
    const device T* q                 [[buffer(0)]],
    const device T* k                 [[buffer(1)]],
    const device T* v                 [[buffer(2)]],
    device T* y                       [[buffer(3)]],
    const device int* grid            [[buffer(4)]],
    const constant uint& channels     [[buffer(5)]],
    const constant uint& clips        [[buffer(6)]],
    const constant int& seg_frames    [[buffer(7)]],
    const constant float& sm_scale    [[buffer(8)]],
    const constant int& base          [[buffer(9)]],
    uint tgid                         [[threadgroup_position_in_grid]],
    uint simd_gid                     [[simdgroup_index_in_threadgroup]],
    uint simd_lid                     [[thread_index_in_simdgroup]]) {
  constexpr int VPT = C_MAX / 32;
  constexpr int THREADS = SIMDS * 32;
  constexpr float NEG_INF = -3.0e38f;

  threadgroup float q_s[C_MAX];
  threadgroup float wacc[SIMDS * C_MAX];
  threadgroup float wm[SIMDS];
  threadgroup float wl[SIMDS];
  threadgroup int span[2];

  const int row = int(tgid);
  const int lane = int(simd_lid);
  const int warp = int(simd_gid);
  const int tid = warp * 32 + lane;
  const uint c = channels;

  if (tid == 0) {
    SpatialClip g;
    const int l = clip_of(grid, int(clips), row + base, g);
    int begin = 0, end = 0;
    if (l >= 0) {
      g.off -= base;
      segment_of(g, row - g.off, seg_frames, begin, end);
    }
    span[0] = begin;
    span[1] = end;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int begin = span[0];
  const int end = span[1];
  device T* out = y + size_t(row) * size_t(c);
  if (end <= begin) {
    for (uint i = uint(tid); i < c; i += THREADS) out[i] = static_cast<T>(0.0f);
    return;
  }

  const device T* q_row = q + size_t(row) * size_t(c);
  for (uint i = uint(tid); i < c; i += THREADS) q_s[i] = float(q_row[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  thread float acc[VPT];
  for (int u = 0; u < VPT; ++u) acc[u] = 0.0f;
  float running_max = NEG_INF;
  float running_sum = 0.0f;

  for (int j = begin + warp; j < end; j += SIMDS) {
    const device T* k_row = k + size_t(j) * size_t(c);
    float dot = 0.0f;
    for (int u = 0; u < VPT; ++u) {
      const uint i = uint(lane + u * 32);
      if (i < c) dot += q_s[i] * float(k_row[i]);
    }
    dot = simd_sum(dot);

    const float score = dot * sm_scale;
    const float widened = max(running_max, score);
    const float rescale = fast::exp(running_max - widened);
    const float weight = fast::exp(score - widened);

    const device T* v_row = v + size_t(j) * size_t(c);
    for (int u = 0; u < VPT; ++u) {
      const uint i = uint(lane + u * 32);
      if (i < c) acc[u] = acc[u] * rescale + weight * float(v_row[i]);
    }
    running_sum = running_sum * rescale + weight;
    running_max = widened;
  }

  if (lane == 0) {
    wm[warp] = running_max;
    wl[warp] = running_sum;
  }
  for (int u = 0; u < VPT; ++u) {
    const uint i = uint(lane + u * 32);
    if (i < c) wacc[warp * C_MAX + i] = acc[u];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float folded_max = NEG_INF;
  for (int w = 0; w < SIMDS; ++w) folded_max = max(folded_max, wm[w]);
  float denominator = 0.0f;
  for (int w = 0; w < SIMDS; ++w) denominator += wl[w] * fast::exp(wm[w] - folded_max);
  const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;

  for (uint i = uint(tid); i < c; i += THREADS) {
    float sum = 0.0f;
    for (int w = 0; w < SIMDS; ++w) {
      sum += wacc[w * C_MAX + i] * fast::exp(wm[w] - folded_max);
    }
    out[i] = static_cast<T>(sum * inv);
  }
}

#define instantiate_voxel_attention(name, itype, cmax)                   \
  template [[host_name("voxel_attention_" #name "_c_" #cmax)]]           \
  [[kernel]] void voxel_attention<itype, cmax, 4>(                       \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const device int*, const constant uint&,              \
      const constant uint&, const constant int&, const constant float&,    \
      const constant int&, uint, uint, uint);

instantiate_voxel_attention(bfloat16, bfloat, 256)
instantiate_voxel_attention(bfloat16, bfloat, 512)
instantiate_voxel_attention(bfloat16, bfloat, 1024)
