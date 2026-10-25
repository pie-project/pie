#include <metal_simdgroup>
#include "grid.h"


inline void welford_merge(
    thread float& cnt, thread float& mean, thread float& m2,
    float cnt_b, float mean_b, float m2_b) {
  const float n = cnt + cnt_b;
  if (n == 0.0f) return;
  const float delta = mean_b - mean;
  const float f = cnt_b / n;
  mean += delta * f;
  m2 += m2_b + delta * delta * cnt * f;
  cnt = n;
}

template <typename T>
[[kernel]] void group_norm_stats(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    device float4* partials           [[buffer(2)]],
    const constant uint& channels     [[buffer(3)]],
    const constant uint& groups       [[buffer(4)]],
    const constant uint& splits       [[buffer(5)]],
    const constant uint& base         [[buffer(6)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint3 lid3                        [[thread_position_in_threadgroup]],
    uint3 tg3                         [[threads_per_threadgroup]],
    uint simd_lid                     [[thread_index_in_simdgroup]],
    uint simd_gid                     [[simdgroup_index_in_threadgroup]]) {

  threadgroup float s_cnt[32], s_mean[32], s_m2[32];
  const uint lid = lid3.x;
  const uint tg_size = tg3.x;

  const uint split = tgid.x;
  const uint clip = tgid.y;
  const uint group = tgid.z;
  const SpatialClip g = clip_at(grid, int(clip));
  const uint n = uint(max(clip_voxels(g), 0));
  const uint chunk = (n + splits - 1) / splits;
  const uint begin = split * chunk;
  const uint end = min(n, begin + chunk);
  const uint cg = channels / groups;
  const uint c0 = group * cg;

  float cnt = 0.0f, mean = 0.0f, m2 = 0.0f;

  for (uint v = begin; v < end; ++v) {
    // `base`: the row `grid` addresses `x`'s row 0 at.
    const device T* row =
        x + (size_t(g.off - int(base)) + size_t(v)) * size_t(channels) + size_t(c0);
    for (uint c = lid; c < cg; c += tg_size) {
      const float value = float(row[c]);
      cnt += 1.0f;
      const float d = value - mean;
      mean += d / cnt;
      m2 += d * (value - mean);
    }
  }

  for (uint off = 16; off > 0; off >>= 1) {
    const float o_cnt = simd_shuffle_xor(cnt, off);
    const float o_mean = simd_shuffle_xor(mean, off);
    const float o_m2 = simd_shuffle_xor(m2, off);
    welford_merge(cnt, mean, m2, o_cnt, o_mean, o_m2);
  }
  if (simd_lid == 0) {
    s_cnt[simd_gid] = cnt;
    s_mean[simd_gid] = mean;
    s_m2[simd_gid] = m2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid != 0) return;
  const uint simds = (tg_size + 31u) / 32u;
  float tcnt = 0.0f, tmean = 0.0f, tm2 = 0.0f;
  for (uint s = 0; s < simds; ++s) {
    welford_merge(tcnt, tmean, tm2, s_cnt[s], s_mean[s], s_m2[s]);
  }
  partials[(size_t(clip) * size_t(splits) + size_t(split)) * size_t(groups) + size_t(group)] =
      float4(tcnt, tmean, tm2, 0.0f);
}

#define instantiate_gn_stats(name, itype)                                  \
  template [[host_name("group_norm_stats_" #name)]]                \
  [[kernel]] void group_norm_stats<itype>(                         \
      const device itype*, const device int*, device float4*,              \
      const constant uint&, const constant uint&, const constant uint&,    \
      const constant uint&, uint3, uint3, uint3, uint, uint);

instantiate_gn_stats(bfloat16, bfloat)

[[kernel]] void group_norm_finalize(
    const device float4* partials     [[buffer(0)]],
    device float2* stats              [[buffer(1)]],
    const constant uint& groups       [[buffer(2)]],
    const constant uint& splits       [[buffer(3)]],
    const constant float& eps         [[buffer(4)]],
    uint2 tgid                        [[threadgroup_position_in_grid]],
    uint simd_lid                     [[thread_index_in_simdgroup]]) {
  const uint group = tgid.x;
  const uint clip = tgid.y;
  float cnt = 0.0f, mean = 0.0f, m2 = 0.0f;
  for (uint s = simd_lid; s < splits; s += 32u) {
    const float4 p =
        partials[(size_t(clip) * size_t(splits) + size_t(s)) * size_t(groups) + size_t(group)];
    welford_merge(cnt, mean, m2, p.x, p.y, p.z);
  }
  for (uint off = 16; off > 0; off >>= 1) {
    const float o_cnt = simd_shuffle_xor(cnt, off);
    const float o_mean = simd_shuffle_xor(mean, off);
    const float o_m2 = simd_shuffle_xor(m2, off);
    welford_merge(cnt, mean, m2, o_cnt, o_mean, o_m2);
  }
  if (simd_lid == 0) {
    const float var = cnt > 0.0f ? m2 / cnt : 0.0f;
    stats[size_t(clip) * size_t(groups) + size_t(group)] =
        float2(mean, precise::rsqrt(var + eps));
  }
}

template <typename T>
[[kernel]] void group_norm_apply(
    const device T* x                 [[buffer(0)]],
    const device int* grid            [[buffer(1)]],
    const device float2* stats        [[buffer(2)]],
    const device float* weight        [[buffer(3)]],
    const device float* bias          [[buffer(4)]],
    device T* y                       [[buffer(5)]],
    const constant uint& channels     [[buffer(6)]],
    const constant uint& groups       [[buffer(7)]],
    const constant uint& clips        [[buffer(8)]],
    const constant uint& silu         [[buffer(9)]],
    const constant uint& base         [[buffer(10)]],
    uint2 tid                         [[thread_position_in_grid]]) {
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= channels) return;
  const size_t e = size_t(row) * size_t(channels) + size_t(col);
  SpatialClip box;
  const int l = clip_of(grid, int(clips), int(row + base), box);
  float v = 0.0f;
  if (l >= 0) {
    const uint group = col / (channels / groups);
    const float2 st = stats[size_t(l) * size_t(groups) + size_t(group)];
    v = fma((float(x[e]) - st.x) * st.y, weight[col], bias[col]);
    if (silu != 0) v = v / (1.0f + precise::exp(-v));
  }
  y[e] = static_cast<T>(v);
}

#define instantiate_gn_apply(name, itype)                                  \
  template [[host_name("group_norm_apply_" #name)]]                \
  [[kernel]] void group_norm_apply<itype>(                         \
      const device itype*, const device int*, const device float2*,        \
      const device float*, const device float*, device itype*,             \
      const constant uint&, const constant uint&, const constant uint&,    \
      const constant uint&, const constant uint&, uint2);

instantiate_gn_apply(bfloat16, bfloat)
