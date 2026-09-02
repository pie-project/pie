#include <metal_stdlib>
using namespace metal;

/// **ONE ROW'S ARGMAX, WRITTEN INTO ONE COLUMN OF AN I32 PLANE.**
///
/// `y[row * depth + column] = argmax_c x[row, c]`, one threadgroup per row.
/// Every thread scans a strided share of the row keeping its best `(value,
/// index)`, the simdgroups fold with `simd_max` and a `simd_min` over the
/// indices that hit it, and thread 0 folds the simdgroups. Ties go to the
/// LOWEST column and a NaN never wins — the epilogue's `reduce_argmax` rule,
/// stated here so a draft the head chained on is the token the verifier
/// reads back from the same logits.
constant constexpr uint kArgmaxSimdgroups = 32;
constant constexpr float NEG_INF_F = -INFINITY;

template <typename T>
[[kernel]] void argmax_rows(
    const device T* x          [[buffer(0)]],
    device int* y              [[buffer(1)]],
    const constant int& width  [[buffer(2)]],
    const constant int& depth  [[buffer(3)]],
    const constant int& column [[buffer(4)]],
    uint2 tid                  [[thread_position_in_grid]],
    uint2 tid_group            [[thread_position_in_threadgroup]],
    uint2 group_size           [[threads_per_threadgroup]],
    uint simd_lid              [[thread_index_in_simdgroup]],
    uint simd_gid              [[simdgroup_index_in_threadgroup]]) {
  const uint lid = tid_group.x;
  const uint threads = group_size.x;
  const size_t row = size_t(tid.y);
  const device T* src = x + row * size_t(width);

  float best = NEG_INF_F;
  uint best_i = 0xFFFFFFFFu;
  for (uint c = lid; c < uint(width); c += threads) {
    const float v = float(src[c]);
    if (!isnan(v) && (v > best || (v == best && c < best_i))) {
      best = v;
      best_i = c;
    }
  }

  threadgroup float part_v[kArgmaxSimdgroups];
  threadgroup uint part_i[kArgmaxSimdgroups];
  const float m = simd_max(best);
  const uint w = simd_min(best == m ? best_i : 0xFFFFFFFFu);
  if (simd_lid == 0) {
    part_v[simd_gid] = m;
    part_i[simd_gid] = w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    const uint n_simd = (threads + 31u) / 32u;
    float top = NEG_INF_F;
    uint top_i = 0xFFFFFFFFu;
    for (uint sg = 0; sg < n_simd; ++sg) {
      const float v = part_v[sg];
      const uint i = part_i[sg];
      if (i != 0xFFFFFFFFu && (v > top || (v == top && i < top_i))) {
        top = v;
        top_i = i;
      }
    }
    y[row * size_t(depth) + size_t(column)] = int(top_i == 0xFFFFFFFFu ? 0u : top_i);
  }
}

#define instantiate_argmax_rows(name, itype)                                \
  template [[host_name("argmax_rows_" #name)]]                              \
  [[kernel]] void argmax_rows<itype>(                                       \
      const device itype*, device int*, const constant int&,               \
      const constant int&, const constant int&, uint2, uint2, uint2, uint,  \
      uint);

instantiate_argmax_rows(bfloat16, bfloat)
instantiate_argmax_rows(float32, float)
