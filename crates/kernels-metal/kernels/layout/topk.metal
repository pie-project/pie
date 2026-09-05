// topk.metal — the k largest entries of every row, sorted, indices beside.
//
// One threadgroup per row. Every thread walks a strided share of the row
// keeping its own sorted list of K `(value, index)` pairs — most candidates
// fail the list's floor and cost one compare — then the lists meet in
// threadgroup memory and the group pops the global maximum K times: each
// thread offers its list's head, the simdgroups fold with `simd_max` and a
// `simd_min` over the indices that hit it (`argmax_rows`' rule: ties to the
// LOWEST column, a NaN never chosen), thread 0 picks across simdgroups, the
// owner advances. K rounds of two barriers for a handful of rows a fire.
//
// `THREADS x K x 8 bytes` of threadgroup memory: 128 x 16 x 8 = 16 KB.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint kTopkThreads = 128;
constant constexpr uint kTopkSimdgroups = kTopkThreads / 32;
constant constexpr float TOPK_NEG_INF = -INFINITY;
constant constexpr uint TOPK_NONE = 0xFFFFFFFFu;

/// `a` beats `b` when it is larger, or equal at a lower index.
inline bool topk_beats(float av, uint ai, float bv, uint bi) {
  return av > bv || (av == bv && ai < bi);
}

template <typename T, int K>
[[kernel]] void topk_rows(
    const device T* x          [[buffer(0)]],
    device float* values       [[buffer(1)]],
    device int* indices        [[buffer(2)]],
    const constant int& width  [[buffer(3)]],
    uint2 tid                  [[thread_position_in_grid]],
    uint2 tid_group            [[thread_position_in_threadgroup]],
    uint simd_lid              [[thread_index_in_simdgroup]],
    uint simd_gid              [[simdgroup_index_in_threadgroup]]) {
  const uint lid = tid_group.x;
  const size_t row = size_t(tid.y);
  const device T* src = x + row * size_t(width);

  // This thread's sorted list, best first.
  float lv[K];
  uint li[K];
  for (int j = 0; j < K; ++j) {
    lv[j] = TOPK_NEG_INF;
    li[j] = TOPK_NONE;
  }
  for (uint c = lid; c < uint(width); c += kTopkThreads) {
    const float v = float(src[c]);
    if (isnan(v) || !topk_beats(v, c, lv[K - 1], li[K - 1])) {
      continue;
    }
    // Insert, shifting the tail down.
    int at = K - 1;
    while (at > 0 && topk_beats(v, c, lv[at - 1], li[at - 1])) {
      lv[at] = lv[at - 1];
      li[at] = li[at - 1];
      --at;
    }
    lv[at] = v;
    li[at] = c;
  }

  threadgroup float lists_v[kTopkThreads * K];
  threadgroup uint lists_i[kTopkThreads * K];
  threadgroup float part_v[kTopkSimdgroups];
  threadgroup uint part_i[kTopkSimdgroups];
  threadgroup uint part_owner[kTopkSimdgroups];
  threadgroup uint winner;
  for (int j = 0; j < K; ++j) {
    lists_v[lid * K + j] = lv[j];
    lists_i[lid * K + j] = li[j];
  }
  uint head = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j = 0; j < K; ++j) {
    // Offer this list's head.
    const float hv = head < uint(K) ? lists_v[lid * K + head] : TOPK_NEG_INF;
    const uint hi = head < uint(K) ? lists_i[lid * K + head] : TOPK_NONE;
    const float m = simd_max(hv);
    const uint w = simd_min(hv == m ? hi : TOPK_NONE);
    const uint owner = simd_min(hv == m && hi == w ? lid : TOPK_NONE);
    if (simd_lid == 0) {
      part_v[simd_gid] = m;
      part_i[simd_gid] = w;
      part_owner[simd_gid] = owner;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float top = TOPK_NEG_INF;
      uint top_i = TOPK_NONE;
      uint top_owner = TOPK_NONE;
      for (uint sg = 0; sg < kTopkSimdgroups; ++sg) {
        if (part_i[sg] != TOPK_NONE && topk_beats(part_v[sg], part_i[sg], top, top_i)) {
          top = part_v[sg];
          top_i = part_i[sg];
          top_owner = part_owner[sg];
        }
      }
      values[row * size_t(K) + size_t(j)] = top_i == TOPK_NONE ? 0.0f : top;
      indices[row * size_t(K) + size_t(j)] = int(top_i == TOPK_NONE ? 0u : top_i);
      winner = top_owner;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (winner == lid) {
      ++head;
    }
  }
}

#define instantiate_topk_rows(name, itype, k)                                 \
  template [[host_name("topk_rows_" #name "_k_" #k)]]                         \
  [[kernel]] void topk_rows<itype, k>(                                        \
      const device itype*, device float*, device int*, const constant int&,   \
      uint2, uint2, uint, uint);

instantiate_topk_rows(bfloat16, bfloat, 8)
instantiate_topk_rows(bfloat16, bfloat, 16)
instantiate_topk_rows(float32, float, 8)
instantiate_topk_rows(float32, float, 16)
