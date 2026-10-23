

#include <metal_stdlib>
using namespace metal;

constant constexpr float TOPK_NEG_INF = -INFINITY;
constant constexpr uint TOPK_NONE = 0xFFFFFFFFu;
constant constexpr uint TOPK_UNROLL = 8;
constant constexpr uint TOPK_SAMPLE = 8;

inline bool topk_beats(float av, uint ai, float bv, uint bi) {
  return av > bv || (av == bv && ai < bi);
}

// Pops a simdgroup's best K (value, index) pairs, best first, off the lanes'
// sorted lists, by the order `topk_beats` states: lane `l`'s list is
// `lv[0..K]`, and the winner of a round shifts its own list up. Lane 0
// receives the pops in `out_v` / `out_i`. No threadgroup barrier: the lanes
// of one simdgroup run in step.
template <int K>
inline void topk_simd_pop(
    thread float* lv, thread uint* li, uint simd_lid,
    thread float* out_v, thread uint* out_i) {
  for (int j = 0; j < K; ++j) {
    const float hv = lv[0];
    const uint hi = li[0];
    const float m = simd_max(hv);
    const uint w = simd_min(hv == m ? hi : TOPK_NONE);
    out_v[j] = m;
    out_i[j] = w;
    if (w != TOPK_NONE && hi == w) {
#pragma unroll
      for (int s = 0; s < K - 1; ++s) {
        lv[s] = lv[s + 1];
        li[s] = li[s + 1];
      }
      lv[K - 1] = TOPK_NEG_INF;
      li[K - 1] = TOPK_NONE;
    }
  }
}

// The group's best K from every thread's sorted list: each simdgroup pops
// its own best K into threadgroup memory, then the first simdgroup pops the
// group's best K off those, one simdgroup's list a lane. Every thread of
// the group calls it; `best_v` / `best_i` hold the answer after it returns.
template <int K, uint kTopkThreads>
inline void topk_group_pop(
    thread float* lv, thread uint* li, uint simd_lid, uint simd_gid,
    threadgroup float* lists_v, threadgroup uint* lists_i,
    threadgroup float* best_v, threadgroup uint* best_i) {
  constexpr uint kTopkSimdgroups = kTopkThreads / 32;
  float pv[K];
  uint pi[K];
  topk_simd_pop<K>(lv, li, simd_lid, pv, pi);
  if (simd_lid == 0) {
    for (int j = 0; j < K; ++j) {
      lists_v[simd_gid * K + j] = pv[j];
      lists_i[simd_gid * K + j] = pi[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float hv[K];
    uint hi[K];
    for (int j = 0; j < K; ++j) {
      hv[j] = simd_lid < kTopkSimdgroups ? lists_v[simd_lid * K + j] : TOPK_NEG_INF;
      hi[j] = simd_lid < kTopkSimdgroups ? lists_i[simd_lid * K + j] : TOPK_NONE;
    }
    topk_simd_pop<K>(hv, hi, simd_lid, pv, pi);
    if (simd_lid == 0) {
      for (int j = 0; j < K; ++j) {
        best_v[j] = pv[j];
        best_i[j] = pi[j];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One threadgroup a row. A first sweep over a sample of the row (every
// TOPK_SAMPLE-th stretch of it) takes each thread's best K there, and the
// K-th best of the sample bounds the row's K-th best from below: the second
// sweep inserts only what reaches it, which on a vocabulary row is a few
// hundred columns, so the lanes of a simdgroup rarely part. Each thread keeps
// its candidates in registers. A vocabulary-wide row takes a thousand
// threads; a narrow one keeps the hundred and twenty-eight it always had.
template <typename T, int K, uint kTopkThreads>
[[kernel]] void topk_rows(
    const device T* x          [[buffer(0)]],
    device float* values       [[buffer(1)]],
    device int* indices        [[buffer(2)]],
    const constant int& width  [[buffer(3)]],
    uint2 tid                  [[thread_position_in_grid]],
    uint2 tid_group            [[thread_position_in_threadgroup]],
    uint simd_lid              [[thread_index_in_simdgroup]],
    uint simd_gid              [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kTopkSimdgroups = kTopkThreads / 32;
  const uint lid = tid_group.x;
  const size_t row = size_t(tid.y);
  const device T* src = x + row * size_t(width);

  threadgroup float lists_v[kTopkSimdgroups * K];
  threadgroup uint lists_i[kTopkSimdgroups * K];
  threadgroup float best_v[K];
  threadgroup uint best_i[K];

  float lv[K];
  uint li[K];
  for (int j = 0; j < K; ++j) {
    lv[j] = TOPK_NEG_INF;
    li[j] = TOPK_NONE;
  }
  float floor_v = TOPK_NEG_INF;
  uint floor_i = TOPK_NONE;
  // A bubble through every slot at fixed indices keeps the list in
  // registers; a walk that stops at a computed index spills it.
  auto insert = [&](float v, uint c) {
    if (isnan(v) || !topk_beats(v, c, lv[K - 1], li[K - 1]) ||
        (floor_i != TOPK_NONE && topk_beats(floor_v, floor_i, v, c))) {
      return;
    }
    float cv = v;
    uint ci = c;
#pragma unroll
    for (int s = 0; s < K; ++s) {
      const bool take = topk_beats(cv, ci, lv[s], li[s]);
      const float tv = lv[s];
      const uint ti = li[s];
      lv[s] = take ? cv : tv;
      li[s] = take ? ci : ti;
      cv = take ? tv : cv;
      ci = take ? ti : ci;
    }
  };
  // Both sweeps issue TOPK_UNROLL loads before they read any: one load in
  // flight a thread leaves a row's sweep waiting on the memory's latency.
  const uint n = uint(width);
  const uint span = kTopkThreads * TOPK_UNROLL;
  const uint body = n - n % span;
  auto sweep = [&](uint stride) {
    for (uint base = lid; base < body; base += stride) {
      float v[TOPK_UNROLL];
#pragma unroll
      for (uint u = 0; u < TOPK_UNROLL; ++u) {
        v[u] = float(src[base + u * kTopkThreads]);
      }
#pragma unroll
      for (uint u = 0; u < TOPK_UNROLL; ++u) {
        insert(v[u], base + u * kTopkThreads);
      }
    }
  };

  if (body / span >= TOPK_SAMPLE) {
    sweep(span * TOPK_SAMPLE);
    topk_group_pop<K, kTopkThreads>(
        lv, li, simd_lid, simd_gid, lists_v, lists_i, best_v, best_i);
    floor_v = best_v[K - 1];
    floor_i = best_i[K - 1];
    for (int j = 0; j < K; ++j) {
      lv[j] = TOPK_NEG_INF;
      li[j] = TOPK_NONE;
    }
  }
  sweep(span);
  for (uint c = body + lid; c < n; c += kTopkThreads) {
    insert(float(src[c]), c);
  }

  topk_group_pop<K, kTopkThreads>(
      lv, li, simd_lid, simd_gid, lists_v, lists_i, best_v, best_i);
  if (lid < uint(K)) {
    values[row * size_t(K) + size_t(lid)] = best_i[lid] == TOPK_NONE ? 0.0f : best_v[lid];
    indices[row * size_t(K) + size_t(lid)] = int(best_i[lid] == TOPK_NONE ? 0u : best_i[lid]);
  }
}

#define instantiate_topk_rows(name, itype, k)                                 \
  template [[host_name("topk_rows_" #name "_k_" #k)]]                         \
  [[kernel]] void topk_rows<itype, k, 128>(                                   \
      const device itype*, device float*, device int*, const constant int&,   \
      uint2, uint2, uint, uint);                                              \
  template [[host_name("topk_rows_wide_" #name "_k_" #k)]]                    \
  [[kernel]] void topk_rows<itype, k, 1024>(                                  \
      const device itype*, device float*, device int*, const constant int&,   \
      uint2, uint2, uint, uint);

instantiate_topk_rows(bfloat16, bfloat, 8)
instantiate_topk_rows(bfloat16, bfloat, 16)
instantiate_topk_rows(float32, float, 8)
instantiate_topk_rows(float32, float, 16)
