

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kWalkThreads = 256;
constant constexpr uint kWalkLanes = 16;
constant constexpr uint kWalkMaxK = kWalkThreads / kWalkLanes;

template <typename T>
[[kernel]] void selector_walk(
    const device int* cand        [[buffer(0)]],
    const device int* indptr      [[buffer(1)]],
    const device float* unary     [[buffer(2)]],
    const device T* hp            [[buffer(3)]],
    const device int* tokens      [[buffer(4)]],
    const device T* pred          [[buffer(5)]],
    const device T* succ          [[buffer(6)]],
    device int* picks             [[buffer(7)]],
    const constant int& k         [[buffer(8)]],
    const constant int& rank      [[buffer(9)]],
    const constant int& vocab     [[buffer(10)]],
    const constant int& has_hp    [[buffer(11)]],
    const constant int& first     [[buffer(12)]],
    uint2 pos                     [[thread_position_in_grid]],
    uint2 lpos                    [[thread_position_in_threadgroup]]) {
  const int r = int(pos.y);
  const uint tid = lpos.x;
  const uint c = tid / kWalkLanes;
  const uint lane = tid % kWalkLanes;
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin) {
    return;
  }

  threadgroup float score[kWalkMaxK];
  threadgroup int prev_id;
  if (tid == 0) {

    if (first > 0) {
      picks[begin] = cand[size_t(begin) * size_t(k)];
    }
    prev_id = tokens[begin];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int row = begin + first; row < end; ++row) {
    const int my_prev = prev_id;
    float partial = 0.0f;
    if (int(c) < k) {
      const int cid = cand[size_t(row) * size_t(k) + size_t(c)];
      const bool live = my_prev >= 0 && my_prev < vocab && cid >= 0 && cid < vocab;
      if (live) {
        const device T* a = pred + size_t(my_prev) * size_t(rank);
        const device T* b = succ + size_t(cid) * size_t(rank);
        if (has_hp) {
          const device T* h = hp + size_t(row) * size_t(rank);
          for (int d = int(lane); d < rank; d += int(kWalkLanes)) {
            partial += float(a[d]) * float(h[d]) * float(b[d]);
          }
        } else {
          for (int d = int(lane); d < rank; d += int(kWalkLanes)) {
            partial += float(a[d]) * float(b[d]);
          }
        }
      }
    }

    partial += simd_shuffle_xor(partial, 8u);
    partial += simd_shuffle_xor(partial, 4u);
    partial += simd_shuffle_xor(partial, 2u);
    partial += simd_shuffle_xor(partial, 1u);
    if (lane == 0 && int(c) < k) {
      score[c] = unary[size_t(row) * size_t(k) + size_t(c)] + partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      int best = 0;
      float best_v = score[0];
      for (int j = 1; j < k; ++j) {
        if (score[j] > best_v) {
          best_v = score[j];
          best = j;
        }
      }
      const int pick = cand[size_t(row) * size_t(k) + size_t(best)];
      picks[row] = pick;
      prev_id = pick;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_selector_walk(name, itype)                                \
  template [[host_name("selector_walk_" #name)]]                              \
  [[kernel]] void selector_walk<itype>(                                       \
      const device int*, const device int*, const device float*,              \
      const device itype*, const device int*, const device itype*,            \
      const device itype*, device int*, const constant int&,                  \
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, uint2, uint2);

instantiate_selector_walk(bfloat16, bfloat)
