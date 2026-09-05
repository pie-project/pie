// selector_walk.metal — DFlash2's candidate selector, walked.
//
// The reference (`mlx_dspark.dflash_model.CandidateSelector`) scores every
// `(predecessor, candidate)` pair of adjacent slots,
//
//     scores[s, p, c] = unary[s, c] + < A[pred[s, p]] * hp[s], B[cand[s, c]] >
//
// and `walk_greedy` follows the best successor from the anchor: only the ROW
// of the predecessor actually chosen is ever read, so a walk is `slots x K`
// dot products of `rank` terms, not `slots x K x K`. One threadgroup per
// request: 256 threads are sixteen lanes a candidate, the lanes stride the
// rank and fold with shuffles inside their aligned sixteen (two candidates a
// simdgroup), thread 0 takes the argmax (ties to the lower candidate) and the
// pick becomes the next slot's predecessor. Rows are the request's span in
// order: the first is the anchor (its pick is its first candidate, unread by
// any guest), the rest are mask slots.
//
// bf16 in, f32 accumulation; the reference is bf16 bilinear plus f32 unary.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kWalkThreads = 256;
constant constexpr uint kWalkLanes = 16;          // lanes a candidate
constant constexpr uint kWalkMaxK = kWalkThreads / kWalkLanes;

template <typename T>
[[kernel]] void selector_walk(
    const device int* cand        [[buffer(0)]],   // [rows, k]
    const device int* indptr      [[buffer(1)]],   // [lanes + 1]
    const device float* unary     [[buffer(2)]],   // [rows, k]
    const device T* hp            [[buffer(3)]],   // [rows, rank]
    const device int* tokens      [[buffer(4)]],   // [rows]
    const device T* pred          [[buffer(5)]],   // [vocab, rank]
    const device T* succ          [[buffer(6)]],   // [vocab, rank]
    device int* picks             [[buffer(7)]],   // [rows]
    const constant int& k         [[buffer(8)]],
    const constant int& rank      [[buffer(9)]],
    const constant int& vocab     [[buffer(10)]],
    uint2 pos                     [[thread_position_in_grid]],
    uint2 lpos                    [[thread_position_in_threadgroup]]) {
  const int r = int(pos.y);
  const uint tid = lpos.x;
  const uint c = tid / kWalkLanes;     // this thread's candidate
  const uint lane = tid % kWalkLanes;  // its lane inside the candidate
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin) {
    return;
  }

  threadgroup float score[kWalkMaxK];
  threadgroup int prev_id;
  if (tid == 0) {
    // The anchor row proposes nothing; its pick is its own first candidate.
    picks[begin] = cand[size_t(begin) * size_t(k)];
    prev_id = tokens[begin];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int row = begin + 1; row < end; ++row) {
    const int my_prev = prev_id;
    float partial = 0.0f;
    if (int(c) < k) {
      const int cid = cand[size_t(row) * size_t(k) + size_t(c)];
      const bool live = my_prev >= 0 && my_prev < vocab && cid >= 0 && cid < vocab;
      if (live) {
        const device T* a = pred + size_t(my_prev) * size_t(rank);
        const device T* b = succ + size_t(cid) * size_t(rank);
        const device T* h = hp + size_t(row) * size_t(rank);
        for (int d = int(lane); d < rank; d += int(kWalkLanes)) {
          partial += float(a[d]) * float(h[d]) * float(b[d]);
        }
      }
    }
    // Fold the sixteen lanes of this candidate; the xor tree stays inside
    // the aligned sixteen, so the two candidates sharing a simdgroup do not
    // mix.
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
      const constant int&, const constant int&, uint2, uint2);

instantiate_selector_walk(bfloat16, bfloat)
