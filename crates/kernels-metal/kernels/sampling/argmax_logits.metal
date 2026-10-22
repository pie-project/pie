#include <metal_stdlib>
using namespace metal;

struct ArgmaxParams {
  uint vocab;
  uint n_eos;
  uint eos_ids[8];
};

template <typename T>
[[kernel]] void argmax_logits(
    const device T* logits      [[buffer(0)]],
    device uint* next_token     [[buffer(1)]],
    constant ArgmaxParams& p    [[buffer(2)]],
    device uint* eos_flag       [[buffer(3)]],
    uint3 tg_pos                [[threadgroup_position_in_grid]],
    uint3 lid3                  [[thread_position_in_threadgroup]],
    uint3 tg_size3              [[threads_per_threadgroup]],
    uint simd_lane_id           [[thread_index_in_simdgroup]],
    uint simd_group_id          [[simdgroup_index_in_threadgroup]]) {
  constexpr uint SIMD_SIZE = 32;
  const uint row = tg_pos.y;
  const uint lid = lid3.x;
  const uint tg_size = tg_size3.x;
  const uint vocab = p.vocab;
  const device T* row_logits = logits + size_t(row) * vocab;

  float best_v = -INFINITY;
  uint  best_i = 0;
  for (uint i = lid; i < vocab; i += tg_size) {
    float v = float(row_logits[i]);
    if (v > best_v) { best_v = v; best_i = i; }
  }

  for (uint off = SIMD_SIZE / 2; off > 0; off >>= 1) {
    float ov = simd_shuffle_down(best_v, off);
    uint  oi = simd_shuffle_down(best_i, off);
    if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
  }

  threadgroup float tg_v[SIMD_SIZE];
  threadgroup uint  tg_i[SIMD_SIZE];
  if (simd_lane_id == 0) { tg_v[simd_group_id] = best_v; tg_i[simd_group_id] = best_i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    const uint n_simd = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
    best_v = simd_lane_id < n_simd ? tg_v[simd_lane_id] : -INFINITY;
    best_i = simd_lane_id < n_simd ? tg_i[simd_lane_id] : 0u;
    for (uint off = SIMD_SIZE / 2; off > 0; off >>= 1) {
      float ov = simd_shuffle_down(best_v, off);
      uint  oi = simd_shuffle_down(best_i, off);
      if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (simd_lane_id == 0) {
      next_token[row] = best_i;
      uint flag = 0;
      for (uint e = 0; e < p.n_eos; ++e) { if (best_i == p.eos_ids[e]) { flag = 1; break; } }
      eos_flag[row] = flag;
    }
  }
}

#define instantiate_argmax(name, itype)                                    \
  template [[host_name("argmax_logits_" #name)]]                           \
  [[kernel]] void argmax_logits<itype>(                                    \
      const device itype*, device uint*, constant ArgmaxParams&,           \
      device uint*, uint3, uint3, uint3, uint, uint);

instantiate_argmax(bfloat16, bfloat)
