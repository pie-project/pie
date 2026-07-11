// Raw-Metal port of MLX sdpa_vector (single-pass), scoped to Phase-0 decode.
//
// Source: mlx/backend/metal/kernels/sdpa_vector.h (sdpa_vector, single-pass).
// Port notes (M=1 decode, B=1):
//   * Single query token attends all N past keys -> the causal mask is trivially
//     all-true (use_key always true), so do_causal / has_mask / bool_mask /
//     float_mask / has_sinks / query_transposed function-constants are all DROPPED
//     and their branches removed. qwen3.6 full-attn has no sinks, no mask at decode.
//   * N == kv_len IS a per-token IO-derived scalar (grows each step) -> kept as a
//     constant *buffer* (buffer 5), NOT setBytes, per decode_abi I1 so the CB stays
//     byte-identical (executor writes the new kv_len into the slot each token).
//   * head/seq strides are STATIC paged-KV geometry -> constant buffers, fine.
//   * Limits<U>::finite_min replaced with an explicit lowest-float constant.
//   * scale = 1/sqrt(head_dim), applied to q. bfloat native on Metal 4.
// Launch: group=(1024,1,1), grid=(n_q_heads,1,1). D=V=256, gqa_factor=4 for qwen3.6.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_decode(
    const device T* queries [[buffer(0)]],
    const device T* keys    [[buffer(1)]],
    const device T* values  [[buffer(2)]],
    device T* out           [[buffer(3)]],
    const constant int& gqa_factor      [[buffer(4)]],
    const constant int& N               [[buffer(5)]],
    const constant size_t& k_head_stride[[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride[[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale         [[buffer(10)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;  // < -FLT_MAX/... finite lowest sentinel
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;  // 0 at decode
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset = o_offset;  // query_transposed == false

  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < N; i += BN) {
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define instantiate_sdpa_decode(name, itype, d, v)                       \
  template [[host_name("sdpa_vector_decode_" #name "_d_" #d)]]            \
  [[kernel]] void sdpa_vector_decode<itype, d, v>(                       \
      const device itype*, const device itype*, const device itype*,     \
      device itype*, const constant int&, const constant int&,           \
      const constant size_t&, const constant size_t&,                    \
      const constant size_t&, const constant size_t&,                    \
      const constant float&, uint3, uint3, uint, uint);

instantiate_sdpa_decode(float32, float, 256, 256)
instantiate_sdpa_decode(float16, half, 256, 256)
instantiate_sdpa_decode(bfloat16, bfloat, 256, 256)
