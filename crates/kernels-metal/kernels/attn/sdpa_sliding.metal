#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

METAL_FUNC void sdpa_online_update(
    float score, thread float& max_score, thread float& sum_exp_score,
    thread float& history_scale, thread float& score_scale) {
  const float new_max = max(max_score, score);
  history_scale = fast::exp(max_score - new_max);
  score_scale = fast::exp(score - new_max);
  max_score = new_max;
  sum_exp_score = sum_exp_score * history_scale + score_scale;
}

METAL_FUNC float sdpa_merge_sink(
    float sink, float reference_max, thread float& sum_exp_score) {
  const float merged_max = max(reference_max, sink);
  const float output_scale = fast::exp(reference_max - merged_max);
  sum_exp_score =
      sum_exp_score * output_scale + fast::exp(sink - merged_max);
  return output_scale;
}

METAL_FUNC float sdpa_lse_base2(float max_score, float sum_exp_score) {
  constexpr float kLog2E = 1.44269504088896340736f;
  return sum_exp_score > 0.0f ? (max_score * kLog2E + log2(sum_exp_score))
                              : -INFINITY;
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_decode_swa(
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
    const constant int& window          [[buffer(11)]],
    const constant int& q_row_stride    [[buffer(12)]],
    const constant int& o_row_stride    [[buffer(13)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;

  const int n_rows = int(tpg.y);
  const int N_row = N - (n_rows - 1 - int(tid.y));

  const int kv_start = (window > 0 && N_row > window) ? (N_row - window) : 0;
  const int Nw = N_row - kv_start;

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
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;

  const int q_offset = q_seq_idx * q_row_stride + q_batch_head_idx * D;
  const int o_offset = q_seq_idx * o_row_stride + q_batch_head_idx * V;

  queries += q_offset + simd_lid * qk_per_thread;

  keys += kv_head_idx * k_head_stride + (kv_start + simd_gid) * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + (kv_start + simd_gid) * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < Nw; i += BN) {
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    U factor, exp_score;
    sdpa_online_update(
        score, max_score, sum_exp_score, factor, exp_score);
    for (int j = 0; j < v_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * values[j];
    }
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  SDPA_ONLINE_FINISH()
}

#define instantiate_sdpa_swa(name, itype, d, v)                          \
  template [[host_name("sdpa_vector_decode_swa_" #name "_d_" #d)]]        \
  [[kernel]] void sdpa_vector_decode_swa<itype, d, v>(                   \
      const device itype*, const device itype*, const device itype*,     \
      device itype*, const constant int&, const constant int&,           \
      const constant size_t&, const constant size_t&,                    \
      const constant size_t&, const constant size_t&,                    \
      const constant float&, const constant int&,                        \
      const constant int&, const constant int&,                          \
      uint3, uint3, uint, uint);

instantiate_sdpa_swa(bfloat16, bfloat, 256, 256)

instantiate_sdpa_swa(bfloat16, bfloat, 512, 512)

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_decode_sink(
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
    const constant int& window          [[buffer(11)]],
    const constant int& q_row_stride    [[buffer(12)]],
    const constant int& o_row_stride    [[buffer(13)]],
    const device T* sinks               [[buffer(14)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;

  const int n_rows = int(tpg.y);
  const int N_row = N - (n_rows - 1 - int(tid.y));

  const int kv_start = (window > 0 && N_row > window) ? (N_row - window) : 0;
  const int Nw = N_row - kv_start;

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
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;

  const int q_offset = q_seq_idx * q_row_stride + q_batch_head_idx * D;
  const int o_offset = q_seq_idx * o_row_stride + q_batch_head_idx * V;

  queries += q_offset + simd_lid * qk_per_thread;

  keys += kv_head_idx * k_head_stride + (kv_start + simd_gid) * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + (kv_start + simd_gid) * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  for (int i = simd_gid; i < Nw; i += BN) {
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys[j];
    }
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    U factor, exp_score;
    sdpa_online_update(
        score, max_score, sum_exp_score, factor, exp_score);
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

  const U sink = static_cast<U>(sinks[tid.x]);
  const U orescale = sdpa_merge_sink(sink, new_max, sum_exp_score);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) * orescale;
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define instantiate_sdpa_sink(name, itype, d, v)                         \
  template [[host_name("sdpa_vector_decode_sink_" #name "_d_" #d)]]       \
  [[kernel]] void sdpa_vector_decode_sink<itype, d, v>(                  \
      const device itype*, const device itype*, const device itype*,     \
      device itype*, const constant int&, const constant int&,           \
      const constant size_t&, const constant size_t&,                    \
      const constant size_t&, const constant size_t&,                    \
      const constant float&, const constant int&,                        \
      const constant int&, const constant int&, const device itype*,     \
      uint3, uint3, uint, uint);

instantiate_sdpa_sink(bfloat16, bfloat, 64, 64)
