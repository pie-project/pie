// Raw-Metal sliding-window SDPA (single-pass decode), gemma4 lane.
//
// Derived from delta's sdpa_vector.metal (full-attn single-pass decode). The ONLY
// structural change is the key/value iteration window:
//
//   * gemma4 sliding layers attend only the last `window` (=512) positions; full
//     layers attend all past keys. At M=1 decode the query sits at logical position
//     N-1 (the current token's K/V is already appended), so the sliding window
//     [p-window+1 .. p] maps to key indices [max(0, N-window) .. N-1]. Causal mask
//     stays trivially all-true WITHIN the window, so no mask branch is needed.
//
//   * `window <= 0` selects full attention (kv_start = 0) -> this single kernel also
//     serves gemma4's full-attn layers {4,9,14,19,24,29,34}; the encode_fn binds the
//     per-layer window (512 sliding / 0 full).
//
//   * kv_start = (window > 0 && N > window) ? N - window : 0. We advance the keys/
//     values base pointers by kv_start*seq_stride and treat the window as a
//     contiguous sub-sequence of length Nw = N - kv_start, leaving the inner
//     simd-reduction identical to the full-attn port.
//
// Buffers mirror sdpa_vector_decode; `window` is appended at buffer(11) (static
// geometry, but kept a constant *buffer* to match the decode_abi I1 convention so
// the command buffer stays byte-identical step-to-step).
//
// M>1 (prefill). The kernel already carried a query-row dimension in tid.y; what
// it lacked was a per-row causal position, so every row would have attended all N
// keys. Row m of an M-row launch holds the token at logical end N-(M-1-m), which
// is the ONLY thing that changes -- `kv_start`/`Nw` are then the existing
// expressions with N replaced by that. Decode is M=1, where N-(M-1-m) == N, so
// the two paths run the same arithmetic rather than two spellings of it.
//
// The query/output row strides are stated by the caller (buffers 12/13) instead
// of inferred. mlx's original infers [head][row][D] from `q_seq_idx`, but a
// prefill GEMM writes [row][head][D]; an inferred layout that disagrees with the
// producer reads plausible garbage rather than failing.
//
// Launch: group=(1024,1,1), grid=(n_q_heads,M,1). D=V=256, gqa_factor=8 for gemma4.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

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

  // Where this row's token sits. The M rows of a prefill are the last M
  // positions, so row m ends at N-(M-1-m); at decode M==1 and this is N.
  const int n_rows = int(tpg.y);
  const int N_row = N - (n_rows - 1 - int(tid.y));

  // Sliding window: restrict to the last `window` keys. window<=0 => full attention.
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
  const int q_seq_idx = tid.y;  // 0 at decode
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  // Row-major over tokens: row m starts at m*row_stride, head h at h*D within it.
  const int q_offset = q_seq_idx * q_row_stride + q_batch_head_idx * D;
  const int o_offset = q_seq_idx * o_row_stride + q_batch_head_idx * V;

  queries += q_offset + simd_lid * qk_per_thread;
  // Advance the K/V base past the keys outside the window, then index within [0, Nw).
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

// gemma4 full-attention layers {4,9,14,19,24,29,34} use head_dim 512 (theta 1e6,
// full rotation). The kernel is compile-time templated on D, so a distinct PSO is
// required; the encode_fn selects it per-layer for is_full_attn() dispatches.
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

  // Where this row's token sits. The M rows of a prefill are the last M
  // positions, so row m ends at N-(M-1-m); at decode M==1 and this is N.
  const int n_rows = int(tpg.y);
  const int N_row = N - (n_rows - 1 - int(tid.y));

  // Sliding window: restrict to the last `window` keys. window<=0 => full attention.
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
  const int q_seq_idx = tid.y;  // 0 at decode
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  // Row-major over tokens: row m starts at m*row_stride, head h at h*D within it.
  const int q_offset = q_seq_idx * q_row_stride + q_batch_head_idx * D;
  const int o_offset = q_seq_idx * o_row_stride + q_batch_head_idx * V;

  queries += q_offset + simd_lid * qk_per_thread;
  // Advance the K/V base past the keys outside the window, then index within [0, Nw).
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

  // The sink: a learned per-head scalar that joins the softmax denominator as
  // though it were one more key, with no value to contribute. It is what lets a
  // head attend to nothing -- as the sink grows, every real weight shrinks and
  // the output goes to zero. Folded in at a shared reference so a sink larger
  // than every logit does not overflow the exponential.
  const U sink = static_cast<U>(sinks[tid.x]);
  const U m2 = max(new_max, sink);
  const U orescale = fast::exp(new_max - m2);
  sum_exp_score = sum_exp_score * orescale + fast::exp(sink - m2);

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


// ── GPT-OSS: the same attention with per-head sinks ─────────────────────────
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
