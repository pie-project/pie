// Raw-Metal batched PAGED attention (read side, M>1) — beta's lane.
//
// Generalizes sdpa_vector_decode from the M=1 CONTIGUOUS-ring K/V to the M>1 PAGED KV
// cache that the Pie runtime owns. The flash online-softmax math is IDENTICAL to
// sdpa_vector.metal; ONLY the per-key/value fetch changes: instead of striding a
// contiguous [n_kv_heads, max_ctx, head_dim] ring, each kv position is gathered through
// the page table (the byte-for-byte seam with delta's kv_append — wiki mac-paged-kv-bridge).
//
// Page layout (LOCKED, token-major NHD): k/v pages = [num_pages, page_size, n_kv_heads, head_dim].
//   phys_slot(kp) = kv_page_indices[ kv_page_indptr[r] + kp/page_size ] * page_size + kp%page_size
//   element       = (phys_slot * n_kv_heads + kv_head) * head_dim + d        (== delta's append dst)
//
// Causal bound: query token `row` at absolute position q_pos = position_ids[row] attends
// exactly kv positions [0, q_pos] (its own K was appended before attention). So the read
// walks kp = 0..q_pos — no kv_last_page_lens needed on the read side (the position IS the
// bound). Decode (1 tok/req) and prefill (qo-span tokens, each its own q_pos) use the SAME
// loop; ragged lengths fall out of per-row q_pos. req_of_token[row] (host-precomputed from
// qo_indptr, batch_schedule::tok_req) gives the owning request → its page list.
//
// At N=1, R=1 this reduces to a single query row walking one request's pages == the M=1
// decode path (a 1-page table over the old ring), so the shipped fast path is preserved.
//
// Launch: one threadgroup per (q_batch_head, query_row): grid threads
//   = (n_q_heads*1024, N, 1), tg = (1024, 1, 1)   (1024 = BN*BD, matches sdpa_vector).
// D = head_dim (template; qwen 256, gemma4 256/512 — host picks the instantiation +
// per-layer pages/kv_source redirect). gqa_factor = n_q_heads/n_kv_heads.
//
// Buffer ORDER here is the binding contract; alpha assigns the matching bind::SdpaPaged
// ordinals in decode_abi.hpp (I bind against his published layout). u32 page tables per schema.

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename T, int D, int V = D>
[[kernel]] void sdpa_paged_decode(
    const device T* queries     [[buffer(0)]],   // [N, n_q_heads, D]
    const device T* k_pages     [[buffer(1)]],   // [num_pages, page_size, n_kv_heads, D]
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],   // [N, n_q_heads, V]
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],   // [N] abs pos → causal bound
    const device int* req_of_token          [[buffer(6)]],   // [N] owning request r
    const device uint* kv_page_indices      [[buffer(7)]],   // [total_pages]
    const device uint* kv_page_indptr       [[buffer(8)]],   // [R+1]
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;         // 0..n_q_heads-1
  const int row              = tid.y;         // query token index in [0, N)
  const int kv_head_idx      = q_batch_head_idx / gqa_factor;
  const int n_q_heads        = int(tpg.x);

  // This query row's request + causal bound + its request's page-list base.
  const int r          = req_of_token[row];
  const int q_pos      = position_ids[row];   // attends kv positions [0, q_pos]
  const int page_base  = int(kv_page_indptr[r]);
  const int kv_row     = n_kv_heads * D;       // elements per token-slot across kv heads

  // queries[row, q_head, :]; out[row, q_head, :]
  queries += (size_t(row) * n_q_heads + q_batch_head_idx) * D + simd_lid * qk_per_thread;
  out     += (size_t(row) * n_q_heads + q_batch_head_idx) * V + simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale) * queries[i];
  for (int i = 0; i < v_per_thread; i++) o[i] = 0;

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  // Online-softmax over kv positions kp = simd_gid, +BN, ... up to and including q_pos.
  for (int kp = simd_gid; kp <= q_pos; kp += BN) {
    // Page-table gather (== delta's kv_append phys_slot, byte-for-byte).
    const int page = int(kv_page_indices[page_base + kp / page_size]);
    const size_t slot = size_t(page) * page_size + (kp % page_size);
    const device T* kptr = k_pages + (slot * n_kv_heads + kv_head_idx) * D + simd_lid * qk_per_thread;
    const device T* vptr = v_pages + (slot * n_kv_heads + kv_head_idx) * D + simd_lid * v_per_thread;

    for (int j = 0; j < qk_per_thread; j++) k[j] = kptr[j];
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * vptr[j];
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

  if (simd_lid == 0)
    for (int i = 0; i < v_per_thread; i++) out[i] = static_cast<T>(o[i]);
}

#define instantiate_sdpa_paged(name, itype, d, v)                          \
  template [[host_name("sdpa_paged_decode_" #name "_d_" #d)]]               \
  [[kernel]] void sdpa_paged_decode<itype, d, v>(                          \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged(float32, float, 256, 256)
instantiate_sdpa_paged(float16, half, 256, 256)
instantiate_sdpa_paged(bfloat16, bfloat, 256, 256)
instantiate_sdpa_paged(bfloat16, bfloat, 512, 512)  // gemma4 full-attn (head_dim 512)
