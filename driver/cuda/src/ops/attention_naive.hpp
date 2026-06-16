#pragma once

// Single-sequence causal attention with GQA, no paging, no batching.
// **For numeric-parity testing only** — not a hot path. M1.2.3 swaps this
// for the flashinfer paged kernels.
//
// Layout:
//   q [num_tokens, num_q_heads,  head_dim]   bf16
//   k [num_tokens, num_kv_heads, head_dim]   bf16
//   v [num_tokens, num_kv_heads, head_dim]   bf16
//   o [num_tokens, num_q_heads,  head_dim]   bf16
//
// Each query at position p attends causally to keys at positions [0..p].
// GQA broadcast: query head h attends to KV head h * num_kv_heads / num_q_heads.

#include <cuda_runtime.h>
#include <cstdint>

namespace pie_cuda_driver::ops {

void launch_attention_naive_bf16(
    const void* q, const void* k, const void* v,
    void* o,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

void launch_attention_mtp_history_bf16(
    const void* q,
    const void* k_history,
    const void* v_history,
    void* o,
    int num_tokens,
    int history_steps,
    int history_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream);

void launch_attention_mtp_paged_history_bf16(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* k_history,
    const void* v_history,
    void* o,
    const std::int32_t* position_ids,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_tokens,
    int history_steps,
    int history_stride,
    int max_global_tokens,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    bool global_cache_uses_prefix_position,
    cudaStream_t stream);

void launch_mtp_shift_hidden_bf16(
    const void* target_hidden,
    const void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    void* out,
    int total_tokens,
    int num_requests,
    int hidden_size,
    cudaStream_t stream);

void launch_mtp_update_pending_hidden_bf16(
    const void* target_hidden,
    void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    int num_requests,
    int hidden_size,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops
