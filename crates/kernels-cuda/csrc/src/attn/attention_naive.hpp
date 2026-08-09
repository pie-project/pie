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

namespace pie_cuda_driver::kernels::attn {

// `attention_naive_bf16`, `attention_mtp_history_bf16` and
// `attention_mtp_paged_history_bf16` WERE declared here. All three launchers
// are deleted -- the audit measured the cluster's consumer set as the
// cluster, which is empty from outside. The `.cu` says what went and why the
// device text stayed.

void mtp_shift_hidden_bf16(
    const void* target_hidden,
    const void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    void* out,
    int total_tokens,
    int num_requests,
    int hidden_size,
    cudaStream_t stream);

void mtp_update_pending_hidden_bf16(
    const void* target_hidden,
    void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    int num_requests,
    int hidden_size,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::attn
