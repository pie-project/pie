#pragma once

// Paged-KV naive attention. Streams over each request's KV pages and
// computes the softmax inline — no tensor cores, no shared-memory
// double-buffering, no flashinfer template machinery. Fallback for
// HEAD_DIM values flashinfer's TC prefill template can't handle
// (currently HEAD_DIM=512 — Gemma-4 full-attention layers).
//
// Causal mask is hard-wired against `(qo_idx + kv_baseline)` so a
// request's queries see all of its prior KV rows plus all of its own
// queries up to and including itself.
//
// One block per (request, qo_token, head). Each block:
//   1. Two-pass softmax across the request's KV span:
//      pass 1 — find row max,
//      pass 2 — accumulate exp + V weighted sum.
//   2. Writes the head's output row.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

// `q`             [total_tokens, num_q_heads, head_dim]   bf16
// `k_pages`/      [num_pages, page_size, num_kv_heads, head_dim]
//   `v_pages`
// `o`             [total_tokens, num_q_heads, head_dim]   bf16
// `qo_indptr_d`   [R+1] device  — start row of each request in `q`/`o`
// `kv_page_indices_d`   device  — concatenated page-id list
// `kv_page_indptr_d`    [R+1] device — request page-list bounds
// `kv_last_page_lens_d` [R] device   — last-page valid token count
// `sm_scale`           softmax scale; pass `-1.f` for `1/sqrt(head_dim)`
// `window_left`        non-negative enables sliding window
void launch_attention_naive_paged_bf16(
    const void* q,
    const void* k_pages, const void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    cudaStream_t stream,
    int window_left = -1,
    float sm_scale = -1.f);

}  // namespace pie_cuda_driver::ops
