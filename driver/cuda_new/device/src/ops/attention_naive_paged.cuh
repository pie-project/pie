#pragma once

// Paged-KV naive attention — the self-contained fallback that needs NO
// flashinfer. Streams over each request's KV pages and computes the
// softmax inline (no tensor cores, no shared-memory double-buffering, no
// flashinfer template machinery). Fallback for HEAD_DIM values
// flashinfer's TC prefill template can't handle (currently HEAD_DIM=512 —
// Gemma-4 full-attention layers).
//
// Lifted from driver/cuda/src/ops/attention_naive_paged.{hpp,cu}: ONLY the
// raw-pointer bf16 prefill path (`launch_attention_naive_paged_bf16` plus
// its `naive_paged_attn_kernel` and the `__device__`/anonymous-namespace
// helper closure it touches: check_head_dim_supported, load_kv_scalar,
// fp8_to_float, fp4_e2m1_value, transform_logit, custom_mask_allows). The
// KvCacheLayerView overloads, the decode kernel and the `_custom` variant
// are dropped, so this never includes kv_cache.hpp; the raw variant takes
// plain device pointers.
//
// Causal mask is hard-wired against `(qo_idx + kv_baseline)` so a request's
// queries see all of its prior KV rows plus all of its own queries up to
// and including themselves.
//
// One block per (request, qo_token, head). Each block:
//   1. Two-pass softmax across the request's KV span:
//      pass 1 — find row max,
//      pass 2 — accumulate exp + V weighted sum.
//   2. Writes the head's output row.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::ops {

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
void attention_naive_paged_bf16(
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
    float sm_scale = -1.f,
    float logits_soft_cap = 0.f,
    float* lse_out = nullptr);

}  // namespace pie_cuda_device::ops
