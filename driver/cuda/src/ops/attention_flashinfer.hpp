#pragma once

// flashinfer-backed paged attention. Phase 1: decode-only (every request
// has qo_len == 1). Phase 2 will add the prefill path. Same call signature
// as `attention_paged.hpp` so the forward pass can dispatch on a flag.

#include <cstdint>
#include <cuda_runtime.h>

#include "attention_workspace.hpp"

namespace pie_cuda_driver::ops {

// Decode-only: total_tokens must equal num_requests. Each query attends to
// the full KV history of its request as described by the page-indptr arrays.
void launch_attention_flashinfer_decode_bf16(
    const void* q,                                 // [num_requests, h_q, d]
    void* k_pages, void* v_pages,                  // [num_pages, page_size, h_kv, d]
    void* o,                                       // [num_requests, h_q, d]
    const std::uint32_t* kv_page_indices_d,        // device
    const std::uint32_t* kv_page_indptr_d,         // device, [R+1]
    const std::uint32_t* kv_last_page_lens_d,      // device, [R]
    const std::uint32_t* kv_page_indptr_h,         // host pointer (for plan)
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream);

// Prefill (or mixed prefill+decode): per-request qo_len comes from
// qo_indptr. Causal mask is hard-wired (DefaultAttention + MaskMode::kCausal).
void launch_attention_flashinfer_prefill_bf16(
    const void* q,                                 // [total_tokens, h_q, d]
    void* k_pages, void* v_pages,                  // [num_pages, page_size, h_kv, d]
    void* o,                                       // [total_tokens, h_q, d]
    const std::uint32_t* qo_indptr_d,              // device, [R+1]
    const std::uint32_t* kv_page_indices_d,        // device
    const std::uint32_t* kv_page_indptr_d,         // device, [R+1]
    const std::uint32_t* kv_last_page_lens_d,      // device, [R]
    const std::uint32_t* qo_indptr_h,              // host (for plan)
    const std::uint32_t* kv_page_indptr_h,         // host (for plan)
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream);

// Same prefill, with a custom packed-bit mask per request. `mask_d` is the
// concatenation of all per-request bitmaps; `mask_indptr_d[r]` is the byte
// offset of request r's mask. Each request's mask is `qo_len_r × kv_len_r`
// bits, row-major (qo_idx × kv_len + kv_idx).
void launch_attention_flashinfer_prefill_custom_bf16(
    const void* q,
    void* k_pages, void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t*  mask_d,                   // device, packed bitmap
    const std::int32_t*  mask_indptr_d,            // device, [R+1] byte offsets
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops
