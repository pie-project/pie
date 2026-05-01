#pragma once

// flashinfer-backed paged attention. Phase 1: decode-only (every request
// has qo_len == 1). Phase 2 will add the prefill path. Same call signature
// as `attention_paged.hpp` so the forward pass can dispatch on a flag.

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"

namespace pie_cuda_driver::ops {

// Opaque cache of flashinfer's `DecodePlanInfo` plus the few scheduling
// fields the dispatch needs. Lifecycle: created once (e.g. in
// ForwardContext), reset each fire by `plan_attention_flashinfer_decode_bf16`,
// then reused by 28 per-layer dispatch calls within that fire. Hoisting
// the plan out of the per-layer loop saves ~27 redundant DecodePlan
// invocations per fire — the plan is identical across all layers in
// pure-decode mode.
struct DecodePlanCache;

struct DecodePlanCacheDeleter {
    void operator()(DecodePlanCache* p) const noexcept;
};
using DecodePlanCachePtr = std::unique_ptr<DecodePlanCache, DecodePlanCacheDeleter>;

DecodePlanCachePtr make_decode_plan();

// Compute decode plan once per fire. Stores results in `cache` and the
// workspace's int/float buffers (so per-layer dispatch can read them).
void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream);

// Per-layer dispatch reusing the cached plan. `q`/`k_pages`/`v_pages`/`o`
// vary per layer; everything else comes from the cache + workspace.
//
// `window_left`: non-negative enables sliding-window attention (only the
// last `window_left + 1` KV tokens are visible to each query). `-1`
// means full causal — the same flashinfer kernel is used either way
// (the variant is compiled with `use_sliding_window=true` but the
// runtime check is a no-op when `window_left == -1`).
//
// `logits_soft_cap`: positive enables Gemma-2 style `cap*tanh(logits/cap)`
// inside the attention softmax. Zero disables — no overhead, no
// alternative compile path is taken (a second template variant is
// compiled with `use_logits_soft_cap=true`; we runtime-dispatch).
// `sm_scale`: softmax scaling factor before the exp(). Negative means
// "use `1/sqrt(head_dim)`" (the default flashinfer behaviour).
// Override is needed when (a) the model wants a non-standard scale —
// e.g. Gemma-4 sets `sm_scale=1.0` because q/k norm absorbs the
// `1/sqrt(d)` factor — or (b) the kernel runs at a *padded* HEAD_DIM
// (e.g. Phi-3 at 128 with logical head_dim=96), in which case
// `1/sqrt(96)` rather than `1/sqrt(128)` is the correct scale.
void dispatch_attention_flashinfer_decode_bf16(
    const DecodePlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f);

// Prefill (or mixed prefill+decode): per-request qo_len comes from
// qo_indptr. Causal mask is hard-wired (DefaultAttention + MaskMode::kCausal).
// `window_left` mirrors the decode entry point — non-negative enables
// sliding-window attention.
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
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f);

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
    cudaStream_t stream,
    int window_left = -1);

}  // namespace pie_cuda_driver::ops
