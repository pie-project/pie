#pragma once

// flashinfer-backed paged attention. Phase 1: decode-only (every request
// has qo_len == 1). Phase 2 will add the prefill path. Same call signature
// as `attention_paged.hpp` so the forward pass can dispatch on a flag.

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "kernels/kv_cache_view.hpp"
#include "ops/attention_workspace.hpp"

namespace pie_cuda_driver::ops {

// Opaque cache of flashinfer's `DecodePlanInfo` plus the few scheduling
// fields the dispatch needs. Lifecycle: created once (e.g. in
// BatchEngine), reset each fire by `plan_attention_flashinfer_decode_bf16`,
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

struct PrefillPlanCache;

struct PrefillPlanCacheDeleter {
    void operator()(PrefillPlanCache* p) const noexcept;
};
using PrefillPlanCachePtr = std::unique_ptr<PrefillPlanCache, PrefillPlanCacheDeleter>;

PrefillPlanCachePtr make_prefill_plan();

// Compact graph-layout class for the most recent decode plan. CUDA graph
// replay records the host-side dispatch branch, so split-KV and non-split
// plans need distinct graph keys.
std::uint32_t decode_plan_graph_layout(const DecodePlanCache& cache);
std::uint32_t prefill_plan_graph_layout(const PrefillPlanCache& cache);

// Whether the plan ran in graph mode, i.e. whether its launch geometry is a
// pure function of (total_tokens, num_requests) rather than of the KV content
// it was planned against. Reads `PrefillPlanCache::graph_capturable`, which
// only the `.cuh` defining the struct can see -- the callers that need the
// answer are plain `.cpp` translation units.
bool prefill_plan_graph_capturable(const PrefillPlanCache& cache);

// Whether this plan's schedule is independent of the page counts it was
// planned against, and therefore whether the launch may be handed a different
// (compacted) page list than the plan saw. Only the static non-split decode
// plan qualifies. See `DecodePlanCache::page_count_independent`.
bool decode_plan_is_page_count_independent(const DecodePlanCache& cache);

// Compute decode plan once per fire. Stores results in `cache` and the
// workspace's int/float buffers (so per-layer dispatch can read them).
// Place this plan's descriptor `bytes` into the shared int workspace. Callers
// holding two plans at once need it: the planners otherwise all carve from
// offset 0, and two plans over different request counts do not agree on where
// their fields sit. Declared here because `DecodePlanCache` is opaque to the
// model translation units.
void set_decode_plan_int_base(DecodePlanCache& cache, std::size_t bytes);

void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph = true,
    bool full_attention_variant = false,
    bool hnd_layout = false);

inline void plan_attention_flashinfer_decode(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph = true,
    bool full_attention_variant = false,
    bool hnd_layout = false) {
    plan_attention_flashinfer_decode_bf16(
        cache, kv_page_indptr_h, num_requests, num_q_heads, num_kv_heads,
        head_dim, page_size, workspace, stream, enable_cuda_graph,
        full_attention_variant, hnd_layout);
}

void plan_attention_flashinfer_prefill_bf16(
    PrefillPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph = true,
    int window_left = -1,
    bool full_attention_variant = false,
    bool hnd_layout = false,
    bool causal_mask = true,
    bool custom_mask = false,
    // Set when the caller intends to dispatch through
    // `dispatch_attention_flashinfer_prefill_capture_bf16`. Only the FA2
    // kernel is instrumented, and SM90-vs-FA2 is decided HERE, at plan time --
    // so the intent has to reach the planner or the capture dispatch would
    // find an SM90 plan it can only refuse.
    bool wants_prefill_score = false);

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
// `lse_out`: when non-null, flashinfer writes per-(token, q_head) log-sum-exp
// (natural log of the unnormalized softmax denominator) into this buffer
// before the final divide. Used by GPT-OSS sink-attention to apply the
// post-hoc denominator-extension correction `o *= sigmoid(lse - sink_h)`.
// Layout: row-major [num_tokens, num_q_heads] floats. nullptr = skip
// (default; no overhead).
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
    float sm_scale = -1.f,
    float* lse_out = nullptr,
    // Every request reads the same Q row -- the KV split issues one query
    // against several page slices, so only the input is shared.
    bool broadcast_q = false);

// Score-observing decode (design doc §3): identical attention output, plus
// `p[head, kv_idx]` written to `score_out` for every request in the batch.
// This is what H2O (arXiv:2306.14048) and TOVA (arXiv:2305.19370) evict on.
//
// `score_out` is RAGGED, because requests in a decode batch have unrelated
// `kv_len`s and a dense `[R, H, max_kv_len]` buffer would be mostly padding:
//
//     score_out[score_indptr[r] + h * kv_len(r) + kv_idx]
//
// `score_indptr` is a device buffer of `num_requests + 1` int32 element
// offsets; the caller sizes `score_out` to `score_indptr[num_requests]`.
//
// What lands there is the NORMALISED attention probability: the variant
// records the scaled pre-softmax logit and a follow-up kernel divides by the
// row's own softmax denominator, so each `[h, :]` row sums to 1 over the
// request's live KV. That second pass is exact, not an approximation — at
// decode `qo_len == 1`, so the captured row IS the full softmax input.
//
// Throws `std::invalid_argument` for configurations where a captured score
// would not mean what the eviction policies assume: `logits_soft_cap > 0`
// (the score is rewritten by `cap * tanh(s/cap)`) or `window_left >= 0`
// (sliding window masks positions *after* the capture hook runs).
void dispatch_attention_flashinfer_decode_capture_bf16(
    const DecodePlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* score_out,
    const std::int32_t* score_indptr_d,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

void dispatch_attention_flashinfer_decode(
    const DecodePlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

// As `dispatch_attention_flashinfer_decode`, but also records the attention
// probability each live KV position received. Same refusals as the `_bf16`
// entry point above.
void dispatch_attention_flashinfer_decode_capture(
    const DecodePlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* score_out,
    const std::int32_t* score_indptr_d,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

// Average the `[num_q_heads, kv_len(r)]` probability rows `score_out` holds
// into one `[kv_len(r)]` row per request, written at
// `folded + score_indptr[r] / num_q_heads`.
//
// Folding is not a convenience: the paged layout carries one page list per
// request, so an eviction policy cannot act on a per-head keep-set. Averaging
// (not summing) keeps the result a distribution over the live prefix.
void launch_attn_score_fold_heads(
    const float* scores,
    const std::int32_t* score_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int page_size,
    int num_requests,
    int num_q_heads,
    float* folded,
    cudaStream_t stream);

void dispatch_attention_flashinfer_prefill_bf16(
    const PrefillPlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

// Score-observing prefill (design doc §12): identical attention output, plus
// the OBSERVATION WINDOW's attention probabilities. This is what SnapKV
// (arXiv:2404.14469) selects on -- it asks which prefix positions the tail of
// the prompt actually looked at, then keeps those and drops the rest before
// the first decode step.
//
// Only the last `window` query rows are recorded, because those are the ones
// SnapKV's selection is defined over and recording all of them would be
// O(qo_len * kv_len) per head. Layout, ragged over the batch:
//
//     score_out[score_indptr[r] + (h * window + w) * kv_len(r) + kv_idx]
//
// with `w = qo_idx - (qo_len - rows)` and `rows = min(window, qo_len)`. A
// prompt shorter than the window records fewer rows and LEAVES THE REST OF ITS
// SLOT UNTOUCHED, so `score_out` must be zeroed by the caller.
//
// What lands there is the causal softmax of the recorded rows: the variant
// records the scaled pre-softmax logit for every `(q, kv)` pair the kernel
// evaluates -- including pairs the causal mask later discards, since
// `LogitsMask` runs after `LogitsTransform` -- and the normalisation pass
// zeroes everything past window row `w`'s causal limit before taking the
// softmax over what remains.
//
// `folded_out` receives the head- and row-averaged distribution, one
// `[kv_len(r)]` row per request at `folded_out + score_indptr[r] / (heads *
// window)`. The divisor is `heads * rows`, not `heads * window`: rows that do
// not exist must not dilute a short prompt's mass.
//
// Throws for configurations where a captured score would not mean what SnapKV
// assumes (`logits_soft_cap > 0`, `window_left >= 0`, non-full-attention
// variant), and for an SM90 plan -- the Hopper kernel takes a different
// variant API and is not instrumented. Plan with `wants_prefill_score` set so
// the planner picks FA2.
void dispatch_attention_flashinfer_prefill_capture_bf16(
    const PrefillPlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* score_out,
    float* folded_out,
    const std::int32_t* score_indptr_d,
    int window,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

// Custom-mask dispatch against a plan prepared outside the graph capture
// region. Pointer arguments are device-persistent and may be captured/replayed.
void dispatch_attention_flashinfer_prefill_custom_bf16(
    const PrefillPlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

void dispatch_attention_flashinfer_prefill_custom(
    const PrefillPlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

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
    float sm_scale = -1.f,
    // See decode entry point. [total_tokens, num_q_heads] fp32, nullptr = skip.
    float* lse_out = nullptr,
    bool hnd_layout = false);

void launch_attention_flashinfer_prefill(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

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
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    // See decode entry point. [total_tokens, num_q_heads] fp32, nullptr = skip.
    float* lse_out = nullptr,
    bool hnd_layout = false);

void launch_attention_flashinfer_prefill_custom(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t*  mask_d,
    const std::int32_t*  mask_indptr_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

}  // namespace pie_cuda_driver::ops
