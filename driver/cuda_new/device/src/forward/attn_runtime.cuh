#pragma once

// Attention runtime glue between the forwards and the fast paged-attention
// wrapper (ops/attention_paged). Keeps the lifted `attention_paged.{cu,hpp}`
// pristine: this TU owns the per-fire *policy* (lazy workspace alloc, the tiny
// D2H of the index arrays the host scheduler needs, decode-vs-prefill gating,
// and the naive-fallback decision) so each forward just calls
// `plan_attention_for_fire` once then `run_attention_layer` per layer.

#include <cstdint>

#include <cuda_runtime.h>

#include "ops/attention_paged.hpp"

struct PieWorkspace;  // workspace.hpp

namespace pie_cuda_device::forward {

// Resolved attention plan for one fire, reused across every layer. When
// `ws == nullptr` the head_dim isn't one the tensor-core templates instantiate
// ({64,128,256,512}) and `run_attention_layer` falls back to naive paged attn.
struct AttnPlan {
    ops::AttentionWorkspace* ws = nullptr;
    const ops::PrefillPlanCache* prefill = nullptr;
    const ops::DecodePlanCache* decode = nullptr;
    bool use_decode = false;     // decode kernel (qo_len==1 + GQA in {1,2,3,4,8})
    int num_tokens = 0;
    int num_requests = 0;
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int page_size = 0;
};

// Lazily allocate the attention plan scratch + caches in `ws` (first use), then
// plan THIS fire on `stream`. Does a small synchronous D2H of qo_indptr /
// kv_page_indptr (R+1 ints each) so the host scheduler can run. Returns an
// AttnPlan ready for per-layer dispatch (ws==nullptr ⇒ caller gets the naive
// fallback automatically). `full_attention_variant` selects the no-sliding-
// window template family; `window_left` >= 0 / `causal_mask` mirror the
// dispatch knobs (forwarded into the prefill plan).
AttnPlan plan_attention_for_fire(
    PieWorkspace* ws,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    int num_tokens, int num_requests,
    int n_q_heads, int n_kv_heads, int head_dim, int page_size,
    cudaStream_t stream,
    bool enable_cuda_graph = false,
    bool full_attention_variant = false,
    int window_left = -1,
    bool causal_mask = true);

// One layer's attention. Dispatches the cached tensor-core decode/prefill
// kernel, or naive paged attention when `p.ws == nullptr`. `o` receives
// [num_tokens, n_q_heads, head_dim] bf16 (same layout as the naive op).
void run_attention_layer(
    const AttnPlan& p,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

}  // namespace pie_cuda_device::forward
