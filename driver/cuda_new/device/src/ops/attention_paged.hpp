#pragma once

// Fast paged attention — the perf path that replaces the correctness-first
// naive two-pass softmax (ops/attention_naive_paged.cu). Tensor-core kernels
// over a paged KV cache, driven by a host-side scheduler (plan) that the
// per-layer device dispatch reuses.
//
// Internalized + de-branded lift of driver/cuda's flashinfer attention wrapper
// (cf. `qgemm` = internalized Marlin): only the upstream `::flashinfer::`
// template *types* remain (a header-only dependency, like cutlass); our file,
// symbols, and surface are brand-neutral. Lifted scope = the raw-pointer bf16
// entry points only. Dropped relative to the source:
//   - the `KvCacheLayerView` overloads + their dequant-active prologue (the new
//     device lib threads raw page pointers; KV-dtype dequant runs via
//     kernels/dequant_kv_active before the bf16 dispatch instead),
//   - the SM90/Hopper FA3 fast path — base tensor-core prefill/decode is the v1
//     perf path on sm_90; FA3 is a follow-on.
//
// Two-phase plan/dispatch split (the CUDA-graph-safety contract): `plan_*` runs
// the host-side scheduler ONCE per fire (vector fills + async H2D into the
// workspace int buffer), writing the scheduler's PlanInfo into a cache;
// `dispatch_*` then runs the (graph-capturable) kernel per layer reusing that
// cache. q/k_pages/v_pages/o vary per layer; everything else comes from the
// plan + workspace.

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "ops/attention_workspace.hpp"

namespace pie_cuda_device::ops {

// Opaque cache of the scheduler's Decode/Prefill PlanInfo + the fields dispatch
// needs. Created once (e.g. per model), reset each fire by the matching
// `plan_*`, reused by the per-layer `dispatch_*` calls in that fire.
struct DecodePlanCache;
struct DecodePlanCacheDeleter { void operator()(DecodePlanCache*) const noexcept; };
using DecodePlanCachePtr = std::unique_ptr<DecodePlanCache, DecodePlanCacheDeleter>;
DecodePlanCachePtr make_decode_plan();

struct PrefillPlanCache;
struct PrefillPlanCacheDeleter { void operator()(PrefillPlanCache*) const noexcept; };
using PrefillPlanCachePtr = std::unique_ptr<PrefillPlanCache, PrefillPlanCacheDeleter>;
PrefillPlanCachePtr make_prefill_plan();

// Compact graph-layout key for the most recent plan. CUDA-graph replay records
// the host-side dispatch branch, so split-KV / non-split / tile-size plans need
// distinct graph keys.
std::uint32_t decode_plan_graph_layout(const DecodePlanCache& cache);
std::uint32_t prefill_plan_graph_layout(const PrefillPlanCache& cache);

// ── Decode (every request qo_len == 1) ──────────────────────────────────────

// Plan once per fire. Stores results in `cache` + the workspace int/float
// buffers (so per-layer dispatch can read them).
void plan_attention_paged_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph = true,
    bool full_attention_variant = false,
    bool hnd_layout = false);

// Per-layer dispatch reusing the cached plan. See the prefill notes below for
// window_left / logits_soft_cap / sm_scale / lse_out semantics.
void dispatch_attention_paged_decode_bf16(
    const DecodePlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

// ── Prefill (or mixed prefill+decode); per-request qo_len from qo_indptr ─────

void plan_attention_paged_prefill_bf16(
    PrefillPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph = true,
    int window_left = -1,
    bool full_attention_variant = false,
    bool hnd_layout = false,
    bool causal_mask = true);

void dispatch_attention_paged_prefill_bf16(
    const PrefillPlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr);

// One-shot prefill (plan + dispatch, no cached plan reuse). Causal mask
// hard-wired. `window_left` >= 0 enables sliding window. `logits_soft_cap` > 0
// enables Gemma-style cap*tanh(logits/cap). `sm_scale` < 0 => 1/sqrt(head_dim).
// `lse_out` (optional) [total_tokens, num_q_heads] fp32 receives per-row LSE.
void launch_attention_paged_prefill_bf16(
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr,
    bool hnd_layout = false);

// Prefill with a custom packed-bit mask per request (constrained decoding).
// `mask_d` concatenates per-request bitmaps; `mask_indptr_d[r]` is request r's
// byte offset. Each request's mask is qo_len_r × kv_len_r bits, row-major
// (qo_idx*kv_len + kv_idx).
void launch_attention_paged_prefill_custom_bf16(
    const void* q, void* k_pages, void* v_pages, void* o,
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
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left = -1,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr,
    bool hnd_layout = false);

}  // namespace pie_cuda_device::ops
