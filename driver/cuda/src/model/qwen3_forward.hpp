#pragma once

// Qwen3 forward pass — single-sequence prefill, no KV cache, no batching.
// **Parity-test path only.** The full wire-driven path with paged KV lands
// in M1.3.

#include <cstdint>

#include "engine.hpp"
#include "model/qwen3.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

// Reusable scratch buffers, sized once for `max_tokens`. The forward pass
// only writes prefixes of these, so reusing across calls is safe as long as
// you don't exceed `max_tokens`.
struct Qwen3Workspace {
    DeviceTensor y;          // [max_tokens, hidden]
    DeviceTensor norm_x;     // [max_tokens, hidden]
    DeviceTensor qkv_fused;  // [max_tokens, Hq + 2*Hk]   — only allocated when fused
                             // QKV path is in use; empty otherwise.
    DeviceTensor q;          // [max_tokens, h_q  * head_dim]   — packed
    DeviceTensor k;          // [max_tokens, h_kv * head_dim]   — packed
    DeviceTensor v;          // [max_tokens, h_kv * head_dim]   — packed
    DeviceTensor attn_out;   // [max_tokens, h_q  * head_dim]   — packed
    DeviceTensor norm_y;     // [max_tokens, hidden]
    DeviceTensor gate_up_fused; // [max_tokens, 2*I] — fused gate+up output, empty
                                // when unfused
    DeviceTensor gate;       // [max_tokens, intermediate]
    DeviceTensor up;         // [max_tokens, intermediate]
    DeviceTensor logits;     // [max_tokens, vocab]
    DeviceTensor probs;      // [max_tokens, vocab] FP32 — softmax scratch for sampling

    // Padded variants for the attention kernel when `head_dim_kernel >
    // head_dim` (Phi-3 ships head_dim=96; flashinfer's TC kernel only
    // works at {64, 128, 256, 512}, so we round up to 128). Empty
    // (numel()==0) for every other model — the forward graph aliases
    // the packed buffers directly.
    DeviceTensor q_padded;        // [max_tokens, h_q  * head_dim_kernel]
    DeviceTensor k_padded;        // [max_tokens, h_kv * head_dim_kernel]
    DeviceTensor v_padded;        // [max_tokens, h_kv * head_dim_kernel]
    DeviceTensor attn_out_padded; // [max_tokens, h_q  * head_dim_kernel]

    static Qwen3Workspace allocate(const HfConfig& cfg, int max_tokens);

    // Variant for architectures whose per-layer MLP `intermediate_size`
    // exceeds the base `cfg.intermediate_size` (Gemma-4's
    // `use_double_wide_mlp` doubles the width on shared layers).
    // Caller passes the worst-case value; ws.gate / ws.up / logits are
    // sized accordingly. Other shapes match the standard `allocate`.
    static Qwen3Workspace allocate_with_max_intermediate(
        const HfConfig& cfg, int max_tokens, int max_intermediate);

    // Variant for architectures whose per-layer attention dimensions
    // (Hq = num_q_heads * head_dim, Hk = num_kv_heads * head_dim) vary
    // across layers — Gemma-4's full-attention layers run at
    // head_dim_global=512 while sliding layers run at head_dim=256, so
    // a single ws.q sized at the sliding width overflows on full
    // layers. Caller passes the worst-case `Hq` and `Hk`.
    static Qwen3Workspace allocate_full(
        const HfConfig& cfg, int max_tokens,
        int max_intermediate, int max_Hq, int max_Hk);
};

// Run prefill on `num_tokens` consecutive tokens starting at position 0.
// Inputs `token_ids` and `positions` are device pointers. Writes
// `ws.logits[:num_tokens, :]`.
void qwen3_forward_prefill(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,    // device, [num_tokens]
    const std::int32_t* positions,    // device, [num_tokens]
    int num_tokens);

}  // namespace pie_cuda_driver::model

// ── Paged variant ──────────────────────────────────────────────────────────
#include "attention_workspace.hpp"
#include "kv_cache.hpp"

namespace pie_cuda_driver::model {

// Same forward pass but routes K/V through the paged KV pool, matching the
// wire contract. Each layer's K/V is written into `cache` at the locations
// described by the page-indptr arrays, and the attention call reads back
// from those same pages.
//
// `is_pure_decode` selects between the flashinfer paged decode kernel
// (when every request has qo_len == 1) and the naive reference path.
// Phase 2 will add the flashinfer prefill kernel; until then prefill
// stays on the naive path.
void qwen3_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,           // device, [total_tokens]
    const std::int32_t* positions,           // device, [total_tokens]
    const std::uint32_t* qo_indptr,          // device, [num_requests + 1]
    const std::uint32_t* kv_page_indices,    // device
    const std::uint32_t* kv_page_indptr,     // device, [num_requests + 1]
    const std::uint32_t* kv_last_page_lens,  // device, [num_requests]
    const std::uint32_t* qo_indptr_h,        // host pointer, [num_requests + 1]
    const std::uint32_t* kv_page_indptr_h,   // host pointer, [num_requests + 1]
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    // Optional custom mask. When non-null, the prefill path uses
    // flashinfer's MaskMode::kCustom; ignored on the decode path.
    const std::uint8_t*  custom_mask_d = nullptr,
    const std::int32_t*  custom_mask_indptr_d = nullptr);

}  // namespace pie_cuda_driver::model
