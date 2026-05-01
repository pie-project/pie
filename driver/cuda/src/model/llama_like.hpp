#pragma once

// Llama-like decoder forward — covers every "transformer block with
// pre-norm + QKV/o + gate-up-down" architecture in pie_driver:
// llama, qwen2, qwen3, phi3, olmo (post-norm variant), mistral (bf16
// fallback). Per-arch knobs live in `LlamaLikeForwardCfg`; the schema
// builder (per-model `bind_*`) chooses the right combination.
//
// Out-of-scope here (handled by their own forwards):
//   * Gemma family — needs four-norm-per-layer, query pre-scale, GELU,
//     embed √-scale, logit soft-cap.
//   * Mixtral / GPT-OSS — sparse MoE replaces the MLP block.
//   * Qwen-3.5 — hybrid full + linear-attention layers.
//   * Gemma-4 — KV sharing across layers, per-layer embeds.

#include <cstdint>
#include <vector>

#include "engine.hpp"
#include "model/qwen3.hpp"           // Qwen3Weights / Qwen3Workspace shared
#include "model/qwen3_forward.hpp"   // Qwen3Workspace (already declared)
#include "ops/attention_flashinfer.hpp"

namespace pie_cuda_driver::model {

enum class RopeKind {
    Standard,  // pure theta-based, used by Qwen 2/3, Phi-3, Mistral
    YaRN,      // Llama-3 / OLMo-3 / GPT-OSS scaling
};

enum class NormPlacement {
    Pre,    // standard Llama / Qwen / Mistral / Phi: norm before sub-layer
    Post,   // OLMo-3: norm after sub-layer, then residual add
};

struct LlamaLikeForwardCfg {
    // Per-fire toggles.
    bool use_qk_norm        = false;  // Qwen3 / Gemma-3 / OLMo-3
    bool use_qkv_bias       = false;  // Qwen-2 / OLMo-3 / GPT-OSS
    NormPlacement norm_placement = NormPlacement::Pre;
    RopeKind rope_kind      = RopeKind::Standard;

    // YaRN params (only consumed when `rope_kind == YaRN`).
    float yarn_factor               = 1.0f;
    float yarn_low_freq_factor      = 1.0f;
    float yarn_high_freq_factor     = 4.0f;
    int   yarn_original_max_position = 8192;

    // Sliding-window attention. `sliding_window = -1` means full causal
    // for every layer; positive values switch flashinfer's
    // `window_left`. When `per_layer_window_left` is non-empty, it
    // overrides the scalar (one entry per layer; -1 = full causal,
    // ≥ 0 = sliding window with that left-context). Used by OLMo-3 and
    // Gemma-3 to alternate full / sliding attention per layer.
    int sliding_window = -1;
    std::vector<int> per_layer_window_left;

    // Force the prefill kernel even for is_pure_decode batches. Used
    // for models whose GQA group size isn't in flashinfer's decode
    // dispatch table {1, 2, 3, 4, 8} — Qwen2-0.5B (group=7),
    // Qwen2-1.5B (group=6), etc. The prefill kernel uses a runtime
    // fastdiv for group_size and accepts arbitrary values; cost is
    // ~1.3× per-step latency vs the dedicated decode kernel.
    bool force_prefill_path = false;
};

// Same call signature as `qwen3_forward_paged`, plus a `cfg` knob block.
void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* custom_mask_d = nullptr,
    const std::int32_t* custom_mask_indptr_d = nullptr);

}  // namespace pie_cuda_driver::model
