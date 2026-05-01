#pragma once

// Gemma-4 (E2B / E4B family — text-only stripped of the multimodal
// towers). Architectural surface beyond Gemma-3:
//
//   * Per-Layer Embeddings (PLE): a 256-dim auxiliary residual stream
//     fed by `embed_tokens_per_layer` (one row per (token, layer))
//     plus a shared `per_layer_model_projection` of the main embed.
//     A per-layer (gate, projection, post-norm) triple injects this
//     signal back into the main residual stream after each MLP block.
//   * KV-cache sharing: the last `num_kv_shared_layers` layers reuse
//     K/V from the most recent non-shared layer of the same attention
//     type (sliding vs full). Shared layers store no K/V, run only a
//     Q-projection, and read attention through the source layer's
//     paged cache.
//   * Per-layer-type head_dim: sliding layers use `head_dim` (256 on
//     E2B), full-attention layers use `global_head_dim` (512). The
//     KV cache is allocated per-layer via `KvCache::allocate_per_layer`.
//   * Per-layer intermediate (`use_double_wide_mlp`): shared layers
//     have `intermediate_size * 2` MLP width.
//   * `sm_scale = 1.0` — Q/K-norm absorbs the usual `1/sqrt(d)` factor.
//   * Plain RMSNorm (`w * x_hat`, *not* Gemma-2's `(1+w) * x_hat`).
//   * V-Norm: pure RMSNorm (no learnable gamma) on V before the KV
//     write. Implemented as an RMSNorm with gamma=1.
//   * Per-layer learnable scalar: each layer's output is multiplied
//     by a scalar `layer_scalar`.
//   * Final logit soft-cap (cap=30).
//   * Proportional RoPE on full-attention layers (only the lower
//     `partial_rotary_factor * head_dim` dims are rotated). The first
//     pass of this implementation rotates the full head_dim — that's
//     a documented approximation; we'll wire `partial_rotary_factor`
//     into the rope kernel as a follow-up.
//
// Out of scope here: the multimodal towers (`audio_config`,
// `vision_config`) and the `enable_moe_block` MoE variant of Gemma-4.

#include <cstdint>
#include <vector>

#include "engine.hpp"
#include "kv_cache.hpp"
#include "model/llama_like.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

struct Gemma4LayerWeights {
    // Four RMSNorms — same placement as Gemma-2/3.
    const DeviceTensor* attn_norm_pre  = nullptr;  // input_layernorm
    const DeviceTensor* attn_norm_post = nullptr;  // post_attention_layernorm
    const DeviceTensor* mlp_norm_pre   = nullptr;  // pre_feedforward_layernorm
    const DeviceTensor* mlp_norm_post  = nullptr;  // post_feedforward_layernorm

    // Q is always present; K/V/V-norm are nullptr on shared layers.
    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* k_proj = nullptr;
    const DeviceTensor* v_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;
    const DeviceTensor* q_norm = nullptr;
    const DeviceTensor* k_norm = nullptr;

    // MLP. `intermediate` may differ per layer (double-wide on shared
    // layers when `use_double_wide_mlp` is set).
    const DeviceTensor* gate_proj = nullptr;
    const DeviceTensor* up_proj   = nullptr;
    const DeviceTensor* down_proj = nullptr;

    // PLE per-layer triple.
    const DeviceTensor* ple_input_gate = nullptr;  // [hidden_per_layer_input, hidden]
    const DeviceTensor* ple_projection = nullptr;  // [hidden, hidden_per_layer_input]
    const DeviceTensor* ple_norm       = nullptr;  // [hidden]

    // Per-layer learnable scalar. 1-element bf16 tensor.
    const DeviceTensor* layer_scalar  = nullptr;

    // Per-layer dimensions (filled in by `bind_gemma4` from layer_types).
    int head_dim     = 0;
    int intermediate = 0;
    int kv_source    = 0;   // == layer index when not shared
    bool is_full     = false;
    bool is_shared   = false;
};

struct Gemma4Weights {
    const DeviceTensor* embed       = nullptr;       // model.language_model.embed_tokens.weight
    const DeviceTensor* embed_per_layer = nullptr;   // [vocab, num_layers * ple_dim]
    const DeviceTensor* ple_model_proj = nullptr;    // [num_layers * ple_dim, hidden]
    const DeviceTensor* ple_model_norm = nullptr;    // [ple_dim]
    const DeviceTensor* final_norm  = nullptr;       // model.language_model.norm
    const DeviceTensor* lm_head     = nullptr;       // tied to embed unless lm_head.weight present
    std::vector<Gemma4LayerWeights> layers;

    // Cached per-layer arrays for the forward / KV-cache allocator.
    std::vector<int> per_layer_head_dim;
    std::vector<int> per_layer_intermediate;
    std::vector<int> kv_source_layer;
    std::vector<int> per_layer_window_left;
    std::vector<float> per_layer_rope_theta;
    std::vector<float> per_layer_partial_rotary_factor;
    std::vector<int> full_layer_indices;  // for debug / introspection
};

struct Gemma4ForwardCfg {
    // Final logit soft-cap (defaults to Gemma-4-E2B's 30).
    float final_logit_softcap = 30.f;

    // Force the prefill kernel for batches whose GQA group size isn't
    // in flashinfer's decode dispatch table — same role as in
    // `LlamaLikeForwardCfg`.
    bool force_prefill_path = false;
};

// Bind helper: validates the Gemma-4 schema, populates per-layer
// dimensions + KV-source mapping from `HfConfig::layer_types` and
// `num_kv_shared_layers`. Throws on missing tensors.
Gemma4Weights bind_gemma4(const Engine& engine);

void gemma4_forward_paged(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
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
