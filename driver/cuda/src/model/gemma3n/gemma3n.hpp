#pragma once

// Gemma-3n (E2B / E4B "Nano" family). A separate architecture from
// Gemma-4 despite the overlapping E2B/E4B naming — Gemma-3n adds three
// non-trivial blocks to the standard pre-norm Gemma layer:
//
//   * AltUp — "Alternating Updates"
//     Each layer maintains `altup_num_inputs` (4 by default) parallel
//     residual streams. Only the active stream (`altup_active_idx`)
//     flows through attention + MLP; the other streams are updated via
//     per-layer "predict" and "correct" matmuls routed by a per-token
//     "modality" vector.
//
//     predict step (called BEFORE the layer body):
//       modalities = tanh(modality_router(router_norm(active) * H^-1))   # [B,T,K]
//       all_coefs  = prediction_coefs(modalities).reshape(K,K).permute   # [B,T,K,K]
//       predictions = matmul(hidden_states.permute, all_coefs).permute + hidden_states
//
//     correct step (after the layer body):
//       modalities = tanh(modality_router(router_norm(activated) * H^-1))
//       innovation = activated - predictions[active]
//       all_coefs  = correction_coefs(modalities) + 1.0                  # [B,T,K]
//       corrected  = innovation.broadcast(K) * all_coefs.permute + predictions
//       active'    = corrected[active] * correct_output_scale            # if altup_correct_scale
//
//   * Laurel — "Learned Augmented Residual Layer"
//     Per-layer low-rank skip:
//       laurel_out = active_normed + post_laurel_norm(linear_right(linear_left(active_normed)))
//     `linear_left`  is [laurel_rank, hidden]; `linear_right` is [hidden, laurel_rank].
//
//   * Activation sparsity (per layer)
//     For layers with `activation_sparsity_pattern[layer] > 0` (only the
//     first ~10 layers on E2B), the SwiGLU gate is hard-sparsified via
//     a Gaussian-quantile cutoff before SiLU:
//       cutoff = mean(gate) + std(gate) * Φ⁻¹(target_sparsity)
//       gate'  = relu(gate - cutoff)
//
// Plus PLE (Per-Layer Embeddings — same structure as Gemma-4) and a
// per-layer (input_gate, projection, post_norm) trio that gates the PLE
// signal into the residual stream.
//
// Out of scope for this header: the multimodal towers (`audio_config`,
// `vision_config`) and any MoE variant.

#include <cstdint>
#include <vector>

#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "store/kv_cache.hpp"
#include "model/llama_like/qwen3.hpp"
#include "model/workspace.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

struct Gemma3nLayerWeights {
    // Standard pre-norm + post-norm RMSNorms (Gemma-2/3/4 placement).
    const DeviceTensor* attn_norm_pre  = nullptr;  // input_layernorm
    const DeviceTensor* attn_norm_post = nullptr;  // post_attention_layernorm
    const DeviceTensor* mlp_norm_pre   = nullptr;  // pre_feedforward_layernorm
    const DeviceTensor* mlp_norm_post  = nullptr;  // post_feedforward_layernorm

    // Q/K/V/O. K/V may be nullptr on KV-shared layers (last
    // `num_kv_shared_layers` layers reuse from earlier).
    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* k_proj = nullptr;
    const DeviceTensor* v_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;
    const DeviceTensor* q_norm = nullptr;
    const DeviceTensor* k_norm = nullptr;

    // MLP. `intermediate` may differ per layer (HF stores the list in
    // `intermediate_size`).
    const DeviceTensor* gate_proj = nullptr;
    const DeviceTensor* up_proj   = nullptr;
    const DeviceTensor* down_proj = nullptr;

    // ── AltUp ──
    const DeviceTensor* altup_correct_output_scale = nullptr; // [hidden]
    const DeviceTensor* altup_correction_coefs    = nullptr;  // [K, K]
    const DeviceTensor* altup_prediction_coefs    = nullptr;  // [K^2, K]
    const DeviceTensor* altup_modality_router     = nullptr;  // [K, hidden]
    const DeviceTensor* altup_router_norm         = nullptr;  // [hidden]

    // ── Laurel ──
    const DeviceTensor* laurel_left      = nullptr;  // [laurel_rank, hidden]
    const DeviceTensor* laurel_right     = nullptr;  // [hidden, laurel_rank]
    const DeviceTensor* laurel_post_norm = nullptr;  // [hidden]

    // ── PLE per-layer trio ──
    // Same structure as gemma4: gate (H → H_ple), projection
    // (H_ple → H), then RMSNorm.
    const DeviceTensor* ple_input_gate = nullptr;      // [H_ple, H]
    const DeviceTensor* ple_projection = nullptr;      // [H, H_ple]
    const DeviceTensor* ple_post_norm  = nullptr;      // [H]

    // Per-layer activation sparsity target (0.0 = no sparsity).
    float activation_sparsity = 0.f;
    // Per-layer intermediate size (HF stores the list).
    int intermediate = 0;
    // KV source layer: == layer index when not shared, otherwise the
    // earlier layer this one reuses K/V from.
    int kv_source = 0;
    bool is_full   = false;
    bool is_shared = false;
};

struct Gemma3nWeights {
    const DeviceTensor* embed       = nullptr;      // model.language_model.embed_tokens.weight

    // PLE token-table + projection + norm. Same names as Gemma-4.
    const DeviceTensor* embed_per_layer    = nullptr;  // [vocab_per_layer, num_layers * H_ple]
    const DeviceTensor* ple_model_proj     = nullptr;  // [num_layers * H_ple, H]
    const DeviceTensor* ple_model_proj_norm = nullptr; // [H_ple]

    // AltUp top-level projections. K-1 input projections + K-1 unembed
    // projections, where K = altup_num_inputs.
    std::vector<const DeviceTensor*> altup_projections;          // [H, H] each, size K-1
    std::vector<const DeviceTensor*> altup_unembed_projections;  // [H, H] each, size K-1

    const DeviceTensor* final_norm = nullptr;  // model.language_model.norm.weight
    const DeviceTensor* lm_head    = nullptr;  // tied to embed unless lm_head.weight present

    std::vector<Gemma3nLayerWeights> layers;

    // Cached per-layer arrays for the forward / KV-cache allocator.
    std::vector<int>   per_layer_intermediate;
    std::vector<int>   per_layer_window_left;
    std::vector<float> per_layer_rope_theta;
    std::vector<int>   kv_source_layer;
};

// Forward config — shared structure with gemma4 for now (gemma3n adds
// the AltUp/Laurel/sparsity flags via HfConfig directly).
struct Gemma3nForwardCfg {
    float final_logit_softcap = 30.f;  // gemma3n config sets this
    bool  force_prefill_path  = false;

    // TP state. tp_size > 1 shards the per-layer attention/MLP weights
    // and inserts all-reduces after o_proj and the per-layer down_proj.
    // AltUp / Laurel / activation-sparsity / PLE weights stay replicated
    // — those sub-streams operate over the full hidden dim on every rank.
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;
};

Gemma3nWeights bind_gemma3n(const LoadedModel& engine);

// Stub for now — throws "not yet implemented". The bind function
// loads every tensor; the forward path is the next milestone.
void gemma3n_forward_paged(
    const Gemma3nWeights& w,
    const HfConfig& cfg,
    const Gemma3nForwardCfg& fwd_cfg,
    Workspace& ws,
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
