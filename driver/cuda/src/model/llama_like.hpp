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

#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "model/qwen3.hpp"           // Qwen3Weights / Qwen3Workspace shared
#include "model/qwen3_forward.hpp"   // Qwen3Workspace (already declared)
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_xqa.hpp"

namespace pie_cuda_driver::model {

enum class RopeKind {
    Standard,      // pure theta-based, used by Qwen 2/3, Phi-3, Mistral
    YaRN,          // Llama-3 smoothed-interpolation YaRN
    YaRNOriginal,  // Original YaRN (OLMo-3, gpt-oss): dim-index ramp +
                   // attention_factor mscale (Peng et al. 2023)
    MRopeInterleaved,  // Qwen3-VL interleaved 3-axis M-RoPE (t,h,w). Reads
                       // `mrope_positions` ([N,3]); requires per-head q/k norm.
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

    // YaRN params (only consumed when `rope_kind == YaRN` or
    // `YaRNOriginal`).
    float yarn_factor               = 1.0f;
    float yarn_low_freq_factor      = 1.0f;
    float yarn_high_freq_factor     = 4.0f;
    int   yarn_original_max_position = 8192;
    // Original-YaRN extras (consumed only when `rope_kind ==
    // YaRNOriginal`).
    float yarn_beta_fast            = 32.0f;
    float yarn_beta_slow            = 1.0f;
    float yarn_attention_factor     = 1.0f;

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
    bool use_xqa_decode = false;
    bool decode_plan_cuda_graph = true;
    bool use_prefill_decode_plan = false;
    int prefill_decode_full_attention_min_requests = 0;
    int prefill_decode_full_attention_min_kv_pages = 0;
    int prefill_decode_min_kv_pages = 0;

    // Tensor-parallel state. `tp_size = 1` (default) keeps the original
    // single-GPU forward; `tp_size > 1` activates the sharded GEMM dims
    // and drops in two NCCL all-reduces per layer (after o_proj and after
    // down_proj). `tp_comm` must be non-null whenever tp_size > 1.
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;

    // TP followers do not sample or build responses. After the final layer
    // all-reduce there are no more collectives, so they can skip the rank-0
    // logits tail.
    bool emit_logits = true;

    // ── Qwen3-VL M-RoPE ──────────────────────────────────────────────
    // mrope_section partitions head_dim/2 across the (t,h,w) axes. Consumed
    // only when `rope_kind == MRopeInterleaved`. The 3-axis positions are
    // supplied per-fire via `mrope_positions` (see llama_like_forward_paged).
    int mrope_section_t = 0;
    int mrope_section_h = 0;
    int mrope_section_w = 0;
};

// Per-fire Qwen3-VL multimodal side-inputs threaded into the shared
// llama_like forward. All null / nullptr disables every multimodal hook
// (the forward reduces to a plain Qwen3 decode). See Qwen3VLModel::body.
struct LlamaLikeVisionInputs {
    // Vision encode + scatter after the embed (gated by num_images > 0).
    const struct Qwen3VLVisionInputs* vision_in = nullptr;
    // DeepStack: each deepstack merger output is added to the hidden state on
    // image rows after decoder layers 0/1/2. `deepstack_scratch` is the
    // [num_deep, N, H] bf16 buffer the scatter wrote; `num_deepstack` blocks.
    void* deepstack_scratch = nullptr;
    int   num_deepstack = 0;
    // 3-axis M-RoPE positions [N,3] int32 (device). When non-null and
    // rope_kind==MRopeInterleaved, used in place of the 1-D `positions`.
    const std::int32_t* mrope_positions = nullptr;
};

// Persistent decode-plan cache. Owned in main.cpp's serving setup so the
// per-fire `prepare` hook (which calls `prepare_llama_like_decode_plan`)
// can refresh the plan before the captured body reads from it. Hoisting
// the plan out of the body lets the body live entirely inside a CUDA
// graph capture region — no host-side work, no allocations.
struct LlamaLikePlanState {
    ops::DecodePlanCachePtr decode_plan;
    ops::PrefillPlanCachePtr prefill_plan;
    ops::PrefillPlanCachePtr prefill_decode_plan;
    bool use_prefill_plan = false;
    bool use_prefill_decode_plan = false;
    bool use_xqa_decode = false;
    int xqa_max_pages_per_seq = 0;
    std::vector<std::uint32_t> prefill_decode_qo_indptr_h;
};

// Refresh the decode plan for the current fire. Caller invokes this
// BEFORE either a direct forward call OR a graph replay, outside any
// capture region. Pure decode plans the flashinfer decode/predecode path;
// ordinary prefill plans the reusable flashinfer prefill path when a single
// layer layout is valid for every layer.
void prepare_llama_like_decode_plan(
    LlamaLikePlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_h,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    bool is_pure_decode);

std::uint32_t llama_like_decode_graph_layout(
    const LlamaLikePlanState& state);

// Same call signature as `qwen3_forward_paged`, plus a `cfg` knob block
// and an externally-owned `LlamaLikePlanState`. The body never plans —
// it only reads `state.decode_plan` (already populated by the prepare
// hook) which makes the body graph-capture-safe.
void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
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
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0,
    bool tp_greedy_argmax = false,
    const std::uint8_t* custom_mask_d = nullptr,
    const std::int32_t* custom_mask_indptr_d = nullptr,
    // Qwen3-VL multimodal side-inputs (nullptr = plain text forward).
    const LlamaLikeVisionInputs* vision = nullptr);

// Map HF's rope_scaling_kind enum onto the driver's RopeKind. Llama3-style
// frequency scaling maps to YaRN; the "original_yarn" branch keeps
// HuggingFace's original formulation.
RopeKind rope_kind_from_hf_config(const HfConfig& hf);

// Populate the RoPE-related fields on LlamaLikeForwardCfg from the
// HF config in one place — every arch that builds an LlamaLikeForwardCfg
// in entry.cpp pulls in the same eight fields.
void apply_rope_config(LlamaLikeForwardCfg& fwd_cfg, const HfConfig& hf);

}  // namespace pie_cuda_driver::model
