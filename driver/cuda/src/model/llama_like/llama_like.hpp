#pragma once

#include <memory>

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
#include <string>
#include <vector>

#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "model/llama_like/qwen3.hpp"           // Qwen3Weights
#include "model/workspace.hpp"       // Workspace
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_xqa.hpp"
#include "store/kv_cache.hpp"

namespace pie_cuda_driver::model {

struct StageHooks;
struct LoraTable;

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

    // TP followers do not publish PTIR channels. After the final layer
    // all-reduce there are no more collectives, so they can skip rank-0 logits.
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
class LoraFireStateHandle;

struct LlamaLikePlanState {
    ops::DecodePlanCachePtr decode_plan;
    ops::PrefillPlanCachePtr prefill_plan;
    ops::PrefillPlanCachePtr prefill_decode_plan;
    // Custom-mask PURE-DECODE fires get their OWN plan slot (the
    // supergraph axiom, S3: an arm may not share a mutable plan slot
    // with a foreign fire class). Before this, masked decodes re-planned
    // `prefill_plan` and every request's PREFILL re-planned it back —
    // the layout oscillation cost the union key one orphan capture per
    // request, and today's masked-variant graphs the same churn.
    ops::PrefillPlanCachePtr mask_decode_plan;
    // NS-2: when >= 0, this fire's attention splits at this REQUEST
    // index — the prefix plans cover requests [0, split),
    // mask_decode_plan covers the REBASED suffix. -1 = fire-level
    // plans (the pre-NS-2 shape).
    int spatial_mask_split = -1;
    // The mixed fire (M-2): the split's TOKEN-ROW offset (q/out
    // pointer arithmetic). Equal to spatial_mask_split on pure-decode
    // fires; diverges when prefill rows share the fire. -1 with
    // spatial_mask_split >= 0 never happens (set together).
    int spatial_mask_row_split = -1;
    bool use_prefill_plan = false;
    bool use_prefill_decode_plan = false;
    bool use_mask_decode_plan = false;
    // Set when the prefill plan was built for the FA2 score-capturing
    // dispatch. SM90-vs-FA2 is decided at PLAN time, so the body cannot
    // decide to capture on its own -- it can only honour what the prepare
    // hook already committed to.
    std::uint32_t prefill_score_window = 0;
    // Lora campaign step 3a: the fire's PRE-STAGED lora state, staged by
    // the engine OUTSIDE any capture region (ForwardFn::invoke_lora_stage
    // -> LlamaLikeModel::lora_stage). Bodies consume it read-only and
    // fall back to local staging only when the engine did not stage
    // (e.g. a path that never calls the stage hook). Cleared/re-staged
    // per fire by the stage call itself.
    std::unique_ptr<LoraFireStateHandle> lora_staged;
    const LoraTable* lora_staged_table = nullptr;
    bool use_xqa_decode = false;
    int xqa_max_pages_per_seq = 0;
    std::vector<std::uint32_t> prefill_decode_qo_indptr_h;
};

// The mixed fire's dedicated suffix-plan workspace (llama_like.cpp):
// the prefix causal plan and the suffix custom plan are both
// prefill-family and must not share one AttentionWorkspace's scheduling
// buffers. Prepare plans the suffix against this; the mixed dispatch
// sites pair with it.
AttentionWorkspace& spatial_suffix_attn_ws();

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
    bool is_pure_decode,
    bool have_custom_mask,
    // Non-zero when the fire's PTIR programs read `AttnScore`; the prefill
    // plan is then built for the FA2 score-capturing dispatch. Decided here
    // and not in the body because SM90-vs-FA2 is a plan-time choice.
    std::uint32_t attn_score_window = 0,
    // NS-2: the planned unmasked wire-row prefix (UINT32_MAX = no split).
    // When 0 < value < R on a masked pure-decode fire and PIE_SPATIAL_MASK
    // is armed, prepare builds the PREFIX decode plan and the rebased
    // SUFFIX mask plan, and records the split on the plan state.
    std::uint32_t unmasked_prefix_rows = 0xffffffffu,
    // NS-2: resolved suffix geometry for the mask plan (see
    // ForwardFn::PrepareInputs) — required when the split is active on a
    // composed-envelope fire, else the host CSR slices serve.
    const std::uint32_t* mask_suffix_page_counts_h = nullptr,
    const std::uint32_t* mask_suffix_last_lens_h = nullptr);

std::uint32_t llama_like_supergraph_graph_layout(
    const LlamaLikePlanState& state);

std::uint32_t llama_like_decode_graph_layout(
    const LlamaLikePlanState& state);

// PIE_CUDA_DECODE_FUSED_POST kill switch for the fused decode QKV
// postprocess (default on; the A/B rationale is at the definition).
// Exposed so the declared executor's peephole (declared_forward.cpp) and
// the hand-written `fused_decode_qkv_post` branch read ONE gate — the two
// paths must fuse, or not, together.
bool decode_fused_post_enabled();

// Wire-driven forward body, plus a `cfg` knob block and an
// externally-owned `LlamaLikePlanState`. The body never plans — it only
// reads `state.decode_plan` (already populated by the prepare hook) which
// makes the body graph-capture-safe.
void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
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
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0,
    const std::uint8_t* custom_mask_d = nullptr,
    const std::int32_t* custom_mask_indptr_d = nullptr,
    // Explicit KV-write descriptor (device-geometry WSlot/WOff, B2). When
    // `has_write_desc`, the per-layer KV append routes through the explicit
    // (physical page, offset) kernel instead of the page-derived write.
    const std::uint32_t* w_page_d = nullptr,
    const std::uint32_t* w_off_d = nullptr,
    const std::uint8_t* row_valid_d = nullptr,
    bool has_write_desc = false,
    int runtime_window_left = -2,
    // Qwen3-VL multimodal side-inputs (nullptr = plain text forward).
    const LlamaLikeVisionInputs* vision = nullptr,
    // The fire's stage hooks (`ForwardInputs::stage_hooks`, observation
    // attached by `invoke_body`). Null = no program attached, and every
    // hook-conditional path in the body then folds to its fast form.
    const StageHooks* hooks = nullptr,
    // The fire's resolved lora configuration (`ForwardInputs::lora`,
    // "model/lora.hpp"). Null = no program in the launch carries the sink,
    // and the body is exactly what it was (§5.1: the CORRECTION term
    // vanishes with no adapters). Non-null: the body applies
    // `x(W+BA)^T = xW^T + (xA^T)B^T` at each lane's declared sites, scoped
    // to that lane's token rows.
    const LoraTable* lora = nullptr,
    // The Peel device window word ({tail_start, tail_len} in device
    // memory), non-null ONLY on hook-graph captures: the fused-decode
    // Peel then emits BOTH regions through the devwin kernel forms so the
    // captured exec replays across row splits. Null keeps host windows.
    const std::uint32_t* peel_window_d = nullptr,
    // NS-2: see declared_forward.hpp — the spatial mask split and the
    // rebased suffix CSRs (UINT32_MAX / null = fire-level mask arm).
    std::uint32_t unmasked_prefix_rows = 0xffffffffu,
    const std::uint32_t* mask_suffix_qo_indptr_d = nullptr,
    const std::uint32_t* mask_suffix_kv_page_indptr_d = nullptr,
    // STRUCTURAL v0 (S-1): run layers [0, k) and take the head at k
    // (UINT32_MAX = full model). The layerskip-draft / logit-lens class:
    // the tail (final norm + lm_head) is unchanged — it simply reads
    // layer k's hidden state.
    std::uint32_t max_layers = 0xffffffffu);

// The fire-scoped lora staging (`LoraFireState` in llama_like.cpp — the
// adapter cast + grouping built once per fire), behind an opaque handle so
// the declared executor can run the SAME §5.1 correction the hand-written
// body runs (the `pie_lora_qkv_correction` pseudo-symbol's launcher).
// Constructor stages; `apply` lands one layer's delta on the materialized
// q/v (the hand-written call, argument for argument).
class LoraFireStateHandle {
public:
    LoraFireStateHandle(
        const LoraTable& table,
        const HfConfig& cfg,
        int total_tokens,
        int hidden,
        int q_width,
        int kv_width,
        int intermediate,
        int tp_size,
        cudaStream_t stream,
        Workspace& ws,
        // The fire-constant buffers the STAGE phase bakes into the
        // pointer slab (campaign step 2): the projection input the
        // correction reads (placement-dependent: ws.y post-norm,
        // ws.norm_x pre-norm), the q/v outputs, and the xA^T scratch.
        const void* qkv_in,
        void* q_out,
        void* v_out,
        void* xa_scratch);
    ~LoraFireStateHandle();
    LoraFireStateHandle(const LoraFireStateHandle&) = delete;
    LoraFireStateHandle& operator=(const LoraFireStateHandle&) = delete;
    void apply(
        cublasHandle_t handle,
        int layer,
        const void* qkv_in,
        int hidden,
        int q_width,
        int kv_width,
        void* q_out,
        void* v_out,
        void* xa_scratch) const;
    /// The one-line grouping description the hand-written fire trace
    /// prints (`PIE_LORA_FIRE_TRACE`).
    std::string grouping_desc() const;

private:
    void* impl_;
};

// Map HF's rope_scaling_kind enum onto the driver's RopeKind. Llama3-style
// frequency scaling maps to YaRN; the "original_yarn" branch keeps
// HuggingFace's original formulation.
// Lora campaign step 3a: stage the fire's lora state into
// `state.lora_staged` OUTSIDE any capture region and answer a
// fingerprint of what was staged (0 = no lora). The fingerprint covers
// everything a captured lora body bakes: the lane structure (count,
// ranks, widths, sites, token spans), the adapter device pointers, the
// grouping mode, and the post-staging arena base (a growth changes
// addresses and must recapture).
std::uint64_t llama_like_lora_stage(
    LlamaLikePlanState& state,
    Workspace& ws,
    const LoraTable* lora,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    int total_tokens,
    cudaStream_t stream);

RopeKind rope_kind_from_hf_config(const HfConfig& hf);

// Populate the RoPE-related fields on LlamaLikeForwardCfg from the
// HF config in one place — every arch that builds an LlamaLikeForwardCfg
// in context.cpp pulls in the same eight fields.
void apply_rope_config(LlamaLikeForwardCfg& fwd_cfg, const HfConfig& hf);

}  // namespace pie_cuda_driver::model
