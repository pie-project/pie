#pragma once

// THE arch registry — the single table mapping a checkpoint's HfConfig
// `model_type` string to { family, config-validate hook, weight binder /
// plan factory, IModel factory, pre-construction capabilities }. This is
// the only place in the driver that knows the full set of supported
// `model_type` strings; adding a model means adding one row here plus its
// own family directory. See cpp-refact.md Phase 5.
//
// Design (cpp-refact.md Phase 5 "kill the union"):
//   * `ModelPlan` is a polymorphic, move-only result of binding one
//     checkpoint's weights for one arch family. Exactly one concrete
//     derived plan per family (defined in registry.cpp, next to the arch
//     table row that constructs it); each owns exactly its own concrete
//     weights struct(s) by value. No `std::any`/`void*`/variant-of-every-
//     family — `Context` never downcasts a `ModelPlan*`.
//   * `PlanInfo` is the narrow, typed, pre-construction planning surface
//     `Context` reads (per-layer shape overrides, recurrent/linear-attn
//     layer maps, lm-head sharding, MTP presence, multimodal limits).
//     Every field defaults to "not applicable"; a family only populates
//     the subset it needs.
//   * `ModelResources` is the flip side: Context-owned workspaces/stores/
//     comms/config a family's `create_model` factory may read. Unused
//     pointers stay null. Context allocates these (switching on `Family`
//     only where the underlying store physically differs — e.g. MLA vs
//     paged KV cache), then makes exactly one `create_model` call.
//   * The one place that downcasts `ModelPlan*` to a concrete family type
//     is that family's own `create_model` factory in registry.cpp — it is
//     the same code that built the plan via `bind`, so the cast is always
//     correct by construction (checked with `dynamic_cast` + a hard error
//     if the registry table is ever mis-wired).

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model/imodel.hpp"

namespace pie_cuda_driver {
struct HfConfig;
class LoadedModel;
class MlaCache;
struct DsaCache;
class NcclComm;
class RecurrentStateCache;
}  // namespace pie_cuda_driver

namespace pie_cuda_driver::model {

struct LlamaLikeForwardCfg;
struct Gemma2ForwardCfg;
struct Gemma4ForwardCfg;
struct Gemma4MoeMlpWorkspace;
struct Qwen3_5LinearAttnWorkspace;
struct Qwen3_5MoeMlpWorkspace;
struct Qwen3_5PlanState;
struct NemotronHWorkspace;
struct DsV4Workspace;
struct KimiWorkspace;
struct Glm5Workspace;

// Every architecture family the registry can construct. Distinct from the
// `model_type` string: several strings alias onto the same family (e.g.
// "mistral3"/"ministral3"/"phi3"/"olmo2"/"olmo3" all bind into
// `LlamaLike`, each through its own binder). Context may switch on this
// enum ONLY to allocate family-specific stores/workspaces that physically
// differ (KV-cache shape, MLA/DSA caches, recurrent-state caches); the
// registry table alone decides which family a `model_type` maps to.
enum class Family {
    LlamaLike,
    Gemma,
    Gemma4,
    Gemma3n,
    Mixtral,
    Qwen3_5,
    Qwen3_5Moe,
    NemotronH,
    Kimi,
    DeepSeekV4,
    Glm5,
    Qwen3VL,
    Csm,
};

const char* family_name(Family family) noexcept;

// Narrow, typed pre-construction planning info a bound `ModelPlan`
// reports. Context reads exactly these fields to size caches/workspaces
// ahead of model construction instead of downcasting a plan/weights
// pointer. Every family leaves the fields it doesn't use at their
// defaults (empty vector / false / 0).
struct PlanInfo {
    Family family = Family::LlamaLike;
    std::size_t num_layers = 0;

    // Per-layer shape overrides (Gemma-4 heterogeneous head_dim/kv-sharing;
    // Gemma-3n per-layer MLP width). Empty => uniform, use HfConfig dims.
    std::vector<int> per_layer_intermediate;
    std::vector<int> per_layer_head_dim;
    std::vector<int> per_layer_num_kv_heads;
    std::vector<int> kv_source_layer;

    // Recurrent/linear-attention layer maps. `layer_is_linear_attn` is
    // populated for Qwen3.5 [+MoE] (Kind::LinearAttn vs FullAttn);
    // `layer_is_mamba` for Nemotron-H (Kind::Mamba vs Attention/MoE).
    std::vector<bool> layer_is_linear_attn;
    std::vector<bool> layer_is_mamba;

    // True when the bound lm_head carries a non-null TP row-shard view
    // (Qwen3Weights::lm_head_tp_shard / KimiWeights::lm_head_tp_sharded).
    // Gates `supports_tp_greedy_argmax` for LlamaLike/Qwen3VL/Kimi.
    bool lm_head_tp_sharded = false;

    // True when the bound weights carry an MTP head (Qwen3.5 [+MoE]).
    bool has_mtp = false;

    // Multimodal adapters (Gemma-4 vision/audio, Qwen3-VL vision).
    bool has_vision = false;
    bool has_audio = false;
    int gemma4_pool_kernel = 0;
    int gemma4_position_table = 0;
    int qwen3_vl_patch_dim = 0;
    int qwen3_vl_merge_unit = 0;
    int audio_mel_bins = 0;
};

// Polymorphic result of binding one checkpoint's weights for one arch
// family. Exactly one concrete derived plan per family (registry.cpp);
// each owns exactly its own concrete weights struct(s) by value (plus any
// same-family adapters — e.g. Gemma-4's optional vision/audio towers).
// Move-only: binding is a one-shot, one-checkpoint operation.
class ModelPlan {
public:
    virtual ~ModelPlan() = default;

    // Pre-construction planning surface. Stable for the plan's lifetime
    // (until `create_model` moves the concrete weights out of it).
    virtual const PlanInfo& plan_info() const = 0;

    // Optional per-family scratch-buffer byte budget, mirroring
    // `IModel::workspace_bytes()` but available before any `IModel`
    // exists (bind happens ahead of the memory planner). Defaults to the
    // universal `model::workspace_bytes` formula; a family whose forward
    // needs extra per-fire scratch (Qwen3.5 linear-attn, Nemotron-H
    // Mamba2, Gemma-4 MoE) overrides it with the same formula the memory
    // planner already applies via its own family switch
    // (`store/memory_planner.cpp`). NOTE: the planner's candidate-shape
    // sweep runs its own copy of that formula today (computed before this
    // hook existed); wiring the sweep itself to call through here is a
    // follow-up, not blocked on anything architectural.
    virtual std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                        int output_rows) const;
};

// Context-owned resources a family's `create_model` factory may read.
// Not every family uses every field; unused pointers/values stay at their
// defaults. Context fills this in once, after the memory planner and all
// per-family stores/workspaces are allocated, immediately before the
// single `create_model` call.
struct ModelResources {
    const HfConfig* hf_config = nullptr;
    KvCache* kv_cache = nullptr;
    MlaCache* mla_cache = nullptr;
    DsaCache* dsa_cache = nullptr;

    int tp_size = 1;
    int tp_rank = 0;
    NcclComm* tp_comm = nullptr;
    bool verbose = false;

    // Forward configs Context pre-populates from HfConfig + runtime env
    // (same values every family's construction reads today).
    LlamaLikeForwardCfg* llama_fwd_cfg = nullptr;
    Gemma2ForwardCfg* gemma_fwd_cfg = nullptr;
    Gemma4ForwardCfg* gemma4_fwd_cfg = nullptr;

    int max_workspace_tokens = 0;
    // Shared small-spec-graph token threshold (env-derived; Gemma-4 and
    // Qwen3.5 both consult it at construction time).
    int small_spec_graph_tokens = 0;

    // Gemma-4 MoE per-fire scratch (allocated regardless of family; inert
    // when unused).
    Gemma4MoeMlpWorkspace* gemma4_moe_ws = nullptr;

    // Qwen3.5 [+MoE] hybrid linear/full-attention scratch.
    Qwen3_5LinearAttnWorkspace* qwen3_5_la_ws = nullptr;
    Qwen3_5MoeMlpWorkspace* qwen3_5_moe_ws = nullptr;
    Qwen3_5PlanState* qwen3_5_plan_state = nullptr;
    RecurrentStateCache* qwen3_5_state_cache = nullptr;
    // MTP system-drafter wiring (Qwen3.5 [+MoE] only; null elsewhere).
    NativeSystemDrafter* system_drafter = nullptr;
    int native_mtp_num_drafts = 0;

    // Nemotron-H Mamba2 scratch + recurrent state.
    NemotronHWorkspace* nemotron_h_ws = nullptr;
    RecurrentStateCache* nemotron_h_state_cache = nullptr;

    // DeepSeek-V4 / Kimi / GLM-5 per-fire scratch.
    DsV4Workspace* dsv4_ws = nullptr;
    KimiWorkspace* kimi_ws = nullptr;
    Glm5Workspace* glm5_ws = nullptr;
};

// One row in the arch table. Every supported `model_type` string is its
// own row; kind-sharing archs (gpt_oss, mistral3/ministral3/phi3/olmo2/
// olmo3, deepseek_v2/deepseek_v3/kimi_k2, qwen3_5_moe/qwen3_5_moe_text/
// qwen3_moe, ...) are explicit aliases pointing at the same family/plan
// type through their own binder — never fallthrough logic.
struct ArchEntry {
    std::string model_type;
    Family family;

    // Human-readable id of which concrete binder this row uses (e.g.
    // "llama_like" vs "mistral3" vs "phi3" vs "olmo3" — all `LlamaLike`
    // family, four distinct binders). Diagnostics + the registry test's
    // proof that aliases route to their own binder, not a shared
    // fallback.
    std::string binder_key;

    // Optional post-parse validation of family-required HfConfig fields
    // (e.g. CSM requires `hf.csm.has_value()`). Returns an error message
    // on failure; most rows use `default_validate_config`, which only
    // checks the dimensions every family needs (the parser already
    // enforces per-field requirements — this is a defense-in-depth check,
    // not a parser rewrite).
    std::function<std::optional<std::string>(const HfConfig&)> validate_config;

    // Bind checkpoint weights (+ same-family adapters) into a concrete
    // `ModelPlan`. `LoadedModel&` is mutable because the Qwen3.5 and CSM
    // binders mutate engine-owned scratch (materializing bf16/fp32
    // copies) — see `model/qwen3_5/qwen3_5.cpp`, `model/csm/csm.cpp`.
    std::function<std::unique_ptr<ModelPlan>(LoadedModel&, bool verbose)> bind;

    // Construct the `IModel` from the bound plan + Context resources.
    // Consumes (moves out of) the plan.
    std::function<std::unique_ptr<IModel>(std::unique_ptr<ModelPlan>,
                                          ModelResources&)>
        create_model;
};

// The single arch table, one row per supported `model_type` string.
const std::vector<ArchEntry>& arch_table();

// Look up by exact `HfConfig.model_type`. Returns nullptr for an unknown
// arch — callers must treat that as a load/status error; there is no
// silent fallback to any family.
const ArchEntry* find_arch_entry(const std::string& model_type);

}  // namespace pie_cuda_driver::model
