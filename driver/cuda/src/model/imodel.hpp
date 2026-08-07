#pragma once

// IModel — polymorphic interface for the per-arch forward.
//
// Every arch implements IModel and registers itself with the batch engine via
// `ForwardFn::attach_model(model.get())` at context.cpp setup time. The
// batch engine dispatches each fire's `prepare` / `body` / `graph_layout` /
// fused-argmax hooks through this interface, replacing what used to be
// per-arch lambda assignments scattered across context.cpp.
//
// Each concrete impl lives in `model/<arch>_model.{hpp,cpp}` next to its
// forward function and weight struct, so adding a new arch is a single
// directory-local change (new pair of files + one CMakeLists entry +
// one branch in the arch-detect chain in context.cpp).

#include <cstddef>
#include <cstdint>
#include <memory>

#include "batch/forward.hpp"
#include "ops/attention_workspace.hpp"
#include "store/kv_cache.hpp"

namespace pie_forward {
class ForwardPlan;
}

namespace pie_cuda_driver::batch {
class SupergraphBuilder;
}  // namespace pie_cuda_driver::batch

namespace pie_cuda_driver {

class LoadedModel;
class HfConfig;
class RecurrentStateCache;

namespace ops {
class CublasHandle;
}

namespace model {

struct Workspace;
struct LoraTable;

struct MediaEncodeInputs {
    const float* image_pixels_h = nullptr;
    const std::uint32_t* image_pixel_byte_indptr_h = nullptr;
    const std::uint32_t* image_patch_positions_h = nullptr;
    const std::uint32_t* image_anchor_rows_h = nullptr;
    int num_images = 0;
    const float* audio_features_h = nullptr;
    const std::uint32_t* audio_feature_byte_indptr_h = nullptr;
    const std::uint32_t* audio_anchor_rows_h = nullptr;
    int num_clips = 0;
    std::uint16_t* output_rows_h = nullptr;
    std::size_t output_bytes = 0;
    std::uint32_t* output_row_indptr_h = nullptr;
};

// Capability flags previously scattered as individual `forward_fn.supports_*`
// booleans. Bundled here so a model declares them in one place at construction
// time. The executor consults these to decide graph capture, compact-logits,
// and small-prefill-graph eligibility.
struct ModelCapabilities {
    bool graph_safe                   = false;
    bool graph_padding_kv_write_safe  = false;
    bool supports_compact_logits      = false;
    bool supports_small_prefill_graph = false;
    bool supports_runtime_window       = false;
    bool supports_media_encode         = false;
    // Stage 6 increment 4: this model's BODY with live stage hooks is
    // capture-legal for pure-decode fires — every hook-adjacent branch it
    // takes under capture (score capture, page-mask seeding) is stream work
    // against stable addresses, and its per-layer hook invocation order is
    // deterministic. Only llama_like asserts this; the batch engine
    // additionally requires wants_page_mask == false (host-side control flow
    // on the mask's written_layer cannot be captured), no lora, and a single
    // rank before a hook fire may replay a graph.
    bool supports_hook_graph_capture   = false;
    // The unionized supergraph (S3): this model can spell its decode body
    // as a SupergraphBuilder program (the emitted `..._supergraph_build`
    // exists for the LIVE deployment's digest). The batch engine gates the
    // capture on this plus the fire-side eligibility (pure decode, no
    // hooks/lora/score, window == -2).
    bool supports_supergraph           = false;
    // Whether `body()` honours `ForwardInputs::logits_argmax_chunk_tokens` and
    // writes `ws.sampled_tokens` instead of `ws.logits` when it is set
    // (§20.37). Most families ignore the field, so the default must be false:
    // the driver commits the epilogue's token source before the forward runs,
    // and a family that quietly materialised logits anyway would leave the
    // epilogue publishing uninitialised memory as token ids.
    bool supports_fused_lm_head_argmax = false;
    // Whether the boot-time graph lattice (synthetic geometry: one shared
    // page, kv_len=1) may pre-capture this deployment's decode buckets.
    // On plan-free force_prefill deployments (GQA ratio keeps the decode
    // kernel out; decode rides BatchPrefill) the capture-time attention
    // launch is configured for the synthetic shape and REPLAYS ~700x slow
    // against real geometry (7.2 ms/layer measured on Qwen2.5-14B/L40S —
    // the "bimodal collapse", dev_merge_playbook.md). First-use capture
    // with real metadata is correct — the nemotron_h precedent, declared
    // as a capability instead of a name check.
    bool upfront_capture_safe          = true;
};

// Polymorphic per-model interface. Implementations hold refs to per-arch
// weights, workspaces (NemotronHWorkspace, Qwen3_5MoeMlpWorkspace, etc.),
// plan state, and state-cache (when applicable). The executor invokes
// `prepare` / `body` / `graph_layout` once per fire through this vtable.
class IModel {
public:
    virtual ~IModel() = default;

    // Per-step host-side plan setup. Mirrors current ForwardFn::prepare.
    virtual void prepare(AttentionWorkspace& attn_ws,
                         const ForwardFn::PrepareInputs& in) = 0;

    // Per-step device-side forward body. Mirrors current ForwardFn::body.
    virtual void body(Workspace& ws,
                      KvCache& kv,
                      AttentionWorkspace& attn_ws,
                      ops::CublasHandle& cublas,
                      const ForwardFn::ForwardInputs& in) = 0;

    // Optional: per-arch scratch-buffer byte budget for the persistent
    // forward-workspace arena, consulted by the memory planner while it
    // sweeps candidate (max_tokens, output_rows) shapes ahead of model
    // construction. Defaults to the universal `Workspace` formula
    // (`model::workspace_bytes`); a family whose forward diverges from the
    // universal shape can override this without touching `body()`'s
    // signature. NOTE: the planner currently sizes the arena directly from
    // `HfConfig` before any `IModel` exists, so this hook is not yet wired
    // into that call site — it exists so per-arch divergence has a home
    // that doesn't require another interface change later.
    virtual std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                        int output_rows) const {
        return ::pie_cuda_driver::model::workspace_bytes(
            cfg, max_tokens, output_rows, cfg.intermediate_size,
            cfg.num_attention_heads * cfg.head_dim,
            cfg.num_key_value_heads * cfg.head_dim);
    }

    // Static-at-construction capability flags.
    virtual ModelCapabilities capabilities() const = 0;

    // Optional: the traced + structurally validated declared-forward plan
    // this model built at construction (PIE_DECLARED_FORWARD opted in AND
    // the configuration was representable AND the validation passed).
    // nullptr otherwise — including for every family without a declared
    // trace. Read once at load, when the capability payload derives the
    // plan's model-structural site summary (`model_site_summary` in
    // context.cpp): the driver is the party holding a VALIDATED plan, so
    // the summary the engine's fire planner consumes is stated here rather
    // than re-derived runtime-side from binding facts the engine lacks.
    virtual const pie_forward::ForwardPlan* declared_plan() const {
        return nullptr;
    }

    // Optional: per-model recurrent state cache (Mamba2 / linear-attn / MTP
    // hidden snapshot). nullptr = model has no recurrent state.
    virtual RecurrentStateCache* state_cache() { return nullptr; }

    // Optional: graph layout key for CUDA-graph cache (forward_fn.graph_layout
    // equivalent). 0 = a single graph variant suffices.
    virtual std::uint32_t graph_layout() { return 0; }

    // The union key's layout (S3): spans every plan the supergraph's arms
    // dispatch against. Defaults to the plain layout for models without a
    // supergraph build.
    virtual std::uint32_t supergraph_graph_layout() { return graph_layout(); }

    // Optional: whether the fire just planned by `prepare` carries a PREFILL
    // whose dispatch has content-independent launch geometry, i.e. one the
    // batch engine may capture and replay.
    //
    // `forward_graph_replay_eligible` gates on `is_pure_decode`, so a wave with
    // a single arriving request loses replay for all of its decode lanes too --
    // measured at 7.3 ms of host enqueue against 10 us for the same width when
    // pure (see the campaign notes). The planner already computes the answer
    // per fire (`PrefillPlanCache::graph_capturable`); this is the vtable seam
    // that lets the gate read it. Default false keeps every arch that has not
    // opted in on exactly the old path.
    virtual bool prefill_graph_capturable() const { return false; }

    virtual bool encode_media(const MediaEncodeInputs&, cudaStream_t) { return false; }

    // Lora campaign step 3a: stage this fire's lora state OUTSIDE any
    // capture region (cast uploads, slab build — host+stream work the
    // captured body must not contain). Returns a fingerprint of what
    // was staged (0 = no lora / unsupported); the engine keys lora
    // graph replay on it. A null table clears the staged state.
    virtual std::uint64_t lora_stage(Workspace&,
                                     const LoraTable*,
                                     int /*total_tokens*/,
                                     cudaStream_t /*stream*/) {
        return 0;
    }

    // The unionized supergraph's capture body (S3): spell this fire's
    // decode as conditional-armed graph work on the builder. Returns false
    // when the deployment has no emitted build (the caller falls back to
    // the plain capture). Only called under an active stream capture.
    virtual bool supergraph_body(Workspace&,
                                 KvCache&,
                                 AttentionWorkspace&,
                                 ops::CublasHandle&,
                                 const ForwardFn::ForwardInputs&,
                                 batch::SupergraphBuilder&) {
        return false;
    }
};

}  // namespace model
}  // namespace pie_cuda_driver
