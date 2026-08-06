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

namespace pie_cuda_driver {

class LoadedModel;
class HfConfig;
class RecurrentStateCache;

namespace ops {
class CublasHandle;
}

namespace model {

struct Workspace;

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
    // Whether `body()` honours `ForwardInputs::logits_argmax_chunk_tokens` and
    // writes `ws.sampled_tokens` instead of `ws.logits` when it is set
    // (§20.37). Most families ignore the field, so the default must be false:
    // the driver commits the epilogue's token source before the forward runs,
    // and a family that quietly materialised logits anyway would leave the
    // epilogue publishing uninitialised memory as token ids.
    bool supports_fused_lm_head_argmax = false;
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

    // Optional: per-model recurrent state cache (Mamba2 / linear-attn / MTP
    // hidden snapshot). nullptr = model has no recurrent state.
    virtual RecurrentStateCache* state_cache() { return nullptr; }

    // Optional: graph layout key for CUDA-graph cache (forward_fn.graph_layout
    // equivalent). 0 = a single graph variant suffices.
    virtual std::uint32_t graph_layout() { return 0; }

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
};

}  // namespace model
}  // namespace pie_cuda_driver
