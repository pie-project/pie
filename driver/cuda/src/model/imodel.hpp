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

// Capability flags previously scattered as individual `forward_fn.supports_*`
// booleans. Bundled here so a model declares them in one place at construction
// time. The executor consults these to decide graph capture, fused-argmax,
// compact-logits, and small-prefill-graph eligibility.
struct ModelCapabilities {
    bool graph_safe                   = false;
    bool supports_tp_greedy_argmax    = false;
    bool supports_compact_logits      = false;
    bool supports_small_prefill_graph = false;
    bool supports_fused_lmhead_argmax = false;
};

// Polymorphic per-model interface. Implementations hold refs to per-arch
// weights, workspaces (NemotronHWorkspace, Qwen3_5MoeMlpWorkspace, etc.),
// plan state, and state-cache (when applicable). The executor invokes
// `prepare` / `body` / `graph_layout` once per fire through this vtable;
// the optional fused-argmax hooks default to no-op so most archs leave
// them alone.
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

    // Optional fused-argmax hooks. Models that opt in must implement all
    // three; the default no-op set treats fused argmax as unsupported.
    virtual void set_logits_argmax_only(bool /*enabled*/) {}
    virtual void set_fused_argmax_output(std::int32_t* /*ptr*/) {}
    virtual bool fused_argmax_done() { return false; }
};

}  // namespace model
}  // namespace pie_cuda_driver
