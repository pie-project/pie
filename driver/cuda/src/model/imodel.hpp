#pragma once

// IModel — polymorphic interface for the per-arch forward.
//
// Every arch implements IModel and registers itself with the executor via
// `ForwardFn::attach_model(model.get())` at entry.cpp setup time. The
// executor dispatches each fire's `prepare` / `body` / `graph_layout` /
// fused-argmax hooks through this interface, replacing what used to be
// per-arch lambda assignments scattered across entry.cpp.
//
// Each concrete impl lives in `model/<arch>_model.{hpp,cpp}` next to its
// forward function and weight struct, so adding a new arch is a single
// directory-local change (new pair of files + one CMakeLists entry +
// one branch in the arch-detect chain in entry.cpp).

#include <cstdint>
#include <memory>

#include "attention_workspace.hpp"
#include "executor/executor.hpp"
#include "kv_cache.hpp"

namespace pie_cuda_driver {

class LoadedModel;
class HfConfig;
class RecurrentStateCache;

namespace ops {
class CublasHandle;
}

namespace model {

struct Qwen3Workspace;

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
    virtual void body(Qwen3Workspace& ws,
                      KvCache& kv,
                      AttentionWorkspace& attn_ws,
                      ops::CublasHandle& cublas,
                      const ForwardFn::ForwardInputs& in) = 0;

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
