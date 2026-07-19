#pragma once

#include "distributed.hpp"
#include "model/imodel.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/nemotron_h/nemotron_h.hpp"
#include "model/nemotron_h/nemotron_h_forward.hpp"
#include "store/recurrent_state_cache.hpp"

namespace pie_cuda_driver::model {

// First per-model IModel impl. Façade over the existing Nemotron-H state in
// context.cpp — holds references to externally-allocated workspaces and the
// state cache, owns its own LlamaLikeForwardCfg + LlamaLikePlanState (these
// are per-arch and not shared anywhere else).
//
// Construction is deferred until after the memory planner has sized the
// workspaces, so allocation order in context.cpp doesn't change. Future PRs
// can flip this to OWN the workspaces once the planner is also abstracted
// into the model boundary.
class NemotronHModel final : public IModel {
public:
    NemotronHModel(
        NemotronHWeights weights,
        const HfConfig& hf_config,
        NemotronHWorkspace& ws,
        RecurrentStateCache& state_cache,
        KvCache& kv_cache,
        const LlamaLikeForwardCfg& base_fwd_cfg,
        int tp_size,
        NcclComm* tp_comm);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    RecurrentStateCache* state_cache() override { return &state_cache_; }
    std::uint32_t graph_layout() override;

    // Mamba2 layers need extra per-fire scratch (conv/SSM scan buffers) on
    // top of the universal `Workspace` formula, sized by the model's own
    // runtime tp_size.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        return IModel::workspace_bytes(cfg, max_tokens, output_rows) +
               nemotron_h_workspace_bytes(cfg, max_tokens, fwd_cfg_.tp_size);
    }

private:
    NemotronHWeights weights_;
    const HfConfig& hf_config_;
    NemotronHWorkspace& ws_;
    RecurrentStateCache& state_cache_;
    KvCache& kv_cache_;
    LlamaLikeForwardCfg fwd_cfg_;
    LlamaLikePlanState plan_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
