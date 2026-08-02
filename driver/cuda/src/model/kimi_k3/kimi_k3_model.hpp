#pragma once

// Kimi-K3 IModel. The one family that holds *both* a paged compressed-latent
// MLA cache and a recurrent-state cache: three of every four layers keep a
// KDA state slab plus three conv windows, the fourth pages KV.

#include "distributed.hpp"
#include "model/imodel.hpp"
#include "model/kimi/kimi_forward.hpp"  // KimiPlanState (shared MLA plan type)
#include "model/kimi_k3/kimi_k3.hpp"
#include "model/kimi_k3/kimi_k3_forward.hpp"
#include "store/mla_cache.hpp"
#include "store/recurrent_state_cache.hpp"

namespace pie_cuda_driver::model {

class KimiK3Model final : public IModel {
public:
    KimiK3Model(
        KimiK3Weights weights,
        const HfConfig& hf_config,
        KimiK3Workspace& ws,
        MlaCache& mla_cache,
        RecurrentStateCache& kda_cache,
        int tp_size,
        NcclComm* tp_comm,
        bool emit_logits);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }

private:
    KimiK3Weights weights_;
    const HfConfig& hf_config_;
    KimiK3Workspace& ws_;
    MlaCache& mla_cache_;
    RecurrentStateCache& kda_cache_;
    KimiPlanState mla_plan_;
    KimiK3ForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
