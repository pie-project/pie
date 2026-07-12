#pragma once

#include "store/dsa_cache.hpp"
#include "distributed.hpp"
#include "store/mla_cache.hpp"
#include "model/imodel.hpp"
#include "model/glm5/glm5.hpp"
#include "model/glm5/glm5_forward.hpp"
#include "model/kimi/kimi_forward.hpp"  // KimiPlanState (shared MLA plan type)

namespace pie_cuda_driver::model {

// GLM-5.1 (DeepSeek Sparse Attention + MLA) IModel. Wraps
// `prepare_kimi_mla_plan` (shared MLA plan) + `glm5_forward_paged`.
// Holds references to both the compressed `MlaCache` and the DSA
// indexer `DsaCache`. The executor's standard kv_cache is a 1×1
// placeholder.
class Glm5Model final : public IModel {
public:
    Glm5Model(
        Glm5Weights weights,
        const HfConfig& hf_config,
        Glm5Workspace& ws,
        MlaCache& mla_cache,
        DsaCache& dsa_cache,
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
    Glm5Weights weights_;
    const HfConfig& hf_config_;
    Glm5Workspace& ws_;
    MlaCache& mla_cache_;
    DsaCache& dsa_cache_;
    KimiPlanState mla_plan_;
    Glm5ForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
