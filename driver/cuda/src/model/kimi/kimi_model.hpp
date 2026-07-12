#pragma once

#include "distributed.hpp"
#include "store/mla_cache.hpp"
#include "model/imodel.hpp"
#include "model/kimi/kimi.hpp"
#include "model/kimi/kimi_forward.hpp"

namespace pie_cuda_driver::model {

// Kimi K2 (DeepSeek-V3-style MLA) IModel. Wraps `prepare_kimi_mla_plan`
// + `kimi_forward_paged`. The real KV lives in the compressed
// `MlaCache` (held by reference); the executor's standard kv_cache is a
// 1×1 placeholder. Owns its KimiPlanState (rebuilt each `prepare`).
class KimiModel final : public IModel {
public:
    KimiModel(
        KimiWeights weights,
        const HfConfig& hf_config,
        KimiWorkspace& ws,
        MlaCache& mla_cache,
        int tp_size,
        NcclComm* tp_comm,
        bool emit_logits,
        bool supports_tp_greedy_argmax);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }

private:
    KimiWeights weights_;
    const HfConfig& hf_config_;
    KimiWorkspace& ws_;
    MlaCache& mla_cache_;
    KimiPlanState plan_state_;
    KimiForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
