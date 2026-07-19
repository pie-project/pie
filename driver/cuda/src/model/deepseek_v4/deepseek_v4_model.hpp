#pragma once

#include "distributed.hpp"
#include "model/imodel.hpp"
#include "model/deepseek_v4/deepseek_v4.hpp"
#include "model/deepseek_v4/deepseek_v4_forward.hpp"

namespace pie_cuda_driver::model {

// DeepSeek-V4 IModel. Thin façade over `dsv4_forward_paged`; owns its
// own DsV4ForwardCfg + DsV4PlanState. Uses the standard paged KvCache
// (no MLA-compressed cache), so it slots into the executor's per-fire
// flow like the other dense/MoE archs. DSV4 has no host-side prepare
// step, so `prepare()` is a no-op.
class DsV4Model final : public IModel {
public:
    DsV4Model(
        DsV4Weights weights,
        const HfConfig& hf_config,
        DsV4Workspace& ws,
        int tp_size,
        int tp_rank,
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
    DsV4Weights weights_;
    const HfConfig& hf_config_;
    DsV4Workspace& ws_;
    DsV4ForwardCfg fwd_cfg_;
    DsV4PlanState plan_state_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
