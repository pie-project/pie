#pragma once

#include "model/gemma2.hpp"
#include "model/imodel.hpp"

namespace pie_cuda_driver::model {

// Covers Gemma 2 and Gemma 3 — both bind into Gemma2Weights and route
// through gemma2_forward_paged. No prepare hook, no graph capture, no
// fused argmax: capabilities stay at their defaults (all false).
class Gemma2Model final : public IModel {
public:
    Gemma2Model(const Gemma2Weights& weights,
                const HfConfig& hf_config,
                const Gemma2ForwardCfg& fwd_cfg);

    void prepare(AttentionWorkspace&, const ForwardFn::PrepareInputs&) override {}
    void body(Qwen3Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return {}; }

private:
    const Gemma2Weights& weights_;
    const HfConfig& hf_config_;
    Gemma2ForwardCfg fwd_cfg_;
};

}  // namespace pie_cuda_driver::model
