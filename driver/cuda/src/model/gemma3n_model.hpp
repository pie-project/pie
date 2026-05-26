#pragma once

#include "model/gemma3n.hpp"
#include "model/imodel.hpp"

namespace pie_cuda_driver::model {

// Loader-only milestone. bind_gemma3n loads every tensor; the forward
// (AltUp predict/correct + Laurel + activation sparsity + PLE input gate)
// is a follow-up — the stub throws with a clear message at the first
// fire_batch.
class Gemma3nModel final : public IModel {
public:
    Gemma3nModel(const Gemma3nWeights& weights,
                 const HfConfig& hf_config,
                 const Gemma3nForwardCfg& fwd_cfg);

    void prepare(AttentionWorkspace&, const ForwardFn::PrepareInputs&) override {}
    void body(Qwen3Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return {}; }

private:
    const Gemma3nWeights& weights_;
    const HfConfig& hf_config_;
    Gemma3nForwardCfg fwd_cfg_;
};

}  // namespace pie_cuda_driver::model
