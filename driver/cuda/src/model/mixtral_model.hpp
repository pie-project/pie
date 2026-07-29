#pragma once

#include "model/imodel.hpp"
#include "model/llama_like.hpp"
#include "model/mixtral.hpp"

namespace pie_cuda_driver::model {

// Mixtral reuses LlamaLikeForwardCfg for its identical attention half;
// the MoE block reads num_experts / top_k from HfConfig at construction.
// No prepare hook, no graph capture, no fused argmax.
class MixtralModel final : public IModel {
public:
    MixtralModel(const MixtralWeights& weights,
                 const HfConfig& hf_config,
                 const LlamaLikeForwardCfg& fwd_cfg,
                 int num_experts,
                 int top_k);

    void prepare(AttentionWorkspace&, const ForwardFn::PrepareInputs&) override {}
    void body(Qwen3Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return {}; }

private:
    const MixtralWeights& weights_;
    const HfConfig& hf_config_;
    LlamaLikeForwardCfg fwd_cfg_;
    int num_experts_;
    int top_k_;
};

}  // namespace pie_cuda_driver::model
