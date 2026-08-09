#pragma once

#include <algorithm>

#include "attention_workspace.hpp"
#include "model/declared/value_arena.hpp"
#include "model/imodel.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/mixtral/declared_forward.hpp"
#include "model/mixtral/mixtral.hpp"

namespace pie_cuda_driver::model {

// Mixtral reuses LlamaLikeForwardCfg for its identical attention half;
// the MoE block reads num_experts / top_k from HfConfig at construction.
// No prepare hook, no graph capture, no fused argmax.
class MixtralModel final : public IModel {
public:
    // The activation block this deployment's declaration needs, over
    // BOTH traced classes — gemma-4's rule, and for its reason: one
    // block serves whichever fires, and it is allocated once outside any
    // capture, so it must hold the wider.
    std::size_t declared_arena_bytes(int max_tokens,
                                     int max_sampled) const override {
        if (!declared_.usable) return 0;
        return std::max(
            declared::arena_bytes_for_widest(declared_.decode, max_tokens,
                                             max_sampled),
            declared::arena_bytes_for_widest(declared_.prefill, max_tokens,
                                             max_sampled));
    }

    MixtralModel(MixtralWeights weights,
                 const HfConfig& hf_config,
                 const LlamaLikeForwardCfg& fwd_cfg,
                 int num_experts,
                 int top_k);

    void prepare(AttentionWorkspace&, const ForwardFn::PrepareInputs&) override {}
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              kernels::gemm::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }

private:
    MixtralWeights weights_;
    const HfConfig& hf_config_;
    LlamaLikeForwardCfg fwd_cfg_;
    int num_experts_;
    int top_k_;
    GptOssDeclaredPlan declared_;
    ModelCapabilities caps_{};
};

}  // namespace pie_cuda_driver::model
