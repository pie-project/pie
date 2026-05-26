#pragma once

#include "model/gemma4.hpp"
#include "model/imodel.hpp"

namespace pie_cuda_driver::model {

// Gemma-4 IModel. Same façade pattern as NemotronHModel — holds references
// to externally-allocated workspaces (weights_gemma4, gemma4_moe_ws) plus
// the per-arch Gemma4ForwardCfg. Forwards prepare/body/graph_layout +
// fused-argmax hooks.
class Gemma4Model final : public IModel {
public:
    Gemma4Model(
        const Gemma4Weights& weights,
        const HfConfig& hf_config,
        Gemma4MoeMlpWorkspace& moe_ws,
        KvCache& kv_cache,
        const Gemma4ForwardCfg& fwd_cfg,
        // Token threshold for the small-prefill graph fast path. Set to 0
        // when the runtime's spec-graph window is disabled.
        int small_spec_graph_tokens);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Qwen3Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    std::uint32_t graph_layout() override;

    // Fused argmax — module-level free functions in the gemma4 namespace
    // today, exposed through the IModel interface so the executor's
    // capability dispatch sees a uniform API.
    void set_logits_argmax_only(bool enabled) override;
    void set_fused_argmax_output(std::int32_t* ptr) override;
    bool fused_argmax_done() override;

private:
    const Gemma4Weights& weights_;
    const HfConfig& hf_config_;
    Gemma4MoeMlpWorkspace& moe_ws_;
    KvCache& kv_cache_;
    Gemma4ForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
