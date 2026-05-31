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
        int small_spec_graph_tokens,
        // Vision tower for multimodal gemma-4 (nullptr = text-only).
        const Gemma4VisionWeights* vision = nullptr,
        // Audio tower for multimodal gemma-4 (nullptr = no audio).
        const Gemma4AudioWeights* audio = nullptr);

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
    // Vision encoder weights (raw bf16 pointers), built once from the bound
    // tower. `has_vision_` gates the encode+scatter in `body()`.
    VisRawWeights vision_raw_;
    bool has_vision_ = false;
    // Audio encoder weights (raw bf16 pointers), built once from the bound
    // tower. `has_audio_` gates the encode+scatter in `body()`.
    AudioRawWeights audio_raw_;
    bool has_audio_ = false;
};

}  // namespace pie_cuda_driver::model
