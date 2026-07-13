#pragma once

#include <optional>

#include "model/gemma4/gemma4.hpp"
#include "model/imodel.hpp"

namespace pie_cuda_driver::model {

// Gemma-4 IModel. Owns its bound `Gemma4Weights` plus optional vision/audio
// tower weights (moved in from the registry's `Gemma4Plan`); holds
// references to Context-allocated workspaces (`moe_ws`, `kv_cache`) and its
// own per-arch `Gemma4ForwardCfg`. Forwards prepare/body/graph_layout +
// fused-argmax hooks.
class Gemma4Model final : public IModel {
public:
    Gemma4Model(
        Gemma4Weights weights,
        const HfConfig& hf_config,
        Gemma4MoeMlpWorkspace& moe_ws,
        KvCache& kv_cache,
        const Gemma4ForwardCfg& fwd_cfg,
        // Token threshold for the small-prefill graph fast path. Set to 0
        // when the runtime's spec-graph window is disabled.
        int small_spec_graph_tokens,
        // Vision tower for multimodal gemma-4 (nullopt = text-only). Only
        // needed transiently: the constructor converts it to `vision_raw_`
        // and does not retain the original struct.
        std::optional<Gemma4VisionWeights> vision = std::nullopt,
        // Audio tower for multimodal gemma-4 (nullopt = no audio). Same
        // transient-conversion note as `vision`.
        std::optional<Gemma4AudioWeights> audio = std::nullopt);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    std::uint32_t graph_layout() override;

    // Gemma-4's MoE variant needs extra per-fire scratch on top of the
    // universal `Workspace` formula (router/expert buffers); dense
    // checkpoints (`!gemma4_enable_moe`) fall back to the base formula.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        std::size_t bytes = IModel::workspace_bytes(cfg, max_tokens, output_rows);
        if (cfg.gemma4_enable_moe) {
            bytes += gemma4_moe_workspace_bytes(cfg, max_tokens);
        }
        return bytes;
    }

    // Fused argmax — module-level free functions in the gemma4 namespace
    // today, exposed through the IModel interface so the executor's
    // capability dispatch sees a uniform API.
    void set_logits_argmax_only(bool enabled) override;
    void set_fused_argmax_output(std::int32_t* ptr) override;
    bool fused_argmax_done() override;
    bool encode_media(const MediaEncodeInputs& in, cudaStream_t stream) override;

private:
    Gemma4Weights weights_;
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
