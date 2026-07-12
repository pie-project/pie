#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "device_buffer.hpp"
#include "model/imodel.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/llama_like/qwen3.hpp"
#include "model/qwen3_vl/qwen3_vl.hpp"
#include "model/qwen3_vl/qwen3_vl_vision_forward.hpp"  // QwenVisRawWeights

namespace pie_cuda_driver::model {

// Qwen3-VL IModel. The text tower is a standard Qwen3 decoder (28L, GQA,
// per-head q/k norm) but with three multimodal additions over llama_like:
//   * vision encode + scatter after the embed (overwrites image-token rows);
//   * interleaved M-RoPE in attention (3-axis position ids per token);
//   * DeepStack injection — the 3 deepstack merger outputs are added into
//     the hidden state on image rows after decoder layers 0/1/2.
// Text-only fires (no images) reduce to a plain Qwen3 forward.
class Qwen3VLModel final : public IModel {
public:
    Qwen3VLModel(
        Qwen3Weights text_weights,
        const HfConfig& hf_config,
        KvCache& kv_cache,
        const LlamaLikeForwardCfg& fwd_cfg,
        int max_workspace_tokens,
        // Vision tower (nullopt = text-only checkpoint, should not happen
        // for qwen3_vl but kept for symmetry with gemma4). Only needed
        // transiently: the constructor converts it to `vision_raw_` and
        // does not retain the original struct.
        std::optional<Qwen3VLVisionWeights> vision = std::nullopt);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    std::uint32_t graph_layout() override;

private:
    Qwen3Weights weights_;
    const HfConfig& hf_config_;
    KvCache& kv_cache_;
    LlamaLikeForwardCfg fwd_cfg_;
    LlamaLikePlanState plan_;
    ModelCapabilities caps_;

    // Vision encoder (raw bf16 pointers built once from the bound tower).
    QwenVisRawWeights vision_raw_;
    bool has_vision_ = false;
    int num_deepstack_ = 0;

    // Owned per-fire scratch (allocated lazily on the first image fire).
    //   deepstack_scratch_: [num_deep, max_tokens, out_hidden] bf16
    //   mrope_positions_d_: [max_tokens, 3] int32
    DeviceBuffer<std::uint16_t> deepstack_scratch_;
    DeviceBuffer<std::int32_t>  mrope_positions_d_;
    int max_tokens_ = 0;
};

}  // namespace pie_cuda_driver::model
