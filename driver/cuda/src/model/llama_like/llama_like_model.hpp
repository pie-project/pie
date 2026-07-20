#pragma once

#include "model/imodel.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/llama_like/qwen3.hpp"

namespace pie_cuda_driver::model {

// Llama-like IModel — handles every arch that falls through to
// `llama_like_forward_paged`: Qwen3, Mixtral, Mistral3, GPT-OSS, Gemma2,
// and any other shape that shares the standard transformer pipeline.
// All those arches bind their weights into a `Qwen3Weights` (which the
// driver overloads as the "llama-like weights" type) and use the same
// LlamaLikeForwardCfg + LlamaLikePlanState, so a single class covers them.
//
// (Gemma4/Gemma3n/Nemotron-H/Qwen3.5-{,MoE} have their own forwards and
// their own IModel classes — they're not routed here.)
class LlamaLikeModel final : public IModel {
public:
    LlamaLikeModel(
        Qwen3Weights weights,
        const HfConfig& hf_config,
        KvCache& kv_cache,
        const LlamaLikeForwardCfg& fwd_cfg);

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
};

}  // namespace pie_cuda_driver::model
