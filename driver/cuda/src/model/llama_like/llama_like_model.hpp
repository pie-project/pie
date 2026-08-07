#pragma once

#include "model/imodel.hpp"
#include "model/llama_like/declared_forward.hpp"
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

    bool supergraph_body(Workspace& ws,
                         KvCache& kv,
                         AttentionWorkspace& attn_ws,
                         ops::CublasHandle& cublas,
                         const ForwardFn::ForwardInputs& in,
                         batch::SupergraphBuilder& sg) override;
    std::uint32_t graph_layout() override;
    std::uint32_t supergraph_graph_layout() override;
    std::uint64_t lora_stage(Workspace& ws,
                             const LoraTable* lora,
                             int total_tokens,
                             cudaStream_t stream) override;

    // The validated declared plan (empty → nullptr), for the load-time
    // capability site summary (imodel.hpp).
    const pie_forward::ForwardPlan* declared_plan() const override {
        return declared_ ? &declared_.plan : nullptr;
    }

    bool prefill_graph_capturable() const override;

private:
    Qwen3Weights weights_;
    const HfConfig& hf_config_;
    KvCache& kv_cache_;
    LlamaLikeForwardCfg fwd_cfg_;
    LlamaLikePlanState plan_;
    ModelCapabilities caps_;
    // Stage 3 (pie-application-plan.md §7): the traced form of this
    // deployment's forward, when PIE_DECLARED_FORWARD opted in and the
    // config is representable. Empty otherwise; `body` then never consults
    // it. Traced at construction because the facts are load-time facts.
    LlamaLikeDeclaredPlan declared_;
};

}  // namespace pie_cuda_driver::model
