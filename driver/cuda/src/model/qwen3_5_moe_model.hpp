#pragma once

#include "distributed.hpp"
#include "model/imodel.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_forward.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_5_moe_forward.hpp"
#include "recurrent_state_cache.hpp"

namespace pie_cuda_driver {
class ExpertStreamCache;
}

namespace pie_cuda_driver::model {

// Qwen3.5-MoE IModel (covers Qwen3.6-35B-A3B). Façade over the existing
// qwen3_5_la_ws, qwen3_5_moe_ws, and qwen3_5_state_cache locals in
// entry.cpp; owns its own Qwen3_5ForwardCfg + Qwen3_5PlanState.
//
// MTP system_drafter wiring stays in entry.cpp for now — it's a separate
// concern that interacts with NativeSystemDrafter, not IModel.
class Qwen35MoeModel final : public IModel {
public:
    Qwen35MoeModel(
        const Qwen3_5MoeWeights& weights,
        const HfConfig& hf_config,
        Qwen3_5LinearAttnWorkspace& la_ws,
        Qwen3_5MoeMlpWorkspace& moe_ws,
        RecurrentStateCache& state_cache,
        Qwen3_5PlanState& plan_state,
        KvCache& kv_cache,
        int tp_size,
        NcclComm* tp_comm,
        // Static runtime knobs computed once at construction.
        bool force_prefill_path,
        int small_prefill_naive_attention_max_tokens,
        bool graph_safe,
        bool supports_small_prefill_graph,
        ExpertStreamCache* expert_cache = nullptr);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Qwen3Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    RecurrentStateCache* state_cache() override { return &state_cache_; }
    std::uint32_t graph_layout() override;

    // Wires up the MoE variant of MTP onto the executor's NativeSystemDrafter.
    void wire_system_drafter(NativeSystemDrafter& drafter,
                             int max_drafts,
                             int draft_position_offset,
                             bool prefix_global_cache);

private:
    const Qwen3_5MoeWeights& weights_;
    const HfConfig& hf_config_;
    Qwen3_5LinearAttnWorkspace& la_ws_;
    Qwen3_5MoeMlpWorkspace& moe_ws_;
    RecurrentStateCache& state_cache_;
    Qwen3_5PlanState& plan_state_;
    KvCache& kv_cache_;
    Qwen3_5ForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
