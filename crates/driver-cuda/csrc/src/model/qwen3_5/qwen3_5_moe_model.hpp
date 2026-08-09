#pragma once

#include "attention_workspace.hpp"
#include "distributed.hpp"
#include "model/imodel.hpp"
#include "model/qwen3_5/declared_facts.hpp"
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "store/recurrent_state_cache.hpp"

namespace pie_cuda_driver::model {

// Qwen3.5-MoE IModel (covers Qwen3.6-35B-A3B). Owns its bound weights plus
// references to the Context-allocated qwen3_5_la_ws/qwen3_5_moe_ws/
// qwen3_5_state_cache workspaces and its own Qwen3_5ForwardCfg +
// Qwen3_5PlanState.
//
// MTP system_drafter wiring is done by the registry's `create_model`
// factory (model/registry.cpp) right after construction — it is a
// separate concern that interacts with NativeSystemDrafter, not IModel.
class Qwen35MoeModel final : public IModel {
public:
    Qwen35MoeModel(
        Qwen3_5MoeWeights weights,
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
        bool supports_small_prefill_graph);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              kernels::gemm::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    RecurrentStateCache* state_cache() override { return &state_cache_; }
    std::uint32_t graph_layout() override;

    // The validated declared plan (empty → nullptr), for the load-time
    // capability site summary (imodel.hpp) — the one family whose plan can
    // carry expert sites when the facts are MoE.
    const pie_forward::ForwardPlan* declared_plan() const override {
        return declared_ ? &declared_.plan : nullptr;
    }

    // Same linear-attention scratch as `Qwen35Model::workspace_bytes`, plus
    // the routed/shared-MoE MLP scratch.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        return IModel::workspace_bytes(cfg, max_tokens, output_rows) +
               qwen3_5_la_workspace_bytes(cfg, max_tokens, fwd_cfg_.tp_size) +
               qwen3_5_moe_workspace_bytes(cfg, max_tokens, fwd_cfg_.tp_size);
    }

    // Wires up the MoE variant of MTP onto the executor's NativeSystemDrafter.
    void wire_system_drafter(NativeSystemDrafter& drafter,
                             int max_drafts,
                             int draft_position_offset,
                             bool prefix_global_cache);

private:
    Qwen3_5MoeWeights weights_;
    const HfConfig& hf_config_;
    Qwen3_5LinearAttnWorkspace& la_ws_;
    Qwen3_5MoeMlpWorkspace& moe_ws_;
    RecurrentStateCache& state_cache_;
    Qwen3_5PlanState& plan_state_;
    KvCache& kv_cache_;
    Qwen3_5ForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
    // Arc 1 of the declared executor (declared_facts.hpp): the traced +
    // structurally validated plan, built at construction when
    // PIE_DECLARED_FORWARD opted in. Stored for arc 2; body() does NOT
    // consume it — cold-start validation only.
    Qwen35DeclaredPlan declared_;
};

}  // namespace pie_cuda_driver::model
