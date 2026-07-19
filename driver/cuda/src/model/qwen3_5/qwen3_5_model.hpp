#pragma once

#include "model/imodel.hpp"
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"

namespace pie_cuda_driver {
class NcclComm;
}

namespace pie_cuda_driver::model {

class Qwen35Model final : public IModel {
public:
    Qwen35Model(Qwen3_5Weights weights,
                const HfConfig& hf_config,
                Qwen3_5LinearAttnWorkspace& la_ws,
                RecurrentStateCache& state_cache,
                Qwen3_5PlanState& plan_state,
                KvCache& kv_cache,
                int tp_size,
                NcclComm* tp_comm,
                bool force_prefill_path,
                int small_prefill_naive_attention_max_tokens,
                bool graph_safe,
                bool supports_small_prefill_graph);

    void prepare(AttentionWorkspace& attn_ws,
                 const ForwardFn::PrepareInputs& in) override;
    void body(Workspace& ws,
              KvCache& kv,
              AttentionWorkspace& attn_ws,
              ops::CublasHandle& cublas,
              const ForwardFn::ForwardInputs& in) override;

    ModelCapabilities capabilities() const override { return caps_; }
    std::uint32_t graph_layout() override;

    // Qwen3.5's linear-attention layers need extra per-fire scratch
    // (mixed_qkv/conv/gating buffers) on top of the universal `Workspace`
    // formula, sized by the model's own runtime tp_size.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        return IModel::workspace_bytes(cfg, max_tokens, output_rows) +
               qwen3_5_la_workspace_bytes(cfg, max_tokens, fwd_cfg_.tp_size);
    }

    // Wires up commit_verified_prefix + draft_step on the executor's
    // NativeSystemDrafter using qwen3_5_mtp_{process_cache,forward}.
    // Caller has already confirmed weights_.mtp.has_value() and that
    // native_mtp_num_drafts > 0.
    void wire_system_drafter(NativeSystemDrafter& drafter,
                             int max_drafts,
                             int draft_position_offset,
                             bool prefix_global_cache,
                             bool mtp_fused_gemv_enabled);

private:
    Qwen3_5Weights weights_;
    const HfConfig& hf_config_;
    Qwen3_5LinearAttnWorkspace& la_ws_;
    RecurrentStateCache& state_cache_;
    Qwen3_5PlanState& plan_state_;
    KvCache& kv_cache_;
    Qwen3_5ForwardCfg fwd_cfg_;
    ModelCapabilities caps_;
};

}  // namespace pie_cuda_driver::model
