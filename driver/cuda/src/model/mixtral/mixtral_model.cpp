#include "model/mixtral/mixtral_model.hpp"

#include <utility>

namespace pie_cuda_driver::model {

MixtralModel::MixtralModel(MixtralWeights weights,
                           const HfConfig& hf_config,
                           const LlamaLikeForwardCfg& fwd_cfg,
                           int num_experts,
                           int top_k)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      fwd_cfg_(fwd_cfg),
      num_experts_(num_experts),
      top_k_(top_k) {
    // The lm_head reads only the sampled rows; see `mixtral_forward_paged`.
    caps_.supports_compact_logits = true;
    // Every KV write is gated on `row_valid`, so padded rows from a
    // device-composed descriptor set cannot touch the cache. This is what
    // lets the family use device composition; CUDA-graph capture is a
    // separate, stronger claim the host-side MoE routing cannot make.
    caps_.graph_padding_kv_write_safe = true;
    // Trace and VALIDATE the declaration at load, the tree-wide polarity.
    // A refusal is a fallback: `usable` stays false and every fire takes
    // the hand-written pass.
    if (gpt_oss_declared_forward_enabled()) {
        declared_ = build_gpt_oss_declared_plan(hf_config_, weights_, fwd_cfg_,
                                                num_experts_, top_k_);
    }
}

void MixtralModel::body(Workspace& ws,
                        KvCache& kv,
                        AttentionWorkspace& attn_ws,
                        ops::CublasHandle& cublas,
                        const ForwardFn::ForwardInputs& in) {
    // The declared drive gets the fire first. It answers false for
    // anything outside the decode class — a prefill, a masked or hooked
    // fire, a fire past the fused leg's route cap — and the hand-written
    // pass runs it unchanged.
    const bool declared_eligible =
        gpt_oss_declared_drive_enabled() && declared_.usable &&
        in.custom_mask_d == nullptr && in.stage_hooks == nullptr &&
        fwd_cfg_.tp_size == 1;
    if (declared_eligible &&
        gpt_oss_forward_declared(
            declared_, weights_, hf_config_, fwd_cfg_, num_experts_, top_k_,
            ws, kv, attn_ws, cublas, in.token_ids, in.positions,
            in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
            in.kv_last_page_lens_d, in.kv_page_indptr_h,
            in.total_tokens, in.num_requests, in.is_pure_decode,
            in.row_valid_d, in.logit_row_indices_d, in.num_logit_rows)) {
        return;
    }
    mixtral_forward_paged(
        weights_, hf_config_, fwd_cfg_,
        num_experts_, top_k_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d,
        in.kv_page_indptr_d, in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.logit_row_indices_d, in.num_logit_rows,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.row_valid_d,
        in.stage_hooks);
}

}  // namespace pie_cuda_driver::model
