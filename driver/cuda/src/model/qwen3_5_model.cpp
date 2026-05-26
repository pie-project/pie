#include "model/qwen3_5_model.hpp"

namespace pie_cuda_driver::model {

Qwen35Model::Qwen35Model(
    const Qwen3_5Weights& weights,
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
    bool supports_small_prefill_graph)
    : weights_(weights),
      hf_config_(hf_config),
      la_ws_(la_ws),
      state_cache_(state_cache),
      plan_state_(plan_state),
      kv_cache_(kv_cache)
{
    fwd_cfg_.force_prefill_path = force_prefill_path;
    fwd_cfg_.small_prefill_naive_attention_max_tokens =
        small_prefill_naive_attention_max_tokens;
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_comm = tp_comm;

    caps_.graph_safe                   = graph_safe;
    caps_.supports_compact_logits      = true;
    caps_.supports_small_prefill_graph = supports_small_prefill_graph;
}

void Qwen35Model::prepare(AttentionWorkspace& attn_ws,
                          const ForwardFn::PrepareInputs& in) {
    prepare_qwen3_5_decode_plan(
        plan_state_, attn_ws, kv_cache_, hf_config_,
        fwd_cfg_, in.qo_indptr_h, in.kv_page_indptr_h,
        in.kv_last_page_lens_h, in.total_tokens,
        in.num_requests, in.is_pure_decode);
}

void Qwen35Model::body(Qwen3Workspace& ws,
                       KvCache& kv,
                       AttentionWorkspace& attn_ws,
                       ops::CublasHandle& cublas,
                       const ForwardFn::ForwardInputs& in) {
    qwen3_5_forward_paged(
        weights_, hf_config_, fwd_cfg_, plan_state_,
        ws, la_ws_, kv, state_cache_,
        attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d,
        in.kv_page_indptr_d, in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.slot_ids_h, in.is_fresh_h, in.slot_ids_d,
        in.logit_row_indices_d, in.num_logit_rows);
}

std::uint32_t Qwen35Model::graph_layout() {
    return qwen3_5_decode_graph_layout(plan_state_);
}

void Qwen35Model::wire_system_drafter(
    NativeSystemDrafter& drafter,
    int max_drafts,
    int draft_position_offset,
    bool prefix_global_cache,
    bool mtp_fused_gemv_enabled)
{
    drafter.max_drafts = max_drafts;
    drafter.draft_position_offset = draft_position_offset;
    drafter.draft_global_cache_uses_prefix_position = prefix_global_cache;
    drafter.draft_step_writes_sampled_tokens =
        weights_.mtp->lm_head_scale_inv != nullptr || mtp_fused_gemv_enabled;
    drafter.commit_verified_prefix =
        [this](const NativeSystemCommitInputs& in) {
            Qwen3_5ForwardCfg q35{};
            q35.tp_size = fwd_cfg_.tp_size;
            q35.tp_comm = fwd_cfg_.tp_comm;
            qwen3_5_mtp_process_cache(
                weights_, hf_config_, q35,
                in.target_ws, la_ws_, in.kv_cache,
                state_cache_, in.cublas,
                in.token_ids, in.positions, in.qo_indptr,
                in.kv_page_indices, in.kv_page_indptr,
                in.kv_last_page_lens, in.slot_ids,
                in.source_row_indices, in.total_tokens,
                in.num_requests);
        };
    drafter.draft_step =
        [this, prefix_global_cache](
            Qwen3Workspace& ws, KvCache& cache, ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::int32_t* base_hidden_row_indices,
            const std::int32_t* request_ids,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            std::int32_t* sampled_token_ids,
            int N, int draft_step, int max_global_tokens) {
            Qwen3_5ForwardCfg q35{};
            q35.tp_size = fwd_cfg_.tp_size;
            q35.tp_comm = fwd_cfg_.tp_comm;
            q35.mtp_global_cache_uses_prefix_position = prefix_global_cache;
            qwen3_5_mtp_forward(
                weights_, hf_config_, q35,
                ws, la_ws_, cache, cublas,
                tok, pos, base_hidden_row_indices, request_ids,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                sampled_token_ids, N, draft_step, max_global_tokens);
        };
}

}  // namespace pie_cuda_driver::model
