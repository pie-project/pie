#include "model/qwen3_5/qwen3_5_moe_model.hpp"

#include <cstdio>
#include <utility>

#include "model/qwen3_5/declared_forward.hpp"

namespace pie_cuda_driver::model {

Qwen35MoeModel::Qwen35MoeModel(
    Qwen3_5MoeWeights weights,
    const HfConfig& hf_config,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    RecurrentStateCache& state_cache,
    Qwen3_5PlanState& plan_state,
    KvCache& kv_cache,
    int tp_size,
    NcclComm* tp_comm,
    bool force_prefill_path,
    int small_prefill_naive_attention_max_tokens,
    bool graph_safe,
    bool supports_small_prefill_graph)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      la_ws_(la_ws),
      moe_ws_(moe_ws),
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
    caps_.graph_padding_kv_write_safe  = true;
    caps_.supports_compact_logits      = true;
    caps_.supports_small_prefill_graph = supports_small_prefill_graph;

    // Declared executor: same cold-start trace + validation as the dense
    // model (qwen3_5_model.cpp) — the two classes do not share a
    // construction site, so each opts in here. body() never consumes the
    // MoE plan this slice (no MoE emission arms yet; see body()).
    if (qwen35_declared_forward_enabled()) {
        declared_ = build_qwen3_5_declared_plan(hf_config_, weights_, tp_size);
    }
}

void Qwen35MoeModel::prepare(AttentionWorkspace& attn_ws,
                             const ForwardFn::PrepareInputs& in) {
    prepare_qwen3_5_decode_plan(
        plan_state_, attn_ws, kv_cache_, hf_config_,
        fwd_cfg_, in.qo_indptr_h, in.kv_page_indptr_h,
        in.kv_last_page_lens_h, in.total_tokens,
        in.num_requests, in.is_pure_decode);
}

void Qwen35MoeModel::body(Workspace& ws,
                          KvCache& kv,
                          AttentionWorkspace& attn_ws,
                          ops::CublasHandle& cublas,
                          const ForwardFn::ForwardInputs& in) {
    // Arc-2 decode slice: the MoE model ALWAYS falls back this slice. Its
    // validated plan carries the dyn MoE vocabulary (TopK, selector
    // Matmuls, WeightedSum, SigmoidGateAdd) which the executor's op-kind
    // switch has no emission rule for — the grouped-GEMM emission is a
    // later, much larger lift (commit 9c54b9b6's list). The trace-gated
    // line keeps the exclusion visible in A/B runs.
    if (static_cast<bool>(declared_) &&
        qwen35_declared_exec_trace_enabled()) {
        std::fprintf(stderr,
                     "[declared-qwen35-exec] fallback N=%d R=%d decode=%d "
                     "reason=moe emission arms absent this slice\n",
                     in.total_tokens, in.num_requests,
                     in.is_pure_decode ? 1 : 0);
    }
    qwen3_5_moe_forward_paged(
        weights_, hf_config_, fwd_cfg_, plan_state_,
        ws, la_ws_, moe_ws_, kv, state_cache_,
        attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.w_page_d, in.w_off_d, in.row_valid_d, in.has_write_desc,
        in.slot_ids_h, in.is_fresh_h, in.slot_ids_d, in.is_fresh_d,
        in.logit_row_indices_d, in.num_logit_rows,
        in.commit_advance_gather_d,
        in.rs_buffer_slot_ids_h, in.rs_buffer_slot_indptr_h,
        in.rs_fold_lens_d,
        in.rs_buffer_write, in.rs_buffer_fold,
        in.stage_hooks);
}

std::uint32_t Qwen35MoeModel::graph_layout() {
    return qwen3_5_decode_graph_layout(plan_state_);
}

void Qwen35MoeModel::wire_system_drafter(
    NativeSystemDrafter& drafter,
    int max_drafts,
    int draft_position_offset,
    bool prefix_global_cache)
{
    drafter.max_drafts = max_drafts;
    drafter.draft_position_offset = draft_position_offset;
    drafter.draft_global_cache_uses_prefix_position = prefix_global_cache;
    drafter.commit_verified_prefix =
        [this](const NativeSystemCommitInputs& in) {
            Qwen3_5ForwardCfg q35{};
            q35.tp_size = fwd_cfg_.tp_size;
            q35.tp_comm = fwd_cfg_.tp_comm;
            qwen3_5_moe_mtp_process_cache(
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
            Workspace& ws, KvCache& cache, ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::int32_t* base_hidden_row_indices,
            const std::int32_t* request_ids,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            std::int32_t* /*sampled_token_ids*/,
            int N, int draft_step, int max_global_tokens) {
            Qwen3_5ForwardCfg q35{};
            q35.tp_size = fwd_cfg_.tp_size;
            q35.tp_comm = fwd_cfg_.tp_comm;
            q35.mtp_global_cache_uses_prefix_position = prefix_global_cache;
            qwen3_5_moe_mtp_forward(
                weights_, hf_config_, q35,
                ws, la_ws_, moe_ws_, cache, cublas,
                tok, pos, base_hidden_row_indices, request_ids,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                nullptr, N, draft_step, max_global_tokens);
        };
}

}  // namespace pie_cuda_driver::model
