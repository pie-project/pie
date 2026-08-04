#include "model/qwen3_5/qwen3_5_model.hpp"

#include <cstdio>
#include <utility>

#include "model/qwen3_5/declared_forward.hpp"

namespace pie_cuda_driver::model {

Qwen35Model::Qwen35Model(
    Qwen3_5Weights weights,
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
    : weights_(std::move(weights)),
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
    caps_.graph_padding_kv_write_safe  = true;
    caps_.supports_compact_logits      = true;
    caps_.supports_small_prefill_graph = supports_small_prefill_graph;

    // Declared executor: trace + structurally validate the hybrid
    // declaration against this deployment's config and bindings, now
    // rather than on first fire (the facts are load-time facts —
    // llama_like_model.cpp's reasoning). An unrepresentable config leaves
    // `declared_` empty with the reason logged once and body() keeps the
    // hand-written path; a validated plan is consumed by body()'s
    // eligibility gate below.
    if (qwen35_declared_forward_enabled()) {
        declared_ = build_qwen3_5_declared_plan(hf_config_, weights_, tp_size);
    }
}

void Qwen35Model::prepare(AttentionWorkspace& attn_ws,
                          const ForwardFn::PrepareInputs& in) {
    prepare_qwen3_5_decode_plan(
        plan_state_, attn_ws, kv_cache_, hf_config_,
        fwd_cfg_, in.qo_indptr_h, in.kv_page_indptr_h,
        in.kv_last_page_lens_h, in.total_tokens,
        in.num_requests, in.is_pure_decode);
}

void Qwen35Model::body(Workspace& ws,
                       KvCache& kv,
                       AttentionWorkspace& attn_ws,
                       ops::CublasHandle& cublas,
                       const ForwardFn::ForwardInputs& in) {
    // The declared executor covers the hand-written TP=1 base pass —
    // decode AND prefill fires (arc 3; mixed prefill+decode fires are the
    // same qo_indptr-windowed prefill shape) — anything it cannot express
    // falls back, per fire, to the hand-written body below. Build-time
    // exclusions (TP>1, quantized projections, irregular layer schedule,
    // mixed fused bindings, ...) already left `declared_` empty. Each term
    // names the hand-written per-fire service the walk does not mirror;
    // `fallback_reason` keeps the exclusion honest in the trace log (a
    // silent fallback would be indistinguishable from a passing A/B run).
    const char* fallback_reason = nullptr;
    if (static_cast<bool>(declared_)) {
        const int qgkv_dim =
            2 * hf_config_.num_attention_heads * hf_config_.head_dim +
            2 * hf_config_.num_key_value_heads * hf_config_.head_dim;
        // (Stage hooks are IN scope since A4: the class traces carry the
        // HookSite ops — qwen3_5's observation-only sites — and the walk
        // invokes them with the hand-written buffers.)
        if (in.lora != nullptr) {
            // The plan has no correction op; running the walk would
            // silently drop the adapter (llama_like's reasoning). The
            // hand-written qwen3_5 body ignores lora too, but the honest
            // gate is exclusion, not shared omission.
            fallback_reason = "lora fire";
        } else if (in.custom_mask_d != nullptr) {
            fallback_reason = "custom mask";
        } else if (in.rs_buffer_write || in.rs_buffer_fold) {
            // Excluded by DESIGN, not backlog — the Stage-2 recon verdict
            // (stage1-notes.md, "RS-buffer solo relax: rejected"):
            // FoldBuffered is unbatchable by WIT contract (the driver
            // rejects a batch mixing folded and forward rows) and Buffer
            // has zero production callers, so the per-slab host-driven
            // memcpy loops stay a hand-written service the walk does not
            // mirror.
            fallback_reason =
                "rs-buffer write/fold fire (stage-2 verdict: unbatchable "
                "by contract / zero callers)";
        } else if (in.has_write_desc &&
                   (in.w_page_d == nullptr || in.w_off_d == nullptr)) {
            // Same guard the hand-written explicit-write validation makes.
            fallback_reason = "write descriptors missing";
        } else if (declared_.fused_full_attn_qgkv &&
                   (ws.gate_up_fused.empty() ||
                    ws.gate_up_fused.numel() <
                        static_cast<std::size_t>(in.total_tokens) *
                            qgkv_dim)) {
            // The trace committed to the fused qgkv bank; a workspace that
            // cannot stage it would make the hand-written body fall back
            // to the unfused GEMMs per layer — a shape the fused trace
            // cannot express (the hand-written `use_fused_qgkv` check).
            fallback_reason = "fused qgkv staging buffer unavailable";
        }
        // (MTP draft fires need no term here: drafting enters through
        // wire_system_drafter's own entry points, never body(). The
        // MTP-adjacent shapes that DO route through body() are declared
        // since this arc: state-only fires (num_logit_rows < 0), frozen
        // verify (state_cache_.verify_frozen() read inside the walk), and
        // commit-advance fires (commit_advance_gather_d threaded below as
        // the walk's commit_lens; its rs_buffer_fold flavor stays behind
        // the rs-buffer term above).)
        if (fallback_reason == nullptr) {
            const bool handled = qwen3_5_forward_declared(
                declared_, weights_, hf_config_, fwd_cfg_, plan_state_,
                ws, la_ws_, kv, state_cache_, attn_ws, cublas,
                in.token_ids, in.positions,
                in.qo_indptr_d, in.kv_page_indices_d,
                in.kv_page_indptr_d, in.kv_last_page_lens_d,
                in.qo_indptr_h, in.kv_page_indptr_h,
                in.total_tokens, in.num_requests, in.is_pure_decode,
                in.w_page_d, in.w_off_d, in.row_valid_d, in.has_write_desc,
                in.slot_ids_h, in.is_fresh_h, in.slot_ids_d, in.is_fresh_d,
                in.logit_row_indices_d, in.num_logit_rows,
                in.commit_advance_gather_d,
                in.stage_hooks);
            if (handled) return;
            // No class for this fire (rung 5): the hand-written path
            // below serves it, exactly as before the declared gate.
        }
        if (qwen35_declared_exec_trace_enabled()) {
            std::fprintf(stderr,
                         "[declared-qwen35-exec] fallback N=%d R=%d "
                         "decode=%d reason=%s\n",
                         in.total_tokens, in.num_requests,
                         in.is_pure_decode ? 1 : 0, fallback_reason);
        }
    }
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
        in.w_page_d, in.w_off_d, in.row_valid_d, in.has_write_desc,
        in.slot_ids_h, in.is_fresh_h, in.slot_ids_d, in.is_fresh_d,
        in.logit_row_indices_d, in.num_logit_rows,
        in.commit_advance_gather_d,
        in.rs_buffer_slot_ids_h, in.rs_buffer_slot_indptr_h,
        in.rs_fold_lens_d,
        in.rs_fold_lens_h,
        in.rs_buffer_write, in.rs_buffer_fold,
        in.rs_buffer_read_slot_ids_h, in.rs_buffer_read_indptr_h,
        in.rs_buffer_read_lens_h, in.rs_buffer_heads_h);
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
            Workspace& ws, KvCache& cache, ops::CublasHandle& cublas,
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
