#include "model/qwen3_5/qwen3_5_moe_model.hpp"
#include "model/qwen3_5/declared_forward.hpp"
#include <algorithm>
#include "ops/flashinfer_moe.hpp"

#include <cstdlib>
#include <utility>

#include "model/stage_hooks.hpp"

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

    // The per-expert dispatch reads the routing table back off the device and
    // syncs on it, and page-in adds a host-chosen slot and a plan run on top.
    // None of that is capturable.
    //
    // Unlike DeepSeek-V4 this is not conditional on the cache being able to
    // miss: streamed experts have no fused slab to stride, so every device-side
    // path is off the table and the forward takes the syncing one even when the
    // slab holds the whole group.
    bool host_work_in_forward = qwen35_moe_force_general_path();
    for (const auto& L : weights_.layers) {
        if (L.expert_cache != nullptr) {
            host_work_in_forward = true;
            break;
        }
    }
    // A profiled forward times stages with CUDA events, which is illegal on a
    // capturing stream — so profiling and graph capture are mutually
    // exclusive. Drop capture rather than silently producing no timings.
    const char* moe_profile = std::getenv("PIE_QWEN35_MOE_PROFILE");
    const bool moe_profile_on =
        moe_profile != nullptr && moe_profile[0] != '\0' && moe_profile[0] != '0';
    if (host_work_in_forward || moe_profile_on) {
        // Only the capture caps. `graph_padding_kv_write_safe` states that this
        // family's KV writes are gated on `row_valid`, which is a property of
        // its kernels and stays true either way -- startup validates it against
        // the padding aliasing.
        caps_.graph_safe = false;
        caps_.supports_small_prefill_graph = false;
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
    // Arc 2, MoE half. Behind its own gate and OFF by default: the dense
    // arc has an A/B behind it, this one is new, and a MoE fire that
    // silently takes a different leg is not something a logit comparison
    // would necessarily catch.
    //
    // The declaration states the ALIGNED leg specifically, so every term
    // here asks the same question in a different way: would this fire have
    // taken some other leg? A fire that would fuse through CUTLASS, or fall
    // to the per-route GEMV, runs a different set of kernels, and the walk
    // would quietly substitute the aligned one.
    const char* fallback_reason = nullptr;
    if (static_cast<bool>(declared_) && qwen35_declared_moe_enabled()) {
        const int routes =
            in.total_tokens * hf_config_.num_experts_per_tok;
        if (in.lora != nullptr) {
            fallback_reason = "lora fire";
        } else if (in.custom_mask_d != nullptr) {
            fallback_reason = "custom mask";
        } else if (in.rs_buffer_write || in.rs_buffer_fold) {
            fallback_reason = "rs-buffer write/fold fire";
        } else if (in.has_write_desc &&
                   (in.w_page_d == nullptr || in.w_off_d == nullptr)) {
            fallback_reason = "write descriptors missing";
        } else if (ops::flashinfer_cutlass_moe_enabled() &&
                   moe_ws_.cutlass_max_rows > 0 &&
                   in.total_tokens <= moe_ws_.cutlass_max_rows) {
            fallback_reason = "fire fits the fused CUTLASS leg";
        } else if (moe_ws_.aligned_block_size <= 1 ||
                   moe_ws_.aligned_expert_in.empty()) {
            fallback_reason = "aligned leg not staged";
        } else if (routes < kQwen35MoeAlignedDecodeMinRoutes) {
            fallback_reason = "too few routes for the aligned leg";
        } else if (!in.is_pure_decode &&
                   in.total_tokens > kQwen35MoeDecodeFastMaxTokens) {
            // Above the bound the hand body leaves the decode fast path,
            // and with it `add_to_residual` -- so the combine stops being
            // the `_add_` form the declaration states.
            fallback_reason = "fire is past the decode fast-path bound";
        } else if (std::any_of(
                       weights_.layers.begin(), weights_.layers.end(),
                       [](const Qwen3_5MoeLayerWeights& l) {
                           return l.expert_cache != nullptr ||
                                  l.shared_gate_up_gate_proj != nullptr ||
                                  l.shared_gate == nullptr ||
                                  l.shared_gate_quant.has_value();
                       })) {
            // Each of these makes the shared expert's landing a DIFFERENT
            // launch than the one declared: a streamed expert cache leaves
            // the fast path entirely, the fused scalar gate rides the
            // gate_up bank's extra column, and a missing or quantized
            // gate takes the gemm-then-broadcast pair.
            fallback_reason = "shared-expert landing is not the declared one";
        }
        if (fallback_reason == nullptr) {
            ScopedStageHooks declared_hooks(in.stage_hooks);
            const bool handled = qwen3_5_forward_declared(
                declared_, weights_, hf_config_, fwd_cfg_, plan_state_,
                ws, &moe_ws_, la_ws_, kv, state_cache_, attn_ws, cublas,
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
        }
        if (qwen35_declared_exec_trace_enabled()) {
            std::fprintf(stderr,
                         "[declared-qwen35-moe-exec] fallback N=%d R=%d "
                         "decode=%d reason=%s\n",
                         in.total_tokens, in.num_requests,
                         in.is_pure_decode ? 1 : 0,
                         fallback_reason ? fallback_reason : "no class");
        }
    }

    // Same ambient install as the dense model: the MoE hand body's hook
    // invocations are the point-first overload and no-op without it.
    ScopedStageHooks ambient_hooks(in.stage_hooks);
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
        in.rs_fold_lens_h,
        in.rs_buffer_write, in.rs_buffer_fold,
        in.rs_buffer_read_slot_ids_h, in.rs_buffer_read_indptr_h,
        in.rs_buffer_read_lens_h, in.rs_buffer_heads_h);
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
