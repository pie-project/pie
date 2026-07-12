#include "model/kimi/kimi_model.hpp"

#include <utility>

namespace pie_cuda_driver::model {

KimiModel::KimiModel(
    KimiWeights weights,
    const HfConfig& hf_config,
    KimiWorkspace& ws,
    MlaCache& mla_cache,
    int tp_size,
    NcclComm* tp_comm,
    bool emit_logits,
    bool supports_tp_greedy_argmax)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      ws_(ws),
      mla_cache_(mla_cache)
{
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_comm = tp_comm;
    fwd_cfg_.emit_logits = emit_logits;

    caps_.supports_tp_greedy_argmax = supports_tp_greedy_argmax;
    caps_.supports_compact_logits = true;
}

void KimiModel::prepare(AttentionWorkspace& attn_ws,
                        const ForwardFn::PrepareInputs& in) {
    prepare_kimi_mla_plan(
        plan_state_, attn_ws, mla_cache_, hf_config_,
        in.kv_page_indices_d, in.qo_indptr_h, in.kv_page_indptr_h,
        in.kv_page_indptr_d, in.kv_last_page_lens_h, in.kv_last_page_lens_d,
        in.total_tokens, in.num_requests, !in.is_pure_decode,
        fwd_cfg_.tp_size);
}

void KimiModel::body(Workspace& ws,
                     KvCache& /*kv*/,
                     AttentionWorkspace& attn_ws,
                     ops::CublasHandle& cublas,
                     const ForwardFn::ForwardInputs& in) {
    kimi_forward_paged(
        weights_, hf_config_, fwd_cfg_, plan_state_, ws_, mla_cache_,
        attn_ws, cublas, ws.logits.data(),
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.logit_row_indices_d, in.num_logit_rows);
}

}  // namespace pie_cuda_driver::model
