#include "model/glm5/glm5_model.hpp"

#include <utility>

namespace pie_cuda_driver::model {

Glm5Model::Glm5Model(
    Glm5Weights weights,
    const HfConfig& hf_config,
    Glm5Workspace& ws,
    MlaCache& mla_cache,
    DsaCache& dsa_cache,
    int tp_size,
    NcclComm* tp_comm,
    bool emit_logits)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      ws_(ws),
      mla_cache_(mla_cache),
      dsa_cache_(dsa_cache)
{
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_comm = tp_comm;
    fwd_cfg_.emit_logits = emit_logits;

    caps_.supports_compact_logits = true;
}

void Glm5Model::prepare(AttentionWorkspace& attn_ws,
                        const ForwardFn::PrepareInputs& in) {
    prepare_kimi_mla_plan(
        mla_plan_, attn_ws, mla_cache_, hf_config_,
        in.kv_page_indices_d, in.qo_indptr_h, in.kv_page_indptr_h,
        in.kv_page_indptr_d, in.kv_last_page_lens_h, in.kv_last_page_lens_d,
        in.total_tokens, in.num_requests, !in.is_pure_decode,
        fwd_cfg_.tp_size);
}

void Glm5Model::body(Workspace& ws,
                     KvCache& /*kv*/,
                     AttentionWorkspace& attn_ws,
                     ops::CublasHandle& cublas,
                     const ForwardFn::ForwardInputs& in) {
    glm5_forward_paged(
        weights_, hf_config_, fwd_cfg_, mla_plan_, ws_,
        mla_cache_, dsa_cache_, attn_ws, cublas, ws.logits.data(),
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.row_valid_d,
        in.logit_row_indices_d, in.num_logit_rows);
}

}  // namespace pie_cuda_driver::model
