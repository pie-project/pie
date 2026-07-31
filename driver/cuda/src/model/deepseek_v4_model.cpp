#include "model/deepseek_v4_model.hpp"

namespace pie_cuda_driver::model {

DsV4Model::DsV4Model(
    const DsV4Weights& weights,
    const HfConfig& hf_config,
    DsV4Workspace& ws,
    int tp_size,
    int tp_rank,
    NcclComm* tp_comm,
    bool emit_logits,
    ExpertStreamCache* expert_cache)
    : weights_(weights),
      hf_config_(hf_config),
      ws_(ws)
{
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_rank = tp_rank;
    fwd_cfg_.tp_comm = tp_comm;
    fwd_cfg_.emit_logits = emit_logits;
    fwd_cfg_.expert_cache = expert_cache;

    // DSV4 emits dense or compact logits via the standard path; it does
    // not opt into CUDA-graph capture or fused-argmax.
    caps_.supports_compact_logits = true;
}

void DsV4Model::prepare(AttentionWorkspace& /*attn_ws*/,
                        const ForwardFn::PrepareInputs& /*in*/) {
    // DSV4 has no host-side per-fire plan setup.
}

void DsV4Model::body(Qwen3Workspace& ws,
                     KvCache& kv,
                     AttentionWorkspace& attn_ws,
                     ops::CublasHandle& cublas,
                     const ForwardFn::ForwardInputs& in) {
    dsv4_forward_paged(
        weights_, hf_config_, fwd_cfg_, ws_, kv, attn_ws, cublas,
        ws.logits.data(),
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.logit_row_indices_d, in.num_logit_rows);
}

}  // namespace pie_cuda_driver::model
