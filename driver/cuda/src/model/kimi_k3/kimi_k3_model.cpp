#include "model/kimi_k3/kimi_k3_model.hpp"

#include <utility>

namespace pie_cuda_driver::model {

KimiK3Model::KimiK3Model(
    KimiK3Weights weights,
    const HfConfig& hf_config,
    KimiK3Workspace& ws,
    MlaCache& mla_cache,
    RecurrentStateCache& kda_cache,
    int tp_size,
    NcclComm* tp_comm,
    bool emit_logits)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      ws_(ws),
      mla_cache_(mla_cache),
      kda_cache_(kda_cache)
{
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_comm = tp_comm;
    fwd_cfg_.emit_logits = emit_logits;

    caps_.supports_compact_logits = true;
    // Every per-fire decision this forward makes is a host decision about host
    // data: the layer kinds come from the config, the AttnRes block count is a
    // host loop over a fixed depth, the KDA recurrence walks its window inside
    // one kernel, and the MoE routes on device through the aligned path. So
    // the launch sequence depends only on the batch shape, which is what graph
    // capture requires. The fresh-slot reset reads `is_fresh_h`, also host.
    caps_.graph_safe = true;
    caps_.graph_padding_kv_write_safe = true;
}

void KimiK3Model::prepare(AttentionWorkspace& attn_ws,
                          const ForwardFn::PrepareInputs& in) {
    // The MLA plan covers only the `full_attention` layers, which is what
    // `MlaCache` was sized for; the KDA layers have no page table.
    prepare_kimi_mla_plan(
        mla_plan_, attn_ws, mla_cache_, hf_config_,
        in.kv_page_indices_d, in.qo_indptr_h, in.kv_page_indptr_h,
        in.kv_page_indptr_d, in.kv_last_page_lens_h, in.kv_last_page_lens_d,
        in.total_tokens, in.num_requests, !in.is_pure_decode,
        fwd_cfg_.tp_size);
}

void KimiK3Model::body(Workspace& ws,
                       KvCache& /*kv*/,
                       AttentionWorkspace& attn_ws,
                       ops::CublasHandle& cublas,
                       const ForwardFn::ForwardInputs& in) {
    kimi_k3_forward_paged(
        weights_, hf_config_, fwd_cfg_, mla_plan_, ws_, mla_cache_, kda_cache_,
        attn_ws, cublas, ws.logits.data(),
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.slot_ids_h, in.slot_ids_d,
        in.is_fresh_h, in.is_fresh_d,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.row_valid_d,
        in.logit_row_indices_d, in.num_logit_rows);
}

}  // namespace pie_cuda_driver::model
