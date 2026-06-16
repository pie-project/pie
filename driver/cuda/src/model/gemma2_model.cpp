#include "model/gemma2_model.hpp"

namespace pie_cuda_driver::model {

Gemma2Model::Gemma2Model(const Gemma2Weights& weights,
                         const HfConfig& hf_config,
                         const Gemma2ForwardCfg& fwd_cfg)
    : weights_(weights), hf_config_(hf_config), fwd_cfg_(fwd_cfg) {}

void Gemma2Model::body(Qwen3Workspace& ws,
                       KvCache& kv,
                       AttentionWorkspace& attn_ws,
                       ops::CublasHandle& cublas,
                       const ForwardFn::ForwardInputs& in) {
    gemma2_forward_paged(
        weights_, hf_config_, fwd_cfg_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d,
        in.kv_page_indptr_d, in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.custom_mask_d, in.custom_mask_indptr_d);
}

}  // namespace pie_cuda_driver::model
