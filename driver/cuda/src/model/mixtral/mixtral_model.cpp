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
      top_k_(top_k) {}

void MixtralModel::body(Workspace& ws,
                        KvCache& kv,
                        AttentionWorkspace& attn_ws,
                        ops::CublasHandle& cublas,
                        const ForwardFn::ForwardInputs& in) {
    mixtral_forward_paged(
        weights_, hf_config_, fwd_cfg_,
        num_experts_, top_k_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d,
        in.kv_page_indptr_d, in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.custom_mask_d, in.custom_mask_indptr_d);
}

}  // namespace pie_cuda_driver::model
