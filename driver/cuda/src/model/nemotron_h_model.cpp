#include "model/nemotron_h_model.hpp"

#include <cstdlib>

namespace pie_cuda_driver::model {

NemotronHModel::NemotronHModel(
    const NemotronHWeights& weights,
    const HfConfig& hf_config,
    NemotronHWorkspace& ws,
    RecurrentStateCache& state_cache,
    KvCache& kv_cache,
    const LlamaLikeForwardCfg& base_fwd_cfg,
    int tp_size,
    NcclComm* tp_comm)
    : weights_(weights),
      hf_config_(hf_config),
      ws_(ws),
      state_cache_(state_cache),
      kv_cache_(kv_cache),
      fwd_cfg_(base_fwd_cfg)
{
    fwd_cfg_.tp_size = tp_size;
    fwd_cfg_.tp_comm = tp_comm;

    // CUDA graphs are opt-in for Nemotron-H: the Mamba/MoE paths trade
    // graph stability for first-fire capture overhead today, and the
    // current graph replay was measured slower on this model in nsys.
    // PIE_NEMOTRON_ENABLE_CUDA_GRAPHS=1 brings it back when the kernel
    // story stabilizes.
    const char* graphs_env = std::getenv("PIE_NEMOTRON_ENABLE_CUDA_GRAPHS");
    caps_.graph_safe =
        kv_cache_.format().is_native_bf16() &&
        graphs_env != nullptr &&
        graphs_env[0] != '\0' &&
        graphs_env[0] != '0';
    caps_.supports_compact_logits = true;
}

void NemotronHModel::prepare(AttentionWorkspace& attn_ws,
                             const ForwardFn::PrepareInputs& in) {
    prepare_llama_like_decode_plan(
        plan_, attn_ws, kv_cache_, hf_config_, fwd_cfg_,
        in.qo_indptr_h,
        in.kv_page_indices_d,
        in.kv_page_indptr_h,
        in.kv_page_indptr_d,
        in.kv_last_page_lens_h,
        in.kv_last_page_lens_d,
        in.total_tokens,
        in.num_requests,
        in.is_pure_decode);
}

void NemotronHModel::body(Qwen3Workspace& ws,
                          KvCache& kv,
                          AttentionWorkspace& attn_ws,
                          ops::CublasHandle& cublas,
                          const ForwardFn::ForwardInputs& in) {
    nemotron_h_forward_paged(
        weights_, hf_config_, fwd_cfg_, plan_,
        ws, ws_, kv, state_cache_,
        attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.slot_ids_h, in.is_fresh_h, in.slot_ids_d,
        in.logit_row_indices_d, in.num_logit_rows);
}

std::uint32_t NemotronHModel::graph_layout() {
    return llama_like_decode_graph_layout(plan_);
}

}  // namespace pie_cuda_driver::model
