#include "model/llama_like/llama_like_model.hpp"

#include <utility>

namespace pie_cuda_driver::model {

LlamaLikeModel::LlamaLikeModel(
    Qwen3Weights weights,
    const HfConfig& hf_config,
    KvCache& kv_cache,
    const LlamaLikeForwardCfg& fwd_cfg)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      kv_cache_(kv_cache),
      fwd_cfg_(fwd_cfg)
{
    // Llama-like decode is graph-replay-safe because (a) the body is
    // host-work-free (the prepare hook hoisted DecodePlan out of the
    // capture region); (b) flashinfer's plan_info layout is pinned across
    // fires when enable_cuda_graph=true — padded_batch_size =
    // max_grid_size / gdy (stable), and the int_buf offsets are
    // deterministic from that. Quantized KV currently dequantizes active
    // physical pages into a BF16 scratch cache before FlashInfer; that
    // dequant launch shape depends on the live page count, while decode
    // graph keys only bucket request count/layout — replay would leave
    // newly-active pages stale, so we gate graph_safe on native BF16.
    caps_.graph_safe = kv_cache_.format().is_native_bf16();
    caps_.supports_compact_logits = true;
    caps_.supports_runtime_window = true;
}

void LlamaLikeModel::prepare(AttentionWorkspace& attn_ws,
                             const ForwardFn::PrepareInputs& in) {
    LlamaLikeForwardCfg runtime_cfg = fwd_cfg_;
    if (in.runtime_window_left >= -1) {
        runtime_cfg.sliding_window = in.runtime_window_left;
        runtime_cfg.per_layer_window_left.clear();
        if (in.runtime_window_left >= 0) {
            runtime_cfg.use_xqa_decode = false;
        }
    }
    prepare_llama_like_decode_plan(
        plan_, attn_ws, kv_cache_, hf_config_, runtime_cfg,
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

void LlamaLikeModel::body(Workspace& ws,
                          KvCache& kv,
                          AttentionWorkspace& attn_ws,
                          ops::CublasHandle& cublas,
                          const ForwardFn::ForwardInputs& in) {
    llama_like_forward_paged(
        weights_, hf_config_, fwd_cfg_, plan_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.logit_row_indices_d, in.num_logit_rows,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.w_page_d, in.w_off_d, in.row_valid_d, in.has_write_desc,
        in.runtime_window_left);
}

std::uint32_t LlamaLikeModel::graph_layout() {
    return llama_like_decode_graph_layout(plan_);
}

}  // namespace pie_cuda_driver::model
