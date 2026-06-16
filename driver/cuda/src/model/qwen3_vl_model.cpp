#include "model/qwen3_vl_model.hpp"

#include <vector>

#include "model/qwen3_vl_vision_adapter.hpp"  // to_vis_raw_qwen
#include "model/qwen3_vl_vision_forward.hpp"   // Qwen3VLVisionInputs, scatter

namespace pie_cuda_driver::model {

Qwen3VLModel::Qwen3VLModel(
    const Qwen3Weights& text_weights,
    const HfConfig& hf_config,
    KvCache& kv_cache,
    const LlamaLikeForwardCfg& fwd_cfg,
    int max_workspace_tokens,
    const Qwen3VLVisionWeights* vision)
    : weights_(text_weights),
      hf_config_(hf_config),
      kv_cache_(kv_cache),
      fwd_cfg_(fwd_cfg),
      max_tokens_(max_workspace_tokens)
{
    if (vision != nullptr) {
        vision_raw_ = to_vis_raw_qwen(*vision);
        has_vision_ = true;
        num_deepstack_ = static_cast<int>(vision->deepstack.size());
    }
    // Decode fires (no image) are plain Qwen3 and graph-replay safe on a
    // native bf16 KV cache; image fires are prefills (never graphed).
    caps_.graph_safe = kv_cache_.format().is_native_bf16();
    caps_.supports_compact_logits = true;
}

void Qwen3VLModel::prepare(AttentionWorkspace& attn_ws,
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

void Qwen3VLModel::body(Qwen3Workspace& ws,
                        KvCache& kv,
                        AttentionWorkspace& attn_ws,
                        ops::CublasHandle& cublas,
                        const ForwardFn::ForwardInputs& in) {
    cudaStream_t stream = cublas.stream();
    const int H = hf_config_.hidden_size;
    const int N = in.total_tokens;

    // Assemble the per-fire multimodal side-inputs. No-op when no images.
    Qwen3VLVisionInputs vision_in;
    LlamaLikeVisionInputs vision;
    const LlamaLikeVisionInputs* vision_ptr = nullptr;

    const bool has_image = has_vision_ && in.num_images > 0;

    // Upload the assembled [N,3] M-RoPE positions on image-carrying fires.
    // The executor builds `mrope_positions_h` (text rows = (p,p,p), image rows
    // = the staged 3-axis positions) only when images are in the batch. On
    // pure-text decode fires `mrope_positions_h` is null and the forward falls
    // back to standard RoPE over the 1-D `positions` — which is exactly M-RoPE
    // with t==h==w for text rows, so decode stays correct.
    if (in.mrope_positions_h != nullptr && in.num_mrope_positions == N &&
        N > 0) {
        if (static_cast<int>(mrope_positions_d_.size()) < N * 3) {
            mrope_positions_d_ = DeviceBuffer<std::int32_t>::alloc(
                static_cast<std::size_t>(max_tokens_) * 3);
        }
        std::vector<std::int32_t> tmp(static_cast<std::size_t>(N) * 3);
        for (int i = 0; i < N * 3; ++i) {
            tmp[i] = static_cast<std::int32_t>(in.mrope_positions_h[i]);
        }
        mrope_positions_d_.copy_from_host(
            std::span<const std::int32_t>(tmp.data(), tmp.size()));
        vision.mrope_positions = mrope_positions_d_.data();
        vision_ptr = &vision;
    }

    if (has_image) {
        // DeepStack scratch: [num_deep, N, H] bf16, zeroed + filled by scatter.
        const std::size_t need =
            static_cast<std::size_t>(num_deepstack_) * max_tokens_ * H;
        if (num_deepstack_ > 0 && deepstack_scratch_.size() < need) {
            deepstack_scratch_ = DeviceBuffer<std::uint16_t>::alloc(need);
        }
        vision_in.weights             = &vision_raw_;
        vision_in.pixels_h            = in.image_pixels_h;
        vision_in.pixel_byte_indptr_h = in.image_pixel_byte_indptr_h;
        vision_in.grids_h             = in.image_grids_h;
        vision_in.anchor_rows_h       = in.image_anchor_rows_h;
        vision_in.num_images          = in.num_images;
        vision.vision_in              = &vision_in;
        vision.deepstack_scratch =
            num_deepstack_ > 0 ? deepstack_scratch_.data() : nullptr;
        vision.num_deepstack = num_deepstack_;
        vision_ptr = &vision;
    }
    (void)stream;

    // Image-only KV-fill fires sample nothing — skip the lm_head so the
    // forward never materializes dense logits over the (large) image span.
    LlamaLikeForwardCfg fwd = fwd_cfg_;
    fwd.emit_logits = in.emit_logits;

    llama_like_forward_paged(
        weights_, hf_config_, fwd, plan_,
        ws, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indptr_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.logit_row_indices_d, in.num_logit_rows,
        in.tp_greedy_argmax,
        in.custom_mask_d, in.custom_mask_indptr_d,
        vision_ptr);
}

std::uint32_t Qwen3VLModel::graph_layout() {
    return llama_like_decode_graph_layout(plan_);
}

}  // namespace pie_cuda_driver::model
