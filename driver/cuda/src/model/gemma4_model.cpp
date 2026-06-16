#include "model/gemma4_model.hpp"

#include <cstdlib>

#include "model/gemma4_vision_adapter.hpp"  // to_vis_raw
#include "model/gemma4_audio_adapter.hpp"   // to_audio_raw

namespace pie_cuda_driver::model {

Gemma4Model::Gemma4Model(
    const Gemma4Weights& weights,
    const HfConfig& hf_config,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& kv_cache,
    const Gemma4ForwardCfg& fwd_cfg,
    int small_spec_graph_tokens,
    const Gemma4VisionWeights* vision,
    const Gemma4AudioWeights* audio)
    : weights_(weights),
      hf_config_(hf_config),
      moe_ws_(moe_ws),
      kv_cache_(kv_cache),
      fwd_cfg_(fwd_cfg)
{
    if (vision != nullptr) {
        vision_raw_ = to_vis_raw(*vision);
        has_vision_ = true;
    }
    if (audio != nullptr) {
        audio_raw_ = to_audio_raw(*audio);
        has_audio_ = true;
    }
    // CUDA graphs default ON for Gemma4 unless intrusive-profile env is set.
    const char* profile_env = std::getenv("PIE_GEMMA4_FORWARD_PROFILE");
    const bool profile_enabled =
        profile_env != nullptr && profile_env[0] != '\0' && profile_env[0] != '0';
    caps_.graph_safe = kv_cache_.format().is_native_bf16() && !profile_enabled;
    caps_.supports_compact_logits = true;
    caps_.supports_small_prefill_graph =
        kv_cache_.format().is_native_bf16() && small_spec_graph_tokens > 0;

    const char* fused_env = std::getenv("PIE_FUSED_LMHEAD_ARGMAX");
    caps_.supports_fused_lmhead_argmax =
        fused_env != nullptr && fused_env[0] != '\0' && fused_env[0] != '0';
}

void Gemma4Model::prepare(AttentionWorkspace& attn_ws,
                          const ForwardFn::PrepareInputs& in) {
    prepare_gemma4_decode_plans(
        weights_, hf_config_, fwd_cfg_,
        moe_ws_, kv_cache_, attn_ws,
        in.qo_indptr_h,
        in.kv_page_indices_h,
        in.kv_page_indptr_h,
        in.kv_last_page_lens_h,
        in.total_tokens,
        in.num_requests,
        in.is_pure_decode);
}

void Gemma4Model::body(Qwen3Workspace& ws,
                       KvCache& kv,
                       AttentionWorkspace& attn_ws,
                       ops::CublasHandle& cublas,
                       const ForwardFn::ForwardInputs& in) {
    // Multimodal: assemble the per-fire vision inputs (no-op when text-only or
    // no images in this fire).
    Gemma4VisionInputs vision_in;
    const Gemma4VisionInputs* vision_in_ptr = nullptr;
    if (has_vision_ && in.num_images > 0) {
        vision_in.weights             = &vision_raw_;
        vision_in.pixels_h            = in.image_pixels_h;
        vision_in.pixel_byte_indptr_h = in.image_pixel_byte_indptr_h;
        vision_in.patch_positions_h   = in.image_patch_positions_h;
        vision_in.anchor_rows_h       = in.image_anchor_rows_h;
        vision_in.num_images          = in.num_images;
        vision_in_ptr = &vision_in;
    }
    // Multimodal: assemble the per-fire audio inputs (no-op when no audio in
    // this fire). Direct analog of vision — log-mel features per clip.
    Gemma4AudioInputs audio_in;
    const Gemma4AudioInputs* audio_in_ptr = nullptr;
    if (has_audio_ && in.num_clips > 0) {
        audio_in.weights               = &audio_raw_;
        audio_in.features_h            = in.audio_features_h;
        audio_in.feature_byte_indptr_h = in.audio_feature_byte_indptr_h;
        audio_in.anchor_rows_h         = in.audio_anchor_rows_h;
        audio_in.n_mel                 = audio_raw_.n_mel;
        audio_in.num_clips             = in.num_clips;
        audio_in_ptr = &audio_in;
    }
    gemma4_forward_paged(
        weights_, hf_config_, fwd_cfg_,
        ws, moe_ws_, kv, attn_ws, cublas,
        in.token_ids, in.positions,
        in.qo_indptr_d, in.kv_page_indices_d, in.kv_page_indptr_d,
        in.kv_last_page_lens_d,
        in.qo_indptr_h, in.kv_page_indices_h, in.kv_page_indptr_h,
        in.kv_last_page_lens_h,
        in.total_tokens, in.num_requests, in.is_pure_decode,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.logit_row_indices_d, in.num_logit_rows, vision_in_ptr, audio_in_ptr);
}

std::uint32_t Gemma4Model::graph_layout() {
    return gemma4_decode_graph_layout(moe_ws_);
}

void Gemma4Model::set_logits_argmax_only(bool enabled) {
    pie_cuda_driver::model::set_gemma4_logits_argmax_only(enabled);
}

void Gemma4Model::set_fused_argmax_output(std::int32_t* ptr) {
    pie_cuda_driver::model::set_gemma4_fused_argmax_output(ptr);
}

bool Gemma4Model::fused_argmax_done() {
    return pie_cuda_driver::model::gemma4_fused_argmax_done();
}

}  // namespace pie_cuda_driver::model
