#include "model/gemma4/gemma4_model.hpp"

#include <cstdlib>
#include <utility>
#include <vector>

#include "model/gemma4/gemma4_vision_adapter.hpp"  // to_vis_raw
#include "model/gemma4/gemma4_audio_adapter.hpp"   // to_audio_raw

namespace pie_cuda_driver::model {

Gemma4Model::Gemma4Model(
    Gemma4Weights weights,
    const HfConfig& hf_config,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& kv_cache,
    const Gemma4ForwardCfg& fwd_cfg,
    int small_spec_graph_tokens,
    std::optional<Gemma4VisionWeights> vision,
    std::optional<Gemma4AudioWeights> audio)
    : weights_(std::move(weights)),
      hf_config_(hf_config),
      moe_ws_(moe_ws),
      kv_cache_(kv_cache),
      fwd_cfg_(fwd_cfg)
{
    if (vision.has_value()) {
        vision_raw_ = to_vis_raw(*vision);
        has_vision_ = true;
    }
    if (audio.has_value()) {
        audio_raw_ = to_audio_raw(*audio);
        has_audio_ = true;
    }
    caps_.supports_media_encode = has_vision_ || has_audio_;
    // The MoE branch reads its routing table back to the host and loops the
    // experts on the CPU for prefill, which cannot run inside a graph capture
    // (`cudaErrorStreamCaptureUnsupported`). Pure decode -- the only shape the
    // executor captures -- takes the on-device GEMV dispatch instead, so
    // capture stays available; `gemma4_moe_block` is handed `is_pure_decode`
    // and treats it as a hard requirement rather than a hint, which is what
    // makes that claim true instead of hopeful.
    const bool kv_capture_ok = kv_cache_.format().is_native_bf16();
    caps_.graph_safe = kv_capture_ok;
    caps_.graph_padding_kv_write_safe = true;
    caps_.supports_compact_logits = true;
    // Small-prefill capture would put the host-dispatched branch inside a
    // capture, so it stays off for the MoE variants.
    caps_.supports_small_prefill_graph =
        kv_capture_ok && !hf_config_.gemma4_enable_moe &&
        small_spec_graph_tokens > 0;

    // Trace and VALIDATE the declaration at load: a drift against the
    // launcher registry, or a weight this checkpoint does not bind,
    // becomes a load-time DECLINE rather than a wrong number. The
    // drive is default-on now, so the decline has to be graceful.
    if (gemma4_declared_forward_enabled()) {
        declared_ = build_gemma4_declared_plan(hf_config_, weights_,
                                               fwd_cfg_.tp_size);
    }
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

void Gemma4Model::body(Workspace& ws,
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
    // A ragged fire whose every request is a SHORT block goes to the
    // hand-written pass's row-decode path — speculative verification,
    // which the declaration does not state. One request longer than the
    // qmax is what makes `prepare_row_decode_kv_table` refuse, so it is
    // what makes the plain prefill class the truthful reading of this
    // fire. Any other reason that prepare refuses leaves us declining a
    // fire the hand pass would have prefilled: a fallback, never a wrong
    // number.
    const bool row_decode_shaped = [&] {
        if (in.is_pure_decode) return false;
        if (in.qo_indptr_h == nullptr) return false;
        const int qmax = gemma4_row_decode_qmax();
        for (int r = 0; r < in.num_requests; ++r) {
            const std::uint32_t len = in.qo_indptr_h[r + 1] - in.qo_indptr_h[r];
            if (len > static_cast<std::uint32_t>(qmax)) return false;
        }
        return true;
    }();

    // The declared drive gets the fire first. It answers false for
    // anything outside what the two classes state — a masked or hooked
    // fire, a multimodal one, a row-decode-shaped one, a deployment
    // whose PLE buffers or cache format do not match — and the
    // hand-written pass runs it unchanged. Eligibility is an ANSWER, not
    // an error.
    const bool declared_eligible =
        gemma4_declared_drive_enabled() && declared_.usable &&
        !row_decode_shaped &&
        in.custom_mask_d == nullptr && in.stage_hooks == nullptr &&
        in.num_images == 0 && in.num_clips == 0 &&
        in.precomputed_embeddings.num_blocks == 0 &&
        fwd_cfg_.tp_size == 1;
    if (declared_eligible &&
        gemma4_forward_declared(
            declared_, weights_, hf_config_, fwd_cfg_, ws, moe_ws_, kv,
            attn_ws, cublas, in.token_ids, in.positions, in.qo_indptr_d,
            in.kv_page_indices_d, in.kv_page_indptr_d, in.kv_last_page_lens_d,
            in.qo_indptr_h, in.kv_page_indptr_h,
            in.total_tokens, in.num_requests, in.is_pure_decode,
            in.row_valid_d, in.logit_row_indices_d, in.num_logit_rows)) {
        return;
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
        in.row_valid_d,
        in.custom_mask_d, in.custom_mask_indptr_d,
        in.logit_row_indices_d, in.num_logit_rows, vision_in_ptr, audio_in_ptr,
        in.precomputed_embeddings.num_blocks > 0
            ? &in.precomputed_embeddings
            : nullptr,
        in.stage_hooks);
}

std::uint32_t Gemma4Model::graph_layout() {
    return gemma4_decode_graph_layout(moe_ws_);
}

bool Gemma4Model::encode_media(const MediaEncodeInputs& in, cudaStream_t stream) {
    if ((in.num_images > 0 && !has_vision_) ||
        (in.num_clips > 0 && !has_audio_) ||
        in.num_images + in.num_clips == 0) {
        return false;
    }
    const int hidden = hf_config_.hidden_size;
    std::size_t row_offset = 0;
    in.output_row_indptr_h[0] = 0;
    if (in.num_images > 0) {
        Gemma4VisionInputs vision_in;
        vision_in.weights = &vision_raw_;
        vision_in.pixels_h = in.image_pixels_h;
        vision_in.pixel_byte_indptr_h = in.image_pixel_byte_indptr_h;
        vision_in.patch_positions_h = in.image_patch_positions_h;
        vision_in.anchor_rows_h = in.image_anchor_rows_h;
        vision_in.num_images = in.num_images;
        std::vector<std::uint32_t> boundaries(in.num_images + 1);
        encode_gemma4_vision(
            vision_in, in.output_rows_h, in.output_bytes,
            boundaries.data(), stream);
        row_offset = boundaries.back();
        for (int image = 0; image < in.num_images; ++image) {
            in.output_row_indptr_h[image + 1] = boundaries[image + 1];
        }
    }
    if (in.num_clips > 0) {
        Gemma4AudioInputs audio_in;
        audio_in.weights = &audio_raw_;
        audio_in.features_h = in.audio_features_h;
        audio_in.feature_byte_indptr_h =
            in.audio_feature_byte_indptr_h;
        audio_in.anchor_rows_h = in.audio_anchor_rows_h;
        audio_in.n_mel = audio_raw_.n_mel;
        audio_in.num_clips = in.num_clips;
        const std::size_t consumed =
            row_offset * hidden * sizeof(std::uint16_t);
        std::vector<std::uint32_t> boundaries(in.num_clips + 1);
        encode_gemma4_audio(
            audio_in, in.output_rows_h + row_offset * hidden,
            in.output_bytes - consumed, boundaries.data(), stream);
        for (int clip = 0; clip < in.num_clips; ++clip) {
            in.output_row_indptr_h[in.num_images + clip + 1] =
                static_cast<std::uint32_t>(row_offset) +
                boundaries[clip + 1];
        }
    }
    return true;
}

}  // namespace pie_cuda_driver::model
