// The gemma-4 tower launchers' bodies: rebuild the C++ weights structs
// from the flat tables (layouts in the header) and hand off to the
// parity-anchored walks. Marshalling only.

#include "vision/gemma4_towers_c.hpp"

#include "vision/gemma4_audio.hpp"
#include "vision/gemma4_vision.hpp"

namespace pie_cuda_driver::kernels::vision {

namespace {

using bf = __nv_bfloat16;

const bf* as_bf(const void* p) { return static_cast<const bf*>(p); }

// Five consecutive slots: [w, imin, imax, omin, omax].
template <typename Clip>
Clip clip_of(const void* const* t) {
    Clip c;
    c.w = as_bf(t[0]);
    c.imin = as_bf(t[1]);
    c.imax = as_bf(t[2]);
    c.omin = as_bf(t[3]);
    c.omax = as_bf(t[4]);
    return c;
}

model::AudioFfnRaw ffn_of(const void* const* t) {
    model::AudioFfnRaw f;
    f.pre_ln = as_bf(t[0]);
    f.post_ln = as_bf(t[1]);
    f.fc1 = clip_of<model::AudioClipRaw>(t + 2);
    f.fc2 = clip_of<model::AudioClipRaw>(t + 7);
    return f;
}

}  // namespace

void gemma4_vision_encode(
    const void* patch_w, const void* pos_table, const void* embed_proj,
    const void* const* layer_w, int depth,
    int hidden, int heads, int intermediate, int pos_table_size,
    int text_hidden, int pool_kernel, float eps, float theta,
    const float* pixels_h, const std::uint32_t* pixel_byte_indptr_h,
    const std::uint32_t* patch_positions_h, const std::uint32_t* anchor_rows_h,
    int num_images,
    std::uint16_t* output_rows_h, std::size_t output_bytes,
    std::uint32_t* output_row_indptr_h,
    cudaStream_t stream) {
    model::VisRawWeights w;
    w.patch_w = as_bf(patch_w);
    w.pos_table = as_bf(pos_table);
    w.embed_proj = as_bf(embed_proj);
    w.layers.resize(static_cast<std::size_t>(depth));
    for (int i = 0; i < depth; ++i) {
        const void* const* t = layer_w + static_cast<std::size_t>(i) * 41;
        model::VisLayerRaw& L = w.layers[static_cast<std::size_t>(i)];
        L.in_ln = as_bf(t[0]);
        L.post_attn_ln = as_bf(t[1]);
        L.pre_ff_ln = as_bf(t[2]);
        L.post_ff_ln = as_bf(t[3]);
        L.q_norm = as_bf(t[4]);
        L.k_norm = as_bf(t[5]);
        L.q = clip_of<model::VisClipRaw>(t + 6);
        L.k = clip_of<model::VisClipRaw>(t + 11);
        L.v = clip_of<model::VisClipRaw>(t + 16);
        L.o = clip_of<model::VisClipRaw>(t + 21);
        L.gate = clip_of<model::VisClipRaw>(t + 26);
        L.up = clip_of<model::VisClipRaw>(t + 31);
        L.down = clip_of<model::VisClipRaw>(t + 36);
    }
    w.hidden = hidden;
    w.heads = heads;
    w.intermediate = intermediate;
    w.pos_table_size = pos_table_size;
    w.text_hidden = text_hidden;
    w.pool_kernel = pool_kernel;
    w.eps = eps;
    w.theta = theta;

    model::Gemma4VisionInputs in;
    in.weights = &w;
    in.pixels_h = pixels_h;
    in.pixel_byte_indptr_h = pixel_byte_indptr_h;
    in.patch_positions_h = patch_positions_h;
    in.anchor_rows_h = anchor_rows_h;
    in.num_images = num_images;
    model::encode_gemma4_vision(in, output_rows_h, output_bytes,
                                output_row_indptr_h, stream);
}

void gemma4_audio_encode(
    const void* sscp0_conv, const void* sscp0_norm, const void* sscp1_conv,
    const void* sscp1_norm, const void* sscp_input_proj,
    const void* output_proj_w, const void* output_proj_b,
    const void* embed_proj,
    const void* const* layer_w, int depth,
    int hidden, int heads, int conv_kernel, int n_mel, int sscp_ch0,
    int sscp_ch1, int out_proj_dims, int text_hidden, int chunk_size,
    int context_left, int context_right, float logit_cap,
    float residual_weight, float eps,
    const float* features_h, const std::uint32_t* feature_byte_indptr_h,
    const std::uint32_t* anchor_rows_h, int num_clips,
    std::uint16_t* output_rows_h, std::size_t output_bytes,
    std::uint32_t* output_row_indptr_h,
    cudaStream_t stream) {
    model::AudioRawWeights w;
    w.sscp0_conv = as_bf(sscp0_conv);
    w.sscp0_norm = as_bf(sscp0_norm);
    w.sscp1_conv = as_bf(sscp1_conv);
    w.sscp1_norm = as_bf(sscp1_norm);
    w.sscp_input_proj = as_bf(sscp_input_proj);
    w.output_proj_w = as_bf(output_proj_w);
    w.output_proj_b = as_bf(output_proj_b);
    w.embed_proj = as_bf(embed_proj);
    w.layers.resize(static_cast<std::size_t>(depth));
    for (int i = 0; i < depth; ++i) {
        const void* const* t = layer_w + static_cast<std::size_t>(i) * 62;
        model::AudioLayerRaw& L = w.layers[static_cast<std::size_t>(i)];
        L.ff1 = ffn_of(t);
        L.ff2 = ffn_of(t + 12);
        L.norm_pre_attn = as_bf(t[24]);
        L.norm_post_attn = as_bf(t[25]);
        L.q = clip_of<model::AudioClipRaw>(t + 26);
        L.k = clip_of<model::AudioClipRaw>(t + 31);
        L.v = clip_of<model::AudioClipRaw>(t + 36);
        L.post = clip_of<model::AudioClipRaw>(t + 41);
        L.relative_k = as_bf(t[46]);
        L.per_dim_scale = as_bf(t[47]);
        L.lconv_pre_ln = as_bf(t[48]);
        L.lconv_conv_norm = as_bf(t[49]);
        L.lconv_start = clip_of<model::AudioClipRaw>(t + 50);
        L.lconv_end = clip_of<model::AudioClipRaw>(t + 55);
        L.depthwise_conv = as_bf(t[60]);
        L.norm_out = as_bf(t[61]);
    }
    w.hidden = hidden;
    w.heads = heads;
    w.conv_kernel = conv_kernel;
    w.n_mel = n_mel;
    w.sscp_ch0 = sscp_ch0;
    w.sscp_ch1 = sscp_ch1;
    w.out_proj_dims = out_proj_dims;
    w.text_hidden = text_hidden;
    w.chunk_size = chunk_size;
    w.context_left = context_left;
    w.context_right = context_right;
    w.logit_cap = logit_cap;
    w.residual_weight = residual_weight;
    w.eps = eps;

    model::Gemma4AudioInputs in;
    in.weights = &w;
    in.features_h = features_h;
    in.feature_byte_indptr_h = feature_byte_indptr_h;
    in.anchor_rows_h = anchor_rows_h;
    in.n_mel = n_mel;
    in.num_clips = num_clips;
    model::encode_gemma4_audio(in, output_rows_h, output_bytes,
                               output_row_indptr_h, stream);
}

}  // namespace pie_cuda_driver::kernels::vision
