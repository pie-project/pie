// Host bridge: Gemma4AudioWeights (bound DeviceTensors) → AudioRawWeights.
// See gemma4_audio_adapter.hpp. Compiled by the host C++ compiler (g++), so it
// may include the toml++-heavy model headers. Mirrors gemma4_vision_adapter.cpp.

#include "model/gemma4/gemma4_audio_adapter.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

namespace {
using bf = __nv_bfloat16;
const bf* P(const DeviceTensor* t) {
    return t ? static_cast<const bf*>(t->data()) : nullptr;
}
AudioClipRaw to_clip(const Gemma4AudioClippedLinear& c) {
    AudioClipRaw r;
    r.w = P(c.weight);
    r.imin = P(c.input_min);
    r.imax = P(c.input_max);
    r.omin = P(c.output_min);
    r.omax = P(c.output_max);
    return r;
}
AudioFfnRaw to_ffn(const Gemma4AudioFfnWeights& f) {
    AudioFfnRaw r;
    r.pre_ln = P(f.pre_layer_norm);
    r.post_ln = P(f.post_layer_norm);
    r.fc1 = to_clip(f.ffw_layer_1);
    r.fc2 = to_clip(f.ffw_layer_2);
    return r;
}
}  // namespace

AudioRawWeights to_audio_raw(const Gemma4AudioWeights& w) {
    AudioRawWeights r;

    r.sscp0_conv = P(w.sscp_layer0_conv);
    r.sscp0_norm = P(w.sscp_layer0_norm);
    r.sscp1_conv = P(w.sscp_layer1_conv);
    r.sscp1_norm = P(w.sscp_layer1_norm);
    r.sscp_input_proj = P(w.sscp_input_proj);

    r.output_proj_w = P(w.output_proj_weight);
    r.output_proj_b = P(w.output_proj_bias);
    r.embed_proj = P(w.embed_audio_projection);

    const auto& c = w.config;
    r.hidden = c.hidden_size;
    r.heads = c.num_attention_heads;
    r.conv_kernel = c.conv_kernel_size;
    r.n_mel = c.feature_size;
    r.sscp_ch0 = c.subsampling_conv_channels0;
    r.sscp_ch1 = c.subsampling_conv_channels1;
    r.out_proj_dims = c.output_proj_dims;
    r.chunk_size = c.attention_chunk_size;
    r.context_left = c.attention_context_left;
    r.context_right = c.attention_context_right;
    r.logit_cap = c.attention_logit_cap;
    r.residual_weight = c.residual_weight;
    r.eps = c.rms_norm_eps;
    // text_hidden derived from the embed projection's row count when present.
    if (w.embed_audio_projection) {
        const auto& s = w.embed_audio_projection->shape();
        if (!s.empty()) r.text_hidden = static_cast<int>(s[0]);
    }

    r.layers.reserve(w.layers.size());
    for (const auto& L : w.layers) {
        AudioLayerRaw o;
        o.ff1 = to_ffn(L.feed_forward1);
        o.ff2 = to_ffn(L.feed_forward2);
        o.norm_pre_attn = P(L.norm_pre_attn);
        o.norm_post_attn = P(L.norm_post_attn);
        o.q = to_clip(L.q_proj);
        o.k = to_clip(L.k_proj);
        o.v = to_clip(L.v_proj);
        o.post = to_clip(L.post);
        o.relative_k = P(L.relative_k_proj);
        o.per_dim_scale = P(L.per_dim_scale);
        o.lconv_pre_ln = P(L.lconv_pre_layer_norm);
        o.lconv_conv_norm = P(L.lconv_conv_norm);
        o.lconv_start = to_clip(L.lconv_linear_start);
        o.lconv_end = to_clip(L.lconv_linear_end);
        o.depthwise_conv = P(L.lconv_depthwise_conv);
        o.norm_out = P(L.norm_out);
        r.layers.push_back(o);
    }
    return r;
}

void run_gemma4_audio(const Gemma4AudioWeights& w,
                      const float* features,
                      int n_frames,
                      int n_mel,
                      int out_len,
                      __nv_bfloat16* out_proj,
                      cudaStream_t stream) {
    run_gemma4_audio(to_audio_raw(w), features, n_frames, n_mel, out_len, out_proj, stream);
}

}  // namespace pie_cuda_driver::model
