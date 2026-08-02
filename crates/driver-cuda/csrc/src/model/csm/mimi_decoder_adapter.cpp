// Host bridge: MimiDecoderWeights (bound DeviceTensors) → MimiDecoderRawWeights.
// See mimi_decoder_adapter.hpp. Compiled by the host C++ compiler (g++), so it
// may include the toml++-heavy model headers. Mirrors gemma4_audio_adapter.cpp.

#include "model/csm/mimi_decoder_adapter.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

namespace {
using bf = __nv_bfloat16;
const bf* P(const DeviceTensor* t) {
    return t ? static_cast<const bf*>(t->data()) : nullptr;
}

// Build a MimiConvRaw from a bound Conv1d-style weight. `in/out/kernel` are read
// from the weight shape: Conv1d weight is [out, in, k]; ConvTranspose1d weight
// is [in, out/groups, k] — handled separately below.
MimiConvRaw to_conv(const MimiConvWeights& c) {
    MimiConvRaw r;
    r.w = P(c.weight);
    r.b = P(c.bias);
    r.stride = c.stride;
    r.dilation = c.dilation;
    if (c.weight) {
        const auto& s = c.weight->shape();  // [out, in, k]
        if (s.size() == 3) {
            r.out_ch = static_cast<int>(s[0]);
            r.in_ch = static_cast<int>(s[1]);
            r.kernel = static_cast<int>(s[2]);
        }
    }
    return r;
}

// ConvTranspose1d weight layout is [in_ch, out_ch/groups, k].
MimiConvTRaw to_convt(const MimiConvWeights& c) {
    MimiConvTRaw r;
    r.w = P(c.weight);
    r.b = P(c.bias);
    r.stride = c.stride;
    r.groups = c.groups;
    if (c.weight) {
        const auto& s = c.weight->shape();  // [in, out/groups, k]
        if (s.size() == 3) {
            r.in_ch = static_cast<int>(s[0]);
            r.out_ch = static_cast<int>(s[1]) * c.groups;
            r.kernel = static_cast<int>(s[2]);
        }
    }
    return r;
}
}  // namespace

MimiDecoderRawWeights to_mimi_decoder_raw(const MimiDecoderWeights& w) {
    MimiDecoderRawWeights r;
    const auto& c = w.config;

    r.hidden = c.hidden_size;
    r.codebook_dim = c.codebook_dim;
    r.codebook_size = c.codebook_size;
    r.num_codebooks = c.num_quantizers;
    r.num_semantic = c.num_semantic_quantizers;
    r.num_filters = c.num_filters;
    r.upsampling_ratios = c.upsampling_ratios;
    r.xf_heads = c.xf_num_attention_heads;
    r.xf_kv_heads = c.xf_num_key_value_heads;
    r.xf_head_dim = c.xf_head_dim;
    r.xf_intermediate = c.xf_intermediate_size;
    r.xf_sliding_window = c.xf_sliding_window;
    r.xf_rope_theta = c.xf_rope_theta;
    r.norm_eps = c.norm_eps;
    r.sampling_rate = c.sampling_rate;
    r.causal = c.use_causal_conv;

    // RVQ codebook embeds (resolved at load) + per-group output projections.
    r.codebook_embed.reserve(w.codebook_embed.size());
    for (const auto* t : w.codebook_embed) r.codebook_embed.push_back(P(t));
    if (static_cast<int>(r.codebook_embed.size()) != r.num_codebooks) {
        throw std::runtime_error("mimi_decoder: codebook_embed count != num_codebooks");
    }
    r.semantic_output_proj = P(w.semantic_output_proj);
    r.acoustic_output_proj = P(w.acoustic_output_proj);

    // upsample ConvTranspose1d (groups=512, no bias). Stride/groups come from
    // config since the weight shape can't disambiguate groups.
    r.upsample = to_convt(w.upsample);
    if (r.upsample.kernel == 0) {  // weight present but shape missing → fall back
        r.upsample.kernel = c.upsample_kernel;
    }
    r.upsample.stride = c.upsample_stride;
    r.upsample.groups = c.upsample_groups;
    r.upsample.in_ch = c.hidden_size;
    r.upsample.out_ch = c.hidden_size;

    // decoder_transformer layers.
    r.xf_layers.reserve(w.xf_layers.size());
    for (const auto& L : w.xf_layers) {
        MimiXfLayerRaw o;
        o.in_ln_w = P(L.input_layernorm_weight);
        o.in_ln_b = P(L.input_layernorm_bias);
        o.q = P(L.q_proj);
        o.k = P(L.k_proj);
        o.v = P(L.v_proj);
        o.o = P(L.o_proj);
        o.attn_scale = P(L.self_attn_layer_scale);
        o.post_ln_w = P(L.post_attention_layernorm_weight);
        o.post_ln_b = P(L.post_attention_layernorm_bias);
        o.fc1 = P(L.mlp_fc1);
        o.fc2 = P(L.mlp_fc2);
        o.mlp_scale = P(L.mlp_layer_scale);
        r.xf_layers.push_back(o);
    }
    r.xf_final_ln_w = P(w.xf_final_ln_weight);
    r.xf_final_ln_b = P(w.xf_final_ln_bias);

    // SEANet decoder.
    r.seanet_in = to_conv(w.seanet_in);
    r.seanet_stages.reserve(w.seanet_stages.size());
    for (size_t i = 0; i < w.seanet_stages.size(); ++i) {
        const auto& st = w.seanet_stages[i];
        MimiDecoderStageRaw o;
        o.convtr = to_convt(st.convtr);
        o.convtr.stride = c.upsampling_ratios[i];        // stride = ratio
        o.convtr.groups = 1;
        o.resnet.conv1 = to_conv(st.resnet.conv1);
        o.resnet.conv2 = to_conv(st.resnet.conv2);
        r.seanet_stages.push_back(o);
    }
    r.seanet_out = to_conv(w.seanet_out);

    return r;
}

int run_mimi_decoder(const MimiDecoderWeights& w,
                     const std::int32_t* codes,
                     int n_frames,
                     float* out_wave,
                     cudaStream_t stream) {
    return run_mimi_decoder(to_mimi_decoder_raw(w), codes, n_frames, out_wave, stream);
}

}  // namespace pie_cuda_driver::model
