// CSM-1B bind + Raw-weight adapters. See csm.hpp / AUDIO_OUTPUT.md.
// Host C++ (g++) — may include the toml++-heavy model/loader headers.

#include "model/csm/csm.hpp"

#include <stdexcept>
#include <string>

#include "model/csm/csm_backbone_forward.hpp"  // csm_cast_f32_to_bf16
#include "model/loaded_model.hpp"

namespace pie_cuda_driver::model {

namespace {
using bf = __nv_bfloat16;

const bf* P(const DeviceTensor* t) {
    return t ? static_cast<const bf*>(t->data()) : nullptr;
}
}  // namespace

// Resolve a tensor by name and ensure it is bf16. eustlb/csm-1b ships
// torch_dtype float32 and the default loader ABI preserves the on-disk dtype,
// so the bound DeviceTensors are F32. The CSM kernels are bf16-store/fp32-
// compute, so any F32 tensor is cast to an owned bf16 copy once here (kept in
// `bf16_owned` so the bf16 storage outlives the model). Already-bf16 tensors
// pass through (the returned pointer aliases the loader-owned storage).
const DeviceTensor* CsmWeights::bf16_tensor(LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("bind_csm: missing tensor '" + name + "'");
    }
    const DeviceTensor& t = e.get(name);
    if (t.dtype() == DType::BF16) return &t;
    if (t.dtype() != DType::FP32) {
        throw std::runtime_error("bind_csm: tensor '" + name +
                                 "' has unsupported dtype (expected FP32 or BF16)");
    }
    auto out = std::make_unique<DeviceTensor>(
        DeviceTensor::allocate(DType::BF16, t.shape()));
    csm_cast_f32_to_bf16(static_cast<const float*>(t.data()),
                         static_cast<bf*>(out->data()),
                         static_cast<long>(t.numel()));
    const DeviceTensor* ptr = out.get();
    bf16_owned.push_back(std::move(out));
    return ptr;
}

CsmBackboneRawWeights CsmWeights::backbone_raw() const {
    CsmBackboneRawWeights r;
    r.embed_text = P(embed_text);
    r.embed_audio = P(embed_audio);
    r.norm_w = P(norm);
    r.lm_head = P(lm_head);
    r.layers.reserve(backbone_layers.size());
    for (const auto& L : backbone_layers) {
        CsmBackboneLayerRaw o;
        o.in_ln_w = P(L.in_ln); o.post_ln_w = P(L.post_ln);
        o.q = P(L.q); o.k = P(L.k); o.v = P(L.v); o.o = P(L.o);
        o.gate = P(L.gate); o.up = P(L.up); o.down = P(L.down);
        r.layers.push_back(o);
    }
    r.hidden = config.hidden_size;
    r.num_layers = config.num_hidden_layers;
    r.num_heads = config.num_attention_heads;
    r.num_kv_heads = config.num_key_value_heads;
    r.head_dim = config.head_dim;
    r.intermediate = config.intermediate_size;
    r.norm_eps = config.rms_norm_eps;
    r.rope_theta = config.rope_theta;
    r.rope_factor = config.rope_factor;
    r.rope_low_freq_factor = config.rope_low_freq_factor;
    r.rope_high_freq_factor = config.rope_high_freq_factor;
    r.rope_original_max_position = config.rope_original_max_position;
    if (config.csm.has_value()) {
        const auto& c = *config.csm;
        r.text_vocab = c.text_vocab_size;
        r.audio_vocab = c.audio_vocab_size;
        r.num_codebooks = c.num_codebooks;
        r.codebook_eos_token_id = c.codebook_eos_token_id;
    }
    return r;
}

CsmDepthRawWeights CsmWeights::depth_raw() const {
    CsmDepthRawWeights r;
    r.embed_tokens = P(depth_embed_tokens);
    r.inputs_embeds_projector = P(depth_inputs_proj);
    r.norm_w = P(depth_norm);
    r.codebooks_head = P(depth_codebooks_head);
    r.layers.reserve(depth_layers.size());
    for (const auto& L : depth_layers) {
        CsmDepthLayerRaw o;
        o.in_ln_w = P(L.in_ln); o.post_ln_w = P(L.post_ln);
        o.q = P(L.q); o.k = P(L.k); o.v = P(L.v); o.o = P(L.o);
        o.gate = P(L.gate); o.up = P(L.up); o.down = P(L.down);
        r.layers.push_back(o);
    }
    if (config.csm.has_value()) {
        const auto& d = config.csm->depth;
        r.hidden = d.hidden_size;
        r.backbone_hidden = d.backbone_hidden_size;
        r.num_layers = d.num_hidden_layers;
        r.num_heads = d.num_attention_heads;
        r.num_kv_heads = d.num_key_value_heads;
        r.head_dim = d.head_dim;
        r.intermediate = d.intermediate_size;
        r.num_codebooks = d.num_codebooks;
        r.vocab_size = d.vocab_size;
        r.norm_eps = d.rms_norm_eps;
        r.rope_theta = d.rope_theta;
        r.rope_factor = d.rope_factor;
        r.rope_low_freq_factor = d.rope_low_freq_factor;
        r.rope_high_freq_factor = d.rope_high_freq_factor;
        r.rope_original_max_position = d.rope_original_max_position;
    }
    return r;
}

MimiDecoderRawWeights CsmWeights::mimi_raw() const {
    return to_mimi_decoder_raw(mimi);
}

// ── bind ────────────────────────────────────────────────────────────────────
CsmWeights bind_csm(LoadedModel& engine, bool verbose) {
    CsmWeights w;
    w.config = engine.hf_config();
    if (!w.config.csm.has_value()) {
        throw std::runtime_error("bind_csm: hf_config.csm not populated (model_type != csm?)");
    }
    const auto& c = *w.config.csm;

    // ── Backbone ──────────────────────────────────────────────────────────
    w.embed_text  = w.bf16_tensor(engine, "embed_text_tokens.weight");
    w.embed_audio = w.bf16_tensor(engine, "backbone_model.embed_tokens.embed_audio_tokens.weight");
    w.norm        = w.bf16_tensor(engine, "backbone_model.norm.weight");
    w.lm_head     = w.bf16_tensor(engine, "lm_head.weight");
    const int BBL = w.config.num_hidden_layers;
    w.backbone_layers.resize(BBL);
    for (int i = 0; i < BBL; ++i) {
        const std::string p = "backbone_model.layers." + std::to_string(i) + ".";
        auto& L = w.backbone_layers[i];
        L.in_ln   = w.bf16_tensor(engine, p + "input_layernorm.weight");
        L.post_ln = w.bf16_tensor(engine, p + "post_attention_layernorm.weight");
        L.q = w.bf16_tensor(engine, p + "self_attn.q_proj.weight");
        L.k = w.bf16_tensor(engine, p + "self_attn.k_proj.weight");
        L.v = w.bf16_tensor(engine, p + "self_attn.v_proj.weight");
        L.o = w.bf16_tensor(engine, p + "self_attn.o_proj.weight");
        L.gate = w.bf16_tensor(engine, p + "mlp.gate_proj.weight");
        L.up   = w.bf16_tensor(engine, p + "mlp.up_proj.weight");
        L.down = w.bf16_tensor(engine, p + "mlp.down_proj.weight");
    }

    // ── Depth decoder ─────────────────────────────────────────────────────
    w.depth_embed_tokens   = w.bf16_tensor(engine, "depth_decoder.model.embed_tokens.weight");
    w.depth_inputs_proj    = w.bf16_tensor(engine, "depth_decoder.model.inputs_embeds_projector.weight");
    w.depth_norm           = w.bf16_tensor(engine, "depth_decoder.model.norm.weight");
    w.depth_codebooks_head = w.bf16_tensor(engine, "depth_decoder.codebooks_head.weight");
    const int DDL = c.depth.num_hidden_layers;
    w.depth_layers.resize(DDL);
    for (int i = 0; i < DDL; ++i) {
        const std::string p = "depth_decoder.model.layers." + std::to_string(i) + ".";
        auto& L = w.depth_layers[i];
        L.in_ln   = w.bf16_tensor(engine, p + "input_layernorm.weight");
        L.post_ln = w.bf16_tensor(engine, p + "post_attention_layernorm.weight");
        L.q = w.bf16_tensor(engine, p + "self_attn.q_proj.weight");
        L.k = w.bf16_tensor(engine, p + "self_attn.k_proj.weight");
        L.v = w.bf16_tensor(engine, p + "self_attn.v_proj.weight");
        L.o = w.bf16_tensor(engine, p + "self_attn.o_proj.weight");
        L.gate = w.bf16_tensor(engine, p + "mlp.gate_proj.weight");
        L.up   = w.bf16_tensor(engine, p + "mlp.up_proj.weight");
        L.down = w.bf16_tensor(engine, p + "mlp.down_proj.weight");
    }

    // ── Mimi decoder ──────────────────────────────────────────────────────
    auto& mc = w.mimi.config;
    mc.hidden_size = c.codec.hidden_size;
    mc.codebook_dim = c.codec.codebook_dim;
    mc.codebook_size = c.codec.codebook_size;
    mc.num_quantizers = c.codec.num_quantizers;
    mc.num_semantic_quantizers = c.codec.num_semantic_quantizers;
    mc.num_filters = c.codec.num_filters;
    mc.upsampling_ratios = c.codec.upsampling_ratios;
    mc.xf_num_attention_heads = c.codec.xf_num_attention_heads;
    mc.xf_num_key_value_heads = c.codec.xf_num_key_value_heads;
    mc.xf_head_dim = c.codec.xf_head_dim;
    mc.xf_intermediate_size = c.codec.xf_intermediate_size;
    mc.xf_sliding_window = c.codec.xf_sliding_window;
    mc.xf_rope_theta = c.codec.xf_rope_theta;
    mc.norm_eps = c.codec.norm_eps;
    mc.sampling_rate = c.codec.sampling_rate;
    mc.use_causal_conv = c.codec.use_causal_conv;
    mc.upsample_groups = c.codec.upsample_groups;
    mc.upsample_kernel = 4;
    mc.upsample_stride = 2;

    // RVQ codebook embeds: resolve embed_sum / cluster_usage.clamp(eps) into
    // owned DeviceTensors [codebook_size, codebook_dim].
    const int NCB = c.codec.num_quantizers;
    const int dim = c.codec.codebook_dim, rows = c.codec.codebook_size;
    const float eps = 1e-5f;
    w.mimi.codebook_embed.resize(NCB);
    auto resolve = [&](const std::string& base) -> const DeviceTensor* {
        const DeviceTensor* es = w.bf16_tensor(engine, base + ".embed_sum");
        const DeviceTensor* cu = w.bf16_tensor(engine, base + ".cluster_usage");
        auto out = std::make_unique<DeviceTensor>(
            DeviceTensor::allocate(DType::BF16,
                                   {static_cast<std::int64_t>(rows),
                                    static_cast<std::int64_t>(dim)}));
        mimi_resolve_codebook_embed(P(es), P(cu), static_cast<bf*>(out->data()),
                                    rows, dim, eps);
        const DeviceTensor* ptr = out.get();
        w.mimi_owned.push_back(std::move(out));
        return ptr;
    };
    w.mimi.codebook_embed[0] = resolve(
        "codec_model.quantizer.semantic_residual_vector_quantizer.layers.0.codebook");
    for (int i = 1; i < NCB; ++i) {
        w.mimi.codebook_embed[i] = resolve(
            "codec_model.quantizer.acoustic_residual_vector_quantizer.layers." +
            std::to_string(i - 1) + ".codebook");
    }
    w.mimi.semantic_output_proj =
        w.bf16_tensor(engine, "codec_model.quantizer.semantic_residual_vector_quantizer.output_proj.weight");
    w.mimi.acoustic_output_proj =
        w.bf16_tensor(engine, "codec_model.quantizer.acoustic_residual_vector_quantizer.output_proj.weight");

    // upsample ConvTranspose1d (groups=512, no bias).
    w.mimi.upsample.weight = w.bf16_tensor(engine, "codec_model.upsample.conv.weight");
    w.mimi.upsample.bias = nullptr;
    w.mimi.upsample.stride = 2;
    w.mimi.upsample.groups = c.codec.upsample_groups;

    // decoder_transformer: 8 layers.
    const int XFL = c.codec.xf_num_hidden_layers;
    w.mimi.xf_layers.resize(XFL);
    for (int l = 0; l < XFL; ++l) {
        const std::string p = "codec_model.decoder_transformer.layers." + std::to_string(l) + ".";
        auto& L = w.mimi.xf_layers[l];
        L.input_layernorm_weight = w.bf16_tensor(engine, p + "input_layernorm.weight");
        L.input_layernorm_bias   = w.bf16_tensor(engine, p + "input_layernorm.bias");
        L.q_proj = w.bf16_tensor(engine, p + "self_attn.q_proj.weight");
        L.k_proj = w.bf16_tensor(engine, p + "self_attn.k_proj.weight");
        L.v_proj = w.bf16_tensor(engine, p + "self_attn.v_proj.weight");
        L.o_proj = w.bf16_tensor(engine, p + "self_attn.o_proj.weight");
        L.self_attn_layer_scale = w.bf16_tensor(engine, p + "self_attn_layer_scale.scale");
        L.post_attention_layernorm_weight = w.bf16_tensor(engine, p + "post_attention_layernorm.weight");
        L.post_attention_layernorm_bias   = w.bf16_tensor(engine, p + "post_attention_layernorm.bias");
        L.mlp_fc1 = w.bf16_tensor(engine, p + "mlp.fc1.weight");
        L.mlp_fc2 = w.bf16_tensor(engine, p + "mlp.fc2.weight");
        L.mlp_layer_scale = w.bf16_tensor(engine, p + "mlp_layer_scale.scale");
    }
    w.mimi.xf_final_ln_weight = nullptr;  // MimiTransformerModel has no final norm
    w.mimi.xf_final_ln_bias = nullptr;

    // SEANet decoder. layer indices in codec_model.decoder.layers:
    //   0: Conv1d k7 512→1024; {2,5,8,11}: ConvTr upsamplers; {3,6,9,12}:
    //   Resnet; 14: Conv1d k3 64→1.
    auto conv = [&](const std::string& base, int stride, int dil) {
        MimiConvWeights cw;
        cw.weight = w.bf16_tensor(engine, base + ".weight");
        cw.bias = w.bf16_tensor(engine, base + ".bias");
        cw.stride = stride; cw.dilation = dil; cw.groups = 1;
        return cw;
    };
    w.mimi.seanet_in = conv("codec_model.decoder.layers.0.conv", 1, 1);
    struct StageDef { int idx; int stride; };
    StageDef stages[4] = {{2, c.codec.upsampling_ratios[0]},
                          {5, c.codec.upsampling_ratios[1]},
                          {8, c.codec.upsampling_ratios[2]},
                          {11, c.codec.upsampling_ratios[3]}};
    w.mimi.seanet_stages.resize(4);
    for (int si = 0; si < 4; ++si) {
        const auto& s = stages[si];
        auto& st = w.mimi.seanet_stages[si];
        st.convtr.weight = w.bf16_tensor(engine,
            "codec_model.decoder.layers." + std::to_string(s.idx) + ".conv.weight");
        st.convtr.bias = w.bf16_tensor(engine,
            "codec_model.decoder.layers." + std::to_string(s.idx) + ".conv.bias");
        st.convtr.stride = s.stride; st.convtr.groups = 1;
        const int rb = s.idx + 1;  // resnet block layer index
        st.resnet.conv1 = conv("codec_model.decoder.layers." + std::to_string(rb) + ".block.1.conv", 1, 1);
        st.resnet.conv2 = conv("codec_model.decoder.layers." + std::to_string(rb) + ".block.3.conv", 1, 1);
    }
    w.mimi.seanet_out = conv("codec_model.decoder.layers.14.conv", 1, 1);

    if (verbose) {
        std::fprintf(stderr,
            "[csm] bound: backbone %d layers (hidden %d), depth %d layers, mimi %d xf layers, "
            "%d codebooks; text_vocab %d audio_vocab %d\n",
            BBL, w.config.hidden_size, DDL, XFL, NCB, c.text_vocab_size, c.audio_vocab_size);
    }
    return w;
}

}  // namespace pie_cuda_driver::model
