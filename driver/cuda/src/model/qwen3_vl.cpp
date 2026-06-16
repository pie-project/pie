#include "model/qwen3_vl.hpp"

#include <stdexcept>
#include <string>

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("qwen3_vl: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

}  // namespace

// ── Text decoder (standard Qwen3 under the `model.language_model.` prefix) ──
Qwen3Weights bind_qwen3_vl_text(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();

    const std::string mp = "model.language_model.";

    Qwen3Weights w;
    w.embed      = &must(engine, mp + "embed_tokens.weight");
    w.final_norm = &must(engine, mp + "norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "qwen3_vl: lm_head missing and tie_word_embeddings=false");
    }

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = mp + "layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.attn_norm = &must(engine, p + "input_layernorm.weight");
        L.mlp_norm  = &must(engine, p + "post_attention_layernorm.weight");

        L.q_proj = &must(engine, p + "self_attn.q_proj.weight");
        L.k_proj = &must(engine, p + "self_attn.k_proj.weight");
        L.v_proj = &must(engine, p + "self_attn.v_proj.weight");
        L.o_proj = &must(engine, p + "self_attn.o_proj.weight");

        if (cfg.attention_bias) {
            L.q_bias = &must(engine, p + "self_attn.q_proj.bias");
            L.k_bias = &must(engine, p + "self_attn.k_proj.bias");
            L.v_bias = &must(engine, p + "self_attn.v_proj.bias");
        }
        if (cfg.use_qk_norm) {
            L.q_norm = &must(engine, p + "self_attn.q_norm.weight");
            L.k_norm = &must(engine, p + "self_attn.k_norm.weight");
        }

        L.gate_proj = &must(engine, p + "mlp.gate_proj.weight");
        L.up_proj   = &must(engine, p + "mlp.up_proj.weight");
        L.down_proj = &must(engine, p + "mlp.down_proj.weight");

        // QuantMeta companions (empty on the bf16 Qwen3-VL-2B ckpt).
        L.q_proj_quant    = engine.quant_meta(p + "self_attn.q_proj.weight");
        L.k_proj_quant    = engine.quant_meta(p + "self_attn.k_proj.weight");
        L.v_proj_quant    = engine.quant_meta(p + "self_attn.v_proj.weight");
        L.o_proj_quant    = engine.quant_meta(p + "self_attn.o_proj.weight");
        L.gate_proj_quant = engine.quant_meta(p + "mlp.gate_proj.weight");
        L.up_proj_quant   = engine.quant_meta(p + "mlp.up_proj.weight");
        L.down_proj_quant = engine.quant_meta(p + "mlp.down_proj.weight");
    }
    return w;
}

// ── Vision tower (`model.visual.`) ──────────────────────────────────────────
Qwen3VLVisionWeights bind_qwen3_vl_vision(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (!cfg.qwen3_vl_vision.has_value()) {
        throw std::runtime_error(
            "qwen3_vl vision: HfConfig.qwen3_vl_vision is empty");
    }
    const Qwen3VLVisionConfig& vc = *cfg.qwen3_vl_vision;

    Qwen3VLVisionWeights w;
    w.config = vc;

    const std::string vp = "model.visual.";
    w.patch_weight = &must(engine, vp + "patch_embed.proj.weight");
    w.patch_bias   = &must(engine, vp + "patch_embed.proj.bias");
    w.pos_embed    = &must(engine, vp + "pos_embed.weight");

    const int L = vc.depth;
    w.layers.resize(static_cast<std::size_t>(L));
    for (int i = 0; i < L; ++i) {
        const std::string lp = vp + "blocks." + std::to_string(i) + ".";
        auto& Lw = w.layers[static_cast<std::size_t>(i)];
        Lw.norm1_weight = &must(engine, lp + "norm1.weight");
        Lw.norm1_bias   = &must(engine, lp + "norm1.bias");
        Lw.norm2_weight = &must(engine, lp + "norm2.weight");
        Lw.norm2_bias   = &must(engine, lp + "norm2.bias");
        Lw.qkv_weight   = &must(engine, lp + "attn.qkv.weight");
        Lw.qkv_bias     = &must(engine, lp + "attn.qkv.bias");
        Lw.proj_weight  = &must(engine, lp + "attn.proj.weight");
        Lw.proj_bias    = &must(engine, lp + "attn.proj.bias");
        Lw.fc1_weight   = &must(engine, lp + "mlp.linear_fc1.weight");
        Lw.fc1_bias     = &must(engine, lp + "mlp.linear_fc1.bias");
        Lw.fc2_weight   = &must(engine, lp + "mlp.linear_fc2.weight");
        Lw.fc2_bias     = &must(engine, lp + "mlp.linear_fc2.bias");
    }

    // Main merger (norm over hidden before the 2×2 shuffle).
    {
        const std::string mp = vp + "merger.";
        auto& m = w.merger;
        m.norm_weight = &must(engine, mp + "norm.weight");
        m.norm_bias   = &must(engine, mp + "norm.bias");
        m.fc1_weight  = &must(engine, mp + "linear_fc1.weight");
        m.fc1_bias    = &must(engine, mp + "linear_fc1.bias");
        m.fc2_weight  = &must(engine, mp + "linear_fc2.weight");
        m.fc2_bias    = &must(engine, mp + "linear_fc2.bias");
        m.use_postshuffle_norm = false;
    }

    // DeepStack mergers (norm over 4*hidden after the 2×2 shuffle).
    const int nd = static_cast<int>(vc.deepstack_visual_indexes.size());
    w.deepstack.resize(static_cast<std::size_t>(nd));
    for (int d = 0; d < nd; ++d) {
        const std::string dp =
            vp + "deepstack_merger_list." + std::to_string(d) + ".";
        auto& m = w.deepstack[static_cast<std::size_t>(d)];
        m.norm_weight = &must(engine, dp + "norm.weight");
        m.norm_bias   = &must(engine, dp + "norm.bias");
        m.fc1_weight  = &must(engine, dp + "linear_fc1.weight");
        m.fc1_bias    = &must(engine, dp + "linear_fc1.bias");
        m.fc2_weight  = &must(engine, dp + "linear_fc2.weight");
        m.fc2_bias    = &must(engine, dp + "linear_fc2.bias");
        m.use_postshuffle_norm = true;
    }
    w.deepstack_layer_idx = vc.deepstack_visual_indexes;

    (void)maybe;  // silence unused in case of future optional tensors
    return w;
}

}  // namespace pie_cuda_driver::model
