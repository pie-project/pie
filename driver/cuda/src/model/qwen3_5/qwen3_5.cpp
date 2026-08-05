#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("qwen3_5: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}


// Qwen3.5 (multimodal config) nests the text tower under
// `model.language_model.`; the vision tower lives under `model.visual.`
// and is unused on the text-only path.
constexpr const char* kPrefix = "model.language_model.";

}  // namespace

Qwen3_5Weights bind_qwen3_5(LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    const int L = cfg.num_hidden_layers;

    if (cfg.layer_types.empty() ||
        static_cast<int>(cfg.layer_types.size()) != L) {
        throw std::runtime_error(
            "qwen3_5: HfConfig.layer_types must have num_hidden_layers entries");
    }
    if (cfg.linear_num_value_heads <= 0 || cfg.linear_num_key_heads <= 0
            || cfg.linear_key_head_dim <= 0 || cfg.linear_value_head_dim <= 0
            || cfg.linear_conv_kernel_dim <= 0) {
        throw std::runtime_error(
            "qwen3_5: linear-attn dimensions are unset; check the loader's "
            "HfConfig parsing for linear_num_*_heads / linear_*_head_dim / "
            "linear_conv_kernel_dim.");
    }

    Qwen3_5Weights w;
    w.layers.resize(static_cast<std::size_t>(L));

    const std::string p = kPrefix;

    w.embed      = &must(engine, p + "embed_tokens.weight");
    w.final_norm = &must(engine, p + "norm.weight");
    // Tied lm_head: HF omits the tensor and aliases to embed_tokens.
    w.lm_head    = cfg.tie_word_embeddings
                       ? w.embed
                       : &must(engine, "lm_head.weight");

    // KV cache slot is assigned only to full-attention layers, in
    // ascending order. Linear layers don't occupy KV-cache slots —
    // their state lives in the recurrent/conv caches built by the
    // forward.

    int kv_slot = 0;
    for (int li = 0; li < L; ++li) {
        const std::string lp = p + "layers." + std::to_string(li) + ".";
        auto& Lw = w.layers[li];
        const auto& kind = cfg.layer_types[li];

        Lw.attn_norm_pre = &must(engine, lp + "input_layernorm.weight");
        Lw.mlp_norm_pre  = &must(engine, lp + "post_attention_layernorm.weight");

        // MLP weights are present on every layer (linear or full).
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj   = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");
        Lw.gate_proj_quant = engine.quant_meta(lp + "mlp.gate_proj.weight");
        Lw.up_proj_quant   = engine.quant_meta(lp + "mlp.up_proj.weight");
        Lw.down_proj_quant = engine.quant_meta(lp + "mlp.down_proj.weight");
        // The contract publishes the join plus views of it under the original
        // names, so this reads the same way whether or not the group was fused
        // -- and when it was, gate/up are views into the fused bank rather than
        // a second copy of it.
        Lw.gate_up_proj_fused = maybe(engine, lp + "mlp.gate_up_proj.fused.weight");

        if (kind == "linear_attention") {
            Lw.kind = Qwen3_5LayerWeights::Kind::LinearAttn;
            const std::string la = lp + "linear_attn.";
            // Whichever layout the contract published. The fused switch makes
            // the join a load-time `Concat`, so the separate tensors are simply
            // absent -- there is never a moment when both are resident.
            Lw.la_in_proj_qkv = maybe(engine, la + "in_proj_qkv.weight");
            Lw.la_in_proj_z = maybe(engine, la + "in_proj_z.weight");
            Lw.la_in_proj_b = maybe(engine, la + "in_proj_b.weight");
            Lw.la_in_proj_a = maybe(engine, la + "in_proj_a.weight");
            // This rank's `[K/T | K/T | V/T]`: the contract states the
            // per-block shard, so the whole tensor is never resident here.
            Lw.la_conv1d_w = &must(engine, la + "conv1d.weight");
            Lw.la_conv1d_b = maybe(engine, la + "conv1d.bias");
            Lw.la_dt_bias  = &must(engine, la + "dt_bias");
            // fp32 by contract (`gdn_fp32_parameters`), so these are the
            // loaded bytes rather than a bind-time copy of them.
            Lw.la_A_log_fp32 =
                static_cast<const float*>(must(engine, la + "A_log").data());
            Lw.la_norm_w_fp32 =
                static_cast<const float*>(must(engine, la + "norm.weight").data());
            Lw.la_out_proj = &must(engine, la + "out_proj.weight");
            Lw.kv_layer = -1;
        } else if (kind == "full_attention") {
            Lw.kind = Qwen3_5LayerWeights::Kind::FullAttn;
            const std::string fa = lp + "self_attn.";
            // q_proj is 2× wide (query + gate fused).
            Lw.fa_q_proj = &must(engine, fa + "q_proj.weight");
            Lw.fa_k_proj = &must(engine, fa + "k_proj.weight");
            Lw.fa_v_proj = &must(engine, fa + "v_proj.weight");
            Lw.fa_o_proj = &must(engine, fa + "o_proj.weight");
            Lw.fa_q_norm = &must(engine, fa + "q_norm.weight");
            Lw.fa_k_norm = &must(engine, fa + "k_norm.weight");
            Lw.fa_q_proj_quant = engine.quant_meta(fa + "q_proj.weight");
            Lw.fa_k_proj_quant = engine.quant_meta(fa + "k_proj.weight");
            Lw.fa_v_proj_quant = engine.quant_meta(fa + "v_proj.weight");
            Lw.fa_o_proj_quant = engine.quant_meta(fa + "o_proj.weight");
            Lw.kv_layer = kv_slot++;
        } else {
            throw std::runtime_error(
                "qwen3_5: unknown layer_type '" + kind + "' at layer " +
                std::to_string(li));
        }
    }

    if (cfg.mtp_num_hidden_layers > 0 && engine.has("mtp.fc.weight")) {
        Qwen3_5Weights::MtpWeights mtp;
        mtp.pre_fc_norm_embedding = &must(engine, "mtp.pre_fc_norm_embedding.weight");
        mtp.pre_fc_norm_hidden = &must(engine, "mtp.pre_fc_norm_hidden.weight");
        mtp.fc = &must(engine, "mtp.fc.weight");
        mtp.norm = &must(engine, "mtp.norm.weight");
        mtp.embed = cfg.mtp_use_dedicated_embeddings
            ? &must(engine, "mtp.embed_tokens.weight")
            : w.embed;
        mtp.lm_head = w.lm_head;

        const std::string lp = "mtp.layers.0.";
        auto& Lw = mtp.layer;
        Lw.kind = Qwen3_5LayerWeights::Kind::FullAttn;
        Lw.attn_norm_pre = &must(engine, lp + "input_layernorm.weight");
        Lw.mlp_norm_pre = &must(engine, lp + "post_attention_layernorm.weight");
        const std::string fa = lp + "self_attn.";
        Lw.fa_q_proj = &must(engine, fa + "q_proj.weight");
        Lw.fa_k_proj = &must(engine, fa + "k_proj.weight");
        Lw.fa_v_proj = &must(engine, fa + "v_proj.weight");
        Lw.fa_o_proj = &must(engine, fa + "o_proj.weight");
        Lw.fa_q_norm = &must(engine, fa + "q_norm.weight");
        Lw.fa_k_norm = &must(engine, fa + "k_norm.weight");
        Lw.fa_q_proj_quant = engine.quant_meta(fa + "q_proj.weight");
        Lw.fa_k_proj_quant = engine.quant_meta(fa + "k_proj.weight");
        Lw.fa_v_proj_quant = engine.quant_meta(fa + "v_proj.weight");
        Lw.fa_o_proj_quant = engine.quant_meta(fa + "o_proj.weight");
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");
        Lw.gate_proj_quant = engine.quant_meta(lp + "mlp.gate_proj.weight");
        Lw.up_proj_quant = engine.quant_meta(lp + "mlp.up_proj.weight");
        Lw.down_proj_quant = engine.quant_meta(lp + "mlp.down_proj.weight");
        Lw.gate_up_proj_fused =
            maybe(engine, lp + "mlp.gate_up_proj.fused.weight");
        Lw.kv_layer = kv_slot++;
        // `mtp_int8_lm_head` publishes this beside the bf16 head when
        // `PIE_QWEN35_MTP_INT8_LM_HEAD` is set; absent, the draft step reads
        // the same bf16 head as the main path.
        if (const DeviceTensor* int8_head = maybe(engine, "mtp.lm_head")) {
            const std::optional<ops::QuantMeta> meta = engine.quant_meta("mtp.lm_head");
            if (meta.has_value() && meta->scale != nullptr) {
                mtp.lm_head = int8_head;
                mtp.lm_head_scale_inv = meta->scale;
            }
        }
        w.mtp = mtp;
    }

    return w;
}

}  // namespace pie_cuda_driver::model
