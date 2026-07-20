#include "model/llama_like/qwen3.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("llama-like: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe_tensor(const LoadedModel& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

const DeviceTensor* bind_projection_or_fused_view(
    const LoadedModel& e,
    const std::string& name,
    const DeviceTensor* fused,
    const std::string& fused_name,
    std::int64_t row_offset,
    std::int64_t rows,
    std::int64_t cols,
    std::unique_ptr<DeviceTensor>& view_slot)
{
    if (const DeviceTensor* direct = maybe_tensor(e, name)) {
        return direct;
    }
    if (fused == nullptr) {
        throw std::runtime_error("llama-like: missing weight '" + name + "'");
    }
    const auto& shape = fused->shape();
    if (fused->dtype() != DType::BF16 || shape.size() != 2 ||
        shape[1] != cols || row_offset < 0 || rows <= 0 ||
        row_offset + rows > shape[0]) {
        throw std::runtime_error(
            "llama-like: fused weight '" + fused_name +
            "' cannot provide view for missing weight '" + name + "'");
    }
    auto* base = static_cast<std::uint8_t*>(
        const_cast<void*>(fused->data()));
    auto* ptr = base + static_cast<std::size_t>(row_offset) *
                       static_cast<std::size_t>(cols) *
                       dtype_bytes(fused->dtype());
    view_slot = std::make_unique<DeviceTensor>(
        DeviceTensor::view(ptr, fused->dtype(), {rows, cols}));
    return view_slot.get();
}

}  // namespace

Qwen3Weights bind_llama_like(const LoadedModel& engine, bool verbose) {
    const auto& cfg = engine.hf_config();
    (void)verbose;

    Qwen3Weights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");

    // Some configs (Llama 3 1B, Qwen3 with tie_word_embeddings) drop the
    // separate lm_head. Fall back to the embed table when allowed.
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error("llama-like: lm_head missing and tie_word_embeddings=false");
    }

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.attn_norm = &must(engine, p + "input_layernorm.weight");
        L.mlp_norm  = &must(engine, p + "post_attention_layernorm.weight");

        const int H = cfg.hidden_size;
        const int Hq = cfg.num_attention_heads * cfg.head_dim;
        const int Hk = cfg.num_key_value_heads * cfg.head_dim;
        const int I = cfg.intermediate_size;
        const std::string qkv_fused_name =
            p + "self_attn.qkv_proj.fused.weight";
        const std::string gate_up_fused_name =
            p + "mlp.gate_up_proj.fused.weight";
        const DeviceTensor* qkv_fused = maybe_tensor(engine, qkv_fused_name);
        const DeviceTensor* gate_up_fused =
            maybe_tensor(engine, gate_up_fused_name);

        L.q_proj = bind_projection_or_fused_view(
            engine, p + "self_attn.q_proj.weight",
            qkv_fused, qkv_fused_name, 0, Hq, H, L.q_proj_view);
        L.k_proj = bind_projection_or_fused_view(
            engine, p + "self_attn.k_proj.weight",
            qkv_fused, qkv_fused_name, Hq, Hk, H, L.k_proj_view);
        L.v_proj = bind_projection_or_fused_view(
            engine, p + "self_attn.v_proj.weight",
            qkv_fused, qkv_fused_name, Hq + Hk, Hk, H, L.v_proj_view);
        L.o_proj = &must(engine, p + "self_attn.o_proj.weight");

        // QKV biases (Qwen-2 / OLMo-3 / GPT-OSS). HF stores them on the
        // same module as the weight, so the convention is `*_proj.bias`.
        if (cfg.attention_bias) {
            L.q_bias = &must(engine, p + "self_attn.q_proj.bias");
            L.k_bias = &must(engine, p + "self_attn.k_proj.bias");
            L.v_bias = &must(engine, p + "self_attn.v_proj.bias");
        }

        // Per-head q/k norm: required on Qwen3 / Gemma-3 / OLMo-3; absent
        // on Llama 3 / Mistral / Qwen 2 / Phi-3.
        if (cfg.use_qk_norm) {
            L.q_norm = &must(engine, p + "self_attn.q_norm.weight");
            L.k_norm = &must(engine, p + "self_attn.k_norm.weight");
        }

        L.gate_proj = bind_projection_or_fused_view(
            engine, p + "mlp.gate_proj.weight",
            gate_up_fused, gate_up_fused_name, 0, I, H, L.gate_proj_view);
        L.up_proj = bind_projection_or_fused_view(
            engine, p + "mlp.up_proj.weight",
            gate_up_fused, gate_up_fused_name, I, I, H, L.up_proj_view);
        L.down_proj = &must(engine, p + "mlp.down_proj.weight");

        // Pull QuantMeta side-map entries — one per projection. Stays
        // empty for unquantized models (the common case).
        L.q_proj_quant    = engine.quant_meta(p + "self_attn.q_proj.weight");
        L.k_proj_quant    = engine.quant_meta(p + "self_attn.k_proj.weight");
        L.v_proj_quant    = engine.quant_meta(p + "self_attn.v_proj.weight");
        L.o_proj_quant    = engine.quant_meta(p + "self_attn.o_proj.weight");
        L.gate_proj_quant = engine.quant_meta(p + "mlp.gate_proj.weight");
        L.up_proj_quant   = engine.quant_meta(p + "mlp.up_proj.weight");
        L.down_proj_quant = engine.quant_meta(p + "mlp.down_proj.weight");

        // Use planned packed Q/K/V and gate/up projections when the loader
        // installed them. Older/unplanned paths may still materialize them
        // here when the memory guard allows it, so the forward path can issue
        // one wide gemm per group instead of three or two narrow ones.
        //
        // Skipped when any projection in the group is quantized (FP8 /
        // INT4 paths carry per-weight scales that don't compose across a
        // concat) or when bf16 is required for the post-load fuse memcpy
        // and the projection isn't bf16. In both cases the forward path
        // sees a null `*_fused` pointer and stays on the unfused branch.
        const bool qkv_quantized =
            L.q_proj_quant.has_value() || L.k_proj_quant.has_value() ||
            L.v_proj_quant.has_value();
        const bool gu_quantized =
            L.gate_proj_quant.has_value() || L.up_proj_quant.has_value();
        const bool qkv_bf16 =
            L.q_proj->dtype() == DType::BF16 &&
            L.k_proj->dtype() == DType::BF16 &&
            L.v_proj->dtype() == DType::BF16;
        const bool gu_bf16 =
            L.gate_proj->dtype() == DType::BF16 &&
            L.up_proj->dtype() == DType::BF16;
        if (!qkv_quantized && qkv_bf16) {
            if (qkv_fused != nullptr) {
                L.qkv_proj_fused = qkv_fused;
            }
        }
        if (!gu_quantized && gu_bf16) {
            if (gate_up_fused != nullptr) {
                L.gate_up_proj_fused = gate_up_fused;
            }
        }
    }

    return w;
}

Qwen3Weights bind_phi3(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();

    // The Rust loader always splits Phi-3 fused QKV and gate/up checkpoint
    // tensors into the canonical Llama-like names before binding.
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        if (!engine.has(p + "self_attn.q_proj.weight") ||
            !engine.has(p + "self_attn.k_proj.weight") ||
            !engine.has(p + "self_attn.v_proj.weight")) {
            throw std::runtime_error(
                "bind_phi3: storage loader did not materialize q/k/v projections");
        }

        if (!engine.has(p + "mlp.gate_proj.weight") ||
            !engine.has(p + "mlp.up_proj.weight")) {
            throw std::runtime_error(
                "bind_phi3: storage loader did not materialize gate/up projections");
        }
    }
    return bind_llama_like(engine);
}

Qwen3Weights bind_olmo3(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();

    Qwen3Weights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "olmo3: lm_head missing and tie_word_embeddings=false");
    }

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];

        // Post-norm placement: HF's `post_attention_layernorm` is the
        // norm we apply *after* attention; `post_feedforward_layernorm`
        // is applied *after* MLP. There is no `input_layernorm` on
        // OLMo-3 — the forward pass reads `y` (residual stream) into
        // QKV directly.
        L.attn_norm = &must(engine, p + "post_attention_layernorm.weight");
        L.mlp_norm  = &must(engine, p + "post_feedforward_layernorm.weight");

        L.q_proj = &must(engine, p + "self_attn.q_proj.weight");
        L.k_proj = &must(engine, p + "self_attn.k_proj.weight");
        L.v_proj = &must(engine, p + "self_attn.v_proj.weight");
        L.o_proj = &must(engine, p + "self_attn.o_proj.weight");

        if (cfg.attention_bias) {
            L.q_bias = &must(engine, p + "self_attn.q_proj.bias");
            L.k_bias = &must(engine, p + "self_attn.k_proj.bias");
            L.v_bias = &must(engine, p + "self_attn.v_proj.bias");
        }

        // OLMo-3 always has q/k norms (its key feature alongside post-norm).
        L.q_norm = &must(engine, p + "self_attn.q_norm.weight");
        L.k_norm = &must(engine, p + "self_attn.k_norm.weight");

        L.gate_proj = &must(engine, p + "mlp.gate_proj.weight");
        L.up_proj   = &must(engine, p + "mlp.up_proj.weight");
        L.down_proj = &must(engine, p + "mlp.down_proj.weight");
    }
    return w;
}

}  // namespace pie_cuda_driver::model
