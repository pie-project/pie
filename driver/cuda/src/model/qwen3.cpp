#include "model/qwen3.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("llama-like: missing weight '" + name + "'");
    }
    return e.get(name);
}

}  // namespace

Qwen3Weights bind_llama_like(const Engine& engine) {
    const auto& cfg = engine.hf_config();

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

        L.q_proj = &must(engine, p + "self_attn.q_proj.weight");
        L.k_proj = &must(engine, p + "self_attn.k_proj.weight");
        L.v_proj = &must(engine, p + "self_attn.v_proj.weight");
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

        L.gate_proj = &must(engine, p + "mlp.gate_proj.weight");
        L.up_proj   = &must(engine, p + "mlp.up_proj.weight");
        L.down_proj = &must(engine, p + "mlp.down_proj.weight");
    }

    return w;
}

namespace {

// Helper: register a non-owning sub-view into an already-loaded fused
// weight. `row_offset` is in elements (not bytes); `rows` is the slice
// height. The slice is contiguous along the leading axis (HF stores
// projection weights as row-major `[out_dim, in_dim]`, which is the
// flashinfer/cublas convention used downstream).
void register_row_slice(
    Engine& e,
    const std::string& fused_name,
    const std::string& slice_name,
    std::int64_t row_offset, std::int64_t rows, std::int64_t cols,
    DType dtype)
{
    const auto& fused = e.get(fused_name);
    if (fused.dtype() != dtype) {
        throw std::runtime_error(
            "register_row_slice: dtype mismatch on '" + fused_name + "'");
    }
    const std::int64_t element_bytes = static_cast<std::int64_t>(dtype_bytes(dtype));
    auto* base = static_cast<std::uint8_t*>(const_cast<void*>(fused.data()));
    auto* slice_ptr = base + row_offset * cols * element_bytes;
    e.insert(slice_name,
             DeviceTensor::view(slice_ptr, dtype, {rows, cols}));
}

}  // namespace

Qwen3Weights bind_phi3(Engine& engine) {
    const auto& cfg = engine.hf_config();
    const std::int64_t H  = cfg.hidden_size;
    const std::int64_t Hq = static_cast<std::int64_t>(cfg.num_attention_heads) * cfg.head_dim;
    const std::int64_t Hk = static_cast<std::int64_t>(cfg.num_key_value_heads) * cfg.head_dim;
    const std::int64_t I  = cfg.intermediate_size;
    const DType dtype = DType::BF16;

    // Phi-3 stores QKV as one fused row-major `[Hq + 2*Hk, H]`. Slice
    // it into the q/k/v_proj names that `bind_llama_like` expects.
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        const std::string fused_qkv = p + "self_attn.qkv_proj.weight";
        if (!engine.has(fused_qkv)) {
            throw std::runtime_error(
                "bind_phi3: expected fused weight " + fused_qkv);
        }
        register_row_slice(engine, fused_qkv, p + "self_attn.q_proj.weight",
                           /*row_offset=*/0,           Hq, H, dtype);
        register_row_slice(engine, fused_qkv, p + "self_attn.k_proj.weight",
                           /*row_offset=*/Hq,          Hk, H, dtype);
        register_row_slice(engine, fused_qkv, p + "self_attn.v_proj.weight",
                           /*row_offset=*/Hq + Hk,     Hk, H, dtype);

        // Same trick for fused `gate_up_proj`: rows 0..I are gate, rows
        // I..2I are up.
        const std::string fused_gu = p + "mlp.gate_up_proj.weight";
        if (!engine.has(fused_gu)) {
            throw std::runtime_error(
                "bind_phi3: expected fused weight " + fused_gu);
        }
        register_row_slice(engine, fused_gu, p + "mlp.gate_proj.weight",
                           /*row_offset=*/0, I, H, dtype);
        register_row_slice(engine, fused_gu, p + "mlp.up_proj.weight",
                           /*row_offset=*/I, I, H, dtype);
    }
    return bind_llama_like(engine);
}

Qwen3Weights bind_olmo3(const Engine& engine) {
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
