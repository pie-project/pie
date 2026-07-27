#include "model/llama_like/qwen3.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

}  // namespace

namespace {

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

        // The contract publishes a fused bank plus views of it under the
        // original names, so q/k/v and gate/up are read the same way whether
        // or not the group was fused, at every `tp_size`.
        const DeviceTensor* qkv_fused =
            maybe_tensor(engine, p + "self_attn.qkv_proj.fused.weight");
        const DeviceTensor* gate_up_fused =
            maybe_tensor(engine, p + "mlp.gate_up_proj.fused.weight");

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
        // installed them, so the forward path can issue one wide gemm per
        // group instead of three or two narrow ones.
        //
        // Whether a group is fused at all is the contract's call
        // (`contract.hpp::dense_fused_projection_joins`): it declines
        // quantized and non-BF16 groups, because per-weight scales do not
        // compose across a concat. When it declines, the tensor is absent and
        // the forward path stays on the unfused branch.
        L.qkv_proj_fused = qkv_fused;
        L.gate_up_proj_fused = gate_up_fused;
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
