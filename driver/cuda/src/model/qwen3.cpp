#include "model/qwen3.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("llama-like: missing weight '" + name + "'");
    }
    return e.get(name);
}

// Allocate a fresh bf16 `[2*I, H]` device tensor and copy gate_proj /
// up_proj into the two row blocks. Used by the fused-MLP path; mirror
// of pack_qkv_weight_bf16 below.
void pack_gate_up_weight_bf16(
    Engine& e,
    const std::string& name,
    const DeviceTensor& gate, const DeviceTensor& up,
    std::int64_t I, std::int64_t H)
{
    const std::int64_t two_I = 2 * I;
    const std::size_t row_bytes = static_cast<std::size_t>(H) * 2u;
    const std::size_t half_bytes = row_bytes * static_cast<std::size_t>(I);

    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {two_I, H});
    auto* dst_b = static_cast<std::uint8_t*>(fused.data());

    CUDA_CHECK(cudaMemcpy(dst_b,               gate.data(),
                          half_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst_b + half_bytes,  up.data(),
                          half_bytes, cudaMemcpyDeviceToDevice));

    e.insert(name, std::move(fused));
}

// Allocate a fresh bf16 `[Hq + 2*Hk, H]` device tensor and copy q_proj,
// k_proj, v_proj into the three row blocks. Registers the new tensor in
// the engine under `name`. Skipped when any of the three is quantized
// (the caller falls back to separate GEMMs in that case).
//
// Memory cost: 2 * sizeof(bf16) * (Hq + 2*Hk) * H per layer. For
// Qwen3-8B that's 48 MiB/layer × 36 layers = 1.7 GiB on top of the
// original weights. Worth it for the per-layer GEMM-count reduction;
// can be revisited (free the originals after packing) if VRAM is tight.
void pack_qkv_weight_bf16(
    Engine& e,
    const std::string& name,
    const DeviceTensor& q, const DeviceTensor& k, const DeviceTensor& v,
    std::int64_t Hq, std::int64_t Hk, std::int64_t H)
{
    const std::int64_t Hqkv = Hq + 2 * Hk;
    const std::size_t bytes_per_row = static_cast<std::size_t>(H) * 2u;
    const std::size_t q_bytes = bytes_per_row * static_cast<std::size_t>(Hq);
    const std::size_t k_bytes = bytes_per_row * static_cast<std::size_t>(Hk);

    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {Hqkv, H});
    auto* dst_b = static_cast<std::uint8_t*>(fused.data());

    // Row layout matches HF: [out_dim, in_dim] row-major. Concatenating
    // along out_dim is three contiguous byte copies.
    CUDA_CHECK(cudaMemcpy(dst_b,                              q.data(),
                          q_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst_b + q_bytes,                    k.data(),
                          k_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst_b + q_bytes + k_bytes,          v.data(),
                          k_bytes, cudaMemcpyDeviceToDevice));

    e.insert(name, std::move(fused));
}

}  // namespace

Qwen3Weights bind_llama_like(Engine& engine) {
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

        // Pull QuantMeta side-map entries — one per projection. Stays
        // empty for unquantized models (the common case).
        L.q_proj_quant    = engine.quant_meta(p + "self_attn.q_proj.weight");
        L.k_proj_quant    = engine.quant_meta(p + "self_attn.k_proj.weight");
        L.v_proj_quant    = engine.quant_meta(p + "self_attn.v_proj.weight");
        L.o_proj_quant    = engine.quant_meta(p + "self_attn.o_proj.weight");
        L.gate_proj_quant = engine.quant_meta(p + "mlp.gate_proj.weight");
        L.up_proj_quant   = engine.quant_meta(p + "mlp.up_proj.weight");
        L.down_proj_quant = engine.quant_meta(p + "mlp.down_proj.weight");

        // Optional QKV fusion: materialise a packed `[Hq + 2*Hk, H]`
        // weight when q/k/v are all bf16. Skipped for quantized layers
        // (each weight has its own scale; merging would require
        // recomputing a unified quant scheme — separate dispatch path
        // is simpler than rewriting the quant kernel for fused inputs).
        if (!L.q_proj_quant && !L.k_proj_quant && !L.v_proj_quant &&
            L.q_proj->dtype() == DType::BF16 &&
            L.k_proj->dtype() == DType::BF16 &&
            L.v_proj->dtype() == DType::BF16)
        {
            const std::int64_t H  = cfg.hidden_size;
            const std::int64_t Hq =
                static_cast<std::int64_t>(cfg.num_attention_heads) * cfg.head_dim;
            const std::int64_t Hk =
                static_cast<std::int64_t>(cfg.num_key_value_heads) * cfg.head_dim;
            const std::string packed_name =
                p + "self_attn.qkv_proj_packed.weight";
            if (!engine.has(packed_name)) {
                pack_qkv_weight_bf16(engine, packed_name,
                                     *L.q_proj, *L.k_proj, *L.v_proj,
                                     Hq, Hk, H);
            }
            L.qkv_proj = &engine.get(packed_name);
        }

        // Same gating logic for gate/up. The chunked-swiglu kernel
        // (`launch_chunked_swiglu_bf16`) already consumes the `[N, 2*I]`
        // packed layout (originally introduced for Qwen3.6 MoE), so the
        // forward only needs the fused weight + workspace — no new
        // kernel.
        if (!L.gate_proj_quant && !L.up_proj_quant &&
            L.gate_proj->dtype() == DType::BF16 &&
            L.up_proj->dtype()   == DType::BF16)
        {
            const std::int64_t H = cfg.hidden_size;
            const std::int64_t I = cfg.intermediate_size;
            const std::string packed_name =
                p + "mlp.gate_up_proj_packed.weight";
            if (!engine.has(packed_name)) {
                pack_gate_up_weight_bf16(engine, packed_name,
                                         *L.gate_proj, *L.up_proj, I, H);
            }
            L.gate_up_proj = &engine.get(packed_name);
        }
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

    // Phi-3 stores QKV as one fused row-major `[Hq + 2*Hk, H]`. On TP=1 we
    // load the fused tensor and expose q/k/v as non-owning sub-views; on
    // TP>1 the engine load loop has already pre-split the fused weight
    // into per-rank q_proj/k_proj/v_proj tensors (a naive axis-0 split of
    // the fused tensor straddles the Q/K/V block boundaries), so this
    // function just verifies the unfused names exist and lets
    // `bind_llama_like` pick them up.
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        const std::string fused_qkv = p + "self_attn.qkv_proj.weight";
        if (engine.has(fused_qkv)) {
            register_row_slice(engine, fused_qkv, p + "self_attn.q_proj.weight",
                               /*row_offset=*/0,           Hq, H, dtype);
            register_row_slice(engine, fused_qkv, p + "self_attn.k_proj.weight",
                               /*row_offset=*/Hq,          Hk, H, dtype);
            register_row_slice(engine, fused_qkv, p + "self_attn.v_proj.weight",
                               /*row_offset=*/Hq + Hk,     Hk, H, dtype);
        } else if (!engine.has(p + "self_attn.q_proj.weight")) {
            throw std::runtime_error(
                "bind_phi3: neither fused " + fused_qkv +
                " nor unfused q_proj is loaded");
        }

        const std::string fused_gu = p + "mlp.gate_up_proj.weight";
        if (engine.has(fused_gu)) {
            register_row_slice(engine, fused_gu, p + "mlp.gate_proj.weight",
                               /*row_offset=*/0, I, H, dtype);
            register_row_slice(engine, fused_gu, p + "mlp.up_proj.weight",
                               /*row_offset=*/I, I, H, dtype);
        } else if (!engine.has(p + "mlp.gate_proj.weight")) {
            throw std::runtime_error(
                "bind_phi3: neither fused " + fused_gu +
                " nor unfused gate_proj is loaded");
        }
    }
    return bind_llama_like(engine);
}

Qwen3Weights bind_olmo3(Engine& engine) {
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
