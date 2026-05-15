#include "model/qwen3.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda_check.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("llama-like: missing weight '" + name + "'");
    }
    return e.get(name);
}

// Stack three BF16 row-major tensors along axis 0 into one fused tensor
// of shape [r0+r1+r2, cols]. All inputs must share `cols`; their actual
// row counts are read from `.shape()[0]`. Inserts the result into the
// engine under `fused_name`. Idempotent on duplicate registration:
// callers should check `!engine.has(fused_name)` first when re-binding.
//
// Returns a non-owning pointer to the inserted DeviceTensor.
const DeviceTensor* fuse_three_rowwise_bf16(
    LoadedModel& engine,
    const DeviceTensor& a, const DeviceTensor& b, const DeviceTensor& c,
    const std::string& fused_name)
{
    if (a.dtype() != DType::BF16 || b.dtype() != DType::BF16 ||
        c.dtype() != DType::BF16) {
        throw std::runtime_error(
            "fuse_three_rowwise_bf16: only BF16 inputs supported (got "
            + std::string(dtype_name(a.dtype())) + "/"
            + dtype_name(b.dtype()) + "/" + dtype_name(c.dtype()) + ")");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2 || c.shape().size() != 2) {
        throw std::runtime_error("fuse_three_rowwise_bf16: expected 2-D tensors");
    }
    const std::int64_t cols = a.shape()[1];
    if (b.shape()[1] != cols || c.shape()[1] != cols) {
        throw std::runtime_error("fuse_three_rowwise_bf16: column mismatch");
    }
    const std::int64_t r0 = a.shape()[0], r1 = b.shape()[0], r2 = c.shape()[0];
    const std::int64_t out_rows = r0 + r1 + r2;

    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {out_rows, cols});

    const std::size_t row_bytes_a = static_cast<std::size_t>(r0) * cols * sizeof(std::uint16_t);
    const std::size_t row_bytes_b = static_cast<std::size_t>(r1) * cols * sizeof(std::uint16_t);
    const std::size_t row_bytes_c = static_cast<std::size_t>(r2) * cols * sizeof(std::uint16_t);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(dst,                              a.data(), row_bytes_a, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + row_bytes_a,                b.data(), row_bytes_b, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + row_bytes_a + row_bytes_b,  c.data(), row_bytes_c, cudaMemcpyDeviceToDevice));

    engine.insert(fused_name, std::move(fused));
    return &engine.get(fused_name);
}

// Two-input variant used by gate+up.
const DeviceTensor* fuse_two_rowwise_bf16(
    LoadedModel& engine,
    const DeviceTensor& a, const DeviceTensor& b,
    const std::string& fused_name)
{
    if (a.dtype() != DType::BF16 || b.dtype() != DType::BF16) {
        throw std::runtime_error(
            "fuse_two_rowwise_bf16: only BF16 inputs supported");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2 ||
        a.shape()[1] != b.shape()[1]) {
        throw std::runtime_error("fuse_two_rowwise_bf16: shape mismatch");
    }
    const std::int64_t cols = a.shape()[1];
    const std::int64_t r0 = a.shape()[0], r1 = b.shape()[0];

    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {r0 + r1, cols});

    const std::size_t row_bytes_a = static_cast<std::size_t>(r0) * cols * sizeof(std::uint16_t);
    const std::size_t row_bytes_b = static_cast<std::size_t>(r1) * cols * sizeof(std::uint16_t);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(dst,               a.data(), row_bytes_a, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + row_bytes_a, b.data(), row_bytes_b, cudaMemcpyDeviceToDevice));

    engine.insert(fused_name, std::move(fused));
    return &engine.get(fused_name);
}

}  // namespace

Qwen3Weights bind_llama_like(LoadedModel& engine) {
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

        // P1.1 — Fuse Q/K/V and gate/up projections at bind time so the
        // forward path can issue one wide gemm per group instead of three
        // or two narrow ones. cuBLAS picks the `gemvx` fallback at small
        // M (decode steps with concurrency 8 sit at M=8 per individual
        // projection); the fused matmul lifts M to Hq+2*Hk / 2*I, which
        // sits comfortably inside the tensor-core gemm path.
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
            const std::string fused_name = p + "self_attn.qkv_proj.fused.weight";
            if (!engine.has(fused_name)) {
                fuse_three_rowwise_bf16(
                    engine, *L.q_proj, *L.k_proj, *L.v_proj, fused_name);
            }
            L.qkv_proj_fused = &engine.get(fused_name);
        }
        if (!gu_quantized && gu_bf16) {
            const std::string fused_name = p + "mlp.gate_up_proj.fused.weight";
            if (!engine.has(fused_name)) {
                fuse_two_rowwise_bf16(
                    engine, *L.gate_proj, *L.up_proj, fused_name);
            }
            L.gate_up_proj_fused = &engine.get(fused_name);
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
    LoadedModel& e,
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

Qwen3Weights bind_phi3(LoadedModel& engine) {
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

Qwen3Weights bind_olmo3(LoadedModel& engine) {
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
