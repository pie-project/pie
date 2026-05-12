#pragma once

// Llama-style transformer weight schema. Holds non-owning pointers into the
// Engine's weight pool, grouped by transformer block. Unfused — Q/K/V and
// gate/up are kept separate; QKV fusion is an optimization for later.
//
// Same struct shape covers Qwen3, Llama 3, Qwen 2, and Mistral. The Qwen3
// quirk (per-head q_norm / k_norm) is captured by leaving those pointers
// null on architectures that don't have them; the forward pass skips the
// extra RMSNorm in that case.

#include <cstdint>
#include <optional>
#include <vector>

#include "engine.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct Qwen3LayerWeights {
    // RMSNorm weights (1D, [hidden] each).
    const DeviceTensor* attn_norm = nullptr;   // input_layernorm
    const DeviceTensor* mlp_norm  = nullptr;   // post_attention_layernorm

    // Self-attention projections (all bf16).
    const DeviceTensor* q_proj = nullptr;      // [num_q_heads*head_dim, hidden]
    const DeviceTensor* k_proj = nullptr;      // [num_kv_heads*head_dim, hidden]
    const DeviceTensor* v_proj = nullptr;      // [num_kv_heads*head_dim, hidden]
    const DeviceTensor* o_proj = nullptr;      // [hidden, num_q_heads*head_dim]

    // Optional fused QKV weight `[Hq + 2*Hk, H]` = [q_proj; k_proj; v_proj]
    // concatenated along the output dim. Populated at bind time for bf16
    // layers; null when any of q/k/v is quantized (separate-GEMM
    // fallback). When non-null, the forward calls one GEMM into a
    // scratch buffer + an unpack kernel that fills the existing ws.q,
    // ws.k, ws.v contiguous buffers.
    const DeviceTensor* qkv_proj = nullptr;

    // Optional QKV bias terms. Set on Qwen-2 / OLMo-3 / GPT-OSS, null on
    // Llama-3 / Qwen-3 / Phi-3 / Mistral. When non-null, applied
    // post-projection via `launch_add_bias_bf16`.
    const DeviceTensor* q_bias = nullptr;      // [num_q_heads*head_dim]
    const DeviceTensor* k_bias = nullptr;      // [num_kv_heads*head_dim]
    const DeviceTensor* v_bias = nullptr;      // [num_kv_heads*head_dim]

    // Per-head QK normalization (Qwen3 / Gemma-3 / OLMo-3; weight length
    // = head_dim). Null on Llama 3 / Qwen 2 / Mistral.
    const DeviceTensor* q_norm = nullptr;
    const DeviceTensor* k_norm = nullptr;

    // MLP.
    const DeviceTensor* gate_proj = nullptr;   // [intermediate, hidden]
    const DeviceTensor* up_proj   = nullptr;   // [intermediate, hidden]
    const DeviceTensor* down_proj = nullptr;   // [hidden, intermediate]

    // Optional fused gate+up weight `[2*I, H]` = [gate_proj; up_proj].
    // Populated for bf16 layers; null when either gate or up is
    // quantized. Forward calls one GEMM into ws.gate_up + strided
    // swiglu (no unpack — swiglu reads gate at offset 0, up at offset
    // I within each row, then writes to a separate contiguous ws.gate).
    const DeviceTensor* gate_up_proj = nullptr;

    // Optional QuantMeta companions for each weight. Null when the
    // weight is plain bf16 (the common case). When set, the forward
    // pass routes the corresponding GEMM through ops::gemm_act_x_w with
    // a quantized WeightView (FP8 / INT4 / etc.). Bind functions
    // populate these by calling `engine.quant_meta(weight_name)` after
    // resolving each pointer. The QuantMeta value lives in the engine's
    // side-map; this is just a pointer into it.
    std::optional<QuantMeta> q_proj_quant;
    std::optional<QuantMeta> k_proj_quant;
    std::optional<QuantMeta> v_proj_quant;
    std::optional<QuantMeta> o_proj_quant;
    std::optional<QuantMeta> gate_proj_quant;
    std::optional<QuantMeta> up_proj_quant;
    std::optional<QuantMeta> down_proj_quant;
};

// Pick the right WeightView for a (weight, optional-quant-meta) pair.
// When `meta` is null, returns a plain bf16 view; when set, returns a
// quantized view that the GEMM dispatcher routes to the appropriate
// kernel (cuBLASLt FP8, marlin int4, …). Same call shape works for
// every model that wires QuantMeta companions in.
inline ops::WeightView make_weight_view(const DeviceTensor* w,
                                        const std::optional<QuantMeta>& meta) {
    if (meta.has_value()) {
        return ops::WeightView::quantized(*w, *meta);
    }
    return ops::WeightView(*w);
}

struct Qwen3Weights {
    const DeviceTensor* embed       = nullptr;  // [vocab, hidden]
    const DeviceTensor* final_norm  = nullptr;  // [hidden]
    const DeviceTensor* lm_head     = nullptr;  // [vocab, hidden] (may alias embed)
    std::vector<Qwen3LayerWeights> layers;
};

/// Build the schema by name-binding tensors out of the engine. Throws if a
/// required weight is missing; tolerates a missing `lm_head` (falls back to
/// `embed` when `tie_word_embeddings` is set). Reads `cfg.use_qk_norm` to
/// decide whether to require q/k_norm weights, and `cfg.use_qkv_bias` to
/// decide whether to bind q/k/v bias terms.
///
/// Also materialises a fused `qkv_proj` weight per layer when q/k/v are all
/// plain bf16 (most common case). The fused weight is registered in the
/// engine under `model.layers.{i}.self_attn.qkv_proj_packed.weight` and
/// drives a single-GEMM-plus-unpack path in the forward.
Qwen3Weights bind_llama_like(Engine& engine);

// Backward-compatible alias for callers still using `bind_qwen3`.
inline Qwen3Weights bind_qwen3(Engine& engine) { return bind_llama_like(engine); }

// Phi-3 ships fused `qkv_proj` and `gate_up_proj` weights. The bind
// function below splits them into the standard q/k/v/gate/up slots
// expected by the Llama-like forward, by registering virtual sub-views
// in the engine's weight pool. Returns the same `Qwen3Weights` shape.
Qwen3Weights bind_phi3(Engine& engine);

// OLMo-3 ships separate Q/K/V (no fused weights), but stores its norms
// at HF positions that don't match Llama. Map:
//   * `post_attention_layernorm` → attn_norm (used as the post-attn
//     RMSNorm in the post-norm forward graph).
//   * `post_feedforward_layernorm` → mlp_norm (post-MLP).
// OLMo-3 has no `input_layernorm` because the architecture is
// post-norm. The forward path for OLMo-3 selects post-norm via
// `LlamaLikeForwardCfg::norm_placement = NormPlacement::Post`.
Qwen3Weights bind_olmo3(Engine& engine);

}  // namespace pie_cuda_driver::model
