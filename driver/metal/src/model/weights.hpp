#pragma once

// Per-architecture weight schemas. Each struct holds MLX `Tensor`s (no
// ownership semantics beyond MLX's own refcounting) that the loader (delta)
// binds by HF name and the graph builders (this dir) read. Optional tensors
// use std::optional<Tensor> and are left empty on architectures that lack
// them — the forward path skips the corresponding op (e.g. Qwen2 bias,
// Qwen3/Gemma3 q/k-norm, Gemma's extra norms).
//
// One struct shape (`LayerWeights`) covers the entire Llama-like family AND
// Gemma2/3: the Gemma-specific norms live alongside the Llama ones and are
// simply unused (empty) on Llama/Qwen. MoE tensors are empty on dense models.
//
// Binding is delta's job: it produces a populated `ModelWeights` from the
// weight_loader. The `bind_*` free functions declared here are the agreed
// seam — delta implements name resolution, this dir defines the target shape.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ops/gemm.hpp"     // ops::linear / ops::quantized_linear (beta)
#include "ops/tensor.hpp"   // pie_metal_driver::Tensor + ops::empty_tensor() (beta)
#include "model/config.hpp" // ModelConfig (quant_bits / quant_group_size)

namespace pie_metal_driver::model {

// QuantLinear — a single linear projection's weight, either dense BF16 or
// mlx-community affine-quantized. Quantized checkpoints pack each nn.Linear
// into a uint32 `weight` + per-group `scales`/`biases`; dense weights carry no
// scales. `quantized()` (== scales present) is the per-tensor flag the graph
// dispatches on (see `apply_linear`): present -> ops::quantized_linear, absent
// -> the plain BF16 ops::linear. The presence of `.scales` in the safetensors
// index IS the quant map — no separate flag needed from the loader.
struct QuantLinear {
    Tensor weight = ops::empty_tensor();   // BF16 [out,in] OR uint32-packed
    std::optional<Tensor> scales;          // present <=> affine-quantized
    std::optional<Tensor> biases;
    int group_size = 64;
    int bits       = 4;
    bool quantized() const { return scales.has_value(); }

    QuantLinear() = default;
    // A dense linear is just its weight tensor — implicit so existing call
    // sites (and synthetic-weight harnesses) that assign a bare Tensor keep
    // working; quantized bundles are built field-wise by bind_linear.
    QuantLinear(Tensor w) : weight(std::move(w)) {}
};

// Dispatch a linear projection: fused dequant-in-GEMV when quantized, plain
// matmul otherwise. Numerically identical to the eager op sequence either way.
inline Tensor apply_linear(const QuantLinear& w, const Tensor& x) {
    if (w.quantized()) {
        return ops::quantized_linear(w.weight, *w.scales, *w.biases, x,
                                     w.group_size, w.bits);
    }
    return ops::linear(w.weight, x);
}

// `mlx::core::array` has no default constructor, so structs holding a
// non-optional `Tensor` aren't default-constructible — which `bind_*`'s
// `ModelWeights w; w.layers.resize(n)` (and `std::vector` resize) require.
// We default-init those members to beta's canonical `ops::empty_tensor()`
// placeholder; every required tensor is overwritten by `bind_*` before any
// forward runs.

struct LayerWeights {
    // ── Norms ──
    // attn_norm    : input_layernorm (pre-attention). Null on post-norm archs.
    // ffn_norm     : pre-FFN norm (Llama post_attention_layernorm, or Gemma
    //                pre_feedforward_layernorm).
    // post_attn_norm / post_ffn_norm : Gemma-only extra norms in the sandwich.
    std::optional<Tensor> attn_norm;
    std::optional<Tensor> ffn_norm;
    std::optional<Tensor> post_attn_norm;   // Gemma2/3
    std::optional<Tensor> post_ffn_norm;    // Gemma2/3

    // ── Attention projections (W[out,in]; linear(w,x) -> [n,out]) ──
    // QuantLinear: dense BF16 by default, or affine-quantized when the loader
    // surfaces `.scales`/`.biases` siblings. Default-constructed (empty weight)
    // so the struct stays default-constructible; bind_* overwrites.
    QuantLinear q_proj;
    QuantLinear k_proj;
    QuantLinear v_proj;
    QuantLinear o_proj;

    // Optional additive QKV bias (Qwen2). Empty on Llama3 / Qwen3 / Mistral.
    std::optional<Tensor> q_bias;
    std::optional<Tensor> k_bias;
    std::optional<Tensor> v_bias;

    // Optional per-head Q/K RMSNorm weights (Qwen3 / Gemma3); length head_dim.
    std::optional<Tensor> q_norm;
    std::optional<Tensor> k_norm;

    // ── Dense MLP (SwiGLU / GeGLU). Empty on MoE layers. ──
    std::optional<QuantLinear> gate_proj;
    std::optional<QuantLinear> up_proj;
    std::optional<QuantLinear> down_proj;

    // ── MoE (Qwen3-MoE / Mixtral). Empty on dense layers. ──
    // router      : [n_experts, hidden]
    // gate/up/down_exps : stacked per-expert weights, expert-major.
    std::optional<Tensor> moe_router;
    std::optional<Tensor> moe_gate_exps;
    std::optional<Tensor> moe_up_exps;
    std::optional<Tensor> moe_down_exps;

    // ── gemma4 Per-Layer-Embedding (PLE) triple. Empty on non-gemma4. ──
    // ple_input_gate : [ple_dim, hidden]  (projects residual -> ple gate)
    // ple_projection : [hidden, ple_dim]  (projects gated signal back)
    // ple_norm       : [hidden]           (post-projection RMSNorm)
    std::optional<QuantLinear> ple_input_gate;
    std::optional<QuantLinear> ple_projection;
    std::optional<Tensor> ple_norm;
    // Per-layer learnable output scalar (1-element). Empty = 1.0.
    std::optional<Tensor> layer_scalar;

    // ── qwen3.6 Gated-DeltaNet linear-attention. Empty on full-attn layers. ──
    // De-interleaved at bind time from HF's per-head-group `in_proj_qkvz` /
    // `in_proj_ba` into contiguous tensors matching beta's gated_delta_net:
    //   la_in_proj_qkv : [conv_dim, hidden]   (conv_dim = 2*K_h*K_d + V_h*V_d,
    //                     row-contiguous [q | k | v])
    //   la_in_proj_z   : [V_dim, hidden]      (gate for RMSNormGated)
    //   la_in_proj_a/b : [V_h, hidden]        (g / beta sources)
    //   la_conv1d_w    : [conv_dim, conv_K]   la_conv1d_b: [conv_dim] (optional)
    //   la_A_log/la_dt_bias : [V_h]           la_gate_norm: [V_d]
    //   la_out_proj    : [hidden, V_dim]
    std::optional<Tensor> la_in_proj_qkv;
    std::optional<Tensor> la_in_proj_z;
    std::optional<Tensor> la_in_proj_a;
    std::optional<Tensor> la_in_proj_b;
    std::optional<Tensor> la_conv1d_w;
    std::optional<Tensor> la_conv1d_b;
    std::optional<Tensor> la_A_log;
    std::optional<Tensor> la_dt_bias;
    std::optional<Tensor> la_gate_norm;
    std::optional<Tensor> la_out_proj;
};

struct ModelWeights {
    Tensor                embed      = ops::empty_tensor();  // [vocab, hidden]
    Tensor                final_norm = ops::empty_tensor();  // [hidden]
    std::optional<QuantLinear> lm_head;  // empty when tie_word_embeddings (use embed)

    // LLaMA-3.1 NTK-by-parts per-dim RoPE scaling, synthesized at load time.
    // Empty for plain theta-only RoPE (qwen2/qwen3/llama3.0/gemma).
    std::optional<Tensor> freq_factors;

    // ── gemma4 Per-Layer-Embedding model-level tensors. Empty otherwise. ──
    // embed_per_layer : [vocab, n_layers * ple_dim]  (per-layer token table)
    // ple_model_proj  : [n_layers * ple_dim, hidden] (projects main embed)
    // ple_model_norm  : [ple_dim]                     (RMSNorm over ple_dim)
    std::optional<Tensor> embed_per_layer;
    std::optional<QuantLinear> ple_model_proj;
    std::optional<Tensor> ple_model_norm;

    std::vector<LayerWeights> layers;
};

// WeightSource — the binding seam delta implements. Resolves an HF tensor
// name to an MLX `Tensor` (already resident on-device), or std::nullopt when
// the tensor is absent. The `bind_*` builders below call this; keeping the
// interface abstract lets delta back it with the weight_loader / a name map
// without this dir depending on loader internals.
class WeightSource {
public:
    virtual ~WeightSource() = default;
    // Required tensor: throws (or the impl decides) if missing.
    virtual Tensor get(const std::string& hf_name) const = 0;
    // Optional tensor: std::nullopt when absent.
    virtual std::optional<Tensor> try_get(const std::string& hf_name) const = 0;
    virtual bool has(const std::string& hf_name) const = 0;
};

// bind_linear — resolve a linear projection's weight (required) plus the
// optional affine-quant siblings. Quantized iff `<base>.scales` is present in
// the source (the index-as-quant-map convention); group_size/bits come from
// the config (mlx-community default 64/4). `base` is the HF name WITHOUT the
// `.weight` suffix (e.g. "...self_attn.q_proj").
inline QuantLinear bind_linear(const WeightSource& src, const std::string& base,
                               const ModelConfig& cfg) {
    QuantLinear q;
    q.weight     = src.get(base + ".weight");
    q.scales     = src.try_get(base + ".scales");
    q.biases     = src.try_get(base + ".biases");
    q.group_size = cfg.quant_group_size > 0 ? cfg.quant_group_size : 64;
    q.bits       = cfg.quant_bits > 0 ? cfg.quant_bits : 4;
    return q;
}

// try_bind_linear — std::nullopt when the projection's `.weight` is absent.
inline std::optional<QuantLinear> try_bind_linear(const WeightSource& src,
                                                  const std::string& base,
                                                  const ModelConfig& cfg) {
    if (!src.has(base + ".weight")) return std::nullopt;
    return bind_linear(src, base, cfg);
}

// Bind the Llama-like schema (llama3/qwen2/qwen3/mistral + qwen3-moe/mixtral).
// Reads config flags (qk-norm, qkv-bias, MoE) to decide which optional
// tensors to bind. Implemented in llama_like.cpp.
ModelWeights bind_llama_like(const WeightSource& src, const ModelConfig& cfg);

// Bind the Gemma2/3 schema (the four-norm sandwich + optional qk-norm).
// Implemented in gemma.cpp.
ModelWeights bind_gemma(const WeightSource& src, const ModelConfig& cfg);

// Bind the Gemma-4 schema (dense E2B/E4B): the four-norm sandwich + q/k-norm,
// the PLE triple (per-layer-embedding gate/projection/norm + model-level
// table/projection/norm) and the per-layer output scalar. Cross-layer KV-share
// and per-layer head_dim are resolved in the graph from config. Implemented in
// gemma4.cpp.
ModelWeights bind_gemma4(const WeightSource& src, const ModelConfig& cfg);

// Bind the Qwen3.6 (Qwen3-Next "qwen3_5" family) hybrid schema. Full-attn
// layers reuse the Llama-like q/k/v/o (q_proj is 2x wide: per-head [q|gate]) +
// gemma-style q/k RMSNorm. Linear-attn layers de-interleave HF's per-head-group
// `in_proj_qkvz` / `in_proj_ba` into the contiguous tensors beta's
// gated_delta_net consumes, plus conv1d / A_log / dt_bias / gate-norm / out_proj.
// Both kinds carry a dense SwiGLU MLP. Implemented in qwen36.cpp.
ModelWeights bind_qwen36(const WeightSource& src, const ModelConfig& cfg);

}  // namespace pie_metal_driver::model
