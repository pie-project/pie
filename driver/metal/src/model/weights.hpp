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
#include <vector>

#include "ops/tensor.hpp"   // pie_metal_driver::Tensor + ops::empty_tensor() (beta)

namespace pie_metal_driver::model {

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
    // Default-initialized to a placeholder so the struct is default-
    // constructible (MLX `array` has no default ctor); bind_* overwrites.
    Tensor q_proj = ops::empty_tensor();
    Tensor k_proj = ops::empty_tensor();
    Tensor v_proj = ops::empty_tensor();
    Tensor o_proj = ops::empty_tensor();

    // Optional additive QKV bias (Qwen2). Empty on Llama3 / Qwen3 / Mistral.
    std::optional<Tensor> q_bias;
    std::optional<Tensor> k_bias;
    std::optional<Tensor> v_bias;

    // Optional per-head Q/K RMSNorm weights (Qwen3 / Gemma3); length head_dim.
    std::optional<Tensor> q_norm;
    std::optional<Tensor> k_norm;

    // ── Dense MLP (SwiGLU / GeGLU). Empty on MoE layers. ──
    std::optional<Tensor> gate_proj;
    std::optional<Tensor> up_proj;
    std::optional<Tensor> down_proj;

    // ── MoE (Qwen3-MoE / Mixtral). Empty on dense layers. ──
    // router      : [n_experts, hidden]
    // gate/up/down_exps : stacked per-expert weights, expert-major.
    std::optional<Tensor> moe_router;
    std::optional<Tensor> moe_gate_exps;
    std::optional<Tensor> moe_up_exps;
    std::optional<Tensor> moe_down_exps;
};

struct ModelWeights {
    Tensor                embed      = ops::empty_tensor();  // [vocab, hidden]
    Tensor                final_norm = ops::empty_tensor();  // [hidden]
    std::optional<Tensor> lm_head;      // empty when tie_word_embeddings (use embed)

    // LLaMA-3.1 NTK-by-parts per-dim RoPE scaling, synthesized at load time.
    // Empty for plain theta-only RoPE (qwen2/qwen3/llama3.0/gemma).
    std::optional<Tensor> freq_factors;

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

struct ModelConfig;  // fwd

// Bind the Llama-like schema (llama3/qwen2/qwen3/mistral + qwen3-moe/mixtral).
// Reads config flags (qk-norm, qkv-bias, MoE) to decide which optional
// tensors to bind. Implemented in llama_like.cpp.
ModelWeights bind_llama_like(const WeightSource& src, const ModelConfig& cfg);

// Bind the Gemma2/3 schema (the four-norm sandwich + optional qk-norm).
// Implemented in gemma.cpp.
ModelWeights bind_gemma(const WeightSource& src, const ModelConfig& cfg);

}  // namespace pie_metal_driver::model
