#pragma once

// Parsed HuggingFace `config.json`. We keep the dataclass narrow — only the
// fields the forward pass and capability handshake need. Add fields here as
// new architectures land.

#include <filesystem>
#include <string>
#include <vector>

namespace pie_cuda_driver {

struct HfConfig {
    // Architecture discriminator (e.g. "Qwen3ForCausalLM"). The first entry
    // of HF's `architectures` list — we keep the exact string for
    // DriverCapabilities.arch_name and for our own model-registry dispatch.
    std::string arch_name;

    // Lower-case `model_type` ("qwen3", "llama", …). Used for registry lookup
    // because some HF configs ship with multiple architectures.
    std::string model_type;

    // ── Transformer dimensions ────────────────────────────────────────
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;   // GQA. Equal to num_attention_heads if MHA.
    int head_dim;              // Some configs imply head_dim = hidden / heads;
                               // Qwen3 sets it explicitly.
    // The HEAD_DIM the attention kernel actually runs at. Equal to
    // `head_dim` for every model whose head_dim is in flashinfer's
    // dispatch set ({64, 128, 256, 512}). For Phi-3-mini (head_dim=96)
    // we round up to 128 and pad Q/K/V/O with zeros — flashinfer's
    // tensor-core prefill produces NaN at HEAD_DIM=96 (a swizzle/
    // alignment bug in 0.6.x).
    int head_dim_kernel;
    int vocab_size;
    int max_position_embeddings;

    // ── Norm / activation ─────────────────────────────────────────────
    float rms_norm_eps;
    std::string hidden_act;    // "silu" — only one supported for now.

    // ── RoPE ──────────────────────────────────────────────────────────
    float rope_theta;
    bool  has_rope_scaling;    // True when `rope_scaling.rope_type ==
                               // "llama3"` (YaRN-style) is set.
    // YaRN parameters. Inert when `has_rope_scaling == false`.
    float rope_factor;
    float rope_low_freq_factor;
    float rope_high_freq_factor;
    int   rope_original_max_position;

    // ── Sliding-window attention ──────────────────────────────────────
    // -1 means full causal. Positive = `window_left` per request to
    // pass through to flashinfer's prefill plan.
    int sliding_window;

    // Per-layer attention type. Empty when the model uses a single
    // attention type across all layers (homogeneous Llama / Mistral /
    // Phi-3). Populated for OLMo-3 (HF `layer_types`) and Gemma-2/3
    // (Gemma-2 hardcodes `i%2==0` = sliding in `modeling_gemma2.py`;
    // Gemma-3 uses `sliding_window_pattern` — every Nth layer is
    // full, others sliding). Values are kept as the same strings HF
    // stores: "sliding_attention" or "full_attention".
    std::vector<std::string> layer_types;

    // Gemma-3-style dual-RoPE base. Sliding layers in Gemma-3 use a
    // *separate* rope_theta (`rope_local_base_freq`) while full-attention
    // layers use the standard `rope_theta`. Zero means "fall back to
    // `rope_theta` for sliding layers too" (every other model).
    float rope_local_base_freq;

    // ── Architectural quirks ──────────────────────────────────────────
    bool tie_word_embeddings;  // If true, lm_head shares weight with embed.
    bool attention_bias;       // QKV/O biases (Qwen2 yes, Qwen3 no).
    bool use_qk_norm;          // Qwen3 / Gemma-3 / OLMo-3.

    // ── Sparse MoE (Mixtral / GPT-OSS / Qwen-3.5 hybrid) ─────────────
    // Zero on dense models. `num_experts` is HF's `num_local_experts`;
    // `num_experts_per_tok` is the top-K used by the router.
    int num_experts;
    int num_experts_per_tok;

    // GPT-OSS-specific knobs. Inert on every other model.
    //   * `swiglu_limit` — clipping threshold applied to gate values
    //     before SiLU (`x = clamp(x, -limit, limit)`). 0 = no clip.
    //   * `mlp_has_bias` — gate_up / down each carry an additive bias
    //     vector per expert.
    //   * `router_has_bias` — router projection has a bias term added
    //     before the top-K softmax.
    //   * `attention_has_sinks` — per-head learnable sink scalar that
    //     extends the softmax denominator (`Z + exp(sink_h)`). The
    //     forward path applies this as a post-attention rescale based
    //     on the kernel's log-sum-exp output.
    float swiglu_limit;
    bool  mlp_has_bias;
    bool  router_has_bias;
    bool  attention_has_sinks;

    // ── Gemma family ─────────────────────────────────────────────────
    // `query_pre_attn_scalar` defaults to `head_dim` so non-Gemma models
    // that don't set the field still get the standard `1/sqrt(head_dim)`
    // attention scale. `gemma_final_logit_softcap` is 0 when no cap is
    // requested (the kernel checks `> 0` and skips the call). The
    // attention-logit cap (Gemma-2 only — `attn_logit_softcapping=50`)
    // routes through flashinfer's `use_logits_soft_cap=true` variant.
    float gemma_query_pre_attn_scalar;
    float gemma_final_logit_softcap;
    float gemma_attn_logit_softcap;

    // Gemma-4 — Per-Layer Embeddings + KV-cache sharing. `ple_dim` is
    // 0 on every other model (no PLE residual block runs). When
    // `num_kv_shared_layers > 0` the last N layers reuse K/V from a
    // source layer of the same `layer_types[i]`.
    int gemma_hidden_size_per_layer_input;
    int num_kv_shared_layers;

    // Gemma-4 per-layer rope_theta (HF nests under `rope_parameters`),
    // including `partial_rotary_factor` for full-attention layers
    // (typically 0.25 — only the first 25% of head_dim is rotated).
    // Empty for non-Gemma-4 models. Indexed by layer.
    std::vector<float> gemma_per_layer_rope_theta;
    std::vector<float> gemma_per_layer_partial_rotary_factor;

    // ── Qwen3.6-MoE specific ─────────────────────────────────────────
    // Sparse-MoE block dims (zero on non-MoE archs). `moe_intermediate_size`
    // is the per-expert hidden width; `shared_expert_intermediate_size`
    // is the (always-on) shared expert's MLP width.
    int moe_intermediate_size;
    int shared_expert_intermediate_size;

    // ── Qwen3.5 hybrid (linear-attention SSM + full attention) ──────
    // Per-layer attention type is in `layer_types` (values
    // "linear_attention" / "full_attention"). The linear layers run a
    // Gated DeltaNet recurrence; the full layers run standard scaled-
    // dot-product attention with a per-token output gate (a' = a *
    // sigmoid(gate), where gate is the second half of `q_proj`).
    //
    // Linear-attention dimensions. Inert (zero) on every other model.
    int   linear_num_value_heads;     // 32 on Qwen3.5-4B
    int   linear_num_key_heads;       // 16 on Qwen3.5-4B
    int   linear_key_head_dim;        // 128 on Qwen3.5-4B
    int   linear_value_head_dim;      // 128 on Qwen3.5-4B
    int   linear_conv_kernel_dim;     // 4 (depthwise causal conv kernel)
    // Full-attention output gating: q_proj output is split (query, gate);
    // attn output is multiplied by sigmoid(gate) before o_proj.
    bool  attn_output_gate;
    // Partial RoPE: only the first `partial_rotary_factor * head_dim`
    // dimensions are rotated. Defaults to 1.0 (full rotation).
    float partial_rotary_factor;

    // Gemma-3n (E2B / E4B "Nano") additions on top of Gemma-4.
    // Gemma-3n is a *different* architecture from Gemma-4 (despite the
    // overlapping E2B / E4B naming) — it adds three new building blocks:
    //
    //   * AltUp ("Alternating Updates"): each layer maintains
    //     `altup_num_inputs` (4 by default) parallel residual streams; the
    //     active one (idx `altup_active_idx`) flows through attention +
    //     MLP, the others are updated via per-layer prediction /
    //     correction matmuls routed by a per-token "modality" vector.
    //   * Laurel ("Learned Augmented Residual Layer"): a per-layer
    //     low-rank skip — `linear_left` (H → laurel_rank), `linear_right`
    //     (laurel_rank → H), then RMSNorm and residual-add.
    //   * Activation sparsity: per-layer hard sparsity gate on the SwiGLU
    //     gate via Gaussian-quantile cutoff (only nonzero on the early
    //     layers of E2B per `activation_sparsity_pattern`).
    //
    // Plus `intermediate_size` is per-layer (HF stores it as a list).
    // We populate `gemma3n_per_layer_intermediate` from that list and
    // mirror the first element into the scalar `intermediate_size` for
    // back-compat with code that reads it as a scalar.
    int   altup_num_inputs;       // 4 on E2B/E4B
    int   altup_active_idx;       // 0 on E2B/E4B
    bool  altup_correct_scale;    // true on E2B/E4B
    float altup_coef_clip;        // 120.0 on E2B/E4B (training-only clip)
    int   laurel_rank;            // 64 on E2B/E4B
    int   vocab_size_per_layer_input;  // 262144 on E2B
    float gemma3n_rope_local_base_freq; // sliding-layer rope theta
    std::vector<int>   gemma3n_per_layer_intermediate;
    std::vector<float> gemma3n_activation_sparsity;

    // ── Storage dtype as declared on disk (for the safetensors loader).
    std::string torch_dtype;   // "bfloat16", "float16", "float32".
};

// Parse `<snapshot_dir>/config.json`. Throws on missing required fields.
HfConfig parse_hf_config(const std::filesystem::path& path);

}  // namespace pie_cuda_driver
