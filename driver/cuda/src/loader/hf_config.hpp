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

    // ── Storage dtype as declared on disk (for the safetensors loader).
    std::string torch_dtype;   // "bfloat16", "float16", "float32".
};

// Parse `<snapshot_dir>/config.json`. Throws on missing required fields.
HfConfig parse_hf_config(const std::filesystem::path& path);

}  // namespace pie_cuda_driver
