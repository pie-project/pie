#pragma once

// Parser for HuggingFace `config.json` plus arch-name canonicalization.
//
// Pie's reference loader at `pie/src/pie_driver/loader.py` reads the same
// fields; we mirror its surface so the C++ driver and the Python driver
// agree on hyperparameter interpretation. See `model/<arch>.py::ModelConfig.from_dict`
// for the per-arch field expectations.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace pie_portable_driver {

// Canonical Pie architecture identifiers. Must match the strings the
// runtime accepts in `bootstrap::ModelConfig::arch_name` and the inferlet
// instruct-template registry at `runtime/src/model/instruct.rs`.
enum class PieArch {
    Qwen3,
    Qwen2,
    Llama3,
    Gemma2,
    Gemma3,
    Gemma4,
    Gemma3n,     // Gemma 3n (E2B / E4B); HF model_type "gemma3n" /
                 // "gemma3n_text". Dense per-layer attention with the
                 // Gemma 3 sliding/full mix. v1 implementation skips
                 // AltUp / PLE / Laurel (the parameter-efficient
                 // tricks); the runtime executes a vanilla Gemma 3-
                 // style graph and produces approximately-correct text.
                 // Full architectural fidelity is a follow-up.
    Mistral3,
    Olmo3,
    GptOss,
    Phi3,
    Mixtral,
    Qwen3Moe,    // Qwen3-MoE family (Qwen3-30B-A3B etc.); HF model_type "qwen3_moe"
    Qwen3_5,     // Qwen 3.5 / 3.6 hybrid (gated delta + GQA);
                 // HF model_type "qwen3_5" (dense) or "qwen3_5_moe" (MoE)
    Phi3Small,   // Phi-3-small (microsoft); HF model_type "phi3small". Uses
                 // LayerNorm with bias, fused query_key_value, packed
                 // mlp.up_proj (gate||up), self_attn.dense, mup parameter-
                 // ization. v1 graph treats blocksparse as causal (correct
                 // for prompts up to num_local_blocks * block_size = 1024
                 // tokens).
    Phi3_5Moe,   // Phi-3.5-MoE (microsoft); HF model_type "phimoe". Mixtral-
                 // style sparse MoE (per-expert w1/w2/w3) with LayerNorm-
                 // with-bias norms, Q/K/V/O biases, lm_head bias.
};

const char* pie_arch_name(PieArch a);

// Maps an HF `model_type` (e.g. "qwen3", "llama") to its Pie counterpart.
// Throws on unknown types.
PieArch hf_model_type_to_pie_arch(const std::string& hf_model_type);

// Subset of `config.json` we use. Optional fields have std::optional values
// so per-arch builders can branch on presence.
struct Hparams {
    PieArch arch;
    std::string hf_model_type;     // raw "qwen3"
    std::string torch_dtype;       // "bfloat16" / "float16" / "float32"

    // Common transformer hparams.
    std::int32_t num_hidden_layers = 0;
    std::int32_t num_attention_heads = 0;
    std::int32_t num_key_value_heads = 0;
    // Gemma 4 alternative attention: full_attention layers use a
    // SMALLER kv_heads count than sliding layers (e.g., 4 vs 16 on 31B).
    // 0 = not provided / not applicable.
    std::int32_t num_global_key_value_heads = 0;
    // Gemma 4 26B-A4B: parallel sparse-MoE block alongside the dense MLP.
    // When true, the loader expects per-layer experts.gate_up_proj /
    // experts.down_proj / router.{proj,scale,per_expert_scale} plus
    // pre_feedforward_layernorm_2 + post_feedforward_layernorm_{1,2}.
    bool         gemma4_enable_moe          = false;
    // Phi-3-small mup parameterization.
    float        mup_attn_multiplier         = 0.0f;  // 0 = unused
    float        mup_embedding_multiplier    = 0.0f;
    float        mup_width_multiplier        = 0.0f;
    // Phi-3-small blocksparse attention parameters.
    std::int32_t phi3small_block_size                  = 0;
    std::int32_t phi3small_num_local_blocks            = 0;
    std::int32_t phi3small_vert_stride                 = 0;
    std::int32_t phi3small_dense_attention_every_n_layers = 0;
    std::int32_t gemma4_moe_intermediate_size = 0;
    std::int32_t hidden_size = 0;
    std::int32_t intermediate_size = 0;
    std::int32_t head_dim = 0;             // computed if missing
    std::int32_t vocab_size = 0;
    std::int32_t max_position_embeddings = 0;

    // Norm / RoPE.
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1e6f;
    // Gemma3 / Gemma4: separate base frequency for sliding-window layers.
    // 0.0 = no override (fall back to rope_theta).
    float rope_local_base_freq = 0.0f;

    // Tied embeddings (qwen3 default true; llama3 typically false).
    bool tie_word_embeddings = true;

    // Sliding window attention. Some archs use it; populated when present.
    std::optional<std::int32_t> sliding_window;
    bool use_sliding_window = false;

    // RoPE scaling (LLaMA-3.1 NTK-by-parts, YARN, linear). Stored raw —
    // the per-arch graph builder interprets the structure.
    bool has_rope_scaling = false;
    std::string rope_scaling_type;             // "llama3" / "yarn" / "linear" / ""
    float rope_scaling_factor = 1.0f;
    float rope_scaling_low_freq_factor = 1.0f;
    float rope_scaling_high_freq_factor = 4.0f;
    std::int32_t rope_scaling_original_max_position = 0;
    // YaRN-specific (olmo3, Ministral 3, gpt-oss). Defaults match HF.
    float rope_yarn_attention_factor = 0.0f;   // 0 = use 0.1*ln(factor)+1
    float rope_yarn_beta_fast = 32.0f;
    float rope_yarn_beta_slow = 1.0f;

    // Per-layer attention type list (olmo3, gpt-oss, gemma4 etc.). One char
    // per layer: 's' (sliding_attention) or 'g' (full_attention). Empty if
    // the config doesn't specify a per-layer pattern.
    std::vector<char> layer_types;

    // ── Gemma 4 / 3n ──
    // Per-attention-type RoPE (gemma4 uses a `rope_parameters` dict keyed by
    // "full_attention" / "sliding_attention" with per-entry rope_theta and
    // partial_rotary_factor). For non-gemma4 archs these stay defaulted and
    // are unused.
    float gemma4_rope_theta_full = 1e6f;
    float gemma4_rope_theta_sliding = 1e4f;
    float gemma4_rope_partial_factor_full = 1.0f;     // 1.0 = full rotation
    float gemma4_rope_partial_factor_sliding = 1.0f;
    // Per-layer-type head_dim. Sliding head_dim lives in `head_dim`; full
    // attention reads `gemma4_head_dim_global`. 0 = same as head_dim.
    std::int32_t gemma4_head_dim_global = 0;
    // KV-cache sharing: last `gemma4_num_kv_shared_layers` layers reuse
    // upstream non-shared layer's K/V (matched by attention type). 0 = no
    // sharing. The shared layers still have k_proj/v_proj weights in the
    // safetensors; the driver loads them but never feeds them at inference.
    std::int32_t gemma4_num_kv_shared_layers = 0;
    // Per-Layer Embeddings: extra ple_dim residual injected after MLP each
    // layer. 0 = PLE disabled (gemma4 always >0 in shipped checkpoints).
    std::int32_t gemma4_ple_dim = 0;
    std::int32_t gemma4_ple_vocab = 0;
    bool         gemma4_use_double_wide_mlp = false;

    // ── Gemma 3n: AltUp / Laurel ──
    // altup_num_inputs (=4): number of parallel hidden streams.
    // altup_active_idx (=0): which stream the transformer block actually
    //   reads/writes; the others are predicted/corrected from the active
    //   stream's behavior via small router + 4×4 / 4×16 coefficient
    //   matrices learned per layer.
    // altup_correct_scale (=true): apply correct_output_scale to the
    //   active stream after each layer's correction step.
    // laurel_rank: low-rank dim for the Laurel residual MLP (=64 in E2B/E4B).
    std::int32_t altup_num_inputs    = 0;   // 0 = arch doesn't use AltUp
    std::int32_t altup_active_idx    = 0;
    bool         altup_correct_scale = false;
    std::int32_t laurel_rank         = 0;
    // Per-layer "Gaussian top-k" pre-activation sparsity. Layers where this
    // is > 0 zero out everything below `mean + icdf(p)*std` in the gate
    // projection (E2B/E4B: 0.95 on layers 0-9, 0.0 on the rest). Empty
    // vector = activation sparsity disabled.
    std::vector<float> activation_sparsity_pattern;

    // Logit softcap (gemma2; future gemma).
    std::optional<float> attn_logit_softcapping;
    std::optional<float> final_logit_softcapping;
    // Custom Q scaling — gemma2/3 use 1/sqrt(query_pre_attn_scalar)
    // instead of 1/sqrt(head_dim). 0 = use head_dim default.
    std::optional<float> query_pre_attn_scalar;

    // ── Mixture-of-Experts (Mixtral, Qwen2/3-MoE, GPT-OSS, etc.) ──
    // num_local_experts (Mixtral) / num_experts (Qwen-MoE) — total experts
    // per layer. 0 = not an MoE model.
    std::int32_t num_experts = 0;
    // num_experts_per_tok — top-k routing.
    std::int32_t num_experts_per_tok = 0;
    // moe_intermediate_size — per-expert FFN hidden size. Defaults to
    // intermediate_size when not specified separately.
    std::int32_t moe_intermediate_size = 0;
    // norm_topk_prob (qwen-moe / mixtral) — renormalize the selected
    // top-k expert weights so they sum to 1.
    bool norm_topk_prob = true;
    // Qwen 3.6 (qwen3_5_moe): shared dense expert that runs alongside the
    // routed experts. 0 disables (Qwen-MoE / Mixtral / gpt-oss don't have it).
    std::int32_t shared_expert_intermediate_size = 0;

    // ── Qwen 3.5 / 3.6 ──
    // Hybrid arch: 3-of-4 layers are gated-delta-rule "linear attention"
    // with recurrent matrix state, every 4th layer is standard GQA with
    // mrope + output gate. Vision tower + multi-token-prediction head
    // are present in checkpoints but ignored by the driver.
    bool         qwen35_attn_output_gate     = false;
    std::int32_t qwen35_full_attn_interval   = 0;
    std::int32_t qwen35_linear_num_k_heads   = 0;
    std::int32_t qwen35_linear_num_v_heads   = 0;
    std::int32_t qwen35_linear_k_head_dim    = 0;
    std::int32_t qwen35_linear_v_head_dim    = 0;
    std::int32_t qwen35_linear_conv_kernel   = 0;
    bool         qwen35_mrope_interleaved    = false;
    std::int32_t qwen35_mrope_section[3]     = {0, 0, 0};
    float        qwen35_partial_rotary_factor = 1.0f;
};

Hparams parse_hf_config(const std::filesystem::path& config_json_path);

}  // namespace pie_portable_driver
