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

namespace pie_ggml_driver {

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
    Mistral3,
    Olmo3,
    GptOss,
    Phi3,
    Mixtral,
    Qwen3_5,
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
};

Hparams parse_hf_config(const std::filesystem::path& config_json_path);

}  // namespace pie_ggml_driver
