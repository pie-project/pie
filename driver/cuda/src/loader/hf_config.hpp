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
    int vocab_size;
    int max_position_embeddings;

    // ── Norm / activation ─────────────────────────────────────────────
    float rms_norm_eps;
    std::string hidden_act;    // "silu" — only one supported for now.

    // ── RoPE ──────────────────────────────────────────────────────────
    float rope_theta;
    bool  has_rope_scaling;    // For YaRN / linear scaling later.

    // ── Architectural quirks ──────────────────────────────────────────
    bool tie_word_embeddings;  // If true, lm_head shares weight with embed.
    bool attention_bias;       // QKV/O biases (Qwen2 yes, Qwen3 no).
    bool use_qk_norm;          // Qwen3-specific per-head q/k RMSNorm.

    // ── Storage dtype as declared on disk (for the safetensors loader).
    std::string torch_dtype;   // "bfloat16", "float16", "float32".
};

// Parse `<snapshot_dir>/config.json`. Throws on missing required fields.
HfConfig parse_hf_config(const std::filesystem::path& path);

}  // namespace pie_cuda_driver
