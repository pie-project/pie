#include "loader/hf_config.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pie_cuda_driver {

namespace {

template <typename T>
T require(const nlohmann::json& j, const char* key, const std::string& path) {
    if (!j.contains(key)) {
        throw std::runtime_error("config.json (" + path + "): missing key '" + key + "'");
    }
    return j[key].get<T>();
}

template <typename T>
T optional(const nlohmann::json& j, const char* key, T default_value) {
    if (!j.contains(key) || j[key].is_null()) return default_value;
    return j[key].get<T>();
}

// Qwen3-specific signal: HF marks `use_qk_norm` implicitly via model_type.
// Some other archs use the same flag explicitly. Until we add per-arch
// metadata, derive it here.
bool infer_qk_norm(const std::string& model_type, const nlohmann::json& j) {
    if (j.contains("use_qk_norm")) return j["use_qk_norm"].get<bool>();
    return model_type == "qwen3" || model_type == "qwen3_5";
}

}  // namespace

HfConfig parse_hf_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config.json: " + path.string());

    nlohmann::json j;
    in >> j;
    const auto path_str = path.string();

    HfConfig cfg;

    if (!j.contains("architectures") || !j["architectures"].is_array() || j["architectures"].empty()) {
        throw std::runtime_error("config.json: missing or empty 'architectures'");
    }
    cfg.arch_name = j["architectures"][0].get<std::string>();
    cfg.model_type = optional<std::string>(j, "model_type", "");

    cfg.hidden_size              = require<int>(j, "hidden_size", path_str);
    cfg.intermediate_size        = require<int>(j, "intermediate_size", path_str);
    cfg.num_hidden_layers        = require<int>(j, "num_hidden_layers", path_str);
    cfg.num_attention_heads      = require<int>(j, "num_attention_heads", path_str);
    cfg.num_key_value_heads      = optional<int>(j, "num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim                 = optional<int>(j, "head_dim", cfg.hidden_size / cfg.num_attention_heads);

    // Round head_dim up to the nearest flashinfer-supported dispatch
    // value for kernel bookkeeping. Models in our supported set hit
    // this path only for Phi-3-mini (96 → 128); everything else maps
    // identically (64, 128, 256, 512).
    auto round_up_head_dim = [](int hd) {
        if (hd <= 64)   return 64;
        if (hd <= 128)  return 128;
        if (hd <= 256)  return 256;
        if (hd <= 512)  return 512;
        return hd;   // unsupported, but let the dispatch error surface
    };
    cfg.head_dim_kernel = round_up_head_dim(cfg.head_dim);
    cfg.vocab_size               = require<int>(j, "vocab_size", path_str);
    cfg.max_position_embeddings  = require<int>(j, "max_position_embeddings", path_str);

    cfg.rms_norm_eps = require<float>(j, "rms_norm_eps", path_str);
    cfg.hidden_act   = optional<std::string>(j, "hidden_act", "silu");

    cfg.rope_theta       = optional<float>(j, "rope_theta", 10000.0f);
    cfg.has_rope_scaling = j.contains("rope_scaling") && !j["rope_scaling"].is_null();

    // YaRN (Llama-3 style). Defaults match Llama-3.2-1B / 3-8B.
    cfg.rope_factor              = 1.0f;
    cfg.rope_low_freq_factor     = 1.0f;
    cfg.rope_high_freq_factor    = 4.0f;
    cfg.rope_original_max_position = cfg.max_position_embeddings;
    if (cfg.has_rope_scaling) {
        const auto& s = j["rope_scaling"];
        cfg.rope_factor              = optional<float>(s, "factor", 8.0f);
        cfg.rope_low_freq_factor     = optional<float>(s, "low_freq_factor", 1.0f);
        cfg.rope_high_freq_factor    = optional<float>(s, "high_freq_factor", 4.0f);
        cfg.rope_original_max_position =
            optional<int>(s, "original_max_position_embeddings",
                          cfg.max_position_embeddings);
    }

    cfg.sliding_window      = optional<int>(j, "sliding_window", -1);
    cfg.rope_local_base_freq = optional<float>(j, "rope_local_base_freq", 0.f);

    // Per-layer attention type. Three sources:
    //   1. Explicit `layer_types` array (OLMo-3, some HF Gemma-3 dumps).
    //   2. Gemma-3 `sliding_window_pattern`: every N-th layer is full;
    //      others are sliding.
    //   3. Gemma-2's hardcoded `i%2==0` pattern (HF `modeling_gemma2.py`)
    //      — synthesized here since the JSON doesn't carry it.
    if (j.contains("layer_types") && j["layer_types"].is_array()) {
        for (const auto& t : j["layer_types"]) {
            cfg.layer_types.push_back(t.get<std::string>());
        }
    } else if ((cfg.model_type == "gemma3" || cfg.model_type == "gemma3_text") &&
               j.contains("sliding_window_pattern")) {
        const int pat = j["sliding_window_pattern"].get<int>();
        cfg.layer_types.reserve(static_cast<std::size_t>(cfg.num_hidden_layers));
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            // Gemma-3: every `sliding_window_pattern`-th layer is full
            // attention; the others are sliding. HF puts the full layer
            // at the *end* of each pattern block (`(i+1) % pat == 0`).
            cfg.layer_types.push_back(((i + 1) % pat == 0)
                                          ? "full_attention"
                                          : "sliding_attention");
        }
    } else if (cfg.model_type == "gemma2" && cfg.sliding_window > 0) {
        // Gemma-2 hardcodes alternation in `modeling_gemma2.py`:
        //     is_sliding = (layer_idx % 2 == 0)
        cfg.layer_types.reserve(static_cast<std::size_t>(cfg.num_hidden_layers));
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            cfg.layer_types.push_back((i % 2 == 0) ? "sliding_attention"
                                                   : "full_attention");
        }
    }

    // HF's Gemma{2,3}TextConfig defaults `tie_word_embeddings=True` and
    // omits the field from the published config.json. Other arches in
    // our supported set ship the field explicitly when they want
    // tying, so a default of false is the right call there. Applying
    // the right per-arch default here saves a "lm_head missing" load
    // failure on every Gemma checkpoint.
    const bool tie_default =
        cfg.model_type == "gemma" || cfg.model_type == "gemma2" ||
        cfg.model_type == "gemma3" || cfg.model_type == "gemma3_text" ||
        cfg.model_type == "gemma4";
    cfg.tie_word_embeddings = optional<bool>(j, "tie_word_embeddings", tie_default);
    // `attention_bias` is what Llama-3 / Qwen-3 use to flag bias on
    // QKV. Qwen-2 always ships biased QKV but its HF config omits the
    // flag — default to true for that model_type.
    cfg.attention_bias      = optional<bool>(j, "attention_bias",
                                             cfg.model_type == "qwen2");
    cfg.use_qk_norm         = infer_qk_norm(cfg.model_type, j);

    // Sparse MoE (zero on dense models). HF spelling: `num_local_experts`
    // and `num_experts_per_tok` for Mixtral / GPT-OSS; some configs use
    // `num_experts` instead.
    cfg.num_experts         = optional<int>(j, "num_local_experts",
                                            optional<int>(j, "num_experts", 0));
    cfg.num_experts_per_tok = optional<int>(j, "num_experts_per_tok", 0);

    // Gemma-family knobs. Both are "missing means default" — Gemma's
    // attention scale defaults to `head_dim` (i.e. `1/sqrt(head_dim)`,
    // matching standard scaled-dot-product) and the final-logit soft-cap
    // defaults to 0 (no cap). HF stores `final_logit_softcapping` as
    // `null` when disabled; `optional<float>` skips the field in that
    // case and returns the default.
    cfg.gemma_query_pre_attn_scalar =
        optional<float>(j, "query_pre_attn_scalar",
                        static_cast<float>(cfg.head_dim));
    cfg.gemma_final_logit_softcap = 0.f;
    if (j.contains("final_logit_softcapping") && !j["final_logit_softcapping"].is_null()) {
        cfg.gemma_final_logit_softcap = j["final_logit_softcapping"].get<float>();
    }
    cfg.gemma_attn_logit_softcap = 0.f;
    if (j.contains("attn_logit_softcapping") && !j["attn_logit_softcapping"].is_null()) {
        cfg.gemma_attn_logit_softcap = j["attn_logit_softcapping"].get<float>();
    }

    cfg.torch_dtype = optional<std::string>(j, "torch_dtype", "bfloat16");

    return cfg;
}

}  // namespace pie_cuda_driver
