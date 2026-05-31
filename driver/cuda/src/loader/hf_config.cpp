#include "loader/hf_config.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <pie_driver_common/hf_config_json.hpp>

namespace pie_cuda_driver {

namespace {

template <typename T>
T require(const nlohmann::json& j, const char* key, const std::string& path) {
    return pie_driver_common::json_require<T>(j, key, path);
}

template <typename T>
T optional(const nlohmann::json& j, const char* key, T default_value) {
    return pie_driver_common::json_get_or<T>(j, key, default_value);
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

    nlohmann::json j_root;
    in >> j_root;
    const auto path_str = path.string();

    HfConfig cfg;

    if (!j_root.contains("architectures") || !j_root["architectures"].is_array() || j_root["architectures"].empty()) {
        throw std::runtime_error("config.json: missing or empty 'architectures'");
    }
    cfg.arch_name = j_root["architectures"][0].get<std::string>();

    // Gemma-4 (and other multimodal Gemma checkpoints) nest the text-
    // tower's hyperparameters under `text_config` while keeping the
    // architecture name + a stub `model_type=gemma4` at the top level.
    // Dereference once so the rest of the parser reads from the text
    // sub-config when present.
    const auto view = pie_driver_common::hf_config_json_view(j_root);
    const auto& j = view.text;
    cfg.model_type = view.text_or_outer_model_type();

    cfg.hidden_size              = require<int>(j, "hidden_size", path_str);
    // `intermediate_size` is normally a scalar, but Gemma-3n stores a
    // per-layer list. Accept either; populate
    // `gemma3n_per_layer_intermediate` from the list and mirror the
    // first element into the scalar field for back-compat.
    if (j.contains("intermediate_size") && j["intermediate_size"].is_array()) {
        const auto& arr = j["intermediate_size"];
        if (arr.empty()) {
            throw std::runtime_error(
                "config.json (" + path_str + "): intermediate_size list is empty");
        }
        cfg.gemma3n_per_layer_intermediate.reserve(arr.size());
        for (const auto& v : arr) {
            cfg.gemma3n_per_layer_intermediate.push_back(v.get<int>());
        }
        cfg.intermediate_size = cfg.gemma3n_per_layer_intermediate.front();
    } else {
        // qwen3_5_moe and similar pure-MoE configs omit `intermediate_size`
        // and use only `moe_intermediate_size` / `shared_expert_intermediate_size`.
        // Default the scalar to 0 in that case; the bind side reads the MoE-
        // specific fields directly.
        cfg.intermediate_size = optional<int>(j, "intermediate_size", 0);
    }
    cfg.num_hidden_layers        = require<int>(j, "num_hidden_layers", path_str);
    cfg.num_attention_heads      = require<int>(j, "num_attention_heads", path_str);
    cfg.num_key_value_heads      = optional<int>(j, "num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim                 = optional<int>(j, "head_dim", cfg.hidden_size / cfg.num_attention_heads);
    cfg.q_lora_rank              = optional<int>(j, "q_lora_rank", 0);
    cfg.kv_lora_rank             = optional<int>(j, "kv_lora_rank", 0);
    cfg.qk_nope_head_dim         = optional<int>(j, "qk_nope_head_dim", 0);
    cfg.qk_rope_head_dim         = optional<int>(j, "qk_rope_head_dim", 0);
    cfg.v_head_dim               = optional<int>(j, "v_head_dim", 0);
    if ((cfg.model_type == "kimi_k2" || cfg.model_type == "deepseek_v2" ||
         cfg.model_type == "deepseek_v3" || cfg.model_type == "glm_moe_dsa") &&
        cfg.qk_nope_head_dim > 0 && cfg.qk_rope_head_dim > 0) {
        // MLA attention has a query/key width that is independent from the
        // value width and from hidden_size / num_heads. Keep `head_dim` as
        // the QK width so RoPE/attention capability checks see the right
        // logical dimension; `v_head_dim` carries the output-value width.
        cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    }

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

    cfg.rms_norm_eps = optional<float>(
        j, "rms_norm_eps",
        optional<float>(j, "layer_norm_epsilon",
                        optional<float>(j, "norm_eps", 1e-5f)));
    cfg.hidden_act   = optional<std::string>(j, "hidden_act", "silu");
    cfg.mlp_hidden_act = optional<std::string>(j, "mlp_hidden_act", cfg.hidden_act);

    cfg.rope_theta       = optional<float>(j, "rope_theta", 10000.0f);

    // Two YaRN variants are recognised:
    //   * Llama-3 YaRN (`rope_type == "llama3"` or `low_freq_factor`
    //     present) — smoothed interpolation in the wavelen domain.
    //   * Original YaRN (`rope_type == "yarn"` with `beta_fast` /
    //     `beta_slow` / `attention_factor`) — linear ramp in the
    //     dim-index domain. Used by OLMo-3, gpt-oss, DeepSeek-V3.
    // Defaults are inert (`scaling_kind = None`) so plain-RoPE
    // ckpts (Mistral / Qwen) take the un-scaled path.
    cfg.rope_factor              = 1.0f;
    cfg.rope_low_freq_factor     = 1.0f;
    cfg.rope_high_freq_factor    = 4.0f;
    cfg.rope_beta_fast           = 32.0f;
    cfg.rope_beta_slow           = 1.0f;
    cfg.rope_attention_factor    = 1.0f;
    cfg.rope_original_max_position = cfg.max_position_embeddings;
    cfg.rope_scaling_kind        = HfConfig::RopeScaling::None;
    cfg.has_rope_scaling         = false;
    if (const auto* rope_cfg = pie_driver_common::flat_rope_config_view(j)) {
        const auto& s = *rope_cfg;
        cfg.rope_theta = optional<float>(s, "rope_theta", cfg.rope_theta);
        const std::string rope_type = optional<std::string>(
            s, "rope_type", optional<std::string>(s, "type", ""));
        const bool has_llama3_keys = s.contains("low_freq_factor") ||
                                     s.contains("high_freq_factor");
        if (rope_type == "llama3" || has_llama3_keys) {
            cfg.rope_scaling_kind = HfConfig::RopeScaling::Llama3;
            cfg.has_rope_scaling  = true;
            cfg.rope_factor              = optional<float>(s, "factor", 8.0f);
            cfg.rope_low_freq_factor     = optional<float>(s, "low_freq_factor", 1.0f);
            cfg.rope_high_freq_factor    = optional<float>(s, "high_freq_factor", 4.0f);
            cfg.rope_original_max_position =
                optional<int>(s, "original_max_position_embeddings",
                              cfg.max_position_embeddings);
        } else if (rope_type == "yarn") {
            cfg.rope_scaling_kind = HfConfig::RopeScaling::OriginalYaRN;
            cfg.has_rope_scaling  = true;
            cfg.rope_factor           = optional<float>(s, "factor", 1.0f);
            cfg.rope_beta_fast        = optional<float>(s, "beta_fast", 32.0f);
            cfg.rope_beta_slow        = optional<float>(s, "beta_slow", 1.0f);
            // DeepSeek/Kimi models use `mscale_all_dim` (typically 1.0)
            // as the attention factor. OLMo-3 uses `attention_factor`
            // directly. Fall back to `0.1 * ln(factor) + 1` if neither
            // is present.
            const float mscale_all_dim =
                optional<float>(s, "mscale_all_dim", 0.0f);
            const float default_mscale = mscale_all_dim > 0.f
                ? mscale_all_dim
                : (cfg.rope_factor > 1.f
                    ? 0.1f * std::log(cfg.rope_factor) + 1.0f
                    : 1.0f);
            cfg.rope_attention_factor = optional<float>(
                s, "attention_factor", default_mscale);
            cfg.rope_original_max_position =
                optional<int>(s, "original_max_position_embeddings",
                              cfg.max_position_embeddings);
        }
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
    } else if (cfg.model_type == "nemotron_h" &&
               j.contains("hybrid_override_pattern")) {
        const std::string pattern =
            j["hybrid_override_pattern"].get<std::string>();
        if (static_cast<int>(pattern.size()) != cfg.num_hidden_layers) {
            throw std::runtime_error(
                "config.json (" + path_str +
                "): hybrid_override_pattern size != num_hidden_layers");
        }
        cfg.layer_types.reserve(pattern.size());
        for (char c : pattern) {
            if (c == 'M') {
                cfg.layer_types.push_back("mamba");
            } else if (c == '*') {
                cfg.layer_types.push_back("attention");
            } else if (c == 'E') {
                cfg.layer_types.push_back("moe");
            } else if (c == '-') {
                cfg.layer_types.push_back("mlp");
            } else {
                throw std::runtime_error(
                    "config.json (" + path_str +
                    "): unsupported hybrid_override_pattern character");
            }
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
        cfg.model_type == "gemma"  || cfg.model_type == "gemma2" ||
        cfg.model_type == "gemma3" || cfg.model_type == "gemma3_text" ||
        cfg.model_type == "gemma4" || cfg.model_type == "gemma4_text" ||
        cfg.model_type == "gemma3n" || cfg.model_type == "gemma3n_text";
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
    cfg.num_experts         = optional<int>(
        j, "num_local_experts",
        optional<int>(j, "num_experts",
                      optional<int>(j, "n_routed_experts", 0)));
    // `num_experts_per_tok` is the canonical name (Mixtral / Qwen MoE);
    // Gemma-4 uses `top_k_experts`. Accept either.
    cfg.num_experts_per_tok = optional<int>(j, "num_experts_per_tok",
                                            optional<int>(j, "top_k_experts", 0));
    cfg.first_k_dense_replace =
        optional<int>(j, "first_k_dense_replace", 0);
    cfg.n_shared_experts =
        optional<int>(j, "n_shared_experts", 0);
    cfg.norm_topk_prob =
        optional<bool>(j, "norm_topk_prob", false);
    cfg.routed_scaling_factor =
        optional<float>(j, "routed_scaling_factor", 1.0f);

    // ── DeepSeek V4 specific ────────────────────────────────────────
    if (cfg.model_type == "deepseek_v4") {
        cfg.dsv4_o_lora_rank       = optional<int>(j, "o_lora_rank", 0);
        cfg.dsv4_o_groups          = optional<int>(j, "o_groups", 0);
        cfg.dsv4_index_head_dim    = optional<int>(j, "index_head_dim", 0);
        cfg.dsv4_index_n_heads     = optional<int>(j, "index_n_heads", 0);
        cfg.dsv4_index_topk        = optional<int>(j, "index_topk", 0);
        cfg.dsv4_hc_mult           = optional<int>(j, "hc_mult", 0);
        cfg.dsv4_hc_eps            = optional<float>(j, "hc_eps", 1e-6f);
        cfg.dsv4_num_hash_layers   = optional<int>(j, "num_hash_layers", 0);
        cfg.dsv4_sliding_window    = optional<int>(j, "sliding_window", 0);
        cfg.dsv4_compress_rope_theta = optional<float>(j, "compress_rope_theta", 0.f);
        cfg.dsv4_scoring_func      = optional<std::string>(j, "scoring_func", "");
        cfg.dsv4_expert_dtype      = optional<std::string>(j, "expert_dtype", "");
        cfg.attention_has_sinks    = true;
        if (j.contains("compress_ratios") && j["compress_ratios"].is_array()) {
            for (const auto& v : j["compress_ratios"]) {
                cfg.dsv4_compress_ratios.push_back(v.get<int>());
            }
        }
    }

    // Gemma-4 26B-A4B sets `enable_moe_block: true` to flip its layers
    // from dense-MLP-only to dense + parallel MoE.
    cfg.gemma4_enable_moe   = optional<bool>(j, "enable_moe_block", false);
    // Gemma-4 26B-A4B's "k_eq_v" full-attention path. Defaults inert
    // on all the dense Gemma-4 / Gemma-3n / unrelated archs.
    cfg.gemma4_attention_k_eq_v = optional<bool>(j, "attention_k_eq_v", false);
    cfg.gemma4_num_global_key_value_heads =
        optional<int>(j, "num_global_key_value_heads",
                      cfg.num_key_value_heads);

    // GPT-OSS knobs. The flags are inferred from `model_type` (rather
    // than from explicit fields) because HF's gpt_oss config doesn't
    // carry them — the architecture itself encodes them. `swiglu_limit`
    // is set explicitly in the config as the activation clip threshold.
    cfg.swiglu_limit         = optional<float>(j, "swiglu_limit", 0.f);
    cfg.mlp_has_bias         = (cfg.model_type == "gpt_oss");
    cfg.router_has_bias      = (cfg.model_type == "gpt_oss");
    cfg.attention_has_sinks  = (cfg.model_type == "gpt_oss");

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
    cfg.gemma_hidden_size_per_layer_input =
        optional<int>(j, "hidden_size_per_layer_input", 0);
    cfg.num_kv_shared_layers = optional<int>(j, "num_kv_shared_layers", 0);
    cfg.gemma4_use_ordered_embeddings =
        optional<bool>(j_root, "use_ordered_embeddings",
                       optional<bool>(j, "use_ordered_embeddings", false));
    cfg.gemma4_num_centroids =
        optional<int>(j_root, "num_centroids",
                      optional<int>(j, "num_centroids", 0));
    cfg.gemma4_centroid_intermediate_top_k =
        optional<int>(j_root, "centroid_intermediate_top_k",
                      optional<int>(j, "centroid_intermediate_top_k", 0));

    // Gemma-4 nests RoPE settings under `rope_parameters` keyed by
    // attention type. Each entry has `rope_theta` and (full only)
    // `partial_rotary_factor`. We pre-expand into per-layer vectors
    // so the forward doesn't have to consult layer_types twice.
    if (j.contains("rope_parameters") && j["rope_parameters"].is_object()
            && !cfg.layer_types.empty()) {
        const auto& rope_params = j["rope_parameters"];
        cfg.gemma_per_layer_rope_theta.assign(
            cfg.layer_types.size(), cfg.rope_theta);
        cfg.gemma_per_layer_partial_rotary_factor.assign(
            cfg.layer_types.size(), 1.0f);
        for (std::size_t i = 0; i < cfg.layer_types.size(); ++i) {
            const auto& t = cfg.layer_types[i];
            if (rope_params.contains(t) && rope_params[t].is_object()) {
                const auto& e = rope_params[t];
                cfg.gemma_per_layer_rope_theta[i] =
                    optional<float>(e, "rope_theta", cfg.rope_theta);
                cfg.gemma_per_layer_partial_rotary_factor[i] =
                    optional<float>(e, "partial_rotary_factor", 1.0f);
            }
        }
    }

    // Qwen3.6-MoE knobs (zero on non-MoE archs).
    cfg.moe_intermediate_size =
        optional<int>(j, "moe_intermediate_size", 0);
    // shared_expert_intermediate_size: explicit field, then the
    // `moe_shared_expert_intermediate_size` alias, then (DeepSeek/GLM
    // style) computed as n_shared_experts × moe_intermediate_size.
    cfg.shared_expert_intermediate_size = optional<int>(
        j, "shared_expert_intermediate_size",
        optional<int>(j, "moe_shared_expert_intermediate_size", 0));
    if (cfg.shared_expert_intermediate_size == 0 &&
        cfg.n_shared_experts > 0 &&
        cfg.moe_intermediate_size > 0) {
        cfg.shared_expert_intermediate_size =
            cfg.n_shared_experts * cfg.moe_intermediate_size;
    }
    // `routed_scaling_factor` and `norm_topk_prob` are set earlier
    // (with the MoE expert-count fields). Only the group-routing knobs
    // are read here.
    cfg.n_group = optional<int>(j, "n_group",
                                optional<int>(j, "n_groups", 1));
    cfg.topk_group = optional<int>(j, "topk_group", 1);

    if (cfg.model_type == "nemotron_h") {
        cfg.mamba_num_heads =
            optional<int>(j, "mamba_num_heads", 0);
        cfg.mamba_head_dim =
            optional<int>(j, "mamba_head_dim", 0);
        cfg.mamba_state_size =
            optional<int>(j, "ssm_state_size", 0);
        cfg.mamba_n_groups =
            optional<int>(j, "n_groups",
                          optional<int>(j, "mamba_n_groups", 0));
        cfg.mamba_conv_kernel =
            optional<int>(j, "conv_kernel",
                          optional<int>(j, "mamba_d_conv", 0));
        cfg.mamba_chunk_size = optional<int>(j, "chunk_size", 0);
        cfg.mamba_time_step_min =
            optional<float>(j, "time_step_min",
                            optional<float>(j, "mamba_dt_min", 0.001f));
    }

    // Qwen3.5 hybrid (linear-attention SSM) knobs. Defaults are zero so
    // non-qwen3.5 models leave the linear-attn dimensions unset; the
    // bind side branches on `linear_num_value_heads > 0`.
    cfg.linear_num_value_heads  = optional<int>(j, "linear_num_value_heads", 0);
    cfg.linear_num_key_heads    = optional<int>(j, "linear_num_key_heads", 0);
    cfg.linear_key_head_dim     = optional<int>(j, "linear_key_head_dim", 0);
    cfg.linear_value_head_dim   = optional<int>(j, "linear_value_head_dim", 0);
    cfg.linear_conv_kernel_dim  = optional<int>(j, "linear_conv_kernel_dim", 0);
    cfg.attn_output_gate        = optional<bool>(j, "attn_output_gate",
                                                 cfg.model_type == "qwen3_5" ||
                                                 cfg.model_type == "qwen3_5_text");

    // Partial RoPE (Qwen3.5). HF stores it under `rope_parameters` for
    // qwen3_5 (single dict, not per-layer-type) and at the top level for
    // some other models. Defaults to 1.0 (full rotation) for everything
    // else.
    cfg.partial_rotary_factor = 1.0f;
    if (const auto* rp = pie_driver_common::flat_rope_parameters_view(j);
        rp != nullptr && rp->contains("partial_rotary_factor")) {
        cfg.partial_rotary_factor =
            (*rp)["partial_rotary_factor"].get<float>();
    } else {
        cfg.partial_rotary_factor =
            optional<float>(j, "partial_rotary_factor", 1.0f);
    }
    // Qwen3.5 stores `rope_theta` under `rope_parameters` rather than
    // at the top level. Use that as the source of truth when present.
    if (const auto* rp = pie_driver_common::flat_rope_parameters_view(j);
        rp != nullptr && rp->contains("rope_theta")) {
        if ((*rp)["rope_theta"].is_number()) {
            cfg.rope_theta = (*rp)["rope_theta"].get<float>();
        }
    }
    cfg.mtp_num_hidden_layers = optional<int>(j, "mtp_num_hidden_layers", 0);
    cfg.mtp_use_dedicated_embeddings =
        optional<bool>(j, "mtp_use_dedicated_embeddings", false);

    // Gemma-3n knobs. Defaults match HF's GptOssConfig defaults so non-
    // gemma3n models leave them inert (laurel_rank=0 disables Laurel,
    // altup_num_inputs=0 disables AltUp from the bind side).
    cfg.altup_num_inputs    = optional<int>(j, "altup_num_inputs", 0);
    cfg.altup_active_idx    = optional<int>(j, "altup_active_idx", 0);
    cfg.altup_correct_scale = optional<bool>(j, "altup_correct_scale", false);
    cfg.altup_coef_clip     = optional<float>(j, "altup_coef_clip", 0.f);
    cfg.laurel_rank         = optional<int>(j, "laurel_rank", 0);
    cfg.vocab_size_per_layer_input =
        optional<int>(j, "vocab_size_per_layer_input", 0);
    cfg.gemma3n_rope_local_base_freq =
        optional<float>(j, "rope_local_base_freq", cfg.rope_local_base_freq);
    if (j.contains("activation_sparsity_pattern") &&
        j["activation_sparsity_pattern"].is_array()) {
        for (const auto& v : j["activation_sparsity_pattern"]) {
            cfg.gemma3n_activation_sparsity.push_back(v.get<float>());
        }
    }

    // GLM-5.1 DSA indexer fields. Inert (zero/empty) on every other model.
    cfg.index_topk     = optional<int>(j, "index_topk", 0);
    cfg.index_head_dim = optional<int>(j, "index_head_dim", 0);
    cfg.index_n_heads  = optional<int>(j, "index_n_heads", 0);
    if (j.contains("indexer_types") && j["indexer_types"].is_array()) {
        for (const auto& v : j["indexer_types"]) {
            cfg.indexer_types.push_back(v.get<std::string>());
        }
    } else if (cfg.index_topk > 0 && cfg.num_hidden_layers > 0) {
        const int freq = optional<int>(j, "index_topk_freq", 1);
        cfg.indexer_types.reserve(
            static_cast<std::size_t>(cfg.num_hidden_layers));
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            const bool is_full = (std::max(i - 1, 0) % freq) == 0;
            cfg.indexer_types.push_back(is_full ? "full" : "shared");
        }
    }

    cfg.torch_dtype = optional<std::string>(j, "torch_dtype", "bfloat16");

    // Multimodal text-tower extraction. The CUDA driver runs the LLM
    // forward only; for vision-language ckpts we strip the
    // "language_model." prefix from every weight and skip the vision-
    // side tensors entirely. Detection key is the top-level model_type:
    // mistral3 (Mistral-Small-3.1-FP8), llava (Llava-1.6), llava_next,
    // qwen2_5_vl, gemma3 (multimodal variant has vision_config), …
    const bool has_vision_config =
        j_root.contains("vision_config") && j_root["vision_config"].is_object();
    const bool is_kimi_k25_wrapper =
        view.outer_model_type == "kimi_k25" && j_root.contains("text_config");
    const bool is_multimodal_wrapper =
        (has_vision_config && j_root.contains("text_config")) ||
        is_kimi_k25_wrapper;
    if (is_multimodal_wrapper) {
        cfg.mm_lm_strip_prefix = "language_model.";
        cfg.mm_skip_prefixes = {
            "vision_tower.",
            "vision_model.",
            "visual.",
            "multi_modal_projector.",
            "mm_projector.",
        };
    }
    if (cfg.model_type == "nemotron_h") {
        cfg.mm_skip_prefixes = {
            "vision_model.",
            "mlp1.",
            "sound_encoder.",
            "sound_projection.",
        };
    }

    // ── Gemma-4 vision encoder (gemma4_vision) ──────────────────────
    // Parse vision_config + the top-level soft-token count for multimodal
    // Gemma-4 checkpoints. Parsed but not yet consumed — the encoder graph
    // (MULTIMODAL.md Phase 2.2) reads this. The vision/audio towers stay in
    // `mm_skip_prefixes` above until the encoder binds them. Confirmed against
    // `google/gemma-4-E4B`.
    const bool is_gemma4_vision =
        has_vision_config &&
        j_root["vision_config"].value("model_type", std::string()) == "gemma4_vision";
    if (is_gemma4_vision) {
        const auto& v = j_root["vision_config"];
        GemmaVisionConfig gv;
        gv.hidden_size         = optional<int>(v, "hidden_size", gv.hidden_size);
        gv.intermediate_size   = optional<int>(v, "intermediate_size", gv.intermediate_size);
        gv.num_hidden_layers   = optional<int>(v, "num_hidden_layers", gv.num_hidden_layers);
        gv.num_attention_heads = optional<int>(v, "num_attention_heads", gv.num_attention_heads);
        gv.num_key_value_heads = optional<int>(v, "num_key_value_heads", gv.num_attention_heads);
        gv.head_dim            = optional<int>(v, "head_dim", gv.head_dim);
        gv.patch_size          = optional<int>(v, "patch_size", gv.patch_size);
        gv.rms_norm_eps        = optional<float>(v, "rms_norm_eps", gv.rms_norm_eps);
        gv.pooling_kernel_size = optional<int>(v, "pooling_kernel_size", gv.pooling_kernel_size);
        gv.use_clipped_linears = optional<bool>(v, "use_clipped_linears", gv.use_clipped_linears);
        if (v.contains("rope_parameters") && v["rope_parameters"].is_object()) {
            gv.rope_theta = optional<float>(v["rope_parameters"], "rope_theta", gv.rope_theta);
        }
        gv.soft_tokens_per_image =
            optional<int>(j_root, "vision_soft_tokens_per_image", gv.soft_tokens_per_image);
        cfg.gemma_vision = gv;
    }

    // ── Gemma-4 audio encoder (gemma4_audio) ────────────────────────
    // Parse audio_config for multimodal Gemma-4 checkpoints. The audio tower
    // (`model.audio_tower.*` + `model.embed_audio.*`) loads alongside vision;
    // the gemma4 audio encoder graph reads this. Confirmed against
    // `google/gemma-4-E4B`. See audio_frontend.md.
    const bool has_audio_config =
        j_root.contains("audio_config") && j_root["audio_config"].is_object();
    const bool is_gemma4_audio =
        has_audio_config &&
        j_root["audio_config"].value("model_type", std::string()) == "gemma4_audio";
    if (is_gemma4_audio) {
        const auto& a = j_root["audio_config"];
        GemmaAudioConfig ga;
        ga.hidden_size           = optional<int>(a, "hidden_size", ga.hidden_size);
        ga.num_attention_heads   = optional<int>(a, "num_attention_heads", ga.num_attention_heads);
        ga.num_hidden_layers     = optional<int>(a, "num_hidden_layers", ga.num_hidden_layers);
        ga.conv_kernel_size      = optional<int>(a, "conv_kernel_size", ga.conv_kernel_size);
        ga.output_proj_dims      = optional<int>(a, "output_proj_dims", ga.output_proj_dims);
        ga.attention_chunk_size  = optional<int>(a, "attention_chunk_size", ga.attention_chunk_size);
        ga.attention_context_left  = optional<int>(a, "attention_context_left", ga.attention_context_left);
        ga.attention_context_right = optional<int>(a, "attention_context_right", ga.attention_context_right);
        ga.feature_size          = optional<int>(a, "feature_size", ga.feature_size);
        ga.attention_logit_cap   = optional<float>(a, "attention_logit_cap", ga.attention_logit_cap);
        ga.residual_weight       = optional<float>(a, "residual_weight", ga.residual_weight);
        ga.rms_norm_eps          = optional<float>(a, "rms_norm_eps", ga.rms_norm_eps);
        ga.use_clipped_linears   = optional<bool>(a, "use_clipped_linears", ga.use_clipped_linears);
        if (a.contains("subsampling_conv_channels") &&
            a["subsampling_conv_channels"].is_array() &&
            a["subsampling_conv_channels"].size() >= 2) {
            ga.subsampling_conv_channels0 = a["subsampling_conv_channels"][0].get<int>();
            ga.subsampling_conv_channels1 = a["subsampling_conv_channels"][1].get<int>();
        }
        cfg.gemma_audio = ga;
    }

    // ── Qwen3-VL vision encoder (qwen3_vl) ──────────────────────────
    // Parse the vision_config + the text M-RoPE params + delimiter token ids
    // for multimodal Qwen3-VL checkpoints. The text tower (model_type
    // "qwen3_vl_text") is standard Qwen3; the Qwen3VLModel consumes these.
    const bool is_qwen3_vl_vision =
        has_vision_config &&
        j_root["vision_config"].value("model_type", std::string()) == "qwen3_vl";
    if (is_qwen3_vl_vision) {
        const auto& v = j_root["vision_config"];
        Qwen3VLVisionConfig qv;
        qv.hidden_size          = optional<int>(v, "hidden_size", qv.hidden_size);
        qv.intermediate_size    = optional<int>(v, "intermediate_size", qv.intermediate_size);
        qv.depth                = optional<int>(v, "depth", qv.depth);
        qv.num_heads            = optional<int>(v, "num_heads", qv.num_heads);
        qv.patch_size           = optional<int>(v, "patch_size", qv.patch_size);
        qv.temporal_patch_size  = optional<int>(v, "temporal_patch_size", qv.temporal_patch_size);
        qv.spatial_merge_size   = optional<int>(v, "spatial_merge_size", qv.spatial_merge_size);
        qv.in_channels          = optional<int>(v, "in_channels", qv.in_channels);
        qv.out_hidden_size      = optional<int>(v, "out_hidden_size", qv.out_hidden_size);
        qv.num_position_embeddings =
            optional<int>(v, "num_position_embeddings", qv.num_position_embeddings);
        if (v.contains("deepstack_visual_indexes") &&
            v["deepstack_visual_indexes"].is_array()) {
            qv.deepstack_visual_indexes.clear();
            for (const auto& x : v["deepstack_visual_indexes"]) {
                qv.deepstack_visual_indexes.push_back(x.get<int>());
            }
        }
        cfg.qwen3_vl_vision = qv;

        // Text tower is Qwen3 (per-head q/k norm) — config has no explicit
        // `use_qk_norm`, so set it here (the tensors carry q_norm/k_norm).
        cfg.use_qk_norm = true;

        // M-RoPE: text rope_scaling holds the section split + interleave flag.
        if (j.contains("rope_scaling") && j["rope_scaling"].is_object()) {
            const auto& rs = j["rope_scaling"];
            if (rs.contains("mrope_section") && rs["mrope_section"].is_array()) {
                for (const auto& x : rs["mrope_section"]) {
                    cfg.qwen3_vl_mrope_section.push_back(x.get<int>());
                }
            }
            cfg.qwen3_vl_mrope_interleaved =
                optional<bool>(rs, "mrope_interleaved", false);
        }
        // M-RoPE is not a frequency-scaling variant — keep plain RoPE math.
        cfg.rope_scaling_kind = HfConfig::RopeScaling::None;
        cfg.has_rope_scaling = false;

        cfg.qwen3_vl_image_token_id        = optional<int>(j_root, "image_token_id", -1);
        cfg.qwen3_vl_vision_start_token_id = optional<int>(j_root, "vision_start_token_id", -1);
        cfg.qwen3_vl_vision_end_token_id   = optional<int>(j_root, "vision_end_token_id", -1);
    }

    // ── quantization_config ─────────────────────────────────────────
    // GPTQ / AWQ checkpoints attach a `quantization_config` block. We
    // read just enough to drive marlin dispatch — the loader can match
    // GPTQ's tensor names independently of this block, but the block
    // tells us which layout to expect (sym/asym, group_size, desc_act).
    //
    // compressed-tensors style FP8 (RedHatAI mistral3-FP8) is also
    // tagged here with `quant_method = "compressed-tensors"`; the FP8
    // path doesn't currently use these fields and treats it like an
    // un-tagged ckpt. M1 follow-up.
    // Mistral3 / Llava-style multimodal configs put `quantization_config`
    // at the top level (alongside `text_config` / `vision_config`). The
    // text-only branch reads it from `j` (= text_config); fall back to the
    // root when it's only present there.
    const nlohmann::json* qcfg_src = nullptr;
    if (j.contains("quantization_config") && j["quantization_config"].is_object()) {
        qcfg_src = &j["quantization_config"];
    } else if (j_root.contains("quantization_config") && j_root["quantization_config"].is_object()) {
        qcfg_src = &j_root["quantization_config"];
    }
    // ── CSM native audio output (model_type == "csm") ──────────────────
    // The backbone Llama hyperparameters are already parsed from the top level
    // above (hidden 2048, 16 layers, etc.). Here we lift the two nested pieces:
    // `depth_decoder_config` (the RVQ depth sampler) and `codec_config` (Mimi
    // codes→waveform). See AUDIO_OUTPUT.md.
    if (cfg.model_type == "csm") {
        CsmConfig csm;
        csm.text_vocab_size = optional<int>(j_root, "text_vocab_size", csm.text_vocab_size);
        csm.audio_vocab_size = cfg.vocab_size;  // top-level vocab_size = per-codebook 2051
        csm.num_codebooks = optional<int>(j_root, "num_codebooks", csm.num_codebooks);
        csm.codebook_eos_token_id = optional<int>(j_root, "codebook_eos_token_id", csm.codebook_eos_token_id);
        csm.audio_eos_token_id = optional<int>(j_root, "audio_eos_token_id", csm.audio_eos_token_id);
        csm.audio_token_id = optional<int>(j_root, "audio_token_id", csm.audio_token_id);

        if (j_root.contains("depth_decoder_config") &&
            j_root["depth_decoder_config"].is_object()) {
            const auto& d = j_root["depth_decoder_config"];
            auto& dd = csm.depth;
            dd.hidden_size          = optional<int>(d, "hidden_size", dd.hidden_size);
            dd.backbone_hidden_size = optional<int>(d, "backbone_hidden_size", dd.backbone_hidden_size);
            dd.num_hidden_layers    = optional<int>(d, "num_hidden_layers", dd.num_hidden_layers);
            dd.num_attention_heads  = optional<int>(d, "num_attention_heads", dd.num_attention_heads);
            dd.num_key_value_heads  = optional<int>(d, "num_key_value_heads", dd.num_key_value_heads);
            dd.head_dim             = optional<int>(d, "head_dim", dd.head_dim);
            dd.intermediate_size    = optional<int>(d, "intermediate_size", dd.intermediate_size);
            dd.num_codebooks        = optional<int>(d, "num_codebooks", dd.num_codebooks);
            dd.vocab_size           = optional<int>(d, "vocab_size", dd.vocab_size);
            dd.max_position_embeddings = optional<int>(d, "max_position_embeddings", dd.max_position_embeddings);
            dd.rms_norm_eps         = optional<float>(d, "rms_norm_eps", dd.rms_norm_eps);
            dd.rope_theta           = optional<float>(d, "rope_theta", dd.rope_theta);
            if (d.contains("rope_scaling") && d["rope_scaling"].is_object()) {
                const auto& s = d["rope_scaling"];
                dd.rope_factor              = optional<float>(s, "factor", dd.rope_factor);
                dd.rope_low_freq_factor     = optional<float>(s, "low_freq_factor", dd.rope_low_freq_factor);
                dd.rope_high_freq_factor    = optional<float>(s, "high_freq_factor", dd.rope_high_freq_factor);
                dd.rope_original_max_position = optional<int>(s, "original_max_position_embeddings", dd.rope_original_max_position);
            }
        }

        if (j_root.contains("codec_config") &&
            j_root["codec_config"].is_object()) {
            const auto& c = j_root["codec_config"];
            auto& mc = csm.codec;
            mc.hidden_size            = optional<int>(c, "hidden_size", mc.hidden_size);
            mc.codebook_dim           = optional<int>(c, "vector_quantization_hidden_dimension", optional<int>(c, "codebook_dim", mc.codebook_dim));
            mc.codebook_size          = optional<int>(c, "codebook_size", mc.codebook_size);
            mc.num_quantizers         = optional<int>(c, "num_quantizers", mc.num_quantizers);
            mc.num_semantic_quantizers = optional<int>(c, "num_semantic_quantizers", mc.num_semantic_quantizers);
            mc.num_filters            = optional<int>(c, "num_filters", mc.num_filters);
            mc.xf_num_attention_heads = optional<int>(c, "num_attention_heads", mc.xf_num_attention_heads);
            mc.xf_num_key_value_heads = optional<int>(c, "num_key_value_heads", mc.xf_num_key_value_heads);
            mc.xf_head_dim            = optional<int>(c, "head_dim", mc.xf_head_dim);
            mc.xf_intermediate_size   = optional<int>(c, "intermediate_size", mc.xf_intermediate_size);
            mc.xf_num_hidden_layers   = optional<int>(c, "num_hidden_layers", mc.xf_num_hidden_layers);
            mc.xf_sliding_window      = optional<int>(c, "sliding_window", mc.xf_sliding_window);
            mc.xf_rope_theta          = optional<float>(c, "rope_theta", mc.xf_rope_theta);
            mc.norm_eps               = optional<float>(c, "norm_eps", mc.norm_eps);
            mc.sampling_rate          = optional<int>(c, "sampling_rate", mc.sampling_rate);
            mc.use_causal_conv        = optional<bool>(c, "use_causal_conv", mc.use_causal_conv);
            mc.upsample_groups        = optional<int>(c, "upsample_groups", mc.upsample_groups);
            mc.residual_kernel_size   = optional<int>(c, "residual_kernel_size", mc.residual_kernel_size);
            mc.kernel_size            = optional<int>(c, "kernel_size", mc.kernel_size);
            mc.last_kernel_size       = optional<int>(c, "last_kernel_size", mc.last_kernel_size);
            if (c.contains("upsampling_ratios") && c["upsampling_ratios"].is_array()) {
                mc.upsampling_ratios.clear();
                for (const auto& r : c["upsampling_ratios"]) mc.upsampling_ratios.push_back(r.get<int>());
            }
        }
        cfg.csm = csm;
    }

    if (qcfg_src) {
        const auto& q = *qcfg_src;
        cfg.quant_method = optional<std::string>(q, "quant_method", "");
        cfg.quant_bits = optional<int>(q, "bits", 0);
        cfg.quant_group_size = optional<int>(q, "group_size", 0);
        cfg.quant_desc_act = optional<bool>(q, "desc_act", false);
        cfg.quant_sym = optional<bool>(q, "sym", true);
        // GPTQ's `sym=true` implies no zero points; AWQ's `zero_point=true`
        // is set explicitly even when sym is unset. Treat them as
        // independent signals.
        cfg.quant_zero_point = optional<bool>(q, "zero_point", !cfg.quant_sym);
        // Compressed-tensors-style KV cache quantization. If non-null
        // the ckpt expects the runtime to store K/V in FP8/INT8 with a
        // per-tensor or per-token scale. We currently always use bf16
        // KV — surface the mismatch so the user knows quality may be
        // slightly off vs the calibrated reference.
        if (q.contains("kv_cache_scheme") && !q["kv_cache_scheme"].is_null()) {
            cfg.kv_cache_scheme_present = true;
        }
    }

    return cfg;
}

}  // namespace pie_cuda_driver
