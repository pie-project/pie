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
    if (j.contains("rope_scaling") && j["rope_scaling"].is_object()) {
        const auto& s = j["rope_scaling"];
        const std::string rope_type = optional<std::string>(s, "rope_type", "");
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
            cfg.rope_factor           = optional<float>(s, "factor", 1.0f);
            cfg.rope_beta_fast        = optional<float>(s, "beta_fast", 32.0f);
            cfg.rope_beta_slow        = optional<float>(s, "beta_slow", 1.0f);
            // HF's `_compute_yarn_parameters` sets the default mscale
            // to `0.1 * ln(factor) + 1` when `attention_factor` is
            // absent. OLMo-3 ships it explicitly (1.2079...).
            const float default_mscale = (cfg.rope_factor > 1.f)
                ? 0.1f * std::log(cfg.rope_factor) + 1.0f
                : 1.0f;
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
    cfg.num_experts         = optional<int>(j, "num_local_experts",
                                            optional<int>(j, "num_experts", 0));
    // `num_experts_per_tok` is the canonical name (Mixtral / Qwen MoE);
    // Gemma-4 uses `top_k_experts`. Accept either.
    cfg.num_experts_per_tok = optional<int>(j, "num_experts_per_tok",
                                            optional<int>(j, "top_k_experts", 0));
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
    cfg.shared_expert_intermediate_size =
        optional<int>(j, "shared_expert_intermediate_size", 0);

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

    cfg.torch_dtype = optional<std::string>(j, "torch_dtype", "bfloat16");

    // Multimodal text-tower extraction. The CUDA driver runs the LLM
    // forward only; for vision-language ckpts we strip the
    // "language_model." prefix from every weight and skip the vision-
    // side tensors entirely. Detection key is the top-level model_type:
    // mistral3 (Mistral-Small-3.1-FP8), llava (Llava-1.6), llava_next,
    // qwen2_5_vl, gemma3 (multimodal variant has vision_config), …
    const bool has_vision_config =
        j_root.contains("vision_config") && j_root["vision_config"].is_object();
    const bool is_multimodal_wrapper =
        has_vision_config && j_root.contains("text_config");
    if (is_multimodal_wrapper) {
        cfg.mm_lm_strip_prefix = "language_model.";
        cfg.mm_skip_prefixes = {
            "vision_tower.",
            "vision_model.",
            "multi_modal_projector.",
        };
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
