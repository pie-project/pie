#include "hf_config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace pie_portable_driver {

const char* pie_arch_name(PieArch a) {
    switch (a) {
        case PieArch::Qwen3:    return "qwen3";
        case PieArch::Qwen2:    return "qwen2";
        case PieArch::Llama3:   return "llama3";
        case PieArch::Gemma2:   return "gemma2";
        case PieArch::Gemma3:   return "gemma3";
        case PieArch::Gemma4:   return "gemma4";
        case PieArch::Gemma3n:  return "gemma3n";
        case PieArch::Mistral3: return "mistral3";
        case PieArch::Olmo3:    return "olmo3";
        case PieArch::GptOss:   return "gptoss";
        case PieArch::Phi3:     return "phi3";
        case PieArch::Mixtral:  return "mixtral";
        case PieArch::Qwen3Moe: return "qwen3_moe";
        case PieArch::Qwen3_5:  return "qwen3_5";
    }
    return "?";
}

PieArch hf_model_type_to_pie_arch(const std::string& hf_model_type) {
    // Mirrors `pie/src/pie_driver/hf_utils.py::HF_TO_PIE_ARCH`. Source-of-truth
    // is that file; keep this table in sync.
    if (hf_model_type == "qwen3")       return PieArch::Qwen3;
    if (hf_model_type == "qwen2")       return PieArch::Qwen2;
    if (hf_model_type == "llama")       return PieArch::Llama3;
    if (hf_model_type == "gemma2")        return PieArch::Gemma2;
    if (hf_model_type == "gemma3" ||
        hf_model_type == "gemma3_text")   return PieArch::Gemma3;
    if (hf_model_type == "gemma4" ||
        hf_model_type == "gemma4_text")   return PieArch::Gemma4;
    if (hf_model_type == "gemma3n" ||
        hf_model_type == "gemma3n_text")  return PieArch::Gemma3n;
    if (hf_model_type == "mistral")     return PieArch::Mistral3;
    if (hf_model_type == "mistral3")    return PieArch::Mistral3;
    if (hf_model_type == "olmo3" ||
        hf_model_type == "olmo2")       return PieArch::Olmo3;
    if (hf_model_type == "gpt_oss")     return PieArch::GptOss;
    if (hf_model_type == "gptoss")      return PieArch::GptOss;
    if (hf_model_type == "phi3")        return PieArch::Phi3;
    if (hf_model_type == "mixtral")     return PieArch::Mixtral;
    if (hf_model_type == "qwen3_moe")   return PieArch::Qwen3Moe;
    if (hf_model_type == "qwen3_5" ||
        hf_model_type == "qwen3_5_text" ||
        hf_model_type == "qwen3_5_moe" ||
        hf_model_type == "qwen3_5_moe_text") return PieArch::Qwen3_5;
    throw std::runtime_error(
        "hf_config: unsupported model_type '" + hf_model_type + "'");
}

namespace {

template <typename T>
std::optional<T> get_opt(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    return it->get<T>();
}

template <typename T>
T get_or(const nlohmann::json& j, const char* key, T def) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    return it->get<T>();
}

}  // namespace

Hparams parse_hf_config(const std::filesystem::path& config_json_path) {
    std::ifstream f(config_json_path);
    if (!f) {
        throw std::runtime_error(
            "hf_config: cannot open " + config_json_path.string());
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "hf_config: parse failed for " + config_json_path.string() +
            ": " + e.what());
    }

    Hparams h;

    // Multimodal wrappers (gemma4, mistral3, gemma3 multimodal) nest the
    // text decoder config under `text_config`. Resolve once: the rest of
    // this function reads from the unwrapped JSON. We still derive `arch`
    // from the *outer* model_type because the runtime catalogue keys on
    // it (e.g. "gemma4" → Gemma4ForConditionalGeneration wraps gemma4_text).
    const std::string outer_model_type =
        j.contains("model_type") && !j["model_type"].is_null()
            ? j["model_type"].get<std::string>() : std::string();
    nlohmann::json text = j;
    if (j.contains("text_config") && j["text_config"].is_object()) {
        text = j["text_config"];
    }

    h.hf_model_type =
        text.contains("model_type") && !text["model_type"].is_null()
            ? text["model_type"].get<std::string>()
            : outer_model_type;
    // Use the outer wrapper's model_type for arch dispatch when present;
    // it's how the wrapper architecture is declared. For Ministral 3 the
    // outer says "mistral3" which is the registered HF type.
    h.arch = hf_model_type_to_pie_arch(
        !outer_model_type.empty() ? outer_model_type : h.hf_model_type);
    h.torch_dtype =
        get_or<std::string>(text, "torch_dtype",
            get_or<std::string>(j, "torch_dtype",
                get_or<std::string>(j, "dtype", "float16")));

    // FUTURE — quantized HF checkpoint support.
    // The safetensors loader currently accepts only F32 / F16 / BF16
    // (plus the MXFP4 path bolted on for gpt-oss in build_gpt_oss_).
    // Some shipped checkpoints use other formats:
    //   - Ministral 3 Instruct (mistralai/Ministral-3-3B-Instruct-2512):
    //     fp8 weight format (`quantization_config.quant_method == "fp8"`,
    //     with `weight_scale_inv` + `activation_scale` companion tensors
    //     per linear). The Base variant is BF16 and works today; supporting
    //     Instruct would require an FP8-block dequant similar to MXFP4.
    //   - Other potentially-quantized arch checkpoints (GPTQ, AWQ, etc.)
    //     likewise unsupported. We ignore `quantization_config` here; the
    //     safetensors loader throws on the first unsupported dtype.

    // Per-arch defaults for fields HF's PretrainedConfig fills implicitly.
    // Gemma 3 4B+ multimodal checkpoints ship with a minimal text_config
    // (only hidden_size / intermediate_size / num_hidden_layers / etc.),
    // relying on Gemma3TextConfig class defaults. Inject those here.
    if (h.arch == PieArch::Gemma3) {
        auto fill = [&](const char* k, double v) {
            if (!text.contains(k) || text[k].is_null()) text[k] = v;
        };
        fill("num_attention_heads", 8);
        fill("num_key_value_heads", 4);
        fill("head_dim", 256);
        fill("vocab_size", 262208);
        fill("rms_norm_eps", 1e-6);
        fill("rope_theta", 1e6);
        fill("max_position_embeddings", 131072);
        fill("sliding_window", 4096);
        fill("rope_local_base_freq", 10000.0);
        fill("query_pre_attn_scalar", 256);
    }

    h.num_hidden_layers = text.at("num_hidden_layers").get<std::int32_t>();
    h.num_attention_heads = text.at("num_attention_heads").get<std::int32_t>();
    // Some configs default num_key_value_heads to num_attention_heads.
    h.num_key_value_heads =
        get_or<std::int32_t>(text, "num_key_value_heads", h.num_attention_heads);
    // Gemma 4 alternative-attention only.
    h.num_global_key_value_heads =
        get_or<std::int32_t>(text, "num_global_key_value_heads", 0);
    h.hidden_size = text.at("hidden_size").get<std::int32_t>();
    // Pure-MoE checkpoints (Qwen 3.6) omit `intermediate_size` because
    // every layer's FFN is the routed-expert path; the per-expert width
    // lives in `moe_intermediate_size`. Fall back to 0 there.
    //
    // Gemma 3n stores `intermediate_size` as a per-layer array (e.g.
    // [8192, 8192, …]). v1 supports the constant-across-layers case by
    // taking the first element; reject heterogeneous values until the
    // graph builder can route per-layer FFN widths.
    if (auto it = text.find("intermediate_size");
        it != text.end() && it->is_array()) {
        if (it->empty()) {
            h.intermediate_size = 0;
        } else {
            const auto first = (*it)[0].get<std::int32_t>();
            for (const auto& v : *it) {
                if (v.get<std::int32_t>() != first) {
                    throw std::runtime_error(
                        "hf_config: per-layer intermediate_size with mixed "
                        "values is not yet supported");
                }
            }
            h.intermediate_size = first;
        }
    } else {
        h.intermediate_size =
            get_or<std::int32_t>(text, "intermediate_size", 0);
    }
    h.vocab_size = text.at("vocab_size").get<std::int32_t>();
    h.max_position_embeddings =
        get_or<std::int32_t>(text, "max_position_embeddings", 4096);

    if (text.contains("head_dim") && !text["head_dim"].is_null()) {
        h.head_dim = text["head_dim"].get<std::int32_t>();
    } else {
        if (h.num_attention_heads <= 0) {
            throw std::runtime_error("hf_config: num_attention_heads is zero");
        }
        h.head_dim = h.hidden_size / h.num_attention_heads;
    }

    h.rms_norm_eps = get_or<float>(text, "rms_norm_eps", 1e-6f);
    h.rope_theta = get_or<float>(text, "rope_theta", 1e6f);
    h.rope_local_base_freq = get_or<float>(text, "rope_local_base_freq", 0.0f);
    // HF's PretrainedConfig defaults this to True; specific archs override
    // (Llama-3 base / Llama-3.1+ untie). Multimodal wrappers (Qwen 3.6 MoE,
    // Gemma 4 etc.) put the flag on the OUTER config — the inner
    // `text_config` doesn't carry it. Outer wins; otherwise default tied.
    if (j.contains("tie_word_embeddings") && !j["tie_word_embeddings"].is_null()) {
        h.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    } else {
        h.tie_word_embeddings = get_or<bool>(text, "tie_word_embeddings", true);
    }

    if (auto sw = get_opt<std::int32_t>(text, "sliding_window")) {
        h.sliding_window = sw;
    }
    h.use_sliding_window = get_or<bool>(text, "use_sliding_window", false);

    // RoPE scaling. Two HF schemas:
    //   - `rope_scaling` (older; olmo3, llama3.1, gpt-oss)
    //   - `rope_parameters` (newer; ministral3) — when a single dict, same
    //     shape as `rope_scaling`. Gemma4 uses a per-attention-type dict-of-
    //     dicts and is parsed in build_gemma4_().
    const nlohmann::json* rs_ptr = nullptr;
    auto rs_it = text.find("rope_scaling");
    if (rs_it != text.end() && !rs_it->is_null()) rs_ptr = &(*rs_it);
    if (!rs_ptr) {
        auto rp_it = text.find("rope_parameters");
        if (rp_it != text.end() && rp_it->is_object()) {
            // Skip per-attention-type dicts (gemma4-style); only consume a
            // flat dict.
            const auto& rp = *rp_it;
            const bool nested = !rp.empty() && rp.begin().value().is_object()
                && rp.begin().value().contains("rope_type");
            if (!nested) rs_ptr = &rp;
        }
    }
    if (rs_ptr) {
        h.has_rope_scaling = true;
        const auto& rs = *rs_ptr;
        // Newer HF configs use `rope_type`; older ones use `type`.
        h.rope_scaling_type =
            get_or<std::string>(rs, "rope_type",
                                get_or<std::string>(rs, "type", ""));
        h.rope_scaling_factor = get_or<float>(rs, "factor", 1.0f);
        h.rope_scaling_low_freq_factor =
            get_or<float>(rs, "low_freq_factor", 1.0f);
        h.rope_scaling_high_freq_factor =
            get_or<float>(rs, "high_freq_factor", 4.0f);
        h.rope_scaling_original_max_position =
            get_or<std::int32_t>(rs, "original_max_position_embeddings", 0);
        // Ministral 3's rope_parameters carries `rope_theta` (overrides the
        // top-level rope_theta when present). Olmo3 keeps rope_theta at the
        // outer scope; this branch is harmless there.
        if (rs.contains("rope_theta") && !rs["rope_theta"].is_null()) {
            h.rope_theta = rs["rope_theta"].get<float>();
        }
        // YaRN-specific.
        h.rope_yarn_attention_factor =
            get_or<float>(rs, "attention_factor", 0.0f);
        h.rope_yarn_beta_fast = get_or<float>(rs, "beta_fast", 32.0f);
        h.rope_yarn_beta_slow = get_or<float>(rs, "beta_slow", 1.0f);
    }

    // ── Gemma 4 specifics ──
    // Gemma 3n shares the PLE dimensions and num_kv_shared_layers fields
    // (same HF config keys), so reuse the same hparams. The Gemma4-only
    // bits (proportional RoPE, layer_scalar, double-wide MLP, dual head_dim)
    // are gated below on arch == Gemma4.
    if (h.arch == PieArch::Gemma4 || h.arch == PieArch::Gemma3n) {
        h.gemma4_num_kv_shared_layers =
            get_or<std::int32_t>(text, "num_kv_shared_layers", 0);
        h.gemma4_ple_dim =
            get_or<std::int32_t>(text, "hidden_size_per_layer_input", 0);
        h.gemma4_ple_vocab =
            get_or<std::int32_t>(text, "vocab_size_per_layer_input", 0);
    }
    if (h.arch == PieArch::Gemma3n) {
        h.altup_num_inputs    = get_or<std::int32_t>(text, "altup_num_inputs", 4);
        h.altup_active_idx    = get_or<std::int32_t>(text, "altup_active_idx", 0);
        h.altup_correct_scale = get_or<bool>(text, "altup_correct_scale", true);
        h.laurel_rank         = get_or<std::int32_t>(text, "laurel_rank", 0);
        if (auto it = text.find("activation_sparsity_pattern");
            it != text.end() && it->is_array()) {
            for (const auto& v : *it) {
                if (v.is_number()) {
                    h.activation_sparsity_pattern.push_back(v.get<float>());
                }
            }
        }
    }
    if (h.arch == PieArch::Gemma4) {
        h.gemma4_head_dim_global =
            get_or<std::int32_t>(text, "global_head_dim", h.head_dim);
        h.gemma4_use_double_wide_mlp =
            get_or<bool>(text, "use_double_wide_mlp", false);
        // rope_parameters in gemma4 is a dict keyed by attention type.
        if (auto rp_it = text.find("rope_parameters");
            rp_it != text.end() && rp_it->is_object()) {
            const auto& rp = *rp_it;
            auto read_kind = [&](const char* key, float default_theta,
                                  float& theta_out, float& partial_out) {
                auto it = rp.find(key);
                if (it == rp.end() || !it->is_object()) {
                    theta_out = default_theta;
                    partial_out = 1.0f;
                    return;
                }
                theta_out = get_or<float>(*it, "rope_theta", default_theta);
                partial_out = get_or<float>(*it, "partial_rotary_factor", 1.0f);
            };
            read_kind("full_attention", 1e6f,
                      h.gemma4_rope_theta_full,
                      h.gemma4_rope_partial_factor_full);
            read_kind("sliding_attention", 1e4f,
                      h.gemma4_rope_theta_sliding,
                      h.gemma4_rope_partial_factor_sliding);
        } else {
            h.gemma4_rope_theta_full = h.rope_theta;
            h.gemma4_rope_theta_sliding =
                (h.rope_local_base_freq > 0.0f) ? h.rope_local_base_freq : 1e4f;
        }
    }

    // Per-layer attention type list (olmo3, gemma4, gpt-oss, qwen3.5).
    // Encodes:
    //   's' = sliding_attention
    //   'g' = full_attention (global)
    //   'l' = linear_attention (Qwen 3.5 / 3.6 gated-delta-rule layer)
    if (auto lt_it = text.find("layer_types");
        lt_it != text.end() && lt_it->is_array()) {
        h.layer_types.reserve(lt_it->size());
        for (const auto& v : *lt_it) {
            const auto s = v.get<std::string>();
            if (s == "sliding_attention")  h.layer_types.push_back('s');
            else if (s == "full_attention")   h.layer_types.push_back('g');
            else if (s == "linear_attention") h.layer_types.push_back('l');
            else throw std::runtime_error(
                "hf_config: unknown layer_type '" + s + "'");
        }
    }

    if (auto v = get_opt<float>(text, "attn_logit_softcapping")) {
        h.attn_logit_softcapping = v;
    }
    if (auto v = get_opt<float>(text, "final_logit_softcapping")) {
        h.final_logit_softcapping = v;
    }
    if (auto v = get_opt<float>(text, "query_pre_attn_scalar")) {
        h.query_pre_attn_scalar = v;
    }

    // ── MoE ──
    // Different repos use different field names: Mixtral uses
    // `num_local_experts`, Qwen-MoE / GPT-OSS use `num_experts`.
    h.num_experts =
        get_or<std::int32_t>(text, "num_experts",
            get_or<std::int32_t>(text, "num_local_experts", 0));
    h.num_experts_per_tok =
        get_or<std::int32_t>(text, "num_experts_per_tok", 0);
    h.moe_intermediate_size =
        get_or<std::int32_t>(text, "moe_intermediate_size",
            h.intermediate_size);  // fallback
    h.norm_topk_prob = get_or<bool>(text, "norm_topk_prob", true);
    h.shared_expert_intermediate_size =
        get_or<std::int32_t>(text, "shared_expert_intermediate_size", 0);

    // ── Qwen 3.5 / 3.6 ──
    if (h.arch == PieArch::Qwen3_5) {
        h.qwen35_attn_output_gate =
            get_or<bool>(text, "attn_output_gate", false);
        h.qwen35_full_attn_interval =
            get_or<std::int32_t>(text, "full_attention_interval", 4);
        h.qwen35_linear_num_k_heads =
            get_or<std::int32_t>(text, "linear_num_key_heads", 0);
        h.qwen35_linear_num_v_heads =
            get_or<std::int32_t>(text, "linear_num_value_heads", 0);
        h.qwen35_linear_k_head_dim =
            get_or<std::int32_t>(text, "linear_key_head_dim", 0);
        h.qwen35_linear_v_head_dim =
            get_or<std::int32_t>(text, "linear_value_head_dim", 0);
        h.qwen35_linear_conv_kernel =
            get_or<std::int32_t>(text, "linear_conv_kernel_dim", 4);

        // mrope (multimodal RoPE) lives under rope_parameters. Qwen 3.6's
        // config carries rope_theta there too (no flat top-level copy).
        if (auto rp_it = text.find("rope_parameters");
            rp_it != text.end() && rp_it->is_object()) {
            const auto& rp = *rp_it;
            h.qwen35_mrope_interleaved =
                get_or<bool>(rp, "mrope_interleaved", false);
            h.qwen35_partial_rotary_factor =
                get_or<float>(rp, "partial_rotary_factor", 1.0f);
            if (rp.contains("rope_theta") && !rp["rope_theta"].is_null()) {
                h.rope_theta = rp["rope_theta"].get<float>();
            }
            if (auto sec_it = rp.find("mrope_section");
                sec_it != rp.end() && sec_it->is_array() &&
                sec_it->size() == 3) {
                for (std::size_t i = 0; i < 3; ++i) {
                    h.qwen35_mrope_section[i] =
                        (*sec_it)[i].get<std::int32_t>();
                }
            }
        }
    }

    return h;
}

}  // namespace pie_portable_driver
