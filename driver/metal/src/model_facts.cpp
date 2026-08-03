#include "model_facts.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "model/contract.hpp"

namespace pie::metal {

ModelFacts read_model_facts(const std::string& hf_path) {
    ModelFacts facts;
    if (hf_path.empty()) return facts;
    const std::filesystem::path cfg =
        std::filesystem::path(hf_path) / "config.json";
    std::ifstream f(cfg);
    if (!f) return facts;
    try {
        nlohmann::json j;
        f >> j;
        // A multimodal release nests the text decoder's facts under
        // `text_config` and leaves the root to the wrapper. `model_type` and
        // the linear-attention probe below already read through that view;
        // `vocab_size` and `max_position_embeddings` used to read the root
        // only, so on the very family this driver targets they silently kept
        // their defaults and the vocab cross-check rejected the checkpoint as
        // "32000 != 248320".
        const nlohmann::json& tc =
            (j.contains("text_config") && j["text_config"].is_object())
                ? j["text_config"]
                : j;
        const auto u32_of = [](const nlohmann::json& obj, const char* key,
                               std::uint32_t& out) {
            if (obj.contains(key) && obj[key].is_number_integer()) {
                out = obj[key].get<std::uint32_t>();
                return true;
            }
            return false;
        };
        if (!u32_of(tc, "vocab_size", facts.vocab_size)) {
            u32_of(j, "vocab_size", facts.vocab_size);
        }
        if (!u32_of(tc, "max_position_embeddings", facts.max_model_len)) {
            u32_of(j, "max_position_embeddings", facts.max_model_len);
        }
        const auto f32_of = [](const nlohmann::json& obj, const char* key,
                               float& out) {
            if (obj.contains(key) && obj[key].is_number()) {
                out = obj[key].get<float>();
                return true;
            }
            return false;
        };
        // `rope_parameters` first (the current schema), then the flat key.
        const nlohmann::json* rp = nullptr;
        for (const nlohmann::json* scope : {&tc, const_cast<const nlohmann::json*>(&j)}) {
            if (scope->contains("rope_parameters") &&
                (*scope)["rope_parameters"].is_object()) {
                rp = &(*scope)["rope_parameters"];
                break;
            }
        }
        if (rp == nullptr || !f32_of(*rp, "rope_theta", facts.rope_theta)) {
            if (!f32_of(tc, "rope_theta", facts.rope_theta)) {
                f32_of(j, "rope_theta", facts.rope_theta);
            }
        }
        if (rp == nullptr ||
            !f32_of(*rp, "partial_rotary_factor", facts.partial_rotary_factor)) {
            if (!f32_of(tc, "partial_rotary_factor", facts.partial_rotary_factor)) {
                f32_of(j, "partial_rotary_factor", facts.partial_rotary_factor);
            }
        }
        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty()) {
            std::string a = j["architectures"][0].get<std::string>();
            for (auto& c : a) c = static_cast<char>(std::tolower(c));
            const std::string suffix = "forcausallm";
            if (a.size() > suffix.size() &&
                a.compare(a.size() - suffix.size(), suffix.size(), suffix) == 0) {
                a.erase(a.size() - suffix.size());
            }
            if (!a.empty()) facts.arch_name = a;
        }
        if (tc.contains("linear_num_value_heads") &&
            tc["linear_num_value_heads"].is_number_integer() &&
            tc["linear_num_value_heads"].get<int>() > 0) {
            facts.has_linear_attn = true;
        }
        if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
            for (const auto& t : tc["layer_types"]) {
                if (t.is_string() && t.get<std::string>() == "linear_attention") {
                    facts.has_linear_attn = true;
                    break;
                }
            }
        }
        const auto str_of = [](const nlohmann::json& obj, const char* key) {
            return obj.contains(key) && obj[key].is_string()
                       ? obj[key].get<std::string>()
                       : std::string{};
        };
        facts.model_type = str_of(tc, "model_type");
        if (facts.model_type.empty()) facts.model_type = str_of(j, "model_type");

        // ── GPT-OSS ──
        // Read only when the config says so, on the same principle. Its shape
        // is at the TOP level, not in a nested `text_config`, and its rope
        // parameters are YaRN's four in `rope_scaling`.
        if (facts.model_type == "gpt_oss") {
            const auto gi = [](const nlohmann::json& obj, const char* key, int& out) {
                if (obj.contains(key) && obj[key].is_number_integer()) {
                    out = obj[key].get<int>();
                }
            };
            const auto gf = [](const nlohmann::json& obj, const char* key, float& out) {
                if (obj.contains(key) && obj[key].is_number()) {
                    out = obj[key].get<float>();
                }
            };
            gi(j, "num_hidden_layers", facts.go_num_hidden_layers);
            gi(j, "hidden_size", facts.go_hidden_size);
            gi(j, "vocab_size", facts.go_vocab_size);
            gi(j, "num_attention_heads", facts.go_num_attention_heads);
            gi(j, "num_key_value_heads", facts.go_num_key_value_heads);
            gi(j, "head_dim", facts.go_head_dim);
            gi(j, "sliding_window", facts.go_sliding_window);
            gi(j, "num_local_experts", facts.go_num_local_experts);
            gi(j, "num_experts_per_tok", facts.go_num_experts_per_tok);
            gi(j, "intermediate_size", facts.go_intermediate_size);
            gf(j, "rms_norm_eps", facts.go_rms_norm_eps);
            gf(j, "swiglu_limit", facts.go_swiglu_limit);
            gf(j, "rope_theta", facts.go_rope_theta);
            if (j.contains("rope_scaling") && j["rope_scaling"].is_object()) {
                const auto& rs = j["rope_scaling"];
                gf(rs, "factor", facts.go_rope_factor);
                gf(rs, "beta_fast", facts.go_rope_beta_fast);
                gf(rs, "beta_slow", facts.go_rope_beta_slow);
                gi(rs, "original_max_position_embeddings",
                   facts.go_rope_original_max_position);
            }
        }

        // ── The llama-shaped families ──
        // Flat, top-level, and the same keys HF has used since llama 1. Read
        // only when `model_type` names one of them, on the same principle as
        // the two above: a config that never mentions this family cannot
        // accidentally select it.
        if (pie::metal::model::llama::is_supported_model_type(facts.model_type)) {
            const auto gi = [](const nlohmann::json& obj, const char* key, int& out) {
                if (obj.contains(key) && obj[key].is_number_integer()) {
                    out = obj[key].get<int>();
                }
            };
            const auto gf = [](const nlohmann::json& obj, const char* key, float& out) {
                if (obj.contains(key) && obj[key].is_number()) {
                    out = obj[key].get<float>();
                }
            };
            gi(j, "num_hidden_layers", facts.ll_num_hidden_layers);
            gi(j, "hidden_size", facts.ll_hidden_size);
            gi(j, "vocab_size", facts.ll_vocab_size);
            gi(j, "num_attention_heads", facts.ll_num_attention_heads);
            gi(j, "num_key_value_heads", facts.ll_num_key_value_heads);
            gi(j, "head_dim", facts.ll_head_dim);
            gi(j, "intermediate_size", facts.ll_intermediate_size);
            gi(j, "num_experts", facts.ll_num_experts);
            // Qwen spells the expert count `num_experts`; Mixtral-derived
            // configs spell it `num_local_experts`. Whichever is present wins;
            // neither present is a dense model.
            gi(j, "num_local_experts", facts.ll_num_experts);
            gi(j, "num_experts_per_tok", facts.ll_num_experts_per_tok);
            gi(j, "moe_intermediate_size", facts.ll_moe_intermediate_size);
            if (j.contains("norm_topk_prob") && j["norm_topk_prob"].is_boolean()) {
                facts.ll_norm_topk_prob = j["norm_topk_prob"].get<bool>();
            }
            gf(j, "rms_norm_eps", facts.ll_rms_norm_eps);
            gf(j, "rope_theta", facts.ll_rope_theta);
            if (j.contains("rope_scaling") && j["rope_scaling"].is_object()) {
                const auto& rs = j["rope_scaling"];
                gf(rs, "factor", facts.ll_rope_scale);
                // `rope_type` is the current key and `type` the older one. The
                // KIND is carried verbatim rather than reduced to a bool here,
                // because the geometry refuses the schedules it cannot run and
                // its message should name the one the config asked for.
                for (const char* key : {"rope_type", "type"}) {
                    if (rs.contains(key) && rs[key].is_string()) {
                        facts.ll_rope_scaling_kind = rs[key].get<std::string>();
                        break;
                    }
                }
            }
            if (j.contains("tie_word_embeddings") && j["tie_word_embeddings"].is_boolean()) {
                facts.ll_tied_embeddings = j["tie_word_embeddings"].get<bool>();
            }
            // Qwen3 RMS-normalises q and k per head; llama, mistral and qwen2
            // do not. The config has no key for it -- it is implied by the
            // architecture -- so the `model_type` says it.
            facts.ll_qk_norm =
                facts.model_type == "qwen3" || facts.model_type == "qwen3_moe";
        }

        // ── Qwen3.5 / Qwen3-Next: the GDN hybrid ──
        // Read only when `model_type` names it, like every block around it.
        if (pie::metal::model::qwen3_5::is_supported_model_type(facts.model_type)) {
            const auto gi = [](const nlohmann::json& obj, const char* key, int& out) {
                if (obj.contains(key) && obj[key].is_number_integer()) {
                    out = obj[key].get<int>();
                }
            };
            const auto gf = [](const nlohmann::json& obj, const char* key, float& out) {
                if (obj.contains(key) && obj[key].is_number()) {
                    out = obj[key].get<float>();
                }
            };
            gi(j, "num_hidden_layers", facts.q35_num_hidden_layers);
            gi(j, "hidden_size", facts.q35_hidden_size);
            gi(j, "vocab_size", facts.q35_vocab_size);
            gi(j, "num_attention_heads", facts.q35_num_attention_heads);
            gi(j, "num_key_value_heads", facts.q35_num_key_value_heads);
            gi(j, "head_dim", facts.q35_head_dim);
            gi(j, "intermediate_size", facts.q35_intermediate_size);
            gi(j, "linear_num_key_heads", facts.q35_linear_key_heads);
            gi(j, "linear_num_value_heads", facts.q35_linear_value_heads);
            gi(j, "linear_key_head_dim", facts.q35_linear_key_head_dim);
            gi(j, "linear_value_head_dim", facts.q35_linear_value_head_dim);
            gi(j, "linear_conv_kernel_dim", facts.q35_linear_conv_kernel);
            gi(j, "full_attention_interval", facts.q35_full_attn_interval);
            gi(j, "num_experts", facts.q35_num_experts);
            gi(j, "num_experts_per_tok", facts.q35_num_experts_per_tok);
            gi(j, "moe_intermediate_size", facts.q35_moe_intermediate_size);
            gi(j, "shared_expert_intermediate_size", facts.q35_shared_expert_intermediate);
            gi(j, "decoder_sparse_step", facts.q35_decoder_sparse_step);
            gf(j, "rms_norm_eps", facts.q35_rms_norm_eps);
            if (j.contains("norm_topk_prob") && j["norm_topk_prob"].is_boolean()) {
                facts.q35_norm_topk_prob = j["norm_topk_prob"].get<bool>();
            }
            if (j.contains("tie_word_embeddings") && j["tie_word_embeddings"].is_boolean()) {
                facts.q35_tied_embeddings = j["tie_word_embeddings"].get<bool>();
            }
            if (j.contains("mlp_only_layers") && j["mlp_only_layers"].is_array()) {
                facts.q35_mlp_only_layer_count = int(j["mlp_only_layers"].size());
            }
            // Some releases spell the layer pattern as a list instead of an
            // interval. Reduced to the interval it implies, or to -1 when it
            // implies none -- the geometry refuses -1 rather than rounding an
            // irregular stack to a regular one, which would put full attention
            // on layers that are linear.
            if (facts.q35_full_attn_interval == 0 && j.contains("layer_types") &&
                j["layer_types"].is_array()) {
                std::vector<int> full;
                int idx = 0;
                for (const auto& lt : j["layer_types"]) {
                    if (lt.is_string() && lt.get<std::string>() == "full_attention") {
                        full.push_back(idx);
                    }
                    ++idx;
                }
                if (!full.empty()) {
                    const int interval = full[0] + 1;
                    bool regular = true;
                    for (std::size_t k = 0; k < full.size(); ++k) {
                        regular = regular && full[k] == int(k + 1) * interval - 1;
                    }
                    facts.q35_full_attn_interval = regular ? interval : -1;
                }
            }
        }

        // ── Gemma 4 ──
        // Read only when the config says so, so nothing here can perturb the
        // family that already works.
        if (facts.model_type == "gemma4" || facts.model_type == "gemma4_text") {
            const auto i32_of = [](const nlohmann::json& obj, const char* key, int& out) {
                if (obj.contains(key) && obj[key].is_number_integer()) {
                    out = obj[key].get<int>();
                    return true;
                }
                return false;
            };
            i32_of(tc, "num_hidden_layers", facts.g4_num_hidden_layers);
            i32_of(tc, "hidden_size", facts.g4_hidden_size);
            i32_of(tc, "intermediate_size", facts.g4_intermediate_size);
            i32_of(tc, "num_attention_heads", facts.g4_num_attention_heads);
            i32_of(tc, "num_key_value_heads", facts.g4_num_key_value_heads);
            i32_of(tc, "head_dim", facts.g4_head_dim);
            i32_of(tc, "global_head_dim", facts.g4_global_head_dim);
            i32_of(tc, "sliding_window", facts.g4_sliding_window);
            i32_of(tc, "num_kv_shared_layers", facts.g4_num_kv_shared_layers);
            i32_of(tc, "hidden_size_per_layer_input", facts.g4_per_layer_emb_dim);
            if (tc.contains("use_double_wide_mlp") && tc["use_double_wide_mlp"].is_boolean()) {
                facts.g4_double_wide_mlp = tc["use_double_wide_mlp"].get<bool>();
            }
            f32_of(tc, "final_logit_softcapping", facts.g4_final_softcap);
            // Per-attention-type rope.
            if (rp != nullptr) {
                if (rp->contains("full_attention") && (*rp)["full_attention"].is_object()) {
                    const auto& full = (*rp)["full_attention"];
                    f32_of(full, "rope_theta", facts.g4_rope_theta_full);
                    f32_of(full, "partial_rotary_factor", facts.g4_full_partial_rotary);
                }
                if (rp->contains("sliding_attention") && (*rp)["sliding_attention"].is_object()) {
                    f32_of((*rp)["sliding_attention"], "rope_theta",
                           facts.g4_rope_theta_sliding);
                }
            }
            // The full-attention schedule, derived from `layer_types` rather
            // than assumed: the interval is the distance between the first two
            // full-attention layers, and the list is then checked against it so
            // an irregular stack is refused instead of silently mis-scheduled.
            if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
                std::vector<int> full;
                int idx = 0;
                for (const auto& t : tc["layer_types"]) {
                    if (t.is_string() && t.get<std::string>() == "full_attention") {
                        full.push_back(idx);
                    }
                    ++idx;
                }
                if (!full.empty()) {
                    const int interval = full[0] + 1;
                    bool regular = true;
                    for (std::size_t k = 0; k < full.size(); ++k) {
                        regular = regular &&
                                  full[k] == static_cast<int>(k + 1) * interval - 1;
                    }
                    facts.g4_full_attn_interval = regular ? interval : -1;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] warning: failed to parse "
                  << cfg.string() << ": " << e.what() << "\n";
    }
    return facts;
}

/// Fill a SetupConfig's model geometry from the facts read out of config.json.
///
/// Shared by the capabilities pass and by setup, because the two must agree
/// about the SAME model: the row budget below is DERIVED from this geometry,
/// and a capability advertising more rows than setup allocates is a fire the
/// driver accepts and cannot hold.
void fill_family_geometry(pie::metal::batch::SetupConfig& cfg, const ModelFacts& facts) {
    cfg.model_type = facts.model_type;
    cfg.rope_theta = facts.rope_theta;
    cfg.partial_rotary_factor = facts.partial_rotary_factor;
    cfg.gptoss.n_layers = facts.go_num_hidden_layers;
    cfg.gptoss.hidden = facts.go_hidden_size;
    cfg.gptoss.vocab = facts.go_vocab_size;
    cfg.gptoss.n_q_heads = facts.go_num_attention_heads;
    cfg.gptoss.n_kv_heads = facts.go_num_key_value_heads;
    cfg.gptoss.head_dim = facts.go_head_dim;
    cfg.gptoss.sliding_window = facts.go_sliding_window;
    cfg.gptoss.n_experts = facts.go_num_local_experts;
    cfg.gptoss.experts_per_token = facts.go_num_experts_per_tok;
    cfg.gptoss.intermediate = facts.go_intermediate_size;
    cfg.gptoss.eps = facts.go_rms_norm_eps;
    cfg.gptoss.swiglu_limit = facts.go_swiglu_limit;
    cfg.gptoss.rope_theta = facts.go_rope_theta;
    cfg.gptoss.rope_factor = facts.go_rope_factor;
    cfg.gptoss.rope_beta_fast = facts.go_rope_beta_fast;
    cfg.gptoss.rope_beta_slow = facts.go_rope_beta_slow;
    cfg.gptoss.rope_original_max_position = facts.go_rope_original_max_position;
    cfg.llama.n_layers = facts.ll_num_hidden_layers;
    cfg.llama.hidden = facts.ll_hidden_size;
    cfg.llama.vocab = facts.ll_vocab_size;
    cfg.llama.n_q_heads = facts.ll_num_attention_heads;
    cfg.llama.n_kv_heads = facts.ll_num_key_value_heads;
    cfg.llama.head_dim = facts.ll_head_dim;
    cfg.llama.intermediate = facts.ll_intermediate_size;
    cfg.llama.n_experts = facts.ll_num_experts;
    cfg.llama.experts_per_token = facts.ll_num_experts_per_tok;
    cfg.llama.moe_intermediate = facts.ll_moe_intermediate_size;
    cfg.llama.eps = facts.ll_rms_norm_eps;
    cfg.llama.rope_theta = facts.ll_rope_theta;
    cfg.llama.rope_scale = facts.ll_rope_scale;
    cfg.llama.rope_scaling_kind = facts.ll_rope_scaling_kind;
    cfg.llama.norm_topk_prob = facts.ll_norm_topk_prob;
    cfg.llama.qk_norm = facts.ll_qk_norm;
    cfg.llama.tied_embeddings = facts.ll_tied_embeddings;
    cfg.qwen35.n_layers = facts.q35_num_hidden_layers;
    cfg.qwen35.hidden = facts.q35_hidden_size;
    cfg.qwen35.vocab = facts.q35_vocab_size;
    cfg.qwen35.n_q_heads = facts.q35_num_attention_heads;
    cfg.qwen35.n_kv_heads = facts.q35_num_key_value_heads;
    cfg.qwen35.head_dim = facts.q35_head_dim;
    cfg.qwen35.intermediate = facts.q35_intermediate_size;
    cfg.qwen35.gdn_k_heads = facts.q35_linear_key_heads;
    cfg.qwen35.gdn_v_heads = facts.q35_linear_value_heads;
    cfg.qwen35.gdn_k_dim = facts.q35_linear_key_head_dim;
    cfg.qwen35.gdn_v_dim = facts.q35_linear_value_head_dim;
    cfg.qwen35.gdn_conv_k = facts.q35_linear_conv_kernel;
    cfg.qwen35.full_attn_interval = facts.q35_full_attn_interval;
    cfg.qwen35.n_experts = facts.q35_num_experts;
    cfg.qwen35.experts_per_token = facts.q35_num_experts_per_tok;
    cfg.qwen35.moe_intermediate = facts.q35_moe_intermediate_size;
    cfg.qwen35.shared_expert_intermediate = facts.q35_shared_expert_intermediate;
    cfg.qwen35.decoder_sparse_step = facts.q35_decoder_sparse_step;
    cfg.qwen35.mlp_only_layer_count = facts.q35_mlp_only_layer_count;
    cfg.qwen35.eps = facts.q35_rms_norm_eps;
    cfg.qwen35.tied_embeddings = facts.q35_tied_embeddings;
    cfg.qwen35.norm_topk_prob = facts.q35_norm_topk_prob;
    cfg.gemma4.n_layers = facts.g4_num_hidden_layers;
    cfg.gemma4.hidden = facts.g4_hidden_size;
    cfg.gemma4.intermediate = facts.g4_intermediate_size;
    cfg.gemma4.n_q_heads = facts.g4_num_attention_heads;
    cfg.gemma4.n_kv_heads = facts.g4_num_key_value_heads;
    cfg.gemma4.head_dim = facts.g4_head_dim;
    cfg.gemma4.global_head_dim = facts.g4_global_head_dim;
    cfg.gemma4.sliding_window = facts.g4_sliding_window;
    cfg.gemma4.num_kv_shared_layers = facts.g4_num_kv_shared_layers;
    cfg.gemma4.per_layer_emb_dim = facts.g4_per_layer_emb_dim;
    cfg.gemma4.full_attn_interval = facts.g4_full_attn_interval;
    cfg.gemma4.double_wide_mlp = facts.g4_double_wide_mlp;
    cfg.gemma4.final_softcap = facts.g4_final_softcap;
    cfg.gemma4.rope_theta_full = facts.g4_rope_theta_full;
    cfg.gemma4.rope_theta_sliding = facts.g4_rope_theta_sliding;
    cfg.gemma4.full_partial_rotary = facts.g4_full_partial_rotary;
}

}  // namespace pie::metal
