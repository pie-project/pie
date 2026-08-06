#include "model_facts.hpp"

#include "batch/forward.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "model/facts.hpp"

namespace pie::metal {

/// `architectures[0]` reduced to the stem the registries key on.
///
/// HuggingFace names an architecture `<Stem>For<Task>`, and every registry on
/// both sides of the ABI -- this driver's family switch and the runtime's
/// `instruct::create` -- keys on the STEM. So the task suffix comes off.
///
/// The list is explicit rather than "cut at the first `for`", and that is the
/// whole of the care this needs: `ReformerForCausalLM` lowercases to
/// `reformerforcausallm`, whose FIRST `for` is inside `reformer`. Cutting
/// there yields `re`, which is not a family and does not announce itself as
/// one -- it just misses the registry and takes the fallback.
///
/// `forconditionalgeneration` is here because leaving it out is what a
/// multimodal release costs: `Gemma4ForConditionalGeneration` stayed whole,
/// missed `instruct::create`'s `gemma4` row, and took the ChatML fallback. The
/// weights still loaded (the load contract keys on `model_type`, not on this)
/// and the kernels were right, so the model produced fluent text -- of a
/// conversation that was never had, terminated by `<|im_end|>`, a token that
/// is not in gemma's vocabulary and was being spelled out a piece at a time.
/// Both gemma-4 checkpoints ship that architecture name.
std::string arch_stem(std::string name) {
    for (auto& c : name) c = static_cast<char>(std::tolower(c));
    // Longest first, so a suffix that contains another cannot be half-stripped.
    for (const char* suffix : {"forconditionalgeneration", "forcausallm"}) {
        const std::string s = suffix;
        if (name.size() > s.size() &&
            name.compare(name.size() - s.size(), s.size(), s) == 0) {
            name.erase(name.size() - s.size());
            break;
        }
    }
    return name;
}

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
        // `quantization` stays at the ROOT even for a multimodal release: it
        // describes the file's tensors, not the text decoder's architecture.
        if (j.contains("quantization") && j["quantization"].is_object()) {
            const nlohmann::json& q = j["quantization"];
            if (q.contains("bits") && q["bits"].is_number_integer()) {
                facts.quant_bits = q["bits"].get<int>();
            }
            if (q.contains("group_size") && q["group_size"].is_number_integer()) {
                facts.quant_group_size = q["group_size"].get<int>();
            }
        }
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
            const std::string a =
                arch_stem(j["architectures"][0].get<std::string>());
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
        if (pie::metal::model::model_family_of(facts.model_type) ==
            pie::metal::model::ModelFamily::Llama) {
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
                gf(rs, "low_freq_factor", facts.ll_rope_low_freq_factor);
                gf(rs, "high_freq_factor", facts.ll_rope_high_freq_factor);
                if (rs.contains("original_max_position_embeddings") &&
                    rs["original_max_position_embeddings"].is_number_integer()) {
                    facts.ll_rope_original_max_position =
                        rs["original_max_position_embeddings"].get<int>();
                }
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
            // do not. Usually implied by the architecture, so the `model_type`
            // says it -- but an explicit `use_qk_norm` wins where a config
            // states one. `normalize.rs::infer_qk_norm` and the CUDA driver's
            // `infer_qk_norm` both give the explicit flag priority, and so does
            // the descriptor branch below; this path was the only one of the
            // four ignoring it, so a checkpoint carrying `"use_qk_norm": true`
            // ran q/k-normed from an artifact and un-normed from raw
            // `config.json` -- the fluent-but-wrong-tokens failure.
            facts.ll_qk_norm = (j.contains("use_qk_norm") &&
                                j["use_qk_norm"].is_boolean() &&
                                j["use_qk_norm"].get<bool>()) ||
                               facts.model_type == "qwen3" ||
                               facts.model_type == "qwen3_moe";
        }

        // ── Qwen3.5 / Qwen3-Next: the GDN hybrid ──
        // Read only when `model_type` names it, like every block around it.
        if (pie::metal::model::model_family_of(facts.model_type) ==
            pie::metal::model::ModelFamily::Qwen35) {
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
            gi(tc, "num_hidden_layers", facts.q35_num_hidden_layers);
            gi(tc, "hidden_size", facts.q35_hidden_size);
            gi(tc, "vocab_size", facts.q35_vocab_size);
            gi(tc, "num_attention_heads", facts.q35_num_attention_heads);
            gi(tc, "num_key_value_heads", facts.q35_num_key_value_heads);
            gi(tc, "head_dim", facts.q35_head_dim);
            gi(tc, "intermediate_size", facts.q35_intermediate_size);
            gi(tc, "linear_num_key_heads", facts.q35_linear_key_heads);
            gi(tc, "linear_num_value_heads", facts.q35_linear_value_heads);
            gi(tc, "linear_key_head_dim", facts.q35_linear_key_head_dim);
            gi(tc, "linear_value_head_dim", facts.q35_linear_value_head_dim);
            gi(tc, "linear_conv_kernel_dim", facts.q35_linear_conv_kernel);
            gi(tc, "full_attention_interval", facts.q35_full_attn_interval);
            gi(tc, "num_experts", facts.q35_num_experts);
            gi(tc, "num_experts_per_tok", facts.q35_num_experts_per_tok);
            gi(tc, "moe_intermediate_size", facts.q35_moe_intermediate_size);
            gi(tc, "shared_expert_intermediate_size", facts.q35_shared_expert_intermediate);
            gi(tc, "decoder_sparse_step", facts.q35_decoder_sparse_step);
            gf(tc, "rms_norm_eps", facts.q35_rms_norm_eps);
            if (tc.contains("norm_topk_prob") && tc["norm_topk_prob"].is_boolean()) {
                facts.q35_norm_topk_prob = tc["norm_topk_prob"].get<bool>();
            }
            // A multimodal wrapper spells this at the TOP level, beside
            // `text_config` rather than inside it -- Qwen3.5-35B-A3B says
            // `"tie_word_embeddings": false` there and says nothing in the text
            // config, so reading only the inner one defaulted to tied and asked
            // the load for a tensor the checkpoint does not have. The inner
            // spelling still wins where both appear: it is the text decoder's
            // own statement about the text decoder.
            if (tc.contains("tie_word_embeddings") && tc["tie_word_embeddings"].is_boolean()) {
                facts.q35_tied_embeddings = tc["tie_word_embeddings"].get<bool>();
            } else if (j.contains("tie_word_embeddings") &&
                       j["tie_word_embeddings"].is_boolean()) {
                facts.q35_tied_embeddings = j["tie_word_embeddings"].get<bool>();
            }
            if (tc.contains("mlp_only_layers") && tc["mlp_only_layers"].is_array()) {
                facts.q35_mlp_only_layer_count = int(tc["mlp_only_layers"].size());
            }
            // Some releases spell the layer pattern as a list instead of an
            // interval. Reduced to the interval it implies, or to -1 when it
            // implies none -- the geometry refuses -1 rather than rounding an
            // irregular stack to a regular one, which would put full attention
            // on layers that are linear.
            if (facts.q35_full_attn_interval == 0 && tc.contains("layer_types") &&
                tc["layer_types"].is_array()) {
                std::vector<int> full;
                int idx = 0;
                for (const auto& lt : tc["layer_types"]) {
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
            const auto bool_of = [](const nlohmann::json& obj, const char* key, bool& out) {
                if (obj.contains(key) && obj[key].is_boolean()) out = obj[key].get<bool>();
            };
            // The mixture, and the k-eq-V attention that comes with it. Absent
            // on every dense gemma 4, which is how the geometry tells them
            // apart -- so these are read where they are and defaulted nowhere.
            bool_of(tc, "enable_moe_block", facts.g4_enable_moe);
            bool_of(tc, "attention_k_eq_v", facts.g4_attention_k_eq_v);
            i32_of(tc, "num_experts", facts.g4_num_experts);
            i32_of(tc, "top_k_experts", facts.g4_experts_per_token);
            i32_of(tc, "moe_intermediate_size", facts.g4_moe_intermediate);
            i32_of(tc, "num_global_key_value_heads", facts.g4_num_global_kv_heads);
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
    cfg.llama.rope_low_freq_factor = facts.ll_rope_low_freq_factor;
    cfg.llama.rope_high_freq_factor = facts.ll_rope_high_freq_factor;
    cfg.llama.rope_original_max_position = facts.ll_rope_original_max_position;
    cfg.llama.norm_topk_prob = facts.ll_norm_topk_prob;
    cfg.llama.qk_norm = facts.ll_qk_norm;
    cfg.llama.tied_embeddings = facts.ll_tied_embeddings;
    cfg.quant_bits = facts.quant_bits;
    cfg.quant_group_size = facts.quant_group_size;
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
    cfg.gemma4.enable_moe = facts.g4_enable_moe;
    cfg.gemma4.n_experts = facts.g4_num_experts;
    cfg.gemma4.experts_per_token = facts.g4_experts_per_token;
    cfg.gemma4.moe_intermediate = facts.g4_moe_intermediate;
    cfg.gemma4.attention_k_eq_v = facts.g4_attention_k_eq_v;
    cfg.gemma4.n_global_kv_heads = facts.g4_num_global_kv_heads;
}

// Reading the facts out of a `pie.model/1` descriptor instead of deriving them
// from `config.json`.
//
// The descriptor is flat and already normalized -- no `text_config` to step
// into, no alternate spellings, no per-family defaulting -- so every read here
// is one key.
//
// Returns nullopt when there is no descriptor to read, and the caller then
// parses `config.json` whole. It is all or nothing per model: half the facts
// normalized and half probed from files is the exact skew the artifact exists
// to remove.
//
// `global_head_dim` and `use_double_wide_mlp` are read by this driver and by
// no other, and they are in the descriptor anyway -- `pie.model/1` is the
// artifact's model config, not one driver's. They arrive under their
// `HfConfig` field names, `gemma4_*`, because the schema is generated from
// that struct.
std::optional<ModelFacts> read_model_facts_from_descriptor(
    std::string_view descriptor_json) {
    if (descriptor_json.empty()) return std::nullopt;

    // Text in, not a path: the caller reads the file, because it keeps the
    // document anyway — the compile request carries it whole, and opening it
    // twice would be two chances to read two different things.
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(descriptor_json);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (j.value("version", std::string{}) != "pie.model/1") return std::nullopt;

    ModelFacts facts;
    const auto u32_of = [&j](const char* key, std::uint32_t& out) {
        if (j.contains(key) && j[key].is_number_integer()) {
            out = j[key].get<std::uint32_t>();
        }
    };
    const auto i32_of = [&j](const char* key, int& out) {
        if (j.contains(key) && j[key].is_number_integer()) {
            out = j[key].get<int>();
        }
    };
    const auto f32_of = [&j](const char* key, float& out) {
        if (j.contains(key) && j[key].is_number()) {
            out = j[key].get<float>();
        }
    };

    facts.model_type = j.value("model_type", std::string{});
    u32_of("vocab_size", facts.vocab_size);
    u32_of("max_position_embeddings", facts.max_model_len);
    f32_of("rope_theta", facts.rope_theta);
    f32_of("partial_rotary_factor", facts.partial_rotary_factor);
    // The affine width and group, which the `config.json` path reads off the
    // root `quantization` block. Leaving them zero here is not a smaller set of
    // facts, it is a *wrong* one: every geometry defaults to `{4, 64}` and only
    // overrides when these are non-zero, so an 8-bit checkpoint served as an
    // artifact would be decoded by 4-bit kernels. The descriptor spells them
    // `quant_bits` / `quant_group_size` because the schema is generated from
    // `HfConfig`, which normalizes both quantization dialects into one pair.
    i32_of("quant_bits", facts.quant_bits);
    i32_of("quant_group_size", facts.quant_group_size);

    // The declared quantization width. `read_model_facts` took these from
    // `mlx_lm`'s bare `quantization` block, and the descriptor now carries
    // them under the `quant_*` names the schema uses -- the normalizer reads
    // that spelling as well as `quantization_config`.
    //
    // Not optional detail: zero here means "undeclared", and both the load
    // contract (`push_mlx_affine_declared`) and the forward geometry
    // (`batch/forward.cpp`) answer undeclared with 4 bits. An 8-bit
    // checkpoint read as 4-bit is not a refusal, it is wrong numbers.
    i32_of("quant_bits", facts.quant_bits);
    i32_of("quant_group_size", facts.quant_group_size);

    // `arch_name` is `architectures[0]` verbatim in the descriptor; this driver
    // keys on the lowercased stem, so apply the same reduction it applies to
    // `config.json` -- the SAME function, because two copies of this is how
    // one of them came to strip only `forcausallm`.
    const std::string arch = arch_stem(j.value("arch_name", std::string{}));
    if (!arch.empty()) facts.arch_name = arch;

    if (j.value("linear_num_value_heads", 0) > 0) facts.has_linear_attn = true;
    if (j.contains("layer_types") && j["layer_types"].is_array()) {
        for (const auto& t : j["layer_types"]) {
            if (t.is_string() && t.get<std::string>() == "linear_attention") {
                facts.has_linear_attn = true;
                break;
            }
        }
    }

    if (facts.model_type == "gpt_oss") {
        i32_of("num_hidden_layers", facts.go_num_hidden_layers);
        i32_of("hidden_size", facts.go_hidden_size);
        i32_of("vocab_size", facts.go_vocab_size);
        i32_of("num_attention_heads", facts.go_num_attention_heads);
        i32_of("num_key_value_heads", facts.go_num_key_value_heads);
        i32_of("head_dim", facts.go_head_dim);
        i32_of("sliding_window", facts.go_sliding_window);
        // The descriptor folds `num_local_experts` / `num_experts` /
        // `n_routed_experts` into one field at import.
        i32_of("num_experts", facts.go_num_local_experts);
        i32_of("num_experts_per_tok", facts.go_num_experts_per_tok);
        i32_of("intermediate_size", facts.go_intermediate_size);
        f32_of("rms_norm_eps", facts.go_rms_norm_eps);
        f32_of("swiglu_limit", facts.go_swiglu_limit);
        f32_of("rope_theta", facts.go_rope_theta);
        // YaRN's four, already resolved out of `rope_scaling`.
        f32_of("rope_factor", facts.go_rope_factor);
        f32_of("rope_beta_fast", facts.go_rope_beta_fast);
        f32_of("rope_beta_slow", facts.go_rope_beta_slow);
        i32_of("rope_original_max_position", facts.go_rope_original_max_position);
    }

    // ── The llama-shaped families ──
    // The mirror of the `config.json` block above: read only when `model_type`
    // names one of them, so a config that never mentions this family cannot
    // accidentally select it. Without this branch every llama-shaped artifact
    // reached the geometry with `ll_num_hidden_layers == 0` and was refused as
    // "config carried no decoder shape" -- the descriptor path claimed to carry
    // a normalized config and then carried no decoder for this family at all.
    if (pie::metal::model::model_family_of(facts.model_type) ==
        pie::metal::model::ModelFamily::Llama) {
        i32_of("num_hidden_layers", facts.ll_num_hidden_layers);
        i32_of("hidden_size", facts.ll_hidden_size);
        i32_of("vocab_size", facts.ll_vocab_size);
        i32_of("num_attention_heads", facts.ll_num_attention_heads);
        i32_of("num_key_value_heads", facts.ll_num_key_value_heads);
        i32_of("head_dim", facts.ll_head_dim);
        i32_of("intermediate_size", facts.ll_intermediate_size);
        // The descriptor folds `num_local_experts` / `num_experts` /
        // `n_routed_experts` into one field at import, as it does for gpt-oss.
        i32_of("num_experts", facts.ll_num_experts);
        i32_of("num_experts_per_tok", facts.ll_num_experts_per_tok);
        i32_of("moe_intermediate_size", facts.ll_moe_intermediate_size);
        f32_of("rms_norm_eps", facts.ll_rms_norm_eps);
        f32_of("rope_theta", facts.ll_rope_theta);
        facts.ll_norm_topk_prob = j.value("norm_topk_prob", facts.ll_norm_topk_prob);
        facts.ll_tied_embeddings =
            j.value("tie_word_embeddings", facts.ll_tied_embeddings);
        // `use_qk_norm` is explicit in the descriptor, but import only infers it
        // for `qwen3` -- not for `qwen3_moe`, which normalizes q and k just the
        // same. OR in the `model_type` the `config.json` path keys on so the two
        // paths cannot disagree about a checkpoint's numerics.
        facts.ll_qk_norm = j.value("use_qk_norm", false) ||
                           facts.model_type == "qwen3" ||
                           facts.model_type == "qwen3_moe";
        // Import resolves `rope_scaling` into a kind plus already-defaulted
        // knobs; the driver keys on HF's spelling, so translate back. `none`
        // stays empty, which is the "plain geometric series" the geometry runs
        // for an absent scaling. `original_yarn` becomes `yarn` so the refusal
        // names the schedule the config actually asked for.
        const std::string kind = j.value("rope_scaling_kind", std::string{});
        if (kind == "llama3") {
            facts.ll_rope_scaling_kind = "llama3";
        } else if (kind == "original_yarn") {
            facts.ll_rope_scaling_kind = "yarn";
        }
        // `rope_factor` is read whatever the kind. A linear scaling normalizes
        // to kind `none` plus a factor -- it IS the plain geometric series,
        // just with positions divided -- so gating this read on a non-empty
        // kind would drop the factor and silently disagree with the
        // `config.json` path above, which reads it unconditionally.
        f32_of("rope_factor", facts.ll_rope_scale);
        if (!facts.ll_rope_scaling_kind.empty()) {
            f32_of("rope_low_freq_factor", facts.ll_rope_low_freq_factor);
            f32_of("rope_high_freq_factor", facts.ll_rope_high_freq_factor);
            i32_of("rope_original_max_position",
                   facts.ll_rope_original_max_position);
        }
    }

    // ── Qwen3.5 / Qwen3-Next: the GDN hybrid ──
    // The mirror of the `config.json` block above, and the reason a Qwen3.5
    // checkpoint imported through `pie model import` could not boot on Metal:
    // the descriptor path had no branch for this family at all, so the geometry
    // saw `q35_num_hidden_layers == 0` and refused the model as carrying no
    // decoder shape -- from a descriptor that carried the whole decoder.
    if (pie::metal::model::model_family_of(facts.model_type) ==
        pie::metal::model::ModelFamily::Qwen35) {
        i32_of("num_hidden_layers", facts.q35_num_hidden_layers);
        i32_of("hidden_size", facts.q35_hidden_size);
        i32_of("vocab_size", facts.q35_vocab_size);
        i32_of("num_attention_heads", facts.q35_num_attention_heads);
        i32_of("num_key_value_heads", facts.q35_num_key_value_heads);
        i32_of("head_dim", facts.q35_head_dim);
        i32_of("intermediate_size", facts.q35_intermediate_size);
        i32_of("linear_num_key_heads", facts.q35_linear_key_heads);
        i32_of("linear_num_value_heads", facts.q35_linear_value_heads);
        i32_of("linear_key_head_dim", facts.q35_linear_key_head_dim);
        i32_of("linear_value_head_dim", facts.q35_linear_value_head_dim);
        i32_of("linear_conv_kernel_dim", facts.q35_linear_conv_kernel);
        // The descriptor folds `num_local_experts` / `num_experts` /
        // `n_routed_experts` into one field at import, as it does for the
        // families above.
        i32_of("num_experts", facts.q35_num_experts);
        i32_of("num_experts_per_tok", facts.q35_num_experts_per_tok);
        i32_of("moe_intermediate_size", facts.q35_moe_intermediate_size);
        i32_of("shared_expert_intermediate_size",
               facts.q35_shared_expert_intermediate);
        f32_of("rms_norm_eps", facts.q35_rms_norm_eps);
        facts.q35_norm_topk_prob =
            j.value("norm_topk_prob", facts.q35_norm_topk_prob);
        // The multimodal `text_config` split the `config.json` path has to
        // reconcile does not exist here: import already resolved the text
        // decoder's own statement into one top-level field.
        facts.q35_tied_embeddings =
            j.value("tie_word_embeddings", facts.q35_tied_embeddings);
        // `full_attention_interval` is not a descriptor field -- import
        // expands the schedule into `layer_types`, one entry per layer. Reduce
        // it the same way the `config.json` path reduces a list-spelled
        // pattern, including the -1 for an irregular stack: rounding that to a
        // regular interval would put full attention on layers that are linear.
        if (j.contains("layer_types") && j["layer_types"].is_array()) {
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
        // `decoder_sparse_step` and `mlp_only_layers` have no descriptor
        // spelling; their defaults stand. Both describe which layers are dense
        // in a routed stack, so they are inert for the dense releases and are
        // the known gap for a MoE one imported through the descriptor.
    }

    if (facts.model_type == "gemma4" || facts.model_type == "gemma4_text") {
        i32_of("num_hidden_layers", facts.g4_num_hidden_layers);
        i32_of("hidden_size", facts.g4_hidden_size);
        i32_of("intermediate_size", facts.g4_intermediate_size);
        i32_of("num_attention_heads", facts.g4_num_attention_heads);
        i32_of("num_key_value_heads", facts.g4_num_key_value_heads);
        i32_of("head_dim", facts.g4_head_dim);
        i32_of("gemma4_global_head_dim", facts.g4_global_head_dim);
        // `sliding_window` is -1 in the descriptor when the config omits it,
        // where reading `config.json` would leave this 0. Guard so "absent"
        // means the same thing on both paths.
        if (j.value("sliding_window", -1) > 0) {
            i32_of("sliding_window", facts.g4_sliding_window);
        }
        i32_of("num_kv_shared_layers", facts.g4_num_kv_shared_layers);
        i32_of("gemma_hidden_size_per_layer_input", facts.g4_per_layer_emb_dim);
        facts.g4_double_wide_mlp = j.value("gemma4_double_wide_mlp", false);
        facts.g4_enable_moe = j.value("gemma4_enable_moe", false);
        facts.g4_attention_k_eq_v = j.value("gemma4_attention_k_eq_v", false);
        i32_of("num_experts", facts.g4_num_experts);
        i32_of("num_experts_per_tok", facts.g4_experts_per_token);
        i32_of("moe_intermediate_size", facts.g4_moe_intermediate);
        i32_of("gemma4_num_global_key_value_heads", facts.g4_num_global_kv_heads);
        f32_of("gemma_final_logit_softcap", facts.g4_final_softcap);

        // The descriptor expands the per-attention-type rope into one entry
        // per layer at import, so the per-type values come back by looking at
        // the first layer of each type rather than by re-reading nested JSON.
        if (j.contains("layer_types") && j["layer_types"].is_array()) {
            const auto& types = j["layer_types"];
            const auto at = [&j](const char* key, std::size_t i, float& out) {
                if (j.contains(key) && j[key].is_array() && i < j[key].size() &&
                    j[key][i].is_number()) {
                    out = j[key][i].get<float>();
                }
            };
            std::vector<int> full;
            // Each type's values come from the *first* layer of that type, and
            // the two are tracked independently: keying the sliding read off
            // "no full layer yet" would miss a stack that opens with one.
            bool saw_full = false;
            bool saw_sliding = false;
            for (std::size_t i = 0; i < types.size(); ++i) {
                if (!types[i].is_string()) continue;
                const std::string t = types[i].get<std::string>();
                if (t == "full_attention") {
                    if (!saw_full) {
                        at("gemma_per_layer_rope_theta", i, facts.g4_rope_theta_full);
                        at("gemma_per_layer_partial_rotary_factor", i,
                           facts.g4_full_partial_rotary);
                        saw_full = true;
                    }
                    full.push_back(static_cast<int>(i));
                } else if (t == "sliding_attention" && !saw_sliding) {
                    at("gemma_per_layer_rope_theta", i, facts.g4_rope_theta_sliding);
                    saw_sliding = true;
                }
            }
            // Same schedule check the `config.json` path makes: the interval is
            // the distance between the first two full layers, and an irregular
            // stack is refused (-1) rather than silently mis-scheduled.
            if (!full.empty()) {
                const int interval = full[0] + 1;
                bool regular = true;
                for (std::size_t k = 0; k < full.size(); ++k) {
                    regular = regular && full[k] == static_cast<int>(k + 1) * interval - 1;
                }
                facts.g4_full_attn_interval = regular ? interval : -1;
            }
        }
    }
    return facts;
}

}  // namespace pie::metal
