#include "model/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ops/attention_workspace.hpp"
#include "store/dsa_cache.hpp"
#include "distributed.hpp"
#include "store/mla_cache.hpp"
#include "store/recurrent_state_cache.hpp"
#include "model/config.hpp"
#include "model/loaded_model.hpp"
#include "model/workspace.hpp"        // universal model::workspace_bytes

#ifndef PIE_CUDA_QWEN_ONLY
#include "model/csm/csm.hpp"
#include "model/csm/csm_model.hpp"
#include "model/deepseek_v4/deepseek_v4.hpp"
#include "model/deepseek_v4/deepseek_v4_model.hpp"
#include "model/gemma/gemma2.hpp"
#include "model/gemma/gemma2_model.hpp"
#include "model/gemma3n/gemma3n.hpp"
#include "model/gemma3n/gemma3n_model.hpp"
#include "model/gemma4/gemma4.hpp"
#include "model/gemma4/gemma4_model.hpp"
#include "model/glm5/glm5.hpp"
#include "model/glm5/glm5_forward.hpp"
#include "model/glm5/glm5_model.hpp"
#include "model/kimi/kimi.hpp"
#include "model/kimi/kimi_forward.hpp"
#include "model/kimi/kimi_model.hpp"
#endif
#include "model/llama_like/llama_like.hpp"
#include "model/llama_like/llama_like_model.hpp"
#ifndef PIE_CUDA_QWEN_ONLY
#include "model/llama_like/mistral3.hpp"
#endif
#include "model/llama_like/qwen3.hpp"
#ifndef PIE_CUDA_QWEN_ONLY
#include "model/mixtral/gpt_oss.hpp"
#include "model/mixtral/mixtral.hpp"
#include "model/mixtral/mixtral_model.hpp"
#include "model/nemotron_h/nemotron_h.hpp"
#include "model/nemotron_h/nemotron_h_forward.hpp"
#include "model/nemotron_h/nemotron_h_model.hpp"
#endif
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/qwen3_5/qwen3_5_config.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"
#include "model/qwen3_5/qwen3_5_model.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "model/qwen3_5/qwen3_5_moe_model.hpp"
#ifndef PIE_CUDA_QWEN_ONLY
#include "model/qwen3_vl/qwen3_vl.hpp"
#include "model/qwen3_vl/qwen3_vl_model.hpp"
#endif

namespace pie_cuda_driver::model {

std::size_t ModelPlan::workspace_bytes(const HfConfig& cfg, int max_tokens,
                                       int output_rows) const {
    return ::pie_cuda_driver::model::workspace_bytes(
        cfg, max_tokens, output_rows, cfg.intermediate_size,
        cfg.num_attention_heads * cfg.head_dim,
        cfg.num_key_value_heads * cfg.head_dim);
}

const char* family_name(Family family) noexcept {
    switch (family) {
    case Family::LlamaLike:  return "llama_like";
    case Family::Gemma:      return "gemma";
    case Family::Gemma4:     return "gemma4";
    case Family::Gemma3n:    return "gemma3n";
    case Family::Mixtral:    return "mixtral";
    case Family::Qwen3_5:    return "qwen3_5";
    case Family::Qwen3_5Moe: return "qwen3_5_moe";
    case Family::NemotronH:  return "nemotron_h";
    case Family::Kimi:       return "kimi";
    case Family::DeepSeekV4: return "deepseek_v4";
    case Family::Glm5:       return "glm5";
    case Family::Qwen3VL:    return "qwen3_vl";
    case Family::Csm:        return "csm";
    }
    return "unknown";
}

namespace {

// ── Concrete plans, one per family ──────────────────────────────────────
// Each owns exactly its own concrete weights struct (+ same-family
// adapters) by value. `plan_info()` reports the narrow pre-construction
// surface Context reads; nothing here is ever downcast outside this file.

class LlamaLikePlan final : public ModelPlan {
public:
    explicit LlamaLikePlan(Qwen3Weights w) : weights(std::move(w)) {
        info.family = Family::LlamaLike;
        info.num_layers = weights.layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    Qwen3Weights weights;
    PlanInfo info;
};

#ifndef PIE_CUDA_QWEN_ONLY
class GemmaPlan final : public ModelPlan {
public:
    explicit GemmaPlan(Gemma2Weights w) : weights(std::move(w)) {
        info.family = Family::Gemma;
        info.num_layers = weights.layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    Gemma2Weights weights;
    PlanInfo info;
};

class Gemma4Plan final : public ModelPlan {
public:
    Gemma4Plan(Gemma4Weights w, std::optional<Gemma4VisionWeights> v,
              std::optional<Gemma4AudioWeights> a)
        : weights(std::move(w)), vision(std::move(v)), audio(std::move(a)) {
        info.family = Family::Gemma4;
        info.num_layers = weights.layers.size();
        info.per_layer_intermediate = weights.per_layer_intermediate;
        info.per_layer_head_dim = weights.per_layer_head_dim;
        info.per_layer_num_kv_heads = weights.per_layer_num_kv_heads;
        info.kv_source_layer = weights.kv_source_layer;
        if (vision.has_value()) {
            info.has_vision = true;
            info.gemma4_pool_kernel = vision->config.pooling_kernel_size;
            const auto* positions = vision->patch_position_embedding;
            if (positions != nullptr && positions->shape().size() > 1 &&
                positions->shape()[1] <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                info.gemma4_position_table = static_cast<int>(positions->shape()[1]);
            }
        }
        if (audio.has_value()) {
            info.has_audio = true;
            info.audio_mel_bins = 128;
        }
    }
    const PlanInfo& plan_info() const override { return info; }

    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        std::size_t bytes = ModelPlan::workspace_bytes(cfg, max_tokens, output_rows);
        if (cfg.gemma4_enable_moe) {
            bytes += gemma4_moe_workspace_bytes(cfg, max_tokens);
        }
        return bytes;
    }

    Gemma4Weights weights;
    std::optional<Gemma4VisionWeights> vision;
    std::optional<Gemma4AudioWeights> audio;
    PlanInfo info;
};

class Gemma3nPlan final : public ModelPlan {
public:
    explicit Gemma3nPlan(Gemma3nWeights w) : weights(std::move(w)) {
        info.family = Family::Gemma3n;
        info.num_layers = weights.layers.size();
        info.per_layer_intermediate = weights.per_layer_intermediate;
    }
    const PlanInfo& plan_info() const override { return info; }

    Gemma3nWeights weights;
    PlanInfo info;
};

class MixtralPlan final : public ModelPlan {
public:
    explicit MixtralPlan(MixtralWeights w) : weights(std::move(w)) {
        info.family = Family::Mixtral;
        info.num_layers = weights.layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    MixtralWeights weights;
    PlanInfo info;
};
#endif

class Qwen3_5Plan final : public ModelPlan {
public:
    explicit Qwen3_5Plan(Qwen3_5Weights w) : weights(std::move(w)) {
        info.family = Family::Qwen3_5;
        info.num_layers = weights.layers.size();
        info.layer_is_linear_attn.resize(info.num_layers);
        for (std::size_t l = 0; l < info.num_layers; ++l) {
            info.layer_is_linear_attn[l] =
                (weights.layers[l].kind == Qwen3_5LayerWeights::Kind::LinearAttn);
        }
        info.has_mtp = weights.mtp.has_value();
    }
    const PlanInfo& plan_info() const override { return info; }

    // NOTE: this hook's signature (matching `IModel::workspace_bytes`)
    // doesn't carry `tp_size`; the memory planner's own candidate sweep
    // (`store/memory_planner.cpp`) still computes the exact figure with the
    // real runtime tp_size via its own family switch. This override
    // exists so the plan-level hook is a real formula rather than a
    // decorative default, at the cost of a tp_size=1 approximation.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        return ModelPlan::workspace_bytes(cfg, max_tokens, output_rows) +
               qwen3_5_la_workspace_bytes(cfg, max_tokens);
    }

    Qwen3_5Weights weights;
    PlanInfo info;
};

class Qwen3_5MoePlan final : public ModelPlan {
public:
    explicit Qwen3_5MoePlan(Qwen3_5MoeWeights w) : weights(std::move(w)) {
        info.family = Family::Qwen3_5Moe;
        info.num_layers = weights.layers.size();
        info.layer_is_linear_attn.resize(info.num_layers);
        for (std::size_t l = 0; l < info.num_layers; ++l) {
            info.layer_is_linear_attn[l] =
                (weights.layers[l].kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn);
        }
        info.has_mtp = weights.mtp.has_value();
    }
    const PlanInfo& plan_info() const override { return info; }

    // Same tp_size=1 approximation caveat as `Qwen3_5Plan` above.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        return ModelPlan::workspace_bytes(cfg, max_tokens, output_rows) +
               qwen3_5_la_workspace_bytes(cfg, max_tokens) +
               qwen3_5_moe_workspace_bytes(cfg, max_tokens);
    }

    Qwen3_5MoeWeights weights;
    PlanInfo info;
};

#ifndef PIE_CUDA_QWEN_ONLY
class NemotronHPlan final : public ModelPlan {
public:
    explicit NemotronHPlan(NemotronHWeights w) : weights(std::move(w)) {
        info.family = Family::NemotronH;
        info.num_layers = weights.layers.size();
        info.layer_is_mamba.resize(info.num_layers);
        for (std::size_t l = 0; l < info.num_layers; ++l) {
            info.layer_is_mamba[l] =
                (weights.layers[l].kind == NemotronHLayerWeights::Kind::Mamba);
        }
    }
    const PlanInfo& plan_info() const override { return info; }

    // Same tp_size=1 approximation caveat as `Qwen3_5Plan` above.
    std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                                int output_rows) const override {
        return ModelPlan::workspace_bytes(cfg, max_tokens, output_rows) +
               nemotron_h_workspace_bytes(cfg, max_tokens, /*tp_size=*/1);
    }

    NemotronHWeights weights;
    PlanInfo info;
};

class KimiPlan final : public ModelPlan {
public:
    explicit KimiPlan(KimiWeights w) : weights(std::move(w)) {
        info.family = Family::Kimi;
        info.num_layers = weights.layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    KimiWeights weights;
    PlanInfo info;
};

class DeepSeekV4Plan final : public ModelPlan {
public:
    explicit DeepSeekV4Plan(DsV4Weights w) : weights(std::move(w)) {
        info.family = Family::DeepSeekV4;
        info.num_layers = weights.layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    DsV4Weights weights;
    PlanInfo info;
};

class Glm5Plan final : public ModelPlan {
public:
    explicit Glm5Plan(Glm5Weights w) : weights(std::move(w)) {
        info.family = Family::Glm5;
        info.num_layers = weights.layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    Glm5Weights weights;
    PlanInfo info;
};

class Qwen3VLPlan final : public ModelPlan {
public:
    Qwen3VLPlan(Qwen3Weights w, std::optional<Qwen3VLVisionWeights> v)
        : weights(std::move(w)), vision(std::move(v)) {
        info.family = Family::Qwen3VL;
        info.num_layers = weights.layers.size();
        if (vision.has_value()) {
            info.has_vision = true;
            const auto& vc = vision->config;
            const std::int64_t patch_dim =
                static_cast<std::int64_t>(vc.in_channels) *
                vc.temporal_patch_size * vc.patch_size * vc.patch_size;
            const std::int64_t merge_unit =
                static_cast<std::int64_t>(vc.spatial_merge_size) *
                vc.spatial_merge_size;
            if (patch_dim > 0 &&
                patch_dim <= std::numeric_limits<int>::max() &&
                merge_unit > 0 &&
                merge_unit <= std::numeric_limits<int>::max()) {
                info.qwen3_vl_patch_dim = static_cast<int>(patch_dim);
                info.qwen3_vl_merge_unit = static_cast<int>(merge_unit);
            }
        }
    }
    const PlanInfo& plan_info() const override { return info; }

    Qwen3Weights weights;
    std::optional<Qwen3VLVisionWeights> vision;
    PlanInfo info;
};

class CsmPlan final : public ModelPlan {
public:
    explicit CsmPlan(CsmWeights w) : weights(std::move(w)) {
        info.family = Family::Csm;
        info.num_layers = weights.backbone_layers.size();
    }
    const PlanInfo& plan_info() const override { return info; }

    CsmWeights weights;
    PlanInfo info;
};
#endif

// ── Config validation hooks ──────────────────────────────────────────────
// Most rows only need the generic dimension sanity check; the parser
// (`model/config.cpp`) already enforces per-field requirements, so this is
// defense-in-depth, not a parsing rewrite. CSM additionally requires its
// optional config section (the one case where a missing section would
// otherwise surface as a `bind_csm` throw deep inside binding instead of a
// clear registry-level rejection).

std::optional<std::string> default_validate_config(const HfConfig& cfg) {
    if (cfg.num_hidden_layers <= 0) {
        return std::string("num_hidden_layers must be > 0");
    }
    if (cfg.hidden_size <= 0) {
        return std::string("hidden_size must be > 0");
    }
    return std::nullopt;
}

#ifndef PIE_CUDA_QWEN_ONLY
std::optional<std::string> validate_csm_config(const HfConfig& cfg) {
    if (auto err = default_validate_config(cfg)) return err;
    if (!cfg.csm.has_value()) {
        return std::string(
            "csm: HfConfig.csm not populated (config.json missing "
            "depth_decoder_config/codec_config?)");
    }
    return std::nullopt;
}
#endif

// ── Binders: one per distinct C++ bind function, wrapped into the owning
//    plan for its family. Kind-sharing archs point their own row at one
//    of these; nothing here falls through to another family. ──────────

std::unique_ptr<ModelPlan> bind_row_llama_like(LoadedModel& engine, bool verbose) {
    return std::make_unique<LlamaLikePlan>(bind_llama_like(engine, verbose));
}
#ifndef PIE_CUDA_QWEN_ONLY
std::unique_ptr<ModelPlan> bind_row_mistral3(LoadedModel& engine, bool) {
    return std::make_unique<LlamaLikePlan>(bind_mistral3(engine));
}
std::unique_ptr<ModelPlan> bind_row_phi3(LoadedModel& engine, bool) {
    return std::make_unique<LlamaLikePlan>(bind_phi3(engine));
}
std::unique_ptr<ModelPlan> bind_row_olmo3(LoadedModel& engine, bool) {
    return std::make_unique<LlamaLikePlan>(bind_olmo3(engine));
}
std::unique_ptr<ModelPlan> bind_row_mixtral(LoadedModel& engine, bool) {
    return std::make_unique<MixtralPlan>(bind_mixtral(engine));
}
std::unique_ptr<ModelPlan> bind_row_gpt_oss(LoadedModel& engine, bool) {
    return std::make_unique<MixtralPlan>(bind_gpt_oss(engine));
}
std::unique_ptr<ModelPlan> bind_row_gemma2(LoadedModel& engine, bool) {
    return std::make_unique<GemmaPlan>(bind_gemma2(engine));
}
std::unique_ptr<ModelPlan> bind_row_gemma3(LoadedModel& engine, bool) {
    return std::make_unique<GemmaPlan>(bind_gemma3(engine));
}
std::unique_ptr<ModelPlan> bind_row_gemma4(LoadedModel& engine, bool verbose) {
    Gemma4Weights w = bind_gemma4(engine);
    std::optional<Gemma4VisionWeights> vision;
    std::optional<Gemma4AudioWeights> audio;
    // Multimodal Gemma-4: bind the vision tower too. Its tensors are
    // already in the weight store (not stripped).
    if (engine.hf_config().gemma_vision.has_value()) {
        vision = bind_gemma4_vision(engine);
        if (verbose) {
            std::fprintf(stderr,
                "[gemma4] bound vision tower (%zu layers)\n",
                vision->layers.size());
        }
    }
    // Multimodal Gemma-4: bind the audio tower (USM/Conformer) too,
    // independent of `vision` — a checkpoint may carry both.
    if (engine.hf_config().gemma_audio.has_value()) {
        audio = bind_gemma4_audio(engine);
        if (verbose) {
            std::fprintf(stderr,
                "[gemma4] bound audio tower (%zu layers)\n",
                audio->layers.size());
        }
    }
    return std::make_unique<Gemma4Plan>(std::move(w), std::move(vision), std::move(audio));
}
std::unique_ptr<ModelPlan> bind_row_gemma3n(LoadedModel& engine, bool) {
    return std::make_unique<Gemma3nPlan>(bind_gemma3n(engine));
}
std::unique_ptr<ModelPlan> bind_row_nemotron_h(LoadedModel& engine, bool) {
    return std::make_unique<NemotronHPlan>(bind_nemotron_h(engine));
}
std::unique_ptr<ModelPlan> bind_row_deepseek_v4(LoadedModel& engine, bool) {
    return std::make_unique<DeepSeekV4Plan>(bind_deepseek_v4(engine));
}
std::unique_ptr<ModelPlan> bind_row_kimi(LoadedModel& engine, bool) {
    return std::make_unique<KimiPlan>(bind_kimi(engine));
}
std::unique_ptr<ModelPlan> bind_row_glm5(LoadedModel& engine, bool) {
    return std::make_unique<Glm5Plan>(bind_glm5(engine));
}
#endif
std::unique_ptr<ModelPlan> bind_row_qwen3_5(LoadedModel& engine, bool) {
    return std::make_unique<Qwen3_5Plan>(bind_qwen3_5(engine));
}
std::unique_ptr<ModelPlan> bind_row_qwen3_5_moe(LoadedModel& engine, bool) {
    return std::make_unique<Qwen3_5MoePlan>(bind_qwen3_5_moe(engine));
}
#ifndef PIE_CUDA_QWEN_ONLY
std::unique_ptr<ModelPlan> bind_row_qwen3_vl(LoadedModel& engine, bool verbose) {
    Qwen3Weights text = bind_qwen3_vl_text(engine);
    std::optional<Qwen3VLVisionWeights> vision;
    if (engine.hf_config().qwen3_vl_vision.has_value()) {
        vision = bind_qwen3_vl_vision(engine);
        if (verbose) {
            std::fprintf(stderr,
                "[qwen3_vl] bound vision tower (%zu layers, %zu deepstack)\n",
                vision->layers.size(), vision->deepstack.size());
        }
    }
    return std::make_unique<Qwen3VLPlan>(std::move(text), std::move(vision));
}
std::unique_ptr<ModelPlan> bind_row_csm(LoadedModel& engine, bool verbose) {
    return std::make_unique<CsmPlan>(bind_csm(engine, verbose));
}
#endif

// ── Model factories: one per family. Each is the sole place that
//    downcasts `ModelPlan*` to its concrete type — always safe because
//    the registry table below wires each `create_model` to run only on
//    plans built by that same family's `bind_row_*` above. ──────────────

template <typename Concrete>
Concrete& plan_cast(ModelPlan& base, const char* family_label) {
    auto* p = dynamic_cast<Concrete*>(&base);
    if (p == nullptr) {
        throw std::runtime_error(
            std::string("model registry: ") + family_label +
            " create_model received a mismatched ModelPlan (registry table "
            "mis-wired)");
    }
    return *p;
}

std::unique_ptr<IModel> create_llama_like_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<LlamaLikePlan>(*plan_base, "llama_like");
    return std::make_unique<LlamaLikeModel>(
        std::move(plan.weights), *res.hf_config, *res.kv_cache,
        *res.llama_fwd_cfg);
}

#ifndef PIE_CUDA_QWEN_ONLY
std::unique_ptr<IModel> create_gemma_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<GemmaPlan>(*plan_base, "gemma");
    return std::make_unique<Gemma2Model>(
        std::move(plan.weights), *res.hf_config, *res.gemma_fwd_cfg);
}

std::unique_ptr<IModel> create_gemma4_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<Gemma4Plan>(*plan_base, "gemma4");
    return std::make_unique<Gemma4Model>(
        std::move(plan.weights), *res.hf_config, *res.gemma4_moe_ws,
        *res.kv_cache, *res.gemma4_fwd_cfg, res.small_spec_graph_tokens,
        std::move(plan.vision), std::move(plan.audio));
}

std::unique_ptr<IModel> create_gemma3n_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<Gemma3nPlan>(*plan_base, "gemma3n");
    Gemma3nForwardCfg fwd_cfg{};
    fwd_cfg.final_logit_softcap = res.hf_config->gemma_final_logit_softcap;
    fwd_cfg.tp_size = res.tp_size;
    fwd_cfg.tp_comm = res.tp_comm;
    return std::make_unique<Gemma3nModel>(
        std::move(plan.weights), *res.hf_config, fwd_cfg);
}

std::unique_ptr<IModel> create_mixtral_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<MixtralPlan>(*plan_base, "mixtral");
    return std::make_unique<MixtralModel>(
        std::move(plan.weights), *res.hf_config, *res.llama_fwd_cfg,
        res.hf_config->num_experts, res.hf_config->num_experts_per_tok);
}

std::unique_ptr<IModel> create_nemotron_h_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<NemotronHPlan>(*plan_base, "nemotron_h");
    return std::make_unique<NemotronHModel>(
        std::move(plan.weights), *res.hf_config, *res.nemotron_h_ws,
        *res.nemotron_h_state_cache, *res.kv_cache, *res.llama_fwd_cfg,
        res.tp_size, res.tp_comm);
}

std::unique_ptr<IModel> create_deepseek_v4_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<DeepSeekV4Plan>(*plan_base, "deepseek_v4");
    return std::make_unique<DsV4Model>(
        std::move(plan.weights), *res.hf_config, *res.dsv4_ws, res.tp_size,
        res.tp_rank, res.tp_comm, /*emit_logits=*/true);
}

std::unique_ptr<IModel> create_kimi_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<KimiPlan>(*plan_base, "kimi");
    return std::make_unique<KimiModel>(
        std::move(plan.weights), *res.hf_config, *res.kimi_ws, *res.mla_cache,
        res.tp_size, res.tp_comm, /*emit_logits=*/true);
}

std::unique_ptr<IModel> create_glm5_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<Glm5Plan>(*plan_base, "glm5");
    return std::make_unique<Glm5Model>(
        std::move(plan.weights), *res.hf_config, *res.glm5_ws, *res.mla_cache,
        *res.dsa_cache, res.tp_size, res.tp_comm, /*emit_logits=*/true);
}
#endif

std::unique_ptr<IModel> create_qwen3_5_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<Qwen3_5Plan>(*plan_base, "qwen3_5");
    const HfConfig& hf = *res.hf_config;
    const bool force_prefill_path = !flashinfer_decode_supports_gqa(
        hf.num_attention_heads / std::max(1, hf.num_key_value_heads));
    const int small_spec_tokens = qwen35_small_spec_graph_tokens();
    const bool graph_safe =
        res.kv_cache->format().is_native_bf16() &&
        !qwen35_forward_profile_enabled();
    const bool supports_small_prefill_graph =
        res.kv_cache->format().is_native_bf16() &&
        !res.kv_cache->hnd_layout() && small_spec_tokens > 0;
    auto model = std::make_unique<Qwen35Model>(
        std::move(plan.weights), hf, *res.qwen3_5_la_ws,
        *res.qwen3_5_state_cache, *res.qwen3_5_plan_state, *res.kv_cache,
        res.tp_size, res.tp_comm, force_prefill_path, small_spec_tokens,
        graph_safe, supports_small_prefill_graph);
    if (plan.info.has_mtp && res.native_mtp_num_drafts > 0 &&
        res.system_drafter != nullptr) {
        model->wire_system_drafter(
            *res.system_drafter, res.native_mtp_num_drafts,
            qwen35_mtp_draft_position_offset(),
            qwen35_mtp_prefix_global_cache(),
            qwen35_mtp_fused_gemv_enabled());
    }
    return model;
}

std::unique_ptr<IModel> create_qwen3_5_moe_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<Qwen3_5MoePlan>(*plan_base, "qwen3_5_moe");
    const HfConfig& hf = *res.hf_config;
    const bool force_prefill_path = !flashinfer_decode_supports_gqa(
        hf.num_attention_heads / std::max(1, hf.num_key_value_heads));
    const int small_spec_tokens = qwen35_small_spec_graph_tokens();
    const bool graph_safe = res.kv_cache->format().is_native_bf16();
    const bool supports_small_prefill_graph =
        res.kv_cache->format().is_native_bf16() &&
        !res.kv_cache->hnd_layout() && small_spec_tokens > 0;
    auto model = std::make_unique<Qwen35MoeModel>(
        std::move(plan.weights), hf, *res.qwen3_5_la_ws, *res.qwen3_5_moe_ws,
        *res.qwen3_5_state_cache, *res.qwen3_5_plan_state, *res.kv_cache,
        res.tp_size, res.tp_comm, force_prefill_path, small_spec_tokens,
        graph_safe, supports_small_prefill_graph);
    if (plan.info.has_mtp && res.native_mtp_num_drafts > 0 &&
        res.system_drafter != nullptr) {
        model->wire_system_drafter(
            *res.system_drafter, res.native_mtp_num_drafts,
            qwen35_mtp_draft_position_offset(),
            qwen35_mtp_prefix_global_cache());
    }
    return model;
}

#ifndef PIE_CUDA_QWEN_ONLY
std::unique_ptr<IModel> create_qwen3_vl_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources& res) {
    auto& plan = plan_cast<Qwen3VLPlan>(*plan_base, "qwen3_vl");
    return std::make_unique<Qwen3VLModel>(
        std::move(plan.weights), *res.hf_config, *res.kv_cache,
        *res.llama_fwd_cfg, res.max_workspace_tokens, std::move(plan.vision));
}

std::unique_ptr<IModel> create_csm_model(
    std::unique_ptr<ModelPlan> plan_base, ModelResources&) {
    auto& plan = plan_cast<CsmPlan>(*plan_base, "csm");
    return std::make_unique<CsmModel>(std::move(plan.weights));
}
#endif

// ── The table itself ─────────────────────────────────────────────────────

std::vector<ArchEntry> build_arch_table() {
    std::vector<ArchEntry> t;
#ifdef PIE_CUDA_QWEN_ONLY
    t.reserve(6);
#else
    t.reserve(33);
#endif

    auto push = [&t](const char* model_type, Family family, const char* binder_key,
                     std::function<std::optional<std::string>(const HfConfig&)> validate,
                     std::function<std::unique_ptr<ModelPlan>(LoadedModel&, bool)> bind,
                     std::function<std::unique_ptr<IModel>(std::unique_ptr<ModelPlan>, ModelResources&)> create) {
        ArchEntry entry;
        entry.model_type = model_type;
        entry.family = family;
        entry.binder_key = binder_key;
        entry.validate_config = std::move(validate);
        entry.bind = std::move(bind);
        entry.create_model = std::move(create);
        t.push_back(std::move(entry));
    };

    // ── LlamaLike: dense/GQA decoder. Four distinct binders share the
    //    family (plain, Mistral-3, Phi-3, OLMo-3); "qwen3"/"qwen2"/"llama"/
    //    "llama3"/"mistral" (bare) all use the same `bind_llama_like`.
#ifdef PIE_CUDA_QWEN_ONLY
    push("qwen3", Family::LlamaLike, "llama_like", default_validate_config,
         bind_row_llama_like, create_llama_like_model);
#else
    for (const char* mt : {"qwen3", "qwen2", "llama", "llama3", "mistral"}) {
        push(mt, Family::LlamaLike, "llama_like", default_validate_config,
             bind_row_llama_like, create_llama_like_model);
    }
    for (const char* mt : {"mistral3", "ministral3"}) {
        push(mt, Family::LlamaLike, "mistral3", default_validate_config,
             bind_row_mistral3, create_llama_like_model);
    }
    push("phi3", Family::LlamaLike, "phi3", default_validate_config,
         bind_row_phi3, create_llama_like_model);
    for (const char* mt : {"olmo2", "olmo3"}) {
        push(mt, Family::LlamaLike, "olmo3", default_validate_config,
             bind_row_olmo3, create_llama_like_model);
    }
#endif

#ifndef PIE_CUDA_QWEN_ONLY
    // ── Mixtral: sparse top-k MoE. gpt_oss shares the family/model class
    //    through its own binder (MXFP4 experts + attention sinks).
    push("mixtral", Family::Mixtral, "mixtral", default_validate_config,
         bind_row_mixtral, create_mixtral_model);
    push("gpt_oss", Family::Mixtral, "gpt_oss", default_validate_config,
         bind_row_gpt_oss, create_mixtral_model);

    // ── Gemma 2 / 3 (share Gemma2Weights + gemma2_forward_paged).
    push("gemma2", Family::Gemma, "gemma2", default_validate_config,
         bind_row_gemma2, create_gemma_model);
    for (const char* mt : {"gemma3", "gemma3_text"}) {
        push(mt, Family::Gemma, "gemma3", default_validate_config,
             bind_row_gemma3, create_gemma_model);
    }

    // ── Gemma 4 (dense/MoE, + optional vision/audio towers).
    for (const char* mt : {"gemma4", "gemma4_text"}) {
        push(mt, Family::Gemma4, "gemma4", default_validate_config,
             bind_row_gemma4, create_gemma4_model);
    }

    // ── Gemma 3n (AltUp/Laurel/PLE "Nano" family).
    for (const char* mt : {"gemma3n", "gemma3n_text"}) {
        push(mt, Family::Gemma3n, "gemma3n", default_validate_config,
             bind_row_gemma3n, create_gemma3n_model);
    }

    // ── Nemotron-H (Mamba2 / attention / MoE hybrid).
    push("nemotron_h", Family::NemotronH, "nemotron_h", default_validate_config,
         bind_row_nemotron_h, create_nemotron_h_model);

    // ── DeepSeek-V4 (hypercompressed streams, own family/plan).
    push("deepseek_v4", Family::DeepSeekV4, "deepseek_v4",
         default_validate_config, bind_row_deepseek_v4,
         create_deepseek_v4_model);

    // ── Kimi: DeepSeek-V2/V3-style MLA + Kimi-K2, one binder/family for
    //    all three aliases.
    for (const char* mt : {"deepseek_v2", "deepseek_v3", "kimi_k2"}) {
        push(mt, Family::Kimi, "kimi", default_validate_config,
             bind_row_kimi, create_kimi_model);
    }

    // ── GLM-5.1 (MLA + DSA indexer + routed/shared MoE).
    push("glm_moe_dsa", Family::Glm5, "glm5", default_validate_config,
         bind_row_glm5, create_glm5_model);
#endif

    // ── Qwen3.5 hybrid dense (linear-attn + full-attn + optional MTP).
    for (const char* mt : {"qwen3_5", "qwen3_5_text"}) {
        push(mt, Family::Qwen3_5, "qwen3_5", default_validate_config,
             bind_row_qwen3_5, create_qwen3_5_model);
    }

    // ── Qwen3.5 hybrid MoE (same hybrid attention + sparse MoE MLP).
    for (const char* mt : {"qwen3_5_moe", "qwen3_5_moe_text", "qwen3_moe"}) {
        push(mt, Family::Qwen3_5Moe, "qwen3_5_moe", default_validate_config,
             bind_row_qwen3_5_moe, create_qwen3_5_moe_model);
    }

#ifndef PIE_CUDA_QWEN_ONLY
    // ── Qwen3-VL: Qwen3 text tower (binds into the same `Qwen3Weights`
    //    shape as LlamaLike) + optional ViT vision tower with DeepStack.
    for (const char* mt : {"qwen3_vl", "qwen3_vl_text"}) {
        push(mt, Family::Qwen3VL, "qwen3_vl_text", default_validate_config,
             bind_row_qwen3_vl, create_qwen3_vl_model);
    }

    // ── CSM: native audio-output backbone + depth decoder + Mimi codec.
    //    The only family whose config validation hook is non-trivial.
    push("csm", Family::Csm, "csm", validate_csm_config, bind_row_csm,
         create_csm_model);
#endif

    return t;
}

}  // namespace

const std::vector<ArchEntry>& arch_table() {
    static const std::vector<ArchEntry> table = build_arch_table();
    return table;
}

const ArchEntry* find_arch_entry(const std::string& model_type) {
    for (const auto& entry : arch_table()) {
        if (entry.model_type == model_type) return &entry;
    }
    return nullptr;
}

}  // namespace pie_cuda_driver::model
