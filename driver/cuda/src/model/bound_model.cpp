#include "model/bound_model.hpp"

#include <string>

namespace pie_cuda_driver::model {

std::size_t BoundCudaModel::num_layers() const noexcept {
    switch (kind) {
    case Kind::LlamaLike: return llama.layers.size();
    case Kind::Gemma: return gemma.layers.size();
    case Kind::Gemma4: return gemma4.layers.size();
    case Kind::Gemma3n: return gemma3n.layers.size();
    case Kind::Mixtral: return mixtral.layers.size();
    case Kind::Qwen3_5: return qwen3_5.layers.size();
    case Kind::Qwen3_5Moe: return qwen3_5_moe.layers.size();
    case Kind::NemotronH: return nemotron_h.layers.size();
    case Kind::Kimi: return kimi.layers.size();
    case Kind::DeepSeekV4: return deepseek_v4.layers.size();
    case Kind::Glm5: return glm5.layers.size();
    case Kind::Qwen3VL: return llama.layers.size();
    case Kind::Csm: return csm.backbone_layers.size();
    }
    return 0;
}

BoundCudaModel bind_cuda_model(LoadedModel& engine, bool verbose) {
    const std::string& mt = engine.hf_config().model_type;
    BoundCudaModel bound;

    const bool is_gemma4 =
        (mt == "gemma4" || mt == "gemma4_text");
    const bool is_gemma3n =
        (mt == "gemma3n" || mt == "gemma3n_text");
    const bool is_gpt_oss = (mt == "gpt_oss");
    const bool is_qwen3_5 =
        (mt == "qwen3_5" || mt == "qwen3_5_text");
    const bool is_qwen3_5_moe =
        (mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text" || mt == "qwen3_moe");
    const bool is_nemotron_h = (mt == "nemotron_h");
    const bool is_qwen3_vl = (mt == "qwen3_vl" || mt == "qwen3_vl_text");

    if (is_qwen3_vl) {
        bound.kind = BoundCudaModel::Kind::Qwen3VL;
        bound.llama = bind_qwen3_vl_text(engine);
        if (engine.hf_config().qwen3_vl_vision.has_value()) {
            bound.qwen3_vl_vision = bind_qwen3_vl_vision(engine);
            bound.has_vision = true;
            if (verbose) {
                std::fprintf(stderr,
                    "[qwen3_vl] bound vision tower (%zu layers, %zu deepstack)\n",
                    bound.qwen3_vl_vision.layers.size(),
                    bound.qwen3_vl_vision.deepstack.size());
            }
        }
    } else if (mt == "csm") {
        bound.kind = BoundCudaModel::Kind::Csm;
        bound.csm = bind_csm(engine, verbose);
    } else if (mt == "deepseek_v4") {
        bound.kind = BoundCudaModel::Kind::DeepSeekV4;
        bound.deepseek_v4 = bind_deepseek_v4(engine);
    } else if (mt == "kimi_k2" || mt == "deepseek_v2" || mt == "deepseek_v3") {
        bound.kind = BoundCudaModel::Kind::Kimi;
        bound.kimi = bind_kimi(engine);
    } else if (mt == "glm_moe_dsa") {
        bound.kind = BoundCudaModel::Kind::Glm5;
        bound.glm5 = bind_glm5(engine);
    } else if (mt == "phi3") {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_phi3(engine);
    } else if (mt == "olmo2" || mt == "olmo3") {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_olmo3(engine);
    } else if (mt == "mistral3" || mt == "ministral3") {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_mistral3(engine);
    } else if (mt == "gemma2") {
        bound.kind = BoundCudaModel::Kind::Gemma;
        bound.gemma = bind_gemma2(engine);
    } else if (mt == "gemma3" || mt == "gemma3_text") {
        bound.kind = BoundCudaModel::Kind::Gemma;
        bound.gemma = bind_gemma3(engine);
    } else if (is_gemma4) {
        bound.kind = BoundCudaModel::Kind::Gemma4;
        bound.gemma4 = bind_gemma4(engine);
        // Multimodal gemma-4: bind the vision tower too. Its tensors are
        // already in the weight store (not stripped). The encoder forward
        // (Phase 2.2) reads `bound.gemma4_vision`.
        if (engine.hf_config().gemma_vision.has_value()) {
            bound.gemma4_vision = bind_gemma4_vision(engine);
            bound.has_vision = true;
            if (verbose) {
                std::fprintf(stderr, "[gemma4] bound vision tower (%zu layers)\n",
                             bound.gemma4_vision.layers.size());
            }
        }
        // Multimodal gemma-4: bind the audio tower (USM/Conformer) too. Like
        // vision, its tensors load unstripped; the encoder forward scatters
        // audio soft tokens. See audio_frontend.md.
        if (engine.hf_config().gemma_audio.has_value()) {
            bound.gemma4_audio = bind_gemma4_audio(engine);
            bound.has_audio = true;
            if (verbose) {
                std::fprintf(stderr, "[gemma4] bound audio tower (%zu layers)\n",
                             bound.gemma4_audio.layers.size());
            }
        }
    } else if (is_gemma3n) {
        bound.kind = BoundCudaModel::Kind::Gemma3n;
        bound.gemma3n = bind_gemma3n(engine);
    } else if (is_gpt_oss) {
        bound.kind = BoundCudaModel::Kind::Mixtral;
        bound.mixtral = bind_gpt_oss(engine);
    } else if (mt == "mixtral") {
        bound.kind = BoundCudaModel::Kind::Mixtral;
        bound.mixtral = bind_mixtral(engine);
    } else if (is_qwen3_5) {
        bound.kind = BoundCudaModel::Kind::Qwen3_5;
        bound.qwen3_5 = bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe) {
        bound.kind = BoundCudaModel::Kind::Qwen3_5Moe;
        bound.qwen3_5_moe = bind_qwen3_5_moe(engine);
    } else if (is_nemotron_h) {
        bound.kind = BoundCudaModel::Kind::NemotronH;
        bound.nemotron_h = bind_nemotron_h(engine);
    } else {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_llama_like(engine, verbose);
    }

    return bound;
}

}  // namespace pie_cuda_driver::model
