#pragma once

#include <cstddef>

#include "model/csm.hpp"
#include "model/gemma2.hpp"
#include "model/gemma3n.hpp"
#include "model/gemma4.hpp"
#include "model/glm5.hpp"
#include "model/gpt_oss.hpp"
#include "model/deepseek_v4.hpp"
#include "model/kimi.hpp"
#include "model/mistral3.hpp"
#include "model/mixtral.hpp"
#include "model/nemotron_h.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_vl.hpp"

namespace pie_cuda_driver::model {

struct BoundCudaModel {
    enum class Kind {
        LlamaLike,
        Gemma,
        Gemma4,
        Gemma3n,
        Mixtral,
        Qwen3_5,
        Qwen3_5Moe,
        NemotronH,
        Kimi,
        DeepSeekV4,
        Glm5,
        Qwen3VL,
        Csm,
    };

    Kind kind = Kind::LlamaLike;

    Qwen3Weights llama;
    Gemma2Weights gemma;
    Gemma4Weights gemma4;
    Gemma3nWeights gemma3n;
    MixtralWeights mixtral;
    Qwen3_5Weights qwen3_5;
    Qwen3_5MoeWeights qwen3_5_moe;
    NemotronHWeights nemotron_h;
    KimiWeights kimi;
    DsV4Weights deepseek_v4;
    Glm5Weights glm5;

    // CSM native-audio-output bound weights (backbone + depth + Mimi). Bound
    // only for `model_type == "csm"`. Move-only (owns resolved codebook embeds).
    CsmWeights csm;

    // Gemma-4 vision tower + projector. Bound only for multimodal gemma-4
    // checkpoints (`HfConfig.gemma_vision` present); the encoder forward
    // (MULTIMODAL.md Phase 2.2) consumes these. The vision tensors are already
    // in the weight store — the `vision_tower.` skip prefix does not match
    // gemma-4's `model.vision_tower.` names — so this only wires pointers.
    Gemma4VisionWeights gemma4_vision;
    bool has_vision = false;

    // Gemma-4 audio tower (`model.audio_tower.*` + `model.embed_audio.*`).
    // Bound only for multimodal gemma-4 checkpoints (`HfConfig.gemma_audio`
    // present). Independent of `has_vision` — a checkpoint may carry both.
    Gemma4AudioWeights gemma4_audio;
    bool has_audio = false;

    // Qwen3-VL vision tower (`model.visual.*`). Bound for qwen3_vl checkpoints;
    // the text tower binds into `llama` (standard Qwen3 schema). `has_vision`
    // is shared with the gemma4 path — only one is ever set per model.
    Qwen3VLVisionWeights qwen3_vl_vision;

    bool is_gemma() const noexcept { return kind == Kind::Gemma; }
    bool is_gemma4() const noexcept { return kind == Kind::Gemma4; }
    bool is_gemma3n() const noexcept { return kind == Kind::Gemma3n; }
    bool is_mixtral() const noexcept { return kind == Kind::Mixtral; }
    bool is_qwen3_5() const noexcept { return kind == Kind::Qwen3_5; }
    bool is_qwen3_5_moe() const noexcept { return kind == Kind::Qwen3_5Moe; }
    bool is_nemotron_h() const noexcept { return kind == Kind::NemotronH; }
    bool is_kimi() const noexcept { return kind == Kind::Kimi; }
    bool is_deepseek_v4() const noexcept { return kind == Kind::DeepSeekV4; }
    bool is_glm5() const noexcept { return kind == Kind::Glm5; }
    bool is_qwen3_vl() const noexcept { return kind == Kind::Qwen3VL; }
    bool is_csm() const noexcept { return kind == Kind::Csm; }
    bool is_llama_like() const noexcept { return kind == Kind::LlamaLike; }

    std::size_t num_layers() const noexcept;
};

BoundCudaModel bind_cuda_model(LoadedModel& engine, bool verbose);

}  // namespace pie_cuda_driver::model
