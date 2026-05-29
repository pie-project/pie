#pragma once

#include <cstddef>

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
    bool is_llama_like() const noexcept { return kind == Kind::LlamaLike; }

    std::size_t num_layers() const noexcept;
};

BoundCudaModel bind_cuda_model(LoadedModel& engine, bool verbose);

}  // namespace pie_cuda_driver::model
