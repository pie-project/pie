#include "model/model_graph.hpp"

#include <stdexcept>
#include <string>

#include "model/arch.hpp"
#include "model/gemma.hpp"
#include "model/llama_like.hpp"

namespace pie_metal_driver::model {

std::unique_ptr<ModelGraph> make_model_graph(const ModelConfig& cfg,
                                             ModelWeights weights) {
    // Data-driven dispatch: one runtime switch on the arch detected from the
    // model's config. Llama-like archs share one builder; Gemma has its own.
    switch (cfg.arch) {
        case PieArch::Llama3:
        case PieArch::Qwen2:
        case PieArch::Qwen3:
        case PieArch::Mistral3:
        case PieArch::Qwen3Moe:
        case PieArch::Mixtral:
            return std::make_unique<LlamaLikeGraph>(cfg, std::move(weights));

        case PieArch::Gemma2:
        case PieArch::Gemma3:
            return std::make_unique<GemmaGraph>(cfg, std::move(weights));

        case PieArch::Unknown:
        default:
            throw std::runtime_error(
                std::string("make_model_graph: unsupported architecture '") +
                pie_arch_name(cfg.arch) + "'");
    }
}

}  // namespace pie_metal_driver::model
