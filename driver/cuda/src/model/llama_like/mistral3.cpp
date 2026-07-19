#include "model/llama_like/mistral3.hpp"

#include <string>
#include <vector>

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("mistral3: missing weight '" + name + "'");
    }
    return e.get(name);
}

}  // namespace

Qwen3Weights bind_mistral3(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();

    // Mistral-Small-3.1 keeps norms and the embedding in bf16 (or
    // fp16); only Q/K/V/O and gate/up/down projections come as FP8.
    // For each layer, materialize bf16 copies under the canonical names
    // bind_llama_like expects.
    const auto suffix_pairs = std::vector<std::pair<std::string, std::string>>{
        {"self_attn.q_proj",    "self_attn.q_proj"},
        {"self_attn.k_proj",    "self_attn.k_proj"},
        {"self_attn.v_proj",    "self_attn.v_proj"},
        {"self_attn.o_proj",    "self_attn.o_proj"},
        {"mlp.gate_proj",       "mlp.gate_proj"},
        {"mlp.up_proj",         "mlp.up_proj"},
        {"mlp.down_proj",       "mlp.down_proj"},
    };

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        for (const auto& [src, dst] : suffix_pairs) {
            const std::string weight_name = p + dst + ".weight";
            if (!engine.has(weight_name) ||
                engine.get(weight_name).dtype() != DType::BF16) {
                throw std::runtime_error(
                    "mistral3: storage loader did not materialize bf16 weight '" +
                    weight_name + "'");
            }
        }
    }

    return bind_llama_like(engine);
}

}  // namespace pie_cuda_driver::model
