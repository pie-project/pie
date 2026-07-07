#include "arch.hpp"

namespace pie_metal_driver::model {

const char* pie_arch_name(PieArch a) {
    switch (a) {
        case PieArch::Llama3:    return "llama3";
        case PieArch::Qwen2:     return "qwen2";
        case PieArch::Qwen3:     return "qwen3";
        case PieArch::Mistral3:  return "mistral3";
        case PieArch::Qwen3Moe:  return "qwen3_moe";
        case PieArch::Mixtral:   return "mixtral";
        case PieArch::Gemma2:    return "gemma2";
        case PieArch::Gemma3:    return "gemma3";
        case PieArch::Gemma4:    return "gemma4";
        case PieArch::Qwen36:    return "qwen3.6";
        case PieArch::Unknown:   return "unknown";
    }
    return "unknown";
}

PieArch hf_model_type_to_pie_arch(const std::string& t) {
    // Dense Llama-like.
    if (t == "llama")      return PieArch::Llama3;
    if (t == "qwen2")      return PieArch::Qwen2;
    if (t == "qwen3")      return PieArch::Qwen3;
    if (t == "mistral" ||
        t == "mistral3")   return PieArch::Mistral3;

    // Llama-like MoE.
    if (t == "qwen3_moe")  return PieArch::Qwen3Moe;
    if (t == "mixtral")    return PieArch::Mixtral;

    // Gemma dense.
    if (t == "gemma2")     return PieArch::Gemma2;
    if (t == "gemma3" ||
        t == "gemma3_text") return PieArch::Gemma3;
    if (t == "gemma4" ||
        t == "gemma4_text") return PieArch::Gemma4;

    // Qwen3.6 (Qwen3.5 hybrid linear-attn family). The cuda reference handles
    // this family under the qwen3_5* model_type; the published checkpoint may
    // also carry the literal "qwen3_6"/"qwen3.6". Accept all spellings.
    if (t == "qwen3_5" || t == "qwen3_5_text" ||
        t == "qwen3_5_moe" || t == "qwen3_5_moe_text" ||
        t == "qwen3_6" || t == "qwen3.6" || t == "qwen3_6_moe")
        return PieArch::Qwen36;

    return PieArch::Unknown;
}

bool is_llama_like(PieArch a) {
    switch (a) {
        case PieArch::Llama3:
        case PieArch::Qwen2:
        case PieArch::Qwen3:
        case PieArch::Mistral3:
        case PieArch::Qwen3Moe:
        case PieArch::Mixtral:
            return true;
        default:
            return false;
    }
}

}  // namespace pie_metal_driver::model
