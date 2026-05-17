#include "loader/model_family.hpp"

namespace pie_cuda_driver {

bool is_qwen_moe_model_type(const std::string& model_type) noexcept
{
    return model_type == "qwen3_5_moe" ||
           model_type == "qwen3_5_moe_text" ||
           model_type == "qwen3_moe";
}

bool is_dense_llama_like_model_type(const std::string& model_type) noexcept
{
    return model_type == "qwen3" ||
           model_type == "qwen2" ||
           model_type == "llama" ||
           model_type == "llama3" ||
           model_type == "mistral" ||
           model_type == "mistral3" ||
           model_type == "ministral3" ||
           model_type == "olmo2" ||
           model_type == "olmo3";
}

ModelSchemaFamily model_schema_family_for_type(
    const std::string& model_type) noexcept
{
    if (model_type == "phi3") {
        return ModelSchemaFamily::Phi3;
    }
    if (is_qwen_moe_model_type(model_type)) {
        return ModelSchemaFamily::QwenMoe;
    }
    if (is_dense_llama_like_model_type(model_type)) {
        return ModelSchemaFamily::DenseLlamaLike;
    }
    return ModelSchemaFamily::Generic;
}

bool supports_dense_llama_packed_load(
    const HfConfig& hf,
    const Config& boot_cfg) noexcept
{
    return hf.quant_method.empty() &&
           boot_cfg.model.runtime_quant.empty() &&
           is_dense_llama_like_model_type(hf.model_type);
}

}  // namespace pie_cuda_driver
