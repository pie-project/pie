#pragma once

#include <string>

#include "config.hpp"
#include "loader/hf_config.hpp"
#include "loader/model_schema.hpp"

namespace pie_cuda_driver {

bool is_qwen_moe_model_type(const std::string& model_type) noexcept;
bool is_dense_llama_like_model_type(const std::string& model_type) noexcept;
ModelSchemaFamily model_schema_family_for_type(
    const std::string& model_type) noexcept;
bool supports_dense_llama_packed_load(
    const HfConfig& hf,
    const Config& boot_cfg) noexcept;

}  // namespace pie_cuda_driver
