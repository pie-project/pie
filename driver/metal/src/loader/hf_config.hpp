#pragma once

// HF config.json → model::ModelConfig parser (delta owns the loader seam).
// Populates the field set the per-arch graph builders + arch_spec read. The
// parser is arch-agnostic: it resolves the architecture via
// hf_model_type_to_pie_arch and fills every hparam the metal forward passes
// consume, applying the same per-arch defaults as driver/cuda's hf_config.cpp.

#include <string>

#include "../model/config.hpp"

namespace pie_metal_driver::loader {

// Parse `<hf_path>/config.json`. `hf_path` is a directory (HF snapshot layout).
// Throws std::runtime_error if the file is missing/unparseable or required
// fields (hidden_size, num_hidden_layers, num_attention_heads) are absent.
model::ModelConfig parse_hf_config(const std::string& hf_path);

// Parse a config.json document already in memory (testing / embedded configs).
model::ModelConfig parse_hf_config_json(const std::string& json_text);

}  // namespace pie_metal_driver::loader
