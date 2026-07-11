#pragma once

// SafetensorsWeightSource — backs charlie's model::WeightSource seam with
// MLX's native safetensors reader. Tensors load straight into
// mlx::core::array (on the default device), so no host->device copy or
// dtype reinterpretation is needed: the bf16/fp16/fp32 arrays the file
// stores become the exact Tensors the graph builders bind.
//
// Sharding: if `model.safetensors.index.json` is present its `weight_map`
// enumerates the shard files; otherwise a single `model.safetensors`
// (or any *.safetensors in the dir) is loaded.

#include <string>
#include <unordered_map>

#include "../model/weights.hpp"
#include "../ops/tensor.hpp"

namespace pie_metal_driver::loader {

class SafetensorsWeightSource final : public model::WeightSource {
public:
    // Loads every shard under `hf_path` (an HF snapshot directory) eagerly.
    explicit SafetensorsWeightSource(const std::string& hf_path);

    Tensor get(const std::string& hf_name) const override;
    std::optional<Tensor> try_get(const std::string& hf_name) const override;
    bool has(const std::string& hf_name) const override;

    std::size_t size() const { return tensors_.size(); }

private:
    std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace pie_metal_driver::loader
