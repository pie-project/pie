#include "safetensors_source.hpp"

#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <mlx/io.h>

namespace pie_metal_driver::loader {

namespace fs = std::filesystem;

namespace {

// Collect the set of shard files to load for an HF snapshot directory.
std::vector<fs::path> resolve_shards(const fs::path& dir) {
    const fs::path index = dir / "model.safetensors.index.json";
    if (fs::exists(index)) {
        std::ifstream in(index);
        if (!in) throw std::runtime_error("cannot open " + index.string());
        nlohmann::json j;
        in >> j;
        std::set<std::string> files;
        if (j.contains("weight_map") && j["weight_map"].is_object()) {
            for (const auto& [_, v] : j["weight_map"].items()) {
                files.insert(v.get<std::string>());
            }
        }
        std::vector<fs::path> shards;
        for (const auto& f : files) shards.push_back(dir / f);
        if (!shards.empty()) return shards;
    }

    const fs::path single = dir / "model.safetensors";
    if (fs::exists(single)) return {single};

    // Fallback: any *.safetensors in the directory.
    std::vector<fs::path> shards;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".safetensors") shards.push_back(e.path());
    }
    if (shards.empty()) {
        throw std::runtime_error("no .safetensors weights found under " + dir.string());
    }
    return shards;
}

}  // namespace

SafetensorsWeightSource::SafetensorsWeightSource(const std::string& hf_path) {
    const fs::path dir(hf_path);
    for (const auto& shard : resolve_shards(dir)) {
        // MLX loads each tensor straight into an mlx::core::array on the
        // default device, preserving the stored dtype (bf16/fp16/fp32).
        auto loaded = mlx::core::load_safetensors(shard.string());
        for (auto& [name, arr] : loaded.first) {
            tensors_.emplace(name, std::move(arr));
        }
    }
    if (tensors_.empty()) {
        throw std::runtime_error("safetensors under " + hf_path + " contained no tensors");
    }
}

Tensor SafetensorsWeightSource::get(const std::string& hf_name) const {
    auto it = tensors_.find(hf_name);
    if (it == tensors_.end()) {
        throw std::runtime_error("weight not found in safetensors: " + hf_name);
    }
    return it->second;
}

std::optional<Tensor> SafetensorsWeightSource::try_get(const std::string& hf_name) const {
    auto it = tensors_.find(hf_name);
    if (it == tensors_.end()) return std::nullopt;
    return it->second;
}

bool SafetensorsWeightSource::has(const std::string& hf_name) const {
    return tensors_.find(hf_name) != tensors_.end();
}

}  // namespace pie_metal_driver::loader
