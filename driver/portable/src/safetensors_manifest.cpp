#include "pie_driver_common/safetensors_manifest.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pie_driver_common {

SafetensorsManifest discover_safetensors_manifest(
    const std::filesystem::path& snapshot_dir,
    SafetensorsLayoutPreference preference) {
    const auto index_path = snapshot_dir / "model.safetensors.index.json";
    const auto single_path = snapshot_dir / "model.safetensors";

    SafetensorsManifest manifest;
    if (preference == SafetensorsLayoutPreference::SingleFile &&
        std::filesystem::exists(single_path)) {
        manifest.shard_paths.push_back(single_path);
        manifest.source_path = single_path;
        manifest.sharded = false;
        return manifest;
    }

    if (std::filesystem::exists(index_path)) {
        std::ifstream f(index_path);
        if (!f) {
            throw std::runtime_error(
                "safetensors: cannot open " + index_path.string());
        }
        nlohmann::json idx = nlohmann::json::parse(f);
        if (!idx.contains("weight_map") || !idx["weight_map"].is_object()) {
            throw std::runtime_error("safetensors: index missing 'weight_map'");
        }

        std::vector<std::string> shard_names;
        for (auto it = idx["weight_map"].begin();
             it != idx["weight_map"].end(); ++it) {
            const auto shard = it.value().get<std::string>();
            if (std::find(shard_names.begin(), shard_names.end(), shard) ==
                shard_names.end()) {
                shard_names.push_back(shard);
            }
        }
        std::sort(shard_names.begin(), shard_names.end());
        manifest.shard_paths.reserve(shard_names.size());
        for (const auto& shard : shard_names) {
            manifest.shard_paths.push_back(snapshot_dir / shard);
        }
        manifest.source_path = index_path;
        manifest.sharded = true;
        return manifest;
    }

    if (std::filesystem::exists(single_path)) {
        manifest.shard_paths.push_back(single_path);
        manifest.source_path = single_path;
        manifest.sharded = false;
        return manifest;
    }

    throw std::runtime_error(
        "safetensors: no model.safetensors[.index.json] in " +
        snapshot_dir.string());
}

}  // namespace pie_driver_common
