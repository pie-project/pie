#pragma once

#include <filesystem>
#include <vector>

namespace pie_driver_common {

struct SafetensorsManifest {
    std::vector<std::filesystem::path> shard_paths;
    std::filesystem::path source_path;
    bool sharded = false;
};

enum class SafetensorsLayoutPreference {
    Index,
    SingleFile,
};

SafetensorsManifest discover_safetensors_manifest(
    const std::filesystem::path& snapshot_dir,
    SafetensorsLayoutPreference preference =
        SafetensorsLayoutPreference::Index);

}  // namespace pie_driver_common
