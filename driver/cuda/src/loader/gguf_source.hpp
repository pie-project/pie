#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "loader/checkpoint_source.hpp"

namespace pie_cuda_driver {

// Metadata source for GGUF checkpoints. This class owns only container
// metadata and byte extents; model-name dialect mapping belongs in
// ModelAdapter/SemanticGraph, and runtime layout belongs in RuntimeABI.
class GgufCheckpointSource final : public CheckpointSource {
public:
    static GgufCheckpointSource open(const std::filesystem::path& path);

    std::vector<std::string> tensor_names() const override { return names_; }
    std::size_t num_tensors() const noexcept override {
        return tensors_.size();
    }
    const TensorInfo& info(const std::string& name) const override;
    bool contains(const std::string& name) const noexcept override {
        return tensors_.find(name) != tensors_.end();
    }
    TensorStorageInfo storage_info(const std::string& name) const override;

    std::uint32_t version() const noexcept { return version_; }
    std::uint64_t alignment() const noexcept { return alignment_; }

private:
    std::filesystem::path path_;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = 32;
    std::vector<std::string> names_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, TensorStorageInfo> storage_;
};

std::vector<float> decode_gguf_q4_0_block(
    const std::uint8_t* block,
    std::size_t bytes);

}  // namespace pie_cuda_driver
