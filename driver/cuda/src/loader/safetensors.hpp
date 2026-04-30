#pragma once

// Streaming safetensors loader.
//
// File format:
//   [u64 LE: header_size]
//   [header_size bytes: JSON object — { name -> {dtype, shape, data_offsets[2]} }]
//   [tensor data, concatenated]
//
// We mmap the file, slice into its header, and copy each tensor straight
// into a `DeviceTensor` via `cudaMemcpyAsync`. No extra host-side staging,
// no torch dependency. Sharded models (`model.safetensors.index.json`) are
// transparently split across files.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace pie_cuda_driver {

struct TensorInfo {
    DType dtype;
    std::vector<std::int64_t> shape;
    // Offset into the shard's data segment (NOT into the file).
    std::uint64_t data_offset;
    std::uint64_t nbytes;
    // Index into SafetensorsLoader::shards_.
    std::uint32_t shard_id;
};

class SafetensorsLoader {
public:
    /// Open either `<snapshot_dir>/model.safetensors` (single file) or
    /// `<snapshot_dir>/model.safetensors.index.json` (sharded). The shard
    /// files are mmap'd lazily on first tensor access.
    static SafetensorsLoader open(const std::filesystem::path& snapshot_dir);

    SafetensorsLoader() = default;
    ~SafetensorsLoader();

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&&) noexcept = default;
    SafetensorsLoader& operator=(SafetensorsLoader&&) noexcept = default;

    /// All weight names found across all shards.
    std::vector<std::string> tensor_names() const;

    /// Total number of weights across all shards.
    std::size_t num_tensors() const noexcept { return index_.size(); }

    /// Total bytes across all shards (storage size).
    std::uint64_t total_bytes() const noexcept { return total_bytes_; }

    /// Look up a tensor's metadata. Throws if not found.
    const TensorInfo& info(const std::string& name) const;

    bool contains(const std::string& name) const noexcept {
        return index_.find(name) != index_.end();
    }

    /// Allocate a `DeviceTensor` and copy the weight into it via
    /// `cudaMemcpyAsync`. Synchronous from the caller's POV (the load path
    /// is one-shot at boot, not a hot path).
    DeviceTensor load_to_device(const std::string& name);

private:
    struct Shard {
        std::filesystem::path path;
        int fd = -1;
        std::size_t mapped_size = 0;
        const std::uint8_t* data = nullptr;     // mmap base
        std::uint64_t data_section_offset = 0;  // bytes from `data` to tensor #0
    };

    static void parse_shard_header_(
        Shard& s,
        std::uint32_t shard_id,
        std::unordered_map<std::string, TensorInfo>& index,
        std::uint64_t& total_bytes);

    void open_shard_(Shard& s) const;

    std::vector<Shard> shards_;
    std::unordered_map<std::string, TensorInfo> index_;
    std::uint64_t total_bytes_ = 0;
};

}  // namespace pie_cuda_driver
