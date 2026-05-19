#pragma once

// Streaming safetensors loader.
//
// File format:
//   [u64 LE: header_size]
//   [header_size bytes: JSON object — { name -> {dtype, shape, data_offsets[2]} }]
//   [tensor data, concatenated]
//
// We mmap the file, slice into its header, and copy checkpoint bytes into
// caller-owned device storage via cudaMemcpyAsync. Allocation and runtime
// representation decisions live in the compiled storage program, not here.
// Sharded models (`model.safetensors.index.json`) are transparently split
// across files.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "loader/checkpoint_source.hpp"

namespace pie_cuda_driver {

class SafetensorsCheckpointSource : public CheckpointSource {
public:
    /// Open either `<snapshot_dir>/model.safetensors` (single file) or
    /// `<snapshot_dir>/model.safetensors.index.json` (sharded). The shard
    /// files are mmap'd lazily on first tensor access.
    static SafetensorsCheckpointSource open(const std::filesystem::path& snapshot_dir);

    SafetensorsCheckpointSource() = default;
    ~SafetensorsCheckpointSource();

    SafetensorsCheckpointSource(const SafetensorsCheckpointSource&) = delete;
    SafetensorsCheckpointSource& operator=(const SafetensorsCheckpointSource&) = delete;
    SafetensorsCheckpointSource(SafetensorsCheckpointSource&&) noexcept = default;
    SafetensorsCheckpointSource& operator=(SafetensorsCheckpointSource&&) noexcept = default;

    /// All weight names found across all shards.
    std::vector<std::string> tensor_names() const override;

    /// Total number of weights across all shards.
    std::size_t num_tensors() const noexcept override { return index_.size(); }

    /// Total bytes across all shards (storage size).
    std::uint64_t total_bytes() const noexcept { return total_bytes_; }

    /// Look up a tensor's metadata. Throws if not found.
    const TensorInfo& info(const std::string& name) const override;

    bool contains(const std::string& name) const noexcept override {
        return index_.find(name) != index_.end();
    }

    /// Storage storage location for a checkpoint tensor. `file_offset` is an
    /// absolute byte offset into `path`, suitable for pread/cuFileRead.
    TensorStorageInfo storage_info(const std::string& name) const override;

    /// Copy a full checkpoint tensor into caller-owned device storage.
    void copy_to_device(
        const std::string& name,
        void* dst,
        const std::vector<std::int64_t>& dst_shape);

    /// Copy an explicit byte range from a shard into caller-owned device
    /// storage. `file_offset` is absolute within the shard file.
    void copy_storage_bytes_to_device(
        std::uint32_t shard_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst);

    /// Copy a slice of `name` along `axis`, keeping only this rank's portion
    /// of the world. Used by layout-plan materialization to shard linear weights
    /// directly into their final runtime allocations.
    ///
    /// - 1-D tensors (biases): `axis` must be 0.
    /// - 2-D tensors (linear weights): `axis ∈ {0, 1}`.
    /// - The sharded dimension must be divisible by `world_size`.
    /// - `axis < 0` (or `world_size == 1`) falls through to the unsharded copy.
    void copy_shard_to_device(
        const std::string& name,
        int axis,
        int rank,
        int world_size,
        void* dst,
        const std::vector<std::int64_t>& dst_shape);

    /// Copy an arbitrary rectangular slice from the row-major checkpoint
    /// tensor into a compact device tensor. Omitted axes are copied in full.
    /// This is the generic primitive used by architecture lowering for
    /// fused QKV, MoE expert bands, and tensor-parallel local shards.
    void copy_slice_to_device(
        const std::string& name,
        int axis,
        std::int64_t start,
        std::int64_t length,
        void* dst,
        const std::vector<std::int64_t>& dst_shape);
    void copy_strided_to_device(
        const std::string& name,
        const std::vector<TensorSlice>& slices,
        void* dst,
        const std::vector<std::int64_t>& dst_shape);
    void copy_strided_to_device_async(
        const std::string& name,
        const std::vector<TensorSlice>& slices,
        void* dst,
        const std::vector<std::int64_t>& dst_shape,
        void* stream);

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
