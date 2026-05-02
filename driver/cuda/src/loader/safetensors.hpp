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

    /// Load a slice of `name` along `axis`, keeping only this rank's portion
    /// of the world. Used by tensor-parallel binders to shard linear weights
    /// at load time so each GPU only allocates its `1 / world_size` share.
    ///
    /// - 1-D tensors (biases): `axis` must be 0.
    /// - 2-D tensors (linear weights): `axis ∈ {0, 1}`.
    /// - The sharded dimension must be divisible by `world_size`.
    /// - `axis < 0` (or `world_size == 1`) falls through to the unsharded
    ///   `load_to_device` path.
    DeviceTensor load_to_device_sharded(const std::string& name,
                                        int axis, int rank, int world_size);

    /// Load a contiguous row range `[row_offset, row_offset + rows)` of a
    /// 2-D tensor, then shard *that range* along axis=0 across `world_size`,
    /// returning this rank's slice. Used to unfuse phi-3-style fused
    /// `qkv_proj.weight = [Hq | Hk | Hk, H]` into per-rank Q/K/V tensors
    /// without ever materialising the full fused weight on a rank — naive
    /// axis-0 sharding of the fused tensor would straddle the Q/K/V
    /// block boundaries and is wrong.
    /// Requires `rows % world_size == 0`. Returns a `[rows / world_size, H]`
    /// owned device tensor.
    DeviceTensor load_to_device_row_range_sharded(
        const std::string& name,
        std::int64_t row_offset, std::int64_t rows,
        int rank, int world_size);

    /// Load an MoE fused-experts gate_up tensor `[E, 2*Im, H]` and shard
    /// each expert's `[gate(Im) | up(Im), H]` block along the Im axis,
    /// returning `[E, 2*Im_local, H]` where `Im_local = Im / world_size`.
    /// Reads strided slices straight from the mmap into the per-rank
    /// device buffer — never materialises the full fused tensor on the
    /// rank, which is what allowed Qwen3.6-35B-A3B to OOM at TP=2.
    DeviceTensor load_to_device_moe_gate_up_sharded(
        const std::string& name, int rank, int world_size);

    /// Load an MoE fused-experts down_proj tensor `[E, H, Im]` and shard
    /// it along the Im axis, returning `[E, H, Im_local]` where
    /// `Im_local = Im / world_size`. Same direct-from-mmap strategy as
    /// `load_to_device_moe_gate_up_sharded`. Caller is responsible for
    /// the cross-rank all-reduce after the down-proj GEMM.
    DeviceTensor load_to_device_moe_down_sharded(
        const std::string& name, int rank, int world_size);

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
