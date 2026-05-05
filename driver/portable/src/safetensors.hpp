#pragma once

// Minimal mmap-based safetensors reader.
//
// Format (little-endian):
//   [u64 header_size]
//   [header_size bytes JSON metadata]
//   [tensor data, packed]
//
// Each entry in the JSON metadata maps a tensor name to:
//   { "dtype": "F16"|"BF16"|"F32"|"I32"|...,
//     "shape": [d0, d1, ...],
//     "data_offsets": [begin, end] }     // relative to tensor data block
//
// A reserved key "__metadata__" carries arbitrary author metadata and is
// ignored here.
//
// Multi-shard models ship `model.safetensors.index.json` which maps tensor
// names → shard filename (e.g. "model-00001-of-00002.safetensors"). The
// SafetensorsArchive class loads the index when present, or falls back to
// a single `model.safetensors` file.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace pie_portable_driver {

enum class StDtype : std::uint8_t {
    F32,
    F16,
    BF16,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    BOOL,
    F8_E4M3,
    F8_E5M2,
};

std::size_t st_dtype_size(StDtype dt);
const char* st_dtype_name(StDtype dt);

struct StTensor {
    StDtype dtype;
    std::vector<std::int64_t> shape;        // HF order: outermost first.
    // Pointer into the mmap'd shard (safetensors) or the mmap'd GGUF
    // file. Valid for the lifetime of the owning archive.
    const std::uint8_t* data;
    std::size_t nbytes;

    // GGUF: when set (>= 0), specifies the ggml tensor type directly
    // (Q4_K, Q5_K, F16, ...). The `dtype` field is ignored when this is
    // set; `shape` is still kept in HF order so the existing shape
    // reversal in `new_tensor_from_st` produces the right ggml ne[]
    // layout. Safetensors path leaves this at -1.
    std::int32_t ggml_type_override = -1;
};

class SafetensorsShard {
public:
    explicit SafetensorsShard(const std::filesystem::path& path);
    ~SafetensorsShard();

    SafetensorsShard(const SafetensorsShard&) = delete;
    SafetensorsShard& operator=(const SafetensorsShard&) = delete;

    SafetensorsShard(SafetensorsShard&&) noexcept;
    SafetensorsShard& operator=(SafetensorsShard&&) noexcept;

    const std::unordered_map<std::string, StTensor>& tensors() const noexcept {
        return tensors_;
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
#ifdef _WIN32
    std::vector<std::uint8_t> owned_data_;
#else
    int fd_ = -1;
#endif
    std::size_t mmap_size_ = 0;
    const std::uint8_t* base_ = nullptr;
    std::unordered_map<std::string, StTensor> tensors_;

    void close_mmap() noexcept;
};

// Polymorphic weight archive. Two implementations:
//   * SafetensorsArchive — HF native, F32 / F16 / BF16
//   * GGUFArchive       — llama.cpp GGUF, plus all ggml quant types
// Both expose tensors by HF-style canonical names (the GGUF archive
// translates `blk.{N}.attn_q.weight` → `model.layers.{N}.self_attn.q_proj.weight`
// internally so the per-arch loaders don't branch).
class WeightArchive {
public:
    virtual ~WeightArchive() = default;
    virtual const StTensor* find(const std::string& name) const noexcept = 0;
    virtual const StTensor& at(const std::string& name) const = 0;
    virtual std::size_t num_tensors() const noexcept = 0;
};

// Top-level safetensors archive: handles single-file or sharded models.
// Construct from a HuggingFace snapshot directory; auto-discovers layout.
class SafetensorsArchive : public WeightArchive {
public:
    explicit SafetensorsArchive(const std::filesystem::path& snapshot_dir);

    const StTensor* find(const std::string& name) const noexcept override;
    const StTensor& at(const std::string& name) const override;
    std::size_t num_tensors() const noexcept override;

    // Iterate all (name, tensor) pairs across all shards.
    template <typename F>
    void for_each(F&& fn) const {
        for (const auto& shard : shards_) {
            for (const auto& [name, t] : shard->tensors()) {
                fn(name, t);
            }
        }
    }

    std::size_t num_shards() const noexcept { return shards_.size(); }

private:
    std::vector<std::unique_ptr<SafetensorsShard>> shards_;
    // Composite index across all shards: name → shard index.
    std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace pie_portable_driver
