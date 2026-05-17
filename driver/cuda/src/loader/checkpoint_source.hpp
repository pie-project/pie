#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace pie_cuda_driver {

struct TensorInfo {
    DType dtype;
    std::vector<std::int64_t> shape;
    std::string encoding;
    std::uint32_t block_elements = 0;
    std::uint32_t block_bytes = 0;
    // Offset into the source tensor data segment, when the container has one.
    std::uint64_t data_offset = 0;
    std::uint64_t nbytes = 0;
    std::uint32_t shard_id = 0;
};

struct TensorSlice {
    int axis = -1;
    std::int64_t start = 0;
    std::int64_t length = 0;
};

struct TensorStorageInfo {
    std::filesystem::path path;
    std::uint64_t file_offset = 0;
    std::uint64_t nbytes = 0;
    std::uint32_t shard_id = 0;
};

class CheckpointSource {
public:
    virtual ~CheckpointSource() = default;

    virtual std::vector<std::string> tensor_names() const = 0;
    virtual std::size_t num_tensors() const noexcept = 0;
    virtual const TensorInfo& info(const std::string& name) const = 0;
    virtual bool contains(const std::string& name) const noexcept = 0;
    virtual TensorStorageInfo storage_info(const std::string& name) const {
        const auto& ti = info(name);
        return TensorStorageInfo{
            .path = {},
            .file_offset = ti.data_offset,
            .nbytes = ti.nbytes,
            .shard_id = ti.shard_id,
        };
    }
};

}  // namespace pie_cuda_driver
