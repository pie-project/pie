#ifndef PIE_WEIGHT_LOADER_CPP_HPP
#define PIE_WEIGHT_LOADER_CPP_HPP

#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "weight_loader.h"

namespace pie_weight_loader::cpp {

inline std::string bytes_to_string(PieLoaderBytes bytes)
{
    if (bytes.ptr == nullptr || bytes.len == 0) return {};
    return std::string(
        reinterpret_cast<const char*>(bytes.ptr),
        reinterpret_cast<const char*>(bytes.ptr) + bytes.len);
}

inline std::vector<std::int64_t> i64_slice_to_vector(PieLoaderI64Slice shape)
{
    if (shape.ptr == nullptr || shape.len == 0) return {};
    return std::vector<std::int64_t>(shape.ptr, shape.ptr + shape.len);
}

inline std::vector<std::uint32_t> buffer_id_slice_to_vector(
    PieLoaderBufferIdSlice ids)
{
    if (ids.ptr == nullptr || ids.len == 0) return {};
    return std::vector<std::uint32_t>(ids.ptr, ids.ptr + ids.len);
}

inline std::vector<std::int64_t> extent_shape(
    const PieLoaderStridedExtentView& extent)
{
    std::vector<std::int64_t> shape;
    shape.reserve(extent.dims.len);
    for (std::size_t i = 0; i < extent.dims.len; ++i) {
        shape.push_back(extent.dims.ptr[i].count);
    }
    return shape;
}

inline bool compact_extent(const PieLoaderStridedExtentView& extent)
{
    std::int64_t stride = static_cast<std::int64_t>(extent.element_bytes);
    for (std::size_t i = extent.dims.len; i > 0; --i) {
        const auto& dim = extent.dims.ptr[i - 1];
        if (dim.src_stride != stride || dim.dst_stride != stride) {
            return false;
        }
        stride *= dim.count;
    }
    return true;
}

inline std::uint64_t extent_bytes(
    const PieLoaderStridedExtentView& extent,
    const char* context)
{
    std::uint64_t elements = 1;
    for (std::size_t i = 0; i < extent.dims.len; ++i) {
        const auto count = extent.dims.ptr[i].count;
        if (count < 0) {
            throw std::runtime_error(
                std::string(context) + ": negative extent dimension");
        }
        const auto ucount = static_cast<std::uint64_t>(count);
        if (ucount != 0 &&
            elements > std::numeric_limits<std::uint64_t>::max() / ucount) {
            throw std::runtime_error(
                std::string(context) + ": extent element count overflow");
        }
        elements *= ucount;
    }
    if (extent.element_bytes != 0 &&
        elements >
            std::numeric_limits<std::uint64_t>::max() /
                extent.element_bytes) {
        throw std::runtime_error(
            std::string(context) + ": extent byte count overflow");
    }
    return elements * extent.element_bytes;
}

class StorageProgramIndex {
public:
    explicit StorageProgramIndex(std::string context)
        : context_(std::move(context))
    {}

    void reset(const PieLoaderStorageProgramView& program)
    {
        instr_by_id_.clear();
        buffer_by_id_.clear();
        tensor_by_id_.clear();
        instr_by_id_.reserve(program.instrs.len);
        buffer_by_id_.reserve(program.buffers.len);
        tensor_by_id_.reserve(program.tensors.len);
        for (std::size_t i = 0; i < program.instrs.len; ++i) {
            instr_by_id_.emplace(
                program.instrs.ptr[i].id,
                &program.instrs.ptr[i]);
        }
        for (std::size_t i = 0; i < program.buffers.len; ++i) {
            buffer_by_id_.emplace(
                program.buffers.ptr[i].id,
                &program.buffers.ptr[i]);
        }
        for (std::size_t i = 0; i < program.tensors.len; ++i) {
            tensor_by_id_.emplace(
                program.tensors.ptr[i].id,
                &program.tensors.ptr[i]);
        }
    }

    const PieLoaderStorageInstrView& instruction(std::uint32_t id) const
    {
        const auto it = instr_by_id_.find(id);
        if (it != instr_by_id_.end()) return *it->second;
        throw std::runtime_error(context_ + ": instruction id out of range");
    }

    const PieLoaderBufferDeclView& buffer(std::uint32_t id) const
    {
        const auto it = buffer_by_id_.find(id);
        if (it != buffer_by_id_.end()) return *it->second;
        throw std::runtime_error(context_ + ": buffer id out of range");
    }

    const PieLoaderTensorDeclView& tensor(std::uint32_t id) const
    {
        const auto it = tensor_by_id_.find(id);
        if (it != tensor_by_id_.end()) return *it->second;
        throw std::runtime_error(context_ + ": tensor id out of range");
    }

private:
    std::string context_;
    std::unordered_map<std::uint32_t, const PieLoaderStorageInstrView*>
        instr_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderBufferDeclView*>
        buffer_by_id_;
    std::unordered_map<std::uint32_t, const PieLoaderTensorDeclView*>
        tensor_by_id_;
};

}  // namespace pie_weight_loader::cpp

#endif  // PIE_WEIGHT_LOADER_CPP_HPP
