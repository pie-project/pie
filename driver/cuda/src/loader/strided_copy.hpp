#pragma once

// Strided host->device copy of a checkpoint tensor (the non-compact ExtentWrite
// / Encode-source path). A loader-source primitive shared by the storage
// executor and the transcode engine; depends only on the checkpoint source.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <cstring>

#include <cuda_runtime.h>

#include "pie_native/storage_program.hpp"
#include "tensor.hpp"
#include "loader/checkpoint_source.hpp"

namespace pie_cuda_driver {

inline void copy_strided_extent_to_device(
    CheckpointSource& loader,
    const pie_weight_loader::PieLoaderStorageInstrView& instr,
    void* dst,
    const std::vector<std::int64_t>& dst_shape) {
    const auto& extent = instr.source.stride;
    if (extent.dims.len != dst_shape.size()) {
        throw std::runtime_error(
            "storage executor: strided source rank mismatch");
    }
    std::uint64_t physical_bytes = extent.element_bytes;
    std::uint64_t elements = 1;
    for (std::size_t axis = 0; axis < extent.dims.len; ++axis) {
        const auto& dim = extent.dims.ptr[axis];
        if (dim.count < 0 || dim.src_stride < 0 ||
            dim.count != dst_shape[axis]) {
            throw std::runtime_error(
                "storage executor: invalid strided source geometry");
        }
        const std::uint64_t count = static_cast<std::uint64_t>(dim.count);
        if (count != 0) {
            physical_bytes += (count - 1) *
                static_cast<std::uint64_t>(dim.src_stride);
        }
        if (count != 0 &&
            elements > std::numeric_limits<std::uint64_t>::max() / count) {
            throw std::runtime_error(
                "storage executor: strided element count overflow");
        }
        elements *= count;
    }
    const std::uint64_t compact_bytes =
        elements * static_cast<std::uint64_t>(extent.element_bytes);
    if (compact_bytes != instr.source.span_bytes) {
        throw std::runtime_error(
            "storage executor: strided compact byte count mismatch");
    }
    const auto* source = loader.storage_host_ptr(
        instr.source.file_id,
        instr.source.file_offset + extent.base_offset,
        physical_bytes);
    std::vector<std::uint8_t> compact(
        static_cast<std::size_t>(compact_bytes));
    for (std::uint64_t linear = 0; linear < elements; ++linear) {
        std::uint64_t remaining = linear;
        std::uint64_t source_offset = 0;
        for (std::size_t axis = extent.dims.len; axis > 0; --axis) {
            const std::uint64_t count =
                static_cast<std::uint64_t>(extent.dims.ptr[axis - 1].count);
            const std::uint64_t index = count == 0 ? 0 : remaining % count;
            remaining = count == 0 ? remaining : remaining / count;
            source_offset += index *
                static_cast<std::uint64_t>(
                    extent.dims.ptr[axis - 1].src_stride);
        }
        std::memcpy(
            compact.data() + linear * extent.element_bytes,
            source + source_offset,
            extent.element_bytes);
    }
    CUDA_CHECK(cudaMemcpy(
        dst,
        compact.data(),
        compact.size(),
        cudaMemcpyHostToDevice));
}

}  // namespace pie_cuda_driver
