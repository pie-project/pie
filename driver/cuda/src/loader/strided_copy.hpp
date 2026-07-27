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

#include "pie_loader/plan.hpp"
#include "tensor.hpp"
#include "pie_loader/checkpoint_source.hpp"

namespace pie_cuda_driver {

inline void copy_strided_extent_to_device(
    pie_loader::CheckpointSource& loader,
    const pie_loader::PieLoaderSourceExtentView& src,
    void* dst,
    std::uint64_t dst_capacity_bytes) {
    const auto& extent = src.stride;
    std::uint64_t physical_bytes = extent.element_bytes;
    std::uint64_t elements = 1;
    for (std::size_t axis = 0; axis < extent.dims.len; ++axis) {
        const auto& dim = extent.dims.ptr[axis];
        if (dim.count < 0 || dim.src_stride < 0) {
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
    if (compact_bytes != src.span_bytes) {
        throw std::runtime_error(
            "storage executor: strided compact byte count mismatch");
    }
    // The gathered run lands contiguously at `dst`, so the destination
    // constraint is a byte bound, not a shape. It cannot be a shape: one
    // instruction writes a SUB-REGION of its destination buffer (a single
    // expert's shard of a stacked MoE weight, say), so the buffer's own
    // extent has neither the rank nor the extents of what is being written.
    if (compact_bytes > dst_capacity_bytes) {
        throw std::runtime_error(
            "storage executor: strided write of " +
            std::to_string(compact_bytes) + " bytes overflows its " +
            std::to_string(dst_capacity_bytes) + "-byte destination");
    }
    const auto* source = loader.storage_host_ptr(
        src.file_id,
        src.file_offset + extent.base_offset,
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
