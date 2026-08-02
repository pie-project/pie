#pragma once

// Strided host->device copy of a checkpoint tensor (the non-compact ExtentWrite
// / Encode-source path). A loader-source primitive shared by the storage
// executor and the transcode engine; depends only on the checkpoint source.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <memory>
#include <limits>
#include <cstring>

#include <cuda_runtime.h>

#include "pie_loader/plan.hpp"
#include "cuda_check.hpp"
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

    // A one-dimensional extent is a pitched rectangle -- `element_bytes` wide,
    // `count` rows, `src_stride` apart -- which is exactly what cudaMemcpy2D
    // takes, so the gather below is the driver's job and not ours. Measured on
    // an RTX 4090 over a 32 MiB payload, the crossover is sharp and sits at 16
    // bytes: below it the per-row DMA descriptor dominates and the host gather
    // wins, above it cudaMemcpy2D pulls away (5.3x on a 7168-byte row, which is
    // one Llama-70B tp8 `down_proj` shard). Overlapping rows cannot be a
    // rectangle, hence the pitch check.
    constexpr std::uint64_t kMemcpy2dMinRowBytes = 16;
    if (extent.dims.len == 1 && extent.element_bytes >= kMemcpy2dMinRowBytes &&
        static_cast<std::uint64_t>(extent.dims.ptr[0].src_stride) >=
            extent.element_bytes) {
        CUDA_CHECK(cudaMemcpy2D(
            dst,
            extent.element_bytes,
            source,
            static_cast<std::size_t>(extent.dims.ptr[0].src_stride),
            extent.element_bytes,
            static_cast<std::size_t>(extent.dims.ptr[0].count),
            cudaMemcpyHostToDevice));
        return;
    }

    // Every byte of the staging buffer is overwritten by the gather, so it must
    // not be value-initialised: `std::vector<std::uint8_t> buf(n)` zero-fills,
    // which cost more than the gather itself (15 ms against 4 ms for a 56 MiB
    // shard). `new[]` default-initialises, which for a trivial type is nothing.
    std::unique_ptr<std::uint8_t[]> compact(
        new std::uint8_t[static_cast<std::size_t>(compact_bytes)]);
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
            compact.get() + linear * extent.element_bytes,
            source + source_offset,
            extent.element_bytes);
    }
    CUDA_CHECK(cudaMemcpy(
        dst,
        compact.get(),
        static_cast<std::size_t>(compact_bytes),
        cudaMemcpyHostToDevice));
}

}  // namespace pie_cuda_driver
