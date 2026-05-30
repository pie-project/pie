#pragma once

// Strided host->device copy of a checkpoint tensor (the non-compact ExtentWrite
// / Encode-source path). A loader-source primitive shared by the storage
// executor and the transcode engine; depends only on the checkpoint source.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../weight_loader/include/weight_loader.h"
#include "tensor.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

inline void copy_strided_extent_to_device(
    SafetensorsCheckpointSource& loader,
    const std::vector<std::string>& source_tensor_names,
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        void* dst,
        const std::vector<std::int64_t>& dst_shape)
    {
        const std::string& name = source_tensor_names[instr.source.tensor_id];
        const TensorInfo& info = loader.info(name);
        const TensorStorageInfo storage = loader.storage_info(name);
        if (instr.source.file_offset < storage.file_offset) {
            throw std::runtime_error(
                "rust storage executor: strided source starts before tensor");
        }
        const auto rank = info.shape.size();
        if (instr.source.stride.dims.len != rank) {
            throw std::runtime_error(
                "rust storage executor: strided source rank mismatch for '" +
                name + "'");
        }

        std::vector<std::int64_t> dense_strides(rank, 1);
        std::int64_t stride = static_cast<std::int64_t>(dtype_bytes(info.dtype));
        for (std::size_t i = rank; i > 0; --i) {
            dense_strides[i - 1] = stride;
            stride *= info.shape[i - 1];
        }

        std::uint64_t relative =
            instr.source.file_offset - storage.file_offset +
            instr.source.stride.base_offset;
        std::vector<TensorSlice> slices;
        slices.reserve(rank);
        for (std::size_t axis = 0; axis < rank; ++axis) {
            const auto& dim = instr.source.stride.dims.ptr[axis];
            if (dim.src_stride != dense_strides[axis]) {
                throw std::runtime_error(
                    "rust storage executor: unsupported strided source layout "
                    "for '" + name + "'");
            }
            if (dim.count < 0 || dim.count > info.shape[axis]) {
                throw std::runtime_error(
                    "rust storage executor: strided source count out of range "
                    "for '" + name + "'");
            }
            const auto axis_stride =
                static_cast<std::uint64_t>(dense_strides[axis]);
            const std::int64_t start =
                axis_stride == 0
                    ? 0
                    : static_cast<std::int64_t>(relative / axis_stride);
            relative = axis_stride == 0 ? relative : relative % axis_stride;
            if (start < 0 || start + dim.count > info.shape[axis]) {
                throw std::runtime_error(
                    "rust storage executor: strided source offset out of "
                    "range for '" + name + "'");
            }
            if (start != 0 || dim.count != info.shape[axis]) {
                slices.push_back(TensorSlice{
                    .axis = static_cast<int>(axis),
                    .start = start,
                    .length = dim.count,
                });
            }
        }
        if (relative != 0) {
            throw std::runtime_error(
                "rust storage executor: strided source offset is not aligned "
                "to tensor strides for '" + name + "'");
        }
        loader.copy_strided_to_device(
            name,
            slices,
            dst,
            dst_shape);
    }

}  // namespace pie_cuda_driver
