#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../weight_loader/include/weight_loader.h"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

class RustStorageProgramExecutor {
public:
    RustStorageProgramExecutor(
        SafetensorsCheckpointSource& loader,
        WeightStoreBuilder& weights,
        std::vector<std::string> source_tensor_names)
        : loader_(loader),
          weights_(weights),
          source_tensor_names_(std::move(source_tensor_names))
    {}

    LoadExecutionStats execute(
        const pie_weight_loader::PieLoaderStorageProgramView& program)
    {
        LoadExecutionStats stats;
        stats.planned_tensor_count = program.tensors.len;
        stats.planned_storage_peak_bytes = program.memory.persistent_bytes +
            program.memory.temporary_peak_bytes;
        stats.planned_storage_temp_bytes = program.memory.temporary_peak_bytes;

        for (std::size_t i = 0; i < program.schedule.len; ++i) {
            const std::uint32_t instr_id = program.schedule.ptr[i];
            const auto& instr = instruction(program, instr_id);
            switch (instr.kind) {
            case pie_weight_loader::PieLoaderStorageInstrKind::Allocate:
                allocate(program, instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::ExtentWrite:
                extent_write(instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Finalize:
                finalize(program, instr, stats);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::TileMap:
                throw std::runtime_error(
                    "rust storage executor: TileMap is not yet implemented "
                    "in the dense cutover path");
            case pie_weight_loader::PieLoaderStorageInstrKind::CreateView:
                create_view(program, instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
            case pie_weight_loader::PieLoaderStorageInstrKind::Release:
                throw std::runtime_error(
                    "rust storage executor: instruction kind is not yet "
                    "implemented in the dense cutover path");
            }
        }
        weights_.finalize();
        return stats;
    }

private:
    static std::string bytes_to_string(
        pie_weight_loader::PieLoaderBytes bytes)
    {
        if (bytes.ptr == nullptr || bytes.len == 0) return {};
        return std::string(
            reinterpret_cast<const char*>(bytes.ptr),
            reinterpret_cast<const char*>(bytes.ptr) + bytes.len);
    }

    static DType dtype_from_rust(pie_weight_loader::PieLoaderDType dtype)
    {
        switch (dtype) {
        case pie_weight_loader::PieLoaderDType::F32: return DType::FP32;
        case pie_weight_loader::PieLoaderDType::F16: return DType::FP16;
        case pie_weight_loader::PieLoaderDType::BF16: return DType::BF16;
        case pie_weight_loader::PieLoaderDType::F8E4M3:
            return DType::FP8_E4M3;
        case pie_weight_loader::PieLoaderDType::F8E5M2:
            return DType::FP8_E5M2;
        case pie_weight_loader::PieLoaderDType::I32: return DType::INT32;
        case pie_weight_loader::PieLoaderDType::I8: return DType::INT8;
        case pie_weight_loader::PieLoaderDType::U8:
        case pie_weight_loader::PieLoaderDType::Bool:
        case pie_weight_loader::PieLoaderDType::I16:
        case pie_weight_loader::PieLoaderDType::U16:
        case pie_weight_loader::PieLoaderDType::U32:
            return DType::UINT8;
        }
        return DType::UINT8;
    }

    static std::vector<std::int64_t> shape_from_slice(
        pie_weight_loader::PieLoaderI64Slice shape)
    {
        if (shape.ptr == nullptr || shape.len == 0) return {};
        return std::vector<std::int64_t>(shape.ptr, shape.ptr + shape.len);
    }

    static std::vector<std::int64_t> shape_from_extent(
        const pie_weight_loader::PieLoaderStridedExtentView& extent)
    {
        std::vector<std::int64_t> shape;
        shape.reserve(extent.dims.len);
        for (std::size_t i = 0; i < extent.dims.len; ++i) {
            shape.push_back(extent.dims.ptr[i].count);
        }
        return shape;
    }

    const pie_weight_loader::PieLoaderStorageInstrView& instruction(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        std::uint32_t id) const
    {
        for (std::size_t i = 0; i < program.instrs.len; ++i) {
            if (program.instrs.ptr[i].id == id) return program.instrs.ptr[i];
        }
        throw std::runtime_error(
            "rust storage executor: instruction id out of range");
    }

    const pie_weight_loader::PieLoaderBufferDeclView& buffer_decl(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        std::uint32_t id) const
    {
        for (std::size_t i = 0; i < program.buffers.len; ++i) {
            if (program.buffers.ptr[i].id == id) return program.buffers.ptr[i];
        }
        throw std::runtime_error(
            "rust storage executor: buffer id out of range");
    }

    const pie_weight_loader::PieLoaderTensorDeclView& tensor_decl(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        std::uint32_t id) const
    {
        for (std::size_t i = 0; i < program.tensors.len; ++i) {
            if (program.tensors.ptr[i].id == id) return program.tensors.ptr[i];
        }
        throw std::runtime_error(
                "rust storage executor: tensor id out of range");
    }

    static std::vector<std::uint32_t> ids_from_slice(
        pie_weight_loader::PieLoaderBufferIdSlice ids)
    {
        if (ids.ptr == nullptr || ids.len == 0) return {};
        return std::vector<std::uint32_t>(ids.ptr, ids.ptr + ids.len);
    }

    void allocate(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        const auto& buffer = buffer_decl(program, instr.buffer_id);
        if (!buffer.has_tensor) {
            throw std::runtime_error(
                "rust storage executor: temporary allocate is not yet "
                "implemented");
        }
        const auto& tensor = tensor_decl(program, buffer.tensor_id);
        buffers_.emplace(
            buffer.id,
            DeviceTensor::allocate(
                dtype_from_rust(tensor.dtype),
                shape_from_slice(tensor.shape)));
    }

    void extent_write(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (!instr.has_source || !instr.has_dest) {
            throw std::runtime_error(
                "rust storage executor: ExtentWrite missing source/dest");
        }
        if (instr.source.tensor_id >= source_tensor_names_.size()) {
            throw std::runtime_error(
                "rust storage executor: source tensor id out of range");
        }
        auto dst_it = buffers_.find(instr.dest.buffer_id);
        if (dst_it == buffers_.end()) {
            throw std::runtime_error(
                "rust storage executor: destination buffer missing");
        }
        auto* dst = static_cast<std::uint8_t*>(dst_it->second.data()) +
            instr.dest.offset;
        loader_.copy_strided_to_device(
            source_tensor_names_[instr.source.tensor_id],
            {},
            dst,
            shape_from_extent(instr.dest.stride));
    }

    const DeviceTensor& buffer_or_finalized_tensor(std::uint32_t buffer_id)
    {
        auto buffer = buffers_.find(buffer_id);
        if (buffer != buffers_.end()) {
            return buffer->second;
        }
        auto finalized = finalized_buffer_names_.find(buffer_id);
        if (finalized != finalized_buffer_names_.end()) {
            return weights_.get(finalized->second);
        }
        throw std::runtime_error(
            "rust storage executor: source buffer missing for CreateView");
    }

    void create_view(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        const auto inputs = ids_from_slice(instr.input_buffers);
        const auto outputs = ids_from_slice(instr.output_buffers);
        if (inputs.size() != 1 || outputs.size() != 1 || !instr.has_dest) {
            throw std::runtime_error(
                "rust storage executor: CreateView expects one input, one "
                "output, and a destination view extent");
        }
        const auto input_id = inputs.front();
        const auto output_id = outputs.front();
        const DeviceTensor& input = buffer_or_finalized_tensor(input_id);
        const auto& output_buffer = buffer_decl(program, output_id);
        if (!output_buffer.has_tensor) {
            throw std::runtime_error(
                "rust storage executor: CreateView output buffer has no "
                "tensor declaration");
        }
        const auto& tensor = tensor_decl(program, output_buffer.tensor_id);
        const auto shape = shape_from_slice(tensor.shape);
        const auto* input_base =
            static_cast<const std::uint8_t*>(input.data()) + instr.dest.offset;
        buffers_.emplace(
            output_id,
            DeviceTensor::view(
                const_cast<std::uint8_t*>(input_base),
                dtype_from_rust(tensor.dtype),
                shape));

        std::string backing_name;
        if (auto finalized = finalized_buffer_names_.find(input_id);
            finalized != finalized_buffer_names_.end()) {
            backing_name = finalized->second;
        } else {
            const auto& input_buffer = buffer_decl(program, input_id);
            if (input_buffer.has_tensor) {
                backing_name =
                    bytes_to_string(tensor_decl(program, input_buffer.tensor_id).name);
            }
        }
        if (!backing_name.empty()) {
            view_backing_names_[output_id] = std::move(backing_name);
        }
    }

    void finalize(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        LoadExecutionStats& stats)
    {
        auto buffer = buffers_.extract(instr.buffer_id);
        if (buffer.empty()) {
            throw std::runtime_error(
                "rust storage executor: finalize buffer missing");
        }
        const auto& buffer_info = buffer_decl(program, instr.buffer_id);
        const auto& tensor = tensor_decl(program, buffer_info.tensor_id);
        TensorDecl spec;
        spec.name = bytes_to_string(tensor.name);
        spec.dtype = dtype_from_rust(tensor.dtype);
        spec.shape = shape_from_slice(tensor.shape);
        spec.layout = TensorLayoutKind::Dense;
        spec.ownership = TensorOwnershipKind::Owned;
        spec.parallel = TensorParallelKind::Replicated;
        if (auto backing = view_backing_names_.find(instr.buffer_id);
            backing != view_backing_names_.end()) {
            spec.layout = TensorLayoutKind::View;
            spec.ownership = TensorOwnershipKind::BorrowedView;
            spec.backing_tensor = backing->second;
        }
        stats.loaded_bytes += buffer.mapped().nbytes();
        const std::string runtime_name = bytes_to_string(instr.name);
        weights_.insert(
            runtime_name,
            std::move(buffer.mapped()),
            std::move(spec));
        finalized_buffer_names_[instr.buffer_id] = runtime_name;
    }

    SafetensorsCheckpointSource& loader_;
    WeightStoreBuilder& weights_;
    std::vector<std::string> source_tensor_names_;
    std::unordered_map<std::uint32_t, DeviceTensor> buffers_;
    std::unordered_map<std::uint32_t, std::string> finalized_buffer_names_;
    std::unordered_map<std::uint32_t, std::string> view_backing_names_;
};

}  // namespace pie_cuda_driver
