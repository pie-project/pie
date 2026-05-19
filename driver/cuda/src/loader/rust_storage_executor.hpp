#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../weight_loader/include/weight_loader.h"
#include "../../../weight_loader/include/weight_loader_cpp.hpp"
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA 1
#endif
#endif
#ifndef PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
#define PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA 0
#endif
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
#include "cuda_check.hpp"
#include "kernels/dtype_cast.hpp"
#endif
#include "loader/rust_quant_attachment.hpp"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

namespace wl_cpp = pie_weight_loader::cpp;

class RustStorageProgramExecutor {
public:
    RustStorageProgramExecutor(
        SafetensorsCheckpointSource& loader,
        WeightStoreBuilder& weights,
        std::vector<std::string> source_tensor_names,
        std::vector<RustQuantAttachment> quant_attachments)
        : loader_(loader),
          weights_(weights),
          source_tensor_names_(std::move(source_tensor_names)),
          quant_attachments_(std::move(quant_attachments))
    {}

    LoadExecutionStats execute(
        const pie_weight_loader::PieLoaderStorageProgramView& program)
    {
        LoadExecutionStats stats;
        stats.planned_tensor_count = program.tensors.len;
        stats.planned_storage_peak_bytes = program.memory.persistent_bytes +
            program.memory.temporary_peak_bytes;
        stats.planned_storage_temp_bytes = program.memory.temporary_peak_bytes;
        program_index_.reset(program);

        for (std::size_t i = 0; i < program.schedule.len; ++i) {
            const std::uint32_t instr_id = program.schedule.ptr[i];
            const auto& instr = program_index_.instruction(instr_id);
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
                tile_map(instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::CreateView:
                create_view(program, instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Release:
                buffers_.erase(instr.buffer_id);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
                break;
            }
        }
        attach_quant_metadata();
        weights_.finalize();
        return stats;
    }

private:
    static QuantMeta::Kind quant_meta_kind(QuantGranularity granularity)
    {
        switch (granularity) {
        case QuantGranularity::PerTensor: return QuantMeta::Kind::PerTensor;
        case QuantGranularity::PerChannel: return QuantMeta::Kind::PerChannel;
        case QuantGranularity::PerGroup: return QuantMeta::Kind::PerGroup;
        case QuantGranularity::None: break;
        }
        return QuantMeta::Kind::PerTensor;
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

    void allocate(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        const auto& buffer = program_index_.buffer(instr.buffer_id);
        if (!buffer.has_tensor) {
            buffers_.emplace(
                buffer.id,
                DeviceTensor::allocate(
                    DType::UINT8,
                    {static_cast<std::int64_t>(buffer.bytes)}));
            return;
        }
        const auto& tensor = program_index_.tensor(buffer.tensor_id);
        if (tensor.encoding_kind ==
            pie_weight_loader::PieLoaderEncodingKind::Quant) {
            buffers_.emplace(
                buffer.id,
                DeviceTensor::allocate(
                    DType::UINT8,
                    {static_cast<std::int64_t>(buffer.bytes)}));
            return;
        }
        buffers_.emplace(
            buffer.id,
            DeviceTensor::allocate(
                dtype_from_rust(tensor.dtype),
                wl_cpp::i64_slice_to_vector(tensor.shape)));
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
            instr.dest.offset + instr.dest.stride.base_offset;
        if (!wl_cpp::compact_extent(instr.dest.stride)) {
            throw std::runtime_error(
                "rust storage executor: non-compact ExtentWrite destination is not "
                "implemented");
        }
        if (!wl_cpp::compact_extent(instr.source.stride)) {
            copy_strided_extent_to_device(instr, dst);
            return;
        }
        loader_.copy_storage_bytes_to_device(
            instr.source.file_id,
            instr.source.file_offset + instr.source.stride.base_offset,
            instr.source.span_bytes,
            dst);
    }

    void copy_strided_extent_to_device(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        void* dst)
    {
        const std::string& name = source_tensor_names_[instr.source.tensor_id];
        const TensorInfo& info = loader_.info(name);
        const TensorStorageInfo storage = loader_.storage_info(name);
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
        loader_.copy_strided_to_device(
            name,
            slices,
            dst,
            wl_cpp::extent_shape(instr.dest.stride));
    }

    DeviceTensor& buffer_tensor(std::uint32_t buffer_id)
    {
        auto buffer = buffers_.find(buffer_id);
        if (buffer == buffers_.end()) {
            throw std::runtime_error("rust storage executor: buffer missing");
        }
        return buffer->second;
    }

    static void cast_tensor_to_ptr(
        const DeviceTensor& src,
        void* dst,
        DType dst_dtype)
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        if (src.dtype() == dst_dtype) {
            CUDA_CHECK(cudaMemcpyAsync(
                dst,
                src.data(),
                src.nbytes(),
                cudaMemcpyDeviceToDevice,
                /*stream=*/0));
        } else if (src.dtype() == DType::FP16 && dst_dtype == DType::BF16) {
            kernels::launch_cast_fp16_to_bf16(
                src.data(), dst, src.numel(), /*stream=*/0);
        } else if (src.dtype() == DType::FP32 && dst_dtype == DType::BF16) {
            kernels::launch_cast_fp32_to_bf16(
                src.data(), dst, src.numel(), /*stream=*/0);
        } else if (src.dtype() == DType::BF16 && dst_dtype == DType::FP32) {
            kernels::launch_cast_bf16_to_fp32(
                src.data(), dst, src.numel(), /*stream=*/0);
        } else {
            throw std::runtime_error(
                "rust storage executor: unsupported TileMap Cast " +
                std::string(dtype_name(src.dtype())) + " -> " +
                std::string(dtype_name(dst_dtype)));
        }
#else
        (void)src;
        (void)dst;
        (void)dst_dtype;
        throw std::runtime_error(
            "rust storage executor: CUDA TileMap Cast compiled without CUDA "
            "headers");
#endif
    }

    void tile_map(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        switch (instr.tile_kind) {
        case pie_weight_loader::PieLoaderTileMapKind::Cast:
            cast_tile_map(instr);
            return;
        case pie_weight_loader::PieLoaderTileMapKind::Reblock:
        case pie_weight_loader::PieLoaderTileMapKind::Reorder:
            reblock_tile_map(instr);
            return;
        case pie_weight_loader::PieLoaderTileMapKind::Decode:
        case pie_weight_loader::PieLoaderTileMapKind::Encode:
        case pie_weight_loader::PieLoaderTileMapKind::Transcode:
        case pie_weight_loader::PieLoaderTileMapKind::None:
            throw std::runtime_error(
                "rust storage executor: unsupported TileMap kind in CUDA "
                "cutover path");
        }
        throw std::runtime_error("rust storage executor: unknown TileMap kind");
    }

    void cast_tile_map(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Cast TileMap expects one output");
        }
        const auto output_id = instr.output_buffers.ptr[0];
        DeviceTensor& out = buffer_tensor(output_id);
        const auto dst_offset =
            instr.has_dest ? instr.dest.offset + instr.dest.stride.base_offset : 0;
        auto* dst = static_cast<std::uint8_t*>(out.data()) + dst_offset;

        if (instr.has_source) {
            if (instr.source.tensor_id >= source_tensor_names_.size()) {
                throw std::runtime_error(
                    "rust storage executor: Cast source tensor id out of range");
            }
            if (!wl_cpp::compact_extent(instr.source.stride)) {
                throw std::runtime_error(
                    "rust storage executor: non-compact Cast source is not "
                    "implemented");
            }
            const TensorInfo& info =
                loader_.info(source_tensor_names_[instr.source.tensor_id]);
            DeviceTensor scratch =
                DeviceTensor::allocate(
                    info.dtype,
                    wl_cpp::extent_shape(instr.source.stride));
            if (scratch.nbytes() != instr.source.span_bytes) {
                throw std::runtime_error(
                    "rust storage executor: Cast source byte size mismatch");
            }
            loader_.copy_storage_bytes_to_device(
                instr.source.file_id,
                instr.source.file_offset,
                instr.source.span_bytes,
                scratch.data());
            cast_tensor_to_ptr(scratch, dst, out.dtype());
            return;
        }

        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Cast TileMap expects source or one input");
        }
        cast_tensor_to_ptr(buffer_or_finalized_tensor(instr.input_buffers.ptr[0]), dst, out.dtype());
    }

    void reblock_tile_map(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.input_buffers.len != 1 || instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Reblock TileMap expects one input "
                "and one output");
        }
        const DeviceTensor& input =
            buffer_or_finalized_tensor(instr.input_buffers.ptr[0]);
        DeviceTensor& output = buffer_tensor(instr.output_buffers.ptr[0]);
        const auto dst_offset =
            instr.has_dest ? instr.dest.offset + instr.dest.stride.base_offset : 0;
        const auto bytes = instr.has_dest
            ? wl_cpp::extent_bytes(
                  instr.dest.stride,
                  "rust storage executor")
            : static_cast<std::uint64_t>(input.nbytes());
        if (bytes > input.nbytes() ||
            dst_offset + bytes > output.nbytes()) {
            throw std::runtime_error(
                "rust storage executor: Reblock byte range out of bounds");
        }
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::uint8_t*>(output.data()) + dst_offset,
            input.data(),
            bytes,
            cudaMemcpyDeviceToDevice,
            /*stream=*/0));
#else
        throw std::runtime_error(
            "rust storage executor: CUDA Reblock compiled without CUDA "
            "headers");
#endif
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
        const auto inputs =
            wl_cpp::buffer_id_slice_to_vector(instr.input_buffers);
        const auto outputs =
            wl_cpp::buffer_id_slice_to_vector(instr.output_buffers);
        if (inputs.size() != 1 || outputs.size() != 1 || !instr.has_dest) {
            throw std::runtime_error(
                "rust storage executor: CreateView expects one input, one "
                "output, and a destination view extent");
        }
        const auto input_id = inputs.front();
        const auto output_id = outputs.front();
        const DeviceTensor& input = buffer_or_finalized_tensor(input_id);
        const auto& output_buffer = program_index_.buffer(output_id);
        if (!output_buffer.has_tensor) {
            throw std::runtime_error(
                "rust storage executor: CreateView output buffer has no "
                "tensor declaration");
        }
        const auto& tensor = program_index_.tensor(output_buffer.tensor_id);
        const auto shape = wl_cpp::i64_slice_to_vector(tensor.shape);
        const auto* input_base =
            static_cast<const std::uint8_t*>(input.data()) + instr.dest.offset +
            instr.dest.stride.base_offset;
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
            const auto& input_buffer = program_index_.buffer(input_id);
            if (input_buffer.has_tensor) {
                backing_name =
                    wl_cpp::bytes_to_string(
                        program_index_.tensor(input_buffer.tensor_id).name);
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
        const auto& buffer_info = program_index_.buffer(instr.buffer_id);
        const auto& tensor = program_index_.tensor(buffer_info.tensor_id);
        TensorDecl spec;
        spec.name = wl_cpp::bytes_to_string(tensor.name);
        if (tensor.encoding_kind ==
            pie_weight_loader::PieLoaderEncodingKind::Quant) {
            spec.dtype = DType::UINT8;
            spec.shape = {static_cast<std::int64_t>(buffer.mapped().nbytes())};
        } else {
            spec.dtype = dtype_from_rust(tensor.dtype);
            spec.shape = wl_cpp::i64_slice_to_vector(tensor.shape);
        }
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
        const std::string runtime_name = wl_cpp::bytes_to_string(instr.name);
        weights_.insert(
            runtime_name,
            std::move(buffer.mapped()),
            std::move(spec));
        finalized_buffer_names_[instr.buffer_id] = runtime_name;
    }

    void attach_quant_metadata()
    {
        for (const auto& attachment : quant_attachments_) {
            if (weights_.find(attachment.tensor_name) == weights_.end()) {
                throw std::runtime_error(
                    "rust storage executor: quant tensor '" +
                    attachment.tensor_name + "' was not finalized");
            }
            if (weights_.find(attachment.scale_tensor_name) ==
                weights_.end()) {
                throw std::runtime_error(
                    "rust storage executor: quant scale tensor '" +
                    attachment.scale_tensor_name + "' for '" +
                    attachment.tensor_name + "' was not finalized");
            }
            QuantMeta meta;
            meta.kind = quant_meta_kind(attachment.granularity);
            meta.scale = &weights_.get(attachment.scale_tensor_name);
            meta.group_size = attachment.group_size;
            meta.channel_axis = attachment.channel_axis;
            weights_.set_quant_meta(attachment.tensor_name, std::move(meta));
        }
    }

    SafetensorsCheckpointSource& loader_;
    WeightStoreBuilder& weights_;
    std::vector<std::string> source_tensor_names_;
    std::vector<RustQuantAttachment> quant_attachments_;
    std::unordered_map<std::uint32_t, DeviceTensor> buffers_;
    std::unordered_map<std::uint32_t, std::string> finalized_buffer_names_;
    std::unordered_map<std::uint32_t, std::string> view_backing_names_;
    wl_cpp::StorageProgramIndex program_index_{"rust storage executor"};
};

}  // namespace pie_cuda_driver

#undef PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
