#pragma once

#include <algorithm>
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
#include "kernels/mxfp4_marlin.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif
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
                tile_map(instr, stats);
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

    static DType quant_physical_dtype(
        const pie_weight_loader::PieLoaderTensorDeclView& tensor)
    {
        switch (tensor.quant_scheme) {
        case pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3:
            return DType::FP8_E4M3;
        case pie_weight_loader::PieLoaderQuantScheme::Int8Symmetric:
            return DType::INT8;
        default:
            return DType::UINT8;
        }
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
            const DType physical = quant_physical_dtype(tensor);
            if (physical == DType::FP8_E4M3 || physical == DType::INT8) {
                buffers_.emplace(
                    buffer.id,
                    DeviceTensor::allocate(
                        physical,
                        wl_cpp::i64_slice_to_vector(tensor.shape)));
                return;
            }
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
            copy_strided_extent_to_device(
                instr,
                dst,
                wl_cpp::extent_shape(instr.dest.stride));
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
        void* dst,
        const std::vector<std::int64_t>& dst_shape)
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
            dst_shape);
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
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        LoadExecutionStats& stats)
    {
        switch (instr.tile_kind) {
        case pie_weight_loader::PieLoaderTileMapKind::Cast:
            cast_tile_map(instr);
            return;
        case pie_weight_loader::PieLoaderTileMapKind::Reblock:
        case pie_weight_loader::PieLoaderTileMapKind::Reorder:
            reblock_tile_map(instr);
            return;
        case pie_weight_loader::PieLoaderTileMapKind::Encode:
            encode_tile_map(instr, stats);
            return;
        case pie_weight_loader::PieLoaderTileMapKind::Repack:
            repack_tile_map(instr);
            return;
        case pie_weight_loader::PieLoaderTileMapKind::Decode:
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

    DeviceTensor materialize_encode_input_bf16_rows(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows)
    {
        const int cols = static_cast<int>(full_shape[1]);
        const std::vector<std::int64_t> tile_shape{
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
        };
        DeviceTensor source;
        if (instr.has_source) {
            if (instr.source.tensor_id >= source_tensor_names_.size()) {
                throw std::runtime_error(
                    "rust storage executor: Encode source tensor id out of range");
            }
            const TensorInfo& info =
                loader_.info(source_tensor_names_[instr.source.tensor_id]);
            source = DeviceTensor::allocate(info.dtype, tile_shape);
            if (!wl_cpp::compact_extent(instr.source.stride)) {
                if (row_start != 0 || rows != full_shape[0]) {
                    throw std::runtime_error(
                        "rust storage executor: tiled Encode for non-compact "
                        "sources is not implemented");
                }
                copy_strided_extent_to_device(instr, source.data(), full_shape);
            } else {
                const std::uint64_t elem = dtype_bytes(info.dtype);
                const std::uint64_t row_bytes =
                    static_cast<std::uint64_t>(cols) * elem;
                loader_.copy_storage_bytes_to_device(
                    instr.source.file_id,
                    instr.source.file_offset + instr.source.stride.base_offset +
                        static_cast<std::uint64_t>(row_start) * row_bytes,
                    static_cast<std::uint64_t>(rows) * row_bytes,
                    source.data());
            }
        } else {
            if (instr.input_buffers.len != 1) {
                throw std::runtime_error(
                    "rust storage executor: Encode expects source or one input");
            }
            const DeviceTensor& input =
                buffer_or_finalized_tensor(instr.input_buffers.ptr[0]);
            source = DeviceTensor::allocate(input.dtype(), tile_shape);
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
            const std::uint64_t row_bytes =
                static_cast<std::uint64_t>(cols) * dtype_bytes(input.dtype());
            CUDA_CHECK(cudaMemcpyAsync(
                source.data(),
                static_cast<const std::uint8_t*>(input.data()) +
                    static_cast<std::uint64_t>(row_start) * row_bytes,
                static_cast<std::uint64_t>(rows) * row_bytes,
                cudaMemcpyDeviceToDevice,
                /*stream=*/0));
#else
            throw std::runtime_error(
                "rust storage executor: CUDA Encode compiled without CUDA "
                "headers");
#endif
        }
        if (source.dtype() == DType::BF16) {
            return source;
        }
        DeviceTensor bf16 = DeviceTensor::allocate(DType::BF16, tile_shape);
        cast_tensor_to_ptr(source, bf16.data(), DType::BF16);
        return bf16;
    }

    DType encode_source_dtype(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.has_source) {
            if (instr.source.tensor_id >= source_tensor_names_.size()) {
                throw std::runtime_error(
                    "rust storage executor: Encode source tensor id out of range");
            }
            return loader_.info(source_tensor_names_[instr.source.tensor_id]).dtype;
        }
        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Encode expects source or one input");
        }
        return buffer_or_finalized_tensor(instr.input_buffers.ptr[0]).dtype();
    }

    bool can_tile_encode(
        const pie_weight_loader::PieLoaderStorageInstrView& instr) const
    {
        return !instr.has_source || wl_cpp::compact_extent(instr.source.stride);
    }

    int encode_rows_per_tile(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        DType source_dtype,
        int rows,
        int cols) const
    {
        const std::uint64_t max_tile_bytes =
            instr.max_tile_bytes == 0 ? (64ull << 20) : instr.max_tile_bytes;
        const std::uint64_t source_row_bytes =
            static_cast<std::uint64_t>(cols) * dtype_bytes(source_dtype);
        const std::uint64_t bf16_row_bytes =
            static_cast<std::uint64_t>(cols) * dtype_bytes(DType::BF16);
        const std::uint64_t scratch_per_row =
            source_dtype == DType::BF16
                ? bf16_row_bytes
                : source_row_bytes + bf16_row_bytes;
        const std::uint64_t rows_per_tile = std::max<std::uint64_t>(
            1,
            max_tile_bytes / std::max<std::uint64_t>(1, scratch_per_row));
        return static_cast<int>(
            std::min<std::uint64_t>(
                static_cast<std::uint64_t>(rows),
                rows_per_tile));
    }

    void launch_encode_tile(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        const DeviceTensor& bf16,
        DeviceTensor& out,
        DeviceTensor& scale,
        int row_start,
        int rows,
        int cols)
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        switch (instr.transform_to) {
        case pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3:
            if (out.dtype() != DType::FP8_E4M3) {
                throw std::runtime_error(
                    "rust storage executor: FP8 Encode output dtype mismatch");
            }
            kernels::quantize_bf16_to_fp8_e4m3_per_channel(
                bf16.data(),
                static_cast<std::uint8_t*>(out.data()) +
                    static_cast<std::uint64_t>(row_start) *
                        static_cast<std::uint64_t>(cols),
                static_cast<float*>(scale.data()) + row_start,
                rows,
                cols,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());
            return;
        case pie_weight_loader::PieLoaderQuantScheme::Int8Symmetric:
            if (out.dtype() != DType::INT8) {
                throw std::runtime_error(
                    "rust storage executor: INT8 Encode output dtype mismatch");
            }
            kernels::quantize_bf16_to_int8_per_channel(
                bf16.data(),
                static_cast<std::int8_t*>(out.data()) +
                    static_cast<std::uint64_t>(row_start) *
                        static_cast<std::uint64_t>(cols),
                static_cast<float*>(scale.data()) + row_start,
                rows,
                cols,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());
            return;
        default:
            throw std::runtime_error(
                "rust storage executor: unsupported Encode quant scheme");
        }
#else
        (void)instr;
        (void)bf16;
        (void)out;
        (void)scale;
        (void)row_start;
        (void)rows;
        (void)cols;
        throw std::runtime_error(
            "rust storage executor: CUDA Encode compiled without CUDA headers");
#endif
    }

    void encode_tile_map(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        LoadExecutionStats& stats)
    {
        if (instr.output_buffers.len != 2) {
            throw std::runtime_error(
                "rust storage executor: Encode expects weight and scale outputs");
        }
        DeviceTensor& out = buffer_tensor(instr.output_buffers.ptr[0]);
        DeviceTensor& scale = buffer_tensor(instr.output_buffers.ptr[1]);
        const auto shape = out.shape();
        if (shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: runtime Encode expects a 2-D weight");
        }
        const int rows = static_cast<int>(shape[0]);
        const int cols = static_cast<int>(shape[1]);
        if (scale.dtype() != DType::FP32 ||
            scale.shape() != std::vector<std::int64_t>{shape[0]}) {
            throw std::runtime_error(
                "rust storage executor: Encode scale output must be FP32 [rows]");
        }
        stats.runtime_quantized_weights += 1;
        stats.runtime_quant_bytes_after += out.nbytes();
        if (instr.has_source) {
            stats.runtime_quant_bytes_before += instr.source.span_bytes;
        } else if (instr.input_buffers.len == 1) {
            stats.runtime_quant_bytes_before +=
                buffer_or_finalized_tensor(instr.input_buffers.ptr[0]).nbytes();
        }

        if (can_tile_encode(instr)) {
            const DType source_dtype = encode_source_dtype(instr);
            const int rows_per_tile =
                encode_rows_per_tile(instr, source_dtype, rows, cols);
            for (int row = 0; row < rows; row += rows_per_tile) {
                const int tile_rows = std::min(rows_per_tile, rows - row);
                DeviceTensor bf16_tile =
                    materialize_encode_input_bf16_rows(
                        instr, shape, row, tile_rows);
                launch_encode_tile(
                    instr, bf16_tile, out, scale, row, tile_rows, cols);
            }
            return;
        }

        DeviceTensor bf16 =
            materialize_encode_input_bf16_rows(instr, shape, 0, rows);
        launch_encode_tile(instr, bf16, out, scale, 0, rows, cols);
    }

#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    static kernels::Mxfp4RowSelect repack_row_map(
        pie_weight_loader::PieLoaderRowMap row_map)
    {
        switch (row_map) {
        case pie_weight_loader::PieLoaderRowMap::Identity:
            return kernels::Mxfp4RowSelect::Identity;
        case pie_weight_loader::PieLoaderRowMap::Even:
            return kernels::Mxfp4RowSelect::Even;
        case pie_weight_loader::PieLoaderRowMap::Odd:
            return kernels::Mxfp4RowSelect::Odd;
        }
        throw std::runtime_error(
            "rust storage executor: unknown Repack row map");
    }
#endif

    static std::uint64_t checked_mul_u64(
        std::uint64_t lhs,
        std::uint64_t rhs,
        const char* context)
    {
        if (rhs != 0 && lhs > UINT64_MAX / rhs) {
            throw std::runtime_error(
                std::string("rust storage executor: ") + context +
                " byte size overflow");
        }
        return lhs * rhs;
    }

    static std::uint64_t checked_nibble_bytes(
        std::uint64_t rows,
        std::uint64_t cols,
        const char* context)
    {
        const std::uint64_t elements = checked_mul_u64(rows, cols, context);
        if (elements % 2 != 0) {
            throw std::runtime_error(
                std::string("rust storage executor: ") + context +
                " has odd nibble element count");
        }
        return elements / 2;
    }

    DeviceTensor materialize_repack_source(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.has_source) {
            if (instr.source.tensor_id >= source_tensor_names_.size()) {
                throw std::runtime_error(
                    "rust storage executor: Repack source tensor id out of range");
            }
            DeviceTensor scratch = DeviceTensor::allocate(
                DType::UINT8,
                {static_cast<std::int64_t>(instr.source.span_bytes)});
            if (!wl_cpp::compact_extent(instr.source.stride)) {
                copy_strided_extent_to_device(
                    instr,
                    scratch.data(),
                    wl_cpp::extent_shape(instr.source.stride));
            } else {
                loader_.copy_storage_bytes_to_device(
                    instr.source.file_id,
                    instr.source.file_offset + instr.source.stride.base_offset,
                    instr.source.span_bytes,
                    scratch.data());
            }
            return scratch;
        }
        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Repack expects source or one input buffer");
        }
        const DeviceTensor& input =
            buffer_or_finalized_tensor(instr.input_buffers.ptr[0]);
        DeviceTensor scratch = DeviceTensor::allocate(
            DType::UINT8,
            {static_cast<std::int64_t>(input.nbytes())});
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        CUDA_CHECK(cudaMemcpyAsync(
            scratch.data(),
            input.data(),
            input.nbytes(),
            cudaMemcpyDeviceToDevice,
            /*stream=*/0));
#else
        throw std::runtime_error(
            "rust storage executor: CUDA Repack compiled without CUDA headers");
#endif
        return scratch;
    }

    void repack_tile_map(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        if (instr.output_buffers.len != 1 || !instr.has_dest) {
            throw std::runtime_error(
                "rust storage executor: Repack expects one output and destination extent");
        }
        const int batch = static_cast<int>(instr.transform_batch);
        const int source_rows = static_cast<int>(instr.transform_source_rows);
        const int source_row_offset =
            static_cast<int>(instr.transform_source_row_offset);
        const int target_rows = static_cast<int>(instr.transform_target_rows);
        const int valid_rows = instr.transform_valid_rows == 0
            ? target_rows
            : static_cast<int>(instr.transform_valid_rows);
        const int source_stride_cols = instr.transform_source_stride_cols == 0
            ? static_cast<int>(instr.transform_source_cols)
            : static_cast<int>(instr.transform_source_stride_cols);
        const int source_col_offset =
            static_cast<int>(instr.transform_source_col_offset);
        const int source_cols = static_cast<int>(instr.transform_source_cols);
        const int target_cols = static_cast<int>(instr.transform_target_cols);
        if (batch <= 0 || source_rows <= 0 || target_rows <= 0 ||
            valid_rows <= 0 || valid_rows > target_rows ||
            source_stride_cols <= 0 || source_col_offset < 0 ||
            source_cols <= 0 || target_cols <= 0 ||
            source_col_offset + source_cols > source_stride_cols) {
            throw std::runtime_error(
                "rust storage executor: Repack has invalid transform dimensions");
        }
        DeviceTensor& output = buffer_tensor(instr.output_buffers.ptr[0]);
        auto* dst_base = static_cast<std::uint8_t*>(output.data()) +
            instr.dest.offset + instr.dest.stride.base_offset;
        DeviceTensor source = materialize_repack_source(instr);
        const auto* src_base =
            static_cast<const std::uint8_t*>(source.data());
        const auto row_map = repack_row_map(instr.row_map);

        switch (instr.repack_layout) {
        case pie_weight_loader::PieLoaderRepackLayout::MarlinMxfp4Weight:
            repack_marlin_mxfp4_weight(
                src_base, dst_base, batch, source_rows, source_row_offset,
                target_rows, valid_rows, source_stride_cols,
                source_col_offset, source_cols, target_cols, row_map);
            return;
        case pie_weight_loader::PieLoaderRepackLayout::MarlinMxfp4Scale:
            repack_marlin_mxfp4_scale(
                src_base, dst_base, batch, source_rows, source_row_offset,
                target_rows, valid_rows, source_stride_cols,
                source_col_offset, source_cols, target_cols, row_map);
            return;
        case pie_weight_loader::PieLoaderRepackLayout::DenseRowGather:
            if (source_cols != 1 || target_cols != 1) {
                throw std::runtime_error(
                    "rust storage executor: DenseRowGather Repack expects column count 1");
            }
            kernels::launch_bf16_row_map_to_dense(
                src_base, dst_base, batch, source_rows, source_row_offset,
                target_rows, valid_rows, row_map, /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());
            return;
        case pie_weight_loader::PieLoaderRepackLayout::None:
            break;
        }
        throw std::runtime_error(
            "rust storage executor: Repack has no target layout");
#else
        (void)instr;
        throw std::runtime_error(
            "rust storage executor: CUDA Repack compiled without CUDA headers");
#endif
    }

#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    void repack_marlin_mxfp4_weight(
        const std::uint8_t* src_base,
        std::uint8_t* dst_base,
        int batch,
        int source_rows,
        int source_row_offset,
        int target_rows,
        int valid_rows,
        int source_stride_cols,
        int source_col_offset,
        int source_cols,
        int target_cols,
        kernels::Mxfp4RowSelect row_map)
    {
#if defined(PIE_CUDA_HAS_MARLIN)
        if (source_cols % 8 != 0 || target_cols % 8 != 0 ||
            source_stride_cols % 8 != 0 || source_col_offset % 8 != 0 ||
            target_cols < source_cols ||
            source_col_offset + source_cols > source_stride_cols) {
            throw std::runtime_error(
                "rust storage executor: MarlinMxfp4Weight Repack requires "
                "K/stride/offset divisible by 8 and target K >= source K");
        }
        const std::uint64_t source_bytes =
            checked_nibble_bytes(
                source_rows, source_stride_cols, "MXFP4 source");
        const std::uint64_t target_bytes =
            checked_nibble_bytes(target_rows, target_cols, "MXFP4 target");
        DeviceTensor gptq_stage = DeviceTensor::allocate(
            DType::UINT8,
            {static_cast<std::int64_t>(target_bytes)});
        for (int b = 0; b < batch; ++b) {
            const auto* src =
                src_base + static_cast<std::uint64_t>(b) * source_bytes;
            auto* dst =
                dst_base + static_cast<std::uint64_t>(b) * target_bytes;
            kernels::launch_mxfp4_weight_to_gptq_w4(
                src, gptq_stage.data(),
                source_rows, source_row_offset, target_rows, valid_rows,
                source_stride_cols, source_col_offset, source_cols,
                target_cols, row_map, /*stream=*/0);
            marlin::launch_gptq_repack_w4_no_perm(
                gptq_stage.data(), dst, target_cols, target_rows,
                /*stream=*/0);
        }
        CUDA_CHECK(cudaGetLastError());
#else
        (void)src_base;
        (void)dst_base;
        (void)batch;
        (void)source_rows;
        (void)source_row_offset;
        (void)target_rows;
        (void)valid_rows;
        (void)source_stride_cols;
        (void)source_col_offset;
        (void)source_cols;
        (void)target_cols;
        (void)row_map;
        throw std::runtime_error(
            "rust storage executor: MarlinMxfp4Weight Repack requires Marlin");
#endif
    }

    void repack_marlin_mxfp4_scale(
        const std::uint8_t* src_base,
        std::uint8_t* dst_base,
        int batch,
        int source_rows,
        int source_row_offset,
        int target_rows,
        int valid_rows,
        int source_stride_groups,
        int source_group_offset,
        int source_groups,
        int target_groups,
        kernels::Mxfp4RowSelect row_map)
    {
        if (source_stride_groups <= 0 || source_group_offset < 0 ||
            target_groups < source_groups ||
            source_group_offset + source_groups > source_stride_groups) {
            throw std::runtime_error(
                "rust storage executor: MarlinMxfp4Scale Repack requires "
                "target group count >= source group count and source slice "
                "within stride");
        }
        const std::uint64_t source_bytes =
            checked_mul_u64(
                source_rows, source_stride_groups, "MXFP4 scale source");
        const std::uint64_t target_bytes =
            checked_mul_u64(target_rows, target_groups, "MXFP4 scale target");
        for (int b = 0; b < batch; ++b) {
            kernels::launch_mxfp4_scales_to_marlin_e8m0(
                src_base + static_cast<std::uint64_t>(b) * source_bytes,
                dst_base + static_cast<std::uint64_t>(b) * target_bytes,
                source_rows, source_row_offset, target_rows, valid_rows,
                source_stride_groups, source_group_offset, source_groups,
                target_groups, row_map, /*stream=*/0);
        }
        CUDA_CHECK(cudaGetLastError());
    }
#endif

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
            const DType physical = quant_physical_dtype(tensor);
            if (physical == DType::FP8_E4M3 || physical == DType::INT8) {
                spec.dtype = physical;
                spec.shape = wl_cpp::i64_slice_to_vector(tensor.shape);
            } else {
                spec.dtype = DType::UINT8;
                spec.shape = {static_cast<std::int64_t>(buffer.mapped().nbytes())};
            }
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
            meta.scale_name = attachment.scale_tensor_name;
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
