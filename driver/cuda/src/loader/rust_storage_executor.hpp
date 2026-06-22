#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../weight-loader/include/weight_loader.h"
#include "../../../weight-loader/include/weight_loader_cpp.hpp"
#include "loader_config.hpp"
#include "loader_helpers.hpp"
#include "loader/dtype_map.hpp"
#include "loader/phase_timer.hpp"
#include "loader/weight_copy_engine.hpp"
#include "loader/buffer_resolver.hpp"
#include "loader/strided_copy.hpp"
#include "loader/transcode_engine.hpp"
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA 1
#endif
#endif
#ifndef PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
#define PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA 0
#endif
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/slab_scatter.hpp"  // slab_scatter() — the transcode kernels moved to transcode_engine.hpp
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

    ~RustStorageProgramExecutor()
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        free_slab_buffers_noexcept();
#endif
    }

    LoadExecutionStats execute(
        const pie_weight_loader::PieLoaderStorageProgramView& program)
    {
        LoadExecutionStats stats;
        stats.planned_tensor_count = program.tensors.len;
        stats.planned_storage_peak_bytes = program.memory.persistent_bytes +
            program.memory.temporary_peak_bytes;
        stats.planned_storage_temp_bytes = program.memory.temporary_peak_bytes;
        program_index_.reset(program);
        init_persistent_arena(program);
        copy_engine_.set_stats(&stats);
        const bool trace_executor =
            loader_config::env_present("PIE_CUDA_TRACE_STORAGE_EXECUTOR");
        if (trace_executor) {
            std::cerr << "[pie-driver-cuda] storage executor begin schedule="
                      << program.schedule.len << " instrs=" << program.instrs.len
                      << " buffers=" << program.buffers.len << "\n";
        }

        for (std::size_t i = 0; i < program.schedule.len; ++i) {
            const std::uint32_t instr_id = program.schedule.ptr[i];
            const auto& instr = program_index_.instruction(instr_id);
            if (trace_executor && (i < 128 || i % 1000 == 0 ||
                    instr.kind != pie_weight_loader::PieLoaderStorageInstrKind::Allocate)) {
                std::cerr << "[pie-driver-cuda] storage executor instr[" << i
                          << "] id=" << instr.id
                          << " kind=" << static_cast<int>(instr.kind)
                          << " buffer=" << instr.buffer_id << "\n";
            }
            switch (instr.kind) {
            case pie_weight_loader::PieLoaderStorageInstrKind::Allocate:
                if (allocate_requires_copy_flush(instr)) {
                    copy_engine_.flush();
                }
                {
                    PhaseTimer _pt(&stats.phase_alloc_ms);
                    allocate(program, instr);
                }
                if (trace_executor && (i < 128 || i % 1000 == 0)) {
                    std::cerr << "[pie-driver-cuda] storage executor allocated buffer="
                              << instr.buffer_id << "\n";
                }
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::ExtentWrite:
                extent_write(instr, stats);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::BulkExtentWrite:
                bulk_extent_write(instr, stats);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::SlabScatter:
                copy_engine_.flush();
                {
                    PhaseTimer _pt(&stats.phase_transform_ms);
                    slab_scatter(instr, stats);
                }
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Finalize:
                {
                    PhaseTimer _pt(&stats.phase_transform_ms);
                    finalize(program, instr, stats);
                }
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::TileMap:
                copy_engine_.flush();
                {
                    PhaseTimer _pt(&stats.phase_transform_ms);
                    transcode_.tile_map(instr, stats);
                }
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::CreateView:
                copy_engine_.flush();
                create_view(program, instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Release:
                copy_engine_.flush();
                buffers_.erase(instr.buffer_id);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
                copy_engine_.flush();
                break;
            }
        }
        copy_engine_.flush();
        attach_quant_metadata();
        weights_.finalize();
        copy_engine_.set_stats(nullptr);
        return stats;
    }

private:
    void allocate(
        const pie_weight_loader::PieLoaderStorageProgramView& program,
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        const auto& buffer = program_index_.buffer(instr.buffer_id);
        if (try_allocate_persistent_arena_view(buffer)) {
            return;
        }
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

    bool allocate_requires_copy_flush(
        const pie_weight_loader::PieLoaderStorageInstrView& instr) const
    {
        const auto& buffer = program_index_.buffer(instr.buffer_id);
        return !can_allocate_persistent_arena_view(buffer);
    }

    void init_persistent_arena(
        const pie_weight_loader::PieLoaderStorageProgramView& program)
    {
        if (loader_config::env_truthy("PIE_CUDA_DISABLE_WEIGHT_ARENA")) {
            return;
        }
        if (program.memory.persistent_bytes == 0) {
            return;
        }
        persistent_arena_name_ = "__pie.storage_arena.0";
        DeviceTensor arena = DeviceTensor::allocate(
            DType::UINT8,
            {static_cast<std::int64_t>(program.memory.persistent_bytes)});
        persistent_arena_base_ = static_cast<std::uint8_t*>(arena.data());
        persistent_arena_bytes_ = arena.nbytes();
        TensorDecl spec;
        spec.name = persistent_arena_name_;
        spec.dtype = DType::UINT8;
        spec.shape = {static_cast<std::int64_t>(persistent_arena_bytes_)};
        spec.layout = TensorLayoutKind::Dense;
        spec.ownership = TensorOwnershipKind::Owned;
        spec.parallel = TensorParallelKind::Replicated;
        weights_.insert(persistent_arena_name_, std::move(arena), std::move(spec));
    }

    bool try_allocate_persistent_arena_view(
        const pie_weight_loader::PieLoaderBufferDeclView& buffer)
    {
        if (!can_allocate_persistent_arena_view(buffer)) {
            return false;
        }
        if (!buffer.has_persistent_offset) {
            throw std::runtime_error(
                "rust storage executor: persistent buffer missing arena offset");
        }
        const std::uint64_t aligned = buffer.persistent_offset;
        const std::uint64_t end = aligned + buffer.bytes;
        if (end < aligned || end > persistent_arena_bytes_) {
            throw std::runtime_error(
                "rust storage executor: persistent arena exhausted");
        }

        const auto& tensor = program_index_.tensor(buffer.tensor_id);
        DType dtype = DType::UINT8;
        std::vector<std::int64_t> shape;
        if (tensor.encoding_kind ==
            pie_weight_loader::PieLoaderEncodingKind::Quant) {
            const DType physical = quant_physical_dtype(tensor);
            if (physical == DType::FP8_E4M3 || physical == DType::INT8) {
                dtype = physical;
                shape = wl_cpp::i64_slice_to_vector(tensor.shape);
            } else {
                dtype = DType::UINT8;
                shape = {static_cast<std::int64_t>(buffer.bytes)};
            }
        } else {
            dtype = dtype_from_rust(tensor.dtype);
            shape = wl_cpp::i64_slice_to_vector(tensor.shape);
        }
        buffers_.emplace(
            buffer.id,
            DeviceTensor::view(persistent_arena_base_ + aligned, dtype, shape));
        arena_backing_names_[buffer.id] = persistent_arena_name_;
        return true;
    }

    bool can_allocate_persistent_arena_view(
        const pie_weight_loader::PieLoaderBufferDeclView& buffer) const
    {
        return persistent_arena_base_ != nullptr && buffer.has_tensor &&
            !buffer.temporary && buffer.bytes != 0;
    }

    void extent_write(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        LoadExecutionStats& stats)
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
                loader_, source_tensor_names_, instr,
                dst,
                wl_cpp::extent_shape(instr.dest.stride));
            return;
        }
        copy_engine_.queue(
            instr.source.file_id,
            instr.source.file_offset + instr.source.stride.base_offset,
            instr.source.span_bytes,
            dst);
        ++stats.h2d_copy_count;
        stats.h2d_copy_bytes += instr.source.span_bytes;
    }

    void bulk_extent_write(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        LoadExecutionStats& stats)
    {
        if (persistent_arena_base_ == nullptr) {
            throw std::runtime_error(
                "rust storage executor: BulkExtentWrite requires persistent arena");
        }
        if (!instr.has_source || !instr.has_dest) {
            throw std::runtime_error(
                "rust storage executor: BulkExtentWrite missing source/dest");
        }
        const std::uint64_t dst_offset =
            instr.dest.offset + instr.dest.stride.base_offset;
        if (dst_offset > persistent_arena_bytes_ ||
            instr.source.span_bytes > persistent_arena_bytes_ - dst_offset) {
            throw std::runtime_error(
                "rust storage executor: BulkExtentWrite destination out of bounds");
        }
        copy_engine_.queue(
            instr.source.file_id,
            instr.source.file_offset + instr.source.stride.base_offset,
            instr.source.span_bytes,
            persistent_arena_base_ + dst_offset);
        ++stats.h2d_copy_count;
        ++stats.h2d_bulk_copy_count;
        stats.h2d_copy_bytes += instr.source.span_bytes;
        stats.h2d_bulk_copy_bytes += instr.source.span_bytes;
    }

    void slab_scatter(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        LoadExecutionStats& stats)
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        if (persistent_arena_base_ == nullptr) {
            throw std::runtime_error(
                "rust storage executor: SlabScatter requires persistent arena");
        }
        if (instr.slab_placements.len == 0 || instr.slab_span_bytes == 0) {
            return;
        }
        ensure_slab_staging_capacity(instr.slab_span_bytes);
        ensure_slab_placement_capacity(instr.slab_placements.len);

        slab_placement_host_.resize(instr.slab_placements.len);
        std::uint64_t payload_bytes = 0;
        for (std::size_t i = 0; i < instr.slab_placements.len; ++i) {
            const auto& placement = instr.slab_placements.ptr[i];
            if (placement.src_offset > instr.slab_span_bytes ||
                placement.bytes > instr.slab_span_bytes - placement.src_offset) {
                throw std::runtime_error(
                    "rust storage executor: SlabScatter source placement out of bounds");
            }
            if (placement.dest_offset > persistent_arena_bytes_ ||
                placement.bytes > persistent_arena_bytes_ - placement.dest_offset) {
                throw std::runtime_error(
                    "rust storage executor: SlabScatter destination out of bounds");
            }
            slab_placement_host_[i] = SlabScatterPlacement{
                placement.src_offset,
                placement.dest_offset,
                placement.bytes,
            };
            payload_bytes += placement.bytes;
        }

        cudaStream_t stream = copy_engine_.acquire_stream();
        loader_.copy_storage_bytes_to_device_async(
            instr.slab_file_id,
            instr.slab_file_offset,
            instr.slab_span_bytes,
            slab_staging_,
            stream);
        CUDA_CHECK(cudaMemcpyAsync(
            slab_placements_device_,
            slab_placement_host_.data(),
            instr.slab_placements.len * sizeof(SlabScatterPlacement),
            cudaMemcpyHostToDevice,
            stream));
        launch_slab_scatter(
            static_cast<const std::uint8_t*>(slab_staging_),
            persistent_arena_base_,
            slab_placements_device_,
            instr.slab_placements.len,
            stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        ++stats.h2d_copy_count;
        stats.h2d_copy_bytes += instr.slab_span_bytes;
        ++stats.slab_scatter_count;
        stats.slab_scatter_placements += instr.slab_placements.len;
        stats.slab_scatter_source_bytes += instr.slab_span_bytes;
        stats.slab_scatter_payload_bytes += payload_bytes;
#else
        (void)instr;
        (void)stats;
        throw std::runtime_error(
            "rust storage executor: SlabScatter requires CUDA support");
#endif
    }

#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    void ensure_slab_staging_capacity(std::uint64_t bytes)
    {
        if (slab_staging_capacity_ >= bytes) {
            return;
        }
        if (slab_staging_ != nullptr) {
            CUDA_CHECK(cudaFree(slab_staging_));
            slab_staging_ = nullptr;
            slab_staging_capacity_ = 0;
        }
        CUDA_CHECK(cudaMalloc(&slab_staging_, static_cast<std::size_t>(bytes)));
        slab_staging_capacity_ = bytes;
    }

    void ensure_slab_placement_capacity(std::size_t count)
    {
        if (slab_placement_capacity_ >= count) {
            return;
        }
        if (slab_placements_device_ != nullptr) {
            CUDA_CHECK(cudaFree(slab_placements_device_));
            slab_placements_device_ = nullptr;
            slab_placement_capacity_ = 0;
        }
        CUDA_CHECK(cudaMalloc(
            &slab_placements_device_,
            count * sizeof(SlabScatterPlacement)));
        slab_placement_capacity_ = count;
    }

    void free_slab_buffers_noexcept() noexcept
    {
        if (slab_staging_ != nullptr) {
            (void)cudaFree(slab_staging_);
            slab_staging_ = nullptr;
        }
        if (slab_placements_device_ != nullptr) {
            (void)cudaFree(slab_placements_device_);
            slab_placements_device_ = nullptr;
        }
        slab_staging_capacity_ = 0;
        slab_placement_capacity_ = 0;
    }
#endif

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
        const DeviceTensor& input = resolver_.or_finalized(input_id);
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
        } else if (auto backing = arena_backing_names_.find(instr.buffer_id);
                   backing != arena_backing_names_.end()) {
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

    // Normalize a block/group FP8 weight scale to FP32 (the gemm_act_x_w FP8
    // path requires FP32). Handles both shipped formats: E8M0 bytes (DeepSeek-V4,
    // expanded to 2^(b-127)) and BF16 block scales (Qwen3-FP8 / dense block-FP8,
    // where BF16 is the high 16 bits of the F32). FP32 passes through unchanged.
    void convert_block_scale_to_f32(const std::string& name)
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        const auto& t = weights_.get(name);
        if (t.dtype() == DType::FP32) return;
        const auto& shape = t.shape();
        const std::size_t n = t.numel();
        std::vector<float> f32;
        if (t.dtype() == DType::UINT8) {
            std::vector<std::uint8_t> e8m0(n);
            CUDA_CHECK(cudaMemcpy(e8m0.data(), t.data(), n, cudaMemcpyDeviceToHost));
            f32 = expand_e8m0_to_f32(e8m0.data(), n);
        } else if (t.dtype() == DType::BF16) {
            std::vector<std::uint16_t> bf16(n);
            CUDA_CHECK(cudaMemcpy(bf16.data(), t.data(),
                                  n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            f32.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(bf16[i]) << 16;
                std::memcpy(&f32[i], &bits, sizeof(float));
            }
        } else {
            return;  // unsupported scale dtype — the GEMM reports it clearly
        }

        // A 2D block scale [row_blocks, col_blocks] keeps its shape so per-group
        // dequant can index it; a 1D scale collapses to [n].
        DeviceTensor converted = shape.size() == 2
            ? DeviceTensor::allocate(DType::FP32,
                  {static_cast<int>(shape[0]), static_cast<int>(shape[1])})
            : DeviceTensor::allocate(DType::FP32, {static_cast<int>(n)});
        CUDA_CHECK(cudaMemcpy(converted.data(), f32.data(),
                              n * sizeof(float), cudaMemcpyHostToDevice));
        weights_.insert(name + ".f32", std::move(converted));
#else
        (void)name;  // no-CUDA build: header must still compile
#endif
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
            // MXFP4 (group_size 32) keeps its raw E8M0 byte scale: the
            // MXFP4 GEMM / dequant kernels and make_expert_weight_view all
            // require U8 E8M0 bytes (gemm.cpp asserts scale_dtype==UINT8).
            // Only the block-scaled FP8 path (e.g. DeepSeek-V4, group 128)
            // wants the E8M0 bytes expanded to F32 2^(b-127) factors.
            const bool is_mxfp4_scale = (attachment.group_size == 32);
            std::string scale_name = attachment.scale_tensor_name;
            if (!is_mxfp4_scale) {
                convert_block_scale_to_f32(attachment.scale_tensor_name);
                const std::string f32_name =
                    attachment.scale_tensor_name + ".f32";
                if (weights_.find(f32_name) != weights_.end()) {
                    scale_name = f32_name;
                }
            }
            QuantMeta meta;
            meta.kind = quant_meta_kind(attachment.granularity);
            meta.scale_name = scale_name;
            meta.scale = &weights_.get(scale_name);
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
    std::unordered_map<std::uint32_t, std::string> arena_backing_names_;
    std::string persistent_arena_name_;
    std::uint8_t* persistent_arena_base_ = nullptr;
    std::uint64_t persistent_arena_bytes_ = 0;
    // Host->device copy path (streams, pinned staging, reader lanes, batching).
    WeightCopyEngine copy_engine_{loader_};
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    void* slab_staging_ = nullptr;
    std::uint64_t slab_staging_capacity_ = 0;
    SlabScatterPlacement* slab_placements_device_ = nullptr;
    std::size_t slab_placement_capacity_ = 0;
    std::vector<SlabScatterPlacement> slab_placement_host_;

#endif
    wl_cpp::StorageProgramIndex program_index_{"rust storage executor"};
    BufferResolver resolver_{buffers_, finalized_buffer_names_, weights_};
    TranscodeEngine transcode_{loader_, source_tensor_names_, copy_engine_, program_index_, resolver_};
};

}  // namespace pie_cuda_driver

#undef PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
