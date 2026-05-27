#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
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
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/mxfp4_marlin.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "kernels/slab_scatter.hpp"
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

    ~RustStorageProgramExecutor()
    {
        destroy_copy_streams_noexcept();
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
        active_stats_ = &stats;
        const bool trace_executor =
            std::getenv("PIE_CUDA_TRACE_STORAGE_EXECUTOR") != nullptr;
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
                    flush_copy_streams();
                }
                allocate(program, instr);
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
                flush_copy_streams();
                slab_scatter(instr, stats);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Finalize:
                finalize(program, instr, stats);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::TileMap:
                flush_copy_streams();
                tile_map(instr, stats);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::CreateView:
                flush_copy_streams();
                create_view(program, instr);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Release:
                flush_copy_streams();
                buffers_.erase(instr.buffer_id);
                break;
            case pie_weight_loader::PieLoaderStorageInstrKind::Attach:
                flush_copy_streams();
                break;
            }
        }
        flush_copy_streams();
        attach_quant_metadata();
        weights_.finalize();
        active_stats_ = nullptr;
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
        if (try_allocate_persistent_arena_view(buffer, program)) {
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
        const char* disable = std::getenv("PIE_CUDA_DISABLE_WEIGHT_ARENA");
        if (disable != nullptr && disable[0] != '\0' && disable[0] != '0') {
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
        const pie_weight_loader::PieLoaderBufferDeclView& buffer,
        const pie_weight_loader::PieLoaderStorageProgramView& program)
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
        (void)program;
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
                instr,
                dst,
                wl_cpp::extent_shape(instr.dest.stride));
            return;
        }
        copy_storage_bytes_to_device(
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
        copy_storage_bytes_to_device(
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
        ensure_copy_streams();
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

        cudaStream_t stream = copy_streams_[next_copy_stream_];
        next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
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

    void copy_storage_bytes_to_device(
        std::uint32_t shard_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst)
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        if (copy_streams_enabled()) {
            ensure_copy_streams();
            if (pinned_staging_enabled()) {
                if (enqueue_pinned_staged_copy(
                        shard_id, file_offset, span_bytes, dst)) {
                    return;
                }
            }
            cudaStream_t stream = copy_streams_[next_copy_stream_];
            next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
            if (batched_copies_enabled()) {
                enqueue_batched_copy(
                    shard_id, file_offset, span_bytes, dst, stream);
            } else {
                loader_.copy_storage_bytes_to_device_async(
                    shard_id, file_offset, span_bytes, dst, stream);
            }
            ++pending_copy_count_;
            if (active_stats_ != nullptr) {
                active_stats_->max_pending_copies_seen =
                    std::max(
                        active_stats_->max_pending_copies_seen,
                        pending_copy_count_);
            }
            if (pending_copy_count_ >= max_pending_copies_) {
                flush_copy_streams();
            }
            return;
        }
#endif
        loader_.copy_storage_bytes_to_device(
            shard_id, file_offset, span_bytes, dst);
    }

    bool pinned_staging_enabled() const
    {
        const char* enable =
            std::getenv("PIE_CUDA_ENABLE_PINNED_WEIGHT_STAGING");
        return enable != nullptr && enable[0] != '\0' && enable[0] != '0';
    }

    std::uint64_t pinned_staging_min_bytes() const
    {
        constexpr std::uint64_t kDefault = 256ull * 1024ull;
        if (const char* env = std::getenv("PIE_CUDA_PINNED_WEIGHT_MIN_BYTES")) {
            const auto parsed = std::strtoull(env, nullptr, 10);
            if (parsed > 0) {
                return parsed;
            }
        }
        return kDefault;
    }

    std::uint64_t pinned_staging_pool_bytes() const
    {
        constexpr std::uint64_t kDefault = 512ull * 1024ull * 1024ull;
        if (const char* env = std::getenv("PIE_CUDA_PINNED_WEIGHT_POOL_MB")) {
            const auto parsed = std::strtoull(env, nullptr, 10);
            if (parsed > 0) {
                return parsed * 1024ull * 1024ull;
            }
        }
        return kDefault;
    }

    std::size_t pinned_staging_slot_count() const
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        std::size_t count = std::max<std::size_t>(copy_streams_.size(), 1);
#else
        std::size_t count = 1;
#endif
        if (const char* env = std::getenv("PIE_CUDA_PINNED_WEIGHT_SLOTS")) {
            const auto parsed = std::strtoull(env, nullptr, 10);
            if (parsed > 0) {
                count = std::min<std::size_t>(parsed, 128);
            }
        }
        return count;
    }

    bool batched_copies_enabled() const
    {
#if CUDART_VERSION >= 12080
        const char* disable =
            std::getenv("PIE_CUDA_DISABLE_BATCHED_WEIGHT_COPIES");
        return !(disable != nullptr && disable[0] != '\0' && disable[0] != '0');
#else
        return false;
#endif
    }

#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    void enqueue_batched_copy(
        std::uint32_t shard_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst,
        cudaStream_t stream)
    {
        pending_copies_.push_back(PendingCopy{
            dst,
            const_cast<std::uint8_t*>(
                loader_.storage_host_ptr(shard_id, file_offset, span_bytes)),
            static_cast<std::size_t>(span_bytes),
            stream,
        });
    }

    bool enqueue_pinned_staged_copy(
        std::uint32_t shard_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst)
    {
        const std::uint64_t min_bytes = pinned_staging_min_bytes();
        if (span_bytes < min_bytes) {
            return false;
        }
        ensure_pinned_slots();
        const std::uint64_t pool_bytes = pinned_staging_pool_bytes();
        const std::uint64_t max_slot_bytes =
            pool_bytes / std::max<std::uint64_t>(pinned_slots_.size(), 1);
        if (span_bytes > max_slot_bytes) {
            return false;
        }

        PinnedSlot& slot = pinned_slots_[next_pinned_slot_];
        next_pinned_slot_ = (next_pinned_slot_ + 1) % pinned_slots_.size();
        if (slot.busy) {
            flush_copy_streams();
        }
        if (slot.capacity < span_bytes) {
            const std::uint64_t next_capacity = next_power_of_two(span_bytes);
            if (pinned_pool_capacity_bytes_ - slot.capacity + next_capacity >
                pool_bytes) {
                return false;
            }
            if (slot.ptr != nullptr) {
                CUDA_CHECK(cudaFreeHost(slot.ptr));
                pinned_pool_capacity_bytes_ -= slot.capacity;
                slot.ptr = nullptr;
                slot.capacity = 0;
            }
            CUDA_CHECK(cudaMallocHost(
                &slot.ptr,
                static_cast<std::size_t>(next_capacity)));
            slot.capacity = next_capacity;
            pinned_pool_capacity_bytes_ += next_capacity;
        }

        cudaStream_t stream = copy_streams_[next_copy_stream_];
        next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
        try {
            loader_.read_storage_bytes_to_host(
                shard_id, file_offset, span_bytes, slot.ptr);
        } catch (...) {
            throw;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            dst,
            slot.ptr,
            span_bytes,
            cudaMemcpyHostToDevice,
            stream));
        slot.stream = stream;
        slot.busy = true;
        slot.inflight_bytes = span_bytes;
        pending_pinned_bytes_ += span_bytes;
        ++pending_copy_count_;
        if (active_stats_ != nullptr) {
            ++active_stats_->h2d_pinned_copy_count;
            active_stats_->h2d_pinned_copy_bytes += span_bytes;
            active_stats_->max_pending_copies_seen =
                std::max(
                    active_stats_->max_pending_copies_seen,
                    pending_copy_count_);
        }
        if (pending_copy_count_ >= max_pending_copies_) {
            flush_copy_streams();
        }
        return true;
    }

    static std::uint64_t next_power_of_two(std::uint64_t value)
    {
        if (value <= 1) {
            return 1;
        }
        --value;
        for (std::size_t shift = 1; shift < 64; shift <<= 1) {
            value |= value >> shift;
        }
        return value + 1;
    }

    void ensure_pinned_slots()
    {
        if (!pinned_slots_.empty()) {
            return;
        }
        pinned_slots_.resize(pinned_staging_slot_count());
    }
#endif

    bool copy_streams_enabled() const
    {
        const char* disable =
            std::getenv("PIE_CUDA_DISABLE_PARALLEL_WEIGHT_COPIES");
        return !(disable != nullptr && disable[0] != '\0' && disable[0] != '0');
    }

    void ensure_copy_streams()
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        if (!copy_streams_.empty()) {
            return;
        }
        std::size_t count = 8;
        if (const char* env = std::getenv("PIE_CUDA_WEIGHT_COPY_STREAMS")) {
            const auto parsed = std::strtoull(env, nullptr, 10);
            if (parsed > 0) {
                count = std::min<std::size_t>(parsed, 32);
            }
        }
        copy_streams_.resize(count);
        for (auto& stream : copy_streams_) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
#endif
    }

    void flush_copy_streams()
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        if (pending_copy_count_ == 0) {
            return;
        }
        flush_batched_copies();
        for (auto stream : copy_streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        release_inflight_pinned_slots();
        if (active_stats_ != nullptr) {
            ++active_stats_->copy_stream_flushes;
        }
        pending_copy_count_ = 0;
#endif
    }

#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    void flush_batched_copies()
    {
#if CUDART_VERSION >= 12080
        if (pending_copies_.empty()) {
            return;
        }
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attr.flags = cudaMemcpyFlagDefault;
        std::size_t attr_index = 0;
        for (auto stream : copy_streams_) {
            batched_dsts_.clear();
            batched_srcs_.clear();
            batched_sizes_.clear();
            for (const auto& copy : pending_copies_) {
                if (copy.stream != stream) {
                    continue;
                }
                batched_dsts_.push_back(copy.dst);
                batched_srcs_.push_back(copy.src);
                batched_sizes_.push_back(copy.size);
            }
            if (batched_dsts_.empty()) {
                continue;
            }
            for (std::size_t i = 0; i < batched_dsts_.size(); ++i) {
                CUDA_CHECK(cudaMemcpyAsync(
                    batched_dsts_[i], batched_srcs_[i], batched_sizes_[i],
                    cudaMemcpyHostToDevice, stream));
                if ((i & 1023) == 1023) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (active_stats_ != nullptr) {
                ++active_stats_->h2d_batch_calls;
            }
        }
        pending_copies_.clear();
#else
        pending_copies_.clear();
#endif
    }
#endif

#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    void release_inflight_pinned_slots() noexcept
    {
        for (auto& slot : pinned_slots_) {
            if (slot.busy) {
                slot.busy = false;
                slot.stream = nullptr;
                slot.inflight_bytes = 0;
            }
        }
        pending_pinned_bytes_ = 0;
    }

    void free_pinned_slots_noexcept() noexcept
    {
        for (auto& slot : pinned_slots_) {
            if (slot.ptr != nullptr) {
                (void)cudaFreeHost(slot.ptr);
                slot.ptr = nullptr;
            }
            slot.capacity = 0;
            slot.busy = false;
            slot.stream = nullptr;
            slot.inflight_bytes = 0;
        }
        pinned_slots_.clear();
        pinned_pool_capacity_bytes_ = 0;
        pending_pinned_bytes_ = 0;
        next_pinned_slot_ = 0;
    }
#endif

    void destroy_copy_streams_noexcept()
    {
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
        free_slab_buffers_noexcept();
        if (copy_streams_.empty()) {
            return;
        }
        for (auto stream : copy_streams_) {
            if (stream != nullptr) {
                if (pending_copy_count_ != 0) {
                    (void)cudaStreamSynchronize(stream);
                }
                (void)cudaStreamDestroy(stream);
            }
        }
        free_pinned_slots_noexcept();
        copy_streams_.clear();
        pending_copy_count_ = 0;
        next_copy_stream_ = 0;
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
            copy_storage_bytes_to_device(
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
                copy_storage_bytes_to_device(
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
                copy_storage_bytes_to_device(
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

    void convert_e8m0_scale_to_f32(const std::string& name)
    {
        const auto& t = weights_.get(name);
        if (t.dtype() == DType::FP32) return;
        if (t.dtype() != DType::UINT8) return;
        const auto& shape = t.shape();
        const std::size_t n = t.numel();
        std::vector<std::uint8_t> e8m0(n);
        cudaMemcpy(e8m0.data(), t.data(), n, cudaMemcpyDeviceToHost);

        if (shape.size() == 2) {
            // 2D block scale [row_blocks, col_blocks] → expand to
            // per-channel [row_blocks * 128] using max across columns
            const int row_blocks = static_cast<int>(shape[0]);
            const int col_blocks = static_cast<int>(shape[1]);
            const int rows = row_blocks * 128;
            std::vector<float> per_row(static_cast<std::size_t>(rows));
            for (int rb = 0; rb < row_blocks; ++rb) {
                float max_scale = 0.f;
                for (int cb = 0; cb < col_blocks; ++cb) {
                    float s = std::ldexp(1.0f,
                        static_cast<int>(e8m0[static_cast<std::size_t>(rb) * col_blocks + cb]) - 127);
                    if (s > max_scale) max_scale = s;
                }
                for (int r = 0; r < 128; ++r) {
                    per_row[static_cast<std::size_t>(rb) * 128 + r] = max_scale;
                }
            }
            auto converted = DeviceTensor::allocate(DType::FP32, {rows});
            cudaMemcpy(converted.data(), per_row.data(),
                       static_cast<std::size_t>(rows) * sizeof(float),
                       cudaMemcpyHostToDevice);
            weights_.insert(name + ".f32", std::move(converted));
        } else {
            std::vector<float> f32(n);
            for (std::size_t i = 0; i < n; ++i) {
                f32[i] = std::ldexp(1.0f, static_cast<int>(e8m0[i]) - 127);
            }
            auto converted = DeviceTensor::allocate(DType::FP32,
                {static_cast<int>(n)});
            cudaMemcpy(converted.data(), f32.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice);
            weights_.insert(name + ".f32", std::move(converted));
        }
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
            convert_e8m0_scale_to_f32(attachment.scale_tensor_name);
            const std::string f32_name = attachment.scale_tensor_name + ".f32";
            const bool has_f32 = weights_.find(f32_name) != weights_.end();
            const std::string& scale_name = has_f32 ? f32_name : attachment.scale_tensor_name;
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
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    std::vector<cudaStream_t> copy_streams_;
#endif
    std::size_t next_copy_stream_ = 0;
    std::size_t pending_copy_count_ = 0;
    std::size_t max_pending_copies_ = 512;
    LoadExecutionStats* active_stats_ = nullptr;
#if PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
    struct PendingCopy {
        void* dst = nullptr;
        void* src = nullptr;
        std::size_t size = 0;
        cudaStream_t stream = nullptr;
    };
    struct PinnedSlot {
        void* ptr = nullptr;
        std::uint64_t capacity = 0;
        std::uint64_t inflight_bytes = 0;
        cudaStream_t stream = nullptr;
        bool busy = false;
    };
    std::vector<PendingCopy> pending_copies_;
    std::vector<PinnedSlot> pinned_slots_;
    std::size_t next_pinned_slot_ = 0;
    std::uint64_t pinned_pool_capacity_bytes_ = 0;
    std::uint64_t pending_pinned_bytes_ = 0;
    std::vector<void*> batched_dsts_;
    std::vector<void*> batched_srcs_;
    std::vector<std::size_t> batched_sizes_;
    void* slab_staging_ = nullptr;
    std::uint64_t slab_staging_capacity_ = 0;
    SlabScatterPlacement* slab_placements_device_ = nullptr;
    std::size_t slab_placement_capacity_ = 0;
    std::vector<SlabScatterPlacement> slab_placement_host_;
#endif
    wl_cpp::StorageProgramIndex program_index_{"rust storage executor"};
};

}  // namespace pie_cuda_driver

#undef PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
