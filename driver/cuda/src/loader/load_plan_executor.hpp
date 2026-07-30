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

#include "pie_loader/plan.hpp"
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
#endif
#include "loader/rust_quant_attachment.hpp"
#include "pie_loader/checkpoint_source.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

/// The two things that differ between running a plan as the model's resident
/// load and running one instance of a group into a slot.
///
/// A group plan is a whole plan, compiled at instance 0, which is exactly why
/// the same executor runs it. What the caller has to say is where its
/// persistent buffers land -- a slot it owns, not an arena the executor
/// allocates -- and which instance's bytes to read. Nothing else about
/// execution changes, and nothing here says whether the group is being
/// streamed or made resident; that is the caller's business.
struct LoadPlanExecution {
    /// Where the plan's persistent buffers go. Null means allocate one, which
    /// is the resident load and the default.
    std::uint8_t* persistent_arena = nullptr;
    std::uint64_t persistent_arena_bytes = 0;

    /// One instance's slice of `PieLoaderGroupView::bindings`. Each entry
    /// replaces the source extent of the instruction it names; the rest of the
    /// instruction -- destination, stride, span, dtype -- is index-independent
    /// by construction, which is what the loader proved when it accepted the
    /// group. Empty means read what the plan was compiled to read.
    const pie_loader::PieLoaderSourceBindingView* source_bindings = nullptr;
    std::size_t source_binding_count = 0;
};

class LoadPlanExecutor {
public:
    /// `lent_copy_engine` is for a caller that runs many short plans -- a
    /// group cache paging one instance per miss. An engine creates its copy
    /// streams and pinned pool lazily and tears them down with itself, which
    /// is free amortised over a whole model load and is the dominant cost of
    /// a page-in that moves a few megabytes. Lending one hoists that setup out
    /// of the loop. Null means own one, which is the resident load.
    LoadPlanExecutor(
        pie_loader::CheckpointSource& loader,
        WeightStoreBuilder& weights,
        std::vector<RustQuantAttachment> quant_attachments,
        WeightCopyEngine* lent_copy_engine = nullptr)
        : loader_(loader),
          weights_(weights),
          quant_attachments_(std::move(quant_attachments)),
          copy_engine_(lent_copy_engine != nullptr ? *lent_copy_engine
                                                   : owned_copy_engine_)
    {}

    LoadExecutionStats execute(const pie_loader::LoadPlanView& plan)
    {
        return execute(plan, LoadPlanExecution{});
    }

    LoadExecutionStats execute(
        const pie_loader::LoadPlanView& plan,
        const LoadPlanExecution& how)
    {
        LoadExecutionStats stats;
        stats.planned_tensor_count = plan.tensors.len;
        stats.planned_storage_peak_bytes = plan.memory.persistent_bytes +
            plan.memory.temporary_peak_bytes;
        stats.planned_storage_temp_bytes = plan.memory.temporary_peak_bytes;
        plan_index_.reset(plan);
        bind_sources(how);
        init_persistent_arena(plan, how);
        copy_engine_.set_stats(&stats);
        const bool trace_executor =
            loader_config::env_present("PIE_CUDA_TRACE_STORAGE_EXECUTOR");
        if (trace_executor) {
            std::cerr << "[pie-driver-cuda] storage executor begin schedule="
                      << plan.schedule.len << " instrs=" << plan.instrs.len
                      << " buffers=" << plan.buffers.len << "\n";
        }

        for (std::size_t i = 0; i < plan.schedule.len; ++i) {
            const std::uint32_t instr_id = plan.schedule.ptr[i];
            const auto& instr = plan_index_.instruction(instr_id);
            using Tag = pie_loader::PieLoaderStorageOp::Tag;
            if (trace_executor && (i < 128 || i % 1000 == 0 ||
                    instr.op.tag != Tag::Allocate)) {
                std::cerr << "[pie-driver-cuda] storage executor instr[" << i
                          << "] id=" << instr.id
                          << " tag=" << static_cast<int>(instr.op.tag) << "\n";
            }
            switch (instr.op.tag) {
            case Tag::Allocate:
                if (allocate_requires_copy_flush(instr.op.allocate)) {
                    copy_engine_.flush();
                }
                {
                    PhaseTimer _pt(&stats.phase_alloc_ms);
                    allocate(plan, instr.op.allocate);
                }
                if (trace_executor && (i < 128 || i % 1000 == 0)) {
                    std::cerr << "[pie-driver-cuda] storage executor allocated buffer="
                              << instr.op.allocate.buffer_id << "\n";
                }
                break;
            case Tag::Fill:
                fill(instr.op.fill);
                break;
            case Tag::ExtentWrite:
                extent_write(instr.id, instr.op.extent_write, stats);
                break;
            case Tag::BulkExtentWrite:
                bulk_extent_write(instr.id, instr.op.bulk_extent_write, stats);
                break;
            case Tag::Finalize:
                {
                    PhaseTimer _pt(&stats.phase_transform_ms);
                    finalize(plan, instr.op.finalize, stats);
                }
                break;
            case Tag::TileMap:
                copy_engine_.flush();
                {
                    PhaseTimer _pt(&stats.phase_transform_ms);
                    transcode_.tile_map(
                        instr.op.tile_map,
                        resolved_source(instr.id, instr.op.tile_map.source),
                        stats);
                }
                break;
            case Tag::CreateView:
                copy_engine_.flush();
                create_view(plan, instr.op.create_view);
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
    /// Zero a buffer the plan is about to write only part of.
    ///
    /// The loader emits this when an expression pads: the padded region is
    /// device memory no source covers, and the compiler prices it as one fill
    /// rather than one copy per band. It must precede every write to the same
    /// buffer, which `validate-fill-order` guarantees on the Rust side; the
    /// stream ordering is what guarantees it here, so the pending copies have
    /// to be flushed first.
    void fill(const pie_loader::PieLoaderStorageOp::Fill_Body& instr)
    {
        copy_engine_.flush();
        auto it = buffers_.find(instr.buffer_id);
        if (it == buffers_.end()) {
            throw std::runtime_error("rust storage executor: Fill buffer missing");
        }
        CUDA_CHECK(cudaMemsetAsync(
            it->second.data(), 0, it->second.nbytes(), copy_engine_.acquire_stream()));
    }

    void allocate(
        const pie_loader::LoadPlanView& plan,
        const pie_loader::PieLoaderStorageOp::Allocate_Body& instr)
    {
        const auto& buffer = plan_index_.buffer(instr.buffer_id);
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
        const auto& tensor = plan_index_.tensor(buffer.tensor_id);
        if (tensor.encoding_kind ==
            pie_loader::PieLoaderEncodingKind::Quant) {
            const DType physical = quant_physical_dtype(tensor);
            if (physical == DType::FP8_E4M3 || physical == DType::INT8) {
                buffers_.emplace(
                    buffer.id,
                    DeviceTensor::allocate(
                        physical,
                        pie_loader::i64_slice_to_vector(tensor.shape)));
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
                pie_loader::i64_slice_to_vector(tensor.shape)));
    }

    bool allocate_requires_copy_flush(
        const pie_loader::PieLoaderStorageOp::Allocate_Body& instr) const
    {
        const auto& buffer = plan_index_.buffer(instr.buffer_id);
        return !can_allocate_persistent_arena_view(buffer);
    }

    void init_persistent_arena(
        const pie_loader::LoadPlanView& plan,
        const LoadPlanExecution& how)
    {
        if (how.persistent_arena != nullptr) {
            // A caller-owned destination. The arena is not this executor's to
            // allocate, not the weight store's to hold, and not named -- every
            // buffer that lands in it is a bare view, so tearing down whatever
            // store collected them frees nothing.
            if (how.persistent_arena_bytes < plan.memory.persistent_bytes) {
                throw std::runtime_error(
                    "rust storage executor: caller arena of " +
                    std::to_string(how.persistent_arena_bytes) +
                    " bytes is smaller than the plan's " +
                    std::to_string(plan.memory.persistent_bytes));
            }
            persistent_arena_base_ = how.persistent_arena;
            persistent_arena_bytes_ = how.persistent_arena_bytes;
            external_arena_ = true;
            return;
        }
        if (loader_config::env_truthy("PIE_CUDA_DISABLE_WEIGHT_ARENA")) {
            return;
        }
        if (plan.memory.persistent_bytes == 0) {
            return;
        }
        persistent_arena_name_ = "__pie.storage_arena.0";
        DeviceTensor arena = DeviceTensor::allocate(
            DType::UINT8,
            {static_cast<std::int64_t>(plan.memory.persistent_bytes)});
        persistent_arena_base_ = static_cast<std::uint8_t*>(arena.data());
        persistent_arena_bytes_ = arena.nbytes();
        TensorDecl spec;
        spec.name = persistent_arena_name_;
        spec.dtype = DType::UINT8;
        spec.shape = {static_cast<std::int64_t>(persistent_arena_bytes_)};
        spec.ownership = TensorOwnershipKind::Owned;
        weights_.insert(persistent_arena_name_, std::move(arena), std::move(spec));
    }

    bool try_allocate_persistent_arena_view(
        const pie_loader::PieLoaderBufferDeclView& buffer)
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

        const auto& tensor = plan_index_.tensor(buffer.tensor_id);
        DType dtype = DType::UINT8;
        std::vector<std::int64_t> shape;
        if (tensor.encoding_kind ==
            pie_loader::PieLoaderEncodingKind::Quant) {
            const DType physical = quant_physical_dtype(tensor);
            if (physical == DType::FP8_E4M3 || physical == DType::INT8) {
                dtype = physical;
                shape = pie_loader::i64_slice_to_vector(tensor.shape);
            } else {
                dtype = DType::UINT8;
                shape = {static_cast<std::int64_t>(buffer.bytes)};
            }
        } else {
            dtype = dtype_from_rust(tensor.dtype);
            shape = pie_loader::i64_slice_to_vector(tensor.shape);
        }
        buffers_.emplace(
            buffer.id,
            DeviceTensor::view(persistent_arena_base_ + aligned, dtype, shape));
        if (!persistent_arena_name_.empty()) {
            arena_backing_names_[buffer.id] = persistent_arena_name_;
        }
        return true;
    }

    bool can_allocate_persistent_arena_view(
        const pie_loader::PieLoaderBufferDeclView& buffer) const
    {
        return persistent_arena_base_ != nullptr && buffer.has_tensor &&
            !buffer.temporary && buffer.bytes != 0;
    }

    /// Index this instance's bindings by the instruction each one rebinds.
    void bind_sources(const LoadPlanExecution& how)
    {
        source_binding_by_instr_.clear();
        for (std::size_t i = 0; i < how.source_binding_count; ++i) {
            const auto& binding = how.source_bindings[i];
            source_binding_by_instr_[binding.instr_id] = &binding;
        }
    }

    /// The source `instr_id` reads, after this instance's binding if it has
    /// one. A binding replaces the three fields that name bytes on disk and
    /// nothing else, so the returned extent keeps the template's stride, span
    /// and dtype -- the parts the loader proved index-independent.
    pie_loader::PieLoaderSourceExtentView resolved_source(
        std::uint32_t instr_id,
        const pie_loader::PieLoaderSourceExtentView& source) const
    {
        if (source_binding_by_instr_.empty()) return source;
        const auto it = source_binding_by_instr_.find(instr_id);
        if (it == source_binding_by_instr_.end()) return source;
        pie_loader::PieLoaderSourceExtentView bound = source;
        bound.file_id = it->second->file_id;
        bound.tensor_id = it->second->tensor_id;
        bound.file_offset = it->second->file_offset;
        return bound;
    }

    void extent_write(
        std::uint32_t instr_id,
        const pie_loader::PieLoaderStorageOp::ExtentWrite_Body& instr_body,
        LoadExecutionStats& stats)
    {
        const auto& dest = instr_body.dest;
        const auto source = resolved_source(instr_id, instr_body.source);
        auto dst_it = buffers_.find(dest.buffer_id);
        if (dst_it == buffers_.end()) {
            throw std::runtime_error(
                "rust storage executor: destination buffer missing");
        }
        auto* dst = static_cast<std::uint8_t*>(dst_it->second.data()) +
            dest.offset + dest.stride.base_offset;
        if (!pie_loader::compact_extent(dest.stride)) {
            throw std::runtime_error(
                "rust storage executor: non-compact ExtentWrite destination is not "
                "implemented");
        }
        if (!pie_loader::compact_extent(source.stride)) {
            const std::uint64_t dst_offset =
                dest.offset + dest.stride.base_offset;
            const std::uint64_t dst_total = dst_it->second.nbytes();
            copy_strided_extent_to_device(
                loader_, source,
                dst,
                dst_offset > dst_total ? 0 : dst_total - dst_offset);
            return;
        }
        copy_engine_.queue(
            source.file_id,
            source.file_offset + source.stride.base_offset,
            source.span_bytes,
            dst);
        ++stats.h2d_copy_count;
        stats.h2d_copy_bytes += source.span_bytes;
    }

    void bulk_extent_write(
        std::uint32_t instr_id,
        const pie_loader::PieLoaderStorageOp::BulkExtentWrite_Body& instr_body,
        LoadExecutionStats& stats)
    {
        const auto source = resolved_source(instr_id, instr_body.source);
        if (persistent_arena_base_ == nullptr) {
            throw std::runtime_error(
                "rust storage executor: BulkExtentWrite requires persistent arena");
        }
        const std::uint64_t dst_offset = instr_body.dest_offset;
        if (dst_offset > persistent_arena_bytes_ ||
            source.span_bytes > persistent_arena_bytes_ - dst_offset) {
            throw std::runtime_error(
                "rust storage executor: BulkExtentWrite destination out of bounds");
        }
        copy_engine_.queue(
            source.file_id,
            source.file_offset + source.stride.base_offset,
            source.span_bytes,
            persistent_arena_base_ + dst_offset);
        ++stats.h2d_copy_count;
        ++stats.h2d_bulk_copy_count;
        stats.h2d_copy_bytes += source.span_bytes;
        stats.h2d_bulk_copy_bytes += source.span_bytes;
    }

    void create_view(
        const pie_loader::LoadPlanView& plan,
        const pie_loader::PieLoaderStorageOp::CreateView_Body& instr)
    {
        const auto input_id = instr.input_buffer;
        const auto output_id = instr.output_buffer;
        const DeviceTensor& input = resolver_.or_finalized(input_id);
        const auto& output_buffer = plan_index_.buffer(output_id);
        if (!output_buffer.has_tensor) {
            throw std::runtime_error(
                "rust storage executor: CreateView output buffer has no "
                "tensor declaration");
        }
        const auto& tensor = plan_index_.tensor(output_buffer.tensor_id);
        const auto shape = pie_loader::i64_slice_to_vector(tensor.shape);
        const auto* input_base =
            static_cast<const std::uint8_t*>(input.data()) + instr.view.offset +
            instr.view.stride.base_offset;
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
            const auto& input_buffer = plan_index_.buffer(input_id);
            if (input_buffer.has_tensor) {
                backing_name =
                    pie_loader::bytes_to_string(
                        plan_index_.tensor(input_buffer.tensor_id).name);
            }
        }
        if (!backing_name.empty()) {
            view_backing_names_[output_id] = std::move(backing_name);
        }
    }

    void finalize(
        const pie_loader::LoadPlanView& plan,
        const pie_loader::PieLoaderStorageOp::Finalize_Body& instr,
        LoadExecutionStats& stats)
    {
        auto buffer = buffers_.extract(instr.buffer_id);
        if (buffer.empty()) {
            throw std::runtime_error(
                "rust storage executor: finalize buffer missing");
        }
        const auto& buffer_info = plan_index_.buffer(instr.buffer_id);
        const auto& tensor = plan_index_.tensor(buffer_info.tensor_id);
        TensorDecl spec;
        spec.name = pie_loader::bytes_to_string(tensor.name);
        if (tensor.encoding_kind ==
            pie_loader::PieLoaderEncodingKind::Quant) {
            const DType physical = quant_physical_dtype(tensor);
            if (physical == DType::FP8_E4M3 || physical == DType::INT8) {
                spec.dtype = physical;
                spec.shape = pie_loader::i64_slice_to_vector(tensor.shape);
            } else {
                spec.dtype = DType::UINT8;
                spec.shape = {static_cast<std::int64_t>(buffer.mapped().nbytes())};
            }
        } else {
            spec.dtype = dtype_from_rust(tensor.dtype);
            spec.shape = pie_loader::i64_slice_to_vector(tensor.shape);
        }
        spec.ownership = TensorOwnershipKind::Owned;
        if (auto backing = view_backing_names_.find(instr.buffer_id);
            backing != view_backing_names_.end()) {
            spec.ownership = TensorOwnershipKind::BorrowedView;
            spec.backing_tensor = backing->second;
        } else if (auto backing = arena_backing_names_.find(instr.buffer_id);
                   backing != arena_backing_names_.end()) {
            spec.ownership = TensorOwnershipKind::BorrowedView;
            spec.backing_tensor = backing->second;
        } else if (external_arena_ && !buffer.mapped().owns_memory()) {
            // Lent arena: nothing in this store owns these bytes, and nothing
            // in it can, because the arena is the caller's.
            spec.ownership = TensorOwnershipKind::External;
        }
        stats.loaded_bytes += buffer.mapped().nbytes();
        const std::string runtime_name = pie_loader::bytes_to_string(instr.name);
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
            // Some kernels want the scale bytes as they are — the MXFP4 GEMM,
            // the dequant kernels and make_expert_weight_view all require U8
            // E8M0 and gemm.cpp asserts on it — while the block-scaled FP8 path
            // wants them expanded to F32 2^(b-127) factors. The plan says which
            // (`load_plan.rs`, `ScaleForm`); this used to be inferred from
            // `group_size == 32`, which held only because MXFP4 happens to be
            // the one scheme with that group size.
            std::string scale_name = attachment.scale_tensor_name;
            if (attachment.scale_form ==
                pie_loader::PieLoaderScaleForm::F32Factors) {
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

    pie_loader::CheckpointSource& loader_;
    WeightStoreBuilder& weights_;
    std::vector<RustQuantAttachment> quant_attachments_;
    std::unordered_map<std::uint32_t, DeviceTensor> buffers_;
    std::unordered_map<std::uint32_t, std::string> finalized_buffer_names_;
    std::unordered_map<std::uint32_t, std::string> view_backing_names_;
    std::unordered_map<std::uint32_t, std::string> arena_backing_names_;
    // This instance's source rebindings, by the instruction each replaces.
    // Empty for a resident load, which is every plan that is not a group's.
    std::unordered_map<std::uint32_t, const pie_loader::PieLoaderSourceBindingView*>
        source_binding_by_instr_;
    std::string persistent_arena_name_;
    bool external_arena_ = false;
    std::uint8_t* persistent_arena_base_ = nullptr;
    std::uint64_t persistent_arena_bytes_ = 0;
    // Host->device copy path (streams, pinned staging, reader lanes, batching).
    // Unused, and so never spun up, when the caller lends one.
    WeightCopyEngine owned_copy_engine_{loader_};
    WeightCopyEngine& copy_engine_;
    pie_loader::LoadPlanIndex plan_index_{"load plan executor"};
    BufferResolver resolver_{buffers_, finalized_buffer_names_, weights_};
    TranscodeEngine transcode_{loader_, copy_engine_, plan_index_, resolver_};
};

}  // namespace pie_cuda_driver

#undef PIE_CUDA_RUST_STORAGE_EXECUTOR_HAS_CUDA
