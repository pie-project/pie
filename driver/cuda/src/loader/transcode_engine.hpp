#pragma once

// TranscodeEngine: the quant/transcode TileMap path — Cast, Encode
// (FP8->bf16->FP8/MXFP4, fused or staged), Repack (Marlin) and Reblock/Reorder.
// Factored out of the storage executor; it consumes source bytes (loader + copy
// engine), resolves input/output buffers (resolver), reads the storage program
// (program index), and owns the FP8 encode scratch buffers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pie_native/storage_program.hpp"
#include "loader_config.hpp"
#include "loader_helpers.hpp"
#include "tensor.hpp"
#include "loader/checkpoint_source.hpp"
#include "loader/buffer_resolver.hpp"
#include "loader/strided_copy.hpp"
#include "loader/weight_copy_engine.hpp"

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA 1
#endif
#endif
#ifndef PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
#define PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA 0
#endif
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
#include <cuda_runtime.h>
#include "cuda_check.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/mxfp4_marlin.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "kernels/quant_bf16_to_mxfp4.hpp"
#include "kernels/transcode.hpp"
#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif
#endif

namespace pie_cuda_driver {

namespace wl_cpp = pie_weight_loader::cpp;

class TranscodeEngine {
public:
    TranscodeEngine(CheckpointSource& loader,
                    WeightCopyEngine& copy_engine,
                    const wl_cpp::StorageProgramIndex& program_index,
                    BufferResolver& resolver)
        : loader_(loader), copy_engine_(copy_engine), program_index_(program_index),
          resolver_(resolver) {}

    ~TranscodeEngine() { free_scratch_noexcept(); }
    TranscodeEngine(const TranscodeEngine&) = delete;
    TranscodeEngine& operator=(const TranscodeEngine&) = delete;

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

private:
    void free_scratch_noexcept() noexcept
    {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        if (fp8_bf16_scratch_ptr_ != nullptr) { cudaFree(fp8_bf16_scratch_ptr_); fp8_bf16_scratch_ptr_ = nullptr; }
        if (fp8_scale_local_ptr_ != nullptr) { cudaFree(fp8_scale_local_ptr_); fp8_scale_local_ptr_ = nullptr; }
        if (fp8_source_tile_ptr_ != nullptr) { cudaFree(fp8_source_tile_ptr_); fp8_source_tile_ptr_ = nullptr; }
        if (bf16_source_tile_ptr_ != nullptr) { cudaFree(bf16_source_tile_ptr_); bf16_source_tile_ptr_ = nullptr; }
        for (auto& kv : fp8_scale_cache_) { if (kv.second.data != nullptr) cudaFree(kv.second.data); }
        fp8_scale_cache_.clear();
#endif
    }

    static void cast_tensor_to_ptr(
        const DeviceTensor& src,
        void* dst,
        DType dst_dtype)
    {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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

    void cast_tile_map(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Cast TileMap expects one output");
        }
        const auto output_id = instr.output_buffers.ptr[0];
        DeviceTensor& out = resolver_.tensor(output_id);
        const auto dst_offset =
            instr.has_dest ? instr.dest.offset + instr.dest.stride.base_offset : 0;
        auto* dst = static_cast<std::uint8_t*>(out.data()) + dst_offset;

        if (instr.has_source) {
            if (!wl_cpp::compact_extent(instr.source.stride)) {
                throw std::runtime_error(
                    "rust storage executor: non-compact Cast source is not "
                    "implemented");
            }
            const auto& info = program_index_.source(instr.source.tensor_id);
            DeviceTensor scratch =
                DeviceTensor::allocate(
                    dtype_from_rust(info.dtype),
                    wl_cpp::extent_shape(instr.source.stride));
            if (scratch.nbytes() != instr.source.span_bytes) {
                throw std::runtime_error(
                    "rust storage executor: Cast source byte size mismatch");
            }
            copy_engine_.queue(
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
        cast_tensor_to_ptr(resolver_.or_finalized(instr.input_buffers.ptr[0]), dst, out.dtype());
    }

    // Acquire the Encode source tile on device (FP8 / other source bytes, or a
    // slice of an input buffer) WITHOUT dequantizing. Shared by the BF16
    // materialize path and the fused FP8->MXFP4 transcode path.
    DeviceTensor acquire_encode_source_tile(
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
            const auto& info = program_index_.source(instr.source.tensor_id);
            const DType source_dtype = dtype_from_rust(info.dtype);
            const bool compact = wl_cpp::compact_extent(instr.source.stride);
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
            // Reuse a persistent device tile buffer for compact sources — the
            // encode/dequant kernel consumes it then we move on, so per-tile
            // cudaMalloc/cudaFree is pure overhead (and dominates the FP4 encode
            // phase for tens of thousands of expert tiles). FP8 and
            // BF16/FP16/FP32 sources keep separate persistent buffers (different
            // element sizes); each is wrapped in a non-owning view. The H2D
            // below runs on stream 0, and so does the follow-up kernel, so a
            // single reused buffer is safe (stream-0 in-order) — and there's no
            // flush. Strided (TP-sharded non-compact) sources fall back to a
            // per-tile allocate: the generic strided copy isn't stream-0 ordered.
            if (compact) {
                const std::size_t want_bytes =
                    static_cast<std::size_t>(rows) *
                    static_cast<std::size_t>(cols) *
                    dtype_bytes(source_dtype);
                const bool is_fp8 = source_dtype == DType::FP8_E4M3;
                void*& tile_ptr =
                    is_fp8 ? fp8_source_tile_ptr_ : bf16_source_tile_ptr_;
                std::size_t& tile_cap =
                    is_fp8 ? fp8_source_tile_bytes_ : bf16_source_tile_bytes_;
                ensure_dev_buffer(tile_ptr, tile_cap, want_bytes);
                source = DeviceTensor::view(tile_ptr,                 source_dtype, tile_shape);
            } else
#endif
            {
                source = DeviceTensor::allocate(source_dtype, tile_shape);
            }
            if (!compact) {
                if (row_start != 0 || rows != full_shape[0]) {
                    throw std::runtime_error(
                        "rust storage executor: tiled Encode for non-compact "
                        "sources is not implemented");
                }
                copy_strided_extent_to_device(
                    loader_, instr, source.data(), full_shape);
            } else {
                const std::uint64_t elem = dtype_bytes(source_dtype);
                const std::uint64_t row_bytes =
                    static_cast<std::uint64_t>(cols) * elem;
                const std::uint64_t off =
                    instr.source.file_offset +
                    instr.source.stride.base_offset +
                    static_cast<std::uint64_t>(row_start) * row_bytes;
                const std::uint64_t span =
                    static_cast<std::uint64_t>(rows) * row_bytes;
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
                // Stream-0 H2D: the follow-up encode/dequant kernel (also stream
                // 0) sees it via implicit ordering — no flush, no event wait.
                copy_engine_.queue_on_stream(
                    instr.source.file_id, off, span, source.data(),
                    /*stream=*/0);
#else
                copy_engine_.queue(
                    instr.source.file_id, off, span, source.data());
#endif
            }
        } else {
            if (instr.input_buffers.len != 1) {
                throw std::runtime_error(
                    "rust storage executor: Encode expects source or one input");
            }
            const DeviceTensor& input =
                resolver_.or_finalized(instr.input_buffers.ptr[0]);
            source = DeviceTensor::allocate(input.dtype(), tile_shape);
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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
        return source;
    }

    DeviceTensor materialize_encode_input_bf16_rows(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows)
    {
        const std::vector<std::int64_t> tile_shape{
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(full_shape[1]),
        };
        DeviceTensor source =
            acquire_encode_source_tile(instr, full_shape, row_start, rows);
        if (source.dtype() == DType::BF16) {
            return source;
        }
        if (source.dtype() == DType::FP8_E4M3) {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
            // FP8 (E4M3) source: dequant to BF16 using the per-group block
            // scale that ships alongside the weight. For GLM-5.1 the scale
            // tensor is `<weight>_scale_inv` with shape [rows/128, cols/128]
            // and dtype FP32 (one float per 128x128 block of the weight).
            return dequant_fp8_tile_to_bf16(
                instr, source, full_shape, row_start, rows, tile_shape);
#else
            throw std::runtime_error(
                "rust storage executor: FP8 Encode requires CUDA support");
#endif
        }
        DeviceTensor bf16 = DeviceTensor::allocate(DType::BF16, tile_shape);
        cast_tensor_to_ptr(source, bf16.data(), DType::BF16);
        return bf16;
    }

#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
    // Grow a persistent device buffer to at least `want_bytes`. Uses
    // cudaMalloc/cudaFree only when growth is needed, so steady-state
    // tile encoding does zero per-call allocations.
    void ensure_dev_buffer(void*& ptr, std::size_t& cap, std::size_t want_bytes)
    {
        if (cap >= want_bytes && ptr != nullptr) return;
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            cap = 0;
        }
        if (cudaMalloc(&ptr, want_bytes) != cudaSuccess) {
            throw std::runtime_error(
                "rust storage executor: cudaMalloc for FP8 scratch buffer failed (" +
                std::to_string(want_bytes) + " bytes)");
        }
        cap = want_bytes;
    }

    // Load the FP8 scale tensor for `scale_name` to a persistent device
    // buffer once and reuse for every subsequent tile of the same weight.
    // For GLM-5.1 we have ~58k expert weights × multiple tiles each, so
    // caching saves both disk I/O and cudaMalloc churn.
    void ensure_fp8_scale_loaded(
        const std::string& scale_name,
        const pie_weight_loader::PieLoaderSourceTensorView& scale_info,
        std::size_t scale_nbytes,
        const pie_weight_loader::PieLoaderSourceTensorView& storage)
    {
        auto it = fp8_scale_cache_.find(scale_name);
        if (it != fp8_scale_cache_.end()) return;
        CachedFp8Scale entry;
        if (cudaMalloc(&entry.data, scale_nbytes) != cudaSuccess) {
            throw std::runtime_error(
                "rust storage executor: cudaMalloc for FP8 scale cache failed");
        }
        entry.nbytes = scale_nbytes;
        // Stream-0 H2D so the dequant kernel (also on stream 0) sees the
        // scale via implicit ordering. Previously this did a full
        // flush_copy_streams() per cache miss — ~30k flushes for GLM-5.1's
        // expert weights, each syncing every copy stream. Stream-0 ordering
        // is free.
        copy_engine_.queue_on_stream(
            storage.file_id, storage.file_offset,
            storage.span_bytes, entry.data, /*stream=*/0);
        (void)scale_info;
        fp8_scale_cache_.emplace(scale_name, entry);
    }

    struct Fp8TileScale {
        const float* scale_dev;  // offset to this tile's first scale row
        int group_size;
        int local_cols;
    };

    // Resolve the per-group FP8 block scale for an Encode-source tile: loads/
    // caches `<weight>_scale_inv`, slices the rank-local block for TP shards,
    // and offsets to the tile's first scale row. Shared by the BF16 dequant and
    // the fused FP8->MXFP4 paths so both see identical scale data.
    Fp8TileScale fp8_tile_scale(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows)
    {
        if (!instr.has_source) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode requires a checkpoint source");
        }
        if (full_shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode source must be 2-D");
        }
        const auto& weight_info = program_index_.source(instr.source.tensor_id);
        const std::string weight_name =
            pie_weight_loader::cpp::bytes_to_string(weight_info.name);
        const std::string scale_name = weight_name + "_scale_inv";
        const auto* scale_info = program_index_.find_source(scale_name);
        if (scale_info == nullptr) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode source '" + weight_name +
                "' has no '_scale_inv' sibling tensor");
        }
        const auto scale_shape =
            pie_weight_loader::cpp::i64_slice_to_vector(scale_info->shape);
        if (scale_shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode scale '" + scale_name +
                "' must be 2-D (block-scaled FP8)");
        }
        // Get the FULL (un-sharded) weight shape from the checkpoint so we
        // can compute the true group_size. The tile's `full_shape` may be
        // TP-sharded and not match the on-disk scale dimensions.
        const auto weight_shape =
            pie_weight_loader::cpp::i64_slice_to_vector(weight_info.shape);
        if (weight_shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode weight '" + weight_name +
                "' must be 2-D on disk");
        }
        const int true_rows = checked_int(weight_shape[0], "FP8 weight rows");
        const int true_cols = checked_int(weight_shape[1], "FP8 weight cols");
        const int scale_rows = checked_int(scale_shape[0], "FP8 scale rows");
        const int scale_cols = checked_int(scale_shape[1], "FP8 scale cols");
        const int true_group_rows = (scale_rows > 0) ? (true_rows / scale_rows) : 0;
        const int true_group_cols = (scale_cols > 0) ? (true_cols / scale_cols) : 0;
        if (true_group_rows <= 0 || true_group_cols <= 0
            || true_group_rows != true_group_cols) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode source '" + weight_name +
                "' has unsupported scale shape");
        }
        const int group_size = true_group_rows;  // typically 128

        // Detect TP shard by comparing rank-local full_shape to on-disk shape.
        const int local_rows = checked_int(full_shape[0], "FP8 local rows");
        const int local_cols = checked_int(full_shape[1], "FP8 local cols");
        const int row_shard_factor = (local_rows > 0 && local_rows < true_rows)
            ? (true_rows / local_rows) : 1;
        const int col_shard_factor = (local_cols > 0 && local_cols < true_cols)
            ? (true_cols / local_cols) : 1;

        // Decode this rank's row/col offset within the full weight from
        // source.stride.base_offset (in bytes; FP8 weights are 1 byte/elem).
        const std::uint64_t base_byte = instr.source.stride.base_offset;
        const std::uint64_t rank_row_off_full = base_byte / true_cols;
        const std::uint64_t rank_col_off_full = base_byte % true_cols;

        if (dtype_from_rust(scale_info->dtype) != DType::FP32) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode scale '" + scale_name +
                "' must be FP32");
        }
        // Cache the full FP8 scale per weight: one disk read + one cudaMalloc
        // amortised across every tile of the same Encode instruction.
        const std::size_t scale_nbytes =
            static_cast<std::size_t>(scale_rows) * scale_cols * sizeof(float);
        ensure_fp8_scale_loaded(
            scale_name, *scale_info, scale_nbytes, *scale_info);
        const auto& cached_scale = fp8_scale_cache_[scale_name];
        const float* scale_full_ptr =
            static_cast<const float*>(cached_scale.data);

        // For TP-sharded weights we need a compact rank-local scale slice
        // so the kernel sees contiguous [local_rows/gs, local_cols/gs] data.
        // Use a persistent device buffer that grows on demand.
        const int local_scale_rows = local_rows / group_size;
        const int local_scale_cols = local_cols / group_size;
        const float* scale_for_kernel = scale_full_ptr;
        if (row_shard_factor != 1 || col_shard_factor != 1) {
            const std::size_t want_bytes =
                static_cast<std::size_t>(local_scale_rows) *
                local_scale_cols * sizeof(float);
            ensure_dev_buffer(fp8_scale_local_ptr_, fp8_scale_local_bytes_,
                              want_bytes);
            const int rank_scale_row_off =
                static_cast<int>(rank_row_off_full) / group_size;
            const int rank_scale_col_off =
                static_cast<int>(rank_col_off_full) / group_size;
            // One D2D per scale row of the rank's slice. Tiny copies; the
            // batched async memcpys overlap well on the default stream.
            for (int r = 0; r < local_scale_rows; ++r) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<float*>(fp8_scale_local_ptr_)
                        + static_cast<std::size_t>(r) * local_scale_cols,
                    scale_full_ptr
                        + static_cast<std::size_t>(rank_scale_row_off + r) *
                              scale_cols
                        + rank_scale_col_off,
                    static_cast<std::size_t>(local_scale_cols) * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    /*stream=*/0));
            }
            scale_for_kernel = static_cast<const float*>(fp8_scale_local_ptr_);
        }

        if (row_start % group_size != 0 && row_start + rows != local_rows) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode tile row range must align "
                "to scale group rows");
        }
        const int scale_row_start = row_start / group_size;
        const float* scale_dev =
            scale_for_kernel +
            static_cast<std::size_t>(scale_row_start) * local_scale_cols;
        return Fp8TileScale{scale_dev, group_size, local_cols};
    }

    DeviceTensor dequant_fp8_tile_to_bf16(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        const DeviceTensor& fp8_tile,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows,
        const std::vector<std::int64_t>& tile_shape)
    {
        const Fp8TileScale s = fp8_tile_scale(instr, full_shape, row_start, rows);
        // Persistent BF16 scratch — grown once, reused for every tile.
        const std::size_t bf16_bytes =
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(s.local_cols) * sizeof(std::uint16_t);
        ensure_dev_buffer(fp8_bf16_scratch_ptr_, fp8_bf16_scratch_bytes_,
                          bf16_bytes);
        // The FP8 source tile is enqueued on stream 0 by
        // acquire_encode_source_tile, so this dequant (also stream 0) sees
        // those bytes via implicit stream ordering — no flush needed.
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_group(
            static_cast<const std::uint8_t*>(fp8_tile.data()),
            fp8_bf16_scratch_ptr_,
            s.scale_dev,
            rows,
            s.local_cols,
            s.group_size,
            /*stream=*/0);
        CUDA_CHECK(cudaGetLastError());
        return DeviceTensor::view(
            fp8_bf16_scratch_ptr_, DType::BF16, tile_shape);
    }

    // Fused FP8 (per-group) -> MXFP4 for one tile, writing directly into the
    // MXFP4 packed/scale outputs (no BF16 HBM round-trip). Bit-identical to
    // dequant_fp8_tile_to_bf16 + quantize_bf16_to_mxfp4 — the fused kernel
    // rounds through BF16; see tests/test_transcode_fused.cu.
    void transcode_fp8_tile_to_mxfp4(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        const DeviceTensor& fp8_tile,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows,
        std::uint8_t* packed_dst,
        std::uint8_t* scale_dst)
    {
        const Fp8TileScale s = fp8_tile_scale(instr, full_shape, row_start, rows);
        kernels::TranscodeParams p;
        p.src = fp8_tile.data();
        p.src_scale = s.scale_dev;
        p.src_group_size = s.group_size;
        p.dst_packed = packed_dst;
        p.dst_scale = scale_dst;
        p.rows = rows;
        p.cols = s.local_cols;
        kernels::launch_transcode(
            kernels::TranscodeSource::Fp8E4m3PerGroup,
            kernels::TranscodeTarget::Mxfp4E2m1E8m0, p, /*stream=*/0);
        CUDA_CHECK(cudaGetLastError());
    }
#endif

    DType encode_source_dtype(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.has_source) {
            return dtype_from_rust(
                program_index_.source(instr.source.tensor_id).dtype);
        }
        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Encode expects source or one input");
        }
        return resolver_.or_finalized(instr.input_buffers.ptr[0]).dtype();
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
        // FP8 Encode source needs a [rows/128, cols/128] block scale; tiling
        // the dequant by an arbitrary row count would slice through the 128
        // row block boundary. Disable tiling on FP8 sources — GLM-5.1 expert
        // weights at [2048, 6144] fit comfortably (~50MB BF16 scratch).
        if (source_dtype == DType::FP8_E4M3 ||
            source_dtype == DType::FP8_E5M2) {
            return rows;
        }
        const std::uint64_t max_tile_bytes =
            instr.max_tile_bytes == 0 ? loader_config::kFallbackTileBytes : instr.max_tile_bytes;
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
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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
        case pie_weight_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0: {
            // Output is packed nibbles `[rows, cols/2]` uint8. Scale is
            // E8M0 `[rows, cols/32]` uint8.
            if (out.dtype() != DType::UINT8 && out.dtype() != DType::MXFP4_PACKED) {
                throw std::runtime_error(
                    "rust storage executor: MXFP4 Encode output dtype mismatch");
            }
            if (scale.dtype() != DType::UINT8) {
                throw std::runtime_error(
                    "rust storage executor: MXFP4 Encode scale dtype mismatch");
            }
            if (cols % loader_config::kMxfp4Group != 0) {
                throw std::runtime_error(
                    "rust storage executor: MXFP4 Encode cols must be a "
                    "multiple of 32");
            }
            const std::uint64_t packed_row_bytes =
                static_cast<std::uint64_t>(cols) / loader_config::kMxfp4PackedPerByte;
            const std::uint64_t scale_row_bytes =
                static_cast<std::uint64_t>(cols) / loader_config::kMxfp4Group;
            std::uint8_t* packed_dst =
                static_cast<std::uint8_t*>(out.data()) +
                static_cast<std::uint64_t>(row_start) * packed_row_bytes;
            std::uint8_t* scale_dst =
                static_cast<std::uint8_t*>(scale.data()) +
                static_cast<std::uint64_t>(row_start) * scale_row_bytes;
            kernels::quantize_bf16_to_mxfp4_e2m1_per_block(
                bf16.data(),
                packed_dst,
                scale_dst,
                rows,
                cols,
                /*stream=*/0);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
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

    bool fused_transcode_enabled() const
    {
        return !loader_config::env_truthy("PIE_CUDA_DISABLE_FUSED_TRANSCODE");
    }

    // Fused FP8->MXFP4 for one Encode tile: acquire the FP8 source tile and
    // transcode it straight into the MXFP4 packed/scale outputs at this tile's
    // row offset (same offsets as launch_encode_tile's MXFP4 case).
    void launch_fused_mxfp4_tile(
        const pie_weight_loader::PieLoaderStorageInstrView& instr,
        DeviceTensor& out,
        DeviceTensor& scale,
        const std::vector<std::int64_t>& shape,
        int row_start,
        int rows,
        int cols)
    {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        const std::uint64_t packed_row_bytes =
            static_cast<std::uint64_t>(cols) / loader_config::kMxfp4PackedPerByte;
        const std::uint64_t scale_row_bytes =
            static_cast<std::uint64_t>(cols) / loader_config::kMxfp4Group;
        std::uint8_t* packed_dst =
            static_cast<std::uint8_t*>(out.data()) +
            static_cast<std::uint64_t>(row_start) * packed_row_bytes;
        std::uint8_t* scale_dst =
            static_cast<std::uint8_t*>(scale.data()) +
            static_cast<std::uint64_t>(row_start) * scale_row_bytes;
        DeviceTensor fp8_tile =
            acquire_encode_source_tile(instr, shape, row_start, rows);
        transcode_fp8_tile_to_mxfp4(
            instr, fp8_tile, shape, row_start, rows, packed_dst, scale_dst);
#else
        (void)instr; (void)out; (void)scale; (void)shape;
        (void)row_start; (void)rows; (void)cols;
        throw std::runtime_error(
            "rust storage executor: fused MXFP4 transcode requires CUDA");
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
        DeviceTensor& out = resolver_.tensor(instr.output_buffers.ptr[0]);
        DeviceTensor& scale = resolver_.tensor(instr.output_buffers.ptr[1]);
        // For MXFP4 the output buffer is allocated flat (UINT8 [bytes]); the
        // logical 2-D `[rows, cols]` shape lives on the tensor decl. Recover
        // it from the program index.
        std::vector<std::int64_t> shape = out.shape();
        if (instr.transform_to ==
            pie_weight_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
            const auto& buf = program_index_.buffer(instr.output_buffers.ptr[0]);
            if (buf.has_tensor) {
                const auto& t = program_index_.tensor(buf.tensor_id);
                shape = wl_cpp::i64_slice_to_vector(t.shape);
            }
        }
        if (shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: runtime Encode expects a 2-D weight");
        }
        const int rows = checked_int(shape[0], "Encode rows");
        const int cols = checked_int(shape[1], "Encode cols");
        switch (instr.transform_to) {
        case pie_weight_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0: {
            // MXFP4 scale is `[rows, cols/32]` uint8 (E8M0 byte per block).
            // Scale buffer may also be allocated 1-D flat — fetch the logical
            // shape from the decl for comparison.
            std::vector<std::int64_t> scale_shape = scale.shape();
            const auto& sbuf = program_index_.buffer(instr.output_buffers.ptr[1]);
            if (sbuf.has_tensor) {
                const auto& st = program_index_.tensor(sbuf.tensor_id);
                scale_shape = wl_cpp::i64_slice_to_vector(st.shape);
            }
            const std::vector<std::int64_t> want{shape[0], shape[1] / loader_config::kMxfp4Group};
            if (scale_shape != want) {
                throw std::runtime_error(
                    "rust storage executor: MXFP4 Encode scale must be U8 [rows, cols/32]");
            }
            break;
        }
        default:
            if (scale.dtype() != DType::FP32 ||
                scale.shape() != std::vector<std::int64_t>{shape[0]}) {
                throw std::runtime_error(
                    "rust storage executor: Encode scale output must be FP32 [rows]");
            }
            break;
        }
        stats.runtime_quantized_weights += 1;
        stats.runtime_quant_bytes_after += out.nbytes();
        if (instr.has_source) {
            stats.runtime_quant_bytes_before += instr.source.span_bytes;
        } else if (instr.input_buffers.len == 1) {
            stats.runtime_quant_bytes_before +=
                resolver_.or_finalized(instr.input_buffers.ptr[0]).nbytes();
        }

        // Fuse FP8 -> MXFP4 directly when possible, skipping the BF16 HBM
        // round-trip. Bit-identical to the two-step (kernel parity-tested);
        // opt out with PIE_CUDA_DISABLE_FUSED_TRANSCODE.
        const bool fuse_fp8_mxfp4 =
            fused_transcode_enabled()
            && instr.transform_to ==
                   pie_weight_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0
            && instr.has_source
            && encode_source_dtype(instr) == DType::FP8_E4M3;

        if (can_tile_encode(instr)) {
            const DType source_dtype = encode_source_dtype(instr);
            const int rows_per_tile =
                encode_rows_per_tile(instr, source_dtype, rows, cols);
            for (int row = 0; row < rows; row += rows_per_tile) {
                const int tile_rows = std::min(rows_per_tile, rows - row);
                if (fuse_fp8_mxfp4) {
                    launch_fused_mxfp4_tile(
                        instr, out, scale, shape, row, tile_rows, cols);
                } else {
                    DeviceTensor bf16_tile =
                        materialize_encode_input_bf16_rows(
                            instr, shape, row, tile_rows);
                    launch_encode_tile(
                        instr, bf16_tile, out, scale, row, tile_rows, cols);
                }
            }
            return;
        }

        if (fuse_fp8_mxfp4) {
            launch_fused_mxfp4_tile(instr, out, scale, shape, 0, rows, cols);
        } else {
            DeviceTensor bf16 =
                materialize_encode_input_bf16_rows(instr, shape, 0, rows);
            launch_encode_tile(instr, bf16, out, scale, 0, rows, cols);
        }
    }

#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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

    DeviceTensor materialize_repack_source(
        const pie_weight_loader::PieLoaderStorageInstrView& instr)
    {
        if (instr.has_source) {
            DeviceTensor scratch = DeviceTensor::allocate(
                DType::UINT8,
                {static_cast<std::int64_t>(instr.source.span_bytes)});
            if (!wl_cpp::compact_extent(instr.source.stride)) {
                copy_strided_extent_to_device(
                    loader_, instr,
                    scratch.data(),
                    wl_cpp::extent_shape(instr.source.stride));
            } else {
                copy_engine_.queue(
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
            resolver_.or_finalized(instr.input_buffers.ptr[0]);
        DeviceTensor scratch = DeviceTensor::allocate(
            DType::UINT8,
            {static_cast<std::int64_t>(input.nbytes())});
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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
        DeviceTensor& output = resolver_.tensor(instr.output_buffers.ptr[0]);
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

#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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
            resolver_.or_finalized(instr.input_buffers.ptr[0]);
        DeviceTensor& output = resolver_.tensor(instr.output_buffers.ptr[0]);
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
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
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


    CheckpointSource& loader_;
    WeightCopyEngine& copy_engine_;
    const wl_cpp::StorageProgramIndex& program_index_;
    BufferResolver& resolver_;
    void* fp8_bf16_scratch_ptr_ = nullptr;
    std::size_t fp8_bf16_scratch_bytes_ = 0;
    struct CachedFp8Scale { void* data = nullptr; std::size_t nbytes = 0; };
    std::unordered_map<std::string, CachedFp8Scale> fp8_scale_cache_;
    void* fp8_scale_local_ptr_ = nullptr;
    std::size_t fp8_scale_local_bytes_ = 0;
    void* fp8_source_tile_ptr_ = nullptr;
    std::size_t fp8_source_tile_bytes_ = 0;
    // Pooled tile buffer for non-FP8 (BF16/FP16/FP32) compact encode sources —
    // the symmetric counterpart of fp8_source_tile_ptr_ (see acquire_encode_
    // source_tile). Reused across tiles; freed in free_scratch_noexcept.
    void* bf16_source_tile_ptr_ = nullptr;
    std::size_t bf16_source_tile_bytes_ = 0;
};

}  // namespace pie_cuda_driver
