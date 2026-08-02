#pragma once

// TranscodeEngine: the quant/transcode TileMap path — Cast, Encode
// (FP8->bf16->FP8/MXFP4, fused or staged), Repack (Marlin) and Reblock.
// Factored out of the storage executor; it consumes source bytes (loader + copy
// engine), resolves input/output buffers (resolver), reads the LoadPlan
// (program index), and owns the FP8 encode scratch buffers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pie_loader/plan.hpp"
#include "loader_config.hpp"
#include "loader_helpers.hpp"
#include "tensor.hpp"
#include "pie_loader/checkpoint_source.hpp"
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
#include "kernels/dequant_fp4.hpp"
#include "kernels/dequant_wna16.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "kernels/quant_bf16_to_mxfp4.hpp"
#include "kernels/transcode.hpp"
#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif
#endif

namespace pie_cuda_driver {

namespace lp = pie_loader;

class TranscodeEngine {
public:
    TranscodeEngine(pie_loader::CheckpointSource& loader,
                    WeightCopyEngine& copy_engine,
                    const pie_loader::LoadPlanIndex& plan_index,
                    BufferResolver& resolver)
        : loader_(loader), copy_engine_(copy_engine), plan_index_(plan_index),
          resolver_(resolver) {}

    ~TranscodeEngine() { free_scratch_noexcept(); }
    TranscodeEngine(const TranscodeEngine&) = delete;
    TranscodeEngine& operator=(const TranscodeEngine&) = delete;

    /// `source` is the extent this instruction reads, *after* the caller has
    /// applied any per-instance rebinding.
    ///
    /// A fused TileMap -- one that reads the file itself rather than a buffer
    /// someone else filled -- is the only way a group instance's index reaches
    /// a transform, so reading `instr.source` here instead would quietly load
    /// instance 0 for every instance. It is worth being explicit about because
    /// nothing else notices: the shapes, the dequantize and the destination are
    /// all index-independent, which is precisely why they were provable, and
    /// the only wrong thing would be the bytes.
    void tile_map(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        LoadExecutionStats& stats)
    {
        switch (instr.tile_kind) {
        case lp::PieLoaderTileMapKind::Cast:
            cast_tile_map(instr, source);
            return;
        case lp::PieLoaderTileMapKind::Reblock:
            reblock_tile_map(instr);
            return;
        case lp::PieLoaderTileMapKind::Encode:
            encode_tile_map(instr, source, stats);
            return;
        case lp::PieLoaderTileMapKind::Repack:
            repack_tile_map(instr, source);
            return;
        case lp::PieLoaderTileMapKind::Scale:
            scale_tile_map(instr, source);
            return;
        case lp::PieLoaderTileMapKind::Decode:
        case lp::PieLoaderTileMapKind::Transcode:
        case lp::PieLoaderTileMapKind::None:
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
        } else if (src.dtype() == DType::E8M0 && dst_dtype == DType::FP32) {
            kernels::launch_cast_e8m0_to_fp32(
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
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source)
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
            if (!pie_loader::compact_extent(source.stride)) {
                throw std::runtime_error(
                    "rust storage executor: non-compact Cast source is not "
                    "implemented");
            }
            DeviceTensor scratch =
                DeviceTensor::allocate(
                    dtype_from_rust(source.dtype),
                    pie_loader::extent_shape(source.stride));
            if (scratch.nbytes() != source.span_bytes) {
                throw std::runtime_error(
                    "rust storage executor: Cast source byte size mismatch");
            }
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
            // Stream-0 H2D: `cast_tensor_to_ptr` launches on stream 0 and reads
            // this scratch immediately. The batched/pinned `queue()` path lands
            // on a private copy stream with no flush before the kernel, so the
            // cast would read an unwritten buffer -- every DeepSeek-V4 block
            // scale decoded to zero, and every quantized GEMM with it.
            copy_engine_.queue_on_stream(
                source.file_id,
                source.file_offset + source.stride.base_offset,
                source.span_bytes,
                scratch.data(),
                /*stream=*/0);
#else
            copy_engine_.queue(
                source.file_id,
                source.file_offset + source.stride.base_offset,
                source.span_bytes,
                scratch.data());
#endif
            cast_tensor_to_ptr(scratch, dst, out.dtype());
            return;
        }

        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Cast TileMap expects source or one input");
        }
        cast_tensor_to_ptr(resolver_.or_finalized(instr.input_buffers.ptr[0]), dst, out.dtype());
    }

    /// `dst = src * factor`, elementwise, in the source's own dtype.
    ///
    /// The loader guarantees the shapes and the dtype match -- `Scale` is
    /// type-preserving, and a contract that also narrows gets a separate `Cast`
    /// instruction -- so there is nothing to negotiate here beyond dispatching
    /// on the dtype.
    static void scale_tensor_to_ptr(
        const DeviceTensor& src,
        void* dst,
        float factor)
    {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        switch (src.dtype()) {
        case DType::BF16:
            kernels::launch_scale_bf16(src.data(), dst, src.numel(), factor, /*stream=*/0);
            return;
        case DType::FP32:
            kernels::launch_scale_fp32(src.data(), dst, src.numel(), factor, /*stream=*/0);
            return;
        case DType::FP16:
            kernels::launch_scale_fp16(src.data(), dst, src.numel(), factor, /*stream=*/0);
            return;
        default:
            throw std::runtime_error(
                "rust storage executor: unsupported TileMap Scale dtype " +
                std::string(dtype_name(src.dtype())));
        }
#else
        (void)src;
        (void)dst;
        (void)factor;
        throw std::runtime_error(
            "rust storage executor: CUDA TileMap Scale compiled without CUDA "
            "headers");
#endif
    }

    void scale_tile_map(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source)
    {
        if (instr.output_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Scale TileMap expects one output");
        }
        DeviceTensor& out = resolver_.tensor(instr.output_buffers.ptr[0]);
        const auto dst_offset =
            instr.has_dest ? instr.dest.offset + instr.dest.stride.base_offset : 0;
        auto* dst = static_cast<std::uint8_t*>(out.data()) + dst_offset;

        if (instr.transform_scale_blocks.len != 0) {
            scale_per_block_tile_map(instr, source, out, dst);
            return;
        }

        float factor = 0.f;
        std::memcpy(&factor, &instr.transform_scale_factor_bits, sizeof(factor));

        if (instr.has_source) {
            if (!pie_loader::compact_extent(source.stride)) {
                throw std::runtime_error(
                    "rust storage executor: non-compact Scale source is not "
                    "implemented");
            }
            DeviceTensor scratch =
                DeviceTensor::allocate(
                    dtype_from_rust(source.dtype),
                    pie_loader::extent_shape(source.stride));
            if (scratch.nbytes() != source.span_bytes) {
                throw std::runtime_error(
                    "rust storage executor: Scale source byte size mismatch");
            }
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
            // Stream-0 H2D for the same reason `cast_tile_map` uses one: the
            // kernel below launches on stream 0 and reads this scratch straight
            // away, while the batched `queue()` path lands on a private copy
            // stream with no flush in between.
            copy_engine_.queue_on_stream(
                source.file_id,
                source.file_offset + source.stride.base_offset,
                source.span_bytes,
                scratch.data(),
                /*stream=*/0);
#else
            copy_engine_.queue(
                source.file_id,
                source.file_offset + source.stride.base_offset,
                source.span_bytes,
                scratch.data());
#endif
            scale_tensor_to_ptr(scratch, dst, factor);
            return;
        }

        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Scale TileMap expects source or one input");
        }
        scale_tensor_to_ptr(resolver_.or_finalized(instr.input_buffers.ptr[0]), dst, factor);
    }

    // One factor per block, read from the operand the contract paired with the
    // payload rather than from a sibling tensor whose name was guessed.
    // Dequantization written this way happens once, in the plan, so the packed
    // original never has to be resident: a weight is a view into the shared
    // arena, and a view cannot be freed.
    //
    // `transform_scale_blocks` gives the block size on every axis, so a
    // DeepSeek-style FP8 checkpoint's 128x128 blocks and MXFP4's row-wise 32
    // are the same statement at different ranks. The loader derived it from the
    // two shapes it had already checked; this reads it back and checks it
    // against the destination rather than trusting it.
    void scale_per_block_tile_map(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        const DeviceTensor& out,
        std::uint8_t* dst)
    {
        if (instr.input_buffers.len < 1) {
            throw std::runtime_error(
                "rust storage executor: per-block Scale has no factor operand");
        }
        const DeviceTensor& factors =
            resolver_.or_finalized(instr.input_buffers.ptr[instr.input_buffers.len - 1]);

        const auto& shape = out.shape();
        if (shape.size() < 2) {
            throw std::runtime_error(
                "rust storage executor: per-block Scale expects a matrix output");
        }
        if (instr.transform_scale_blocks.len != shape.size()) {
            throw std::runtime_error(
                "rust storage executor: per-block Scale states a block size for " +
                std::to_string(instr.transform_scale_blocks.len) +
                " axes, but the output has " + std::to_string(shape.size()));
        }
        // Every axis but the last two contributes whole matrices, and both
        // kernels below take one matrix at a time -- so a block that spans them
        // is a layout neither can index. The last two are the row block and the
        // column block.
        for (std::size_t axis = 0; axis + 2 < shape.size(); ++axis) {
            if (instr.transform_scale_blocks.ptr[axis] != 1) {
                throw std::runtime_error(
                    "rust storage executor: per-block Scale blocks only the two "
                    "innermost axes; axis " + std::to_string(axis) + " is blocked by " +
                    std::to_string(instr.transform_scale_blocks.ptr[axis]));
            }
        }
        const std::int64_t row_block =
            instr.transform_scale_blocks.ptr[shape.size() - 2];
        const std::int64_t col_block =
            instr.transform_scale_blocks.ptr[shape.size() - 1];
        const std::int64_t cols = shape.back();
        const std::int64_t rows = out.numel() / cols;
        if (row_block <= 0 || col_block <= 0 || cols % col_block != 0 ||
            shape[shape.size() - 2] % row_block != 0) {
            throw std::runtime_error(
                "rust storage executor: per-block Scale blocks do not divide the "
                "output");
        }
        if (out.dtype() != DType::BF16) {
            throw std::runtime_error(
                "rust storage executor: this dequant writes BF16, but the "
                "output declares " +
                std::string(dtype_name(out.dtype())));
        }

        // FP8 is a whole element per byte, so its payload is `rows * cols`
        // bytes; the four-bit schemes pack two to a byte. The scheme is what
        // says which, and both nibble schemes are packed low nibble first --
        // so reading one as the other is silent, and the check has to be exact
        // rather than a width test.
        const bool mxfp4 = instr.transform_from == lp::PieLoaderQuantScheme::Mxfp4E2M1E8M0;
        const bool int4b8 = instr.transform_from == lp::PieLoaderQuantScheme::Int4B8;
        const bool fp8 = instr.transform_from == lp::PieLoaderQuantScheme::Fp8E4M3;
        if (!mxfp4 && !int4b8 && !fp8) {
            throw std::runtime_error(
                "rust storage executor: per-block Scale is implemented for "
                "MXFP4, Int4B8 and FP8-E4M3 elements only");
        }
        // The nibble kernels index one factor per 32 columns and read the row
        // block as 1; the FP8 kernel takes both blocks as arguments and is the
        // only one that can index a block spanning rows.
        if (!fp8 && (row_block != 1 || col_block != loader_config::kMxfp4Group)) {
            throw std::runtime_error(
                "rust storage executor: these block scales come in row-wise "
                "groups of 32");
        }
        // MXFP4 pairs E2M1 elements with E8M0 exponents; Int4B8 pairs
        // biased nibbles with plain BF16 factors; FP8 pairs a byte per element
        // with F32 reciprocal scales. No kernel reads another's factor format.
        const DType want_factors =
            mxfp4 ? DType::E8M0 : (int4b8 ? DType::BF16 : DType::FP32);
        if (factors.dtype() != want_factors) {
            throw std::runtime_error(
                "rust storage executor: this scheme's block scales are " +
                std::string(dtype_name(want_factors)) +
                ", but the factor operand declares " +
                std::string(dtype_name(factors.dtype())));
        }

        DeviceTensor scratch = acquire_scale_source(
            instr, source, fp8 ? rows * cols : rows * cols / 2);
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        if (mxfp4) {
            kernels::launch_dequant_mxfp4_to_bf16(
                static_cast<const std::uint8_t*>(scratch.data()),
                static_cast<const std::uint8_t*>(factors.data()),
                dst,
                static_cast<int>(rows),
                static_cast<int>(cols),
                /*stream=*/0);
        } else if (fp8) {
            kernels::launch_dequant_fp8_e4m3_to_bf16_blocked(
                static_cast<const std::uint8_t*>(scratch.data()),
                dst,
                static_cast<const float*>(factors.data()),
                static_cast<int>(rows),
                static_cast<int>(cols),
                static_cast<int>(row_block),
                static_cast<int>(col_block),
                /*stream=*/0);
        } else {
            // The kernel reads the payload as 32-bit words. Eight nibbles to a
            // word in little-endian order is the same byte sequence the packed
            // source already holds, so the reinterpret is a type change and
            // not a repack -- but it does require the row to be a multiple of
            // eight elements, which a group of 32 already guarantees.
            kernels::launch_dequant_wna16_int4b8_to_bf16(
                static_cast<const std::int32_t*>(scratch.data()),
                factors.data(),
                dst,
                static_cast<int>(rows),
                static_cast<int>(cols),
                static_cast<int>(col_block),
                /*stream=*/0);
        }
        // A dequant launch that the driver rejects (a grid dimension past its
        // limit, say) leaves a sticky error behind and writes nothing. Without
        // this check the weights are silently garbage and the failure surfaces
        // in whatever unrelated kernel next calls cudaGetLastError().
        CUDA_CHECK(cudaGetLastError());
#else
        (void)scratch;
        (void)dst;
        (void)row_block;
        throw std::runtime_error(
            "rust storage executor: CUDA TileMap Scale compiled without CUDA "
            "headers");
#endif
    }

    // The packed payload on device, whether it arrives as file bytes or as a
    // buffer an earlier instruction filled.
    DeviceTensor acquire_scale_source(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        std::int64_t want_bytes)
    {
        if (!instr.has_source) {
            const DeviceTensor& input = resolver_.or_finalized(instr.input_buffers.ptr[0]);
            if (static_cast<std::int64_t>(input.nbytes()) != want_bytes) {
                throw std::runtime_error(
                    "rust storage executor: per-group Scale input is the wrong "
                    "size for its output");
            }
            return DeviceTensor::view(
                const_cast<void*>(input.data()),
                DType::UINT8,
                {want_bytes});
        }
        if (!pie_loader::compact_extent(source.stride)) {
            throw std::runtime_error(
                "rust storage executor: non-compact Scale source is not "
                "implemented");
        }
        if (static_cast<std::int64_t>(source.span_bytes) != want_bytes) {
            throw std::runtime_error(
                "rust storage executor: per-group Scale source is the wrong "
                "size for its output");
        }
        DeviceTensor scratch = DeviceTensor::allocate(
            DType::UINT8, {static_cast<std::int64_t>(source.span_bytes)});
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        // Stream-0 H2D for the same reason the uniform path uses one: the
        // kernel that reads this scratch launches on stream 0 straight away,
        // while the batched `queue()` path lands on a private copy stream with
        // no flush in between.
        copy_engine_.queue_on_stream(
            source.file_id,
            source.file_offset + source.stride.base_offset,
            source.span_bytes,
            scratch.data(),
            /*stream=*/0);
#else
        copy_engine_.queue(
            source.file_id,
            source.file_offset + source.stride.base_offset,
            source.span_bytes,
            scratch.data());
#endif
        return scratch;
    }

    // Acquire the Encode source tile on device (FP8 / other source bytes, or a
    // slice of an input buffer) WITHOUT dequantizing. Shared by the BF16
    // materialize path and the fused FP8->MXFP4 transcode path.
    DeviceTensor acquire_encode_source_tile(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows)
    {
        const int cols = static_cast<int>(full_shape[1]);
        const std::vector<std::int64_t> tile_shape{
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
        };
        DeviceTensor tile;
        if (instr.has_source) {
            const DType source_dtype = dtype_from_rust(source.dtype);
            const bool compact = pie_loader::compact_extent(source.stride);
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
                tile = DeviceTensor::view(tile_ptr,                 source_dtype, tile_shape);
            } else
#endif
            {
                tile = DeviceTensor::allocate(source_dtype, tile_shape);
            }
            if (!compact) {
                if (row_start != 0 || rows != full_shape[0]) {
                    throw std::runtime_error(
                        "rust storage executor: tiled Encode for non-compact "
                        "sources is not implemented");
                }
                copy_strided_extent_to_device(
                    loader_, source, tile.data(), tile.nbytes());
            } else {
                const std::uint64_t elem = dtype_bytes(source_dtype);
                const std::uint64_t row_bytes =
                    static_cast<std::uint64_t>(cols) * elem;
                const std::uint64_t off =
                    source.file_offset +
                    source.stride.base_offset +
                    static_cast<std::uint64_t>(row_start) * row_bytes;
                const std::uint64_t span =
                    static_cast<std::uint64_t>(rows) * row_bytes;
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
                // Stream-0 H2D: the follow-up encode/dequant kernel (also stream
                // 0) sees it via implicit ordering — no flush, no event wait.
                copy_engine_.queue_on_stream(
                    source.file_id, off, span, tile.data(),
                    /*stream=*/0);
#else
                copy_engine_.queue(
                    source.file_id, off, span, tile.data());
#endif
            }
        } else {
            if (instr.input_buffers.len != 1) {
                throw std::runtime_error(
                    "rust storage executor: Encode expects tile or one input");
            }
            const DeviceTensor& input =
                resolver_.or_finalized(instr.input_buffers.ptr[0]);
            tile = DeviceTensor::allocate(input.dtype(), tile_shape);
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
            const std::uint64_t row_bytes =
                static_cast<std::uint64_t>(cols) * dtype_bytes(input.dtype());
            CUDA_CHECK(cudaMemcpyAsync(
                tile.data(),
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
        return tile;
    }

    DeviceTensor materialize_encode_input_bf16_rows(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows)
    {
        const std::vector<std::int64_t> tile_shape{
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(full_shape[1]),
        };
        DeviceTensor tile =
            acquire_encode_source_tile(instr, source, full_shape, row_start, rows);
        if (tile.dtype() == DType::BF16) {
            return tile;
        }
        if (tile.dtype() == DType::FP8_E4M3) {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
            // FP8 (E4M3) tile: dequant to BF16 using the per-group block
            // scale that ships alongside the weight. For GLM-5.1 the scale
            // tensor is `<weight>_scale_inv` with shape [rows/128, cols/128]
            // and dtype FP32 (one float per 128x128 block of the weight).
            return dequant_fp8_tile_to_bf16(
                instr, source, tile, full_shape, row_start, rows, tile_shape);
#else
            throw std::runtime_error(
                "rust storage executor: FP8 Encode requires CUDA support");
#endif
        }
        DeviceTensor bf16 = DeviceTensor::allocate(DType::BF16, tile_shape);
        cast_tensor_to_ptr(tile, bf16.data(), DType::BF16);
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
        const lp::PieLoaderSourceTensorView& scale_info,
        std::size_t scale_nbytes,
        const lp::PieLoaderSourceTensorView& storage)
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
    // caches the scale tensor the instruction names, slices the rank-local
    // block for TP shards,
    // and offsets to the tile's first scale row. Shared by the BF16 dequant and
    // the fused FP8->MXFP4 paths so both see identical scale data.
    Fp8TileScale fp8_tile_scale(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
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
        const auto& weight_info = plan_index_.source(source.tensor_id);
        const std::string weight_name =
            pie_loader::bytes_to_string(weight_info.name);
        // Which tensor holds the block scales is the checkpoint's naming
        // convention, and the loader read the tensor table. It says so on the
        // instruction rather than leaving this to rebuild the name and hope.
        if (instr.transform_metadata_source == pie_loader::PIE_LOADER_NO_TENSOR) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode source '" + weight_name +
                "' has no block-scale tensor on its instruction");
        }
        const auto& scale_info =
            plan_index_.source(instr.transform_metadata_source);
        const auto scale_shape =
            pie_loader::i64_slice_to_vector(scale_info.shape);
        if (scale_shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode scale '" +
                pie_loader::bytes_to_string(scale_info.name) +
                "' must be 2-D (block-scaled FP8)");
        }
        // Get the FULL (un-sharded) weight shape from the checkpoint so we
        // can compute the true group_size. The tile's `full_shape` may be
        // TP-sharded and not match the on-disk scale dimensions.
        const auto weight_shape =
            pie_loader::i64_slice_to_vector(weight_info.shape);
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
        const std::uint64_t base_byte = source.stride.base_offset;
        const std::uint64_t rank_row_off_full = base_byte / true_cols;
        const std::uint64_t rank_col_off_full = base_byte % true_cols;

        const std::string scale_name = pie_loader::bytes_to_string(scale_info.name);
        if (dtype_from_rust(scale_info.dtype) != DType::FP32) {
            throw std::runtime_error(
                "rust storage executor: FP8 Encode scale '" + scale_name +
                "' must be FP32");
        }
        // Cache the full FP8 scale per weight: one disk read + one cudaMalloc
        // amortised across every tile of the same Encode instruction.
        const std::size_t scale_nbytes =
            static_cast<std::size_t>(scale_rows) * scale_cols * sizeof(float);
        ensure_fp8_scale_loaded(
            scale_name, scale_info, scale_nbytes, scale_info);
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
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        const DeviceTensor& fp8_tile,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows,
        const std::vector<std::int64_t>& tile_shape)
    {
        const Fp8TileScale s = fp8_tile_scale(instr, source, full_shape, row_start, rows);
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
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
        const DeviceTensor& fp8_tile,
        const std::vector<std::int64_t>& full_shape,
        int row_start,
        int rows,
        std::uint8_t* packed_dst,
        std::uint8_t* scale_dst)
    {
        const Fp8TileScale s = fp8_tile_scale(instr, source, full_shape, row_start, rows);
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

    // How many rows to transform per launch. The loader decided this in
    // `backend::cuda` (loader/architecture.md §8.1) after weighing the source
    // dtype, the extent's stride and the tile budget; 0 means "the whole tensor
    // in one pass", which is both the untileable case and the case where one
    // tile already covers everything. Clamped against `rows` so a malformed plan
    // cannot turn the loop below into a spin.
    static int encode_rows_per_tile(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        int rows)
    {
        if (instr.rows_per_tile == 0) {
            return rows;
        }
        return static_cast<int>(std::min<std::uint64_t>(
            instr.rows_per_tile, static_cast<std::uint64_t>(rows)));
    }

    void launch_encode_tile(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const DeviceTensor& bf16,
        DeviceTensor& out,
        DeviceTensor& scale,
        int row_start,
        int rows,
        int cols)
    {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        switch (instr.transform_to) {
        case lp::PieLoaderQuantScheme::Fp8E4M3:
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
        case lp::PieLoaderQuantScheme::Int8Symmetric:
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
        case lp::PieLoaderQuantScheme::Mxfp4E2M1E8M0: {
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
        (void)source;
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

    // Fused FP8->MXFP4 for one Encode tile: acquire the FP8 source tile and
    // transcode it straight into the MXFP4 packed/scale outputs at this tile's
    // row offset (same offsets as launch_encode_tile's MXFP4 case).
    void launch_fused_mxfp4_tile(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
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
            acquire_encode_source_tile(instr, source, shape, row_start, rows);
        transcode_fp8_tile_to_mxfp4(
            instr, source, fp8_tile, shape, row_start, rows, packed_dst, scale_dst);
#else
        (void)instr;
        (void)source; (void)out; (void)scale; (void)shape;
        (void)row_start; (void)rows; (void)cols;
        throw std::runtime_error(
            "rust storage executor: fused MXFP4 transcode requires CUDA");
#endif
    }

    void encode_tile_map(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source,
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
            lp::PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
            const auto& buf = plan_index_.buffer(instr.output_buffers.ptr[0]);
            if (buf.has_tensor) {
                const auto& t = plan_index_.tensor(buf.tensor_id);
                shape = pie_loader::i64_slice_to_vector(t.shape);
            }
        }
        if (shape.size() != 2) {
            throw std::runtime_error(
                "rust storage executor: runtime Encode expects a 2-D weight");
        }
        const int rows = checked_int(shape[0], "Encode rows");
        const int cols = checked_int(shape[1], "Encode cols");
        switch (instr.transform_to) {
        case lp::PieLoaderQuantScheme::Mxfp4E2M1E8M0: {
            // MXFP4 scale is `[rows, cols/32]` uint8 (E8M0 byte per block).
            // Scale buffer may also be allocated 1-D flat — fetch the logical
            // shape from the decl for comparison.
            std::vector<std::int64_t> scale_shape = scale.shape();
            const auto& sbuf = plan_index_.buffer(instr.output_buffers.ptr[1]);
            if (sbuf.has_tensor) {
                const auto& st = plan_index_.tensor(sbuf.tensor_id);
                scale_shape = pie_loader::i64_slice_to_vector(st.shape);
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

        // Fusing FP8 -> MXFP4 skips the BF16 HBM round-trip and is bit-identical
        // to the two-step path (kernel parity-tested). Whether to do it is the
        // loader's call, not this executor's, so that the plan — and therefore
        // the artifact cache key — records which kernel sequence ran.
        const bool fuse_fp8_mxfp4 =
            instr.transform_fusion == lp::PieLoaderTransformFusion::Fp8ToMxfp4;

        const int rows_per_tile = encode_rows_per_tile(instr, rows);
        for (int row = 0; row < rows; row += rows_per_tile) {
            const int tile_rows = std::min(rows_per_tile, rows - row);
            if (fuse_fp8_mxfp4) {
                launch_fused_mxfp4_tile(
                    instr, source, out, scale, shape, row, tile_rows, cols);
            } else {
                DeviceTensor bf16_tile =
                    materialize_encode_input_bf16_rows(
                        instr, source, shape, row, tile_rows);
                launch_encode_tile(
                    instr, bf16_tile, out, scale, row, tile_rows, cols);
            }
        }
    }

    // Stage the bytes a Repack reads, reusing the block staged for the tile map
    // before it when both read the same extent.
    //
    // Written when GPT-OSS cut its gate and up projections out of a single
    // `gate_up_proj` block that both halves had to name in full, so each block
    // was staged twice. The contract now narrows each half's read to its own
    // rows, so that case no longer arises and the reuse is opportunistic: it
    // costs one comparison and still covers any future pair of repacks that
    // land on identical bytes.
    //
    // Reuse is safe because the staging copy and the repack kernel that reads it
    // both run on stream 0, so the copy has landed before any kernel that sees
    // the block, and because repack kernels only ever read their source.
    const DeviceTensor& materialize_repack_source(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source)
    {
        if (instr.has_source) {
            const bool compact = pie_loader::compact_extent(source.stride);
            const StagedSource staged{
                /*valid=*/compact,
                source.file_id,
                source.file_offset + source.stride.base_offset,
                source.span_bytes};
            if (staged.valid && staged == staged_source_) {
                return repack_source_;
            }
            // Drop the previous block before taking the next so that two are
            // never resident at once.
            repack_source_ = DeviceTensor{};
            staged_source_ = StagedSource{};
            DeviceTensor scratch = DeviceTensor::allocate(
                DType::UINT8,
                {static_cast<std::int64_t>(source.span_bytes)});
            if (!compact) {
                copy_strided_extent_to_device(
                    loader_, source,
                    scratch.data(),
                    scratch.nbytes());
            } else {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
                // Stream-0 H2D for the same reason as the Cast path: the repack
                // kernel that consumes this scratch runs on stream 0 with no
                // intervening flush.
                copy_engine_.queue_on_stream(
                    staged.file_id,
                    staged.file_offset,
                    staged.span_bytes,
                    scratch.data(),
                    /*stream=*/0);
#else
                copy_engine_.queue(
                    staged.file_id,
                    staged.file_offset,
                    staged.span_bytes,
                    scratch.data());
#endif
            }
            repack_source_ = std::move(scratch);
            staged_source_ = staged;
            return repack_source_;
        }
        if (instr.input_buffers.len != 1) {
            throw std::runtime_error(
                "rust storage executor: Repack expects source or one input buffer");
        }
        const DeviceTensor& input =
            resolver_.or_finalized(instr.input_buffers.ptr[0]);
        repack_source_ = DeviceTensor{};
        staged_source_ = StagedSource{};
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
        repack_source_ = std::move(scratch);
        return repack_source_;
    }

    void repack_tile_map(
        const lp::PieLoaderStorageOp::TileMap_Body& instr,
        const lp::PieLoaderSourceExtentView& source)
    {
#if PIE_CUDA_TRANSCODE_ENGINE_HAS_CUDA
        if (instr.output_buffers.len != 1 || !instr.has_dest) {
            throw std::runtime_error(
                "rust storage executor: Repack expects one output and destination extent");
        }
        // The source is dense and holds exactly the rows and columns this
        // repack wants: which ones those are was decided by the contract's
        // `Slice`/`Shard`/`Stride` nodes and resolved by the plan, so a kernel
        // sees a block, never a selection.
        const int batch = static_cast<int>(instr.transform_batch);
        const int source_rows = static_cast<int>(instr.transform_source_rows);
        const int target_rows = static_cast<int>(instr.transform_target_rows);
        const int source_cols = static_cast<int>(instr.transform_source_cols);
        const int target_cols = static_cast<int>(instr.transform_target_cols);
        if (batch <= 0 || source_rows <= 0 || target_rows < source_rows ||
            source_cols <= 0 || target_cols < source_cols) {
            throw std::runtime_error(
                "rust storage executor: Repack has invalid transform dimensions");
        }
        DeviceTensor& output = resolver_.tensor(instr.output_buffers.ptr[0]);
        auto* dst_base = static_cast<std::uint8_t*>(output.data()) +
            instr.dest.offset + instr.dest.stride.base_offset;
        const DeviceTensor& src_tensor = materialize_repack_source(instr, source);
        const auto* src_base =
            static_cast<const std::uint8_t*>(src_tensor.data());

        switch (instr.repack_layout) {
        case lp::PieLoaderRepackLayout::MarlinMxfp4Weight:
            repack_marlin_mxfp4_weight(
                src_base, dst_base, batch, source_rows, target_rows,
                source_cols, target_cols);
            return;
        case lp::PieLoaderRepackLayout::MarlinMxfp4Scale:
            repack_marlin_mxfp4_scale(
                src_base, dst_base, batch, source_rows, target_rows,
                source_cols, target_cols);
            return;
        case lp::PieLoaderRepackLayout::None:
            break;
        }
        throw std::runtime_error(
            "rust storage executor: Repack has no target layout");
#else
        (void)instr;
        (void)source;
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
        int target_rows,
        int source_cols,
        int target_cols)
    {
#if defined(PIE_CUDA_HAS_MARLIN)
        if (source_cols % 8 != 0 || target_cols % 8 != 0) {
            throw std::runtime_error(
                "rust storage executor: MarlinMxfp4Weight Repack requires "
                "source and target K divisible by 8");
        }
        const std::uint64_t source_bytes =
            checked_nibble_bytes(source_rows, source_cols, "MXFP4 source");
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
                source_rows, /*source_row_offset=*/0, target_rows,
                /*valid_rows=*/source_rows, /*source_stride_cols=*/source_cols,
                /*source_col_offset=*/0, source_cols, target_cols,
                kernels::Mxfp4RowSelect::Identity, /*stream=*/0);
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
        (void)target_rows;
        (void)source_cols;
        (void)target_cols;
        throw std::runtime_error(
            "rust storage executor: MarlinMxfp4Weight Repack requires Marlin");
#endif
    }

    void repack_marlin_mxfp4_scale(
        const std::uint8_t* src_base,
        std::uint8_t* dst_base,
        int batch,
        int source_rows,
        int target_rows,
        int source_groups,
        int target_groups)
    {
        const std::uint64_t source_bytes =
            checked_mul_u64(source_rows, source_groups, "MXFP4 scale source");
        const std::uint64_t target_bytes =
            checked_mul_u64(target_rows, target_groups, "MXFP4 scale target");
        for (int b = 0; b < batch; ++b) {
            kernels::launch_mxfp4_scales_to_marlin_e8m0(
                src_base + static_cast<std::uint64_t>(b) * source_bytes,
                dst_base + static_cast<std::uint64_t>(b) * target_bytes,
                source_rows, /*source_row_offset=*/0, target_rows,
                /*valid_rows=*/source_rows, /*source_stride_groups=*/source_groups,
                /*source_group_offset=*/0, source_groups, target_groups,
                kernels::Mxfp4RowSelect::Identity, /*stream=*/0);
        }
        CUDA_CHECK(cudaGetLastError());
    }
#endif

    void reblock_tile_map(
        const lp::PieLoaderStorageOp::TileMap_Body& instr)
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
            ? pie_loader::extent_bytes(
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


    // The checkpoint extent currently held in `repack_source_`. A strided read
    // is never cached, so `valid` also says "this holds a whole block".
    struct StagedSource {
        bool valid = false;
        std::uint32_t file_id = 0;
        std::uint64_t file_offset = 0;
        std::uint64_t span_bytes = 0;

        bool operator==(const StagedSource& other) const = default;
    };

    pie_loader::CheckpointSource& loader_;
    WeightCopyEngine& copy_engine_;
    const pie_loader::LoadPlanIndex& plan_index_;
    BufferResolver& resolver_;
    void* fp8_bf16_scratch_ptr_ = nullptr;
    std::size_t fp8_bf16_scratch_bytes_ = 0;
    DeviceTensor repack_source_;
    StagedSource staged_source_;
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
