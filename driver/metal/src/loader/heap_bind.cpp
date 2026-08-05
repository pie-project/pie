// heap_bind.cpp — delta's Metal-side weight staging + per-ordinal arg-table binding.
//
// Executes the runtime-owned LoadPlan into one weights region, allocates
// persistent GDN state + KV pages + IO scalars from heap_layout, then walks beta's
// build_decode_dag and binds each dispatch's WEIGHT / STATE / KV / IO slots BY ORDINAL.
// beta binds the per-dispatch activation/scratch X/Out (his WAR/WAW ping-pong) over the
// SAME ordinal space; alpha's RawMetalContext owns the heap + arg tables.
//
// Lane split (binds delta owns here):
//   * weights  — weight_binds(kind,layer): RO 4-bit/dense/norm tensors (bind::Qmv/Dense/Rms/...)
//   * state    — GdnCore ConvState/RecurrentState/ConvStateOut + a zeroed ConvB (no ckpt bias)
//   * kv       — KvAppend KPages/VPages + Sdpa K/V (the paged cache, per full-attn layer)
//   * io       — the I1 per-token buffers: Embed TokenId, Rope/KvAppend Position, Sdpa SeqLen,
//                QmvLmHead Out=Logits, Argmax Logits/NextToken (I3: logits live in IO)
// beta binds: scratch X/Out activations + const geometry params (setBytes-safe: identical
// every token, so the CB stays byte-identical — only the I1 IO buffer CONTENTS change).

#include "heap_bind.hpp"
#include "heap_bind_metal.hpp"

#include <map>
#include <set>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "heap_layout.hpp"
#include "transcode.hpp"
#include "decode_step.hpp"     // beta: Dispatch{kind,ordinal,layer,grid,tg} + build_decode_dag
#include "mtl4_context.hpp"
#include "pie_loader/checkpoint_source.hpp"
#include "pie_loader/stream_pack.hpp"

namespace pie::metal {

namespace {

SlotHandle slice_slot(const SlotHandle& parent, std::uint64_t offset, std::uint64_t bytes) {
    if (!parent.valid() || offset > parent.size || bytes > parent.size - offset) {
        throw std::runtime_error("LoadPlan buffer exceeds weights region");
    }
    SlotHandle out = parent;
    out.contents_ptr = static_cast<std::uint8_t*>(parent.contents()) + offset;
    out.gpu_address = parent.gpu_address + offset;
    out.offset = parent.offset + offset;
    out.size = static_cast<std::size_t>(bytes);
    return out;
}

SlotHandle alloc_zeroed(
    RawMetalContext& ctx,
    size_t nbytes,
    bool elastic = false,
    size_t initial_commit_bytes = 0) {
    SlotHandle s = elastic
        ? ctx.create_elastic_buffer(nbytes, initial_commit_bytes)
        : ctx.heap_alloc(nbytes);
    if (!s.valid()) throw std::runtime_error("buffer allocation failed");
    const size_t zero_bytes = elastic
        ? std::min(nbytes, initial_commit_bytes)
        : nbytes;
    if (zero_bytes != 0 &&
        !ctx.zero_buffer_range(s, 0, zero_bytes)) {
        throw std::runtime_error("buffer zero failed");
    }
    return s;
}

std::uint64_t extent_bytes(
    const pie_loader::PieLoaderStridedExtentView& extent) {
    return pie_loader::extent_bytes(extent, "metal load executor");
}

/// Walk a strided source extent, calling `band(src_offset, bytes)` on each run
/// of contiguous bytes, in destination order.
///
/// A strided read is how a contract selects part of a source: gpt-oss fuses its
/// gate and up projections into alternating rows of one tensor, and splitting
/// them is a stride over the row axis. The selected bytes are compact once
/// selected -- the destination is the tensor they become -- so the walk only
/// has to enumerate the source's runs, and the ranks of the two extents need
/// not agree.
void walk_source_bands(
    const pie_loader::PieLoaderStridedExtentView& src,
    const std::function<void(std::uint64_t, std::uint64_t)>& band) {
    const std::size_t rank = src.dims.len;
    if (rank == 0) {
        band(0, src.element_bytes);
        return;
    }
    // The longest contiguous suffix moves as one memcpy.
    std::size_t split = rank;
    std::int64_t run = src.element_bytes;
    while (split > 0 && src.dims.ptr[split - 1].src_stride == run) {
        run *= src.dims.ptr[split - 1].count;
        --split;
    }
    const auto bytes = static_cast<std::uint64_t>(run);
    if (split == 0) {
        band(0, bytes);
        return;
    }
    std::vector<std::int64_t> index(split, 0);
    for (;;) {
        std::uint64_t at = 0;
        for (std::size_t i = 0; i < split; ++i) {
            at += static_cast<std::uint64_t>(index[i] * src.dims.ptr[i].src_stride);
        }
        band(at, bytes);
        std::size_t axis = split;
        for (;;) {
            if (axis == 0) return;
            --axis;
            if (++index[axis] < src.dims.ptr[axis].count) break;
            index[axis] = 0;
        }
    }
}

void copy_extent(
    const pie_loader::CheckpointSource& source,
    const pie_loader::PieLoaderSourceExtentView& src,
    const pie_loader::PieLoaderDestExtentView& dst,
    const SlotHandle& target,
    std::uint64_t max_tile_bytes) {
    if (!pie_loader::compact_extent(src.stride)) {
        // A strided source is a selection, not a failure: copy it band by band.
        if (!pie_loader::compact_extent(dst.stride)) {
            throw std::runtime_error(
                "metal storage executor: a strided selection must land compactly");
        }
        auto* out = static_cast<std::uint8_t*>(target.contents());
        std::uint64_t at = dst.offset + dst.stride.base_offset;
        const std::uint64_t limit = at + extent_bytes(dst.stride);
        if (limit > target.size) {
            throw std::runtime_error(
                "metal storage executor: strided destination is out of bounds");
        }
        // Collected rather than copied one at a time: the runs are small and
        // there are millions of them, so what matters is that the mapping is
        // released once for all of them and that the copies can go wide.
        struct Band {
            std::uint64_t from, to, bytes;
        };
        std::vector<Band> bands;
        std::uint64_t footprint = 0;
        walk_source_bands(src.stride, [&](std::uint64_t from, std::uint64_t bytes) {
            if (at + bytes > limit) {
                throw std::runtime_error(
                    "metal storage executor: strided selection overruns its tensor");
            }
            bands.push_back(Band{from, at, bytes});
            footprint = std::max(footprint, from + bytes);
            at += bytes;
        });
        source.with_mapped_span(
            src.file_id, src.file_offset + src.stride.base_offset, footprint,
            [&](const std::uint8_t* base) {
                transcode::parallel_ranges(
                    static_cast<std::int64_t>(bands.size()),
                    [&](std::int64_t begin, std::int64_t end) {
                        for (std::int64_t i = begin; i < end; ++i) {
                            const Band& band = bands[static_cast<std::size_t>(i)];
                            std::memcpy(
                                out + band.to, base + band.from,
                                static_cast<std::size_t>(band.bytes));
                        }
                    });
            });
        (void)max_tile_bytes;
        return;
    }
    if (!pie_loader::compact_extent(dst.stride)) {
        throw std::runtime_error(
            "metal storage executor: non-compact ExtentWrite destination is unsupported");
    }
    const std::uint64_t bytes = extent_bytes(dst.stride);
    if (bytes != src.span_bytes) {
        throw std::runtime_error(
            "metal storage executor: source/destination extent size mismatch");
    }
    const std::uint64_t offset = dst.offset + dst.stride.base_offset;
    if (offset > target.size || bytes > target.size - offset) {
        throw std::runtime_error(
            "metal storage executor: ExtentWrite destination is out of bounds");
    }
    source.copy_storage_bytes(
        src.file_id,
        src.file_offset + src.stride.base_offset,
        bytes,
        static_cast<std::uint8_t*>(target.contents()) + offset,
        max_tile_bytes);
}


/// Where a load's wall clock went, printed when `PIE_METAL_LOAD_TRACE` is set.
///
/// Staging a stock gpt-oss checkpoint runs two transforms over twenty billion
/// weights, and "the load is slow" is not a statement anyone can act on: the
/// question is always whether the time is in the arithmetic, in the file, or in
/// the allocator. This answers it without a profiler.
///
/// One switch, four reports: this breakdown, the `setup:` marks in
/// `forward.cpp`, and the two `load:` lines below. They are read together --
/// `forward.cpp` points at this one by name -- so arming three of them and not
/// the fourth answers the question with its middle term missing.
struct LoadTrace {
    bool on = std::getenv("PIE_METAL_LOAD_TRACE") != nullptr;
    double scratch_alloc_ms = 0, scratch_free_ms = 0;
    double extent_ms = 0, bulk_ms = 0, scale_ms = 0, encode_ms = 0;
    std::uint64_t scale_out_bytes = 0, encode_in_bytes = 0, scratch_bytes = 0;

    struct Span {
        LoadTrace& trace;
        double* into;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~Span() {
            if (!trace.on) return;
            *into += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    };
    Span span(double* into) { return Span{*this, into}; }

    void report() const {
        if (!on) return;
        std::fprintf(
            stderr,
            "[pie-metal] load: bulk %.0f ms  extent %.0f ms  mxfp4-decode %.0f ms (%.1f GB out)"
            "  affine-encode %.0f ms (%.1f GB in)  scratch alloc %.0f ms / free %.0f ms"
            " (%.1f GB)\n",
            bulk_ms, extent_ms, scale_ms, double(scale_out_bytes) / 1e9, encode_ms,
            double(encode_in_bytes) / 1e9, scratch_alloc_ms, scratch_free_ms,
            double(scratch_bytes) / 1e9);
    }
};

#ifndef PIE_METAL_KERNELS_DIR_DEFAULT
#define PIE_METAL_KERNELS_DIR_DEFAULT ""
#endif
constexpr const char* kKernelsDir = PIE_METAL_KERNELS_DIR_DEFAULT;

/// The two load-time transforms, on the GPU.
///
/// Compiled on first use and only then: a checkpoint that needs no transform --
/// one already converted offline -- should not pay for a shader compile, and a
/// device that cannot build these kernels should still load that checkpoint.
/// So a failure here is recorded, not thrown, and the caller falls back to the
/// host loops that define what these kernels have to compute.
class TranscodeGpu {
  public:
    explicit TranscodeGpu(RawMetalContext& ctx) : ctx_(ctx) {}

    ~TranscodeGpu() {
        if (params_.valid()) ctx_.release_standalone_buffer(params_);
        for (int ordinal : {kDequantOrdinal, kEncodeOrdinal}) {
            ctx_.release_argtable_ordinal(ordinal);
        }
    }

    /// Decode `blocks` MXFP4 blocks of `block_size` elements into BF16.
    bool dequant_mxfp4(
        const SlotHandle& payload,
        const SlotHandle& exponents,
        const SlotHandle& out,
        std::uint64_t out_offset,
        std::uint32_t blocks,
        std::uint32_t block_size) {
        if (!ready()) return false;
        const std::uint32_t args[2] = {blocks, block_size};
        write_params(args, sizeof(args));
        ctx_.arg_bind_ordinal(kDequantOrdinal, 0, payload);
        ctx_.arg_bind_ordinal(kDequantOrdinal, 1, exponents);
        ctx_.arg_bind_ordinal(kDequantOrdinal, 2, out, out_offset);
        ctx_.arg_bind_ordinal(kDequantOrdinal, 3, params_);
        return run(dequant_, kDequantOrdinal, blocks);
    }

    /// Quantize `groups` groups of `group_size` elements to affine U4.
    bool encode_affine_u4(
        const SlotHandle& input,
        std::uint32_t input_element_bytes,
        const SlotHandle& codes,
        const SlotHandle& scales,
        const SlotHandle& biases,
        std::uint32_t groups,
        std::uint32_t group_size) {
        if (!ready()) return false;
        const Pso pso = input_element_bytes == 2 ? encode_bf16_ : encode_f32_;
        const std::uint32_t args[2] = {groups, group_size};
        write_params(args, sizeof(args));
        ctx_.arg_bind_ordinal(kEncodeOrdinal, 0, input);
        ctx_.arg_bind_ordinal(kEncodeOrdinal, 1, codes);
        ctx_.arg_bind_ordinal(kEncodeOrdinal, 2, scales);
        ctx_.arg_bind_ordinal(kEncodeOrdinal, 3, biases);
        ctx_.arg_bind_ordinal(kEncodeOrdinal, 4, params_);
        return run(pso, kEncodeOrdinal, groups);
    }

  private:
    // Ordinals the decode DAG does not use. The argument-table space is keyed by
    // the flat dispatch ordinal, which tops out in the hundreds; these sit far
    // above it so a load-time table can never be mistaken for a decode one.
    static constexpr int kDequantOrdinal = 30000;
    static constexpr int kEncodeOrdinal = 30001;

    bool ready() {
        if (built_) return dequant_.valid();
        built_ = true;
        const char* env = std::getenv("PIE_METAL_KERNELS_DIR");
        std::string dir = env != nullptr ? std::string(env) : std::string(kKernelsDir);
        if (dir.empty()) return false;
        if (dir.back() != '/') dir += '/';
        const std::string path = dir + "transcode.metal";
        std::string error;
        dequant_ = ctx_.compile_precise_pso_from_file(path, "mxfp4_dequant_bf16", &error);
        encode_bf16_ = ctx_.compile_precise_pso_from_file(path, "affine_encode_u4_bf16", &error);
        encode_f32_ = ctx_.compile_precise_pso_from_file(path, "affine_encode_u4_f32", &error);
        params_ = ctx_.create_standalone_buffer(256);
        if (!dequant_.valid() || !encode_bf16_.valid() || !encode_f32_.valid() ||
            !params_.valid()) {
            std::fprintf(
                stderr, "[pie-metal] load-time transcode kernels unavailable (%s); "
                        "staging on the host instead\n",
                error.empty() ? "no compiler error reported" : error.c_str());
            dequant_ = Pso{};
            return false;
        }
        // The transforms write into the placement heap, which is only made
        // resident once staging is over. It is idempotent and covers
        // sub-allocations made after it, so asking early costs nothing.
        ctx_.make_resident();
        return true;
    }

    void write_params(const void* args, std::size_t bytes) {
        // Safe to overwrite between dispatches: `run_step` waits for the command
        // buffer it committed before returning.
        std::memcpy(params_.contents(), args, bytes);
    }

    bool run(Pso pso, int ordinal, std::uint32_t threads) {
        if (threads == 0) return true;
        // Every scratch allocation and release edits the residency set, and a
        // set that has been edited has to be requested again before the next
        // command buffer uses it. Asking once per dispatch is the only place
        // that is true by construction.
        ctx_.make_resident();
        // Never wider than the grid: a threadgroup larger than the dispatch is
        // not a launch this encoder performs.
        const std::uint32_t width = std::max(
            1u, std::min({ctx_.pso_max_threads(pso), 256u, threads}));
        const StepTiming timing = ctx_.run_step([&](StepEncoder& encoder) {
            encoder.set_pso(pso);
            encoder.set_argtable_ordinal(ordinal);
            encoder.dispatch(Grid{threads, 1, 1}, Threadgroup{width, 1, 1});
        });
        if (!timing.succeeded()) {
            // This used to read `timing.timed_out`, back when that meant "took
            // longer than five seconds" -- so a transform over a large expert
            // bank on a busy machine was killed for being slow, and one that
            // genuinely never finished was waited on forever instead. It now
            // means the driver gave up, which is the thing worth throwing on.
            throw std::runtime_error(
                "metal storage executor: a load-time transform did not complete" +
                (timing.gpu_error_text.empty() ? std::string()
                                               : ": " + timing.gpu_error_text));
        }
        return true;
    }

    RawMetalContext& ctx_;
    bool built_ = false;
    Pso dequant_, encode_bf16_, encode_f32_;
    SlotHandle params_;
};

/// Execute one load-time transform.
///
/// The Metal heap is Shared storage, so every operand here is a host pointer
/// into the same memory the GPU will read: a transform is a write, not a
/// transfer, and needs no command buffer or fence to become visible.
void run_tile_map(
    const pie_loader::CheckpointSource& source,
    const pie_loader::LoadPlanIndex& index,
    const pie_loader::PieLoaderStorageOp::TileMap_Body& op,
    std::unordered_map<std::uint32_t, SlotHandle>& buffers,
    std::uint64_t max_tile_bytes,
    TranscodeGpu& gpu,
    LoadTrace& trace) {
    namespace tc = transcode;
    const auto slot = [&](std::uint32_t id) -> const SlotHandle& {
        const auto found = buffers.find(id);
        if (found == buffers.end()) {
            throw std::runtime_error(
                "metal storage executor: TileMap operand buffer is missing");
        }
        return found->second;
    };
    const auto shape_of = [&](std::uint32_t id) {
        const auto& decl = index.buffer(id);
        if (!decl.has_tensor) {
            throw std::runtime_error(
                "metal storage executor: TileMap output has no tensor type");
        }
        return pie_loader::i64_slice_to_vector(index.tensor(decl.tensor_id).shape);
    };
    /// The payload's bytes, either straight off disk or already in a buffer.
    std::vector<std::uint8_t> staged;
    const auto payload = [&](std::size_t input_index) -> std::pair<const std::uint8_t*,
                                                                   std::uint64_t> {
        if (op.has_source) {
            if (!pie_loader::compact_extent(op.source.stride)) {
                throw std::runtime_error(
                    "metal storage executor: non-compact TileMap source is unsupported");
            }
            staged.resize(static_cast<std::size_t>(op.source.span_bytes));
            source.copy_storage_bytes(
                op.source.file_id, op.source.file_offset + op.source.stride.base_offset,
                op.source.span_bytes, staged.data(), max_tile_bytes);
            return {staged.data(), op.source.span_bytes};
        }
        if (op.input_buffers.len <= input_index) {
            throw std::runtime_error(
                "metal storage executor: TileMap has neither a source nor an operand");
        }
        const SlotHandle& in = slot(op.input_buffers.ptr[input_index]);
        return {static_cast<const std::uint8_t*>(in.contents()), in.size};
    };

    switch (op.tile_kind) {
    case pie_loader::PieLoaderTileMapKind::Cast: {
        // Raw to raw, and only ever narrowing to BF16 -- the one float format
        // every kernel in this driver reads. `push_direct` emits this whenever
        // a checkpoint ships F16 or F32, which `mlx-community`'s Llama
        // conversions do for every unpacked tensor they have.
        if (op.output_buffers.len != 1) {
            throw std::runtime_error(
                "metal storage executor: a Cast writes exactly one buffer");
        }
        const auto& out_decl = index.buffer(op.output_buffers.ptr[0]);
        if (!out_decl.has_tensor) {
            throw std::runtime_error("metal storage executor: Cast output has no tensor type");
        }
        const auto& out_tensor = index.tensor(out_decl.tensor_id);
        if (out_tensor.encoding_kind != pie_loader::PieLoaderEncodingKind::Raw ||
            out_tensor.dtype != pie_loader::PieLoaderDType::BF16) {
            throw std::runtime_error(
                "metal storage executor: Cast is implemented for raw BF16 outputs only");
        }
        // The input's width comes from whichever side supplied it, so a fused
        // read off disk and a transform of a resident buffer answer the same
        // question in the same units.
        pie_loader::PieLoaderDType from{};
        if (op.has_source) {
            from = index.source(op.source.tensor_id).dtype;
        } else {
            if (op.input_buffers.len == 0) {
                throw std::runtime_error(
                    "metal storage executor: Cast has neither a source nor an operand");
            }
            const auto& in_decl = index.buffer(op.input_buffers.ptr[0]);
            if (!in_decl.has_tensor) {
                throw std::runtime_error(
                    "metal storage executor: Cast operand has no tensor type");
            }
            from = index.tensor(in_decl.tensor_id).dtype;
        }
        std::uint32_t src_bytes = 0;
        switch (from) {
        case pie_loader::PieLoaderDType::F16:
        case pie_loader::PieLoaderDType::BF16: src_bytes = 2; break;
        case pie_loader::PieLoaderDType::F32: src_bytes = 4; break;
        default:
            throw std::runtime_error(
                "metal storage executor: Cast reads F16, BF16 or F32 only");
        }
        const auto [bytes, size] = payload(0);
        if (src_bytes == 0 || size % src_bytes != 0) {
            throw std::runtime_error(
                "metal storage executor: Cast operand is not a whole number of elements");
        }
        const SlotHandle& out = slot(op.output_buffers.ptr[0]);
        const std::uint64_t at = op.has_dest ? op.dest.offset + op.dest.stride.base_offset : 0;
        if (at > out.size) {
            throw std::runtime_error(
                "metal storage executor: Cast destination is out of bounds");
        }
        trace.scale_out_bytes += size;
        const auto _span = trace.span(&trace.scale_ms);
        tc::cast_float_to_bf16(
            bytes, src_bytes, static_cast<std::int64_t>(size / src_bytes),
            tc::Region{static_cast<std::uint8_t*>(out.contents()) + at, out.size - at});
        return;
    }
    case pie_loader::PieLoaderTileMapKind::Scale: {
        if (op.output_buffers.len != 1 || op.transform_scale_blocks.len == 0) {
            throw std::runtime_error(
                "metal storage executor: only per-block Scale is implemented");
        }
        if (op.transform_from != pie_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0) {
            throw std::runtime_error(
                "metal storage executor: per-block Scale is implemented for MXFP4 only");
        }
        // The factor operand is the last input; a fused Scale reads its payload
        // from the file and an unfused one from the input before it.
        const SlotHandle& factors = slot(op.input_buffers.ptr[op.input_buffers.len - 1]);
        const auto [bytes, size] = payload(0);
        const SlotHandle& out = slot(op.output_buffers.ptr[0]);
        const std::uint64_t at = op.has_dest ? op.dest.offset + op.dest.stride.base_offset : 0;
        if (at > out.size) {
            throw std::runtime_error(
                "metal storage executor: Scale destination is out of bounds");
        }
        trace.scale_out_bytes += out.size - at;
        const auto _span = trace.span(&trace.scale_ms);
        const std::uint64_t elements = size * 2;
        if (factors.size == 0 || elements % factors.size != 0) {
            throw std::runtime_error(
                "metal storage executor: MXFP4 blocks do not divide the payload");
        }
        // The GPU reads its operands out of the heap, so it can only run when
        // the payload is already a buffer -- a fused Scale reads the file, and
        // the host has those bytes and the GPU does not.
        if (!op.has_source &&
            gpu.dequant_mxfp4(
                slot(op.input_buffers.ptr[0]), factors, out, at,
                static_cast<std::uint32_t>(factors.size),
                static_cast<std::uint32_t>(elements / factors.size))) {
            return;
        }
        tc::scale_mxfp4_to_bf16(
            bytes, size, static_cast<const std::uint8_t*>(factors.contents()), factors.size,
            tc::Region{static_cast<std::uint8_t*>(out.contents()) + at, out.size - at});
        return;
    }
    case pie_loader::PieLoaderTileMapKind::Encode: {
        if (op.transform_to != pie_loader::PieLoaderQuantScheme::MlxAffineU4) {
            throw std::runtime_error(
                "metal storage executor: Encode is implemented for MLX affine U4 only");
        }
        if (op.output_buffers.len != 3) {
            throw std::runtime_error(
                "metal storage executor: affine Encode wants codes, scales and biases");
        }
        const SlotHandle& codes = slot(op.output_buffers.ptr[0]);
        const SlotHandle& scales = slot(op.output_buffers.ptr[1]);
        const SlotHandle& biases = slot(op.output_buffers.ptr[2]);
        const auto shape = shape_of(op.output_buffers.ptr[0]);
        if (shape.size() != 2) {
            throw std::runtime_error(
                "metal storage executor: affine Encode quantizes a matrix");
        }
        const auto [bytes, size] = payload(0);
        const std::uint64_t elements =
            static_cast<std::uint64_t>(shape[0]) * static_cast<std::uint64_t>(shape[1]);
        if (elements == 0 || size % elements != 0) {
            throw std::runtime_error(
                "metal storage executor: affine Encode operand does not match its shape");
        }
        trace.encode_in_bytes += size;
        const auto _span = trace.span(&trace.encode_ms);
        const auto element_bytes = static_cast<std::uint32_t>(size / elements);
        if (shape[1] % tc::kAffineGroup != 0) {
            throw std::runtime_error(
                "metal storage executor: affine Encode needs a multiple of 64 columns");
        }
        if (!op.has_source &&
            gpu.encode_affine_u4(
                slot(op.input_buffers.ptr[0]), element_bytes, codes, scales, biases,
                static_cast<std::uint32_t>(elements / tc::kAffineGroup),
                static_cast<std::uint32_t>(tc::kAffineGroup))) {
            return;
        }
        tc::encode_mlx_affine_u4(
            bytes, element_bytes, shape[0], shape[1],
            tc::Region{static_cast<std::uint8_t*>(codes.contents()), codes.size},
            tc::Region{static_cast<std::uint8_t*>(scales.contents()), scales.size},
            tc::Region{static_cast<std::uint8_t*>(biases.contents()), biases.size});
        return;
    }
    default:
        throw std::runtime_error(
            "metal storage executor: compiler emitted an unsupported load-time transform");
    }
}

/// One buffer's bytes, located in the checkpoint rather than in the arena.
struct MappedSource {
    std::uint32_t file_id = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t bytes = 0;
    /// The runtime name, carried so the pack can be identified by what is in
    /// it rather than only by which plan produced it.
    std::string name;
};

/// Which buffers the plan builds as a verbatim slice of ONE file range.
///
/// The plan assembles the arena out of `BulkExtentWrite`s: contiguous, verbatim
/// file ranges. A buffer inside one of them is, on disk, already the bytes the
/// GPU wants, so its bytes can be taken from the file rather than rebuilt. A
/// buffer any other op touches is not; this proves the distinction rather than
/// assuming it.
std::unordered_map<std::uint32_t, MappedSource> resolve_mappable(
    const pie_loader::PieLoaderPlan& plan,
    const pie_loader::LoadPlanIndex& index,
    const std::function<bool(const std::string&)>& streams) {
    using Tag = pie_loader::PieLoaderStorageOp::Tag;
    struct Range {
        std::uint64_t dest_begin, dest_end, file_offset;
        std::uint32_t file_id;
    };
    std::vector<Range> ranges;
    std::unordered_map<std::uint32_t, bool> touched_otherwise;
    std::unordered_map<std::uint32_t, std::string> names;
    std::vector<std::uint32_t> allocated;

    for (std::size_t step = 0; step < plan.schedule.len; ++step) {
        const auto& instr = index.instruction(plan.schedule.ptr[step]);
        switch (instr.op.tag) {
        case Tag::Allocate:
            allocated.push_back(instr.op.allocate.buffer_id);
            break;
        case Tag::BulkExtentWrite: {
            const auto& op = instr.op.bulk_extent_write;
            ranges.push_back({op.dest_offset, op.dest_offset + op.source.span_bytes,
                              op.source.file_offset + op.source.stride.base_offset,
                              op.source.file_id});
            break;
        }
        case Tag::Finalize:
            names[instr.op.finalize.buffer_id] =
                pie_loader::bytes_to_string(instr.op.finalize.name);
            break;
        case Tag::ExtentWrite:
            touched_otherwise[instr.op.extent_write.dest.buffer_id] = true;
            break;
        case Tag::Fill:
            touched_otherwise[instr.op.fill.buffer_id] = true;
            break;
        case Tag::CreateView:
            touched_otherwise[instr.op.create_view.output_buffer] = true;
            touched_otherwise[instr.op.create_view.input_buffer] = true;
            break;
        default:
            break;
        }
    }

    std::unordered_map<std::uint32_t, MappedSource> out;
    for (const std::uint32_t id : allocated) {
        if (touched_otherwise.count(id) != 0) continue;
        const auto& decl = index.buffer(id);
        if (!decl.has_persistent_offset || decl.temporary) continue;
        const auto name = names.find(id);
        if (name == names.end() || !streams(name->second)) continue;
        const std::uint64_t begin = decl.persistent_offset;
        const std::uint64_t end = begin + decl.bytes;
        for (const Range& r : ranges) {
            if (begin < r.dest_begin || end > r.dest_end) continue;
            out[id] = MappedSource{r.file_id, r.file_offset + (begin - r.dest_begin),
                                   decl.bytes, name->second};
            break;
        }
    }
    return out;
}


/// Which weights can be read where they already lie.
///
/// A buffer qualifies on two counts. It is a plain span of a file -- no
/// dequantize, no fill, no band copy -- so there is nothing to compute on the
/// way in. And that span begins on `align`, so a device pointer may point at
/// it.
///
/// The second is the one that actually decides, and it is about PLACEMENT
/// rather than format. A stock safetensors checkpoint packs tensors end to end:
/// of gpt-oss's 400 expert tensors, 196 begin on a 4-byte boundary, 36 on 16
/// and none on 32. A `.zt` artifact puts each one on a 64 KiB boundary, which
/// is why converting a checkpoint turns this on -- but nothing here asks what
/// the file is called, and a safetensors checkpoint that happened to be padded
/// would take the same path.
///
/// Answered per BUFFER, not for the checkpoint as a whole. It began as
/// all-or-nothing and that made the cost a cliff rather than a slope:
/// Qwen3.5-0.8B ships eighteen norm tensors as F16, every kernel here reads
/// BF16, and those eighteen casts -- 0.0 MB of a 629 MB checkpoint -- put the
/// other 629 MB back through the copy. The same eighteen tensors would have
/// cost Qwen3.5-35B a twenty-gigabyte copy. Now the heap holds exactly what
/// had to be computed and the file supplies the rest.
///
/// Pure, because the answer is needed TWICE and the two callers must not
/// disagree. The heap is sized before a context exists, and it has to leave
/// these bytes out; the stager then has to leave exactly the same ones out. A
/// heap sized for weights that are then also mapped is the footprint doubled,
/// which on a model that only just fits is the difference between running and
/// an out-of-memory command buffer.
std::unordered_map<std::uint32_t, MappedSource> plan_map_in_place(
    const pie_loader::PieLoaderPlan& plan,
    const pie_loader::LoadPlanIndex& index,
    std::uint64_t align) {
    auto out = resolve_mappable(plan, index, [](const std::string&) { return true; });
    for (auto it = out.begin(); it != out.end();) {
        it = it->second.file_offset % align == 0 ? std::next(it) : out.erase(it);
    }
    return out;
}

}  // namespace

std::uint64_t streamable_plan_bytes(const pie_loader::LoadPlan& load,
                                    const std::function<bool(const std::string&)>& streams,
                                    bool slab_paging) {
    const auto plan = load.view();
    pie_loader::LoadPlanIndex index("metal load executor");
    index.reset(plan);
    // Mapping in place takes EVERY weight out of the heap, so it is asked
    // first and it subsumes streaming: there is no pack to carve out when
    // nothing was copied anywhere to begin with.
    //
    // Paging is the one thing it does NOT subsume. A slab exists to keep the
    // bank off the device entirely, and a mapping cannot do that -- every
    // mapped page is wired -- so `stage_plan_weights` turns mapping off when a
    // budget is set. This must agree with it or the heap is sized for a
    // checkpoint that will not be loaded that way.
    const auto direct = slab_paging
        ? std::unordered_map<std::uint32_t, MappedSource>{}
        : plan_map_in_place(plan, index,
                            std::max<std::uint64_t>(1, load.preferred_alignment()));
    if (!direct.empty()) {
        std::uint64_t bytes = 0;
        for (const auto& [id, src] : direct) bytes += src.bytes;
        return bytes;
    }
    if (!streams) return 0;
    std::uint64_t total = 0;
    for (const auto& [id, src] : resolve_mappable(plan, index, streams)) total += src.bytes;
    return total;
}

namespace {

}  // namespace

// Stage every tensor the plan names into the heap, keyed by its runtime name.
//
// Driven entirely by the LoadPlan -- which the model contract authors -- and so
// by nothing family-specific. Both families' staging calls this rather than
// carrying a copy: the transforms (dequantize, fill, band copies) are where a
// second implementation would quietly diverge, and a checkpoint staged two
// slightly different ways is a model that works for one family and produces
// plausible wrong tokens for the other.
WeightBytes weight_bytes(const std::unordered_map<std::string, SlotHandle>& weights,
                         int n_experts, int experts_per_token) {
    // A mixture with a top-k wider than its bank would be a config defect, and
    // a share above one would silently inflate the bandwidth it claims.
    const double share = (n_experts > 0 && experts_per_token > 0 && experts_per_token < n_experts)
                             ? double(experts_per_token) / double(n_experts)
                             : 1.0;
    WeightBytes out;
    // Counted once per REGION rather than once per name. Expert paging binds
    // every mixture layer's bank to the same slab -- that sharing is the whole
    // point of a budget -- so summing by name reported 48 copies of one slab
    // and claimed a 0.34 GB cache was 16 GB resident. Two names over one
    // region is one region's worth of memory whatever the reason for it.
    std::set<std::pair<const void*, std::uint64_t>> counted;
    for (const auto& [name, slot] : weights) {
        const bool first = counted.emplace(slot.contents(), std::uint64_t(slot.size)).second;
        if (!first) continue;
        out.resident += slot.size;
        // `experts.` is the runtime name every family's routed bank is staged
        // under: llama, qwen3.5 and gpt-oss spell it `mlp.experts.`, gemma4
        // hangs it straight off the layer as `layers.N.experts.`. Matching the
        // longer prefix therefore discounted nothing for gemma4 and reported
        // the 26B reading its whole 13.47 GiB heap every token -- 806 GB/s on
        // a 400 GB/s bus. The shared expert beside the bank is
        // `mlp.shared_expert.`, singular, which this does not match and which
        // is right: every token reads it whole.
        if (name.find("experts.") != std::string::npos) {
            out.routed_resident += slot.size;
            out.per_token += double(slot.size) * share;
        } else if (name.rfind("embed_tokens", 0) == 0) {
            // Gathered, one row a token. Counted at zero rather than at a row
            // because a row is under a millionth of what a layer moves and
            // pretending to that precision would suggest the rest has it.
            //
            // `embed_tokens` is the UNTIED spelling -- `EmbedUntied` binds it
            // and `LmHeadUntied` binds a separate `lm_head`, which is read in
            // full and counted in full. A tied checkpoint stages one tensor
            // called `shared_embedding` that serves both, and it falls to the
            // branch below because the head does read every row of it.
            // `embed_tokens_per_layer` is gemma4's second gathered table and
            // shares the prefix on purpose.
        } else {
            out.per_token += double(slot.size);
        }
    }
    return out;
}

StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes) {
    return stage_plan_weights(ctx, std::move(view), load, weights_bytes, {});
}

StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes,
    const std::function<bool(const std::string&)>& streams) {
    return stage_plan_weights(ctx, std::move(view), load, weights_bytes, streams,
                              ExpertSlabRequest{});
}

StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view_owned,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes,
    const std::function<bool(const std::string&)>& streams,
    const ExpertSlabRequest& slab_req) {
    if (!view_owned) throw std::runtime_error("metal load executor: no checkpoint mapping");
    const pie_loader::CheckpointSource& view = *view_owned;
    StagedWeights b;
    // Backend and tile-map transforms are no longer re-checked: this driver
    // supplied both in the request it compiled from (`architecture.md` §9).
    const auto load_plan = load.view();
    for (std::size_t i = 0; i < load_plan.tensors.len; ++i) {
        const auto& tensor = load_plan.tensors.ptr[i];
        // Every quantized encoding SOME kernel here reads, listed rather than
        // pattern-matched. This is a necessary condition and not a sufficient
        // one: 8-bit g64 is on the list because gpt-oss's router is 8-bit and
        // has its own matvec, which does not mean an 8-bit llama has one. That
        // second question is about which kernel a family names for a width, so
        // it is asked where the width chooses a kernel and not here.
        const auto scheme = tensor.quant_scheme;
        const int bits = int(tensor.quant_bits_per_element);
        const int group = int(tensor.quant_group_size);
        // The affine group is an axis now, not a constant: `quantized_qmv`,
        // `embed_gather` and `quantized_qmm_t` all instantiate at 32, 64 and
        // 128, so the width and the group are two independent choices rather
        // than the one pair g64/b4 every kernel name used to be spelled with.
        //
        // Still necessary and not sufficient, in the same way the width is.
        // The ROUTED matvec takes its group from its codec, which is fixed at
        // 64 because only g64 mixtures are published, so a routed checkpoint
        // at another group passes here and fails where its pipeline is built,
        // by name. That is the same place a width with no instantiation fails.
        const bool affine_group = group == 32 || group == 64 || group == 128;
        const bool readable =
            scheme == pie_loader::PieLoaderQuantScheme::None ||
            (scheme == pie_loader::PieLoaderQuantScheme::MlxAffineU4 && bits == 4 &&
             affine_group) ||
            (scheme == pie_loader::PieLoaderQuantScheme::Int8Asymmetric && bits == 8 &&
             affine_group) ||
            (scheme == pie_loader::PieLoaderQuantScheme::Mxfp4E2M1E8M0 && bits == 4 &&
             group == 32);
        if (!readable) {
            throw std::runtime_error(
                "no metal kernel here reads '" + std::string(reinterpret_cast<const char*>(tensor.name.ptr), tensor.name.len) +
                "': scheme " + std::to_string(int(scheme)) + " at g" + std::to_string(group) +
                "/b" + std::to_string(bits) +
                ". The kernels read MLX affine at b4 or b8 over g32, g64 or g128, "
                "and MXFP4 g32/b4.");
        }
    }
    std::unordered_map<std::uint32_t, SlotHandle> buffers;
    // Transform intermediates, kept apart from `buffers` because they are the
    // only entries this function owns rather than borrows from the heap.
    std::unordered_map<std::uint32_t, SlotHandle> scratch;
    pie_loader::LoadPlanIndex index("metal load executor");
    index.reset(load_plan);

    const auto mapped = streams ? resolve_mappable(load_plan, index, streams)
                                : std::unordered_map<std::uint32_t, MappedSource>{};
    // Rebuilding the arena buffer by buffer is what lets the streamed ones be
    // LEFT OUT of it, and leaving them out is the whole point -- a pack beside
    // a heap that still holds the same bytes doubles the footprint instead of
    // halving it. So every buffer must resolve, not just the streamed ones.
    const std::uint64_t align = std::max<std::uint64_t>(1, load.preferred_alignment());
    // The SAME call `streamable_plan_bytes` made when the heap was sized. If
    // these two ever disagreed the heap would be wrong in one direction or the
    // other, so they ask one function rather than each deciding.
    //
    // Paging the experts turns this OFF, and that is not an optimization being
    // declined -- it is the point. Mapping a file means wrapping it as one
    // no-copy MTLBuffer, and wrapping wires every page of it, expert bank
    // included. A slab beside a wired mapping of the same bytes is the
    // footprint doubled, not bounded. So when a slab is asked for, the dense
    // weights are copied into the heap exactly as they would be from a
    // checkpoint that could not be mapped, and the mmap survives only as the
    // host pointer the slab copies FROM.
    const bool slab_on = slab_req.valid() && !mapped.empty();
    auto direct = slab_on ? std::unordered_map<std::uint32_t, MappedSource>{}
                          : plan_map_in_place(load_plan, index, align);
    // The pack is the OTHER way to keep weights out of the heap, and it is for
    // checkpoints this one cannot help: it re-places misaligned tensors by
    // copying them once. When anything maps, nothing needs re-placing.
    std::unordered_map<std::uint32_t, MappedSource> all_sources =
        resolve_mappable(load_plan, index, [](const std::string&) { return true; });
    bool compact_around_pack = !slab_on && direct.empty() && !mapped.empty() && !all_sources.empty();
    if (compact_around_pack) {
        for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
            const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
            if (instr.op.tag != pie_loader::PieLoaderStorageOp::Tag::Allocate) continue;
            if (all_sources.count(instr.op.allocate.buffer_id) == 0) {
                compact_around_pack = false;
                break;
            }
        }
    }

    // ── Map the checkpoint in place, when the checkpoint lets us. ──
    //
    // Every buffer here is a plain span of a file: that is exactly what
    // `compact` established. On unified memory there is then no reason to copy
    // those bytes anywhere -- `CheckpointSource` has already mmap'd every file
    // the plan names, and a no-copy MTLBuffer over that mapping is a pointer
    // the GPU can read directly.
    //
    // What stops it on a stock safetensors checkpoint is placement, not
    // format: of gpt-oss's 400 expert tensors, 196 begin on a 4-byte boundary
    // and none on 32, so most tensors simply do not start where a device
    // pointer may point. That is why the stream pack below exists at all --
    // it is a re-placement, paid for in one copy.
    //
    // A `.zt` artifact places every tensor on a 64 KiB boundary, so the
    // question answers itself and the copy is pure waste. The gate is the
    // placement, though, and never the file's name: an artifact is just the
    // thing that happens to satisfy it, and a safetensors checkpoint that did
    // would take this path too.
    //
    // One buffer per FILE rather than per tensor: a device allocation carries
    // a residency-set commit, and 1351 of them cost more than the copy would.
    std::unordered_map<std::uint32_t, SlotHandle> file_slots;
    bool map_in_place = !direct.empty();
    const auto map_t0 = std::chrono::steady_clock::now();
    if (map_in_place) {
        std::vector<std::uint32_t> file_ids;
        for (const auto& [id, src] : direct) {
            if (std::find(file_ids.begin(), file_ids.end(), src.file_id) == file_ids.end()) {
                file_ids.push_back(src.file_id);
            }
        }
        for (const std::uint32_t file_id : file_ids) {
            const auto [base, bytes] = view.mapped_file(file_id);
            if (base == nullptr || bytes == 0) { map_in_place = false; break; }
            SlotHandle slot = ctx.wrap_host_memory(const_cast<std::uint8_t*>(base), bytes);
            if (!slot.valid()) { map_in_place = false; break; }
            file_slots[file_id] = slot;
        }
        // Every span must lie inside the mapping it claims. A plan that named a
        // span past the end of its file would otherwise become a GPU read of
        // whatever follows the mapping, which is not a fault and not zero.
        if (map_in_place) {
            for (const auto& [id, src] : direct) {
                const SlotHandle& f = file_slots.at(src.file_id);
                if (src.file_offset <= f.size && src.bytes <= f.size - src.file_offset) continue;
                map_in_place = false;
                break;
            }
        }
        if (map_in_place) {
            b.weight_mapping = std::move(view_owned);
            for (const auto& [id, src] : direct) b.streamed_bytes += src.bytes;
            if (std::getenv("PIE_METAL_LOAD_TRACE") != nullptr) {
                std::fprintf(stderr, "[pie-metal] load: map %.0f ms (%zu files, %zu slices)\n",
                             std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - map_t0).count(),
                             file_slots.size(), direct.size());
            }
        } else {
            file_slots.clear();
        }
    }

    SlotHandle pack_slot;
    std::vector<std::uint32_t> stream_ids;
    std::unordered_map<std::uint32_t, std::uint64_t> pack_offset;
    if (compact_around_pack && !map_in_place) {
        // Buffer-id order, so the same plan lays the pack out the same way on
        // every run and the one built last time is the one found this time.
        for (const auto& [id, src] : mapped) stream_ids.push_back(id);
        std::sort(stream_ids.begin(), stream_ids.end());
        std::vector<pie_loader::StreamPackEntry> entries;
        entries.reserve(stream_ids.size());
        for (const std::uint32_t id : stream_ids) {
            entries.push_back({id, mapped.at(id).name, mapped.at(id).bytes, 0});
        }
        const auto layout = pie_loader::stream_pack_layout(std::move(entries));
        for (const auto& e : layout.entries) pack_offset[e.id] = e.offset;

        std::error_code ec;
        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / "pie-metal-stream";
        std::filesystem::create_directories(root, ec);
        const std::string key = pie_loader::bytes_to_string(load_plan.cache_key);
        auto pack = std::make_shared<pie_loader::StreamPack>(
            pie_loader::StreamPack::open(root.string(), key, layout));
        if (pack->valid()) {
            if (pack->needs_fill()) {
                for (const auto& e : layout.entries) {
                    const auto& src = mapped.at(e.id);
                    view.copy_storage_bytes(
                        src.file_id, src.file_offset, src.bytes,
                        static_cast<std::uint8_t*>(pack->base()) + e.offset,
                        load.max_tile_bytes());
                }
                // A pack that cannot be published is still this run's pack;
                // the next run rebuilds it rather than reading half of one.
                pack->commit();
            }
            pack_slot = ctx.wrap_host_memory(pack->base(), std::size_t(layout.total_bytes));
        }
        if (pack_slot.valid()) {
            b.weight_mapping = pack;  // must outlive every slot pointing into it
            for (const std::uint32_t id : stream_ids) b.streamed_bytes += mapped.at(id).bytes;
        } else {
            compact_around_pack = false;
        }
    }

    // ── Page the routed experts through a bounded slab. ──
    //
    // The banks are not bound; a fixed region is, and the host swaps what is in
    // it between layers. Everything needed to do that is already in `mapped`:
    // each entry is a bank buffer with the file range it occupies, and the
    // arity is the mixture's own. No contract declaration, no plan change and
    // no new loader concept -- the streamed set was already identified by name
    // for the mapping work, and this reuses that answer rather than asking a
    // second question that could disagree with it.
    //
    // Grouped by what the tensor IS and indexed by which layer it belongs to,
    // because a slot has to mean one expert across every routed tensor at once
    // -- see `expert_slab.hpp`. The name is where both facts are written down.
    std::shared_ptr<ExpertSlab> slab;
    std::unordered_map<std::uint32_t, std::size_t> slab_kind_of;  // buffer id -> tensor kind
    std::vector<SlotHandle> slab_regions;
    if (slab_on) {
        std::map<std::string, std::map<int, std::uint32_t>> by_suffix;  // suffix -> layer -> id
        for (const auto& [id, src] : mapped) {
            static const std::string kPrefix = "layers.";
            if (src.name.compare(0, kPrefix.size(), kPrefix) != 0) {
                throw std::runtime_error("metal expert slab: '" + src.name +
                                         "' is streamed but names no layer");
            }
            const std::size_t dot = src.name.find('.', kPrefix.size());
            if (dot == std::string::npos) {
                throw std::runtime_error("metal expert slab: '" + src.name + "' has no suffix");
            }
            const int layer = std::stoi(src.name.substr(kPrefix.size(), dot - kPrefix.size()));
            const std::string suffix = src.name.substr(dot + 1);
            if (!by_suffix[suffix].emplace(layer, id).second) {
                throw std::runtime_error("metal expert slab: two '" + suffix +
                                         "' on layer " + std::to_string(layer));
            }
        }
        const auto experts = std::uint32_t(slab_req.n_experts);
        std::vector<SlabTensor> tensors;
        std::vector<std::vector<std::uint32_t>> ids_of_kind;
        std::uint64_t slot_bytes = 0;
        for (const auto& [suffix, layers] : by_suffix) {
            SlabTensor t;
            t.suffix = suffix;
            std::vector<std::uint32_t> ids;
            for (const auto& [layer, id] : layers) {
                const MappedSource& src = mapped.at(id);
                // A bank that is not a whole number of equal experts is not a
                // bank this can band, and guessing a stride would read as a
                // quality regression rather than a failure.
                if (src.bytes % experts != 0) {
                    throw std::runtime_error(
                        "metal expert slab: '" + src.name + "' is " +
                        std::to_string(src.bytes) + " bytes, which is not " +
                        std::to_string(experts) + " equal experts");
                }
                const std::uint64_t band = src.bytes / experts;
                if (t.band_bytes == 0) t.band_bytes = band;
                if (t.band_bytes != band) {
                    throw std::runtime_error("metal expert slab: '" + src.name +
                                             "' bands differently from its own kind");
                }
                const auto [base, file_bytes] = view.mapped_file(src.file_id);
                if (base == nullptr || src.file_offset + src.bytes > file_bytes) {
                    throw std::runtime_error("metal expert slab: '" + src.name +
                                             "' lies outside its file's mapping");
                }
                t.layer_base.push_back(base + src.file_offset);
                ids.push_back(id);
                (void)layer;
            }
            slot_bytes += t.band_bytes;
            tensors.push_back(std::move(t));
            ids_of_kind.push_back(std::move(ids));
        }
        // Floored, and refused below one: a budget that cannot hold a single
        // expert has not configured a small cache.
        const std::uint64_t want = slab_req.budget_bytes / std::max<std::uint64_t>(1, slot_bytes);
        if (want == 0) {
            throw std::runtime_error(
                "metal expert slab: a budget of " + std::to_string(slab_req.budget_bytes) +
                " bytes cannot hold one expert, which costs " + std::to_string(slot_bytes));
        }
        const std::uint64_t instances = std::uint64_t(tensors.front().layer_base.size()) * experts;
        // Capped at the mixture's arity, and this is a KERNEL constraint rather
        // than a policy: `moe_route_sort` histograms the routing decision into
        // `counts[n_experts]` and drops any id at or past it, so a slot number
        // the slab handed out above the arity would be silently unrouted --
        // an expert contribution zeroed, which reads as a quality regression.
        // It is also the right cap anyway: one layer's worth of experts is all
        // a single layer can want at once.
        const auto slots = std::uint32_t(
            std::min<std::uint64_t>({want, instances, std::uint64_t(experts)}));
        std::vector<std::uint8_t*> bases;
        for (std::size_t k = 0; k < tensors.size(); ++k) {
            SlotHandle region = ctx.heap_alloc(std::size_t(std::uint64_t(slots) * tensors[k].band_bytes),
                                               std::size_t(align));
            if (!region.valid()) {
                throw std::runtime_error("metal expert slab: heap_alloc failed for '" +
                                         tensors[k].suffix + "'");
            }
            bases.push_back(static_cast<std::uint8_t*>(region.contents()));
            for (const std::uint32_t id : ids_of_kind[k]) slab_kind_of[id] = k;
            slab_regions.push_back(region);
        }
        slab = std::make_shared<ExpertSlab>(std::move(tensors), experts, std::move(bases), slots);
        // The mmap is the slab's SOURCE, not something the GPU reads, so it is
        // kept alive without ever being wrapped as a device buffer -- which is
        // exactly the wiring this exists to avoid.
        b.weight_mapping = view_owned;
        b.slab = slab;
        for (const auto& [id, src] : mapped) b.streamed_bytes += src.bytes;
    }

    // What the heap still has to hold.
    //
    // The plan lays every persistent buffer out in one arena at a fixed offset,
    // and both paths that take bytes OUT of that arena -- the pack and the
    // mapping -- leave holes in it. Allocating the arena's full size around
    // those holes would put back exactly the bytes that were just removed, so
    // the survivors are re-laid compactly and the arena is never built.
    //
    // Whatever is left is what genuinely had to be computed: a cast to BF16, a
    // dequantized MXFP4 block, a fill. On a `.zt` this is usually a rounding
    // error against the model -- 0.0 MB for Qwen3.5-0.8B, 201 MB of 1.7 GB for
    // Llama-3.2-3B -- and on a checkpoint that needs no transform at all it is
    // zero and no region is allocated.
    const bool compact = map_in_place || compact_around_pack || slab_on;
    std::unordered_map<std::uint32_t, std::uint64_t> compact_offset;
    std::uint64_t compact_bytes = 0;
    if (compact) {
        std::vector<std::uint32_t> ids;
        for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
            const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
            if (instr.op.tag != pie_loader::PieLoaderStorageOp::Tag::Allocate) continue;
            const auto& decl = index.buffer(instr.op.allocate.buffer_id);
            if (!decl.has_persistent_offset || decl.temporary) continue;
            if (direct.count(decl.id) != 0 || mapped.count(decl.id) != 0) continue;
            ids.push_back(decl.id);
        }
        std::sort(ids.begin(), ids.end());
        for (const std::uint32_t id : ids) {
            compact_offset[id] = compact_bytes;
            compact_bytes += ((index.buffer(id).bytes + align - 1) / align) * align;
        }
    }
    if (!compact || compact_bytes > 0) {
        b.weights_region =
            ctx.heap_alloc(compact ? std::size_t(compact_bytes) : weights_bytes,
                           std::size_t(align));
        if (!b.weights_region.valid()) {
            throw std::runtime_error("heap_alloc failed for program-owned weights region");
        }
    }
    // Said once, here, because here is where it becomes true. It used to be
    // said by each family after staging returned, in four copies of one
    // sentence -- and two of those copies still claimed a pack after the
    // mapping path replaced it, which is what a restatement does when the
    // thing it restates moves.
    if (b.streamed_bytes > 0) {
        std::fprintf(stderr,
                     "[pie-metal] %.2f GB of weights %s, and out of the heap\n",
                     double(b.streamed_bytes) / 1e9,
                     slab_on ? "paged through a bounded slab"
                             : (map_in_place ? "bound where they lie"
                                             : "packed once, then mapped"));
    }
    if (compact) {
        // Each buffer pulls its own bytes, so the bulk writes are skipped below:
        // they address the arena the plan laid out, which no longer exists. Only
        // the survivors that ARE plain file spans are pulled here; the rest are
        // written by the transform that was the reason they survived.
        for (const auto& [id, off] : compact_offset) {
            const auto found = all_sources.find(id);
            if (found == all_sources.end()) continue;
            const auto& src = found->second;
            view.copy_storage_bytes(src.file_id, src.file_offset, src.bytes,
                                    static_cast<std::uint8_t*>(b.weights_region.contents()) + off,
                                    load.max_tile_bytes());
        }
    }
    // What loading this checkpoint still costs, in the only unit that matters
    // on unified memory: bytes the driver has to write. Zero means every weight
    // was bound where it lay, which is what `pie model convert` exists to
    // arrange -- so when it is NOT zero, the ops that kept it from being zero
    // are named, because each one is work that belongs at convert time.
    if (std::getenv("PIE_METAL_LOAD_TRACE") != nullptr) {
        std::fprintf(stderr, "[pie-metal] load: heap %.1f MB of %.1f MB weights\n",
                     double(compact ? compact_bytes : weights_bytes) / 1e6,
                     double(weights_bytes) / 1e6);
        if (compact) {
            std::map<std::string, std::pair<std::size_t, std::uint64_t>> by_op;
            for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
                const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
                std::uint32_t out_id = 0;
                const char* tag = nullptr;
                switch (instr.op.tag) {
                case pie_loader::PieLoaderStorageOp::Tag::TileMap:
                    if (instr.op.tile_map.output_buffers.len == 0) break;
                    out_id = instr.op.tile_map.output_buffers.ptr[0];
                    tag = "TileMap";
                    break;
                case pie_loader::PieLoaderStorageOp::Tag::Fill:
                    out_id = instr.op.fill.buffer_id;
                    tag = "Fill";
                    break;
                case pie_loader::PieLoaderStorageOp::Tag::ExtentWrite:
                    out_id = instr.op.extent_write.dest.buffer_id;
                    tag = "ExtentWrite";
                    break;
                default: break;
                }
                if (tag == nullptr) continue;
                auto& e = by_op[tag];
                e.first += 1;
                e.second += index.buffer(out_id).bytes;
            }
            for (const auto& [op, count] : by_op) {
                std::fprintf(stderr, "[pie-metal] load: %-12s %5zu ops  %8.1f MB\n", op.c_str(),
                             count.first, double(count.second) / 1e6);
            }
        }
    }
    LoadTrace trace;
    TranscodeGpu gpu(ctx);
    // The last step at which each buffer is named, so a scratch allocation can
    // be returned the moment nothing can read it again.
    std::unordered_map<std::uint32_t, std::size_t> last_use;
    std::unordered_map<std::uint32_t, std::uint64_t> pending_scratch;
    // The schedule reuses a handful of intermediate sizes over and over -- one
    // per expert half, per layer -- so a freed buffer is nearly always the
    // right size for the next one. Handing it back instead of asking the
    // device for fresh pages turns the whole staging run's allocation cost
    // into a couple of allocations.
    std::unordered_map<std::uint64_t, std::vector<SlotHandle>> scratch_pool;
    const auto materialize = [&](std::uint32_t id) {
        const auto want = pending_scratch.find(id);
        if (want == pending_scratch.end()) return;
        trace.scratch_bytes += want->second;
        const auto _span = trace.span(&trace.scratch_alloc_ms);
        SlotHandle slot;
        auto& pool = scratch_pool[want->second];
        if (!pool.empty()) {
            slot = pool.back();
            pool.pop_back();
        } else {
            slot = ctx.create_standalone_buffer(want->second);
        }
        if (!slot.valid()) {
            throw std::runtime_error(
                "metal storage executor: transform scratch allocation failed");
        }
        buffers.emplace(id, slot);
        scratch.emplace(id, slot);
        pending_scratch.erase(want);
    };
    for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
        const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
        using Tag = pie_loader::PieLoaderStorageOp::Tag;
        const auto touch = [&](std::uint32_t id) { last_use[id] = step; };
        switch (instr.op.tag) {
        case Tag::Allocate: touch(instr.op.allocate.buffer_id); break;
        case Tag::ExtentWrite: touch(instr.op.extent_write.dest.buffer_id); break;
        case Tag::Fill: touch(instr.op.fill.buffer_id); break;
        case Tag::Finalize: touch(instr.op.finalize.buffer_id); break;
        case Tag::CreateView:
            touch(instr.op.create_view.input_buffer);
            touch(instr.op.create_view.output_buffer);
            break;
        case Tag::TileMap: {
            const auto& op = instr.op.tile_map;
            for (std::size_t i = 0; i < op.input_buffers.len; ++i) touch(op.input_buffers.ptr[i]);
            for (std::size_t i = 0; i < op.output_buffers.len; ++i) touch(op.output_buffers.ptr[i]);
            if (op.has_dest) touch(op.dest.buffer_id);
            break;
        }
        default: break;
        }
    }

    for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
        const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
        using Tag = pie_loader::PieLoaderStorageOp::Tag;
        switch (instr.op.tag) {
        case Tag::Allocate: {
            const auto& decl = index.buffer(instr.op.allocate.buffer_id);
            if (!decl.has_persistent_offset || decl.temporary) {
                // An intermediate a transform produces and consumes. It is not
                // part of the arena the plan laid out — the plan deliberately
                // gave it no offset there — so it comes from its own Shared
                // allocation and is handed back once the last instruction that
                // names it has run. Holding them all would cost more than the
                // model: gpt-oss dequantizes half a gigabyte per expert half.
                // Deferred: the plan declares every intermediate long before the
                // step that fills it, so allocating here would hold ten
                // gigabytes of scratch resident at once — past this device's
                // working set, which silently drops GPU writes. The bytes are
                // taken on first use instead.
                pending_scratch.emplace(decl.id, decl.bytes);
                break;
            }
            if (compact) {
                const auto in_file = direct.find(decl.id);
                if (in_file != direct.end()) {
                    buffers.emplace(decl.id,
                                    slice_slot(file_slots.at(in_file->second.file_id),
                                               in_file->second.file_offset, decl.bytes));
                } else if (slab_on && mapped.count(decl.id) != 0) {
                    // The bank becomes the slab: a SHORTER expert stack, which
                    // the routed matvec cannot tell from the real one because
                    // it addresses experts as `e * rows * row_stride` with `e`
                    // read from a buffer. Writing slot numbers into that
                    // buffer is the whole of the kernel-side change.
                    buffers.emplace(decl.id, slab_regions[slab_kind_of.at(decl.id)]);
                } else if (mapped.count(decl.id) != 0) {
                    buffers.emplace(decl.id,
                                    slice_slot(pack_slot, pack_offset.at(decl.id), decl.bytes));
                } else {
                    buffers.emplace(decl.id, slice_slot(b.weights_region,
                                                        compact_offset.at(decl.id), decl.bytes));
                }
                break;
            }
            buffers.emplace(
                decl.id,
                slice_slot(
                    b.weights_region,
                    decl.persistent_offset,
                    decl.bytes));
            break;
        }
        case Tag::ExtentWrite: {
            const auto& op = instr.op.extent_write;
            materialize(op.dest.buffer_id);
            const auto target = buffers.find(op.dest.buffer_id);
            if (target == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: destination buffer is missing");
            }
            const auto _span = trace.span(&trace.extent_ms);
            copy_extent(
                view,
                op.source,
                op.dest,
                target->second,
                load.max_tile_bytes());
            break;
        }
        case Tag::BulkExtentWrite: {
            // Mapped in place: the bytes are already where they need to be.
            // Compacted: every buffer pulled its own. Either way the arena this
            // write addresses no longer exists.
            if (compact) break;
            const auto _span = trace.span(&trace.bulk_ms);
            const auto& op = instr.op.bulk_extent_write;
            const std::uint64_t offset = op.dest_offset;
            if (offset > b.weights_region.size ||
                op.source.span_bytes > b.weights_region.size - offset) {
                throw std::runtime_error(
                    "metal storage executor: bulk destination is out of bounds");
            }
            view.copy_storage_bytes(
                op.source.file_id,
                op.source.file_offset + op.source.stride.base_offset,
                op.source.span_bytes,
                static_cast<std::uint8_t*>(b.weights_region.contents()) + offset,
                load.max_tile_bytes());
            break;
        }
        case Tag::CreateView: {
            const auto& op = instr.op.create_view;
            materialize(op.input_buffer);
            const auto input = buffers.find(op.input_buffer);
            if (input == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: view input buffer is missing");
            }
            buffers[op.output_buffer] = slice_slot(
                input->second,
                op.view.offset + op.view.stride.base_offset,
                extent_bytes(op.view.stride));
            break;
        }
        case Tag::Finalize: {
            materialize(instr.op.finalize.buffer_id);
            const auto buffer = buffers.find(instr.op.finalize.buffer_id);
            if (buffer == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: finalized buffer is missing");
            }
            const std::string name =
                pie_loader::bytes_to_string(instr.op.finalize.name);
            if (!b.weights.emplace(name, buffer->second).second) {
                throw std::runtime_error(
                    "metal storage executor: duplicate runtime tensor " + name);
            }
            break;
        }
        case Tag::Fill: {
            // The loader emits this when an expression pads: the padded region
            // is memory no source covers, and the compiler prices it as one
            // fill rather than one copy per band. It must precede every write
            // to the same buffer, which `validate-fill-order` guarantees on
            // the Rust side; the writes below are synchronous host stores into
            // Shared storage, so program order is enough to preserve it here.
            materialize(instr.op.fill.buffer_id);
            const auto target = buffers.find(instr.op.fill.buffer_id);
            if (target == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: Fill buffer is missing");
            }
            std::memset(target->second.contents(), 0, target->second.size);
            break;
        }
        case Tag::TileMap: {
            const auto& op = instr.op.tile_map;
            for (std::size_t i = 0; i < op.input_buffers.len; ++i) materialize(op.input_buffers.ptr[i]);
            for (std::size_t i = 0; i < op.output_buffers.len; ++i) materialize(op.output_buffers.ptr[i]);
            if (op.has_dest) materialize(op.dest.buffer_id);
            run_tile_map(view, index, op, buffers, load.max_tile_bytes(), gpu, trace);
            break;
        }
        }
        // Everything the plan can still reach is still allocated; the rest is
        // returned now rather than at the end, which is what keeps the peak at
        // one transform's intermediates instead of every transform's.
        const auto _free_span = trace.span(&trace.scratch_free_ms);
        for (auto it = scratch.begin(); it != scratch.end();) {
            const auto dead = last_use.find(it->first);
            if (dead != last_use.end() && dead->second > step) {
                ++it;
                continue;
            }
            scratch_pool[it->second.size].push_back(it->second);
            buffers.erase(it->first);
            it = scratch.erase(it);
        }
        for (auto it = pending_scratch.begin(); it != pending_scratch.end();) {
            const auto dead = last_use.find(it->first);
            it = (dead != last_use.end() && dead->second > step) ? std::next(it)
                                                                 : pending_scratch.erase(it);
        }
    }
    {
        const auto _free_span = trace.span(&trace.scratch_free_ms);
        for (auto& [bytes, pool] : scratch_pool) {
            for (const SlotHandle& slot : pool) ctx.release_standalone_buffer(slot);
        }
    }
    trace.report();

    return b;
}

BoundDecode stage_decode_storage(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load,
    const DecodeGeometry& g,
    const HeapPlan& heap_plan,
    const std::function<bool(const std::string&)>& streams) {
    BoundDecode b;
    b.plan = heap_plan;
    b.gdn.resize(g.n_layers);
    b.kv.resize(g.n_layers);

    {
        StagedWeights staged =
            stage_plan_weights(ctx, std::move(view), load, heap_plan.weights_bytes, streams);
        b.weights_region = staged.weights_region;
        b.weights = std::move(staged.weights);
        b.weight_mapping = std::move(staged.weight_mapping);
    }

    // ── KV region: k/v pages per full-attn layer (append-only, I4) ──
    const size_t kv_one = heap_plan.kv_per_layer / 2;  // bytes for k (== v)
    for (int L = 0; L < g.n_layers; ++L) {
        if (!g.is_full_attn(L)) continue;
        const size_t initial = std::min(kv_one, size_t{2} << 20);
        b.kv[L].k_pages = alloc_zeroed(ctx, kv_one, true, initial);
        b.kv[L].v_pages = alloc_zeroed(ctx, kv_one, true, initial);
    }

    // ── State region: GDN conv (ping-pong) + recurrent (in-place) + zeroed conv-bias ──
    // S>1: conv/recurrent slabs hold g.max_slots slots packed at the natural per-slot stride
    // (beta's gdn_core_slotted indexes slot*(Kc*CDIM) / slot*(Hv*Vd*Dk)). conv_bias is a
    // shared zeroed slot (slot-independent). At max_slots=1 every alloc is byte-identical.
    const size_t slots = size_t(g.max_slots);
    const size_t conv_state = size_t(g.gdn_conv_dim) * g.gdn_conv_k * 4 * slots;       // f32
    const size_t recur_state = size_t(g.gdn_v_heads) * g.gdn_v_dim * g.gdn_k_dim * 4 * slots;
    const size_t conv_bias = size_t(g.gdn_conv_dim) * 2;                       // bf16, all-zero
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_full_attn(L)) continue;
        const size_t conv_initial =
            std::min(conv_state, size_t(g.gdn_conv_dim) * g.gdn_conv_k * 4);
        const size_t recur_initial =
            std::min(
                recur_state,
                size_t(g.gdn_v_heads) * g.gdn_v_dim * g.gdn_k_dim * 4);
        b.gdn[L].conv_state =
            alloc_zeroed(ctx, conv_state, true, conv_initial);
        b.gdn[L].conv_state_out =
            alloc_zeroed(ctx, conv_state, true, conv_initial);
        b.gdn[L].recurrent_state =
            alloc_zeroed(ctx, recur_state, true, recur_initial);
        b.gdn[L].conv_bias_zero = alloc_zeroed(ctx, conv_bias);  // conv1d has no ckpt bias
    }

    // ── Scratch pool (beta assigns X/Out per dispatch) ──
    for (int i = 0; i < SCRATCH_POOL; ++i)
        b.scratch[i] =
            ctx.create_elastic_buffer(heap_plan.scratch_slot_bytes);

    // ── IO region (I1 per-token scalars + I3 logits) ──
    // M>1: scalar slots widen to u32[max_tokens]; logits stays f32[vocab]. Byte-identical at M=1.
    const size_t tok = 4 * size_t(g.max_tokens);
    b.io[static_cast<int>(IoSlot::TokenId)]   = alloc_zeroed(ctx, tok);
    b.io[static_cast<int>(IoSlot::Position)]  = alloc_zeroed(ctx, tok);
    b.io[static_cast<int>(IoSlot::SeqLen)]    = alloc_zeroed(ctx, tok);
    b.io[static_cast<int>(IoSlot::Logits)] = alloc_zeroed(
        ctx, g.paged_kv_enabled
                 ? size_t(g.vocab) * size_t(std::max(1, g.max_tokens)) * 2u
                 : size_t(g.vocab) * 4u);
    b.io[static_cast<int>(IoSlot::NextToken)] = alloc_zeroed(ctx, tok);

    if (g.paged_kv_enabled) {
        const size_t r = size_t(std::max(1, g.max_requests));
        const size_t n = size_t(std::max(1, g.max_tokens));
        const size_t refs = r * size_t(std::max(1, g.total_pages));
        b.io[static_cast<int>(IoSlot::QoIndptr)]       = alloc_zeroed(ctx, (r + 1) * 4u);
        b.io[static_cast<int>(IoSlot::KvPageIndptr)]   = alloc_zeroed(ctx, (r + 1) * 4u);
        b.io[static_cast<int>(IoSlot::KvPageIndices)]  = alloc_zeroed(ctx, refs * 4u);
        b.io[static_cast<int>(IoSlot::KvLastPageLens)] = alloc_zeroed(ctx, r * 4u);
        b.io[static_cast<int>(IoSlot::RsSlotIds)]      = alloc_zeroed(ctx, r * 4u);
        b.io[static_cast<int>(IoSlot::RsSlotFlags)]    = alloc_zeroed(ctx, r);
        b.io[static_cast<int>(IoSlot::ReqOfToken)]     = alloc_zeroed(ctx, n * 4u);
        b.io[static_cast<int>(IoSlot::SlotOfToken)]    = alloc_zeroed(ctx, n * 4u);
        b.io[static_cast<int>(IoSlot::WPage)]          = alloc_zeroed(ctx, n * 4u);
        b.io[static_cast<int>(IoSlot::WOff)]           = alloc_zeroed(ctx, n * 4u);
        const size_t mask_stride =
            size_t(std::max(1, g.total_pages)) *
            size_t(std::max(1, g.kv_page_size));
        b.io[static_cast<int>(IoSlot::AttnMask)] =
            alloc_zeroed(ctx, n * mask_stride);
        b.io[static_cast<int>(IoSlot::AttnMaskStride)] =
            alloc_zeroed(ctx, sizeof(std::uint32_t));
        b.io[static_cast<int>(IoSlot::AttnMaskEnabled)] =
            alloc_zeroed(ctx, n);
    }

    // device-argmax substrate (inert unless with_argmax): ArgmaxParams const + EosFlag out.
    b.argmax_params = alloc_zeroed(ctx, sizeof(ArgmaxParams));
    b.eos_flag      = alloc_zeroed(ctx, tok);
    {
        auto* p = static_cast<ArgmaxParams*>(b.argmax_params.contents());
        p->vocab = static_cast<uint32_t>(g.vocab);
        p->n_eos = 0;  // executor/resident loop rewrites vocab+eos per generation
    }
    return b;
}

std::string layer_prefix(int layer) {
    return "layers." + std::to_string(layer) + ".";
}

namespace {

void push_quant(std::vector<WeightBind>& out, const std::string& base) {
    out.push_back({0, base + ".weight"});
    out.push_back({1, base + ".scales"});
    out.push_back({2, base + ".biases"});
}

/// The MXFP4 flavour: a `.scales` of block exponents and no zero point. The
/// third slot is left unbound because the format has nothing to put in it, and
/// the kernel that reads these never loads from it.
void push_mxfp4(std::vector<WeightBind>& out, const std::string& base) {
    out.push_back({0, base + ".weight"});
    out.push_back({1, base + ".scales"});
}

}  // namespace

std::vector<WeightBind> weight_binds(
    Kernel kind,
    int layer,
    const DecodeGeometry& g,
    bool gdn_prep) {
    std::vector<WeightBind> weights;
    const auto push_expert = [&](std::vector<WeightBind>& o, const std::string& base) {
        if (g.mxfp4_experts) {
            push_mxfp4(o, base);
        } else {
            push_quant(o, base);
        }
    };
    const std::string prefix = layer >= 0 ? layer_prefix(layer) : std::string();
    switch (kind) {
    case Kernel::EmbedGather:
    case Kernel::QmvLmHead:
        push_quant(weights, "shared_embedding");
        break;
    case Kernel::FinalRms:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Rms::W),
            "final_norm.weight",
        });
        break;
    case Kernel::Rms:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Rms::W),
            prefix + "input_layernorm.weight",
        });
        break;
    case Kernel::FfnRms:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Rms::W),
            prefix + "post_attention_layernorm.weight",
        });
        break;
    case Kernel::QNorm:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Rms::W),
            prefix + "self_attn.q_norm.weight",
        });
        break;
    case Kernel::KNorm:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Rms::W),
            prefix + "self_attn.k_norm.weight",
        });
        break;
    case Kernel::QmvQ: push_quant(weights, prefix + "self_attn.q_proj"); break;
    case Kernel::QmvK: push_quant(weights, prefix + "self_attn.k_proj"); break;
    case Kernel::QmvV: push_quant(weights, prefix + "self_attn.v_proj"); break;
    case Kernel::QmvO: push_quant(weights, prefix + "self_attn.o_proj"); break;
    case Kernel::QmvIn: push_quant(weights, prefix + "linear_attn.in_proj_qkv"); break;
    case Kernel::QmvInZ: push_quant(weights, prefix + "linear_attn.in_proj_z"); break;
    case Kernel::QmvOut: push_quant(weights, prefix + "linear_attn.out_proj"); break;
    case Kernel::GdnInA: push_quant(weights, prefix + "linear_attn.in_proj_a"); break;
    case Kernel::GdnInB: push_quant(weights, prefix + "linear_attn.in_proj_b"); break;
    // ── Gemma 4 ──
    // The norm sandwich: four per layer, so three of them need their own kind.
    case Kernel::G4AttnPostNorm:
        weights.push_back({(std::uint8_t)bind::Rms::W, prefix + "post_attention_layernorm.weight"});
        break;
    case Kernel::G4FfnPreNorm:
        weights.push_back({(std::uint8_t)bind::Rms::W, prefix + "pre_feedforward_layernorm.weight"});
        break;
    case Kernel::G4FfnPostNorm:
        weights.push_back({(std::uint8_t)bind::Rms::W, prefix + "post_feedforward_layernorm.weight"});
        break;
    case Kernel::G4LayerScalar:
        weights.push_back({(std::uint8_t)bind::LayerScalar::Scalar, prefix + "layer_scalar"});
        break;
    case Kernel::G4PleNorm:
        weights.push_back(
            {(std::uint8_t)bind::Rms::W, prefix + "post_per_layer_input_norm.weight"});
        break;
    // The fused norm+residual kinds. `bind::RmsResidual` keeps bind::Rms's
    // prefix, so the norm weight lands at the same slot; the scaled variant also
    // carries the learned gain the separate `LayerScalar` dispatch used to.
    case Kernel::G4AttnPostResidual:
        weights.push_back(
            {(std::uint8_t)bind::RmsResidual::W, prefix + "post_attention_layernorm.weight"});
        break;
    case Kernel::G4FfnPostResidual:
        weights.push_back(
            {(std::uint8_t)bind::RmsResidual::W, prefix + "post_feedforward_layernorm.weight"});
        break;
    // ── Gemma 4's mixture ──
    // The routed branch sits BESIDE the dense FFN rather than replacing it, so
    // the layer carries five norms and both branches' projections. mlx-lm's
    // suffixes are `_1` for the dense branch's closing norm and `_2` for the
    // routed one's pair; they are numbered, not named, so the switch is the
    // only place the mapping is written down.
    case Kernel::G4RouterNorm:
        weights.push_back({(std::uint8_t)bind::Rms::W, prefix + "router.scale"});
        break;
    case Kernel::G4MoeNorm:
        weights.push_back(
            {(std::uint8_t)bind::Rms::W, prefix + "pre_feedforward_layernorm_2.weight"});
        break;
    case Kernel::G4DenseBranchNorm:
        weights.push_back(
            {(std::uint8_t)bind::Rms::W, prefix + "post_feedforward_layernorm_1.weight"});
        break;
    case Kernel::G4MoeBranchNorm:
        weights.push_back(
            {(std::uint8_t)bind::Rms::W, prefix + "post_feedforward_layernorm_2.weight"});
        break;
    case Kernel::G4Router:
        push_quant(weights, prefix + "router.proj");
        break;
    // A WEIGHT on the top-k kernel, which is otherwise weightless: gemma 4
    // rescales the selected softmax by a learned per-expert gain. The slot is
    // the one after `Params`, which is what `router_topk_scaled` declares.
    case Kernel::G4RouterTopK:
        weights.push_back({(std::uint8_t)bind::GoRouterTopK::Params + 1,
                           prefix + "router.per_expert_scale"});
        break;
    // One stacked tensor per projection, indexed by the routing -- `switch_glu`
    // is mlx-lm's name for the stack, not a third module.
    case Kernel::G4ExpertGate:
        push_expert(weights, prefix + "experts.switch_glu.gate_proj");
        break;
    case Kernel::G4ExpertUp:
        push_expert(weights, prefix + "experts.switch_glu.up_proj");
        break;
    case Kernel::G4ExpertDown:
        push_expert(weights, prefix + "experts.switch_glu.down_proj");
        break;
    // Weightless: they move rows and sum them.
    case Kernel::G4MoeSort:
    case Kernel::G4MoeGather:
    case Kernel::G4ExpertGeglu:
    case Kernel::G4ExpertCombine:
    case Kernel::G4BranchAdd:
        break;
    // ── GPT-OSS ──
    // The embedding and the head are separate tensors here, and every
    // projection carries an additive bias at slot 7 alongside its quantized
    // triplet. `.bias` and `.biases` differ by one character and mean nothing
    // alike; the triplet's zero point is the latter.
    // Untied: the embedding table and the head are different tensors. Shared
    // by gpt-oss and by any llama-family checkpoint that does not tie them.
    case Kernel::EmbedUntied:
        push_quant(weights, "embed_tokens");
        break;
    case Kernel::LmHeadUntied:
        push_quant(weights, "lm_head");
        break;
    case Kernel::GoQmvQ:
        push_quant(weights, prefix + "self_attn.q_proj");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "self_attn.q_proj.bias"});
        break;
    case Kernel::GoQmvK:
        push_quant(weights, prefix + "self_attn.k_proj");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "self_attn.k_proj.bias"});
        break;
    case Kernel::GoQmvV:
        push_quant(weights, prefix + "self_attn.v_proj");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "self_attn.v_proj.bias"});
        break;
    case Kernel::GoQmvO:
        push_quant(weights, prefix + "self_attn.o_proj");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "self_attn.o_proj.bias"});
        break;
    case Kernel::GoSdpaSink:
        weights.push_back({(std::uint8_t)bind::SdpaSink::Sinks, prefix + "self_attn.sinks"});
        break;
    case Kernel::GoRouter:
        push_quant(weights, prefix + "mlp.router");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.router.bias"});
        break;
    case Kernel::GoExpertGate:
        push_expert(weights, prefix + "mlp.experts.gate_proj");
        weights.push_back(
            {(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.experts.gate_proj.bias"});
        break;
    case Kernel::GoExpertUp:
        push_expert(weights, prefix + "mlp.experts.up_proj");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.experts.up_proj.bias"});
        break;
    case Kernel::GoExpertDown:
        push_expert(weights, prefix + "mlp.experts.down_proj");
        weights.push_back(
            {(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.experts.down_proj.bias"});
        break;
    // Weightless: the routing decision and the two elementwise stages read
    // activations only.
    case Kernel::GoRouterTopK:
    case Kernel::GoSwiGlu:
    case Kernel::GoExpertCombine:
        break;

    // ── The llama family's routed FFN ──
    // The same tensors gpt-oss's experts use, minus the biases: Qwen's experts
    // carry none, and asking for `mlp.experts.gate_proj.bias` on a checkpoint
    // that has no such tensor is a load failure, not a zero.
    case Kernel::LlRouter:
        push_quant(weights, prefix + "mlp.gate");
        break;
    case Kernel::LlExpertGate:
        push_expert(weights, prefix + "mlp.experts.gate_proj");
        break;
    case Kernel::LlExpertUp:
        push_expert(weights, prefix + "mlp.experts.up_proj");
        break;
    case Kernel::LlExpertDown:
        push_expert(weights, prefix + "mlp.experts.down_proj");
        break;

    // ── The Qwen3.5 mixture's shared expert ──
    // A dense FFN under its own names. These are separate kinds from
    // `QmvGate`/`QmvUp`/`QmvDown` for exactly this switch's sake: a kind is a
    // weight name, and reusing the dense kinds would have made the contract
    // rename `mlp.shared_expert.gate_proj` to `mlp.gate_proj` -- true of the
    // shape, false of the model, and unreadable in any dump.
    case Kernel::LlSharedGate:
        push_quant(weights, prefix + "mlp.shared_expert.gate_proj");
        break;
    case Kernel::LlSharedUp:
        push_quant(weights, prefix + "mlp.shared_expert.up_proj");
        break;
    case Kernel::LlSharedDown:
        push_quant(weights, prefix + "mlp.shared_expert.down_proj");
        break;
    case Kernel::LlSharedGateProj:
        push_quant(weights, prefix + "mlp.shared_expert_gate");
        break;
    // Weightless, like the mixture's own combine: it reads two activations and
    // a gate, all produced this layer.
    case Kernel::LlSharedCombine:
    case Kernel::LlExpertSiluMul:
        break;

    case Kernel::G4PleResidualScaled:
        weights.push_back(
            {(std::uint8_t)bind::RmsResidual::W, prefix + "post_per_layer_input_norm.weight"});
        weights.push_back({(std::uint8_t)bind::RmsResidual::Scalar, prefix + "layer_scalar"});
        break;
    case Kernel::G4PleProjNorm:
        weights.push_back({(std::uint8_t)bind::Rms::W, "per_layer_projection_norm.weight"});
        break;
    // The PLE table is gathered exactly like the token embedding, and the three
    // PLE projections are ordinary quantized matvecs.
    case Kernel::G4PleTokenGather:
        push_quant(weights, "embed_tokens_per_layer");
        break;
    case Kernel::G4PleProjGemv:
        push_quant(weights, "per_layer_model_projection");
        break;
    case Kernel::G4PleGateGemv:
        push_quant(weights, prefix + "per_layer_input_gate");
        break;
    case Kernel::G4PleProjLayerGemv:
        push_quant(weights, prefix + "per_layer_projection");
        break;
    // Weightless: V-norm, both GeGLUs, the softcap, the sliding attention and
    // the PLE residual all read activations only.
    case Kernel::G4VNorm:
    case Kernel::G4Geglu:
    case Kernel::G4Softcap:
    case Kernel::G4SdpaSliding:
    case Kernel::G4PleCombine:
    case Kernel::G4PleGeglu:
    case Kernel::G4PleResidual:
        break;

    case Kernel::GdnPrep:
    case Kernel::GdnPrepSlotted:
        weights.push_back({
            static_cast<std::uint8_t>(bind::GdnPrep::ConvW),
            prefix + "linear_attn.conv1d.weight",
        });
        weights.push_back({
            static_cast<std::uint8_t>(bind::GdnPrep::ALog),
            prefix + "linear_attn.A_log",
        });
        weights.push_back({
            static_cast<std::uint8_t>(bind::GdnPrep::DtBias),
            prefix + "linear_attn.dt_bias",
        });
        break;
    case Kernel::GdnCore:
    case Kernel::GdnCoreSlotted:
        if (gdn_prep || kind == Kernel::GdnCoreSlotted) {
            weights.push_back({
                static_cast<std::uint8_t>(bind::GdnCoreRecurrent::ConvW),
                prefix + "linear_attn.conv1d.weight",
            });
        } else {
            weights.push_back({
                static_cast<std::uint8_t>(bind::GdnCore::ConvW),
                prefix + "linear_attn.conv1d.weight",
            });
            weights.push_back({
                static_cast<std::uint8_t>(bind::GdnCore::ALog),
                prefix + "linear_attn.A_log",
            });
            weights.push_back({
                static_cast<std::uint8_t>(bind::GdnCore::DtBias),
                prefix + "linear_attn.dt_bias",
            });
        }
        break;
    case Kernel::GatedRms:
        weights.push_back({
            static_cast<std::uint8_t>(bind::GatedRms::W),
            prefix + "linear_attn.norm.weight",
        });
        break;
    case Kernel::QmvGate: push_quant(weights, prefix + "mlp.gate_proj"); break;
    case Kernel::QmvUp: push_quant(weights, prefix + "mlp.up_proj"); break;
    case Kernel::QmvDown: push_quant(weights, prefix + "mlp.down_proj"); break;
    default:
        break;
    }
    return weights;
}

namespace {
inline void bind_slot(RawMetalContext& ctx, int ord, uint8_t idx, const SlotHandle& s) {
    ctx.arg_bind_ordinal(ord, idx, s);
}
}  // namespace

// Walk beta's DAG; bind delta's weight/state/KV/IO slots for each dispatch by ordinal.
// (Robust to beta's ordering — reacts to each dispatch's kind+layer, not a fixed sequence.)
void bind_decode_dag(RawMetalContext& ctx, const BoundDecode& b,
                     const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                     bool gdn_prep) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;

        // (a) load-once weights for this dispatch.
        for (const auto& wb : weight_binds(d.kind, L, g, gdn_prep)) {
            auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end())
                throw std::runtime_error("bind: unstaged weight " + wb.tensor);
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }

        // (b) kind-specific state / KV / IO slots delta owns.
        switch (d.kind) {
            case Kernel::EmbedGather:
                bind_slot(ctx, ord, (uint8_t)bind::Embed::TokenId, io(IoSlot::TokenId));
                break;

            case Kernel::GdnPrep: {  // prep-dispatch (PIE_GDN_PREP): q/k path + q/k conv_state writeback
                const auto& s = b.gdn[L];
                bind_slot(ctx, ord, (uint8_t)bind::GdnPrep::ConvState,    s.conv_state);
                bind_slot(ctx, ord, (uint8_t)bind::GdnPrep::ConvStateOut, s.conv_state_out);
                bind_slot(ctx, ord, (uint8_t)bind::GdnPrep::ConvB,        s.conv_bias_zero);
                break;
            }

            case Kernel::GdnCore: {
                const auto& s = b.gdn[L];
                if (gdn_prep) {  // slimmed recurrent: ConvStateOut at 9, no ALog/DtBias/AGate/BGate
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::ConvState,      s.conv_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::RecurrentState, s.recurrent_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::ConvStateOut,   s.conv_state_out);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::ConvB,          s.conv_bias_zero);
                } else {
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::ConvState,    s.conv_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::RecurrentState, s.recurrent_state);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::ConvStateOut, s.conv_state_out);
                    bind_slot(ctx, ord, (uint8_t)bind::GdnCore::ConvB,        s.conv_bias_zero);
                }
                break;
            }

            case Kernel::KvAppend: {
                const auto& kv = b.kv[L];
                bind_slot(ctx, ord, (uint8_t)bind::KvAppend::KPages, kv.k_pages);
                bind_slot(ctx, ord, (uint8_t)bind::KvAppend::VPages, kv.v_pages);
                bind_slot(ctx, ord, (uint8_t)bind::KvAppend::PositionPtr, io(IoSlot::Position));
                break;
            }

            case Kernel::Sdpa: {
                const auto& kv = b.kv[L];
                bind_slot(ctx, ord, (uint8_t)bind::Sdpa::K, kv.k_pages);
                bind_slot(ctx, ord, (uint8_t)bind::Sdpa::V, kv.v_pages);
                bind_slot(ctx, ord, (uint8_t)bind::Sdpa::N, io(IoSlot::SeqLen));
                break;
            }

            case Kernel::Rope:
            case Kernel::RopeK:
                // rope.metal is in-place on buffer 0 (X); position is the IO scalar at
                // buffer 1. scale/base/head_dim are consts (decode_consts). The activation
                // (buffer 0 X) is bound by beta's scratch schedule (in-place).
                bind_slot(ctx, ord, (uint8_t)bind::Rope::Position, io(IoSlot::Position));
                break;

            case Kernel::QmvLmHead:
                // logits ALWAYS produced into the IO region (I3).
                bind_slot(ctx, ord, (uint8_t)bind::Qmv::Out, io(IoSlot::Logits));
                break;

            case Kernel::Argmax:
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::Logits, io(IoSlot::Logits));
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::NextToken, io(IoSlot::NextToken));
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::Params, b.argmax_params);
                bind_slot(ctx, ord, (uint8_t)bind::Argmax::EosFlag, b.eos_flag);
                break;

            default:
                break;  // weight-only or scratch-only (beta) dispatches
        }
    }
}

}  // namespace pie::metal
