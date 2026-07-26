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

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "heap_layout.hpp"
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

void copy_extent(
    const pie_loader::CheckpointSource& source,
    const pie_loader::PieLoaderSourceExtentView& src,
    const pie_loader::PieLoaderDestExtentView& dst,
    const SlotHandle& target,
    std::uint64_t max_tile_bytes) {
    if (!pie_loader::compact_extent(src.stride) ||
        !pie_loader::compact_extent(dst.stride)) {
        throw std::runtime_error(
            "metal storage executor: non-compact ExtentWrite is unsupported");
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


}  // namespace

std::uint64_t streamable_plan_bytes(const pie_loader::LoadPlan& load,
                                    const std::function<bool(const std::string&)>& streams) {
    if (!streams) return 0;
    const auto plan = load.view();
    pie_loader::LoadPlanIndex index("metal load executor");
    index.reset(plan);
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
StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    const pie_loader::CheckpointSource& view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes) {
    return stage_plan_weights(ctx, view, load, weights_bytes, {});
}

StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    const pie_loader::CheckpointSource& view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes,
    const std::function<bool(const std::string&)>& streams) {
    StagedWeights b;
    // Backend and tile-map transforms are no longer re-checked: this driver
    // supplied both in the request it compiled from (`architecture.md` §9).
    const auto load_plan = load.view();
    for (std::size_t i = 0; i < load_plan.tensors.len; ++i) {
        const auto& tensor = load_plan.tensors.ptr[i];
        if (tensor.quant_scheme ==
                pie_loader::PieLoaderQuantScheme::MlxAffineU4 &&
            (tensor.quant_bits_per_element != 4 ||
             tensor.quant_group_size != 64)) {
            throw std::runtime_error(
                "metal qmv kernels require MLX affine-U4 g64/b4; plan requested g" +
                std::to_string(tensor.quant_group_size) + "/b" +
                std::to_string(tensor.quant_bits_per_element));
        }
    }
    std::unordered_map<std::uint32_t, SlotHandle> buffers;
    pie_loader::LoadPlanIndex index("metal load executor");
    index.reset(load_plan);

    const auto mapped = streams ? resolve_mappable(load_plan, index, streams)
                                : std::unordered_map<std::uint32_t, MappedSource>{};
    // Rebuilding the arena buffer by buffer is what lets the streamed ones be
    // LEFT OUT of it, and leaving them out is the whole point -- a pack beside
    // a heap that still holds the same bytes doubles the footprint instead of
    // halving it. So every buffer must resolve, not just the streamed ones.
    std::unordered_map<std::uint32_t, MappedSource> all_sources;
    bool compact = false;
    if (!mapped.empty()) {
        all_sources = resolve_mappable(load_plan, index, [](const std::string&) { return true; });
        compact = true;
        for (std::size_t step = 0; step < load_plan.schedule.len && compact; ++step) {
            const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
            if (instr.op.tag != pie_loader::PieLoaderStorageOp::Tag::Allocate) continue;
            if (all_sources.count(instr.op.allocate.buffer_id) == 0) compact = false;
        }
    }

    SlotHandle pack_slot;
    std::vector<std::uint32_t> stream_ids;
    std::unordered_map<std::uint32_t, std::uint64_t> pack_offset;
    if (compact) {
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

        const char* dir = std::getenv("PIE_METAL_STREAM_DIR");
        std::error_code ec;
        const std::filesystem::path root =
            dir != nullptr ? std::filesystem::path(dir)
                           : std::filesystem::temp_directory_path() / "pie-metal-stream";
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
            b.stream_pack = pack;  // must outlive every slot pointing into it
            for (const std::uint32_t id : stream_ids) b.streamed_bytes += mapped.at(id).bytes;
        } else {
            compact = false;
        }
    }

    const std::uint64_t align = std::max<std::uint64_t>(1, load.preferred_alignment());
    std::unordered_map<std::uint32_t, std::uint64_t> compact_offset;
    std::uint64_t compact_bytes = 0;
    if (compact) {
        std::vector<std::uint32_t> ids;
        for (const auto& [id, src] : all_sources) {
            if (mapped.count(id) == 0) ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end());
        for (const std::uint32_t id : ids) {
            compact_offset[id] = compact_bytes;
            compact_bytes += ((all_sources.at(id).bytes + align - 1) / align) * align;
        }
    }
    b.weights_region = ctx.heap_alloc(compact ? std::size_t(compact_bytes) : weights_bytes,
                                      std::size_t(align));
    if (!b.weights_region.valid()) {
        throw std::runtime_error("heap_alloc failed for program-owned weights region");
    }
    if (compact) {
        // Each buffer pulls its own bytes, so the bulk writes are skipped below:
        // they address the arena the plan laid out, which no longer exists.
        for (const auto& [id, off] : compact_offset) {
            const auto& src = all_sources.at(id);
            view.copy_storage_bytes(src.file_id, src.file_offset, src.bytes,
                                    static_cast<std::uint8_t*>(b.weights_region.contents()) + off,
                                    load.max_tile_bytes());
        }
    }
    for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
        const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
        using Tag = pie_loader::PieLoaderStorageOp::Tag;
        switch (instr.op.tag) {
        case Tag::Allocate: {
            const auto& decl = index.buffer(instr.op.allocate.buffer_id);
            if (!decl.has_persistent_offset || decl.temporary) {
                throw std::runtime_error(
                    "metal storage executor requires arena-resident buffers");
            }
            if (compact) {
                buffers.emplace(decl.id,
                                mapped.count(decl.id) != 0
                                    ? slice_slot(pack_slot, pack_offset.at(decl.id), decl.bytes)
                                    : slice_slot(b.weights_region, compact_offset.at(decl.id),
                                                 decl.bytes));
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
            const auto target = buffers.find(op.dest.buffer_id);
            if (target == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: destination buffer is missing");
            }
            copy_extent(
                view,
                op.source,
                op.dest,
                target->second,
                load.max_tile_bytes());
            break;
        }
        case Tag::BulkExtentWrite: {
            if (compact) break;  // every buffer already pulled its own bytes
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
            const auto target = buffers.find(instr.op.fill.buffer_id);
            if (target == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: Fill buffer is missing");
            }
            std::memset(target->second.contents(), 0, target->second.size);
            break;
        }
        case Tag::TileMap:
            throw std::runtime_error(
                "metal storage executor: compiler emitted an unsupported load-time transform");
        }
    }

    return b;
}

BoundDecode stage_decode_storage(
    RawMetalContext& ctx,
    const pie_loader::CheckpointSource& view,
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
            stage_plan_weights(ctx, view, load, heap_plan.weights_bytes, streams);
        b.weights_region = staged.weights_region;
        b.weights = std::move(staged.weights);
        b.stream_pack = std::move(staged.stream_pack);
        if (staged.streamed_bytes > 0) {
            std::fprintf(stderr,
                         "[pie-metal] %.2f GB of FFN weights streamed from a pack, and out "
                         "of the heap\n",
                         double(staged.streamed_bytes) / 1e9);
        }
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

}  // namespace

std::vector<WeightBind> weight_binds(
    Kernel kind,
    int layer,
    const DecodeGeometry& g,
    bool gdn_prep) {
    (void)g;
    std::vector<WeightBind> weights;
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
    case Kernel::GdnInA:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Dense::W),
            prefix + "linear_attn.in_proj_a.weight",
        });
        break;
    case Kernel::GdnInB:
        weights.push_back({
            static_cast<std::uint8_t>(bind::Dense::W),
            prefix + "linear_attn.in_proj_b.weight",
        });
        break;
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
    // ── GPT-OSS ──
    // The embedding and the head are separate tensors here, and every
    // projection carries an additive bias at slot 7 alongside its quantized
    // triplet. `.bias` and `.biases` differ by one character and mean nothing
    // alike; the triplet's zero point is the latter.
    case Kernel::GoEmbed:
        push_quant(weights, "embed_tokens");
        break;
    case Kernel::GoLmHead:
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
        push_quant(weights, prefix + "mlp.experts.gate_proj");
        weights.push_back(
            {(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.experts.gate_proj.bias"});
        break;
    case Kernel::GoExpertUp:
        push_quant(weights, prefix + "mlp.experts.up_proj");
        weights.push_back({(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.experts.up_proj.bias"});
        break;
    case Kernel::GoExpertDown:
        push_quant(weights, prefix + "mlp.experts.down_proj");
        weights.push_back(
            {(std::uint8_t)bind::GoQmv::Bias, prefix + "mlp.experts.down_proj.bias"});
        break;
    // Weightless: the routing decision and the two elementwise stages read
    // activations only.
    case Kernel::GoRouterTopK:
    case Kernel::GoSwiGlu:
    case Kernel::GoExpertCombine:
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
