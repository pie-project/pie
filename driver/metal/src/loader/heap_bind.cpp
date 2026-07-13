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
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "heap_layout.hpp"
#include "decode_step.hpp"     // beta: Dispatch{kind,ordinal,layer,grid,tg} + build_decode_dag
#include "mtl4_context.hpp"
#include "safetensors_view.hpp"

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

SlotHandle alloc_zeroed(RawMetalContext& ctx, size_t nbytes) {
    SlotHandle s = ctx.heap_alloc(nbytes);
    if (!s.valid()) throw std::runtime_error("heap_alloc failed");
    std::memset(s.contents(), 0, nbytes);
    return s;
}

std::uint64_t extent_bytes(
    const pie_load_planner::PieLoaderStridedExtentView& extent) {
    return pie_load_planner::cpp::extent_bytes(extent, "metal load executor");
}

void copy_extent(
    const SafetensorsView& source,
    const pie_load_planner::PieLoaderSourceExtentView& src,
    const pie_load_planner::PieLoaderDestExtentView& dst,
    const SlotHandle& target,
    std::uint64_t max_tile_bytes) {
    if (!pie_load_planner::cpp::compact_extent(src.stride) ||
        !pie_load_planner::cpp::compact_extent(dst.stride)) {
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

}  // namespace

BoundDecode stage_decode_storage(
    RawMetalContext& ctx,
    const SafetensorsView& view,
    const pie_load_planner::LoadPlan& load,
    const DecodeGeometry& g,
    const HeapPlan& heap_plan) {
    BoundDecode b;
    b.plan = heap_plan;
    b.gdn.resize(g.n_layers);
    b.kv.resize(g.n_layers);

    const auto load_plan = load.view();
    if (load.backend() != pie_load_planner::PieLoaderBackendKind::Metal) {
        throw std::runtime_error("Metal load executor received a non-Metal plan");
    }
    if (load.tile_map_mask() != pie_load_planner::kMetalTileMapMask) {
        throw std::runtime_error(
            "Metal load plan advertises unsupported TileMap transforms");
    }
    for (std::size_t i = 0; i < load_plan.tensors.len; ++i) {
        const auto& tensor = load_plan.tensors.ptr[i];
        if (tensor.quant_scheme ==
                pie_load_planner::PieLoaderQuantScheme::MlxAffineU4 &&
            (tensor.quant_bits_per_element != 4 ||
             tensor.quant_group_size != 64)) {
            throw std::runtime_error(
                "metal qmv kernels require MLX affine-U4 g64/b4; plan requested g" +
                std::to_string(tensor.quant_group_size) + "/b" +
                std::to_string(tensor.quant_bits_per_element));
        }
    }
    b.weights_region = ctx.heap_alloc(
        heap_plan.weights_bytes,
        std::max<std::size_t>(1, load.preferred_alignment()));
    if (!b.weights_region.valid()) {
        throw std::runtime_error("heap_alloc failed for program-owned weights region");
    }
    std::unordered_map<std::uint32_t, SlotHandle> buffers;
    pie_load_planner::cpp::LoadPlanIndex index("metal load executor");
    index.reset(load_plan);
    for (std::size_t step = 0; step < load_plan.schedule.len; ++step) {
        const auto& instr = index.instruction(load_plan.schedule.ptr[step]);
        using K = pie_load_planner::PieLoaderStorageInstrKind;
        switch (instr.kind) {
        case K::Allocate: {
            const auto& decl = index.buffer(instr.buffer_id);
            if (!decl.has_persistent_offset || decl.temporary) {
                throw std::runtime_error(
                    "metal storage executor requires arena-resident buffers");
            }
            buffers.emplace(
                decl.id,
                slice_slot(
                    b.weights_region,
                    decl.persistent_offset,
                    decl.bytes));
            break;
        }
        case K::ExtentWrite: {
            if (!instr.has_source || !instr.has_dest) {
                throw std::runtime_error(
                    "metal storage executor: ExtentWrite missing source/dest");
            }
            const auto target = buffers.find(instr.dest.buffer_id);
            if (target == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: destination buffer is missing");
            }
            copy_extent(
                view,
                instr.source,
                instr.dest,
                target->second,
                load.max_tile_bytes());
            break;
        }
        case K::BulkExtentWrite: {
            if (!instr.has_source || !instr.has_dest) {
                throw std::runtime_error(
                    "metal storage executor: BulkExtentWrite missing source/dest");
            }
            const std::uint64_t offset =
                instr.dest.offset + instr.dest.stride.base_offset;
            if (offset > b.weights_region.size ||
                instr.source.span_bytes > b.weights_region.size - offset) {
                throw std::runtime_error(
                    "metal storage executor: bulk destination is out of bounds");
            }
            view.copy_storage_bytes(
                instr.source.file_id,
                instr.source.file_offset + instr.source.stride.base_offset,
                instr.source.span_bytes,
                static_cast<std::uint8_t*>(b.weights_region.contents()) + offset,
                load.max_tile_bytes());
            break;
        }
        case K::SlabScatter:
            for (std::size_t i = 0; i < instr.slab_placements.len; ++i) {
                const auto& placement = instr.slab_placements.ptr[i];
                if (placement.dest_offset > b.weights_region.size ||
                    placement.bytes >
                        b.weights_region.size - placement.dest_offset) {
                    throw std::runtime_error(
                        "metal storage executor: slab destination is out of bounds");
                }
                view.copy_storage_bytes(
                    instr.slab_file_id,
                    instr.slab_file_offset + placement.src_offset,
                    placement.bytes,
                    static_cast<std::uint8_t*>(b.weights_region.contents()) +
                        placement.dest_offset,
                    load.max_tile_bytes());
            }
            break;
        case K::CreateView: {
            if (instr.input_buffers.len != 1 ||
                instr.output_buffers.len != 1 ||
                !instr.has_dest) {
                throw std::runtime_error(
                    "metal storage executor: malformed CreateView");
            }
            const auto input = buffers.find(instr.input_buffers.ptr[0]);
            if (input == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: view input buffer is missing");
            }
            buffers[instr.output_buffers.ptr[0]] = slice_slot(
                input->second,
                instr.dest.offset + instr.dest.stride.base_offset,
                extent_bytes(instr.dest.stride));
            break;
        }
        case K::Finalize: {
            const auto buffer = buffers.find(instr.buffer_id);
            if (buffer == buffers.end()) {
                throw std::runtime_error(
                    "metal storage executor: finalized buffer is missing");
            }
            const std::string name =
                pie_load_planner::cpp::bytes_to_string(instr.name);
            if (!b.weights.emplace(name, buffer->second).second) {
                throw std::runtime_error(
                    "metal storage executor: duplicate runtime tensor " + name);
            }
            break;
        }
        case K::Attach:
            break;
        case K::Release:
            buffers.erase(instr.buffer_id);
            break;
        case K::TileMap:
            throw std::runtime_error(
                "metal storage executor: compiler emitted an unsupported load-time transform");
        }
    }

    // ── KV region: k/v pages per full-attn layer (append-only, I4) ──
    const size_t kv_one = heap_plan.kv_per_layer / 2;  // bytes for k (== v)
    for (int L = 0; L < g.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        b.kv[L].k_pages = alloc_zeroed(ctx, kv_one);
        b.kv[L].v_pages = alloc_zeroed(ctx, kv_one);
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
        if (DecodeGeometry::is_full_attn(L)) continue;
        b.gdn[L].conv_state     = alloc_zeroed(ctx, conv_state);
        b.gdn[L].conv_state_out = alloc_zeroed(ctx, conv_state);
        b.gdn[L].recurrent_state = alloc_zeroed(ctx, recur_state);
        b.gdn[L].conv_bias_zero = alloc_zeroed(ctx, conv_bias);  // conv1d has no ckpt bias
    }

    // ── Scratch pool (beta assigns X/Out per dispatch) ──
    for (int i = 0; i < SCRATCH_POOL; ++i)
        b.scratch[i] = ctx.heap_alloc(heap_plan.scratch_slot_bytes);

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
