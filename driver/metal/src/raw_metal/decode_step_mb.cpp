#include "decode_step_mb.hpp"

#include <stdexcept>

#include "decode_dispatch.hpp"
#include "decode_dispatch_mb.hpp"
#include "heap_bind.hpp"

namespace pie_metal_driver::raw_metal {
namespace {

Kernel mb_kind(Kernel k) {
    switch (k) {
        case Kernel::GdnPrep: return Kernel::GdnPrepSlotted;
        case Kernel::GdnCore: return Kernel::GdnCoreSlotted;
        case Kernel::KvAppend: return Kernel::KvAppendPaged;
        case Kernel::Sdpa: return Kernel::SdpaPaged;
        default: return k;
    }
}

int qmv_out(Kernel k, const DecodeGeometry& g) {
    switch (k) {
        case Kernel::QmvIn: return g.gdn_conv_dim;
        case Kernel::QmvInZ: return g.gdn_v_total;
        case Kernel::QmvOut:
        case Kernel::QmvO:
        case Kernel::QmvDown: return g.hidden;
        case Kernel::QmvQ: return 2 * g.n_q_heads * g.head_dim;
        case Kernel::QmvK:
        case Kernel::QmvV: return g.n_kv_heads * g.head_dim;
        case Kernel::QmvGate:
        case Kernel::QmvUp: return g.intermediate;
        case Kernel::QmvLmHead: return g.vocab;
        default: return 0;
    }
}

void mb_geometry(Dispatch& d, const DecodeGeometry& g, int n) {
    auto rms = [&](int row, int rows) { rms_mb_dispatch(row, rows, n, d.grid, d.tg); };
    if (const int out = qmv_out(d.kind, g); out != 0) {
        qmv_mb_dispatch(out, n, d.grid, d.tg);
        return;
    }
    switch (d.kind) {
        case Kernel::EmbedGather:
            embed_mb_dispatch(g.hidden, n, d.grid, d.tg); break;
        case Kernel::Rms:
        case Kernel::FfnRms:
        case Kernel::FinalRms:
            rms(g.hidden, 1); break;
        case Kernel::QNorm:
            rms(g.head_dim, g.n_q_heads); break;
        case Kernel::KNorm:
            rms(g.head_dim, g.n_kv_heads); break;
        case Kernel::GdnInA:
        case Kernel::GdnInB:
            d.grid = Grid{32u, uint32_t(g.gdn_v_heads), uint32_t(n)};
            d.tg = Threadgroup{32, 1, 1};
            break;
        case Kernel::GdnPrepSlotted:
            d.grid = Grid{32u, 1u, uint32_t(n * g.gdn_v_heads)};
            d.tg = Threadgroup{32, 1, 1};
            break;
        case Kernel::GdnCoreSlotted:
            d.grid = Grid{32u, uint32_t(g.gdn_v_dim), uint32_t(n * g.gdn_v_heads)};
            d.tg = Threadgroup{32, 4, 1};
            break;
        case Kernel::GatedRms:
            d.grid = Grid{uint32_t(g.gdn_v_dim), uint32_t(g.gdn_v_heads), uint32_t(n)};
            d.tg = Threadgroup{uint32_t(g.gdn_v_dim), 1, 1};
            break;
        case Kernel::QSplit:
            d.grid = Grid{uint32_t(g.head_dim), uint32_t(g.n_q_heads), uint32_t(n)};
            d.tg = Threadgroup{uint32_t(g.head_dim), 1, 1};
            break;
        case Kernel::Rope:
            rope_mb_dispatch(g.rotary_dims, g.n_q_heads, n, d.grid, d.tg); break;
        case Kernel::RopeK:
            rope_mb_dispatch(g.rotary_dims, g.n_kv_heads, n, d.grid, d.tg); break;
        case Kernel::KvAppendPaged:
            kv_append_mb_dispatch(g.head_dim, g.n_kv_heads, n, d.grid, d.tg); break;
        case Kernel::SdpaPaged:
            sdpa_paged_dispatch(g.n_q_heads, n, d.grid, d.tg); break;
        case Kernel::AttnGate:
            elementwise_mb_dispatch(g.n_q_heads * g.head_dim, n, d.grid, d.tg); break;
        case Kernel::Residual:
        case Kernel::LayerOut:
            elementwise_mb_dispatch(g.hidden, n, d.grid, d.tg); break;
        case Kernel::SiluMul:
            elementwise_mb_dispatch(g.intermediate, n, d.grid, d.tg); break;
        default:
            throw std::runtime_error("missing multi-batch launch geometry");
    }
}

bool barrier_after_mb(const std::vector<Dispatch>& dag, size_t i) {
    if (i + 1 >= dag.size()) return true;
    const Dispatch& a = dag[i];
    const Dispatch& b = dag[i + 1];
    if (a.layer != b.layer) return true;
    return !((a.kind == Kernel::QmvK && b.kind == Kernel::QmvV) ||
             (a.kind == Kernel::QNorm && b.kind == Kernel::KNorm) ||
             (a.kind == Kernel::QmvGate && b.kind == Kernel::QmvUp) ||
             (a.kind == Kernel::QmvIn && b.kind == Kernel::QmvInZ) ||
             (a.kind == Kernel::GdnInA && b.kind == Kernel::GdnInB));
}

Pso mb_pso(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb) {
    switch (d.kind) {
        case Kernel::EmbedGather: return mb.embed_mb;
        case Kernel::Rope:
        case Kernel::RopeK: return mb.rope_mb;
        case Kernel::GdnPrepSlotted: return mb.gdn_prep_slotted;
        case Kernel::GdnCoreSlotted: return mb.gdn_recurrent_slotted;
        case Kernel::KvAppendPaged: return mb.kv_append_paged;
        case Kernel::SdpaPaged: return mb.sdpa_paged;
        default: return d.fuse_residual ? base.qmv_residual : base[d.kind];
    }
}

inline void bind_slot(RawMetalContext& ctx, int ord, uint8_t idx, const SlotHandle& slot) {
    ctx.arg_bind_ordinal(ord, idx, slot);
}

}  // namespace

std::vector<Dispatch> build_decode_dag_mb(const DecodeGeometry& g, int n_tokens,
                                          int ordinal_base, bool fuse_residual, bool gdn_prep) {
    if (n_tokens <= 0) throw std::runtime_error("multi-batch DAG requires at least one token");
    std::vector<Dispatch> dag = build_decode_dag(g, false, fuse_residual, gdn_prep);
    for (Dispatch& d : dag) {
        d.kind = mb_kind(d.kind);
        d.ordinal += ordinal_base;
        mb_geometry(d, g, n_tokens);
    }
    return dag;
}

std::vector<std::vector<Dispatch>> build_decode_prefill_dags(
    const DecodeGeometry& g, int n_tokens, bool fuse_residual, bool gdn_prep) {
    if (n_tokens <= 0) throw std::runtime_error("prefill DAG stream requires at least one token");
    std::vector<std::vector<Dispatch>> out;
    out.reserve(size_t(n_tokens));
    for (int t = 0; t < n_tokens; ++t) {
        auto dag = build_decode_dag_mb(
            g, 1, kPrefillOrdinalBase + t * kPrefillOrdinalStride, fuse_residual, gdn_prep);
        if (dag.size() >= size_t(kPrefillOrdinalStride)) {
            throw std::runtime_error("prefill DAG exceeds its argument-table ordinal stride");
        }
        out.push_back(std::move(dag));
    }
    return out;
}

void bind_decode_dag_mb(RawMetalContext& ctx, const BoundDecode& b,
                        const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                        const std::vector<SlotHandle>& k_pages,
                        const std::vector<SlotHandle>& v_pages, bool gdn_prep,
                        const MbBindOffsets& offsets) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };
    if (k_pages.size() < size_t(g.n_layers) || v_pages.size() < size_t(g.n_layers))
        throw std::runtime_error("paged KV bindings do not cover all layers");
    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        for (const WeightBind& wb : weight_binds(d.kind, L, g, gdn_prep)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) throw std::runtime_error("MB bind: unstaged weight " + wb.tensor);
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }
        switch (d.kind) {
            case Kernel::EmbedGather:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Embed::TokenId), io(IoSlot::TokenId),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::GdnPrepSlotted: {
                const auto& s = b.gdn[L];
                bind_slot(ctx, ord, uint8_t(bind::GdnPrep::ConvState), s.conv_state);
                bind_slot(ctx, ord, uint8_t(bind::GdnPrep::ConvStateOut), s.conv_state_out);
                bind_slot(ctx, ord, uint8_t(bind::GdnPrep::ConvB), s.conv_bias_zero);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::GdnPrep::SlotOfToken),
                                     io(IoSlot::SlotOfToken),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            }
            case Kernel::GdnCoreSlotted: {
                const auto& s = b.gdn[L];
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::ConvState), s.conv_state);
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::RecurrentState), s.recurrent_state);
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::ConvStateOut), s.conv_state_out);
                bind_slot(ctx, ord, uint8_t(bind::GdnCoreRecurrent::ConvB), s.conv_bias_zero);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::GdnCoreRecurrent::SlotOfToken),
                                     io(IoSlot::SlotOfToken),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            }
            case Kernel::KvAppendPaged:
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::KPages), k_pages[L]);
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::VPages), v_pages[L]);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::PositionIds),
                                     io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::KvPageIndices),
                          io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, uint8_t(bind::KvAppendPaged::KvPageIndptr),
                          io(IoSlot::KvPageIndptr));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::ReqOfToken),
                                     io(IoSlot::ReqOfToken), offsets.token_row * sizeof(uint32_t));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::WPage), io(IoSlot::WPage),
                                     offsets.token_row * sizeof(uint32_t));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::KvAppendPaged::WOff), io(IoSlot::WOff),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::SdpaPaged:
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::KPages), k_pages[L]);
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::VPages), v_pages[L]);
                ctx.arg_bind_ordinal(ord, uint8_t(bind::SdpaPaged::PositionIds),
                                     io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                ctx.arg_bind_ordinal(ord, uint8_t(bind::SdpaPaged::ReqOfToken),
                                     io(IoSlot::ReqOfToken),
                                     offsets.token_row * sizeof(uint32_t));
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::KvPageIndices),
                          io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, uint8_t(bind::SdpaPaged::KvPageIndptr),
                          io(IoSlot::KvPageIndptr));
                break;
            case Kernel::Rope:
            case Kernel::RopeK:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Rope::Position), io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::QmvLmHead:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Qmv::Out), io(IoSlot::Logits),
                                     offsets.logits_bytes);
                break;
            default:
                break;
        }
    }
}

void bind_prefill_gdn_state(RawMetalContext& ctx, const BoundDecode& b,
                            const std::vector<Dispatch>& dag, uint32_t slot, bool even) {
    for (const Dispatch& d : dag) {
        if (d.kind != Kernel::GdnPrepSlotted && d.kind != Kernel::GdnCoreSlotted) continue;
        const auto& s = b.gdn[size_t(d.layer)];
        const SlotHandle& in = even ? s.conv_state : s.conv_state_out;
        const SlotHandle& out = even ? s.conv_state_out : s.conv_state;
        if (d.kind == Kernel::GdnPrepSlotted) {
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnPrep::ConvState), in);
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnPrep::ConvStateOut), out);
        } else {
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnCoreRecurrent::ConvState), in);
            ctx.arg_bind_ordinal(d.ordinal, uint8_t(bind::GdnCoreRecurrent::ConvStateOut), out);
        }
    }
    (void)slot;  // the slotted shader consumes the per-token SlotOfToken buffer.
}

void encode_decode_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const DecodeStepPsos& base_psos, const MultiBatchPsos& mb_psos,
                           bool force_barriers) {
    for (size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        se.set_pso(mb_pso(d, base_psos, mb_psos));
        se.set_argtable(d.kind, d.ordinal);
        se.dispatch(d.grid, d.tg);
        if (force_barriers || barrier_after_mb(dag, i)) se.barrier();
    }
}

}  // namespace pie_metal_driver::raw_metal
