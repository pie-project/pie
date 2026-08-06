#include <cstdlib>
#include <iostream>
#include "decode_step_mb.hpp"

#include <limits>
#include <stdexcept>

#include "decode_dispatch.hpp"
#include "decode_consts.hpp"
#include "decode_dispatch_mb.hpp"
#include "heap_bind.hpp"

namespace pie::metal {
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

}  // namespace

int qmv_out_size(Kernel k, const DecodeGeometry& g) {
    switch (k) {
        case Kernel::QmvIn: return g.gdn_conv_dim;
        case Kernel::QmvInZ: return g.gdn_v_total;
        case Kernel::GdnInA:
        case Kernel::GdnInB: return g.gdn_v_heads;
        case Kernel::QmvOut:
        case Kernel::QmvO:
        case Kernel::QmvDown:
        case Kernel::LlSharedDown: return g.hidden;
        case Kernel::QmvQ: return 2 * g.n_q_heads * g.head_dim;
        case Kernel::QmvK:
        case Kernel::QmvV: return g.n_kv_heads * g.head_dim;
        case Kernel::QmvGate:
        case Kernel::QmvUp: return g.intermediate;
        case Kernel::QmvLmHead:
        case Kernel::LmHeadUntied: return g.vocab;
        // The router is a DENSE matvec into one logit per expert: it runs over
        // the tokens like any other projection, and only what follows it is
        // routed. The three expert projections are deliberately absent -- they
        // run over the SORTED rows, and answering here would launch them over
        // the token count instead, computing the first `n` sorted rows and
        // leaving the rest of the stack holding the previous layer's output.
        case Kernel::LlRouter: return g.n_experts;
        // The shared expert is dense in every sense -- it runs over the tokens,
        // one row each, so it answers here like any other projection.
        case Kernel::LlSharedGate:
        case Kernel::LlSharedUp: return g.shared_intermediate;
        case Kernel::LlSharedGateProj: return 1;
        default: return 0;
    }
}

namespace {
bool g_ab_arm = false;
}  // namespace

bool ab_all_barriers() {
    static const bool on = std::getenv("PIE_METAL_AB_BARRIERS") != nullptr;
    return on;
}

bool ab_enabled() {
    static const bool on = std::getenv("PIE_METAL_AB") != nullptr;
    return on;
}
bool ab_arm() { return g_ab_arm; }
void ab_set_arm(bool b) { g_ab_arm = b; }

namespace {

void mb_geometry(Dispatch& d, const DecodeGeometry& g, int n) {
    auto rms = [&](int row, int rows) { rms_mb_dispatch(row, rows, n, d.grid, d.tg); };
    if (const int out = qmv_out_size(d.kind, g); out != 0) {
        // `qmm_bn_unsplit`, not `qmm_bn`: the widest-tile rule is correct only
        // where split-K supplies the threadgroups the wide tile gives up, and
        // this family dispatches no split -- see the comment on `qmm_split`
        // just below, which is the same reason gemma4 and gpt-oss have.
        d.qmm_bn = qmm_bn_unsplit(out, n, qmm_min_batch(g.is_moe()));
        d.qmm_bm = qmm_bm(n);
        // The second quantized set has a matvec and no GEMM: a checkpoint that
        // spares its two routing projections at 8 bits gets one extra pipeline
        // table, not an extra table of every batched shape. Both are
        // hidden-by-a-few, so this costs nothing worth a second table.
        if (qwen35_uses_alt_quant(d.kind, g)) d.qmm_bn = 0;
        // NO split-K, for exactly the reason gemma4's `launch_shape_mb` gives:
        // the split GEMM writes `split_k` partial [M, N] slices into a side
        // buffer and needs a reduce pass to sum them, and NOTHING IN THIS
        // DRIVER EVER DISPATCHES THAT PASS. `qmm_splitk_reduce` is compiled and
        // has no caller; `affine_qmm_t_splitk` writes its result to buffer 8,
        // the partials, and buffer 4 -- the projection's real output -- keeps
        // whatever the last fire left in it.
        //
        // It survived because the split only engages at `qmm_bn != 0`, which
        // needs a batch of at least `qmm_min_batch()`, and nothing fired one until
        // the throughput harness did. With the harness's fleet check it is a
        // one-line reproduction: sixteen copies of one prompt in one fire
        // answer 74088 and 1125 at step 0 with the split on, and agree with the
        // single-sequence decode for all 64 steps with it off.
        //
        // The measured cost of turning it off is the honest version of a number
        // that was never real: 717 tok/s becomes 520 at sixteen lanes, and 717
        // was the speed of computing the wrong answer.
        //
        // The fix is to emit the reduce, which this family cannot do here --
        // `mb_geometry` decides the split while walking a DAG that is already
        // built, so it has no way to insert a dispatch after itself.
        d.qmm_split = 1;
        if (d.qmm_split > 1)
            qmm_t_splitk_dispatch(out, n, d.qmm_bm, d.qmm_split, d.grid, d.tg);
        else if (d.qmm_bn > 0)
            qmm_t_dispatch(out, n, d.qmm_bn, d.qmm_bm, d.grid, d.tg);
        else
            qmv_mb_dispatch(out, n, d.grid, d.tg);
        return;
    }
    switch (d.kind) {
        case Kernel::EmbedUntied:
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
            // Routed, the dense SwiGLU that remains is the SHARED expert's --
            // one row a token, at its own width. The mixture's own SwiGLU is
            // `LlExpertSiluMul` below, over the sorted stack, and the two were
            // split precisely because this line cannot say both.
            elementwise_mb_dispatch(g.is_moe() ? g.shared_intermediate : g.intermediate,
                                    n, d.grid, d.tg);
            break;
        case Kernel::LlExpertSiluMul:
            // The slot axis is gone -- a sorted row IS a slot -- so this is the
            // same elementwise shape over a taller batch.
            elementwise_mb_dispatch(g.moe_intermediate, moe_sorted_rows(g, n), d.grid, d.tg);
            break;

        // ── the mixture ──
        // The three expert projections first, because `qmv_out_size` does not
        // answer for them: they run over the sorted rows, which is neither `n`
        // nor `n * k`, because the sort pads every expert's run to a whole
        // tile. Once the batch fills a tile they become matmuls, and the tile
        // is `moe_tile_rows`'s answer -- the same number the sort padded to,
        // asked of the same function rather than restated.
        case Kernel::LlExpertGate:
        case Kernel::LlExpertUp:
        case Kernel::LlExpertDown: {
            const int N = d.kind == Kernel::LlExpertDown ? g.hidden : g.moe_intermediate;
            const int sorted = moe_sorted_rows(g, n);
            // Routed, so the routed crossover.
            if (const int bn = qmm_bn(N, sorted, qmm_min_batch(true));
                bn > 0 && shared_kernels::moe_should_batch(n * g.experts_per_token, g.n_experts)) {
                d.qmm_bn = bn;
                d.qmm_bm = shared_kernels::moe_tile_rows(n * g.experts_per_token, g.n_experts);
                d.qmm_split = 1;
                qmm_t_dispatch(N, sorted, bn, d.qmm_bm, d.grid, d.tg);
            } else {
                // One sorted row per (token, slot) pair and no expert axis: the
                // pair's expert is `row_expert[p]`, not `tid.z`.
                shared_kernels::routed_qmv_dispatch(N, 1, d.grid, d.tg, sorted);
            }
            break;
        }
        case Kernel::GoRouterTopK:
            shared_kernels::router_topk_dispatch(g.n_experts, d.grid, d.tg, n); break;
        case Kernel::LlMoeSort:
            shared_kernels::moe_route_sort_dispatch(g.n_experts, d.grid, d.tg); break;
        case Kernel::LlMoeGather:
            shared_kernels::moe_route_rows_dispatch(g.hidden, moe_sorted_rows(g, n),
                                                    d.grid, d.tg); break;
        case Kernel::LlMoeCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, d.grid, d.tg, n); break;
        case Kernel::LlSharedCombine:
            elementwise_mb_dispatch(g.hidden, n, d.grid, d.tg); break;

        default:
            throw std::runtime_error("missing multi-batch launch geometry");
    }
}

bool barrier_after_mb(const std::vector<Dispatch>& dag, size_t i,
                      const std::vector<int>& run_ends) {
    if (i + 1 >= dag.size()) return true;
    return run_ends[i] == int(i);
}

Pso mb_pso(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb) {
    switch (d.kind) {
        case Kernel::EmbedUntied:
        case Kernel::EmbedGather: return mb.embed_mb;
        case Kernel::Rope:
        case Kernel::RopeK: return mb.rope_mb;
        case Kernel::GdnPrepSlotted: return mb.gdn_prep_slotted;
        case Kernel::GdnCoreSlotted: return mb.gdn_recurrent_slotted;
        case Kernel::KvAppendPaged: return mb.kv_append_paged;
        case Kernel::SdpaPaged: return mb.sdpa_paged;
        // The mixture's projections, asked BEFORE the shared GEMM below and
        // for the same reason `qmv_out_size` declines to answer for them: they
        // carry a `qmm_bn` like any batched projection, so the default arm
        // would hand them the DENSE GEMM -- which indexes one weight for the
        // whole dispatch and would run every expert's rows through expert 0's
        // slice. Fluent, and wrong.
        case Kernel::LlExpertGate:
        case Kernel::LlExpertUp:
        case Kernel::LlExpertDown: {
            if (d.qmm_bn > 0) {
                const int slot = d.qmm_bn == 64 ? 2 : (d.qmm_bn == 32 ? 1 : 0);
                const int bm = shared_kernels::moe_bm_slot(d.qmm_bm);
                if (mb.qmm_routed[bm][slot].valid()) return mb.qmm_routed[bm][slot];
            }
            return base[d.kind];
        }
        default: {
            const int wide_ = qmm_bm_slot(d.qmm_bm);
            if (d.qmm_split > 1 && mb.qmm_t_splitk[wide_].valid())
                return mb.qmm_t_splitk[wide_];
            if (d.qmm_bn > 0) {
                const int slot = d.qmm_bn == 64 ? 2 : (d.qmm_bn == 32 ? 1 : 0);
                const int wide = wide_;
                const Pso& gemm = d.fuse_residual ? mb.qmm_t_residual[wide][slot]
                                                  : mb.qmm_t[wide][slot];
                if (gemm.valid()) return gemm;
            }
            return d.fuse_residual ? base.qmv_residual : base[d.kind];
        }
    }
}

inline void bind_slot(RawMetalContext& ctx, int ord, uint8_t idx, const SlotHandle& slot) {
    ctx.arg_bind_ordinal(ord, idx, slot);
}

}  // namespace

std::size_t paged_attention_mask_pitch_bytes(
    const DecodeGeometry& geometry) {
    if (geometry.total_pages <= 0 ||
        geometry.kv_page_size <= 0 ||
        static_cast<std::size_t>(geometry.total_pages) >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(
                    geometry.kv_page_size)) {
        return 0;
    }
    return static_cast<std::size_t>(geometry.total_pages) *
           static_cast<std::size_t>(geometry.kv_page_size);
}

bool paged_pool_size_supported(
    const DecodeGeometry& geometry,
    std::uint32_t pages) {
    return pages != 0 &&
           geometry.total_pages > 0 &&
           pages <=
               static_cast<std::uint32_t>(
                   geometry.total_pages);
}

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
            case Kernel::EmbedUntied:
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
                ctx.arg_bind_ordinal(
                    ord,
                    uint8_t(bind::SdpaPaged::AttnMask),
                    io(IoSlot::AttnMask),
                    offsets.token_row *
                        paged_attention_mask_pitch_bytes(g));
                bind_slot(
                    ctx, ord,
                    uint8_t(bind::SdpaPaged::AttnMaskStride),
                    io(IoSlot::AttnMaskStride));
                ctx.arg_bind_ordinal(
                    ord,
                    uint8_t(bind::SdpaPaged::AttnMaskEnabled),
                    io(IoSlot::AttnMaskEnabled),
                    offsets.token_row);
                break;
            case Kernel::Rope:
            case Kernel::RopeK:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Rope::Position), io(IoSlot::Position),
                                     offsets.token_row * sizeof(uint32_t));
                break;
            case Kernel::LmHeadUntied:
            case Kernel::QmvLmHead:
                ctx.arg_bind_ordinal(ord, uint8_t(bind::Qmv::Out), io(IoSlot::Logits),
                                     offsets.logits_bytes);
                break;
            default:
                break;
        }
    }
}

void alias_decode_conv_state_out(RawMetalContext& ctx, const BoundDecode& b,
                                 const std::vector<Dispatch>& dag) {
    for (const Dispatch& d : dag) {
        if (d.kind != Kernel::GdnPrepSlotted && d.kind != Kernel::GdnCoreSlotted) continue;
        const auto& s = b.gdn[size_t(d.layer)];
        if (!s.conv_state.valid()) continue;
        const uint8_t idx = d.kind == Kernel::GdnPrepSlotted
                                ? uint8_t(bind::GdnPrep::ConvStateOut)
                                : uint8_t(bind::GdnCoreRecurrent::ConvStateOut);
        ctx.arg_bind_ordinal(d.ordinal, idx, s.conv_state);
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

namespace {

// The only state a prompt token hands to the next one: the GDN recurrent and
// convolution slots, which every token reads and writes in place (and which
// ping-pong, so token t's output buffer is token t+1's input). Paged KV is not
// on this list -- each token appends at its own position, and the group barrier
// below already orders the whole append group ahead of the whole attention
// group.
// The tail that exists only to produce a row's logits.
bool produces_logits(Kernel kind) {
    return kind == Kernel::QmvLmHead || kind == Kernel::LmHeadUntied ||
           kind == Kernel::Argmax;
}

bool carries_cross_token_state(Kernel kind) {
    switch (kind) {
        case Kernel::GdnPrep:
        case Kernel::GdnPrepSlotted:
        case Kernel::GdnCore:
        case Kernel::GdnCoreSlotted:
            return true;
        default:
            return false;
    }
}

}  // namespace

// A prefill walked token by token streams the whole checkpoint once per prompt
// token -- 34 passes over 405MB for a 34-token prompt, which measured 118ms of
// GPU against a 1.0ms bandwidth floor. Every token's DAG is the same DAG with
// different argument tables, so walking it dispatch by dispatch instead reads
// each weight tensor once and hands it to all N tokens back to back, small
// enough to stay in cache, and gives the GPU N-way work where it had one narrow
// dispatch.
//
// The reorder is legal because a barrier closes every dispatch group: token t's
// dispatch i is ordered ahead of any token's dispatch i+1, which is every
// dependency inside a token and every dependency through paged KV. The one
// exception is the GDN state above, which is serialized token by token.
void encode_prefill_dags_mb(StepEncoder& se,
                            const std::vector<std::vector<Dispatch>>& dags,
                            int n_tokens,
                            const DecodeStepPsos& base_psos,
                            const MultiBatchPsos& mb_psos,
                            bool force_barriers,
                            const std::vector<std::uint8_t>& row_needs_logits,
                            const DecodeGeometry* geometry,
                            int max_rows,
                            const std::vector<GdnScanSegment>& gdn_scans) {
    const size_t n = n_tokens > 0 ? size_t(n_tokens) : 0;
    if (n == 0 || dags.size() < n) return;
    const size_t length = dags[0].size();
    // Token-major is the contract this reorders; if the DAGs ever stop being the
    // same shape, that assumption is gone and so is the reorder.
    for (size_t t = 1; t < n; ++t) {
        if (dags[t].size() != length) {
            for (size_t k = 0; k < n; ++k)
                encode_decode_step_mb(se, dags[k], base_psos, mb_psos, force_barriers);
            return;
        }
    }
    // Every group ends in a barrier. Letting the adjacent-group hazard rule
    // relax some of them was measured and moved nothing (93.6ms against 93.0ms):
    // N-way concurrency inside a group already saturates the machine.
    // lm_head reads the whole output embedding -- 127MB for this checkpoint,
    // far past any cache -- so running it for a prompt row nobody samples is the
    // single most expensive wasted dispatch in a prefill.
    const bool skip_unsampled_logits = row_needs_logits.size() >= n;
    // A projection does not depend on any other token's row, so the whole prompt
    // can go through the GEMM in one dispatch instead of one GEMV per token --
    // which is what turns N reads of the checkpoint into one. lm_head is left
    // out: it writes the logits buffer, whose rows are the fire's readout rows
    // and are not padded, and it already runs only for sampled rows.
    const int strided_rows =
        geometry != nullptr ? qmm_strided_rows(int(n), max_rows) : 0;
    for (size_t i = 0; i < length; ++i) {
        const Dispatch& d0 = dags[0][i];
        if (strided_rows > 0 && d0.kind != Kernel::QmvLmHead &&
            d0.kind != Kernel::LmHeadUntied &&
            !qwen35_uses_alt_quant(d0.kind, *geometry)) {
            const int out = qmv_out_size(d0.kind, *geometry);
            if (out == 16 && !d0.fuse_residual && mb_psos.qmv_wide_strided.valid()) {
                constexpr int vecs = 4;
                constexpr int lanes = 8;
                se.set_pso(mb_psos.qmv_wide_strided);
                se.set_argtable(d0.kind, d0.ordinal);
                se.dispatch(
                    Grid{32u * std::uint32_t((int(n) + vecs - 1) / vecs),
                         2u * std::uint32_t(
                             (out + (64 / lanes) - 1) / (64 / lanes)), 1},
                    Threadgroup{32, 2, 1});
                se.barrier();
                continue;
            }
            // N=16 GDN projections deliberately stay matvecs. A BN16 strided
            // GEMM replaced 768 dispatches with six, but end-to-end prefill fell
            // from 1408 to 1396 tok/s. The wide matvec above is the intermediate
            // primitive: five vectors reuse each decoded weight chunk without
            // paying a matrix tile's setup.
            if (out != 0 && out % 32 == 0) {
                const bool wide = qmm_strided_bm(strided_rows) > kQmmBM &&
                                  mb_psos.qmm_t_strided_wide.valid();
                const bool fp16 = mb_psos.qmm_t_strided_cast.valid() &&
                                  mb_psos.qmm_t_strided_fp16_precast.valid();
                const Pso& gemm = fp16
                    ? (wide ? (d0.fuse_residual
                                   ? mb_psos.qmm_t_strided_fp16_precast_wide_residual
                                   : mb_psos.qmm_t_strided_fp16_precast_wide)
                            : (d0.fuse_residual
                                   ? mb_psos.qmm_t_strided_fp16_precast_residual
                                   : mb_psos.qmm_t_strided_fp16_precast))
                    : (wide ? (d0.fuse_residual ? mb_psos.qmm_t_strided_wide_residual
                                                : mb_psos.qmm_t_strided_wide)
                            : (d0.fuse_residual ? mb_psos.qmm_t_strided_residual
                                                : mb_psos.qmm_t_strided));
                if (gemm.valid()) {
                    Grid grid;
                    Threadgroup tg;
                    qmm_t_strided_dispatch(out, strided_rows, grid, tg);
                    if (fp16) {
                        se.set_pso(mb_psos.qmm_t_strided_cast);
                        se.set_argtable(d0.kind, d0.ordinal);
                        se.dispatch(
                            Grid{std::uint32_t(qmv_kn(d0.kind, *geometry).K),
                                 std::uint32_t(strided_rows), 1},
                            Threadgroup{256, 1, 1});
                        se.barrier();
                    }
                    se.set_pso(gemm);
                    se.set_argtable(d0.kind, d0.ordinal);
                    se.dispatch(grid, tg);
                    se.barrier();
                    continue;
                }
            }
        }
        // Row-independent kernels: the prefill's scratch rows are a uniform pitch
        // apart, so one dispatch over the whole prompt replaces one per token.
        // The row-blocked variants take that pitch explicitly and are otherwise
        // byte-for-byte the same arithmetic.
        {
            Pso strided{};
            Grid grid = d0.grid;
            switch (d0.kind) {
                case Kernel::Rms:
                case Kernel::FfnRms:
                case Kernel::FinalRms:
                    // One threadgroup per row only; the per-head norms (q/k) stack
                    // several rows inside one dispatch and do not have a uniform pitch.
                    if (d0.grid.x == d0.tg.x && d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_psos.rms_strided;
                        grid.x = d0.grid.x * uint32_t(n);
                    }
                    break;
                case Kernel::SiluMul:
                    if (d0.grid.y == 1 && d0.grid.z == 1) {
                        strided = mb_psos.silu_mul_strided;
                        grid.y = uint32_t(n);
                    }
                    break;
                case Kernel::GatedRms:
                    if (d0.grid.z == 1) {
                        strided = mb_psos.gated_rms_strided;
                        grid.z = uint32_t(n);
                    }
                    break;
                default:
                    break;
            }
            if (strided.valid()) {
                se.set_pso(strided);
                se.set_argtable(d0.kind, d0.ordinal);
                se.dispatch(grid, d0.tg);
                se.barrier();
                continue;
            }
        }
        const bool serialize = carries_cross_token_state(dags[0][i].kind);
        // The GDN pair is the prefill's only strict chain -- a barrier per token,
        // 34 of them per layer.  When the caller says the prompt is one request's
        // and hands us an even scan length (so the ping-pong lands where the
        // trailing per-token dispatch expects it), the whole prompt goes through
        // in one dispatch each: prep is token-parallel once its conv window comes
        // off `mixed`, and the recurrent scan runs in registers.
        // Each request's recurrence is independent, so one scan per request
        // replaces that request's whole per-token chain; rows a scan does not
        // cover (an even tail) fall through to the per-token path below.
        std::vector<bool> scanned(n, false);
        if (serialize && !gdn_scans.empty()) {
            const Pso& scan = dags[0][i].kind == Kernel::GdnPrepSlotted
                                  ? mb_psos.gdn_prep_prefill
                                  : mb_psos.gdn_core_prefill;
            if (scan.valid()) {
                se.set_pso(scan);
                for (const GdnScanSegment& seg : gdn_scans) {
                    if (seg.rows < 2 || size_t(seg.start) >= n) continue;
                    const Dispatch& d = dags[size_t(seg.start)][i];
                    Grid grid = d.grid;
                    // prep is (dk, 1, rows*heads); the recurrent kernel keeps
                    // its (dk, dv, heads) grid and walks the rows internally.
                    if (d.kind == Kernel::GdnPrepSlotted)
                        grid.z = d.grid.z * uint32_t(seg.rows);
                    se.set_argtable(d.kind, d.ordinal);
                    se.dispatch(grid, d.tg);
                    for (int r = 0; r < seg.rows; ++r)
                        scanned[size_t(seg.start + r)] = true;
                }
                se.barrier();
            }
        }
        const bool logits_tail = skip_unsampled_logits && produces_logits(dags[0][i].kind);
        bool prev_emitted = false;
        for (size_t t = 0; t < n; ++t) {
            if (serialize && scanned[t]) continue;
            if (logits_tail && row_needs_logits[t] == 0) continue;
            const Dispatch& d = dags[t][i];
            if (d.kind != dags[0][i].kind) {
                for (size_t k = 0; k < n; ++k)
                    encode_decode_step_mb(se, dags[k], base_psos, mb_psos, force_barriers);
                return;
            }
            se.set_pso(mb_pso(d, base_psos, mb_psos));
            se.set_argtable(d.kind, d.ordinal);
            se.dispatch(d.grid, d.tg);
            // Barrier between consecutive per-token GDN dispatches only; a
            // scanned row in between has already been ordered by the scan's
            // own barrier above.
            if (serialize && prev_emitted) se.barrier();
            prev_emitted = true;
        }
        se.barrier();
    }
}

void encode_decode_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const DecodeStepPsos& base_psos, const MultiBatchPsos& mb_psos,
                           bool force_barriers) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        se.set_pso(mb_pso(d, base_psos, mb_psos));
        se.set_argtable(d.kind, d.ordinal);
        se.dispatch(d.grid, d.tg);
        // Arm B of the interleaved A/B is a CONTROL by default: identical to
        // arm A, so a nonzero A-B difference means the harness is biased and
        // nothing else. Point it at whatever is being evaluated by adding a
        // term here.
        //
        // It used to default to "barrier after EVERY dispatch", which is a fine
        // question but a terrible default: any other question asked of this
        // harness came back dominated by it, and one such answer (a 12% win
        // that was really 0) reached a commit before being caught.
        // `PIE_METAL_AB_BARRIERS=1` asks the old question explicitly.
        if (force_barriers || barrier_after_mb(dag, i, run_ends) ||
            (ab_enabled() && ab_arm() && ab_all_barriers()))
            se.barrier();
    }
}

}  // namespace pie::metal
