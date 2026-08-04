// Bind gemma4's argument tables: weights, KV pages, IO and activations.
//
// The KV binding is the part that is this family's and not a variant of anyone
// else's. Layers 15-34 do not own KV pages — they attend the pages the most
// recent earlier layer of the SAME attention type wrote. So the page slots a
// dispatch binds come from `kv_source(L)`, not from `L`, and the KV region is
// sized for `n_kv_owning()` layers rather than for every layer.

#include "bind.hpp"

#include <stdexcept>

#include "../../batch/decode_abi.hpp"
#include "../../batch/scratch_color.hpp"
#include "encode.hpp"

namespace pie::metal::gemma4 {

namespace {

void bind_slot(RawMetalContext& ctx, int ord, std::uint8_t idx, const SlotHandle& s,
               std::size_t offset = 0) {
    if (s.valid()) ctx.arg_bind_ordinal(ord, idx, s, offset);
}

/// Where inside a shared activation buffer this dispatch's operand starts.
///
/// One case, and it is the PLE table: `PleCombine` writes all
/// `n_layers * ple_dim` of it once, and every layer's `PleGeglu` consumes its
/// own `ple_dim`-wide slice. Bound at offset 0 for all of them, all 35 layers
/// gate on layer 0's slice — which is not a crash, and not even implausible
/// numbers.
std::size_t slice_offset(const Dispatch& d, const Gemma4Geometry& g, std::uint8_t bind_index) {
    constexpr std::size_t kActBytes = 2;  // bf16
    if (d.kind == Kind::PleGeglu && bind_index == (std::uint8_t)bind::Geglu::Up && d.layer > 0) {
        return std::size_t(d.layer) * std::size_t(g.per_layer_emb_dim) * kActBytes;
    }
    return 0;
}

}  // namespace

ScratchColoring color_gemma4_scratch(const std::vector<Dispatch>& dag,
                                    const ScratchPlan& plan, bool no_recycle) {
    return model::color_family_scratch(dag.size(), plan.uses, gemma4_run_ends(dag),
                                       plan.value_count, no_recycle);
}

void bind_gemma4_dag(RawMetalContext& ctx, const BoundGemma4& b, const std::vector<Dispatch>& dag,
                     const Gemma4Geometry& g, const ScratchColoring& scratch,
                     int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        // (a) Weights. `shared_kind` is the weight-map key — deliberately not
        // `pso_kind`, which answers what the dispatch RUNS on.
        for (const WeightBind& wb : weight_binds(shared_kind(d.kind), L, DecodeGeometry{}, false)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) {
                throw std::runtime_error("gemma4 bind: unstaged weight " + wb.tensor);
            }
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }

        // (b) IO and KV.
        switch (d.kind) {
            case Kind::EmbedGather:
            case Kind::PleTokenGather:
                bind_slot(ctx, ord, (std::uint8_t)bind::Embed::TokenId, io(IoSlot::TokenId));
                break;
            case Kind::RopeQ:
            case Kind::RopeK:
                bind_slot(ctx, ord, (std::uint8_t)bind::Rope::Position, io(IoSlot::Position));
                break;
            case Kind::KvAppend: {
                // Only an owning layer appends, and it appends to its own pages.
                const auto& kv = b.kv[std::size_t(L)];
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::KPages, kv.k);
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::VPages, kv.v);
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::PositionPtr,
                          io(IoSlot::Position));
                break;
            }
            case Kind::Sdpa: {
                // The redirect: a shared layer reads the pages its source wrote.
                const int src = g.kv_source(L);
                if (src < 0) {
                    throw std::runtime_error(
                        "gemma4 bind: layer " + std::to_string(L) +
                        " shares KV but no earlier layer of its attention type owns any");
                }
                const auto& kv = b.kv[std::size_t(src)];
                bind_slot(ctx, ord, (std::uint8_t)bind::Sdpa::K, kv.k);
                bind_slot(ctx, ord, (std::uint8_t)bind::Sdpa::V, kv.v);
                bind_slot(ctx, ord, (std::uint8_t)bind::Sdpa::N, io(IoSlot::SeqLen));
                break;
            }
            // Declared by the shared routed template, never read at
            // BIASED=false. See `BoundGemma4::zero_bias`.
            case Kind::ExpertGate:
            case Kind::ExpertUp:
            case Kind::ExpertDown:
                bind_slot(ctx, ord, (std::uint8_t)bind::GoQmv::Bias, b.zero_bias);
                break;
            case Kind::RowGather:
                bind_slot(ctx, ord, (std::uint8_t)bind::RowGather::Rows,
                          io(IoSlot::SampleRows));
                break;
            case Kind::Argmax:
                bind_slot(ctx, ord, 1, io(IoSlot::TokenId));
                break;
            default:
                break;
        }

        // (c) Activations, from the coloured plan, which is indexed by POSITION
        // -- `d.ordinal` is an argument-table key and the prefill path shifts it.
        if (di < scratch.per_dispatch.size()) {
            for (const auto& sb : scratch.per_dispatch[di]) {
                if (sb.color >= 0 && std::size_t(sb.color) < b.pool.size()) {
                    bind_slot(ctx, ord, sb.bind_index, b.pool[std::size_t(sb.color)],
                              slice_offset(d, g, sb.bind_index));
                }
            }
        }
    }
}

void bind_gemma4_dag_mb(RawMetalContext& ctx, const BoundGemma4& b,
                        const std::vector<Dispatch>& dag, const Gemma4Geometry& g,
                        const ScratchColoring& scratch,
                        const std::vector<SlotHandle>& k_pages,
                        const std::vector<SlotHandle>& v_pages,
                        const MbBindOffsets& offsets, int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };
    // Per-row IO is indexed by token, so every read of it is offset by the row.
    const std::size_t row = offsets.token_row * sizeof(std::uint32_t);
    auto bind_row = [&](int ord, std::uint8_t idx, IoSlot s) {
        if (io(s).valid()) ctx.arg_bind_ordinal(ord, idx, io(s), row);
    };
    if (k_pages.size() < std::size_t(g.n_layers) || v_pages.size() < std::size_t(g.n_layers)) {
        throw std::runtime_error("gemma4 mb bind: paged KV does not cover all layers");
    }

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        for (const WeightBind& wb : weight_binds(shared_kind(d.kind), L, DecodeGeometry{}, false)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) {
                throw std::runtime_error("gemma4 mb bind: unstaged weight " + wb.tensor);
            }
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }

        switch (d.kind) {
            case Kind::EmbedGather:
            case Kind::PleTokenGather:
                bind_row(ord, (std::uint8_t)bind::Embed::TokenId, IoSlot::TokenId);
                break;
            case Kind::RopeQ:
            case Kind::RopeK:
                bind_row(ord, (std::uint8_t)bind::Rope::Position, IoSlot::Position);
                break;
            // Declared by the shared routed template, never read at
            // BIASED=false. See `BoundGemma4::zero_bias`.
            case Kind::ExpertGate:
            case Kind::ExpertUp:
            case Kind::ExpertDown:
                bind_slot(ctx, ord, (std::uint8_t)bind::GoQmv::Bias, b.zero_bias);
                break;
            // NOT row-offset: it indexes the fire's rows, so it reads the slot
            // whole.
            case Kind::RowGather:
                bind_slot(ctx, ord, (std::uint8_t)bind::RowGather::Rows,
                          io(IoSlot::SampleRows));
                break;
            case Kind::KvAppend: {
                // An owning layer scatters into its OWN pages -- `kv_source` is
                // the read redirect, not the write one.
                using P = bind::KvAppendPaged;
                bind_slot(ctx, ord, (std::uint8_t)P::KPages, k_pages[std::size_t(L)]);
                bind_slot(ctx, ord, (std::uint8_t)P::VPages, v_pages[std::size_t(L)]);
                bind_row(ord, (std::uint8_t)P::PositionIds, IoSlot::Position);
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndices, io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndptr, io(IoSlot::KvPageIndptr));
                bind_row(ord, (std::uint8_t)P::ReqOfToken, IoSlot::ReqOfToken);
                bind_row(ord, (std::uint8_t)P::WPage, IoSlot::WPage);
                bind_row(ord, (std::uint8_t)P::WOff, IoSlot::WOff);
                break;
            }
            case Kind::Sdpa: {
                const int src = g.kv_source(L);
                if (src < 0) {
                    throw std::runtime_error(
                        "gemma4 mb bind: layer " + std::to_string(L) +
                        " shares KV but no earlier layer of its attention type owns any");
                }
                using P = bind::SdpaPaged;
                bind_slot(ctx, ord, (std::uint8_t)P::KPages, k_pages[std::size_t(src)]);
                bind_slot(ctx, ord, (std::uint8_t)P::VPages, v_pages[std::size_t(src)]);
                bind_row(ord, (std::uint8_t)P::PositionIds, IoSlot::Position);
                bind_row(ord, (std::uint8_t)P::ReqOfToken, IoSlot::ReqOfToken);
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndices, io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndptr, io(IoSlot::KvPageIndptr));
                // The mask is per token row too, but its pitch is the mask's,
                // not a u32; `AttnMaskEnabled` is one byte per row.
                if (io(IoSlot::AttnMask).valid()) {
                    ctx.arg_bind_ordinal(ord, (std::uint8_t)P::AttnMask, io(IoSlot::AttnMask),
                                         offsets.token_row * gemma4_mask_pitch_bytes(g));
                }
                bind_slot(ctx, ord, (std::uint8_t)P::AttnMaskStride, io(IoSlot::AttnMaskStride));
                if (io(IoSlot::AttnMaskEnabled).valid()) {
                    ctx.arg_bind_ordinal(ord, (std::uint8_t)P::AttnMaskEnabled,
                                         io(IoSlot::AttnMaskEnabled), offsets.token_row);
                }
                break;
            }
            case Kind::LmHead:
                // The logits are the one activation that leaves the pool: a
                // prefill writes one row per token into the IO buffer.
                if (io(IoSlot::Logits).valid()) {
                    ctx.arg_bind_ordinal(ord, (std::uint8_t)bind::Qmv::Out, io(IoSlot::Logits),
                                         offsets.logits_bytes);
                }
                break;
            case Kind::Argmax:
                bind_row(ord, 1, IoSlot::NextToken);
                break;
            default:
                break;
        }

        if (di < scratch.per_dispatch.size()) {
            for (const auto& sb : scratch.per_dispatch[di]) {
                if (sb.color >= 0 && std::size_t(sb.color) < b.pool.size()) {
                    bind_slot(ctx, ord, sb.bind_index, b.pool[std::size_t(sb.color)],
                              slice_offset(d, g, sb.bind_index));
                }
            }
        }
    }
}

}  // namespace pie::metal::gemma4
