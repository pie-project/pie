// Bind gpt-oss's argument tables: weights, KV, IO and activations.

#include "bind.hpp"

#include <stdexcept>

#include "../../batch/decode_abi.hpp"
#include "../../batch/scratch_color.hpp"
#include "encode.hpp"

namespace pie::metal::gptoss {

namespace {

void bind_slot(RawMetalContext& ctx, int ord, std::uint8_t idx, const SlotHandle& s,
               std::size_t offset = 0) {
    if (s.valid()) ctx.arg_bind_ordinal(ord, idx, s, offset);
}

}  // namespace

ScratchColoring color_gptoss_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                     bool no_recycle) {
    std::vector<pie::metal::scratch::Use> uses;
    uses.reserve(plan.uses.size());
    for (const Use& u : plan.uses) {
        uses.push_back({u.index, u.bind_index, u.value, u.is_write});
    }
    const auto colored = pie::metal::scratch::color_live_ranges(uses, gptoss_run_ends(dag),
                                                               plan.value_count, no_recycle);
    ScratchColoring out;
    out.colors_used = colored.colors_used;
    out.hazard_free = colored.hazard_free;
    out.per_dispatch.resize(dag.size());
    for (const Use& u : plan.uses) {
        out.per_dispatch[std::size_t(u.index)].push_back(
            {u.bind_index, colored.color[std::size_t(u.value)]});
    }
    return out;
}

void bind_gptoss_dag(RawMetalContext& ctx, const BoundGptOss& b, const std::vector<Dispatch>& dag,
                     const GptOssGeometry& g, const ScratchColoring& scratch, int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        // (a) Weights.
        for (const WeightBind& wb : weight_binds(shared_kind(d.kind), L, DecodeGeometry{}, false)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) {
                throw std::runtime_error("gpt-oss bind: unstaged weight " + wb.tensor);
            }
            bind_slot(ctx, ord, wb.bind_index, it->second);
        }

        // (b) IO and KV.
        switch (d.kind) {
            case Kind::EmbedGather:
                bind_slot(ctx, ord, (std::uint8_t)bind::Embed::TokenId, io(IoSlot::TokenId));
                break;
            case Kind::RopeQ:
            case Kind::RopeK:
                bind_slot(ctx, ord, (std::uint8_t)bind::RopeFreqs::Position, io(IoSlot::Position));
                break;
            case Kind::KvAppend: {
                const auto& kv = b.kv[std::size_t(L)];
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::KPages, kv.k);
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::VPages, kv.v);
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::PositionPtr,
                          io(IoSlot::Position));
                break;
            }
            case Kind::SdpaSink: {
                const auto& kv = b.kv[std::size_t(L)];
                bind_slot(ctx, ord, (std::uint8_t)bind::SdpaSink::K, kv.k);
                bind_slot(ctx, ord, (std::uint8_t)bind::SdpaSink::V, kv.v);
                bind_slot(ctx, ord, (std::uint8_t)bind::SdpaSink::N, io(IoSlot::SeqLen));
                break;
            }
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

        // (c) Activations, from the coloured plan, which is indexed by POSITION.
        if (di < scratch.per_dispatch.size()) {
            for (const auto& sb : scratch.per_dispatch[di]) {
                if (sb.color >= 0 && std::size_t(sb.color) < b.pool.size()) {
                    bind_slot(ctx, ord, sb.bind_index, b.pool[std::size_t(sb.color)]);
                }
            }
        }
    }
    (void)g;
}

void bind_gptoss_dag_paged(RawMetalContext& ctx, const BoundGptOss& b,
                           const std::vector<Dispatch>& dag, const GptOssGeometry& g,
                           const ScratchColoring& scratch,
                           const std::vector<SlotHandle>& k_pages,
                           const std::vector<SlotHandle>& v_pages, int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };
    if (k_pages.size() < std::size_t(g.n_layers) || v_pages.size() < std::size_t(g.n_layers)) {
        throw std::runtime_error("gpt-oss paged bind: KV pages do not cover all layers");
    }

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        for (const WeightBind& wb : weight_binds(shared_kind(d.kind), L, DecodeGeometry{}, false)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) {
                throw std::runtime_error("gpt-oss paged bind: unstaged weight " + wb.tensor);
            }
            // The sinks tensor moves: `bind::SdpaSink::Sinks` is 14, which on
            // the paged ABI is `AttnMaskEnabled`. Binding it there is not a
            // crash -- it is a weight read as a mask, and a mask read as a
            // weight.
            const std::uint8_t idx =
                d.kind == Kind::SdpaSink && wb.bind_index == (std::uint8_t)bind::SdpaSink::Sinks
                    ? (std::uint8_t)bind::SdpaPaged::Sinks
                    : wb.bind_index;
            bind_slot(ctx, ord, idx, it->second);
        }

        switch (d.kind) {
            case Kind::EmbedGather:
                bind_slot(ctx, ord, (std::uint8_t)bind::Embed::TokenId, io(IoSlot::TokenId));
                break;
            case Kind::RopeQ:
            case Kind::RopeK:
                bind_slot(ctx, ord, (std::uint8_t)bind::RopeFreqs::Position, io(IoSlot::Position));
                break;
            case Kind::KvAppend: {
                using P = bind::KvAppendPaged;
                bind_slot(ctx, ord, (std::uint8_t)P::KPages, k_pages[std::size_t(L)]);
                bind_slot(ctx, ord, (std::uint8_t)P::VPages, v_pages[std::size_t(L)]);
                bind_slot(ctx, ord, (std::uint8_t)P::PositionIds, io(IoSlot::Position));
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndices, io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndptr, io(IoSlot::KvPageIndptr));
                bind_slot(ctx, ord, (std::uint8_t)P::ReqOfToken, io(IoSlot::ReqOfToken));
                bind_slot(ctx, ord, (std::uint8_t)P::WPage, io(IoSlot::WPage));
                bind_slot(ctx, ord, (std::uint8_t)P::WOff, io(IoSlot::WOff));
                break;
            }
            case Kind::SdpaSink: {
                using P = bind::SdpaPaged;
                bind_slot(ctx, ord, (std::uint8_t)P::KPages, k_pages[std::size_t(L)]);
                bind_slot(ctx, ord, (std::uint8_t)P::VPages, v_pages[std::size_t(L)]);
                bind_slot(ctx, ord, (std::uint8_t)P::PositionIds, io(IoSlot::Position));
                bind_slot(ctx, ord, (std::uint8_t)P::ReqOfToken, io(IoSlot::ReqOfToken));
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndices, io(IoSlot::KvPageIndices));
                bind_slot(ctx, ord, (std::uint8_t)P::KvPageIndptr, io(IoSlot::KvPageIndptr));
                bind_slot(ctx, ord, (std::uint8_t)P::AttnMask, io(IoSlot::AttnMask));
                bind_slot(ctx, ord, (std::uint8_t)P::AttnMaskStride, io(IoSlot::AttnMaskStride));
                bind_slot(ctx, ord, (std::uint8_t)P::AttnMaskEnabled,
                          io(IoSlot::AttnMaskEnabled));
                break;
            }
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

        if (di < scratch.per_dispatch.size()) {
            for (const auto& sb : scratch.per_dispatch[di]) {
                if (sb.color >= 0 && std::size_t(sb.color) < b.pool.size()) {
                    bind_slot(ctx, ord, sb.bind_index, b.pool[std::size_t(sb.color)]);
                }
            }
        }
    }
}

}  // namespace pie::metal::gptoss
