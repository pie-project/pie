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

ScratchColoring color_gemma4_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                     bool no_recycle) {
    std::vector<pie::metal::scratch::Use> uses;
    uses.reserve(plan.uses.size());
    for (const Use& u : plan.uses) {
        uses.push_back({u.ordinal, u.bind_index, u.value, u.is_write});
    }
    const auto colored = pie::metal::scratch::color_live_ranges(uses, gemma4_run_ends(dag),
                                                               plan.value_count, no_recycle);
    ScratchColoring out;
    out.colors_used = colored.colors_used;
    out.hazard_free = colored.hazard_free;
    out.per_dispatch.resize(dag.size());
    for (const Use& u : plan.uses) {
        out.per_dispatch[std::size_t(u.ordinal)].push_back(
            {u.bind_index, colored.color[std::size_t(u.value)]});
    }
    return out;
}

void bind_gemma4_dag(RawMetalContext& ctx, const BoundGemma4& b, const std::vector<Dispatch>& dag,
                     const Gemma4Geometry& g, const ScratchColoring& scratch,
                     int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    for (const Dispatch& d : dag) {
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
            case Kind::Argmax:
                bind_slot(ctx, ord, 1, io(IoSlot::TokenId));
                break;
            default:
                break;
        }

        // (c) Activations, from the coloured plan.
        if (std::size_t(d.ordinal) < scratch.per_dispatch.size()) {
            for (const auto& sb : scratch.per_dispatch[std::size_t(d.ordinal)]) {
                if (sb.color >= 0 && std::size_t(sb.color) < b.pool.size()) {
                    bind_slot(ctx, ord, sb.bind_index, b.pool[std::size_t(sb.color)],
                              slice_offset(d, g, sb.bind_index));
                }
            }
        }
    }
}

}  // namespace pie::metal::gemma4
