// Bind the llama families' argument tables: weights, KV pages, IO and
// activations.
//
// The shortest binder in the driver, and for a reason worth stating. Gemma 4
// needs prose here because its KV-shared layers attend pages an EARLIER layer
// wrote, so a dispatch's page slots come from `kv_source(L)` rather than from
// `L`. Llama's layers are uniform: layer L appends to layer L's pages and reads
// them straight back.
//
// The routed FFN adds nothing either, which is less obvious. A mixture-of-
// experts layer needs the router's chosen ids and their softmax weights, and
// those look like they ought to be a new kind of resource. They are not --
// they are ACTIVATIONS, produced by `RouterTopK` and consumed a few dispatches
// later, so they come out of the colour pool like every other value and the
// binder never names them. The only routed-specific work in this whole file is
// the weight map, and that is in `heap_bind.cpp`.

#include "bind.hpp"

#include <stdexcept>

#include "../../batch/decode_abi.hpp"
#include "encode.hpp"

namespace pie::metal::llama {

namespace {

void bind_slot(RawMetalContext& ctx, int ord, std::uint8_t idx, const SlotHandle& s,
               std::size_t offset = 0) {
    if (s.valid()) ctx.arg_bind_ordinal(ord, idx, s, offset);
}

}  // namespace

ScratchColoring color_llama_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                    bool no_recycle) {
    return model::color_family_scratch(dag.size(), plan.uses, llama_run_ends(dag),
                                       plan.value_count, no_recycle);
}

void bind_llama_dag(RawMetalContext& ctx, const BoundLlama& b, const std::vector<Dispatch>& dag,
                    const LlamaGeometry& g, const ScratchColoring& scratch, int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        // (a) Weights. `shared_kind` is the weight-map key -- deliberately not
        // `pso_kind`, which answers what the dispatch RUNS on. The two disagree
        // for the routed matvecs, which bind the expert stack but run the
        // routed kernel.
        for (const WeightBind& wb :
             weight_binds(shared_kind(d.kind, g), L, DecodeGeometry{}, false)) {
            const auto it = b.weights.find(wb.tensor);
            if (it == b.weights.end()) {
                throw std::runtime_error("llama bind: unstaged weight " + wb.tensor);
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
                bind_slot(ctx, ord, (std::uint8_t)bind::Rope::Position, io(IoSlot::Position));
                break;
            case Kind::KvAppend: {
                const auto& kv = b.kv[std::size_t(L)];
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::KPages, kv.k);
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::VPages, kv.v);
                bind_slot(ctx, ord, (std::uint8_t)bind::KvAppend::PositionPtr,
                          io(IoSlot::Position));
                break;
            }
            case Kind::Sdpa: {
                const auto& kv = b.kv[std::size_t(L)];
                bind_slot(ctx, ord, (std::uint8_t)bind::Sdpa::K, kv.k);
                bind_slot(ctx, ord, (std::uint8_t)bind::Sdpa::V, kv.v);
                bind_slot(ctx, ord, (std::uint8_t)bind::Sdpa::N, io(IoSlot::SeqLen));
                break;
            }
            // Declared by the shared routed template, never read at
            // BIASED=false. See `BoundLlama::zero_bias`.
            case Kind::ExpertGate:
            case Kind::ExpertUp:
            case Kind::ExpertDown:
                bind_slot(ctx, ord, (std::uint8_t)bind::GoQmv::Bias, b.zero_bias);
                break;
            case Kind::RowGather:
                bind_slot(ctx, ord, (std::uint8_t)bind::RowGather::Rows, io(IoSlot::SampleRows));
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
                    bind_slot(ctx, ord, sb.bind_index, b.pool[std::size_t(sb.color)]);
                }
            }
        }
    }
}

}  // namespace pie::metal::llama
