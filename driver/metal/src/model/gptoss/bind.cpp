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

int router_bits_from_extents(std::uint64_t weight_bytes, std::uint64_t scale_bytes) {
    if (weight_bytes == 0 || scale_bytes == 0) return 0;
    const std::uint64_t denom = 4 * scale_bytes;
    if (weight_bytes % denom != 0) return 0;
    const std::uint64_t bits = weight_bytes / denom;
    return (bits == 4 || bits == 8) ? int(bits) : 0;
}

bool mxfp4_experts_from_weights(const std::unordered_map<std::string, SlotHandle>& weights) {
    // The absence of a zero point is what distinguishes the two 4-bit formats,
    // and it is the same rule the contract used when it decided not to decode.
    return weights.count("layers.0.mlp.experts.gate_proj.weight") != 0 &&
           weights.count("layers.0.mlp.experts.gate_proj.biases") == 0;
}

int router_bits_from_weights(const std::unordered_map<std::string, SlotHandle>& weights) {
    const auto w = weights.find("layers.0.mlp.router.weight");
    const auto s = weights.find("layers.0.mlp.router.scales");
    if (w == weights.end() || s == weights.end()) return 0;
    return router_bits_from_extents(w->second.size, s->second.size);
}

int proj_bits_from_weights(const std::unordered_map<std::string, SlotHandle>& weights) {
    const auto w = weights.find("layers.0.self_attn.q_proj.weight");
    const auto s = weights.find("layers.0.self_attn.q_proj.scales");
    if (w == weights.end() || s == weights.end()) return 0;
    return router_bits_from_extents(w->second.size, s->second.size);
}

ScratchColoring color_gptoss_scratch(const std::vector<Dispatch>& dag,
                                    const ScratchPlan& plan, bool no_recycle) {
    return model::color_family_scratch(dag.size(), plan.uses, gptoss_run_ends(dag),
                                       plan.value_count, no_recycle);
}

void bind_gptoss_dag(RawMetalContext& ctx, const BoundGptOss& b, const std::vector<Dispatch>& dag,
                     const GptOssGeometry& g, const ScratchColoring& scratch, int ordinal_base) {
    auto io = [&](IoSlot s) -> const SlotHandle& { return b.io[static_cast<int>(s)]; };

    DecodeGeometry wg{};
    wg.mxfp4_experts = g.mxfp4_experts;

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        // (a) Weights.
        for (const WeightBind& wb : weight_binds(shared_kind(d.kind), L, wg, false)) {
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

    DecodeGeometry wg{};
    wg.mxfp4_experts = g.mxfp4_experts;

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        const int ord = ordinal_base + d.ordinal;
        const int L = d.layer;

        for (const WeightBind& wb : weight_binds(shared_kind(d.kind), L, wg, false)) {
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
