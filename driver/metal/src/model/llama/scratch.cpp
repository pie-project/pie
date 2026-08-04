// The llama families' activation dataflow.
//
// The same job `gemma4/scratch.cpp` does for gemma4: walk the DAG in order and
// say, for every dispatch, which activation value each buffer reads or writes.
// Colouring those live ranges onto a pool is the shared half and is reused.
//
// llama is the plain case, and that is worth saying plainly: a pre-norm
// residual stream, `x += block(norm(x))` twice per layer, with nothing
// normalised on the way out and no second stream. Everything interesting here
// is the routed FFN, and it is interesting for one reason -- it is the only
// place in any family where a single dispatch writes `experts_per_token`
// results per row instead of one. The expert activations are [rows, k, width],
// and a pool that sized them like the dense ones would overrun by a factor of
// k. `llama_value_extents` exists to say so.
//
// The routing decision is the other thing this file has to get right. Dense
// temporaries die at their next consumer, so a colouring pass can recycle them
// aggressively. `expert_ids` and `expert_weights` do not: they are written by
// `RouterTopK` and still read by `ExpertCombine`, five dispatches later, with
// three matvecs in between allocating freely. They are tracked as ordinary
// values with honest live ranges so the shared colouring sees the extent rather
// than being told about it.

#include "scratch.hpp"

namespace pie::metal::llama {

namespace bi {
// The bind indices this dataflow touches, named so the walk below reads as
// dataflow rather than as arithmetic.
constexpr std::uint8_t RmsX = 0, RmsOut = 2;
// The quantized matvec's activation in and out. The ROUTED matvec shares them:
// it is the same kernel with the weight stack indexed by `expert_ids`, so the
// activation binds do not move.
constexpr std::uint8_t QmvX = 3, QmvOut = 4;
constexpr std::uint8_t QmvExpertIds = 8;
constexpr std::uint8_t EmbedOut = 4;
constexpr std::uint8_t RopeX = 0;
constexpr std::uint8_t SdpaQ = 0, SdpaOut = 3;
constexpr std::uint8_t KvAppendK = 0, KvAppendV = 1;
constexpr std::uint8_t SiluGate = 0, SiluUp = 1, SiluOut = 2;
constexpr std::uint8_t ResidX = 0, ResidR = 1, ResidOut = 2;
constexpr std::uint8_t RouterLogits = 0, RouterIds = 1, RouterWeights = 2;
constexpr std::uint8_t CombineY = 0, CombineWeights = 1, CombineOut = 2, CombineInv = 4;
constexpr std::uint8_t GatherIn = 0, GatherOut = 1;
// The batched mixture's reordering. `QmvTileExpert` sits clear of the matvec's
// eleven slots because one ordinal serves both pipelines; see `bind::GoQmv`.
constexpr std::uint8_t QmvTileExpert = 12;
constexpr std::uint8_t SortIds = 0, SortPerm = kMoeSortPermBind, SortRowExpert = 2,
                       SortTileExpert = 3, SortInv = 5;
constexpr std::uint8_t RowsIn = 0, RowsOut = 1, RowsPerm = 2;
}  // namespace bi

ScratchPlan build_llama_scratch(const std::vector<Dispatch>& dag, const LlamaGeometry& g) {
    ScratchPlan plan;
    int next_value = 0;
    auto fresh = [&] { return next_value++; };
    auto rd = [&](int o, std::uint8_t b, int v) { plan.uses.push_back({o, b, v, false}); };
    auto wr = [&](int o, std::uint8_t b, int v) { plan.uses.push_back({o, b, v, true}); };

    // The residual stream, and the temporaries that hang off it.
    int resid = -1;
    int normed = -1;
    int q = -1, kk = -1, vv = -1, attn = -1;
    int gp = -1, up = -1, act = -1;
    int block = -1;  // a block's output, between its projection and its residual add
    int router_logits = -1, expert_ids = -1, expert_weights = -1;
    int perm = -1, row_expert = -1, tile_expert = -1, inv = -1, sorted_x = -1, sorted_out = -1;

    for (std::size_t di = 0; di < dag.size(); ++di) {
        const Dispatch& d = dag[di];
        // Position, not `d.ordinal`: the prefill DAG shifts its ordinals clear
        // of the decode path's, and this is a time axis.
        const int o = static_cast<int>(di);
        switch (d.kind) {
            case Kind::EmbedGather:
                resid = fresh();
                wr(o, bi::EmbedOut, resid);
                break;

            // ── attention ──
            case Kind::AttnNorm:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::QmvQ:
                q = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, q);
                break;
            case Kind::QmvK:
                kk = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, kk);
                break;
            case Kind::QmvV:
                vv = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, vv);
                break;
            // Qwen3's per-head norms, in place: they rescale q and k where they
            // already are, before the rotation reads them.
            case Kind::QNorm:
                rd(o, bi::RmsX, q);
                wr(o, bi::RmsOut, q);
                break;
            case Kind::KNorm:
                rd(o, bi::RmsX, kk);
                wr(o, bi::RmsOut, kk);
                break;
            case Kind::RopeQ:
                rd(o, bi::RopeX, q);
                wr(o, bi::RopeX, q);
                break;
            case Kind::RopeK:
                rd(o, bi::RopeX, kk);
                wr(o, bi::RopeX, kk);
                break;
            case Kind::KvAppend:
                // Writes the cache, which the pool does not own -- so this
                // consumes k and v and produces nothing colourable. It is also
                // what ENDS their live ranges, which is why it is recorded at
                // all rather than skipped.
                rd(o, bi::KvAppendK, kk);
                rd(o, bi::KvAppendV, vv);
                break;
            case Kind::Sdpa:
                attn = fresh();
                rd(o, bi::SdpaQ, q);
                wr(o, bi::SdpaOut, attn);
                break;
            case Kind::QmvO:
                block = fresh();
                rd(o, bi::QmvX, attn);
                wr(o, bi::QmvOut, block);
                break;
            case Kind::AttnResidual: {
                const int next = fresh();
                rd(o, bi::ResidX, block);
                rd(o, bi::ResidR, resid);
                wr(o, bi::ResidOut, next);
                resid = next;
                break;
            }

            // ── FFN, dense ──
            case Kind::FfnNorm:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::QmvGate:
                gp = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, gp);
                break;
            case Kind::QmvUp:
                up = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, up);
                break;
            case Kind::SiluMul:
                act = fresh();
                rd(o, bi::SiluGate, gp);
                rd(o, bi::SiluUp, up);
                wr(o, bi::SiluOut, act);
                break;
            case Kind::QmvDown:
                block = fresh();
                rd(o, bi::QmvX, act);
                wr(o, bi::QmvOut, block);
                break;

            // ── FFN, routed ──
            case Kind::Router:
                router_logits = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, router_logits);
                break;
            case Kind::RouterTopK:
                // Two outputs, and both outlive the three matvecs that follow.
                expert_ids = fresh();
                plan.expert_ids_by_layer.push_back(expert_ids);
                expert_weights = fresh();
                plan.expert_weights_by_layer.push_back(expert_weights);
                rd(o, bi::RouterLogits, router_logits);
                wr(o, bi::RouterIds, expert_ids);
                wr(o, bi::RouterWeights, expert_weights);
                break;
            case Kind::ExpertSort:
                // Four outputs: the permutation, the per-row expert the matvec
                // form reads, the per-tile expert the matmul form reads, and
                // the inverse the combine reads. All live to the end of the
                // routed run.
                perm = fresh();
                row_expert = fresh();
                tile_expert = fresh();
                inv = fresh();
                rd(o, bi::SortIds, expert_ids);
                wr(o, bi::SortPerm, perm);
                wr(o, bi::SortRowExpert, row_expert);
                wr(o, bi::SortTileExpert, tile_expert);
                wr(o, bi::SortInv, inv);
                break;
            case Kind::ExpertGather:
                sorted_x = fresh();
                rd(o, bi::RowsIn, normed);
                wr(o, bi::RowsOut, sorted_x);
                rd(o, bi::RowsPerm, perm);
                break;
            case Kind::ExpertGate:
                gp = fresh();
                rd(o, bi::QmvX, sorted_x);
                rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert);
                wr(o, bi::QmvOut, gp);
                break;
            case Kind::ExpertUp:
                up = fresh();
                rd(o, bi::QmvX, sorted_x);
                rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert);
                wr(o, bi::QmvOut, up);
                break;
            case Kind::ExpertSiluMul:
                // Elementwise over the sorted stack. The slot axis is gone --
                // a sorted row IS a slot -- so this is the dense kernel over a
                // taller batch rather than a strided one.
                act = fresh();
                rd(o, bi::SiluGate, gp);
                rd(o, bi::SiluUp, up);
                wr(o, bi::SiluOut, act);
                break;
            case Kind::ExpertDown:
                // Still sorted: the k results per token are only brought back
                // together by the scatter, and only summed by ExpertCombine.
                sorted_out = fresh();
                rd(o, bi::QmvX, act);
                rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert);
                wr(o, bi::QmvOut, sorted_out);
                break;
            case Kind::ExpertCombine: {
                // Reads the results WHERE THE SORT LEFT THEM, through its
                // inverse. The sort knows both directions at the moment it
                // places a pair, so undoing it with a kernel of its own would
                // be a dispatch and a full-width buffer spent to feed a step
                // that is about to gather anyway.
                const int combined = fresh();
                rd(o, bi::CombineY, sorted_out);
                rd(o, bi::CombineWeights, expert_weights);
                rd(o, bi::CombineInv, inv);
                wr(o, bi::CombineOut, combined);
                block = combined;
                break;
            }

            case Kind::FfnResidual: {
                const int next = fresh();
                rd(o, bi::ResidX, block);
                rd(o, bi::ResidR, resid);
                wr(o, bi::ResidOut, next);
                resid = next;
                break;
            }

            // ── tail ──
            // The sampled rows, compacted: everything after this is [S, *]. It
            // writes a value of its own rather than in place, because its input
            // is [N, hidden] and its output is a DIFFERENT shape -- aliasing
            // them would make the pool's slot sizing a lie.
            case Kind::RowGather: {
                const int gathered = fresh();
                rd(o, bi::GatherIn, resid);
                wr(o, bi::GatherOut, gathered);
                resid = gathered;
                break;
            }
            case Kind::FinalRms:
                normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, normed);
                break;
            case Kind::LmHead: {
                const int logits = fresh();
                rd(o, bi::QmvX, normed);
                wr(o, bi::QmvOut, logits);
                plan.logits_value = logits;
                break;
            }
            case Kind::Argmax:
                rd(o, 0, plan.logits_value);
                break;
        }
    }
    plan.value_count = next_value;
    return plan;
}

int llama_moe_sorted_rows(const LlamaGeometry& g, int rows) {
    if (!g.is_moe()) return rows < 1 ? 1 : rows;
    const int n = (rows < 1 ? 1 : rows) * g.experts_per_token;
    return shared_kernels::moe_sorted_rows(n, g.n_experts);
}

std::vector<ValueExtent> llama_value_extents(const std::vector<Dispatch>& dag,
                                             const ScratchPlan& plan, const LlamaGeometry& g) {
    std::vector<ValueExtent> ext(std::size_t(plan.value_count));

    // Attribute each value to the dispatch that WROTE it: a value's shape is
    // its producer's output shape, and every value here has exactly one
    // producer. Reading the extent off the consumer instead would be ambiguous
    // for anything read twice.
    for (const Use& u : plan.uses) {
        if (!u.is_write) continue;
        const Kind k = dag[std::size_t(u.index)].kind;
        ValueExtent e;
        switch (k) {
            case Kind::EmbedGather:
            case Kind::AttnNorm:
            case Kind::FfnNorm:
            case Kind::FinalRms:
            case Kind::AttnResidual:
            case Kind::FfnResidual:
            case Kind::RowGather:
            case Kind::QmvO:
                e.elems = g.hidden;
                break;
            case Kind::QmvQ:
            case Kind::QNorm:
            case Kind::Sdpa:
                e.elems = g.q_width();
                break;
            case Kind::QmvK:
            case Kind::QmvV:
            case Kind::KNorm:
                e.elems = g.kv_width();
                break;
            case Kind::RopeQ:
                e.elems = g.q_width();
                break;
            case Kind::RopeK:
                e.elems = g.kv_width();
                break;
            case Kind::QmvGate:
            case Kind::QmvUp:
            case Kind::SiluMul:
                e.elems = g.intermediate;
                break;
            case Kind::QmvDown:
                e.elems = g.hidden;
                break;
            case Kind::Router:
                e.elems = g.n_experts;
                break;
            case Kind::RouterTopK:
                // Both outputs are k-wide, but they are not the same width:
                // the weights are activations and the ids are `int32`, which is
                // TWO of this pool's elements rather than one. The pool is
                // measured in activation elements and allocated at two bytes
                // each, so sizing the ids like the weights hands `router_topk`
                // a buffer half the length it writes -- 16 bytes for a k=8
                // model against the 32 it stores. Nothing here can tell the two
                // values apart (one kind produces both), so the claim is the
                // wider of them and the weights ride along at twice their need.
                // That is k extra elements per routed layer.
                e.elems = g.experts_per_token * 2;
                break;
            case Kind::ExpertSort:
                // Four int32 outputs. Two are one entry per sorted row, one is
                // per TILE and one is per PAIR, neither of which is ever more,
                // so all four are sized as the tallest. An int32 is TWO of this pool's
                // elements -- the pool is measured in activation elements and
                // allocated at two bytes each -- which is the same trap
                // `RouterTopK` documents above.
                e.elems = 2;
                e.rows_are_sorted = 1;
                break;
            case Kind::ExpertGather:
                e.elems = g.hidden;
                e.rows_are_sorted = 1;
                break;
            case Kind::ExpertGate:
            case Kind::ExpertUp:
            case Kind::ExpertSiluMul:
                e.elems = g.moe_intermediate;
                e.rows_are_sorted = 1;
                break;
            case Kind::ExpertDown:
                e.elems = g.hidden;
                e.rows_are_sorted = 1;
                break;
            case Kind::ExpertCombine:
                // The k slots collapse here: back to one row's worth.
                e.elems = g.hidden;
                break;
            case Kind::LmHead:
                e.elems = g.vocab;
                break;
            case Kind::Argmax:
            case Kind::KvAppend:
                break;
        }
        // A value written more than once (rope and the qk-norms write in place)
        // keeps the widest claim, so an in-place pass can never shrink a slot
        // under the value already living in it.
        ValueExtent& slot = ext[std::size_t(u.value)];
        if (e.elems > slot.elems) slot.elems = e.elems;
        if (e.rows_are_slots != 0) slot.rows_are_slots = 1;
        if (e.rows_are_sorted != 0) slot.rows_are_sorted = 1;
    }
    return ext;
}

/// How many elements each pool colour must hold.
///
/// Unlike the other families' equivalents, this reads the extents off the
/// VALUES rather than off the dispatch kinds, because for this family the two are not the same
/// question. A routed layer's expert stack is `experts_per_token` times taller
/// than the dense tensor it shares a colour with, and a colour is sized by the
/// widest value in it -- so sizing by kind would under-allocate the stack by a
/// factor of k and let the last expert's projection run off the end of the
/// buffer. `llama_value_extents` says which values those are; `color_of_value`
/// says where they landed.
std::vector<std::size_t> llama_pool_elems(const std::vector<Dispatch>& dag,
                                         const ScratchPlan& plan,
                                         const model::ScratchColoring& col,
                                         const LlamaGeometry& g, int rows, int head_rows) {
    const std::vector<ValueExtent> ext = llama_value_extents(dag, plan, g);
    const int k = g.is_moe() ? g.experts_per_token : 1;
    std::vector<std::size_t> elems(std::size_t(col.colors_used), 0);
    for (const Use& u : plan.uses) {
        if (!u.is_write) continue;
        if (u.value < 0 || std::size_t(u.value) >= col.color_of_value.size()) continue;
        const int c = col.color_of_value[std::size_t(u.value)];
        if (c < 0 || c >= col.colors_used) continue;
        // The tail's tensors have one row per SAMPLED row; the body's one per
        // token; the expert stack k per token.
        const bool tail = is_tail(dag[std::size_t(u.index)].kind);
        const ValueExtent& e = ext[std::size_t(u.value)];
        const int n = e.rows_are_sorted != 0 ? llama_moe_sorted_rows(g, rows)
                      : e.rows_are_slots != 0 ? rows * k
                      : (tail ? head_rows : rows);
        const std::size_t need = std::size_t(n) * std::size_t(e.elems);
        elems[std::size_t(c)] = std::max(elems[std::size_t(c)], need);
    }
    for (std::size_t& e : elems) {
        if (e == 0) e = std::size_t(rows) * std::size_t(g.hidden);
    }
    return elems;
}

}  // namespace pie::metal::llama
