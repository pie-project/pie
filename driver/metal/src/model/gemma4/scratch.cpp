// Gemma 4's activation dataflow.
//
// The same job `batch/scratch.cpp` does for qwen3.5: walk the DAG in order and
// say, for every dispatch, which activation value each of its buffers reads or
// writes. Colouring those live ranges onto a pool is the shared half and is
// reused; naming the values is the family's, and gemma4's differ in three ways:
//
//  * **A norm sandwich.** Attention and the FFN are each wrapped: normalise the
//    input, run the block, normalise the OUTPUT, then add the residual. So a
//    block's result is normalised before it rejoins the stream, which qwen3.5
//    never does.
//  * **Per-Layer Embeddings.** A second stream, computed once before the stack
//    and consumed a slice at a time, which no other family here has.
//  * **KV-shared layers read pages they did not write.** Those layers emit no
//    k/v projection at all, so the dataflow simply has nothing to thread.

#include "scratch.hpp"

#include "decode_step.hpp"
#include "../shared_kernels.hpp"

namespace pie::metal::gemma4 {

namespace {
int moe_pairs(const Gemma4Geometry& g, int rows) {
    return (rows < 1 ? 1 : rows) * (g.experts_per_token > 0 ? g.experts_per_token : 1);
}
}  // namespace

int gemma4_moe_sorted_rows(const Gemma4Geometry& g, int rows) {
    if (!g.is_moe()) return rows < 1 ? 1 : rows;
    return shared_kernels::moe_sorted_rows(moe_pairs(g, rows), g.n_experts);
}

int gemma4_moe_tile_rows(const Gemma4Geometry& g, int rows) {
    if (!g.is_moe()) return 1;
    return shared_kernels::moe_tile_rows(moe_pairs(g, rows), g.n_experts);
}

namespace bi {
// The bind indices this dataflow touches, named so the walk below reads as
// dataflow rather than as arithmetic.
constexpr std::uint8_t RmsX = 0, RmsOut = 2;
constexpr std::uint8_t QmvX = 3, QmvOut = 4;
constexpr std::uint8_t EmbedOut = 4;
constexpr std::uint8_t RopeX = 0;
constexpr std::uint8_t SdpaQ = 0, SdpaOut = 3;
constexpr std::uint8_t KvAppendK = 0, KvAppendV = 1;
constexpr std::uint8_t GegluGate = 0, GegluUp = 1, GegluOut = 2;
constexpr std::uint8_t ResidX = 0, ResidR = 1, ResidOut = 2;
// The fused norm+residual keeps bind::Rms's prefix and appends the residual.
constexpr std::uint8_t RRX = 0, RROut = 2, RRResid = 4;
constexpr std::uint8_t VNormX = 0, VNormOut = 1;
constexpr std::uint8_t ScalarX = 0, ScalarOut = 2;
constexpr std::uint8_t CombineProj = 0, CombineToken = 1, CombineOut = 2;
constexpr std::uint8_t SoftcapIn = 0, SoftcapOut = 1;
constexpr std::uint8_t GatherIn = 0, GatherOut = 1;
// The mixture. The ROUTED matvec shares the dense one's activation binds -- it
// is the same kernel with the weight stack indexed by `expert_ids` -- and adds
// two index inputs. `QmvTileExpert` sits clear of the matvec's eleven slots
// because one ordinal serves both pipelines; see `bind::GoQmv`.
constexpr std::uint8_t QmvExpertIds = 8, QmvTileExpert = 12;
constexpr std::uint8_t RouterLogits = 0, RouterIds = 1, RouterWeights = 2;
constexpr std::uint8_t SortIds = 0, SortPerm = 1, SortRowExpert = 2, SortTileExpert = 3,
                       SortInv = 5;
constexpr std::uint8_t RowsIn = 0, RowsOut = 1, RowsPerm = 2;
// `CombineOut` above is `PleCombine`'s; this is the mixture's, which is a
// different kernel with a different signature.
constexpr std::uint8_t CombineY = 0, CombineWeights = 1, CombineMoeOut = 2, CombineInv = 4;
}  // namespace bi

ScratchPlan build_gemma4_scratch(const std::vector<Dispatch>& dag, const Gemma4Geometry& g) {
    ScratchPlan plan;
    int next_value = 0;
    auto fresh = [&] { return next_value++; };
    auto rd = [&](int ord, std::uint8_t b, int v) { plan.uses.push_back({ord, b, v, false}); };
    auto wr = [&](int ord, std::uint8_t b, int v) { plan.uses.push_back({ord, b, v, true}); };

    // The residual stream, and the temporaries that hang off it.
    int resid = -1;    // the stream itself
    int normed = -1;   // the norm output currently feeding projections
    int q = -1, kk = -1, vv = -1, attn = -1;
    int gp = -1, up = -1, act = -1;
    int block = -1;    // a block's output, between its post-norm and its residual add
    // PLE: one table for the whole stack, plus this layer's slice.
    int ple_tok = -1, ple_proj = -1, ple = -1;
    int ple_gate = -1, ple_act = -1, ple_back = -1;
    // The mixture's, all -1 on a dense model.
    int dense_br = -1, moe_br = -1, router_normed = -1, moe_normed = -1;
    int router_logits = -1, expert_ids = -1, expert_weights = -1;
    int perm = -1, row_expert = -1, tile_expert = -1, inv = -1;
    int sorted_x = -1, sorted_out = -1;

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

            // ── PLE precompute, once ──
            case Kind::PleTokenGather:
                ple_tok = fresh();
                wr(o, bi::EmbedOut, ple_tok);
                break;
            case Kind::PleProjGemv:
                ple_proj = fresh();
                rd(o, bi::QmvX, resid);
                wr(o, bi::QmvOut, ple_proj);
                break;
            case Kind::PleProjNorm:
                rd(o, bi::RmsX, ple_proj);
                wr(o, bi::RmsOut, ple_proj);
                break;
            case Kind::PleCombine:
                ple = fresh();
                rd(o, bi::CombineProj, ple_proj);
                rd(o, bi::CombineToken, ple_tok);
                wr(o, bi::CombineOut, ple);
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
            case Kind::QNorm:
                rd(o, bi::RmsX, q);
                wr(o, bi::RmsOut, q);
                break;
            case Kind::KNorm:
                rd(o, bi::RmsX, kk);
                wr(o, bi::RmsOut, kk);
                break;
            case Kind::VNorm:
                // Weightless, and in place: V is normalised on its way to the cache.
                rd(o, bi::VNormX, vv);
                wr(o, bi::VNormOut, vv);
                break;
            case Kind::VNormFromK:
                // The layer projected no V. It reads the K projection -- before
                // `KNorm` rewrites it -- and writes a V of its own.
                vv = fresh();
                rd(o, bi::VNormX, kk);
                wr(o, bi::VNormOut, vv);
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
            case Kind::PostAttnResidual: {
                // The sandwich's second half and the add it always precedes:
                // normalise the BLOCK's output, then rejoin the stream.
                const int next = fresh();
                rd(o, bi::RRX, block);
                rd(o, bi::RRResid, resid);
                wr(o, bi::RROut, next);
                resid = next;
                break;
            }

            // ── FFN ──
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
            case Kind::GegluTanh:
                act = fresh();
                rd(o, bi::GegluGate, gp);
                rd(o, bi::GegluUp, up);
                wr(o, bi::GegluOut, act);
                break;
            case Kind::QmvDown:
                block = fresh();
                rd(o, bi::QmvX, act);
                wr(o, bi::QmvOut, block);
                break;
            // ── the mixture, beside the dense branch ──
            //
            // `resid` here is the POST-ATTENTION residual: mlx-lm routes on it,
            // and both branches read it. `block` carries the dense branch's
            // output through to `BranchAdd`, which is why `dense_br` is a
            // separate name -- the routed branch reuses `block` for its own.
            case Kind::DenseBranchNorm:
                dense_br = fresh();
                rd(o, bi::RmsX, block);
                wr(o, bi::RmsOut, dense_br);
                break;
            case Kind::RouterNorm:
                router_normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, router_normed);
                break;
            case Kind::RouterGemv:
                router_logits = fresh();
                rd(o, bi::QmvX, router_normed);
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
            case Kind::MoeNorm:
                moe_normed = fresh();
                rd(o, bi::RmsX, resid);
                wr(o, bi::RmsOut, moe_normed);
                break;
            case Kind::ExpertSort:
                // Four outputs: the permutation, the per-row expert the matvec
                // form reads, the per-tile expert the matmul form reads, and
                // the inverse the combine reads. All live to the end of the run.
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
                rd(o, bi::RowsIn, moe_normed);
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
            case Kind::ExpertGeglu:
                // Elementwise over the sorted stack. The slot axis is gone -- a
                // sorted row IS a slot -- so this is the dense kernel over a
                // taller batch rather than a strided one.
                act = fresh();
                rd(o, bi::GegluGate, gp);
                rd(o, bi::GegluUp, up);
                wr(o, bi::GegluOut, act);
                break;
            case Kind::ExpertDown:
                sorted_out = fresh();
                rd(o, bi::QmvX, act);
                rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert);
                wr(o, bi::QmvOut, sorted_out);
                break;
            case Kind::ExpertCombine: {
                // Reads the results WHERE THE SORT LEFT THEM, through its
                // inverse: the sort knows both directions at the moment it
                // places a pair.
                const int combined = fresh();
                rd(o, bi::CombineY, sorted_out);
                rd(o, bi::CombineWeights, expert_weights);
                rd(o, bi::CombineInv, inv);
                wr(o, bi::CombineMoeOut, combined);
                moe_br = combined;
                break;
            }
            case Kind::MoeBranchNorm: {
                const int normed_moe = fresh();
                rd(o, bi::RmsX, moe_br);
                wr(o, bi::RmsOut, normed_moe);
                moe_br = normed_moe;
                break;
            }
            case Kind::BranchAdd:
                // The two branches meet. `block` from here is what the
                // sandwich's second half normalises, exactly as on a dense
                // layer -- which is why `PostFfnResidual` needs no mixture case.
                block = fresh();
                rd(o, bi::ResidX, dense_br);
                rd(o, bi::ResidR, moe_br);
                wr(o, bi::ResidOut, block);
                break;

            case Kind::PostFfnResidual: {
                const int next = fresh();
                rd(o, bi::RRX, block);
                rd(o, bi::RRResid, resid);
                wr(o, bi::RROut, next);
                resid = next;
                break;
            }

            // ── per-layer embedding residual ──
            case Kind::PleGateGemv:
                ple_gate = fresh();
                rd(o, bi::QmvX, resid);
                wr(o, bi::QmvOut, ple_gate);
                break;
            case Kind::PleGeglu:
                // Gated by the stream, valued by this layer's slice of the table.
                ple_act = fresh();
                rd(o, bi::GegluGate, ple_gate);
                rd(o, bi::GegluUp, ple);
                wr(o, bi::GegluOut, ple_act);
                break;
            case Kind::PleProjLayerGemv:
                ple_back = fresh();
                rd(o, bi::QmvX, ple_act);
                wr(o, bi::QmvOut, ple_back);
                break;
            case Kind::PleResidualScaled: {
                const int next = fresh();
                rd(o, bi::RRX, ple_back);
                rd(o, bi::RRResid, resid);
                wr(o, bi::RROut, next);
                resid = next;
                break;
            }
            case Kind::LayerScalar: {
                const int next = fresh();
                rd(o, bi::ScalarX, resid);
                wr(o, bi::ScalarOut, next);
                resid = next;
                break;
            }

            // ── tail ──
            // The sampled rows, compacted: everything after this is [S, *].
            // It writes a value of its own rather than in place, because its
            // input is the last layer's output at [N, hidden] and its output is
            // a DIFFERENT shape -- aliasing them would make the pool's slot
            // sizing a lie.
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
            case Kind::FinalSoftcap:
                rd(o, bi::SoftcapIn, plan.logits_value);
                wr(o, bi::SoftcapOut, plan.logits_value);
                break;
            case Kind::Argmax:
                rd(o, 0, plan.logits_value);
                break;
        }
    }
    plan.value_count = next_value;
    (void)g;
    return plan;
}

std::vector<ValueExtent> gemma4_value_extents(const std::vector<Dispatch>& dag,
                                              const ScratchPlan& plan,
                                              const Gemma4Geometry& g) {
    std::vector<ValueExtent> ext(std::size_t(plan.value_count));
    // Attribute each value to the dispatch that WROTE it: a value's shape is
    // its producer's output shape. Reading it off a consumer would be ambiguous
    // for anything read twice, and the routed stack is read three times.
    for (const Use& u : plan.uses) {
        if (!u.is_write) continue;
        const Dispatch& d = dag[std::size_t(u.index)];
        const int L = d.layer;
        const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;
        const int inter = L >= 0 ? g.intermediate_of(L) : g.intermediate;
        ValueExtent e;
        switch (d.kind) {
            case Kind::EmbedGather:
            case Kind::AttnNorm:
            case Kind::FfnNorm:
            case Kind::FinalRms:
            case Kind::RowGather:
            case Kind::QmvO:
            case Kind::QmvDown:
            case Kind::PostAttnResidual:
            case Kind::PostFfnResidual:
            case Kind::PleResidualScaled:
            case Kind::PleProjLayerGemv:
            case Kind::LayerScalar:
            case Kind::RouterNorm:
            case Kind::MoeNorm:
            case Kind::DenseBranchNorm:
            case Kind::MoeBranchNorm:
            case Kind::BranchAdd:
            case Kind::ExpertCombine:  // the k slots collapse here
                e.elems = g.hidden;
                break;
            case Kind::QmvQ:
            case Kind::QNorm:
            case Kind::RopeQ:
            case Kind::Sdpa:
                e.elems = g.n_q_heads * hd;
                break;
            case Kind::QmvK:
            case Kind::QmvV:
            case Kind::KNorm:
            case Kind::VNorm:
            case Kind::VNormFromK:
            case Kind::RopeK:
                e.elems = g.n_kv_heads_of(L) * hd;
                break;
            case Kind::QmvGate:
            case Kind::QmvUp:
            case Kind::GegluTanh:
                e.elems = inter;
                break;
            case Kind::PleTokenGather:
            case Kind::PleProjGemv:
            case Kind::PleProjNorm:
            case Kind::PleCombine:
                e.elems = g.n_layers * g.per_layer_emb_dim;
                break;
            case Kind::PleGateGemv:
            case Kind::PleGeglu:
                e.elems = g.per_layer_emb_dim;
                break;
            case Kind::RouterGemv:
                e.elems = g.n_experts;
                break;
            case Kind::RouterTopK:
                // Both outputs are k-wide and they are not the same width: the
                // weights are activations, the ids are int32, and an int32 is
                // TWO of this pool's elements -- the pool is measured in
                // activation elements and allocated at two bytes each. One kind
                // produces both, so the claim is the wider and the weights ride
                // along at twice their need.
                e.elems = g.experts_per_token * 2;
                break;
            case Kind::ExpertSort:
                // Four int32 outputs; the tallest is one entry per sorted row.
                e.elems = 2;
                e.rows_are_sorted = 1;
                break;
            case Kind::ExpertGather:
            case Kind::ExpertDown:
                e.elems = g.hidden;
                e.rows_are_sorted = 1;
                break;
            case Kind::ExpertGate:
            case Kind::ExpertUp:
            case Kind::ExpertGeglu:
                e.elems = g.moe_intermediate;
                e.rows_are_sorted = 1;
                break;
            case Kind::LmHead:
            case Kind::FinalSoftcap:
                e.elems = g.vocab;
                break;
            case Kind::KvAppend:
            case Kind::Argmax:
                break;
        }
        // A value written more than once -- the ropes and the qk-norms write in
        // place -- keeps the widest claim.
        ValueExtent& slot = ext[std::size_t(u.value)];
        if (e.elems > slot.elems) slot.elems = e.elems;
        if (e.rows_are_sorted != 0) slot.rows_are_sorted = 1;
    }
    return ext;
}

}  // namespace pie::metal::gemma4
