// scratch_schedule.cpp — derive the decode DAG's activation dataflow and linear-scan
// color the live ranges onto the SCRATCH_POOL ping-pong buffers (beta's WAR/WAW lane).
//
// The dataflow below mirrors build_decode_dag's fixed order exactly. Each logical
// activation tensor is an SSA "value"; in-place kernels (q_norm/k_norm/rope/attn_gate)
// read+write the SAME value (one buffer). We compute each value's live interval
// [first_def_ordinal, last_use_ordinal] and greedily color: two values that overlap
// (inclusive endpoints — so a same-dispatch WAR and any concurrent ‖-pair output always
// interfere) get distinct buffers. The result is hazard-free by construction.
//
// Slots delta owns (weights, GDN conv/recurrent state, KV pages, IO scalars/logits) are
// NOT modeled here — only the activation X/Out scratch this lane binds.

#include "scratch.hpp"

#include "model/llama/scratch.hpp"

#include "scratch_color.hpp"

#include <algorithm>

namespace pie::metal {

namespace {

// A single scratch slot a dispatch touches.
struct Use {
    int     ordinal;
    uint8_t bind_index;
    int     value;
    bool    write;
};

// bind-index constants (from decode_abi.hpp bind:: enums) for the activation slots.
namespace bi {
constexpr uint8_t EmbedOut   = 4;             // bind::Embed::Out
constexpr uint8_t RmsX = 0, RmsOut = 2;       // bind::Rms
constexpr uint8_t QmvX = 3, QmvOut = 4, QmvResidual = 7;  // bind::Qmv (Residual = fused epilogue)
constexpr uint8_t QSplitIn = 0, QSplitQ = 1, QSplitGate = 2;  // bind::QSplit
constexpr uint8_t RopeX = 0;                  // bind::Rope: in-place activation (1=pos IO, 2/3/4=consts)
constexpr uint8_t SdpaQ = 0, SdpaOut = 3;     // bind::Sdpa (K/V from KV region)
constexpr uint8_t AttnGateAttn = 0, AttnGateGate = 1;  // bind::AttnGate (in-place)
constexpr uint8_t KvAppendK = 0, KvAppendV = 1;        // bind::KvAppend (out -> KV pages)
constexpr uint8_t GdnMixed = 0, GdnCoreOut = 3, GdnAGate = 8, GdnBGate = 9;  // bind::GdnCore
// Prep-dispatch split (PIE_GDN_PREP): GdnPrep scratch binds + GdnCore-recurrent scratch binds.
constexpr uint8_t GdnPrepMixed = 0, GdnPrepAGate = 6, GdnPrepBGate = 7,
                  GdnPrepPreQ = 8, GdnPrepPreK = 9, GdnPrepPreGate = 10;     // bind::GdnPrep
constexpr uint8_t GdnRecMixed = 0, GdnRecCoreOut = 3,
                  GdnRecPreQ = 6, GdnRecPreK = 7, GdnRecPreGate = 8;         // bind::GdnCoreRecurrent
constexpr uint8_t GatedRmsX = 0, GatedRmsZ = 1, GatedRmsOut = 3;  // bind::GatedRms
constexpr uint8_t ResidX = 0, ResidR = 1, ResidOut = 2;  // bind::Residual
constexpr uint8_t SiluGate = 0, SiluUp = 1, SiluOut = 2;  // bind::SiluMul
// ── the routed FFN ──
// The routed matvec shares the dense one's activation binds: it is the same
// kernel with the weight stack indexed per row, so only the routing slots are
// new. `QmvTileExpert` sits clear of the matvec's eleven because one ordinal
// serves both the matvec and the matmul pipeline; see `bind::GoQmv`.
constexpr uint8_t QmvExpertIds = 8, QmvTileExpert = 12;
constexpr uint8_t RouterLogits = 0, RouterIds = 1, RouterWeights = 2;
constexpr uint8_t SortIds = 0, SortPerm = llama::kMoeSortPermBind, SortRowExpert = 2,
                  SortTileExpert = 3, SortInv = 5;
constexpr uint8_t RowsIn = 0, RowsOut = 1, RowsPerm = 2;
constexpr uint8_t CombineY = 0, CombineWeights = 1, CombineOut = 2, CombineInv = 4;
// bind::SharedCombine -- the routed output, the shared expert's, its
// one-per-row gate, and where the sum goes.
constexpr uint8_t ShRouted = 0, ShShared = 1, ShGate = 2, ShOut = 3;
}  // namespace bi

}  // namespace

ScratchSchedule build_scratch_schedule(const std::vector<Dispatch>& dag,
                                       const DecodeGeometry& g,
                                       bool no_recycle) {
    std::vector<Use> uses;
    int next_value = 0;
    auto fresh = [&]() { return next_value++; };

    // Live activation handles, threaded through the DAG order.
    int resid = -1;     // current residual stream (layer input / accumulator)
    int normed = -1;    // last Rms/FfnRms/FinalRms output feeding projections
    int q = -1, gate = -1, kk = -1, vv = -1, attn = -1;        // attn temporaries
    int mixed = -1, zg = -1, ag = -1, bg = -1, core = -1, gnorm = -1, gdnout = -1;  // GDN
    int pq = -1, pk = -1, pg = -1;          // GDN prep-dispatch scratch (PIE_GDN_PREP)
    bool prep_pending = false;              // a GdnPrep preceded this layer's GdnCore
    int gp = -1, up = -1, hh = -1, dn = -1;                    // mlp temporaries
    // The mixture. These are tracked as ordinary values with honest live ranges
    // rather than pinned, so the shared colouring SEES the extent instead of
    // being told about it -- `expert_weights` and the sort's outputs are
    // written once and still read five dispatches later, with three matvecs in
    // between allocating freely.
    int router_logits = -1, expert_ids = -1, expert_weights = -1;
    // The shared expert's two values that outlive a single dispatch. Its
    // gate/up/silu deliberately reuse `gp`/`up`/`hh`: by the time it runs, the
    // mixture's expert projections have consumed theirs, and it IS the same
    // three-stage SwiGLU dataflow. Giving it private handles would have said
    // the two are different shapes, which they are not.
    int shared_gate = -1, shared_out = -1;
    int perm = -1, row_expert = -1, tile_expert = -1, inv = -1;
    int sorted_x = -1, sorted_out = -1;

    auto rd = [&](int ord, uint8_t b, int val) { uses.push_back({ord, b, val, false}); };
    auto wr = [&](int ord, uint8_t b, int val) { uses.push_back({ord, b, val, true}); };

    for (size_t dispatch_index = 0; dispatch_index < dag.size(); ++dispatch_index) {
        const Dispatch& d = dag[dispatch_index];
        // Schedules are indexed by DAG position, not argument-table ordinal.
        // M=1 happens to use both as 0..N-1; paged DAG ordinals deliberately
        // live in a disjoint range.
        const int o = static_cast<int>(dispatch_index);
        switch (d.kind) {
            case Kernel::EmbedUntied:
            case Kernel::EmbedGather:
                resid = fresh(); wr(o, bi::EmbedOut, resid);
                break;

            // norms feeding the block (input/ffn) produce `normed` from `resid`.
            case Kernel::Rms:
            case Kernel::FfnRms:
                normed = fresh(); rd(o, bi::RmsX, resid); wr(o, bi::RmsOut, normed);
                break;
            case Kernel::FinalRms:
                normed = fresh(); rd(o, bi::RmsX, resid); wr(o, bi::RmsOut, normed);
                break;

            // ── GDN block ──
            case Kernel::QmvIn:
                mixed = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, mixed);
                break;
            case Kernel::QmvInZ:
                zg = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, zg);
                break;
            case Kernel::GdnInA:
                ag = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, ag);
                break;
            case Kernel::GdnInB:
                bg = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, bg);
                break;
            case Kernel::GdnPrep:
            case Kernel::GdnPrepSlotted:  // dv-independent q/k path → fp32 scratch (once/head)
                pq = fresh(); pk = fresh(); pg = fresh();
                rd(o, bi::GdnPrepMixed, mixed); rd(o, bi::GdnPrepAGate, ag); rd(o, bi::GdnPrepBGate, bg);
                wr(o, bi::GdnPrepPreQ, pq); wr(o, bi::GdnPrepPreK, pk); wr(o, bi::GdnPrepPreGate, pg);
                prep_pending = true;
                break;
            case Kernel::GdnCore:
            case Kernel::GdnCoreSlotted:
                core = fresh();
                if (prep_pending) {  // recurrent variant: reads prep's pre_q/k/gate scratch
                    rd(o, bi::GdnRecMixed, mixed);
                    rd(o, bi::GdnRecPreQ, pq); rd(o, bi::GdnRecPreK, pk); rd(o, bi::GdnRecPreGate, pg);
                    wr(o, bi::GdnRecCoreOut, core);
                    prep_pending = false;
                } else {
                    rd(o, bi::GdnMixed, mixed); rd(o, bi::GdnAGate, ag); rd(o, bi::GdnBGate, bg);
                    wr(o, bi::GdnCoreOut, core);
                }
                break;
            case Kernel::GatedRms:
                gnorm = fresh();
                rd(o, bi::GatedRmsX, core); rd(o, bi::GatedRmsZ, zg); wr(o, bi::GatedRmsOut, gnorm);
                break;
            case Kernel::QmvOut:
                if (d.fuse_residual) {  // gdn_out + block residual → new resid (drops Residual)
                    int nr = fresh();
                    rd(o, bi::QmvX, gnorm); rd(o, bi::QmvResidual, resid); wr(o, bi::QmvOut, nr);
                    resid = nr;
                } else {
                    gdnout = fresh(); rd(o, bi::QmvX, gnorm); wr(o, bi::QmvOut, gdnout);
                }
                break;

            // ── Full-attn block ──
            case Kernel::QmvQ:
                q = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, q);
                break;
            case Kernel::QSplit: {
                int Q = fresh(); gate = fresh();
                rd(o, bi::QSplitIn, q); wr(o, bi::QSplitQ, Q); wr(o, bi::QSplitGate, gate);
                q = Q;  // post-split query replaces the 2×-wide buffer
                break;
            }
            case Kernel::QmvK:
                kk = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, kk);
                break;
            case Kernel::QmvV:
                vv = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, vv);
                break;
            case Kernel::QNorm:  // in-place on the query heads
                rd(o, bi::RmsX, q); wr(o, bi::RmsOut, q);
                break;
            case Kernel::KNorm:  // in-place on the key heads
                rd(o, bi::RmsX, kk); wr(o, bi::RmsOut, kk);
                break;
            case Kernel::Rope:   // in-place on the query heads (kernel buffer 0 only)
                rd(o, bi::RopeX, q); wr(o, bi::RopeX, q);
                break;
            case Kernel::RopeK:  // in-place on the key heads (kernel buffer 0 only)
                rd(o, bi::RopeX, kk); wr(o, bi::RopeX, kk);
                break;
            case Kernel::KvAppend:
            case Kernel::KvAppendPaged:  // k/v -> KV pages; k/v read from scratch
                rd(o, bi::KvAppendK, kk); rd(o, bi::KvAppendV, vv);
                break;
            case Kernel::Sdpa:
            case Kernel::SdpaPaged:  // Q from scratch, K/V from KV region; out -> attn
                attn = fresh(); rd(o, bi::SdpaQ, q); wr(o, bi::SdpaOut, attn);
                break;
            case Kernel::AttnGate:  // attn *= sigmoid(gate), in-place on attn
                rd(o, bi::AttnGateAttn, attn); rd(o, bi::AttnGateGate, gate);
                wr(o, bi::AttnGateAttn, attn);
                break;
            case Kernel::QmvO:
                if (d.fuse_residual) {  // o_proj + block residual → new resid (drops Residual)
                    int nr = fresh();
                    rd(o, bi::QmvX, attn); rd(o, bi::QmvResidual, resid); wr(o, bi::QmvOut, nr);
                    resid = nr;
                } else {
                    gdnout = fresh(); rd(o, bi::QmvX, attn); wr(o, bi::QmvOut, gdnout);
                }
                break;

            // attn/gdn epilogue + mlp epilogue both fold into Residual.
            case Kernel::Residual: {
                int nr = fresh();
                rd(o, bi::ResidX, resid); rd(o, bi::ResidR, gdnout); wr(o, bi::ResidOut, nr);
                resid = nr;
                break;
            }

            // ── SwiGLU MLP ──
            case Kernel::QmvGate:
                gp = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, gp);
                break;
            case Kernel::QmvUp:
                up = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, up);
                break;
            case Kernel::SiluMul:
            case Kernel::LlExpertSiluMul:
                hh = fresh(); rd(o, bi::SiluGate, gp); rd(o, bi::SiluUp, up); wr(o, bi::SiluOut, hh);
                break;
            case Kernel::QmvDown:
                if (d.fuse_residual) {  // down_proj + block residual → new resid (drops LayerOut)
                    int nr = fresh();
                    rd(o, bi::QmvX, hh); rd(o, bi::QmvResidual, resid); wr(o, bi::QmvOut, nr);
                    resid = nr;
                } else {
                    dn = fresh(); rd(o, bi::QmvX, hh); wr(o, bi::QmvOut, dn);
                }
                break;
            case Kernel::LayerOut: {
                int nr = fresh();
                rd(o, bi::ResidX, resid); rd(o, bi::ResidR, dn); wr(o, bi::ResidOut, nr);
                resid = nr;
                break;
            }

            // ── the routed FFN ──
            // Mirrors the llama family's dataflow dispatch for dispatch,
            // because it is the same nine kernels reading the same binds.
            case Kernel::LlRouter:
                router_logits = fresh();
                rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, router_logits);
                break;
            case Kernel::GoRouterTopK:
                // Two outputs, and both outlive the three matvecs that follow:
                // the ids name each pair's expert and the weights are what the
                // combine finally sums with.
                expert_ids = fresh(); expert_weights = fresh();
                rd(o, bi::RouterLogits, router_logits);
                wr(o, bi::RouterIds, expert_ids); wr(o, bi::RouterWeights, expert_weights);
                break;
            case Kernel::LlMoeSort:
                // Four outputs: the permutation the gather applies, the
                // per-row expert the matvec form reads, the per-tile expert
                // the matmul form reads, and the inverse the combine reads.
                perm = fresh(); row_expert = fresh(); tile_expert = fresh(); inv = fresh();
                rd(o, bi::SortIds, expert_ids);
                wr(o, bi::SortPerm, perm); wr(o, bi::SortRowExpert, row_expert);
                wr(o, bi::SortTileExpert, tile_expert); wr(o, bi::SortInv, inv);
                break;
            case Kernel::LlMoeGather:
                sorted_x = fresh();
                rd(o, bi::RowsIn, normed); wr(o, bi::RowsOut, sorted_x);
                rd(o, bi::RowsPerm, perm);
                break;
            case Kernel::LlExpertGate:
                gp = fresh();
                rd(o, bi::QmvX, sorted_x); rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert); wr(o, bi::QmvOut, gp);
                break;
            case Kernel::LlExpertUp:
                up = fresh();
                rd(o, bi::QmvX, sorted_x); rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert); wr(o, bi::QmvOut, up);
                break;
            case Kernel::LlExpertDown:
                // Still sorted: the k results per token are only brought back
                // together by the combine, which reads them where the sort
                // left them, through its inverse.
                sorted_out = fresh();
                rd(o, bi::QmvX, hh); rd(o, bi::QmvExpertIds, row_expert);
                rd(o, bi::QmvTileExpert, tile_expert); wr(o, bi::QmvOut, sorted_out);
                break;
            case Kernel::LlMoeCombine:
                // The mixture's output takes the place a dense `down_proj`
                // would have written, so the layer's residual add below needs
                // no case of its own.
                dn = fresh();
                rd(o, bi::CombineY, sorted_out); rd(o, bi::CombineWeights, expert_weights);
                rd(o, bi::CombineInv, inv); wr(o, bi::CombineOut, dn);
                break;

            // ── the shared expert ──
            // A dense SwiGLU off the same `normed` the router read, added to
            // the mixture under a gate that is ONE number per token. The three
            // projections read `normed` and not `sorted_x`: the shared expert
            // sees every token whole, which is the whole point of it.
            case Kernel::LlSharedGateProj:
                shared_gate = fresh();
                rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, shared_gate);
                break;
            case Kernel::LlSharedGate:
                gp = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, gp);
                break;
            case Kernel::LlSharedUp:
                up = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, up);
                break;
            case Kernel::LlSharedDown:
                shared_out = fresh(); rd(o, bi::QmvX, hh); wr(o, bi::QmvOut, shared_out);
                break;
            case Kernel::LlSharedCombine: {
                // Takes `dn`'s place, so the layer's residual add below still
                // reads one value and does not know a mixture happened.
                int nd = fresh();
                rd(o, bi::ShRouted, dn); rd(o, bi::ShShared, shared_out);
                rd(o, bi::ShGate, shared_gate); wr(o, bi::ShOut, nd);
                dn = nd;
                break;
            }

            // tail lm_head: reads normed_final from scratch, writes logits to IO (not scratch).
            case Kernel::LmHeadUntied:
            case Kernel::QmvLmHead:
                rd(o, bi::QmvX, normed);
                break;
            case Kernel::Argmax:  // logits/next-token both in IO; no scratch
                break;
        }
    }

    // Concurrency: the encoders drop the barriers inside a run of mutually
    // independent dispatches (decode_step.cpp::concurrent_run_ends), so a buffer
    // last read at ordinal i must not be rewritten by anything still running
    // alongside i. Extend such a value's interval to the END of i's run -- with
    // runs of more than two that is what "one ordinal" used to approximate, and
    // the approximation would now be wrong.
    const std::vector<int> run_ends = concurrent_run_ends(dag);

    // Colouring is family-agnostic and lives in `scratch_color.hpp`, so gemma4
    // does not arrive with a second copy of it.
    std::vector<pie::metal::scratch::Use> shared_uses;
    shared_uses.reserve(uses.size());
    for (const Use& u : uses) {
        shared_uses.push_back({u.ordinal, u.bind_index, u.value, u.write});
    }
    const auto coloring = pie::metal::scratch::color_live_ranges(shared_uses, run_ends,
                                                                next_value, no_recycle);
    const std::vector<int>& color = coloring.color;

    ScratchSchedule sched;
    sched.per_dispatch.resize(dag.size());
    sched.colors_used = coloring.colors_used;
    for (const Use& u : uses) {
        sched.per_dispatch[u.ordinal].binds.push_back({u.bind_index, color[u.value]});
    }
    sched.hazard_free = coloring.hazard_free;

    (void)g;
    return sched;
}

void bind_scratch(RawMetalContext& ctx,
                  const std::vector<Dispatch>& dag,
                  const ScratchSchedule& sched,
                  const SlotHandle* pool,
                  int pool_n,
                  size_t byte_offset) {
    const size_t n = std::min(dag.size(), sched.per_dispatch.size());
    for (size_t di = 0; di < n; ++di) {
        const int ordinal = dag[di].ordinal;
        for (const ScratchBind& sb : sched.per_dispatch[di].binds) {
            if (sb.buffer_id < pool_n && pool[sb.buffer_id].valid()) {
                ctx.arg_bind_ordinal(ordinal, sb.bind_index, pool[sb.buffer_id], byte_offset);
            }
        }
    }
}

}  // namespace pie::metal
