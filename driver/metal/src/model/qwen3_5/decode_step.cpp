// decode_step.cpp — build + encode the qwen3.6 per-token decode DAG (beta's lane).
//
// DAG order matches the GOLDEN kernel surface (wiki mac-golden-kernel-surface) +
// delta's coverage pass: full gated-attn (q_proj 2×-wide → QSplit, AttnGate), 4-way
// GDN in-projection (QmvIn/QmvInZ 4-bit + GdnInA/GdnInB dense bf16), GdnCore→GatedRms,
// per-class SwiGLU MLP. Launch dims come from delta's decode_dispatch.hpp helpers
// (authoritative); GdnCore's grid is beta's (gdn_core.metal {32,Vd,Vh}/{32,4,1}).
//
// Counts: 1 embed + 18 GDN×15 + 6 full-attn×20 + 2 tail(FinalRms,QmvLmHead) = 393
// raw dispatches (363 golden-tapped: QSplit/KvAppend internal, GdnCore folds into the
// gdn_core tap via GatedRms). +1 if with_argmax.
//
// Barrier modeling (beta's WAR/WAW lane): a barrier follows every dispatch except the
// few adjacent independent pairs (k‖v proj, q‖k norm, gate‖up proj) that read a common
// already-produced input and write disjoint slots.

#include "decode_step.hpp"
#include "decode_dispatch.hpp"
#include "../shared_kernels.hpp"

namespace pie::metal {

namespace {

struct LD { Grid grid; Threadgroup tg; };

// GdnCore launch lives with beta's kernel (dispatchThreads = total threads).
// tg spans a TILE of dv (32 simdgroups) so the q/k prologue is computed once per
// threadgroup (tpit.y==0) + shared via threadgroup memory — kills the Vd-fold
// redundancy while keeping full occupancy. grid {32,Vd,Hv} / tg {32,32,1}.
LD gdncore_ld(const DecodeGeometry& g) {
    return { Grid{32, uint32_t(g.gdn_v_dim), uint32_t(g.gdn_v_heads)}, Threadgroup{32, 32, 1} };
}

// Prep-dispatch split (PIE_GDN_PREP): GdnPrep computes the dv-INDEPENDENT q/k path
// ONCE per (req,v-head) — one simdgroup/head (32 lanes × n_per_t=4 cover Dk=128).
LD gdn_prep_ld(const DecodeGeometry& g) {
    return { Grid{32, 1, uint32_t(g.gdn_v_heads)}, Threadgroup{32, 1, 1} };
}
// Slimmed recurrent core (PIE_GDN_PREP): per-(req,v-head,v-dim) RMW reading the prep
// scratch — no dv-tile share needed, so tg drops back to {32,4,1}.
LD gdncore_rec_ld(const DecodeGeometry& g) {
    return { Grid{32, uint32_t(g.gdn_v_dim), uint32_t(g.gdn_v_heads)}, Threadgroup{32, 4, 1} };
}

}  // namespace

std::vector<Dispatch> build_decode_dag(const DecodeGeometry& g, bool with_argmax,
                                       bool fuse_residual, bool gdn_prep) {
    const int q_dim  = g.n_q_heads * g.head_dim;   // 2048 (post-split query)
    const int qg_dim = 2 * q_dim;                  // 4096 (q_proj is 2×-wide [query|gate])
    const int kv_dim = g.n_kv_heads * g.head_dim;  // 512

    std::vector<Dispatch> dag;
    int ord = 0;
    auto emit = [&](Kernel k, int layer, LD ld) {
        Dispatch d; d.kind = k; d.ordinal = ord++; d.layer = layer; d.grid = ld.grid; d.tg = ld.tg;
        dag.push_back(d);
    };
    auto qmv   = [&](int N) { LD l; qmv_dispatch(N, l.grid, l.tg); return l; };
    auto rms   = [&](int row, int rows) { LD l; rms_dispatch(row, rows, l.grid, l.tg); return l; };
    auto rope  = [&](int nh) { LD l; rope_dispatch(g.rotary_dims, nh, l.grid, l.tg); return l; };
    auto resid = [&]() { LD l; residual_dispatch(g.hidden, l.grid, l.tg); return l; };

    // EMBED ×1 (4-bit dequant gather of the shared lm_head bundle).
    // Tied, one table serves both ends of the model and both kinds bind
    // `shared_embedding`. Untied -- which is every routed member of this
    // family -- they are two tensors, and a kind is a weight name, so they
    // are two kinds. Same kernels, same launch shapes, same constants: what
    // differs is only which tensor is asked for.
    { LD l; embed_dispatch(g.hidden, l.grid, l.tg);
      emit(g.tied_embeddings ? Kernel::EmbedGather : Kernel::EmbedUntied, -1, l); }

    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_full_attn(L)) {
            // Full-attn (20): attn_norm, q_proj(4096), q_split, k_proj, v_proj, q_norm,
            // k_norm, rope_q, rope_k, kv_append, sdpa, attn_gate, o_proj, attn_resid + MLP(6).
            emit(Kernel::Rms,    L, rms(g.hidden, 1));
            // q/k/v adjacent so they form one concurrent run: all three read the
            // block norm's output and write disjoint scratch. QSplit consumes
            // q_proj, so it follows the run rather than splitting it in half.
            emit(Kernel::QmvQ,   L, qmv(qg_dim));                       // 2×-wide [query|gate]
            emit(Kernel::QmvK,   L, qmv(kv_dim));
            emit(Kernel::QmvV,   L, qmv(kv_dim));
            { LD l; q_split_dispatch(g.head_dim, g.n_q_heads, l.grid, l.tg); emit(Kernel::QSplit, L, l); }
            emit(Kernel::QNorm,  L, rms(g.head_dim, g.n_q_heads));
            emit(Kernel::KNorm,  L, rms(g.head_dim, g.n_kv_heads));
            emit(Kernel::Rope,   L, rope(g.n_q_heads));
            emit(Kernel::RopeK,  L, rope(g.n_kv_heads));
            { LD l; kv_append_dispatch(g.head_dim, g.n_kv_heads, l.grid, l.tg); emit(Kernel::KvAppend, L, l); }
            { LD l; sdpa_dispatch(g.n_q_heads, l.grid, l.tg); emit(Kernel::Sdpa, L, l); }
            { LD l; attn_gate_dispatch(g.n_q_heads, g.head_dim, l.grid, l.tg); emit(Kernel::AttnGate, L, l); }
            emit(Kernel::QmvO,   L, qmv(g.hidden));
            if (fuse_residual) dag.back().fuse_residual = true;
            else emit(Kernel::Residual, L, resid());
        } else {
            // GDN (15): attn_norm, gdn_in_qkv(6144), gdn_in_z(2048), gdn_in_a, gdn_in_b,
            // gdn_core, gated_rms, gdn_out, attn_resid + MLP(6).
            emit(Kernel::Rms,     L, rms(g.hidden, 1));
            emit(Kernel::QmvIn,   L, qmv(g.gdn_conv_dim));             // 6144, 4-bit
            emit(Kernel::QmvInZ,  L, qmv(g.gdn_v_total));              // 2048, 4-bit (gate z)
            emit(Kernel::GdnInA,  L, qmv(g.gdn_v_heads));              // 1024 → 16, 4-bit
            emit(Kernel::GdnInB,  L, qmv(g.gdn_v_heads));              // 1024 → 16, 4-bit
            if (gdn_prep) {
                // Prep-dispatch split: dv-independent q/k path hoisted to GdnPrep (once/head),
                // then the slimmed recurrent core reads its fp32 scratch (full 128×→1×).
                emit(Kernel::GdnPrep, L, gdn_prep_ld(g));
                emit(Kernel::GdnCore, L, gdncore_rec_ld(g));           // gdn_core_recurrent
            } else {
                emit(Kernel::GdnCore, L, gdncore_ld(g));               // beta's fused 1-dispatch core
            }
            { LD l; gated_rms_dispatch(g.gdn_v_heads, g.gdn_v_dim, l.grid, l.tg); emit(Kernel::GatedRms, L, l); }
            emit(Kernel::QmvOut,  L, qmv(g.hidden));                   // gdn_out
            if (fuse_residual) dag.back().fuse_residual = true;
            else emit(Kernel::Residual, L, resid());
        }
        // The FFN, in one of two shapes. Everything above this line is the
        // same either way -- the routed and dense members of this family differ
        // in the mixture and in nothing else, which is why they are one family
        // and not two.
        emit(Kernel::FfnRms, L, rms(g.hidden, 1));
        if (g.is_moe()) {
            // The same nine dispatches and the same kernels the llama family's
            // mixture runs. They are shared `Kernel` values, and the weights
            // for them are already keyed by kind in `weights_for_kind`, so
            // what this family was missing was the DAG and not the machinery.
            //
            // The sort is what makes the projections tractable: each of the
            // three reads a DIFFERENT weight matrix per (token, slot) pair, so
            // grouping the pairs by expert makes one expert's rows contiguous
            // and a contiguous run against one weight slice is the matmul the
            // driver already has. At M=1 it is a grouping with no padding and
            // the projections stay matvecs, which is deliberate: one dataflow
            // for both, so the batched path is not a second implementation.
            const int sorted = moe_sorted_rows(g);
            emit(Kernel::LlRouter, L, qmv(g.n_experts));
            { LD l; shared_kernels::router_topk_dispatch(g.n_experts, l.grid, l.tg);
              emit(Kernel::GoRouterTopK, L, l); }
            { LD l; shared_kernels::moe_route_sort_dispatch(g.n_experts, l.grid, l.tg);
              emit(Kernel::LlMoeSort, L, l); }
            { LD l; shared_kernels::moe_route_rows_dispatch(g.hidden, sorted, l.grid, l.tg);
              emit(Kernel::LlMoeGather, L, l); }
            { LD l; shared_kernels::routed_qmv_dispatch(g.moe_intermediate, 1, l.grid, l.tg,
                                                        sorted);
              emit(Kernel::LlExpertGate, L, l); }
            { LD l; shared_kernels::routed_qmv_dispatch(g.moe_intermediate, 1, l.grid, l.tg,
                                                        sorted);
              emit(Kernel::LlExpertUp, L, l); }
            { LD l; shared_kernels::moe_route_rows_dispatch(g.moe_intermediate, sorted,
                                                            l.grid, l.tg);
              emit(Kernel::LlExpertSiluMul, L, l); }
            { LD l; shared_kernels::routed_qmv_dispatch(g.hidden, 1, l.grid, l.tg, sorted);
              emit(Kernel::LlExpertDown, L, l); }
            { LD l; shared_kernels::moe_route_rows_dispatch(g.hidden, 1, l.grid, l.tg);
              emit(Kernel::LlMoeCombine, L, l); }
            // The shared expert. A dense FFN over the SAME `FfnRms` output the
            // router read, at its own width, added to the mixture under a gate
            // that is one number per token.
            //
            // It is emitted after the mixture rather than before only because
            // the combine has to see both; nothing here depends on the routed
            // half, and `parallel_groups` says so, so the two halves overlap on
            // the GPU. Keeping them textually apart would have hidden that they
            // are one FFN in two pieces.
            if (g.has_shared_expert()) {
                emit(Kernel::LlSharedGate, L, qmv(g.shared_intermediate));
                emit(Kernel::LlSharedUp,   L, qmv(g.shared_intermediate));
                { LD l; silu_mul_dispatch(g.shared_intermediate, l.grid, l.tg);
                  emit(Kernel::SiluMul, L, l); }
                emit(Kernel::LlSharedDown, L, qmv(g.hidden));
                // The gate is emitted LAST of the five, immediately before its
                // only reader. It reads `normed`, which is live the whole way
                // through, so it could have gone first -- and going first cost
                // a ninth scratch colour on a pool of eight, because a value
                // written five dispatches before it is read is live across all
                // five. It is `hidden -> 1`: the cheapest dispatch in the
                // layer, and the one with the least to gain from overlapping.
                emit(Kernel::LlSharedGateProj, L, qmv(1));
                { LD l; shared_kernels::moe_route_rows_dispatch(g.hidden, 1, l.grid, l.tg);
                  emit(Kernel::LlSharedCombine, L, l); }
            }
        } else {
            emit(Kernel::QmvGate,  L, qmv(g.intermediate));
            emit(Kernel::QmvUp,    L, qmv(g.intermediate));
            { LD l; silu_mul_dispatch(g.intermediate, l.grid, l.tg); emit(Kernel::SiluMul, L, l); }
            emit(Kernel::QmvDown,  L, qmv(g.hidden));
            if (fuse_residual) dag.back().fuse_residual = true;
        }
        if (!fuse_residual || g.is_moe()) emit(Kernel::LayerOut, L, resid());
    }

    // TAIL: final_norm, lm_head (logits ALWAYS produced, I3), [optional] device argmax.
    emit(Kernel::FinalRms,  -1, rms(g.hidden, 1));
    emit(g.tied_embeddings ? Kernel::QmvLmHead : Kernel::LmHeadUntied, -1, qmv(g.vocab));
    if (with_argmax) {
        emit(Kernel::Argmax, -1, LD{ Grid{1024, 1, 1}, Threadgroup{1024, 1, 1} });
    }
    return dag;
}

// Kinds that may run together inside a layer. Each group reads only activations
// produced before the group starts and writes a distinct scratch value, so the
// members are mutually independent -- not merely pairwise-adjacent-independent,
// which is what lets a whole group drop its internal barriers.
//
//   * the GDN in-projections and the attention q/k/v projections all read the
//     block norm's output;
//   * gate/up both read the FFN norm's output;
//   * q_norm/k_norm rewrite their own heads in place.
//
// Nothing here touches KV pages or GDN recurrent state, which the scratch
// dataflow does not model; that is why this is an explicit list rather than a
// derivation from the scratch schedule alone.
static int concurrency_group(Kernel k) {
    switch (k) {
        case Kernel::QmvIn:
        case Kernel::QmvInZ:
        case Kernel::GdnInA:
        case Kernel::GdnInB:   return 1;
        case Kernel::QmvQ:
        case Kernel::QmvK:
        case Kernel::QmvV:     return 2;
        case Kernel::QmvGate:
        case Kernel::QmvUp:    return 3;
        case Kernel::QNorm:
        case Kernel::KNorm:    return 4;
        // Rope rewrites q in place and RopeK rewrites k in place, from
        // disjoint scratch, exactly like the QNorm/KNorm pair above -- so the
        // barrier that used to sit between them was ordering nothing.
        case Kernel::Rope:
        case Kernel::RopeK:    return 5;
        default:               return 0;  // runs alone
    }
}

std::vector<int> concurrent_run_ends(const std::vector<Dispatch>& dag) {
    std::vector<int> ends(dag.size());
    for (std::size_t i = 0; i < dag.size(); ++i) ends[i] = int(i);
    std::size_t i = 0;
    while (i < dag.size()) {
        const int group = concurrency_group(dag[i].kind);
        std::size_t j = i;
        if (group != 0) {
            while (j + 1 < dag.size() &&
                   dag[j + 1].layer == dag[i].layer &&
                   concurrency_group(dag[j + 1].kind) == group) {
                ++j;
            }
        }
        for (std::size_t k = i; k <= j; ++k) ends[k] = int(j);
        i = j + 1;
    }
    return ends;
}

// Barrier flags: false = this dispatch runs concurrently with the next (no barrier).
// Only adjacent independent pairs that read an already-produced common input and write
// disjoint outputs. Conservative; the gate localizes any over-/under-sync as a port bug.
static bool barrier_after(const std::vector<Dispatch>& dag, size_t i,
                          const std::vector<int>& run_ends) {
    if (i + 1 >= dag.size()) return true;
    return run_ends[i] == int(i);
}

void encode_decode_step(StepEncoder& se,
                        const std::vector<Dispatch>& dag,
                        const DecodeStepPsos& psos,
                        bool force_barriers,
                        const StepTimingHook* timing) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        if (timing && timing->mark) timing->mark(static_cast<int>(i));  // boundary i
        se.set_pso(d.fuse_residual ? psos.qmv_residual : psos[d.kind]);
        se.set_argtable(d.kind, d.ordinal);  // ordinal-keyed (unique, token-stable)
        se.dispatch(d.grid, d.tg);
        if (force_barriers || barrier_after(dag, i, run_ends)) se.barrier();
    }
    if (timing && timing->mark) timing->mark(static_cast<int>(dag.size()));  // final boundary
}

}  // namespace pie::metal
