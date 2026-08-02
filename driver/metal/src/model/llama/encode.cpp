// The llama families' step encoder.
//
// Most of this file is a mapping table, and it is short for the reason the
// kernels file was: the shared `Kernel` enum's common prefix already IS a llama
// decoder, so most kinds borrow both their weight map and their pipeline from
// it unchanged.
//
// Two places that is not true.
//
// The embedding and the head. A tied checkpoint reads `shared_embedding` for
// both; an untied one has `embed_tokens` and `lm_head` as separate tensors.
// That is a property of the CHECKPOINT, not of the kind, which is why
// `shared_kind` takes the geometry. Getting it wrong is a load failure -- the
// requested tensor is simply absent -- rather than a wrong number, which is the
// good outcome.
//
// The routed FFN. Here the weight map and the pipeline genuinely disagree:
// `ExpertGate` binds `mlp.experts.gate_proj` (so its weight key is
// `LlExpertGate`) but runs the ROUTED matvec, not the dense one. `pso_kind` and
// `shared_kind` exist as separate questions precisely for this.

#include "encode.hpp"

#include "../../model/qwen3_5/decode_dispatch.hpp"
#include "../../model/qwen3_5/decode_dispatch_mb.hpp"
#include "decode_consts.hpp"

namespace pie::metal::llama {

// The dense matvec's launch shape is shared, not restated: `affine_qmv_fast`
// is the same kernel for every family that dispatches it.
//
// And the M>1 forms are the SAME shapes, which is why this family has one
// `launch_shape` where gpt-oss has two. Every `*_mb_dispatch` below is its M=1
// counterpart with a row count folded in, and at `rows == 1` they agree
// element for element -- `rms_mb_dispatch(w, r, 1)` is `rms_dispatch(w, r)`,
// `elementwise_mb_dispatch(w, 1)` is `elementwise_dispatch(w)`, and so on. A
// second switch over the same thirty kinds would be a second place for the
// three shape bugs the numerics test just found to grow back.
using pie::metal::elementwise_mb_dispatch;
using pie::metal::embed_mb_dispatch;
using pie::metal::kv_append_mb_dispatch;
using pie::metal::qmm_bm;
using pie::metal::qmm_bn;
using pie::metal::qmm_t_dispatch;
using pie::metal::rms_mb_dispatch;
using pie::metal::rope_mb_dispatch;
using pie::metal::sdpa_paged_dispatch;

Kernel shared_kind(Kind k, const LlamaGeometry& g) {
    switch (k) {
        // Tied models read the one table for both ends of the model. Untied
        // ones have two tensors and two kinds.
        case Kind::EmbedGather:
            return g.tied_embeddings ? Kernel::EmbedGather : Kernel::EmbedUntied;
        case Kind::LmHead:
            return g.tied_embeddings ? Kernel::QmvLmHead : Kernel::LmHeadUntied;

        case Kind::AttnNorm:      return Kernel::Rms;
        case Kind::QmvQ:          return Kernel::QmvQ;
        case Kind::QmvK:          return Kernel::QmvK;
        case Kind::QmvV:          return Kernel::QmvV;
        case Kind::QNorm:         return Kernel::QNorm;
        case Kind::KNorm:         return Kernel::KNorm;
        case Kind::RopeQ:         return Kernel::Rope;
        case Kind::RopeK:         return Kernel::RopeK;
        case Kind::KvAppend:      return Kernel::KvAppend;
        case Kind::Sdpa:          return Kernel::Sdpa;
        case Kind::QmvO:          return Kernel::QmvO;
        case Kind::AttnResidual:  return Kernel::Residual;
        case Kind::FfnNorm:       return Kernel::FfnRms;
        case Kind::QmvGate:       return Kernel::QmvGate;
        case Kind::QmvUp:         return Kernel::QmvUp;
        case Kind::SiluMul:       return Kernel::SiluMul;
        case Kind::QmvDown:       return Kernel::QmvDown;
        case Kind::FfnResidual:   return Kernel::LayerOut;
        case Kind::RowGather:     return Kernel::G4RowGather;
        case Kind::FinalRms:      return Kernel::FinalRms;
        case Kind::Argmax:        return Kernel::Argmax;

        // Routed. The weight keys are this family's own, because gpt-oss's
        // equivalents bind a bias Qwen's experts do not have.
        case Kind::Router:          return Kernel::LlRouter;
        case Kind::ExpertGate:      return Kernel::LlExpertGate;
        case Kind::ExpertUp:        return Kernel::LlExpertUp;
        case Kind::ExpertDown:      return Kernel::LlExpertDown;
        case Kind::RouterTopK:      return Kernel::GoRouterTopK;
        case Kind::ExpertSiluMul:   return Kernel::SiluMul;
        case Kind::ExpertCombine:   return Kernel::GoExpertCombine;
    }
    return Kernel::Rms;
}

Kernel pso_kind(Kind k) {
    switch (k) {
        // All five norms run the one rms kernel; only their weights differ.
        case Kind::AttnNorm:
        case Kind::FfnNorm:
        case Kind::FinalRms:
        case Kind::QNorm:
        case Kind::KNorm:      return Kernel::Rms;
        // Every dense matvec is the same `affine_qmv_fast`; K and N come from
        // the per-ordinal constants, not from the pipeline.
        case Kind::QmvQ:
        case Kind::QmvK:
        case Kind::QmvV:
        case Kind::QmvO:
        case Kind::QmvGate:
        case Kind::QmvUp:
        case Kind::QmvDown:
        case Kind::Router:
        case Kind::LmHead:     return Kernel::QmvGate;
        case Kind::RopeQ:
        case Kind::RopeK:      return Kernel::Rope;
        case Kind::AttnResidual:
        case Kind::FfnResidual: return Kernel::Residual;
        case Kind::SiluMul:
        case Kind::ExpertSiluMul: return Kernel::SiluMul;
        // The routed matvecs run their own pipeline, which is why this is a
        // separate question from `shared_kind`.
        case Kind::ExpertGate:
        case Kind::ExpertUp:
        case Kind::ExpertDown:  return Kernel::LlExpertGate;
        case Kind::RouterTopK:  return Kernel::GoRouterTopK;
        case Kind::ExpertCombine: return Kernel::GoExpertCombine;
        case Kind::EmbedGather: return Kernel::EmbedGather;
        case Kind::KvAppend:    return Kernel::KvAppend;
        case Kind::Sdpa:        return Kernel::Sdpa;
        case Kind::RowGather:   return Kernel::G4RowGather;
        case Kind::Argmax:      return Kernel::Argmax;
    }
    return Kernel::Rms;
}

Pso pso_for(const Dispatch& d, const LlamaGeometry& g, const DecodeStepPsos& base,
            const LlamaPsos& ll, const MultiBatchPsos* mb, int rows, int head_rows) {
    const int R = rows < 1 ? 1 : rows;
    const int S = head_rows < 1 ? R : (head_rows < R ? head_rows : R);

    // The GEMM, when the batch fills a tile. Asked first and asked with the
    // same numbers `launch_shape` uses, because the two answers must agree.
    if (mb != nullptr) {
        const int m = d.kind == Kind::LmHead ? S : R;
        if (const int bn = llama_qmm_bn(d.kind, g, m); bn > 0) {
            const int padded = llama_qmm_rows(m);
            const int wide = qmm_bm(padded) == kQmmBMWide ? 1 : 0;
            const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
            // No bias table: llama's projections have no bias tensor, which is
            // the one place this family is simpler than gpt-oss.
            if (mb->qmm_t[wide][slot].valid()) return mb->qmm_t[wide][slot];
        }
        // Per-row IO. These two are the only kinds whose M>1 form differs in
        // how it INDEXES rather than merely in how wide it launches: the
        // gather reads `id[m]` and the rope reads `position[m]`. Everything
        // else in this family is already row-strided or flat over rows*width.
        if (R > 1) {
            switch (d.kind) {
                case Kind::EmbedGather:
                    if (mb->embed_mb.valid()) return mb->embed_mb;
                    break;
                case Kind::RopeQ:
                case Kind::RopeK:
                    if (mb->rope_mb.valid()) return mb->rope_mb;
                    break;
                default:
                    break;
            }
        }
    }

    switch (d.kind) {
        // The family's own PSOs: a 128-wide head, and the routed set.
        case Kind::Sdpa:
            return g.paged_kv_enabled ? ll.sdpa_paged_d128 : ll.sdpa_d128;
        // The append follows attention. Both KV kinds must agree on the ABI:
        // the binder writes page tables into slots the ring kernel reads as a
        // head stride, so a mismatch here is a scatter through a pointer made
        // of arithmetic, not an unbound slot.
        case Kind::KvAppend:
            if (g.paged_kv_enabled) {
                if (mb == nullptr || !mb->kv_append_paged.valid()) return Pso{};
                return mb->kv_append_paged;
            }
            break;
        case Kind::RowGather:     return ll.row_gather;
        case Kind::RouterTopK:    return ll.router_topk;
        case Kind::ExpertCombine: return ll.expert_combine;
        case Kind::ExpertGate:
        case Kind::ExpertUp:
        case Kind::ExpertDown:    return ll.qmv_routed;
        default:
            break;
    }
    return base[pso_kind(d.kind)];
}

namespace {

/// Dispatches that may run without a barrier between them.
///
/// A group is a set of dispatches with no true RAW edge among them, so the
/// hazard is not that they race but that a barrier between them costs ~6 us and
/// buys nothing. `concurrent_run_ends` only ever merges ADJACENT members of the
/// same group in the same layer, so a group number is a claim about independence
/// and not about ordering.
int concurrency_group(Kind k) {
    switch (k) {
        case Kind::QmvQ:
        case Kind::QmvK:
        case Kind::QmvV:
            return 1;  // all three read the attention norm's output
        case Kind::QNorm:
        case Kind::KNorm:
            return 2;  // each rewrites its own tensor in place
        case Kind::RopeQ:
        case Kind::RopeK:
            return 3;  // q and k, disjoint
        case Kind::QmvGate:
        case Kind::QmvUp:
            return 4;  // both read the FFN norm's output
        case Kind::ExpertGate:
        case Kind::ExpertUp:
            return 5;  // the routed pair, same argument
        default:
            return 0;  // runs alone
    }
}

}  // namespace

std::vector<int> llama_run_ends(const std::vector<Dispatch>& dag) {
    std::vector<int> ends(dag.size());
    for (std::size_t i = 0; i < dag.size(); ++i) ends[i] = static_cast<int>(i);
    std::size_t i = 0;
    while (i < dag.size()) {
        const int group = concurrency_group(dag[i].kind);
        std::size_t j = i;
        if (group != 0) {
            while (j + 1 < dag.size() && dag[j + 1].layer == dag[i].layer &&
                   concurrency_group(dag[j + 1].kind) == group) {
                ++j;
            }
        }
        for (std::size_t k = i; k <= j; ++k) ends[k] = static_cast<int>(j);
        i = j + 1;
    }
    return ends;
}

int llama_qmm_rows(int rows) {
    const int n = rows < 1 ? 1 : rows;
    if (n < kQmmMinBatch) return n;
    const int bm = qmm_bm(n);
    return ((n + bm - 1) / bm) * bm;
}

int llama_qmm_pool_rows(int max_rows) {
    const int n = max_rows < 1 ? 1 : max_rows;
    return ((n + kQmmBMWide - 1) / kQmmBMWide) * kQmmBMWide;
}

bool llama_is_dense_proj(Kind k) {
    // Everything with a K and an N that is not routed. Unlike gpt-oss, whose
    // FFN is always a mixture, a dense llama's gate/up/down are ordinary
    // projections and are the largest matrices in the layer -- excluding them
    // would leave most of a dense prefill running as a matvec.
    //
    // The router is deliberately NOT here. Its N is the expert count, tens of
    // columns against a hidden of thousands, so the GEMM's tile is mostly
    // padding, and it is the one projection whose output every later dispatch
    // in the layer waits on.
    switch (k) {
        case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
        case Kind::QmvGate: case Kind::QmvUp: case Kind::QmvDown:
        case Kind::LmHead:
            return true;
        default:
            return false;
    }
}

int llama_qmm_bn(Kind k, const LlamaGeometry& g, int rows) {
    if (!llama_is_dense_proj(k)) return 0;
    const KN kn = qmv_kn(k, g);
    if (kn.N == 0) return 0;
    return qmm_bn(kn.N, llama_qmm_rows(rows));
}

void launch_shape(const Dispatch& d, const LlamaGeometry& g, Grid& grid, Threadgroup& tg,
                  int rows, int head_rows) {
    const int R = rows < 1 ? 1 : rows;
    // The tail runs on the rows the sampler will READ, which `RowGather`
    // compacted to a dense prefix. The head is `hidden * vocab` per row, so on
    // a prefill it is most of the cost and all of the logits memory.
    const int S = head_rows < 1 ? R : (head_rows < R ? head_rows : R);
    // The dense matvecs first: they are most of the DAG, and the shared
    // `qmv_dispatch` is deliberately reused rather than restated. That kernel
    // is 2 simdgroups of 4 rows reducing K with `simd_sum`, so it needs
    // tg {32,2,1} against a grid of TOTAL THREADS. Restating it as "one
    // threadgroup per 8 rows" gives each threadgroup one thread, which silently
    // skips 4 rows in 8 and 31/32 of K.
    //
    // `is_routed` is asked FIRST because `qmv_kn` answers for the routed kinds
    // too -- they have a K and an N like any other matvec. Falling into the
    // dense shape on the strength of that leaves `grid.z` at 1, and `tid.z` is
    // the expert slot: every slot after the first is never dispatched at all
    // and its output stays whatever the pool held. The first expert is right,
    // so the model still produces text.
    if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
        const int m = d.kind == Kind::LmHead ? S : R;
        // Once the batch fills a tile, a dense projection becomes a matmul.
        // The matvec re-reads the ENTIRE weight for every row, so on a prefill
        // it is the difference between amortizing the weights and not.
        if (const int bn = llama_qmm_bn(d.kind, g, m); bn > 0) {
            const int padded = llama_qmm_rows(m);
            qmm_t_dispatch(kn.N, padded, bn, qmm_bm(padded), grid, tg);
            return;
        }
        // One call for dense and routed alike. `slots` is the expert axis and
        // is 1 for a dense projection, which makes the two cases the same
        // statement rather than two branches with a precedence between them.
        // They WERE two branches, and the dense one tested only `kn.N != 0` --
        // which `qmv_kn` answers for the routed kinds too, so the routed branch
        // below was unreachable, `grid.z` stayed 1, and every expert slot after
        // the first was never dispatched. The first expert is right, so the
        // model still produced text.
        routed_qmv_dispatch(kn.N, is_routed(d.kind) ? g.experts_per_token : 1, grid, tg, m);
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            embed_mb_dispatch(g.hidden, R, grid, tg);
            return;
        case Kind::RowGather:
            grid = Grid{std::uint32_t(g.hidden), std::uint32_t(S), 1};
            tg = Threadgroup{64, 1, 1};
            return;
        case Kind::AttnNorm:
        case Kind::FfnNorm:
            rms_mb_dispatch(g.hidden, 1, R, grid, tg);
            return;
        // The tail's norm runs on the COMPACTED rows, not on every token.
        case Kind::FinalRms:
            rms_mb_dispatch(g.hidden, 1, S, grid, tg);
            return;
        // Qwen3's qk-norms are per HEAD, over head_dim -- not one norm over the
        // whole projection. One threadgroup per head.
        case Kind::QNorm:
            rms_mb_dispatch(g.head_dim, g.n_q_heads, R, grid, tg);
            return;
        case Kind::KNorm:
            rms_mb_dispatch(g.head_dim, g.n_kv_heads, R, grid, tg);
            return;
        // NOT the norms' shape, which is what this used to borrow. `rms_norm`
        // reads four elements per thread and `residual_add` reads one, so the
        // norms' `hidden/4` threads leave three quarters of the residual stream
        // holding whatever the pool buffer held before. That survives -- the
        // first quarter is right, the model still emits tokens.
        case Kind::AttnResidual:
        case Kind::FfnResidual:
            elementwise_mb_dispatch(g.hidden, R, grid, tg);
            return;
        case Kind::RopeQ:
            rope_mb_dispatch(g.rotary_dims(), g.n_q_heads, R, grid, tg);
            return;
        case Kind::RopeK:
            rope_mb_dispatch(g.rotary_dims(), g.n_kv_heads, R, grid, tg);
            return;
        case Kind::KvAppend:
            kv_append_mb_dispatch(g.head_dim, g.n_kv_heads, R, grid, tg);
            return;
        case Kind::Sdpa:
            // The shared shape, not a restatement of it. `sdpa_vector_decode`
            // reads the query head from `tid.x` -- the THREADGROUP's x -- and
            // uses `tid.y` as the query's sequence index, which at decode is 0.
            // Putting the head on y instead launches the right number of
            // threads and computes the wrong thing: `kv_head_idx` is
            // `tid.x / gqa_factor`, so every query head would read KV head 0.
            // The output still lands per-head, so the symptom is not garbage --
            // it is attention with the grouping collapsed.
            //
            // The row axis is `grid.y`, which at R == 1 is the M=1 shape
            // unchanged -- the ring kernel reads y as the query's sequence
            // index and a decode has exactly one.
            sdpa_paged_dispatch(g.n_q_heads, R, grid, tg);
            return;
        case Kind::SiluMul:
            elementwise_mb_dispatch(g.intermediate, R, grid, tg);
            return;

        // ── routed ──
        case Kind::RouterTopK:
            router_topk_dispatch(g.n_experts, grid, tg, R);
            return;
        // The three routed matvecs are handled by the `qmv_kn` branch above,
        // which answers for them: they reach here only if that ever stops
        // being true.
        case Kind::ExpertGate:
        case Kind::ExpertUp:
            routed_qmv_dispatch(g.moe_intermediate, g.experts_per_token, grid, tg, R);
            return;
        case Kind::ExpertDown:
            routed_qmv_dispatch(g.hidden, g.experts_per_token, grid, tg, R);
            return;
        case Kind::ExpertSiluMul:
            expert_silu_dispatch(g.moe_intermediate, g.experts_per_token, grid, tg, R);
            return;
        case Kind::ExpertCombine:
            expert_combine_dispatch(g.hidden, grid, tg, R);
            return;

        case Kind::Argmax:
            grid = Grid{1024, 1, 1};
            tg = Threadgroup{1024, 1, 1};
            return;
        default:
            grid = Grid{1, 1, 1};
            tg = Threadgroup{1, 1, 1};
            return;
    }
}

void encode_llama_step(StepEncoder& se, const std::vector<Dispatch>& dag, const LlamaGeometry& g,
                       const DecodeStepPsos& base, const LlamaPsos& ll, int ordinal_base,
                       const MultiBatchPsos* mb, int rows, int head_rows) {
    const std::vector<int> run_ends = llama_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg, rows, head_rows);
        se.set_pso(pso_for(d, g, base, ll, mb, rows, head_rows));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        // A barrier after every dispatch except inside a concurrency run: the
        // last member of a run carries it for the whole group.
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::llama
