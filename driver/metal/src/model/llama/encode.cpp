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
#include "decode_consts.hpp"

namespace pie::metal::llama {

// The dense matvec's launch shape is shared, not restated: `affine_qmv_fast`
// is the same kernel for every family that dispatches it.
using pie::metal::qmv_dispatch;

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
            const LlamaPsos& ll) {
    switch (d.kind) {
        // The family's own PSOs: a 128-wide head, and the routed set.
        case Kind::Sdpa:
            return g.paged_kv_enabled ? ll.sdpa_paged_d128 : ll.sdpa_d128;
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

void launch_shape(const Dispatch& d, const LlamaGeometry& g, Grid& grid, Threadgroup& tg) {
    // The dense matvecs first: they are most of the DAG, and the shared
    // `qmv_dispatch` is deliberately reused rather than restated. That kernel
    // is 2 simdgroups of 4 rows reducing K with `simd_sum`, so it needs
    // tg {32,2,1} against a grid of TOTAL THREADS. Restating it as "one
    // threadgroup per 8 rows" gives each threadgroup one thread, which silently
    // skips 4 rows in 8 and 31/32 of K.
    if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
        qmv_dispatch(kn.N, grid, tg);
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        case Kind::RowGather:
            grid = Grid{std::uint32_t(g.hidden), 1, 1};
            tg = Threadgroup{64, 1, 1};
            return;
        case Kind::AttnNorm:
        case Kind::FfnNorm:
        case Kind::FinalRms:
        case Kind::AttnResidual:
        case Kind::FfnResidual: {
            const int threads = (g.hidden + 3) / 4;
            grid = Grid{std::uint32_t(threads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        // Qwen3's qk-norms are per HEAD, over head_dim -- not one norm over the
        // whole projection. One threadgroup per head.
        case Kind::QNorm: {
            const int threads = (g.head_dim + 3) / 4;
            grid = Grid{std::uint32_t(threads) * std::uint32_t(g.n_q_heads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::KNorm: {
            const int threads = (g.head_dim + 3) / 4;
            grid = Grid{std::uint32_t(threads) * std::uint32_t(g.n_kv_heads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::RopeQ:
            grid = Grid{std::uint32_t(g.rotary_dims() / 2), std::uint32_t(g.n_q_heads), 1};
            tg = Threadgroup{std::uint32_t(g.rotary_dims() / 2), 1, 1};
            return;
        case Kind::RopeK:
            grid = Grid{std::uint32_t(g.rotary_dims() / 2), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.rotary_dims() / 2), 1, 1};
            return;
        case Kind::KvAppend:
            grid = Grid{std::uint32_t(g.head_dim), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim), 1, 1};
            return;
        case Kind::Sdpa:
            grid = Grid{1024, std::uint32_t(g.n_q_heads), 1};
            tg = Threadgroup{1024, 1, 1};
            return;
        case Kind::SiluMul:
            elementwise_dispatch(g.intermediate, grid, tg);
            return;

        // ── routed ──
        case Kind::RouterTopK:
            router_topk_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertGate:
        case Kind::ExpertUp:
            routed_qmv_dispatch(g.moe_intermediate, g.experts_per_token, grid, tg);
            return;
        case Kind::ExpertDown:
            routed_qmv_dispatch(g.hidden, g.experts_per_token, grid, tg);
            return;
        case Kind::ExpertSiluMul:
            expert_silu_dispatch(g.moe_intermediate, g.experts_per_token, grid, tg);
            return;
        case Kind::ExpertCombine:
            expert_combine_dispatch(g.hidden, grid, tg);
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
                       const DecodeStepPsos& base, const LlamaPsos& ll, int ordinal_base) {
    const std::vector<int> run_ends = llama_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for(d, g, base, ll));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        // A barrier after every dispatch except inside a concurrency run: the
        // last member of a run carries it for the whole group.
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::llama
