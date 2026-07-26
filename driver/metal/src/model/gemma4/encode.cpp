// Gemma 4's step encoder: the DAG, walked with a real encoder.
//
// Everything before this was a plan — which dispatches, in what order, reading
// which values, with which constants. This is where they become commands.
//
// Two choices are per-dispatch rather than per-kind, and both come from the
// layer's attention type:
//
//   * which SDPA pipeline (`d=256` sliding, `d=512` full)
//   * which grid, because head_dim and the MLP width vary with the layer
//
// Barriers follow the same rule qwen3.5's encoder uses: one after every
// dispatch except inside a run of mutually independent ones. On this machine a
// barrier costs ~3.7 us, and the step has 834 dispatches, so the runs are worth
// having.

#include "encode.hpp"

#include "../../batch/decode_abi.hpp"
#include "decode_consts.hpp"

namespace pie::metal::gemma4 {

namespace {

/// Dispatches that may run together: same layer, mutually independent, all
/// reading something produced before the group starts and writing distinct
/// values. This is an explicit list rather than something derived, for the same
/// reason qwen3.5's is — the scratch dataflow does not model the KV pages, so
/// "independent" cannot be read off it.
int concurrency_group(Kind k) {
    switch (k) {
        case Kind::QmvQ:
        case Kind::QmvK:
        case Kind::QmvV:
            return 1;  // all three read the attention norm's output
        case Kind::QNorm:
        case Kind::KNorm:
        case Kind::VNorm:
            return 2;  // each rewrites its own tensor in place
        case Kind::RopeQ:
        case Kind::RopeK:
            return 3;  // q and k, disjoint
        case Kind::QmvGate:
        case Kind::QmvUp:
            return 4;  // both read the FFN norm's output
        default:
            return 0;  // runs alone
    }
}

std::vector<int> concurrent_run_ends(const std::vector<Dispatch>& dag) {
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

}  // namespace

std::vector<int> gemma4_run_ends(const std::vector<Dispatch>& dag) {
    return concurrent_run_ends(dag);
}

/// The shared `Kernel` a gemma4 kind borrows its pipeline from.
///
/// The PSO table is indexed by the shared enum, and most of this family's
/// dispatches run the very same kernel qwen3.5's do — a quantized matvec is a
/// quantized matvec. Only what a kind BINDS and what constants it reads differ,
/// and those are per ordinal.
Kernel shared_kind(Kind k) {
    switch (k) {
        case Kind::EmbedGather:
        case Kind::PleTokenGather:      return Kernel::EmbedGather;
        case Kind::QmvQ:                return Kernel::QmvQ;
        case Kind::QmvK:                return Kernel::QmvK;
        case Kind::QmvV:                return Kernel::QmvV;
        case Kind::QmvO:                return Kernel::QmvO;
        case Kind::QmvGate:             return Kernel::QmvGate;
        case Kind::QmvUp:               return Kernel::QmvUp;
        case Kind::QmvDown:             return Kernel::QmvDown;
        case Kind::LmHead:              return Kernel::QmvLmHead;
        case Kind::PleProjGemv:         return Kernel::G4PleProjGemv;
        case Kind::PleGateGemv:         return Kernel::G4PleGateGemv;
        case Kind::PleProjLayerGemv:    return Kernel::G4PleProjLayerGemv;
        case Kind::AttnNorm:            return Kernel::Rms;
        case Kind::PostAttnNorm:        return Kernel::G4AttnPostNorm;
        case Kind::FfnNorm:             return Kernel::G4FfnPreNorm;
        case Kind::PostFfnNorm:         return Kernel::G4FfnPostNorm;
        case Kind::FinalRms:            return Kernel::FinalRms;
        case Kind::QNorm:               return Kernel::QNorm;
        case Kind::KNorm:               return Kernel::KNorm;
        case Kind::PleNorm:             return Kernel::G4PleNorm;
        case Kind::PleProjNorm:         return Kernel::G4PleProjNorm;
        case Kind::RopeQ:               return Kernel::Rope;
        case Kind::RopeK:               return Kernel::RopeK;
        case Kind::KvAppend:            return Kernel::KvAppend;
        case Kind::Sdpa:                return Kernel::Sdpa;
        case Kind::AttnResidual:        return Kernel::Residual;
        case Kind::FfnResidual:         return Kernel::LayerOut;
        case Kind::PleResidual:         return Kernel::G4PleResidual;
        case Kind::VNorm:               return Kernel::G4VNorm;
        case Kind::GegluTanh:           return Kernel::G4Geglu;
        case Kind::PleGeglu:            return Kernel::G4PleGeglu;
        case Kind::PleCombine:          return Kernel::G4PleCombine;
        case Kind::LayerScalar:         return Kernel::G4LayerScalar;
        case Kind::FinalSoftcap:        return Kernel::G4Softcap;
        case Kind::Argmax:              return Kernel::Argmax;
    }
    return Kernel::Argmax;
}

/// The kind whose PIPELINE a gemma4 kind runs on.
///
/// Distinct from `shared_kind`, which answers a different question: that one is
/// the weight-map key, and gemma4's norms bind different tensors even though
/// they run the identical `rms_single_row`. The PSO table is filled by
/// qwen3.5's loader and only has entries for its kinds, so this maps onto those.
Kernel pso_kind(Kind k) {
    switch (k) {
        case Kind::PostAttnNorm:
        case Kind::FfnNorm:
        case Kind::PostFfnNorm:
        case Kind::PleNorm:
        case Kind::PleProjNorm:      return Kernel::Rms;
        case Kind::PleProjGemv:
        case Kind::PleGateGemv:
        case Kind::PleProjLayerGemv: return Kernel::QmvGate;
        case Kind::PleResidual:      return Kernel::Residual;
        default:                     return shared_kind(k);
    }
}

Pso pso_for(const Dispatch& d, const DecodeStepPsos& base, const Gemma4Psos& g4) {
    switch (d.kind) {
        // The gemma4-only kernels, which the shared table has no entry for.
        case Kind::GegluTanh:
        case Kind::PleGeglu:     return g4.geglu_tanh;
        case Kind::LayerScalar:  return g4.layer_scalar;
        case Kind::FinalSoftcap: return g4.logit_softcap;
        case Kind::PleCombine:   return g4.ple_combine;
        case Kind::VNorm:        return g4.vnorm;
        // The one place the attention type picks the pipeline rather than a
        // constant: the two head widths are separate instantiations.
        case Kind::Sdpa:         return d.sliding ? g4.sdpa_swa_d256 : g4.sdpa_swa_d512;
        default:                 return base[pso_kind(d.kind)];
    }
}

void launch_shape(const Dispatch& d, const Gemma4Geometry& g, Grid& grid, Threadgroup& tg) {
    const int L = d.layer;
    const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;

    if (const KN kn = qmv_kn(d.kind, g, L); kn.N != 0) {
        // affine_qmv_fast: 2 simdgroups x 4 rows each, so 8 output rows per group.
        grid = Grid{1, std::uint32_t((kn.N + 7) / 8), 1};
        tg = Threadgroup{64, 1, 1};
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        case Kind::PleTokenGather:
            elementwise_dispatch(g.n_layers * g.per_layer_emb_dim, grid, tg);
            return;
        case Kind::AttnNorm: case Kind::PostAttnNorm: case Kind::FfnNorm:
        case Kind::PostFfnNorm: case Kind::FinalRms: {
            const int threads = (g.hidden + 3) / 4;
            grid = Grid{std::uint32_t(threads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::QNorm: {
            const int threads = (hd + 3) / 4;
            grid = Grid{std::uint32_t(threads) * std::uint32_t(g.n_q_heads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::KNorm: {
            const int threads = (hd + 3) / 4;
            grid = Grid{std::uint32_t(threads) * std::uint32_t(g.n_kv_heads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::PleNorm: case Kind::PleProjNorm: {
            const int threads = (g.per_layer_emb_dim + 3) / 4;
            const int rows = d.kind == Kind::PleProjNorm ? g.n_layers : 1;
            grid = Grid{std::uint32_t(threads) * std::uint32_t(rows), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::VNorm:
            vnorm_dispatch(g.n_kv_heads, hd, grid, tg);
            return;
        case Kind::RopeQ:
            grid = Grid{std::uint32_t(g.rotary_dims_of(L) / 2), std::uint32_t(g.n_q_heads), 1};
            tg = Threadgroup{std::uint32_t(g.rotary_dims_of(L) / 2), 1, 1};
            return;
        case Kind::RopeK:
            grid = Grid{std::uint32_t(g.rotary_dims_of(L) / 2), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.rotary_dims_of(L) / 2), 1, 1};
            return;
        case Kind::KvAppend:
            grid = Grid{std::uint32_t(hd), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(hd), 1, 1};
            return;
        case Kind::Sdpa:
            sdpa_sliding_dispatch(g.n_q_heads, grid, tg);
            return;
        case Kind::AttnResidual: case Kind::FfnResidual: case Kind::PleResidual:
        case Kind::LayerScalar:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        case Kind::GegluTanh:
            elementwise_dispatch(L >= 0 ? g.intermediate_of(L) : g.intermediate, grid, tg);
            return;
        case Kind::PleGeglu:
            elementwise_dispatch(g.per_layer_emb_dim, grid, tg);
            return;
        case Kind::PleCombine:
            elementwise_dispatch(g.n_layers * g.per_layer_emb_dim, grid, tg);
            return;
        case Kind::FinalSoftcap:
            elementwise_dispatch(g.vocab, grid, tg);
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

void encode_gemma4_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const Gemma4Geometry& g, const DecodeStepPsos& base,
                        const Gemma4Psos& g4, int ordinal_base) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for(d, base, g4));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        // A barrier after every dispatch except inside a concurrency run: the
        // last member of a run carries it for the whole group.
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::gemma4
