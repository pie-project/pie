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

#include "../qwen3_5/decode_dispatch_mb.hpp"

#include "../../batch/decode_abi.hpp"
#include "decode_consts.hpp"
#include "scratch.hpp"

namespace pie::metal::gemma4 {

int gemma4_qmm_rows(int rows) {
    const int n = rows < 1 ? 1 : rows;
    // `kQmmMinBatch` is qwen3.5's crossover between the GEMV and the GEMM, and
    // this family inherits it. MEASURED here rather than assumed: lowering it
    // to 4, so the GEMM engages at 8 rows instead of 12, costs gemma4 17%
    // (8 lanes, 128.5 -> 106.5 tok/s). The inherited number holds.
    if (n < kQmmMinBatch) return n;
    const int bm = qmm_bm(n);
    return ((n + bm - 1) / bm) * bm;
}

int gemma4_qmm_pool_rows(int max_rows) {
    const int n = max_rows < 1 ? 1 : max_rows;
    // Any row count up to `n` pads to at most this.
    const int bm = kQmmBMWide;
    return ((n + bm - 1) / bm) * bm;
}


namespace {

/// The three projections whose weight stack is indexed by the routing.
///
/// Asked BEFORE the dense matvec branch everywhere it is asked, because
/// `qmv_kn` answers for these kinds too: they have a K and an N like any other
/// matvec. Falling into the dense shape on the strength of that launches them
/// over the TOKEN count rather than the sorted one, which computes the first
/// tile correctly and leaves the rest of the stack holding whatever the pool
/// did. The model still produces text.
bool is_routed(Kind k) {
    return k == Kind::ExpertGate || k == Kind::ExpertUp || k == Kind::ExpertDown;
}

/// The routed matmul's column tile, or 0 when the batch stays a matvec.
int gemma4_moe_qmm_bn(Kind k, const Gemma4Geometry& g, int rows, int layer) {
    if (!is_routed(k) || gemma4_moe_tile_rows(g, rows) <= 1) return 0;
    const KN kn = qmv_kn(k, g, layer);
    if (kn.N == 0) return 0;
    return qmm_bn(kn.N, gemma4_moe_sorted_rows(g, rows));
}

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
        // No weights: the gather is pure dataflow.
        case Kind::RowGather:           return Kernel::G4RowGather;
        case Kind::EmbedGather:         return Kernel::EmbedGather;
        // NOT EmbedGather: that key binds `shared_embedding`, and this gathers
        // the SECOND table. Sharing the key silently read the token embedding
        // with the PLE table's row pitch, which is 8960 wide against 1536.
        case Kind::PleTokenGather:      return Kernel::G4PleTokenGather;
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
        case Kind::PostAttnResidual:    return Kernel::G4AttnPostResidual;
        case Kind::FfnNorm:             return Kernel::G4FfnPreNorm;
        case Kind::PostFfnResidual:     return Kernel::G4FfnPostResidual;
        case Kind::PleResidualScaled:   return Kernel::G4PleResidualScaled;
        case Kind::FinalRms:            return Kernel::FinalRms;
        case Kind::QNorm:               return Kernel::QNorm;
        case Kind::KNorm:               return Kernel::KNorm;

        case Kind::PleProjNorm:         return Kernel::G4PleProjNorm;
        case Kind::RopeQ:               return Kernel::Rope;
        case Kind::RopeK:               return Kernel::RopeK;
        case Kind::KvAppend:            return Kernel::KvAppend;
        case Kind::Sdpa:                return Kernel::Sdpa;
        case Kind::VNorm:
        case Kind::VNormFromK:          return Kernel::G4VNorm;
        // ── the mixture ──
        case Kind::RouterGemv:          return Kernel::G4Router;
        case Kind::RouterNorm:          return Kernel::G4RouterNorm;
        case Kind::RouterTopK:          return Kernel::G4RouterTopK;
        case Kind::MoeNorm:             return Kernel::G4MoeNorm;
        case Kind::DenseBranchNorm:     return Kernel::G4DenseBranchNorm;
        case Kind::MoeBranchNorm:       return Kernel::G4MoeBranchNorm;
        case Kind::ExpertGate:          return Kernel::G4ExpertGate;
        case Kind::ExpertUp:            return Kernel::G4ExpertUp;
        case Kind::ExpertDown:          return Kernel::G4ExpertDown;
        case Kind::ExpertGeglu:         return Kernel::G4ExpertGeglu;
        case Kind::ExpertSort:          return Kernel::G4MoeSort;
        case Kind::ExpertGather:        return Kernel::G4MoeGather;
        case Kind::ExpertCombine:       return Kernel::G4ExpertCombine;
        case Kind::BranchAdd:           return Kernel::G4BranchAdd;
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
        case Kind::FfnNorm:
        case Kind::PleProjNorm:
        // The mixture's four extra norms all run the one `rms_single_row`;
        // only the tensor they bind differs, which is `shared_kind`'s question.
        case Kind::RouterNorm:
        case Kind::MoeNorm:
        case Kind::DenseBranchNorm:
        case Kind::MoeBranchNorm:    return Kernel::Rms;
        // An ordinary dense matvec: hidden -> n_experts, no bias.
        case Kind::RouterGemv:       return Kernel::QmvGate;
        case Kind::PleTokenGather:   return Kernel::EmbedGather;
        case Kind::PleProjGemv:
        case Kind::PleGateGemv:
        case Kind::PleProjLayerGemv: return Kernel::QmvGate;
        default:                     return shared_kind(k);
    }
}

/// The pipeline a gemma4 dispatch runs on at M>1.
///
/// Distinct from `pso_for` for exactly the kinds whose KERNEL changes with the
/// batch -- the projections (matvec at M=1, matmul at M>1) and the four that
/// read per-row IO (embed, rope, kv append, attention). Everything else is the
/// identical kernel at a wider `n`, so it falls through to `pso_for` rather than
/// being restated here, where a copy could drift.
///
/// The tile choice mirrors `launch_shape_mb`'s exactly, because a grid computed
/// for one tiling against a pipeline compiled for another is not a crash -- it
/// is wrong numbers.
Pso pso_for_mb(const Dispatch& d, const Gemma4Geometry& g, int rows,
               const DecodeStepPsos& base, const MultiBatchPsos& mb,
               const Gemma4Psos& g4, int head_rows) {
    const int N = rows < 1 ? 1 : rows;
    const int S = head_rows < 1 ? N : (head_rows < N ? head_rows : N);

    // The routed projections' tiling, asked with the SAME numbers
    // `launch_shape_mb` uses: a grid computed for one tiling against a pipeline
    // compiled for another is not a crash, it is wrong numbers.
    if (is_routed(d.kind)) {
        if (const int bn = gemma4_moe_qmm_bn(d.kind, g, N, d.layer); bn > 0) {
            const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
            if (g4.qmm_routed[slot].valid()) return g4.qmm_routed[slot];
        }
        return g4.qmv_routed;
    }
    if (const KN kn = qmv_kn(d.kind, g, d.layer); kn.N != 0) {
        const int M = gemma4_qmm_rows(d.kind == Kind::LmHead ? S : N);
        const int bm = qmm_bm(M);
        const int bn = qmm_bn(kn.N, M);
        const int wide = qmm_bm_slot(bm);
        // No split-K here either; see `launch_shape_mb`.
        const int split = 0;
        if (split > 1 && mb.qmm_t_splitk[wide].valid()) return mb.qmm_t_splitk[wide];
        if (bn > 0) {
            const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
            if (mb.qmm_t[wide][slot].valid()) return mb.qmm_t[wide][slot];
        }
        return pso_for(d, g, base, g4);  // the matvec, when no tiling applies
    }

    switch (d.kind) {
        case Kind::EmbedGather:
        case Kind::PleTokenGather:
            return g4.embed_scaled_mb;
        case Kind::RopeQ:
        case Kind::RopeK:
            return g4.rope_prop_mb;
        // The PLE gate is the one elementwise kernel whose operands have
        // different row pitches at M>1.
        case Kind::PleGeglu:
            return g4.geglu_strided;
        case Kind::KvAppend:
            return mb.kv_append_paged;
        // Paged, and windowed by the same slot for both attention types -- the
        // head width is still what picks the instantiation.
        case Kind::Sdpa:
            return d.sliding ? mb.sdpa_paged : mb.sdpa_paged_d512;
        default:
            return pso_for(d, g, base, g4);
    }
}

/// Whether this K needs the bounds-checked matvec.
///
/// The aligned kernels reduce K in whole blocks of `32 lanes * (32/bits) values
/// * 2 packs` and check nothing: a K that is not a whole number of them reads
/// the next row's bytes into the last block of every output row. At the end of
/// a tensor that is a NaN -- which is how gemma-4-26B was caught -- and in the
/// middle of one it is a plausible wrong number.
bool qmv_needs_tail(int K, const AffineFormat& fmt) {
    const int block = 32 * (32 / (fmt.bits > 0 ? fmt.bits : 4)) * 2;
    return K % block != 0;
}

bool gemma4_uses_alt_quant(Kind k) {
    return k == Kind::QmvGate || k == Kind::QmvUp || k == Kind::QmvDown ||
           k == Kind::RouterGemv;
}

Pso pso_for(const Dispatch& d, const Gemma4Geometry& g, const DecodeStepPsos& base,
            const Gemma4Psos& g4) {
    // Asked before the table, because the table's answer for every one of these
    // kinds is the ALIGNED matvec and the alignment is a property of K, not of
    // the kind. The routed projections are excluded: their kernel is already the
    // bounds-checked one.
    if (!is_routed(d.kind)) {
        if (const KN kn = qmv_kn(d.kind, g, d.layer); kn.N != 0) {
            const bool alt = gemma4_uses_alt_quant(d.kind) && g.has_alt_quant();
            const AffineFormat& fmt = alt ? g.ffn_quant : g.quant;
            if (qmv_needs_tail(kn.K, fmt)) return alt ? g4.qmv_tail_alt : g4.qmv_tail;
        }
    }
    switch (d.kind) {
        // The gemma4-only kernels, which the shared table has no entry for.
        case Kind::GegluTanh:
        case Kind::PleGeglu:     return g4.geglu_tanh;
        case Kind::LayerScalar:  return g4.layer_scalar;
        case Kind::FinalSoftcap: return g4.logit_softcap;
        case Kind::RowGather:    return g4.row_gather;
        case Kind::PleCombine:   return g4.ple_combine;
        case Kind::VNorm:
        case Kind::VNormFromK:   return g4.vnorm;
        // ── the mixture ──
        // GeGLU, not SwiGLU: the dense activation kernel over the sorted stack.
        case Kind::ExpertGeglu:  return g4.geglu_tanh;
        case Kind::RouterTopK:   return g4.router_topk;
        case Kind::ExpertSort:   return g4.moe_sort;
        case Kind::ExpertGather: return g4.moe_gather;
        case Kind::ExpertCombine: return g4.moe_combine;
        // The one add this family does NOT fuse into a norm: the two branches
        // meet before the sandwich's closing norm, not at it.
        case Kind::BranchAdd:    return g4.residual_add;
        case Kind::ExpertGate:
        case Kind::ExpertUp:
        case Kind::ExpertDown:   return g4.qmv_routed;
        // Both gathers are scaled -- by sqrt(hidden) and sqrt(ple_dim) -- and
        // the scale cannot be folded into the weights, which the LM head shares.
        case Kind::EmbedGather:
        case Kind::PleTokenGather: return g4.embed_scaled;
        // Partial rotary over the whole head; see rope.metal.
        case Kind::RopeQ:
        case Kind::RopeK:        return g4.rope_prop;
        // The norm sandwich's closing norm and its residual add, in one.
        case Kind::PostAttnResidual:
        case Kind::PostFfnResidual:  return g4.rms_residual;
        case Kind::PleResidualScaled: return g4.rms_residual_scaled;
        // The one place the attention type picks the pipeline rather than a
        // constant: the two head widths are separate instantiations.
        case Kind::Sdpa:         return d.sliding ? g4.sdpa_swa_d256 : g4.sdpa_swa_d512;
        default:                 return base[pso_kind(d.kind)];
    }
}

/// The M>1 (prefill) launch shape for a gemma4 dispatch.
///
/// Every kernel gemma4 needs at M>1 already exists: the shared ones in
/// qwen3.5's `decode_dispatch_mb.hpp`, and gemma4's own five, which carry no row
/// structure and widen by counting M*width. So this is a shape function, not a
/// second set of kernels -- the arithmetic is the same one the decode path runs.
///
/// `rows` is the token count in the batch. At rows==1 each case reduces to
/// `launch_shape`'s, which the test asserts rather than assuming.
void launch_shape_mb(const Dispatch& d, const Gemma4Geometry& g, int rows, Grid& grid,
                     Threadgroup& tg, int head_rows) {
    const int L = d.layer;
    const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;
    const int N = rows < 1 ? 1 : rows;
    // The tail's row count: what `RowGather` compacted. 0 means every row, for
    // a caller that reads them all.
    const int S = head_rows < 1 ? N : (head_rows < N ? head_rows : N);

    // The projections become matmuls. Tiling is chosen by the shared selectors,
    // so gemma4 gets whatever qwen3.5's prefill measured its way to -- except
    // for the row count, which is PADDED to a whole tile here.
    //
    // `qmm_bn` refuses a batch that does not fill one (`N % bm != 0`), and the
    // fallback is a per-row GEMV that re-reads the whole weight for every row.
    // On a 2.6 GB model that is the difference between amortizing the weights
    // and not: measured at 8 lanes, this family's throughput was 1.12x its
    // single-stream rate, and the GEMM only engaged at a batch of exactly 16.
    // The padding rows compute discardable values into pool rows the fire does
    // not use, which is the same trade the prefill's strided GEMM already makes.
    if (is_routed(d.kind)) {
        const KN kn = qmv_kn(d.kind, g, L);
        const int sorted = gemma4_moe_sorted_rows(g, N);
        if (const int bn = gemma4_moe_qmm_bn(d.kind, g, N, L); bn > 0) {
            qmm_t_dispatch(kn.N, sorted, bn, shared_kernels::kMoeTileRows, grid, tg);
            return;
        }
        // One sorted row per (token, slot) pair, and the expert axis is gone --
        // the pair's expert is `row_expert[p]`, not `tid.z`.
        shared_kernels::routed_qmv_dispatch(kn.N, 1, grid, tg, sorted);
        return;
    }
    if (const KN kn = qmv_kn(d.kind, g, L); kn.N != 0) {
        const int M = gemma4_qmm_rows(d.kind == Kind::LmHead ? S : N);
        const int bm = qmm_bm(M);
        const int bn = qmm_bn(kn.N, M);
        // NO split-K. The split GEMM accumulates partial sums into a split
        // buffer and needs a reduce pass to fold them; qwen3.5 has both, this
        // family has neither -- so a split dispatch here wrote partials nobody
        // summed. It never showed up because the split only engages at
        // `qmm_bn != 0`, which needed a batch of exactly 16, and no test fired
        // one: measured against mlx-lm, a 16-row prefill answered 147040 where
        // the oracle says 476.
        const int split = 0;
        if (split > 1) {
            qmm_t_splitk_dispatch(kn.N, M, bm, split, grid, tg);
        } else if (bn > 0) {
            qmm_t_dispatch(kn.N, M, bn, bm, grid, tg);
        } else {
            qmv_mb_dispatch(kn.N, M, grid, tg);
        }
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            embed_mb_dispatch(g.hidden, N, grid, tg);
            return;
        case Kind::PleTokenGather:
            embed_mb_dispatch(g.n_layers * g.per_layer_emb_dim, N, grid, tg);
            return;
        case Kind::AttnNorm: case Kind::FfnNorm:
        case Kind::PostAttnResidual: case Kind::PostFfnResidual:
        case Kind::PleResidualScaled:
            rms_mb_dispatch(g.hidden, 1, N, grid, tg);
            return;
        // The tail runs on the SAMPLED rows, which `RowGather` compacted.
        case Kind::FinalRms:
            rms_mb_dispatch(g.hidden, 1, S, grid, tg);
            return;
        case Kind::RowGather:
            grid = Grid{std::uint32_t(g.hidden), std::uint32_t(S), 1};
            tg = Threadgroup{64, 1, 1};
            return;
        case Kind::QNorm:
            rms_mb_dispatch(hd, g.n_q_heads, N, grid, tg);
            return;
        case Kind::KNorm:
            rms_mb_dispatch(hd, g.n_kv_heads_of(L), N, grid, tg);
            return;
        case Kind::PleProjNorm:
            rms_mb_dispatch(g.per_layer_emb_dim, g.n_layers, N, grid, tg);
            return;
        case Kind::VNorm:
        case Kind::VNormFromK:
            rms_mb_dispatch(hd, g.n_kv_heads_of(L), N, grid, tg);
            return;

        // ── the mixture ──
        // The four extra norms are hidden-wide over every token, like the two
        // the dense path already has.
        case Kind::RouterNorm:
        case Kind::MoeNorm:
        case Kind::DenseBranchNorm:
        case Kind::MoeBranchNorm:
            rms_mb_dispatch(g.hidden, 1, N, grid, tg);
            return;
        case Kind::RouterTopK:
            shared_kernels::router_topk_dispatch(g.n_experts, grid, tg, N);
            return;
        case Kind::ExpertSort:
            shared_kernels::moe_route_sort_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertGather:
            shared_kernels::moe_route_rows_dispatch(g.hidden, gemma4_moe_sorted_rows(g, N), grid, tg);
            return;
        case Kind::ExpertGeglu:
            // The slot axis is gone -- a sorted row IS a slot -- so this is the
            // dense elementwise shape over a taller batch.
            elementwise_mb_dispatch(g.moe_intermediate, gemma4_moe_sorted_rows(g, N), grid,
                                    tg);
            return;
        case Kind::ExpertCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, grid, tg, N);
            return;
        // NOT the norms' shape: `residual_add` reads ONE element per thread
        // where `rms_norm` reads four, so borrowing the norm shape would leave
        // three quarters of the sum holding whatever the pool did.
        case Kind::BranchAdd:
            elementwise_mb_dispatch(g.hidden, N, grid, tg);
            return;
        case Kind::RopeQ:
            rope_mb_dispatch(g.rotary_dims_of(L), g.n_q_heads, N, grid, tg);
            return;
        case Kind::RopeK:
            rope_mb_dispatch(g.rotary_dims_of(L), g.n_kv_heads_of(L), N, grid, tg);
            return;
        case Kind::KvAppend:
            kv_append_mb_dispatch(hd, g.n_kv_heads_of(L), N, grid, tg);
            return;
        case Kind::Sdpa:
            sdpa_sliding_dispatch(g.n_q_heads, grid, tg, N);
            return;
        case Kind::LayerScalar:
            elementwise_mb_dispatch(g.hidden, N, grid, tg);
            return;
        case Kind::GegluTanh:
            elementwise_mb_dispatch(L >= 0 ? g.intermediate_of(L) : g.intermediate, N, grid, tg);
            return;
        case Kind::PleGeglu:
            // One thread per (channel, token row): the strided kernel indexes
            // both, because its `up` operand strides by the whole PLE table.
            grid = Grid{std::uint32_t(g.per_layer_emb_dim), std::uint32_t(N), 1};
            tg = Threadgroup{std::uint32_t(g.per_layer_emb_dim < 256 ? g.per_layer_emb_dim : 256),
                             1, 1};
            return;
        case Kind::PleCombine:
            elementwise_mb_dispatch(g.n_layers * g.per_layer_emb_dim, N, grid, tg);
            return;
        case Kind::FinalSoftcap:
            // Every SAMPLED row. This used to cap one row on the grounds that
            // only the sampled one matters -- but the row a prefill samples is
            // the LAST, and the dispatch started at offset 0, so it capped a row
            // nobody reads and left the one everybody does uncapped. `RowGather`
            // has since made "the sampled rows" a dense range, so all of them
            // and only them is now the same statement.
            elementwise_mb_dispatch(g.vocab, S, grid, tg);
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

void launch_shape(const Dispatch& d, const Gemma4Geometry& g, Grid& grid, Threadgroup& tg) {
    const int L = d.layer;
    const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;

    if (is_routed(d.kind)) {
        const KN kn = qmv_kn(d.kind, g, L);
        shared_kernels::routed_qmv_dispatch(kn.N, 1, grid, tg, gemma4_moe_sorted_rows(g, 1));
        return;
    }
    if (const KN kn = qmv_kn(d.kind, g, L); kn.N != 0) {
        // The shared `qmv_dispatch`, deliberately, rather than a second answer:
        // `affine_qmv_fast` is 2 simdgroups of 4 rows each and reduces K with
        // `simd_sum`, so it needs tg {32,2,1} against a grid of TOTAL THREADS.
        // Stating it as "one threadgroup per 8 output rows" -- grid {1,N/8,1},
        // tg {64,1,1} -- gives every threadgroup exactly one thread: `simd_gid`
        // is always 0, so rows 4..7 of every 8 are never written, and `simd_lid`
        // is always 0, so the reduction covers 1/32 of K. That is the defect
        // behind "half the vocabulary is exactly zero".
        qmv_dispatch(kn.N, grid, tg);
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        case Kind::PleTokenGather:
            elementwise_dispatch(g.n_layers * g.per_layer_emb_dim, grid, tg);
            return;
        case Kind::RowGather:
            grid = Grid{std::uint32_t(g.hidden), 1, 1};
            tg = Threadgroup{64, 1, 1};
            return;
        // The residual kinds sit with the norms because in this family they
        // ARE norms: they run the FUSED `rms_residual`, which reads 4 elements
        // per thread like any other. A plain `residual_add` here would need one
        // thread per element instead.
        case Kind::AttnNorm: case Kind::FfnNorm: case Kind::FinalRms:
        case Kind::PostAttnResidual: case Kind::PostFfnResidual:
        case Kind::PleResidualScaled:
            rms_dispatch(g.hidden, 1, grid, tg);
            return;
        case Kind::QNorm:
            rms_dispatch(hd, g.n_q_heads, grid, tg);
            return;
        case Kind::KNorm:
            rms_dispatch(hd, g.n_kv_heads_of(L), grid, tg);
            return;
        // Only `per_layer_projection_norm` is the ple_dim-wide norm, over
        // n_layers rows; `post_per_layer_input_norm` is hidden-wide and now
        // rides inside `PleResidualScaled`.
        case Kind::PleProjNorm:
            rms_dispatch(g.per_layer_emb_dim, g.n_layers, grid, tg);
            return;
        case Kind::VNorm:
        case Kind::VNormFromK:
            vnorm_dispatch(g.n_kv_heads_of(L), hd, grid, tg);
            return;

        // ── the mixture ──
        case Kind::RouterNorm:
        case Kind::MoeNorm:
        case Kind::DenseBranchNorm:
        case Kind::MoeBranchNorm:
            rms_dispatch(g.hidden, 1, grid, tg);
            return;
        case Kind::RouterTopK:
            shared_kernels::router_topk_dispatch(g.n_experts, grid, tg, 1);
            return;
        case Kind::ExpertSort:
            shared_kernels::moe_route_sort_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertGather:
            shared_kernels::moe_route_rows_dispatch(g.hidden, gemma4_moe_sorted_rows(g, 1), grid, tg);
            return;
        case Kind::ExpertGeglu:
            elementwise_dispatch(g.moe_intermediate * gemma4_moe_sorted_rows(g, 1), grid, tg);
            return;
        case Kind::ExpertCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, grid, tg, 1);
            return;
        case Kind::BranchAdd:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        // `rope_neox_prop_decode` takes the ROTATED PAIR COUNT as grid.x and
        // reads the pair offset off head_dim, so a full-attention layer rotates
        // 64 pairs of a 512-wide head rather than 64 pairs of a 128-wide prefix.
        case Kind::RopeQ:
            rope_dispatch(g.rotary_dims_of(L), g.n_q_heads, grid, tg);
            return;
        case Kind::RopeK:
            rope_dispatch(g.rotary_dims_of(L), g.n_kv_heads_of(L), grid, tg);
            return;
        case Kind::KvAppend:
            kv_append_dispatch(hd, g.n_kv_heads_of(L), grid, tg);
            return;
        case Kind::Sdpa:
            sdpa_sliding_dispatch(g.n_q_heads, grid, tg);
            return;
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
                        const Gemma4Psos& g4, int ordinal_base,
                        const DecodeStepPsos* base_alt) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        // The dense FFN and the router are 8-bit on a checkpoint whose rest is
        // 4. One pipeline for both would run a 4-bit kernel over 8-bit bytes:
        // it dispatches, it is fast, and every token is wrong.
        const DecodeStepPsos& B =
            (base_alt != nullptr && gemma4_uses_alt_quant(d.kind)) ? *base_alt : base;
        se.set_pso(pso_for(d, g, B, g4));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        // A barrier after every dispatch except inside a concurrency run: the
        // last member of a run carries it for the whole group.
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

void encode_gemma4_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const Gemma4Geometry& g, int rows,
                           const DecodeStepPsos& base, const MultiBatchPsos& mb,
                           const Gemma4Psos& g4, int ordinal_base, int head_rows,
                           const DecodeStepPsos* base_alt, const MultiBatchPsos* mb_alt,
                           std::size_t begin, std::size_t end) {
    // Deliberately the same walk as `encode_gemma4_step`, differing only in
    // which shape and which pipeline each dispatch gets. The DAG, the ordering
    // and the concurrency runs are properties of the MODEL, not of the batch
    // size, so re-deriving them here would be a second answer to a question that
    // already has one.
    // Off the WHOLE dag, not the slice: a run's extent is the model's, and
    // recomputing it per segment would let a boundary invent a barrier.
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    if (end == 0 || end > dag.size()) end = dag.size();
    for (std::size_t i = begin; i < end; ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape_mb(d, g, rows, grid, tg, head_rows);
        const bool alt = base_alt != nullptr && mb_alt != nullptr &&
                         gemma4_uses_alt_quant(d.kind);
        se.set_pso(pso_for_mb(d, g, rows, alt ? *base_alt : base, alt ? *mb_alt : mb, g4,
                              head_rows));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        if (i + 1 >= end || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::gemma4
