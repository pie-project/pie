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

#include <algorithm>
#include <cstdlib>

#include "../../model/qwen3_5/decode_dispatch.hpp"
#include "../../model/qwen3_5/decode_dispatch_mb.hpp"
#include "decode_consts.hpp"
#include "scratch.hpp"

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
using pie::metal::sdpa_paged_tiled_dispatch;
using pie::metal::sdpa_should_tile;

namespace {
int llama_sdpa_simdgroups(const LlamaGeometry& g) {
    return g.head_dim == 64 && g.kv_page_size == 32 ? 8 : 32;
}

int llama_dense_qmm_bm(int rows, int requests) {
    // A 64-request decode fleet has enough independent rows that BM=32's
    // extra threadgroups beat BM=64's weight reuse. A 64-token prefill is one
    // request and keeps BM=64; row count alone cannot distinguish the two.
    if (rows == 64 && requests >= 64) return 32;
    return qmm_bm(rows);
}

bool llama_fp16_qmm(const LlamaGeometry& g, Kind k, int rows, int requests) {
    return fp16_qmm() && !g.is_moe() && g.quant.bits == 4 && g.quant.group == 64 &&
           llama_qmm_bn(k, g, rows, requests) > 0;
}

bool llama_fp16_cast_before(Kind k) {
    switch (k) {
        case Kind::QmvQ:
        case Kind::QmvO:
        case Kind::QmvGate:
        case Kind::QmvDown:
        case Kind::LmHead:
            return true;
        default:
            return false;
    }
}
}

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
        // Weightless: `weight_binds` has no case for these, which is the
        // point -- the reordering moves rows and indices, not parameters.
        case Kind::ExpertSort:      return Kernel::LlMoeSort;
        case Kind::ExpertGather:    return Kernel::LlMoeGather;
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
        // `pso_for` answers for these three off the family's own PSOs and
        // never reaches here -- but naming them keeps the fall-through value
        // meaning "nothing claimed this kind".
        case Kind::ExpertSort:    return Kernel::LlMoeSort;
        case Kind::ExpertGather:  return Kernel::LlMoeGather;
        case Kind::ExpertCombine: return Kernel::LlMoeCombine;
        case Kind::EmbedGather: return Kernel::EmbedGather;
        case Kind::KvAppend:    return Kernel::KvAppend;
        case Kind::Sdpa:        return Kernel::Sdpa;
        case Kind::RowGather:   return Kernel::G4RowGather;
        case Kind::Argmax:      return Kernel::Argmax;
    }
    return Kernel::Rms;
}

Pso pso_for(const Dispatch& d, const LlamaGeometry& g, const DecodeStepPsos& base,
            const LlamaPsos& ll, const MultiBatchPsos* mb, int rows, int head_rows,
            int requests) {
    const int R = rows < 1 ? 1 : rows;
    const int S = head_rows < 1 ? R : (head_rows < R ? head_rows : R);

    // The GEMM, when the batch fills a tile. Asked first and asked with the
    // same numbers `launch_shape` uses, because the two answers must agree.
    if (mb != nullptr) {
        const int m = d.kind == Kind::LmHead ? S : R;
        if (const int bn = llama_qmm_bn_for(d.kind, g, m, requests); bn > 0) {
            const int split = llama_qmm_split(d.kind, g, m, requests);
            const int wide = qmm_bm_slot(llama_dense_qmm_bm(m, requests));
            const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
            // M1's native FP16 simdgroup MMA is substantially faster than
            // BF16. Keep storage and every surrounding kernel in BF16; only
            // the dense g64/b4 GEMM tile is cast to half on load. Routed
            // models stay BF16 because the extra rounding can flip top-k.
            const bool fp16 = llama_fp16_qmm(g, d.kind, m, requests);
            // The split form first, and asked with the same `m` `launch_shape`
            // uses: the split grid is `split` deep in z and writes partials,
            // the plain one is not and writes the output, so disagreeing here
            // leaves the projection's real buffer untouched.
            if (split > 1) {
                const Pso split_pso =
                    fp16 ? (d.kind == Kind::QmvV
                                  ? mb->qmm_t_splitk_fp16_precast_f32[wide]
                                  : mb->qmm_t_splitk_fp16_precast[wide])
                         : (d.kind == Kind::QmvV
                                  ? mb->qmm_t_splitk_f32[wide]
                                  : mb->qmm_t_splitk[wide]);
                if (split_pso.valid()) return split_pso;
            }
            // No bias table: llama's projections have no bias tensor, which is
            // the one place this family is simpler than gpt-oss.
            if (fp16 && mb->qmm_t_fp16_precast[wide][slot].valid()) {
                return mb->qmm_t_fp16_precast[wide][slot];
            }
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
                    // The table form first: a llama3 checkpoint's frequencies
                    // are not a geometric series, and the base kernel would
                    // rotate every dimension by the wrong angle past the
                    // original context while running perfectly well.
                    if (g.rope_freq_table && ll.rope_freqs_mb.valid()) {
                        return ll.rope_freqs_mb;
                    }
                    if (mb->rope_mb.valid()) return mb->rope_mb;
                    break;
                default:
                    break;
            }
        }
    }

    switch (d.kind) {
        case Kind::RopeQ:
        case Kind::RopeK:
            if (g.rope_freq_table && ll.rope_freqs.valid()) return ll.rope_freqs;
            break;
        default:
            break;
    }

    switch (d.kind) {
        // The family's own PSOs: a 128-wide head, and the routed set.
        case Kind::Sdpa:
            // Asked with the same row count `launch_shape` uses, because the
            // two answers must agree: the tiled kernel's grid is N/32 tall and
            // the per-row kernel's is N, so choosing one here and shaping the
            // other there launches a thirty-second of the attention.
            if (!g.paged_kv_enabled) return ll.sdpa;
            if (sdpa_should_tile(R, requests)) return ll.sdpa_paged_tiled;
            if (llama_sdpa_simdgroups(g) == 8 && ll.sdpa_paged_sg8.valid())
                return ll.sdpa_paged_sg8;
            return ll.sdpa_paged;
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
        case Kind::ExpertSort:    return ll.moe_sort;
        case Kind::ExpertGather:  return ll.moe_gather;
        case Kind::ExpertCombine: return ll.moe_combine;
        case Kind::ExpertGate:
        case Kind::ExpertUp:
        case Kind::ExpertDown: {
            // Asked with the same row count `launch_shape` uses, because the
            // two answers must agree: a matmul pipeline under a matvec grid
            // reads `tile_expert` off the end of the routing.
            const int bn = llama_moe_qmm_bn(d.kind, g, R);
            if (bn == 0) return ll.qmv_routed;
            const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
            const int bm = shared_kernels::moe_bm_slot(llama_moe_tile_rows(g, R));
            return ll.qmm_routed[bm][slot].valid() ? ll.qmm_routed[bm][slot]
                                                   : ll.qmv_routed;
        }
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

int llama_qmm_rows(const LlamaGeometry& g, int rows, int requests) {
    const int n = rows < 1 ? 1 : rows;
    if (n < qmm_min_batch(g.is_moe())) return n;
    const int bm = llama_dense_qmm_bm(n, requests);
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

int llama_qmm_bn(Kind k, const LlamaGeometry& g, int rows, int requests) {
    if (!llama_is_dense_proj(k)) return 0;
    const KN kn = qmv_kn(k, g);
    if (kn.N == 0) return 0;
    const int bn = qmm_bn(kn.N, llama_qmm_rows(g, rows, requests),
                          qmm_min_batch(g.is_moe()));
    if (k == Kind::LmHead && bn > 0) return 32;
    return bn;
}

int llama_qmm_split(Kind k, const LlamaGeometry& g, int rows, int requests) {
    static const bool off = std::getenv("PIE_METAL_NO_SPLITK") != nullptr;
    if (off) return 1;
    if (llama_qmm_bn(k, g, rows, requests) <= 0) return 1;
    const KN kn = qmv_kn(k, g);
    // lm_head has thousands of output tiles of its own and needs no split; it
    // is also the one projection whose partials would be hundreds of megabytes.
    if (kn.N > kQmmSplitMaxOut) return 1;
    if (g.quant.bits == 4 && g.quant.group == 64 &&
        (k == Kind::QmvGate || k == Kind::QmvUp)) {
        const int bm = llama_dense_qmm_bm(rows, requests);
        const int tiles =
            (kn.N / kQmmSplitBN) *
            ((llama_qmm_rows(g, rows, requests) + bm - 1) / bm);
        // This checkpoint's gate/up grid already has 256 threadgroups. A
        // second K partition only adds a full-output reduce; measured at
        // batch 32 it is 1.7% slower.
        if (tiles >= 256) return 1;
    }
    return qmm_split_k(kn.N, llama_qmm_rows(g, rows, requests), kn.K,
                       llama_dense_qmm_bm(rows, requests));
}

/// The column tile a dispatch actually launches.
///
/// Two functions because there are two questions and only one of them is
/// recursive. `llama_qmm_bn` answers "is there a tile at all, and how wide if
/// the split is behind it", and `llama_qmm_split` needs that answer to decide
/// whether to split -- so the width cannot depend on the split inside it.
///
/// This is the width the encoder and the grid use, and it asks the split
/// first. `qmm_bn`'s widest-tile rule is correct *because* the split supplies
/// the threadgroups a wide tile gives up; a dispatch this family does NOT
/// split has no such supply, and the same rule that was wrong for gemma4 and
/// gpt-oss is wrong for it. Measured on llama-3.2-1B's own projections at
/// M=448, BM=64, FP16 GEMM, GFLOP/s:
///
///     K     N        BN=16    BN=32    BN=64
///     2048  2048      5832    *6070*    5660
///     2048  8192      6115    *6411*    6166
///     8192  2048      5929    *6180*    5766
///
/// At 448 rows this checkpoint splits k_proj and v_proj and nothing else, so
/// the other five projections per layer -- 73.7% of the prefill's GPU time --
/// were taking a width that is the best of none of the three.
///
/// `LmHead` keeps its own answer: `llama_qmm_bn` pins it to 32 already, and
/// the vocabulary is wide enough that both rules agree anyway.
int llama_qmm_bn_for(Kind k, const LlamaGeometry& g, int rows, int requests) {
    const int bn = llama_qmm_bn(k, g, rows, requests);
    if (bn <= 0 || k == Kind::LmHead) return bn;
    if (llama_qmm_split(k, g, rows, requests) > 1) return bn;
    const KN kn = qmv_kn(k, g);
    const int unsplit = qmm_bn_unsplit(int(kn.N), llama_qmm_rows(g, rows, requests),
                                       qmm_min_batch(g.is_moe()));
    return unsplit > 0 ? unsplit : bn;
}

std::size_t llama_splitk_partial_elems(const LlamaGeometry& g, int max_rows) {
    const Kind dense[] = {
        Kind::QmvQ, Kind::QmvK, Kind::QmvV, Kind::QmvO,
        Kind::QmvGate, Kind::QmvUp, Kind::QmvDown,
    };
    std::size_t one_lane = 0;
    const int limit = max_rows < 1 ? 1 : max_rows;
    for (int rows = 1; rows <= limit; ++rows) {
        for (const Kind k : dense) {
            if (g.is_moe() &&
                (k == Kind::QmvGate || k == Kind::QmvUp || k == Kind::QmvDown)) {
                continue;
            }
            for (const int requests : {1, rows}) {
                const int split = llama_qmm_split(k, g, rows, requests);
                if (split <= 1) continue;
                const KN kn = qmv_kn(k, g);
                one_lane = std::max(
                    one_lane,
                    std::size_t(split) *
                        std::size_t(llama_qmm_rows(g, rows, requests)) *
                        std::size_t(kn.N));
            }
        }
    }
    return std::size_t(kLlamaSplitkConcurrentLanes) * one_lane;
}

int llama_moe_pairs(const LlamaGeometry& g, int rows) {
    return (rows < 1 ? 1 : rows) * (g.is_moe() ? g.experts_per_token : 1);
}

/// The tile the sort pads each expert's run to: 1 leaves the projections
/// matvecs, and `moe_tile_rows`'s width makes them matmuls. One question asked
/// in one place,
/// because the sort, the launch shapes, the pipeline choice and the pool sizer
/// all have to give the same answer -- a sort that padded to 16 under a matvec
/// launched for 8 rows would run the projection over a fraction of its input.
int llama_moe_tile_rows(const LlamaGeometry& g, int rows) {
    if (!g.is_moe()) return 1;
    return shared_kernels::moe_tile_rows(llama_moe_pairs(g, rows), g.n_experts);
}

/// The routed matmul's column tile, or 0 when the batch stays a matvec.
int llama_moe_qmm_bn(Kind k, const LlamaGeometry& g, int rows) {
    if (!is_routed(k) || llama_moe_tile_rows(g, rows) <= 1) return 0;
    const KN kn = qmv_kn(k, g);
    if (kn.N == 0) return 0;
    // Routed, so the routed crossover -- though `moe_should_batch` has already
    // admitted this batch and the sorted count is far above either number.
    return qmm_bn(kn.N, llama_moe_sorted_rows(g, rows), qmm_min_batch(true));
}

void launch_shape(const Dispatch& d, const LlamaGeometry& g, Grid& grid, Threadgroup& tg,
                  int rows, int head_rows, int requests) {
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
    // The routed projections run on the SORTED rows, whose count is neither R
    // nor R*k: the sort pads every expert's run to a whole tile. Asked before
    // the shared matvec branch for the same reason `is_routed` is -- `qmv_kn`
    // answers for these kinds too, and the dense shape would launch them over
    // the token count.
    if (is_routed(d.kind)) {
        const KN kn = qmv_kn(d.kind, g);
        const int sorted = llama_moe_sorted_rows(g, R);
        if (const int bn = llama_moe_qmm_bn(d.kind, g, R); bn > 0) {
            qmm_t_dispatch(kn.N, sorted, bn, llama_moe_tile_rows(g, R), grid, tg);
            return;
        }
        // One sorted row per (token, slot) pair, and the expert axis is gone --
        // the pair's expert is `row_expert[p]`, not `tid.z`.
        routed_qmv_dispatch(kn.N, 1, grid, tg, sorted);
        return;
    }
    if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
        const int m = d.kind == Kind::LmHead ? S : R;
        // Once the batch fills a tile, a dense projection becomes a matmul.
        // The matvec re-reads the ENTIRE weight for every row, so on a prefill
        // it is the difference between amortizing the weights and not.
        if (const int bn = llama_qmm_bn_for(d.kind, g, m, requests); bn > 0) {
            // The row block is asked of the BATCH, not of the padded count:
            // padding rounds up, and a rounded-up count can land on a wider
            // rung than the one the grid was built for.
            if (const int split = llama_qmm_split(d.kind, g, m, requests); split > 1) {
                qmm_t_splitk_dispatch(kn.N, llama_qmm_rows(g, m, requests),
                                      llama_dense_qmm_bm(m, requests), split, grid, tg);
                return;
            }
            qmm_t_dispatch(kn.N, llama_qmm_rows(g, m, requests), bn,
                           llama_dense_qmm_bm(m, requests), grid, tg);
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
            if (sdpa_should_tile(R, requests)) {
                sdpa_paged_tiled_dispatch(g.n_q_heads, R, grid, tg);
                return;
            }
            if (const int sg = llama_sdpa_simdgroups(g); sg < 32) {
                const std::uint32_t threads = std::uint32_t(sg * 32);
                grid = Grid{std::uint32_t(g.n_q_heads) * threads,
                            std::uint32_t(R), 1};
                tg = Threadgroup{threads, 1, 1};
                return;
            }
            sdpa_paged_dispatch(g.n_q_heads, R, grid, tg);
            return;
        case Kind::SiluMul:
            elementwise_mb_dispatch(g.intermediate, R, grid, tg);
            return;

        // ── routed ──
        case Kind::RouterTopK:
            router_topk_dispatch(g.n_experts, grid, tg, R);
            return;
        // The three routed projections are handled by the `is_routed` branch
        // above, which answers for them: they reach here only if that ever
        // stops being true.
        case Kind::ExpertSort:
            moe_route_sort_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertGather:
            moe_route_rows_dispatch(g.hidden, llama_moe_sorted_rows(g, R), grid, tg);
            return;
        case Kind::ExpertSiluMul:
            // The slot axis is gone -- a sorted row IS a slot -- so this is the
            // dense elementwise shape over a taller batch.
            elementwise_mb_dispatch(g.moe_intermediate, llama_moe_sorted_rows(g, R), grid, tg);
            return;
        case Kind::ExpertCombine:
            expert_combine_dispatch(g.hidden, grid, tg, R);
            return;

        case Kind::Argmax:
            grid = Grid{1024, std::uint32_t(S), 1};
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
                       const MultiBatchPsos* mb, int rows, int head_rows, int requests,
                       std::size_t begin, std::size_t end, bool run_argmax) {
    const std::vector<int> run_ends = llama_run_ends(dag);
    const std::size_t last = std::min(end, dag.size());
    for (std::size_t i = begin; i < last; ++i) {
        const Dispatch& d = dag[i];
        if (d.kind == Kind::Argmax && !run_argmax) continue;
        const int m = d.kind == Kind::LmHead
                          ? (head_rows < 1 ? (rows < 1 ? 1 : rows)
                                           : std::min(head_rows, rows < 1 ? 1 : rows))
                          : (rows < 1 ? 1 : rows);
        if (mb != nullptr && llama_fp16_qmm(g, d.kind, m, requests) &&
            llama_fp16_cast_before(d.kind) && mb->qmm_cast_bf16_f16.valid()) {
            se.set_pso(mb->qmm_cast_bf16_f16);
            se.set_argtable_ordinal(ordinal_base + d.ordinal);
            const std::uint32_t count =
                std::uint32_t(llama_qmm_rows(g, m, requests)) *
                std::uint32_t(qmv_kn(d.kind, g).K);
            se.dispatch(Grid{count, 1, 1}, Threadgroup{256, 1, 1});
            se.barrier();
        }
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg, rows, head_rows, requests);
        se.set_pso(pso_for(d, g, base, ll, mb, rows, head_rows, requests));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        // A split projection is TWO dispatches. The GEMM above wrote `split`
        // partial [M, N] slices into a side buffer and left the projection's
        // real output alone; this sums them. It rides the same argument table
        // -- the partials, the stride and the count are bound there beside the
        // weights -- so it needs no DAG entry of its own, which is what let
        // this exist at all: the split is decided while walking a DAG that is
        // already built, and nothing there can insert a dispatch.
        if (mb != nullptr) {
            if (llama_qmm_split(d.kind, g, m, requests) > 1) {
                const Pso reduce = d.kind == Kind::QmvV
                                       ? mb->qmm_splitk_reduce_f32
                                       : mb->qmm_splitk_reduce;
                se.barrier();
                se.set_pso(reduce);
                se.set_argtable_ordinal(ordinal_base + d.ordinal);
                Grid rg;
                Threadgroup rtg;
                qmm_splitk_reduce_dispatch(
                    qmv_kn(d.kind, g).N, llama_qmm_rows(g, m, requests), rg, rtg);
                se.dispatch(rg, rtg);
                // No trailing barrier inside a concurrency run. Its members
                // own different partial slices, so this reduce can overlap the
                // next member's GEMM. The ordinary run-end barrier still waits
                // for every output before a consumer starts.
            }
        }
        // A barrier after every dispatch except inside a concurrency run: the
        // last member of a run carries it for the whole group.
        if (i + 1 >= last || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::llama
