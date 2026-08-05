// GPT-OSS's step encoder: the DAG, walked with a real encoder.
//
// One choice is per-dispatch rather than per kind: the attention window, which
// follows the layer's type. Everything else is uniform — this family has no
// per-layer head width, no KV sharing and no optional stages.
//
// Barriers follow the same rule the other two encoders use: one after every
// dispatch except inside a run of mutually independent ones.

#include "encode.hpp"

#include "../../batch/decode_abi.hpp"
#include "../qwen3_5/decode_dispatch_mb.hpp"
#include "decode_consts.hpp"

namespace pie::metal::gptoss {

namespace {

/// Dispatches that may run together: same layer, mutually independent, all
/// reading something produced before the group starts and writing distinct
/// values. An explicit list rather than something derived, for the same reason
/// the other families' are — the scratch dataflow does not model the KV pages,
/// so "independent" cannot be read off it.
int concurrency_group(Kind k) {
    switch (k) {
        case Kind::QmvQ:
        case Kind::QmvK:
        case Kind::QmvV:
            return 1;  // all three read the attention norm's output
        case Kind::RopeQ:
        case Kind::RopeK:
            return 2;  // q and k, disjoint
        case Kind::ExpertGate:
        case Kind::ExpertUp:
            return 3;  // both read the FFN norm's output and the same expert ids
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

std::vector<int> gptoss_run_ends(const std::vector<Dispatch>& dag) {
    return concurrent_run_ends(dag);
}

Kernel shared_kind(Kind k) {
    switch (k) {
        // No weights: the gather is pure dataflow.
        case Kind::RowGather:     return Kernel::G4RowGather;
        case Kind::EmbedGather:   return Kernel::EmbedUntied;
        case Kind::AttnNorm:      return Kernel::Rms;        // input_layernorm
        case Kind::FfnNorm:       return Kernel::FfnRms;     // post_attention_layernorm
        case Kind::FinalRms:      return Kernel::FinalRms;
        case Kind::QmvQ:          return Kernel::GoQmvQ;
        case Kind::QmvK:          return Kernel::GoQmvK;
        case Kind::QmvV:          return Kernel::GoQmvV;
        case Kind::QmvO:          return Kernel::GoQmvO;
        case Kind::RopeQ:         return Kernel::Rope;
        case Kind::RopeK:         return Kernel::RopeK;
        case Kind::KvAppend:      return Kernel::KvAppend;
        case Kind::SdpaSink:      return Kernel::GoSdpaSink;
        case Kind::AttnResidual:  return Kernel::Residual;
        case Kind::RouterGemv:    return Kernel::GoRouter;
        case Kind::RouterTopK:    return Kernel::GoRouterTopK;
        case Kind::ExpertSort:    return Kernel::LlMoeSort;
        case Kind::ExpertGather:  return Kernel::LlMoeGather;
        case Kind::ExpertGate:    return Kernel::GoExpertGate;
        case Kind::ExpertUp:      return Kernel::GoExpertUp;
        case Kind::ExpertSwiGlu:  return Kernel::GoSwiGlu;
        case Kind::ExpertDown:    return Kernel::GoExpertDown;
        case Kind::ExpertCombine: return Kernel::LlMoeCombine;
        case Kind::FfnResidual:   return Kernel::LayerOut;
        case Kind::LmHead:        return Kernel::LmHeadUntied;
        case Kind::Argmax:        return Kernel::Argmax;
    }
    return Kernel::Argmax;
}

Pso pso_for(const Dispatch& d, const DecodeStepPsos& base, const GptOssPsos& go) {
    switch (d.kind) {
        // Every projection here is K=2880 -- a whole number of quantization
        // groups but not of any reduction block -- so all of them run the
        // tail-handling matvec, and all but the head are biased.
        case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
            return go.qmv_tail_bias;
        case Kind::LmHead:
            return go.qmv_tail;
        case Kind::RouterGemv:
            return go.qmv_router;
        case Kind::ExpertGate: case Kind::ExpertUp: case Kind::ExpertDown:
            return go.qmv_routed_bias;
        case Kind::RouterTopK:    return go.router_topk;
        case Kind::ExpertSort:    return go.moe_sort;
        case Kind::ExpertGather:  return go.moe_gather;
        case Kind::RowGather:     return go.row_gather;
        case Kind::ExpertSwiGlu:  return go.swiglu;
        case Kind::ExpertCombine: return go.moe_combine;
        case Kind::SdpaSink:      return go.sdpa_sink;
        case Kind::RopeQ: case Kind::RopeK: return go.rope_freqs;
        // The shared table only has entries for qwen3.5's kinds, so the norms
        // and the residual adds map onto those.
        case Kind::AttnNorm:      return base[Kernel::Rms];
        case Kind::FfnNorm:       return base[Kernel::Rms];
        case Kind::FinalRms:      return base[Kernel::FinalRms];
        case Kind::EmbedGather:   return base[Kernel::EmbedGather];
        case Kind::KvAppend:      return base[Kernel::KvAppend];
        case Kind::AttnResidual:  return base[Kernel::Residual];
        case Kind::FfnResidual:   return base[Kernel::Residual];
        case Kind::Argmax:        return base[Kernel::Argmax];
    }
    return Pso{};
}

void launch_shape(const Dispatch& d, const GptOssGeometry& g, Grid& grid, Threadgroup& tg) {
    if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
        const bool routed = d.kind == Kind::ExpertGate || d.kind == Kind::ExpertUp ||
                            d.kind == Kind::ExpertDown;
        qmv_dispatch(kn.N, 1, grid, tg,
                     routed ? gptoss_moe_sorted_rows(g, 1) : 1);
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
        case Kind::AttnNorm: case Kind::FfnNorm: case Kind::FinalRms: {
            const int threads = (g.hidden + 3) / 4;
            grid = Grid{std::uint32_t(threads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::RopeQ:
            grid = Grid{std::uint32_t(g.head_dim / 2), std::uint32_t(g.n_q_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim / 2), 1, 1};
            return;
        case Kind::RopeK:
            grid = Grid{std::uint32_t(g.head_dim / 2), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim / 2), 1, 1};
            return;
        case Kind::KvAppend:
            grid = Grid{std::uint32_t(g.head_dim), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim), 1, 1};
            return;
        case Kind::SdpaSink:
            sdpa_sink_dispatch(g.n_q_heads, grid, tg);
            return;
        case Kind::RouterTopK:
            router_topk_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertSort:
            shared_kernels::moe_route_sort_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertGather:
            shared_kernels::moe_route_rows_dispatch(
                g.hidden, gptoss_moe_sorted_rows(g, 1), grid, tg);
            return;
        case Kind::ExpertSwiGlu:
            elementwise_dispatch(
                gptoss_moe_sorted_rows(g, 1) * g.intermediate, grid, tg);
            return;
        case Kind::ExpertCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, grid, tg);
            return;
        case Kind::AttnResidual:
        case Kind::FfnResidual:
            elementwise_dispatch(g.hidden, grid, tg);
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

Pso pso_for_paged(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb,
                  const GptOssPsos& go) {
    switch (d.kind) {
        case Kind::KvAppend:  return mb.kv_append_paged;
        case Kind::SdpaSink:  return go.sdpa_sink_paged;
        default:              return pso_for(d, base, go);
    }
}

/// The pipeline a dispatch runs on at M>1.
///
/// Differs from `pso_for_paged` for exactly the two kinds whose ROW indexing
/// differs, not merely their launch width. Everything else in this family is
/// row-strided already: the matvecs read `tid.x` as the token, the norms and
/// the elementwise kernels are flat over `rows * width`, and the paged
/// attention takes the row on its second grid axis. What that leaves is the
/// embedding gather (a different weight layout at M>1) and the YaRN rope (which
/// must read `position[row]` and stride by a width its grid does not carry).
/// How many rows a DENSE projection's GEMM runs over.
///
/// Padded up to a whole BM tile so the GEMM engages at any batch rather than
/// only at exact multiples of one; the padding computes discardable values into
/// pool rows the fire does not use.

int gptoss_qmm_rows(int rows) {
    const int n = rows < 1 ? 1 : rows;
    // Inherited from qwen3.5 and measured here: lowering it to 4 costs gpt-oss
    // 1% (8 lanes, 55.5 -> 54.9 tok/s), so the inherited number holds.
    if (n < kQmmMinBatch) return n;
    const int bm = qmm_bm(n);
    return ((n + bm - 1) / bm) * bm;
}

int gptoss_qmm_pool_rows(int max_rows) {
    const int n = max_rows < 1 ? 1 : max_rows;
    return ((n + kQmmBMWide - 1) / kQmmBMWide) * kQmmBMWide;
}

int gptoss_moe_pairs(const GptOssGeometry& g, int rows) {
    return (rows < 1 ? 1 : rows) * g.experts_per_token;
}

int gptoss_moe_tile_rows(const GptOssGeometry& g, int rows) {
    return shared_kernels::moe_tile_rows(
        gptoss_moe_pairs(g, rows), g.n_experts);
}

int gptoss_moe_sorted_rows(const GptOssGeometry& g, int rows) {
    return shared_kernels::moe_sorted_rows(
        gptoss_moe_pairs(g, rows), g.n_experts);
}

int gptoss_moe_qmm_bn(Kind k, const GptOssGeometry& g, int rows) {
    const bool routed = k == Kind::ExpertGate || k == Kind::ExpertUp ||
                        k == Kind::ExpertDown;
    if (!routed || !g.mxfp4_experts || gptoss_moe_tile_rows(g, rows) <= 1)
        return 0;
    const KN kn = qmv_kn(k, g);
    int bn = 0;
    for (const int candidate : {16, 32, 64})
        if (kn.N % candidate == 0) bn = candidate;
    return bn;
}

/// The dense projections, which are the ones a batched GEMM can serve.
///
/// NOT the routed three: each row picks its own experts, so their weight is
/// chosen on the GPU and a tile spanning rows would span weights. That is the
/// MoE's whole difficulty and it is not a launch shape.
bool gptoss_is_dense_proj(Kind k) {
    return k == Kind::QmvQ || k == Kind::QmvK || k == Kind::QmvV || k == Kind::QmvO ||
           k == Kind::LmHead;
}

/// The GEMM's tile for a dense projection, or 0 to keep the matvec.
int gptoss_qmm_bn(Kind k, const GptOssGeometry& g, int rows) {
    if (!gptoss_is_dense_proj(k)) return 0;
    const KN kn = qmv_kn(k, g);
    if (kn.N == 0) return 0;
    return qmm_bn(kn.N, gptoss_qmm_rows(rows));
}

Pso pso_for_mb(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb,
               const GptOssPsos& go) {
    switch (d.kind) {
        case Kind::EmbedGather: return mb.embed_mb;
        case Kind::RopeQ:
        case Kind::RopeK:       return go.rope_freqs_mb;
        case Kind::RowGather:   return go.row_gather;
        default:                return pso_for_paged(d, base, mb, go);
    }
}

/// The pipeline for a dispatch whose tiling depends on the batch. Split from
/// `pso_for_mb` because the tile choice must mirror `launch_shape_mb`'s
/// EXACTLY: a grid computed for one tiling against a pipeline compiled for
/// another is not a crash, it is wrong numbers.
Pso pso_for_mb_rows(const Dispatch& d, const GptOssGeometry& g, int rows,
                    const DecodeStepPsos& base, const MultiBatchPsos& mb,
                    const GptOssPsos& go, int head_rows) {
    const int N = rows < 1 ? 1 : rows;
    const int S = head_rows < 1 ? N : (head_rows < N ? head_rows : N);
    const int m = d.kind == Kind::LmHead ? S : N;
    if (const int bn = gptoss_moe_qmm_bn(d.kind, g, N); bn > 0) {
        const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
        if (go.qmm_routed_bias[slot].valid()) return go.qmm_routed_bias[slot];
    }
    if (const int bn = gptoss_qmm_bn(d.kind, g, m); bn > 0) {
        const int wide = qmm_bm_slot(qmm_bm(m));
        const int slot = bn == 64 ? 2 : (bn == 32 ? 1 : 0);
        // The LM head has no bias; every other projection here does.
        const auto& table = d.kind == Kind::LmHead ? mb.qmm_t : mb.qmm_t_bias;
        if (table[wide][slot].valid()) return table[wide][slot];
    }
    return pso_for_mb(d, base, mb, go);
}

void launch_shape_mb(const Dispatch& d, const GptOssGeometry& g, int rows, Grid& grid,
                     Threadgroup& tg, int head_rows) {
    const int N = rows < 1 ? 1 : rows;
    // The tail runs on the rows the fire will SAMPLE, which `RowGather`
    // compacted. The LM head is `hidden * vocab` per row, so on a prefill it is
    // most of the fire's cost and all of its logits memory.
    const int S = head_rows < 1 ? N : (head_rows < N ? head_rows : N);

    if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
        const bool routed = d.kind == Kind::ExpertGate || d.kind == Kind::ExpertUp ||
                            d.kind == Kind::ExpertDown;
        const int m = d.kind == Kind::LmHead ? S : N;
        if (const int bn = gptoss_moe_qmm_bn(d.kind, g, N); bn > 0) {
            qmm_t_dispatch(kn.N, gptoss_moe_sorted_rows(g, N), bn,
                           shared_kernels::kMoeTileRows, grid, tg);
            return;
        }
        // A dense projection becomes a matmul once the batch fills a tile: the
        // matvec re-reads the whole weight PER ROW, which on this checkpoint is
        // 318 MB a layer.
        if (const int bn = gptoss_qmm_bn(d.kind, g, m); bn > 0) {
            // The row block is asked of the BATCH, not of the padded count:
            // padding rounds up, and a rounded-up count can land on a wider
            // rung than the one `pso_for` picked.
            qmm_t_dispatch(kn.N, gptoss_qmm_rows(m), bn, qmm_bm(m), grid, tg);
            return;
        }
        qmv_dispatch(kn.N, 1, grid, tg,
                     routed ? gptoss_moe_sorted_rows(g, N) : m);
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            embed_mb_dispatch(g.hidden, N, grid, tg);
            return;
        case Kind::AttnNorm: case Kind::FfnNorm:
            rms_mb_dispatch(g.hidden, 1, N, grid, tg);
            return;
        case Kind::RowGather:
            grid = Grid{std::uint32_t(g.hidden), std::uint32_t(S), 1};
            tg = Threadgroup{64, 1, 1};
            return;
        case Kind::FinalRms:
            rms_mb_dispatch(g.hidden, 1, S, grid, tg);
            return;
        case Kind::RopeQ:
            grid = Grid{std::uint32_t(g.head_dim / 2), std::uint32_t(g.n_q_heads),
                        std::uint32_t(N)};
            tg = Threadgroup{std::uint32_t(g.head_dim / 2), 1, 1};
            return;
        case Kind::RopeK:
            grid = Grid{std::uint32_t(g.head_dim / 2), std::uint32_t(g.n_kv_heads),
                        std::uint32_t(N)};
            tg = Threadgroup{std::uint32_t(g.head_dim / 2), 1, 1};
            return;
        case Kind::KvAppend:
            kv_append_mb_dispatch(g.head_dim, g.n_kv_heads, N, grid, tg);
            return;
        case Kind::SdpaSink:
            sdpa_sink_dispatch(g.n_q_heads, grid, tg, N);
            return;
        case Kind::RouterTopK:
            router_topk_dispatch(g.n_experts, grid, tg, N);
            return;
        case Kind::ExpertSort:
            shared_kernels::moe_route_sort_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertGather:
            shared_kernels::moe_route_rows_dispatch(
                g.hidden, gptoss_moe_sorted_rows(g, N), grid, tg);
            return;
        case Kind::ExpertSwiGlu:
            elementwise_dispatch(
                gptoss_moe_sorted_rows(g, N) * g.intermediate, grid, tg);
            return;
        case Kind::ExpertCombine:
            shared_kernels::expert_combine_dispatch(g.hidden, grid, tg, N);
            return;
        case Kind::AttnResidual:
        case Kind::FfnResidual:
            elementwise_mb_dispatch(g.hidden, N, grid, tg);
            return;
        default:
            launch_shape(d, g, grid, tg);
            return;
    }
}

void encode_gptoss_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const GptOssGeometry& g, int rows, const DecodeStepPsos& base,
                           const MultiBatchPsos& mb, const GptOssPsos& go, int ordinal_base,
                           int head_rows, std::size_t begin, std::size_t end) {
    // Deliberately the same walk as `encode_gptoss_step`: the DAG, its order
    // and its concurrency runs belong to the MODEL, not to the batch size.
    // Off the WHOLE dag, not the slice: a run's extent is the model's, and
    // recomputing it per segment would let a boundary invent a barrier.
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    if (end == 0 || end > dag.size()) end = dag.size();
    for (std::size_t i = begin; i < end; ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape_mb(d, g, rows, grid, tg, head_rows);
        se.set_pso(pso_for_mb_rows(d, g, rows, base, mb, go, head_rows));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        if (i + 1 >= end || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

void encode_gptoss_step_paged(StepEncoder& se, const std::vector<Dispatch>& dag,
                              const GptOssGeometry& g, const DecodeStepPsos& base,
                              const MultiBatchPsos& mb, const GptOssPsos& go,
                              int ordinal_base) {
    // The same walk and the same shapes as the contiguous encoder: only the KV
    // layout changed, and a layout is not a launch geometry. Both attention
    // kernels are one threadgroup per query head either way.
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for_paged(d, base, mb, go));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

void encode_gptoss_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const GptOssGeometry& g, const DecodeStepPsos& base,
                        const GptOssPsos& go, int ordinal_base) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for(d, base, go));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::gptoss
