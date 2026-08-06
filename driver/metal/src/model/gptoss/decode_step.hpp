#pragma once

/// GPT-OSS's per-token dispatch DAG.
///
/// A pure function, exactly like qwen3.5's `build_decode_dag` and gemma4's: it
/// emits the ordered list of dispatches for one token and touches no Metal.
/// That is what lets the shape of the step — dispatch count, the sliding/full
/// split, how many dispatches an MoE layer actually costs — be checked with no
/// GPU and no checkpoint.
///
/// The kinds stay in this namespace rather than joining the shared
/// `pie::metal::Kernel`: that enum is the M=1 argument-table ABI, and a family
/// that cannot yet be bound has no business appending to it. gemma4 made the
/// same choice for the same reason.

#include <cstdint>
#include <vector>

#include "geometry.hpp"

namespace pie::metal::gptoss {

enum class Kind : std::uint8_t {
    EmbedGather,

    // Attention. Every projection is biased, and the softmax has a per-head
    // sink, but the shape is otherwise the plainest of the three families:
    // no QK-norm, no V-norm, no partial rotary.
    AttnNorm,
    QmvQ, QmvK, QmvV,
    RopeQ, RopeK,
    KvAppend,
    /// Scaled dot-product attention with per-head sinks. The sink is a learned
    /// scalar that joins the softmax denominator as though it were one more key,
    /// so a head can attend to nothing; it is bound, not folded.
    SdpaSink,
    QmvO,
    AttnResidual,

    // The sparse MoE. This is the family's whole difficulty: which experts run
    // is decided on the GPU, so the routed matmuls' operands are gathered.
    FfnNorm,
    /// `router(x) + bias` -> one logit per expert.
    RouterGemv,
    /// top-k over those logits, then softmax over the k that survive. Emits both
    /// the chosen expert ids and their normalized weights.
    RouterTopK,
    /// Expert-major reordering. At decode tile_rows=1 and the routed matvecs
    /// consume the four sorted rows; at a wide fire each expert run pads to a
    /// GEMM tile and amortizes one weight read across its rows.
    ExpertSort,
    ExpertGather,
    /// The three routed projections. Each reads `experts_per_token` expert
    /// slices selected by `RouterTopK`, not a fixed weight.
    ExpertGate, ExpertUp,
    /// gpt-oss's SwiGLU: clamp both operands to +-limit (the gate from above
    /// only), `gate * sigmoid(alpha*gate) * (up + 1)`. The `+1` and the clamp
    /// are not decoration -- they are why this cannot reuse `silu_mul`.
    ExpertSwiGlu,
    ExpertDown,
    /// Weighted sum of the k experts' outputs, by the router's softmax.
    ExpertCombine,
    FfnResidual,

    // Tail.
    /// The rows this fire will sample, compacted. A prefill computes every row
    /// of the body and reads one per request, and the LM head is `hidden *
    /// vocab` per row -- so on a prefill this is most of the fire's cost.
    RowGather,
    FinalRms,
    LmHead,
    Argmax,
};

/// Which kinds run on the rows the sampler will READ rather than on every token.
///
/// `RowGather` compacts the sampled positions to a dense prefix, and everything
/// after it works on that prefix. Two places ask this -- the scratch sizer, so
/// a colour holding one of these is allocated for the shorter count, and the
/// golden-tap dumper, so a published tensor has the rows it really has -- and
/// they have to give the same answer. When they disagree the sizer wins and the
/// dumper reads past the value into whatever the pool put after it.
constexpr bool is_tail(Kind k) {
    return k == Kind::RowGather || k == Kind::FinalRms || k == Kind::LmHead;
}

struct Dispatch {
    Kind kind;
    int layer;     // -1 for the tail
    int ordinal;   // flat argument-table key
    bool sliding;  // this layer's attention type: picks the sdpa window
};

/// Emit the ordered per-token DAG for `g`.
///
/// The order within a stage is clustered so independent dispatches end up
/// adjacent (q/k/v together, then the ropes, then gate/up). That is
/// hazard-neutral — a barrier still lands at every true RAW edge — and only
/// lets a concurrency group form.
inline std::vector<Dispatch> build_gptoss_dag(const GptOssGeometry& g, bool with_argmax = true) {
    std::vector<Dispatch> dag;
    dag.reserve(std::size_t(g.n_layers) * 20 + 8);
    int ord = 0;
    auto emit = [&](Kind k, int layer, bool sliding) {
        dag.push_back({k, layer, ord++, sliding});
    };

    emit(Kind::EmbedGather, -1, false);

    for (int L = 0; L < g.n_layers; ++L) {
        const bool sliding = g.is_sliding(L);

        emit(Kind::AttnNorm, L, sliding);
        emit(Kind::QmvQ, L, sliding);
        emit(Kind::QmvK, L, sliding);
        emit(Kind::QmvV, L, sliding);
        emit(Kind::RopeQ, L, sliding);
        emit(Kind::RopeK, L, sliding);
        emit(Kind::KvAppend, L, sliding);
        emit(Kind::SdpaSink, L, sliding);
        emit(Kind::QmvO, L, sliding);
        emit(Kind::AttnResidual, L, sliding);

        emit(Kind::FfnNorm, L, sliding);
        emit(Kind::RouterGemv, L, sliding);
        emit(Kind::RouterTopK, L, sliding);
        emit(Kind::ExpertSort, L, sliding);
        emit(Kind::ExpertGather, L, sliding);
        emit(Kind::ExpertGate, L, sliding);
        emit(Kind::ExpertUp, L, sliding);
        emit(Kind::ExpertSwiGlu, L, sliding);
        emit(Kind::ExpertDown, L, sliding);
        emit(Kind::ExpertCombine, L, sliding);
        emit(Kind::FfnResidual, L, sliding);
    }

    emit(Kind::RowGather, -1, false);
    emit(Kind::FinalRms, -1, false);
    emit(Kind::LmHead, -1, false);
    if (with_argmax) emit(Kind::Argmax, -1, false);
    return dag;
}

inline bool is_expert_sorted(Kind k) {
    return k == Kind::ExpertGather || k == Kind::ExpertGate ||
           k == Kind::ExpertUp || k == Kind::ExpertSwiGlu ||
           k == Kind::ExpertDown;
}

struct DagStats {
    int total = 0;
    int full_attn_layers = 0;
    int sliding_attn_layers = 0;
    int gemv = 0;    // the matvecs, which are the step's bandwidth
    int routed = 0;  // of those, the ones whose weights the router picks
};

inline DagStats dag_stats(const std::vector<Dispatch>& dag, const GptOssGeometry& g) {
    DagStats s;
    s.total = static_cast<int>(dag.size());
    for (int L = 0; L < g.n_layers; ++L) {
        (g.is_full_attn(L) ? s.full_attn_layers : s.sliding_attn_layers) += 1;
    }
    for (const Dispatch& d : dag) {
        switch (d.kind) {
            case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
            case Kind::RouterGemv: case Kind::LmHead:
                ++s.gemv;
                break;
            case Kind::ExpertGate: case Kind::ExpertUp: case Kind::ExpertDown:
                ++s.gemv;
                ++s.routed;
                break;
            default:
                break;
        }
    }
    return s;
}

/// The same DAG, with argument-table ordinals shifted clear of the decode
/// path's. The dispatch LIST does not depend on the batch size -- only the
/// shapes and pipelines do -- so this offsets rather than rebuilds, and the two
/// paths cannot come to describe different models.
inline std::vector<Dispatch> build_gptoss_dag_mb(const GptOssGeometry& g, int ordinal_base,
                                                 bool with_argmax = true) {
    std::vector<Dispatch> dag = build_gptoss_dag(g, with_argmax);
    for (Dispatch& d : dag) d.ordinal += ordinal_base;
    return dag;
}

}  // namespace pie::metal::gptoss
