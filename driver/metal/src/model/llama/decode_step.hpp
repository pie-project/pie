#pragma once

/// The llama families' per-token dispatch DAG.
///
/// A pure function, exactly like gemma4's `build_gemma4_dag`: it emits the
/// ordered list of dispatches for one token and touches no Metal. That is what
/// lets the shape of the step -- dispatch count, the qk-norm pair, the dense/
/// routed split -- be checked with no GPU, the way gemma4's was.
///
/// The kinds stay in this namespace rather than joining the shared
/// `pie::metal::Kernel`: that enum is the M=1 argument-table ABI, and `encode`
/// maps onto it. Almost every kind here borrows an existing `Kernel` -- this
/// family's whole point is that the driver already had the pieces.

#include <cstdint>
#include <vector>

#include "geometry.hpp"

namespace pie::metal::llama {

enum class Kind : std::uint8_t {
    /// Token embedding. Tied checkpoints read `shared_embedding`, untied ones
    /// their own `embed_tokens`; the geometry says which and `encode` picks the
    /// weight map, but the dispatch is the same either way.
    EmbedGather,

    // ── Attention ──
    AttnNorm,  // input_layernorm
    QmvQ, QmvK, QmvV,
    /// Qwen3 only. Per-head RMS over `head_dim`, before the rotation.
    QNorm, KNorm,
    RopeQ, RopeK,
    KvAppend,
    Sdpa,
    QmvO,
    /// hidden += attn_out.
    AttnResidual,

    // ── FFN, one of two shapes ──
    FfnNorm,  // post_attention_layernorm

    // Dense SwiGLU.
    QmvGate, QmvUp, SiluMul, QmvDown,

    // Routed mixture. `Router` is an ordinary matvec: `mlp.gate` is
    // `[n_experts, hidden]`, which is the same shape as a narrow attention
    // projection, so it goes through the same kernel rather than a new one.
    Router,
    RouterTopK,
    /// Group the (token, slot) pairs by expert.
    ///
    /// The three routed projections read a DIFFERENT weight matrix per pair,
    /// which is the whole reason a mixture's prefill does not improve with
    /// length the way a dense one does. Sorting the pairs by expert makes each
    /// expert's rows contiguous, and a contiguous run against one weight slice
    /// is the matmul this driver already has.
    ///
    /// It runs at M=1 too, where it is a grouping with no padding and the
    /// projections stay matvecs. That is deliberate: the routed dataflow has
    /// one shape rather than a decode shape and a prefill shape that would have
    /// to be kept agreeing.
    ExpertSort,
    /// The FFN norm's rows, in the sorted order.
    ExpertGather,
    /// The three routed projections. Each reads the expert stack at
    /// `expert_id * N * K` and serves one sorted row.
    ExpertGate, ExpertUp,
    /// SwiGLU over the sorted `[sorted_rows, moe_intermediate]` stack.
    ExpertSiluMul,
    ExpertDown,
    /// Sum the k experts' outputs, weighted by the router's softmax, reading
    /// them through the sort's inverse permutation rather than after a kernel
    /// has put them back.
    ExpertCombine,

    /// hidden += ffn_out.
    FfnResidual,

    // ── Tail ──
    /// The rows this fire will sample, compacted. A prefill computes every row
    /// of the body and reads one per request, so everything after this runs on
    /// `S` rows rather than `N` -- and the LM head is what that saves.
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
    int layer;    // -1 for the layer-less embedding and the tail
    int ordinal;  // flat argument-table key
};

/// Emit the ordered per-token DAG for `g`.
///
/// The order within a stage is clustered so independent dispatches end up
/// adjacent (q/k/v together, then the norms, then the ropes). That is
/// hazard-neutral -- a barrier still lands at every true RAW edge -- and only
/// lets a concurrency group form.
inline std::vector<Dispatch> build_llama_dag(const LlamaGeometry& g, bool with_argmax = true) {
    std::vector<Dispatch> dag;
    dag.reserve(std::size_t(g.n_layers) * 24 + 8);
    int ord = 0;
    const auto emit = [&](Kind k, int layer) { dag.push_back({k, layer, ord++}); };

    emit(Kind::EmbedGather, -1);

    for (int L = 0; L < g.n_layers; ++L) {
        emit(Kind::AttnNorm, L);
        emit(Kind::QmvQ, L);
        emit(Kind::QmvK, L);
        emit(Kind::QmvV, L);
        if (g.qk_norm) {
            emit(Kind::QNorm, L);
            emit(Kind::KNorm, L);
        }
        emit(Kind::RopeQ, L);
        emit(Kind::RopeK, L);
        emit(Kind::KvAppend, L);
        emit(Kind::Sdpa, L);
        emit(Kind::QmvO, L);
        emit(Kind::AttnResidual, L);

        emit(Kind::FfnNorm, L);
        if (g.is_moe()) {
            emit(Kind::Router, L);
            emit(Kind::RouterTopK, L);
            emit(Kind::ExpertSort, L);
            emit(Kind::ExpertGather, L);
            emit(Kind::ExpertGate, L);
            emit(Kind::ExpertUp, L);
            emit(Kind::ExpertSiluMul, L);
            emit(Kind::ExpertDown, L);
            emit(Kind::ExpertCombine, L);
        } else {
            emit(Kind::QmvGate, L);
            emit(Kind::QmvUp, L);
            emit(Kind::SiluMul, L);
            emit(Kind::QmvDown, L);
        }
        emit(Kind::FfnResidual, L);
    }

    emit(Kind::RowGather, -1);
    emit(Kind::FinalRms, -1);
    emit(Kind::LmHead, -1);
    if (with_argmax) emit(Kind::Argmax, -1);
    return dag;
}

/// Whether a kind reads a quantized weight stack indexed by the router.
inline bool is_routed(Kind k) {
    return k == Kind::ExpertGate || k == Kind::ExpertUp || k == Kind::ExpertDown;
}

/// Whether a kind's output has one row per SORTED (token, slot) pair.
///
/// Between the gather and the scatter every activation is in expert order and
/// padded to whole tiles, so it has more rows than `rows * experts_per_token`.
/// Three places ask -- the scratch sizer, the launch shapes and the golden-tap
/// dumper -- and a disagreement is an overrun, so they ask here.
inline bool is_expert_sorted(Kind k) {
    return k == Kind::ExpertGather || k == Kind::ExpertGate || k == Kind::ExpertUp ||
           k == Kind::ExpertSiluMul || k == Kind::ExpertDown;
}

/// Whether a kind is a matvec -- the step's bandwidth, and what any performance
/// claim about this family is really about.
inline bool is_gemv(Kind k) {
    switch (k) {
        case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
        case Kind::QmvGate: case Kind::QmvUp: case Kind::QmvDown:
        case Kind::Router: case Kind::ExpertGate: case Kind::ExpertUp:
        case Kind::ExpertDown: case Kind::LmHead:
            return true;
        default:
            return false;
    }
}

struct DagStats {
    int total = 0;
    int layers = 0;
    int gemv = 0;
    int routed = 0;
};

inline DagStats dag_stats(const std::vector<Dispatch>& dag, const LlamaGeometry& g) {
    DagStats s;
    s.total = static_cast<int>(dag.size());
    s.layers = g.n_layers;
    for (const Dispatch& d : dag) {
        if (is_gemv(d.kind)) ++s.gemv;
        if (is_routed(d.kind)) ++s.routed;
    }
    return s;
}

/// The same DAG, with argument-table ordinals shifted clear of the decode
/// path's. The dispatch LIST does not depend on the batch size -- only the
/// shapes and pipelines do -- so this offsets rather than rebuilds, and the two
/// paths cannot describe different models.
inline std::vector<Dispatch> build_llama_dag_mb(const LlamaGeometry& g, int ordinal_base,
                                                bool with_argmax = true) {
    std::vector<Dispatch> dag = build_llama_dag(g, with_argmax);
    for (Dispatch& d : dag) d.ordinal += ordinal_base;
    return dag;
}

}  // namespace pie::metal::llama
