#pragma once

/// Gemma 4's per-token dispatch DAG.
///
/// A pure function, exactly like qwen3.5's `build_decode_dag`: it emits the
/// ordered list of dispatches for one token and touches no Metal. That is what
/// lets the shape of the step — dispatch count, the KV-shared skips, the
/// sliding/full split — be checked with no GPU, the way qwen3.5's "363
/// dispatches" was.
///
/// Ported from the bring-up harness at `tools/rawmetal/gemma4_decode_step.hpp`.
/// The kinds stay in this namespace rather than joining the shared
/// `pie::metal::Kernel`: that enum is the M=1 argument-table ABI, and a family
/// that cannot yet be bound has no business appending to it.

#include <cstdint>
#include <vector>

#include "geometry.hpp"

namespace pie::metal::gemma4 {

enum class Kind : std::uint8_t {
    // PLE precompute, once per step, layer-less.
    EmbedGather,     // embed_tokens gather, scaled by sqrt(hidden)
    PleTokenGather,  // embed_tokens_per_layer gather, scaled by sqrt(ple_dim)
    PleProjGemv,     // per_layer_model_projection, scaled by 1/sqrt(hidden)
    PleProjNorm,     // rms over each ple_dim row
    PleCombine,      // (proj + token) * 1/sqrt(2)

    // Attention, with Gemma's norm sandwich.
    AttnNorm,
    QmvQ, QmvK, QmvV,
    QNorm, KNorm,
    VNorm,  // weightless, before the KV write
    RopeQ, RopeK,
    KvAppend,
    Sdpa,
    QmvO,
    /// `rms(block)*w + resid`, fused. Gemma's norm sandwich puts a norm
    /// immediately before every residual add, and the threadgroup that computes
    /// the row's inverse RMS already holds the whole row -- so the add is one
    /// extra load and no synchronisation, against ~5.8 us for the barrier the
    /// second dispatch would have cost.
    PostAttnResidual,

    // FFN, GeGLU-tanh, with the other half of the sandwich.
    FfnNorm,
    QmvGate, QmvUp,
    GegluTanh,
    QmvDown,
    PostFfnResidual,

    // Per-layer embedding residual and the learned layer scalar.
    PleGateGemv,
    PleGeglu,
    PleProjLayerGemv,
    /// The same fusion, plus the learned per-layer gain that followed it:
    /// `(rms(ple)*w + resid) * layer_scalar`.
    PleResidualScaled,
    /// The gain on its own, for a config with no per-layer embeddings -- there
    /// is then no PLE residual for it to ride on.
    LayerScalar,

    // Tail.
    /// The rows this fire will sample, compacted. A prefill computes every row
    /// of the body and reads one per request, so everything after this runs on
    /// `S` rows rather than `N` -- and the LM head is what that saves.
    RowGather,
    FinalRms,
    LmHead,
    FinalSoftcap,
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
    return k == Kind::RowGather || k == Kind::FinalRms || k == Kind::LmHead ||
           k == Kind::FinalSoftcap;
}

struct Dispatch {
    Kind kind;
    int layer;    // -1 for the layer-less PLE precompute and the tail
    int ordinal;  // flat argument-table key
    bool sliding; // this layer's attention type: picks sdpa variant and rope theta
};

/// Emit the ordered per-token DAG for `g`.
///
/// The order within a stage is clustered so independent dispatches end up
/// adjacent (q/k/v together, then the norms, then the ropes). That is
/// hazard-neutral — a barrier still lands at every true RAW edge — and only
/// lets a concurrency group form, which is worth ~3.7 us per barrier avoided.
inline std::vector<Dispatch> build_gemma4_dag(const Gemma4Geometry& g, bool with_argmax = true) {
    std::vector<Dispatch> dag;
    dag.reserve(std::size_t(g.n_layers) * 32 + 16);
    int ord = 0;
    auto emit = [&](Kind k, int layer, bool sliding) {
        dag.push_back({k, layer, ord++, sliding});
    };

    emit(Kind::EmbedGather, -1, false);
    if (g.per_layer_emb_dim > 0) {
        emit(Kind::PleTokenGather, -1, false);
        emit(Kind::PleProjGemv, -1, false);
        emit(Kind::PleProjNorm, -1, false);
        emit(Kind::PleCombine, -1, false);
    }

    for (int L = 0; L < g.n_layers; ++L) {
        const bool sliding = g.is_sliding(L);
        // A KV-shared layer rotates its own Q and reads the pages its source
        // layer wrote: no k/v projection, no k-norm, no v-norm, no append.
        const bool owns_kv = !g.is_kv_shared(L);

        emit(Kind::AttnNorm, L, sliding);
        emit(Kind::QmvQ, L, sliding);
        if (owns_kv) {
            emit(Kind::QmvK, L, sliding);
            emit(Kind::QmvV, L, sliding);
        }
        emit(Kind::QNorm, L, sliding);
        if (owns_kv) {
            emit(Kind::KNorm, L, sliding);
            emit(Kind::VNorm, L, sliding);
        }
        emit(Kind::RopeQ, L, sliding);
        if (owns_kv) {
            emit(Kind::RopeK, L, sliding);
            emit(Kind::KvAppend, L, sliding);
        }
        emit(Kind::Sdpa, L, sliding);
        emit(Kind::QmvO, L, sliding);
        emit(Kind::PostAttnResidual, L, sliding);

        emit(Kind::FfnNorm, L, sliding);
        emit(Kind::QmvGate, L, sliding);
        emit(Kind::QmvUp, L, sliding);
        emit(Kind::GegluTanh, L, sliding);
        emit(Kind::QmvDown, L, sliding);
        emit(Kind::PostFfnResidual, L, sliding);

        if (g.per_layer_emb_dim > 0) {
            emit(Kind::PleGateGemv, L, sliding);
            emit(Kind::PleGeglu, L, sliding);
            emit(Kind::PleProjLayerGemv, L, sliding);
            emit(Kind::PleResidualScaled, L, sliding);
        } else {
            emit(Kind::LayerScalar, L, sliding);
        }
    }

    emit(Kind::RowGather, -1, false);
    emit(Kind::FinalRms, -1, false);
    emit(Kind::LmHead, -1, false);
    if (g.final_softcap > 0.0f) emit(Kind::FinalSoftcap, -1, false);
    if (with_argmax) emit(Kind::Argmax, -1, false);
    return dag;
}

struct DagStats {
    int total = 0;
    int kv_owning_layers = 0;
    int kv_shared_layers = 0;
    int full_attn_layers = 0;
    int sliding_attn_layers = 0;
    int gemv = 0;  // the matvecs, which are the step's bandwidth
};

inline DagStats dag_stats(const std::vector<Dispatch>& dag, const Gemma4Geometry& g) {
    DagStats s;
    s.total = static_cast<int>(dag.size());
    for (int L = 0; L < g.n_layers; ++L) {
        (g.is_kv_shared(L) ? s.kv_shared_layers : s.kv_owning_layers) += 1;
        (g.is_full_attn(L) ? s.full_attn_layers : s.sliding_attn_layers) += 1;
    }
    for (const Dispatch& d : dag) {
        switch (d.kind) {
            case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
            case Kind::QmvGate: case Kind::QmvUp: case Kind::QmvDown:
            case Kind::PleProjGemv: case Kind::PleGateGemv:
            case Kind::PleProjLayerGemv: case Kind::LmHead:
                ++s.gemv;
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
/// paths cannot describe different models.
inline std::vector<Dispatch> build_gemma4_dag_mb(const Gemma4Geometry& g, int ordinal_base,
                                                 bool with_argmax = true) {
    std::vector<Dispatch> dag = build_gemma4_dag(g, with_argmax);
    for (Dispatch& d : dag) d.ordinal += ordinal_base;
    return dag;
}

}  // namespace pie::metal::gemma4
