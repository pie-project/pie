#pragma once
// ── gemma4 decode-step DAG skeleton (alpha) ──────────────────────────────────
// The gemma4 analog of beta's qwen3.6 decode_step. `build_gemma4_dag` is a PURE
// function (no Metal) that emits the exact per-token dispatch order as a list of
// {kind, layer, ordinal}. The flat ordinal is the arg-table key (mtl4_context's
// ratified scheme); `kind`+`layer` are retained only for charlie's
// `<layer>.<tag>.npy` golden-dump naming.
//
// Walking it with a real StepEncoder (PSOs + grid/tg per kind) is the encode_fn
// for ctx->run_step(...) — layered on once the gemma4 kernels are ported + gated
// against charlie's golden. The pure list lets us verify the DAG shape (dispatch
// count, kv-share skips, sliding/full split) with zero GPU, exactly like beta's
// "322 dispatches verified".

#include <vector>

#include "gemma4_abi.hpp"

namespace pie_metal_driver::raw_metal::gemma4 {

struct Gemma4Dispatch {
    Kernel kind;
    int    layer;     // -1 for layer-less (PLE precompute / tail)
    int    ordinal;   // flat 0..M-1 — the arg-table key
    bool   sliding;   // attention type for this dispatch's layer (sdpa/rope theta)
};

// Build the ordered per-token dispatch DAG for the given geometry.
inline std::vector<Gemma4Dispatch> build_gemma4_dag(const Gemma4Geometry& g) {
    std::vector<Gemma4Dispatch> d;
    d.reserve(900);
    int ord = 0;
    auto emit = [&](Kernel k, int layer, bool sliding) {
        d.push_back({k, layer, ord++, sliding});
    };

    // ── PLE precompute (layer-less, once) ──
    emit(Kernel::EmbedGather,    -1, false);  // embed_tokens gather * sqrt(hidden)
    emit(Kernel::PleTokenGather, -1, false);  // embed_tokens_per_layer gather * sqrt(ple_dim)
    emit(Kernel::PleProjGemv,    -1, false);  // per_layer_model_projection GEMV * 1/sqrt(hidden)
    emit(Kernel::PleProjNorm,    -1, false);  // rms over each ple_dim row
    emit(Kernel::PleCombine,     -1, false);  // (proj + token) * 1/sqrt(2)

    // ── decoder layers ──
    for (int L = 0; L < g.n_layers; ++L) {
        const bool sliding = Gemma4Geometry::is_sliding(L);
        const bool shared  = g.is_kv_shared(L);

        // attention (norm sandwich; qk-norm; sliding/full; kv-share). Dataflow matches the
        // gemma4 reference; the emit ORDER is clustered-by-stage to expose branch concurrency
        // to the greedy RAW predicate: the 3 input GEMVs adjacent (QmvK/QmvV overlap QmvQ's
        // compute — all read AttnNorm-out), then the 3 q/k-norms, then the ropes. Reordering is
        // hazard-neutral (the predicate inserts a barrier exactly at each true RAW boundary);
        // it only lets independent dispatches overlap. Shared layers rotate Q only + reuse src KV.
        emit(Kernel::AttnNorm, L, sliding);
        emit(Kernel::QmvQ,     L, sliding);
        if (!shared) {
            emit(Kernel::QmvK, L, sliding);   // ‖ QmvQ (both read AttnNorm-out)
            emit(Kernel::QmvV, L, sliding);   // ‖ QmvQ
        }
        emit(Kernel::QNorm, L, sliding);
        if (!shared) {
            emit(Kernel::KNorm, L, sliding);  // ‖ QNorm
            emit(Kernel::VNorm, L, sliding);  // ‖ QNorm (weightless V-norm before the KV write)
        }
        emit(Kernel::RopeQ, L, sliding);
        if (!shared) {
            emit(Kernel::RopeK, L, sliding);  // ‖ RopeQ (in-place on disjoint kn/qn)
            emit(Kernel::KvAppend, L, sliding);
        }
        emit(Kernel::Sdpa,         L, sliding);  // shared layers read source-layer pages
        emit(Kernel::QmvO,         L, sliding);
        emit(Kernel::PostAttnNorm, L, sliding);
        emit(Kernel::AttnResidual, L, sliding);

        // FFN (GeGLU-tanh)
        emit(Kernel::FfnNorm,     L, sliding);
        emit(Kernel::QmvGate,     L, sliding);
        emit(Kernel::QmvUp,       L, sliding);
        emit(Kernel::GegluTanh,   L, sliding);
        emit(Kernel::QmvDown,     L, sliding);
        emit(Kernel::PostFfnNorm, L, sliding);
        emit(Kernel::FfnResidual, L, sliding);

        // PLE residual + layer scalar
        emit(Kernel::PleGateGemv,      L, sliding);
        emit(Kernel::PleGeglu,         L, sliding);
        emit(Kernel::PleProjLayerGemv, L, sliding);
        emit(Kernel::PleNorm,          L, sliding);
        emit(Kernel::PleResidual,      L, sliding);
        emit(Kernel::LayerScalar,      L, sliding);
    }

    // ── tail (once) ──
    emit(Kernel::FinalRms,      -1, false);
    emit(Kernel::LmHead,        -1, false);  // tied embed^T, vocab=262144
    emit(Kernel::FinalSoftcap,  -1, false);  // 30 * tanh(logits / 30)
    emit(Kernel::Argmax,        -1, false);  // optional device argmax (I3)

    return d;
}

// Convenience counts (for the skeleton self-test + the harness banner).
struct Gemma4DagStats {
    int total = 0;
    int n_shared_layers = 0;
    int n_full_attn = 0;
    int n_sliding_attn = 0;
    int n_gemv = 0;   // dense bf16 matvecs (the dominant compute)
};

inline Gemma4DagStats dag_stats(const std::vector<Gemma4Dispatch>& d,
                                const Gemma4Geometry& g) {
    Gemma4DagStats s;
    s.total = static_cast<int>(d.size());
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_kv_shared(L)) ++s.n_shared_layers;
        if (Gemma4Geometry::is_full_attn(L)) ++s.n_full_attn; else ++s.n_sliding_attn;
    }
    for (const auto& x : d) {
        switch (x.kind) {
            case Kernel::QmvQ: case Kernel::QmvK: case Kernel::QmvV: case Kernel::QmvO:
            case Kernel::QmvGate: case Kernel::QmvUp: case Kernel::QmvDown:
            case Kernel::PleProjGemv: case Kernel::PleGateGemv:
            case Kernel::PleProjLayerGemv: case Kernel::LmHead:
                ++s.n_gemv; break;
            default: break;
        }
    }
    return s;
}

}  // namespace pie_metal_driver::raw_metal::gemma4
