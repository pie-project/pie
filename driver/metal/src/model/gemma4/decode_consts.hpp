#pragma once

/// Gemma 4's per-dispatch constants.
///
/// Unlike qwen3.5's, these are per LAYER rather than per kind: head_dim, the
/// RoPE base, the rotated fraction and the MLP width all depend on the layer's
/// attention type and on whether it sits in the KV-shared range.

#include <vector>

#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::gemma4 {

/// A matvec's `(in_vec, out_vec)`, or `{0, 0}` when the kind is not one.
///
/// Inline (and not in decode_consts.cpp) deliberately: it is pure geometry,
/// and the host-only shape tests use it on platforms where the binding half
/// of this header's surface — which calls into RawMetalContext — never links.
struct KN {
    int K = 0;
    int N = 0;
};
inline KN qmv_kn(Kind k, const Gemma4Geometry& g, int layer) {
    const int H = g.hidden;
    const int hd = layer >= 0 ? g.head_dim_of(layer) : g.head_dim;
    const int q_dim = g.n_q_heads * hd;
    const int kv_dim = g.n_kv_heads * hd;
    const int inter = layer >= 0 ? g.intermediate_of(layer) : g.intermediate;
    const int ple_total = g.n_layers * g.per_layer_emb_dim;
    switch (k) {
        case Kind::QmvQ: return {H, q_dim};
        case Kind::QmvK: return {H, kv_dim};
        case Kind::QmvV: return {H, kv_dim};
        case Kind::QmvO: return {q_dim, H};
        case Kind::QmvGate: return {H, inter};
        case Kind::QmvUp: return {H, inter};
        case Kind::QmvDown: return {inter, H};
        case Kind::LmHead: return {H, g.vocab};
        // PLE: the model projection fans hidden out to the whole table; the
        // per-layer gate and projection work one layer's slice at a time.
        case Kind::PleProjGemv: return {H, ple_total};
        case Kind::PleGateGemv: return {H, g.per_layer_emb_dim};
        case Kind::PleProjLayerGemv: return {g.per_layer_emb_dim, H};
        default: return {0, 0};
    }
}

/// Bind every constant the DAG's dispatches read. Returns how many were bound,
/// which is the number a test can pin.
int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const Gemma4Geometry& g);

}  // namespace pie::metal::gemma4
