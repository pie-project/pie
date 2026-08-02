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
///
/// `rows` is the token count: a GEMM must be told M, and an elementwise kernel
/// over a contiguous [rows, width] buffer counts rows*width. Defaults to the
/// decode path's single row.
///
/// `paged` picks which attention ABI the two KV kinds are bound against.
/// `Kind::Sdpa` and `Kind::KvAppend` are ONE kind each whose kernel changes with
/// the batch -- contiguous ring at M=1, page table at M>1 -- and the two ABIs
/// put different meanings at the same slot indices. Binding the contiguous
/// constants against the paged shader is not a crash: `bind::Sdpa::KHeadStride`
/// is `bind::SdpaPaged::ReqOfToken`, so it is a stride read as a pointer.
/// `head_rows` is how many rows the fire will SAMPLE -- what `Kind::RowGather`
/// compacts, and what the tail after it runs on. 0 means "all of them", which
/// is what a test that reads every row wants.
int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const Gemma4Geometry& g, int rows = 1, bool paged = false,
                       int head_rows = 0);

}  // namespace pie::metal::gemma4
