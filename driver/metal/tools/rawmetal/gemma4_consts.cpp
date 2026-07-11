// gemma4_consts.cpp — bind all geometry-derived const params for the gemma4 decode DAG.
//
// Replicates the .metal param structs EXACTLY and binds them by ordinal. gemma4 specifics
// vs qwen3.6: RmsParams.plus_one = 0 (plain RMSNorm), SDPA scale = 1.0 (Q/K-norm absorbs
// 1/sqrt(d)) + a per-layer sliding window (512 sliding / 0 full) at buffer 11, per-type
// rope theta, per-layer head_dim / intermediate, the GeGLU-tanh / PLE-combine / softcap /
// layer-scalar pointwise params, and the scaled-embed (sqrt(hidden)/sqrt(ple_dim)) scale.

#include "gemma4_heap_bind.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pie::metal::gemma4 {

namespace {

// param structs — byte-identical to the .metal sources.
struct RmsParams      { float eps; uint32_t axis_size; uint32_t w_stride; uint32_t plus_one; }; // rms_norm.metal buf3
struct VNormParams    { float eps; uint32_t axis_size; };                                       // vnorm.metal buf2
struct GegluParams    { uint32_t n; };                                                          // geglu_tanh.metal buf3
struct PleCombineParams { float inv_sqrt2; uint32_t n; };                                        // ple_combine.metal buf3
struct SoftcapParams  { float cap; uint32_t n; };                                               // logit_softcap.metal buf2
struct LayerScalarParams { uint32_t n; };                                                       // layer_scalar.metal buf3

const uint8_t kSdpaWindow = 11;  // sdpa_sliding.metal buffer(11) — gemma-specific window

template <class V>
void bind_const(RawMetalContext& ctx, int ord, uint8_t idx, const V& val, int* count) {
    SlotHandle s = ctx.heap_alloc(sizeof(V));
    if (!s.valid()) throw std::runtime_error("gemma4_consts: heap_alloc failed");
    std::memcpy(s.contents(), &val, sizeof(V));
    ctx.arg_bind_ordinal(ord, idx, s);
    if (count) ++*count;
}

struct KN { int K, N; };
bool qmv_kn(Kernel k, int L, const Gemma4Geometry& g, KN& out) {
    switch (k) {
        case Kernel::PleProjGemv:      out = {g.hidden, g.n_layers * g.per_layer_emb_dim}; return true;
        case Kernel::QmvQ:             out = {g.hidden, g.q_dim_at(L)};                    return true;
        case Kernel::QmvK:             out = {g.hidden, g.kv_dim_at(L)};                   return true;
        case Kernel::QmvV:             out = {g.hidden, g.kv_dim_at(L)};                   return true;
        case Kernel::QmvO:             out = {g.q_dim_at(L), g.hidden};                    return true;
        case Kernel::QmvGate:          out = {g.hidden, g.intermediate_at(L)};             return true;
        case Kernel::QmvUp:            out = {g.hidden, g.intermediate_at(L)};             return true;
        case Kernel::QmvDown:          out = {g.intermediate_at(L), g.hidden};             return true;
        case Kernel::PleGateGemv:      out = {g.hidden, g.per_layer_emb_dim};              return true;
        case Kernel::PleProjLayerGemv: out = {g.per_layer_emb_dim, g.hidden};              return true;
        case Kernel::LmHead:           out = {g.hidden, g.vocab};                          return true;
        default: return false;
    }
}

// axis size for the rms_single_row norms (hidden / per-head head_dim / ple_dim).
uint32_t rms_axis(Kernel k, int L, const Gemma4Geometry& g) {
    switch (k) {
        case Kernel::QNorm: case Kernel::KNorm: return uint32_t(g.head_dim_at(L));
        case Kernel::PleProjNorm:               return uint32_t(g.per_layer_emb_dim);
        default:                                return uint32_t(g.hidden);  // the hidden-wide norms
    }
}

}  // namespace

int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Gemma4Dispatch>& dag,
                       const Gemma4Geometry& g, int max_ctx) {
    int count = 0;
    const int ple_w = g.n_layers * g.per_layer_emb_dim;

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        const Kernel k = d.kind;

        KN kn;
        if (qmv_kn(k, L, g, kn)) {
            bind_const<int>(ctx, ord, (uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<int>(ctx, ord, (uint8_t)bind::Qmv::N, kn.N, &count);
            continue;
        }

        switch (k) {
            case Kernel::EmbedGather:
                bind_const<int>  (ctx, ord, (uint8_t)bind::EmbedScaled::Hidden, g.hidden, &count);
                bind_const<float>(ctx, ord, (uint8_t)bind::EmbedScaled::Scale,
                                  std::sqrt(float(g.hidden)), &count);
                break;
            case Kernel::PleTokenGather:
                bind_const<int>  (ctx, ord, (uint8_t)bind::EmbedScaled::Hidden, ple_w, &count);
                bind_const<float>(ctx, ord, (uint8_t)bind::EmbedScaled::Scale,
                                  std::sqrt(float(g.per_layer_emb_dim)), &count);
                break;

            // RMSNorm variants — plain rms (plus_one=0).
            case Kernel::PleProjNorm: case Kernel::AttnNorm: case Kernel::QNorm:
            case Kernel::KNorm: case Kernel::PostAttnNorm: case Kernel::FfnNorm:
            case Kernel::PostFfnNorm: case Kernel::PleNorm: case Kernel::FinalRms:
                bind_const<RmsParams>(ctx, ord, (uint8_t)bind::Rms::Params,
                    RmsParams{g.eps, rms_axis(k, L, g), 1u, 0u}, &count);
                break;

            case Kernel::VNorm:
                bind_const<VNormParams>(ctx, ord, (uint8_t)bind::VNorm::Axis,  // buffer 2 = Params
                    VNormParams{g.eps, uint32_t(g.head_dim_at(L))}, &count);
                break;

            case Kernel::RopeQ: case Kernel::RopeK:
                bind_const<float>(ctx, ord, (uint8_t)bind::Rope::Scale,   1.0f, &count);
                bind_const<float>(ctx, ord, (uint8_t)bind::Rope::Base,
                                  std::log2(g.rope_theta_at(L)), &count);
                bind_const<int>  (ctx, ord, (uint8_t)bind::Rope::HeadDim, g.head_dim_at(L), &count);
                break;

            case Kernel::KvAppend: {
                const size_t hstride = size_t(max_ctx) * g.head_dim_at(L);
                bind_const<int>   (ctx, ord, (uint8_t)bind::KvAppend::HeadDim,     g.head_dim_at(L), &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppend::KHeadStride, hstride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppend::KSeqStride,  size_t(g.head_dim_at(L)), &count);
                break;
            }

            case Kernel::Sdpa: {
                const size_t hstride = size_t(max_ctx) * g.head_dim_at(L);
                const size_t sstride = size_t(g.head_dim_at(L));
                const int    window  = Gemma4Geometry::is_full_attn(L) ? 0 : g.sliding_window;
                bind_const<int>   (ctx, ord, (uint8_t)bind::Sdpa::GqaFactor,   g.n_q_heads / g.n_kv_heads, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::KHeadStride, hstride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::KSeqStride,  sstride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::VHeadStride, hstride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::VSeqStride,  sstride, &count);
                bind_const<float> (ctx, ord, (uint8_t)bind::Sdpa::Scale,       g.attn_scale_at(L), &count);
                bind_const<int>   (ctx, ord, kSdpaWindow,                      window, &count);
                break;
            }

            case Kernel::GegluTanh:
                bind_const<GegluParams>(ctx, ord, (uint8_t)bind::Geglu::N,
                    GegluParams{uint32_t(g.intermediate_at(L))}, &count);
                break;
            case Kernel::PleGeglu:
                bind_const<GegluParams>(ctx, ord, (uint8_t)bind::Geglu::N,
                    GegluParams{uint32_t(g.per_layer_emb_dim)}, &count);
                break;

            case Kernel::PleCombine:
                bind_const<PleCombineParams>(ctx, ord, (uint8_t)bind::PleCombine::InvSqrt2,
                    PleCombineParams{0.70710678118654752f, uint32_t(ple_w)}, &count);
                break;

            case Kernel::LayerScalar:
                bind_const<LayerScalarParams>(ctx, ord, (uint8_t)bind::LayerScalar::N,  // buffer 3 = Params
                    LayerScalarParams{uint32_t(g.hidden)}, &count);
                break;

            case Kernel::FinalSoftcap:
                bind_const<SoftcapParams>(ctx, ord, (uint8_t)bind::Softcap::Cap,
                    SoftcapParams{g.final_softcap, uint32_t(g.vocab)}, &count);
                break;

            default:
                break;  // residuals / pass-through have no const params
        }
    }
    return count;
}

}  // namespace pie::metal::gemma4
