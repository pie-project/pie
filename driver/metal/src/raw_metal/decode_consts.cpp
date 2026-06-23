// decode_consts.cpp — bind all geometry-derived const params, by ordinal, over beta's DAG.

#include "decode_consts.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "decode_step.hpp"     // beta: Dispatch{kind,ordinal,layer,grid,tg}
#include "mtl4_context.hpp"

namespace pie_metal_driver::raw_metal {

namespace {

// ── Kernel param structs, replicated EXACTLY from the .metal sources ──
struct RmsParams {       // rms_norm.metal:22  (buffer 3)
    float eps;
    uint32_t axis_size;  // feature dim
    uint32_t w_stride;   // 1 (contiguous)
    uint32_t plus_one;   // qwen3.5: 1 for ALL norms → gain (1+weight)
};
struct GatedRmsParams {  // gated_rms.metal:20  (buffer 4)
    float eps;
    uint32_t vd;         // value-head dim (reduction axis)
};
struct GdnCoreParams {   // gdn_core.metal:39  (buffer 11)
    int32_t Dk, Dv, Hk, Hv, conv_dim, Kc, q_off, k_off, v_off;
    float   eps, inv_sqrt_dk;
};

// Bind a POD constant value into a fresh resident slot at (ordinal, bind_index).
template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, uint8_t idx, const V& val, int* count) {
    SlotHandle s = ctx.heap_alloc(sizeof(V));
    if (!s.valid()) throw std::runtime_error("decode_consts: heap_alloc failed (budget too small)");
    std::memcpy(s.contents(), &val, sizeof(V));
    ctx.arg_bind_ordinal(ord, idx, s);
    if (count) ++*count;
}

// qmv in_vec (K) / out_vec (N) per kind, from geometry (matches the staged weight shapes).
struct KN { int K, N; };
KN qmv_kn(Kernel k, const DecodeGeometry& g) {
    const int H = g.hidden;
    const int q_wide = 2 * g.n_q_heads * g.head_dim;   // 2×-wide gated q_proj (4096)
    const int kv_dim = g.n_kv_heads * g.head_dim;      // 512
    const int q_dim  = g.n_q_heads * g.head_dim;       // 2048
    switch (k) {
        case Kernel::QmvIn:     return {H, g.gdn_conv_dim};   // 1024 → 6144
        case Kernel::QmvInZ:    return {H, g.gdn_v_total};    // 1024 → 2048
        case Kernel::QmvOut:    return {g.gdn_v_total, H};    // 2048 → 1024
        case Kernel::QmvQ:      return {H, q_wide};           // 1024 → 4096
        case Kernel::QmvK:      return {H, kv_dim};           // 1024 → 512
        case Kernel::QmvV:      return {H, kv_dim};           // 1024 → 512
        case Kernel::QmvO:      return {q_dim, H};            // 2048 → 1024
        case Kernel::QmvGate:   return {H, g.intermediate};   // 1024 → 3584
        case Kernel::QmvUp:     return {H, g.intermediate};   // 1024 → 3584
        case Kernel::QmvDown:   return {g.intermediate, H};   // 3584 → 1024
        case Kernel::QmvLmHead: return {H, g.vocab};          // 1024 → 248320
        default:                return {0, 0};
    }
}

bool is_qmv(Kernel k) { return qmv_kn(k, DecodeGeometry{}).N != 0; }

}  // namespace

int bind_decode_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const DecodeGeometry& g, int max_ctx) {
    int count = 0;

    // rope: x[h*head_dim + i], rotary half from grid.x. scale=1.0 (qwen3.6 default mrope),
    // base = log2(theta).
    const float rope_scale = 1.0f;
    const float rope_base  = std::log2(g.rope_theta);

    // sdpa / kv_append cache layout [n_kv_heads, max_ctx, head_dim]:
    const size_t head_stride = size_t(max_ctx) * size_t(g.head_dim);  // *_head_stride
    const size_t seq_stride  = size_t(g.head_dim);                    // *_seq_stride
    const int    gqa_factor  = g.n_q_heads / g.n_kv_heads;            // 4
    const float  sdpa_scale  = 1.0f / std::sqrt(float(g.head_dim));   // 1/sqrt(256)

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const Kernel k = d.kind;

        if (is_qmv(k)) {
            const KN kn = qmv_kn(k, g);
            bind_const<int>(ctx, ord, (uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<int>(ctx, ord, (uint8_t)bind::Qmv::N, kn.N, &count);
            continue;
        }

        switch (k) {
            case Kernel::EmbedGather:
                bind_const<int>(ctx, ord, (uint8_t)bind::Embed::Hidden, g.hidden, &count);
                break;

            // RMSNorm variants — plus_one=1 for ALL (qwen3.5/Gemma gain = 1+weight).
            case Kernel::Rms:
            case Kernel::FfnRms:
            case Kernel::FinalRms:
                bind_const<RmsParams>(ctx, ord, (uint8_t)bind::Rms::Params,
                                      RmsParams{g.eps, (uint32_t)g.hidden, 1u, 1u}, &count);
                break;
            case Kernel::QNorm:
            case Kernel::KNorm:
                bind_const<RmsParams>(ctx, ord, (uint8_t)bind::Rms::Params,
                                      RmsParams{g.eps, (uint32_t)g.head_dim, 1u, 1u}, &count);
                break;

            // GDN in_proj_a / in_proj_b — DENSE bf16 GEMV [16,1024].
            case Kernel::GdnInA:
            case Kernel::GdnInB:
                bind_const<uint32_t>(ctx, ord, (uint8_t)bind::Dense::K, (uint32_t)g.hidden, &count);
                bind_const<uint32_t>(ctx, ord, (uint8_t)bind::Dense::N, (uint32_t)g.gdn_v_heads, &count);
                break;

            case Kernel::GdnCore:
                bind_const<GdnCoreParams>(ctx, ord, (uint8_t)bind::GdnCore::Params,
                    GdnCoreParams{g.gdn_k_dim, g.gdn_v_dim, g.gdn_k_heads, g.gdn_v_heads,
                                  g.gdn_conv_dim, g.gdn_conv_k,
                                  /*q_off*/0, /*k_off*/g.gdn_k_heads * g.gdn_k_dim,
                                  /*v_off*/2 * g.gdn_k_heads * g.gdn_k_dim,
                                  g.eps, 1.0f / std::sqrt(float(g.gdn_k_dim))}, &count);
                break;

            case Kernel::GatedRms:
                bind_const<GatedRmsParams>(ctx, ord, (uint8_t)bind::GatedRms::Params,
                                           GatedRmsParams{g.eps, (uint32_t)g.gdn_v_dim}, &count);
                break;

            case Kernel::QSplit:
                bind_const<int>(ctx, ord, (uint8_t)bind::QSplit::HeadDim, g.head_dim, &count);
                break;

            case Kernel::Rope:
            case Kernel::RopeK:
                bind_const<float>(ctx, ord, (uint8_t)bind::Rope::Scale,   rope_scale, &count);
                bind_const<float>(ctx, ord, (uint8_t)bind::Rope::Base,    rope_base,  &count);
                bind_const<int>  (ctx, ord, (uint8_t)bind::Rope::HeadDim, g.head_dim, &count);
                break;

            case Kernel::KvAppend:
                bind_const<int>   (ctx, ord, (uint8_t)bind::KvAppend::HeadDim,     g.head_dim,  &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppend::KHeadStride, head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppend::KSeqStride,  seq_stride,  &count);
                break;

            case Kernel::Sdpa:
                bind_const<int>   (ctx, ord, (uint8_t)bind::Sdpa::GqaFactor,   gqa_factor,  &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::KHeadStride, head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::KSeqStride,  seq_stride,  &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::VHeadStride, head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::VSeqStride,  seq_stride,  &count);
                bind_const<float> (ctx, ord, (uint8_t)bind::Sdpa::Scale,       sdpa_scale,  &count);
                break;

            // No const params: AttnGate, SiluMul, Residual, LayerOut, Argmax.
            default:
                break;
        }
    }
    return count;
}

size_t decode_consts_budget(const std::vector<Dispatch>& dag) {
    // Worst case 6 const slots/dispatch (sdpa), each ≤ 256-aligned. Be generous.
    return (dag.size() * 6 + 64) * 256;
}

}  // namespace pie_metal_driver::raw_metal
