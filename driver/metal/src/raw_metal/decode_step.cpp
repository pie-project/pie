// decode_step.cpp — build + encode the qwen3.6 per-token decode DAG (beta's lane).
//
// The DAG order mirrors wiki mac-raw-metal-decode-dag. Barrier modeling (which
// dispatches run concurrently vs. behind a barrier edge) is beta's WAR/WAW
// responsibility; the ‖ (concurrent) groups below drop the barrier count from ~322
// dispatches to ~250 barriers, matching the locked DAG.

#include "decode_step.hpp"

namespace pie_metal_driver::raw_metal {

namespace {

// qmv output widths at qwen3.6 shapes.
int qmv_out_n(Kernel k, const DecodeGeometry& g) {
    const int q_dim  = g.n_q_heads * g.head_dim;   // 2048
    const int kv_dim = g.n_kv_heads * g.head_dim;  // 512
    switch (k) {
        case Kernel::QmvQ:      return q_dim;
        case Kernel::QmvK:      return kv_dim;
        case Kernel::QmvV:      return kv_dim;
        case Kernel::QmvO:      return g.hidden;
        // GDN in-proj: conv input (conv_dim) + gate z (v_total) + a,b (v_heads each).
        // TODO(delta): confirm exact in-proj width from the checkpoint.
        case Kernel::QmvIn:     return g.gdn_conv_dim + g.gdn_v_total + 2 * g.gdn_v_heads;
        case Kernel::QmvOut:    return g.hidden;
        case Kernel::QmvGate:   return g.intermediate;
        case Kernel::QmvUp:     return g.intermediate;
        case Kernel::QmvDown:   return g.hidden;
        case Kernel::QmvLmHead: return g.vocab;
        default:                return 0;
    }
}

// A DAG entry with its barrier flag (false = concurrent with the following dispatch).
struct Step { Kernel kind; bool barrier_after; };

}  // namespace

Threadgroup default_tg(Kernel k, const DecodeGeometry& /*g*/) {
    switch (k) {
        case Kernel::GdnCore:   return {32, 4, 1};   // beta-validated
        case Kernel::Sdpa:      return {1024, 1, 1};
        case Kernel::Argmax:    return {1024, 1, 1};
        case Kernel::QmvQ: case Kernel::QmvK: case Kernel::QmvV: case Kernel::QmvO:
        case Kernel::QmvIn: case Kernel::QmvOut: case Kernel::QmvGate: case Kernel::QmvUp:
        case Kernel::QmvDown: case Kernel::QmvLmHead:
            return {32, 2, 1};                       // affine_qmv_fast group; TODO(delta) bn
        default:                return {256, 1, 1};  // rms/rope/qknorm/embed/silu/residual/gatedrms/kvappend
    }
}

Grid default_grid(Kernel k, const DecodeGeometry& g, int out_n) {
    switch (k) {
        case Kernel::GdnCore:
            // grid {32, Vd, R*Vh} (R=1) — beta-validated.
            return {32, static_cast<uint32_t>(g.gdn_v_dim), static_cast<uint32_t>(g.gdn_v_heads)};
        case Kernel::Sdpa:
            return {static_cast<uint32_t>(g.n_q_heads), 1, 1};
        case Kernel::QNorm:
            return {static_cast<uint32_t>(g.n_q_heads), 1, 1};
        case Kernel::KNorm:
            return {static_cast<uint32_t>(g.n_kv_heads), 1, 1};
        case Kernel::QmvQ: case Kernel::QmvK: case Kernel::QmvV: case Kernel::QmvO:
        case Kernel::QmvIn: case Kernel::QmvOut: case Kernel::QmvGate: case Kernel::QmvUp:
        case Kernel::QmvDown: case Kernel::QmvLmHead: {
            // affine_qmv_fast: grid (1, ceil(N/bn), 1). TODO(delta): exact bn per variant.
            constexpr uint32_t kBn = 8;
            const uint32_t n = static_cast<uint32_t>(out_n > 0 ? out_n : 1);
            return {1, (n + kBn - 1) / kBn, 1};
        }
        default:
            return {1, 1, 1};  // single-row rms/rope/embed/etc (M=1)
    }
}

std::vector<Dispatch> build_decode_dag(const DecodeGeometry& g) {
    std::vector<Dispatch> dag;
    int ord = 0;
    auto push = [&](Kernel k, int layer, bool /*barrier_after handled in encode*/) {
        Dispatch d;
        d.kind    = k;
        d.ordinal = ord++;
        d.layer   = layer;
        d.grid    = default_grid(k, g, qmv_out_n(k, g));
        d.tg      = default_tg(k, g);
        dag.push_back(d);
    };

    // EMBED ×1
    push(Kernel::EmbedGather, -1, true);

    for (int L = 0; L < g.n_layers; ++L) {
        if (DecodeGeometry::is_full_attn(L)) {
            // full-attn (11): rms, {qmv_q‖qmv_k‖qmv_v}, {qnorm‖knorm}, rope, kv_append, sdpa, qmv_o, residual
            push(Kernel::Rms,      L, true);
            push(Kernel::QmvQ,     L, false);
            push(Kernel::QmvK,     L, false);
            push(Kernel::QmvV,     L, true);
            push(Kernel::QNorm,    L, false);
            push(Kernel::KNorm,    L, true);
            push(Kernel::Rope,     L, true);
            push(Kernel::KvAppend, L, true);
            push(Kernel::Sdpa,     L, true);
            push(Kernel::QmvO,     L, true);
            push(Kernel::Residual, L, true);
        } else {
            // GDN (6): rms, qmv_in, gdn_core, gated_rms, qmv_out, residual
            push(Kernel::Rms,      L, true);
            push(Kernel::QmvIn,    L, true);
            push(Kernel::GdnCore,  L, true);
            push(Kernel::GatedRms, L, true);
            push(Kernel::QmvOut,   L, true);
            push(Kernel::Residual, L, true);
        }
        // MLP (6, every layer): rms(ffn), {qmv_gate‖qmv_up}, silu_mul, qmv_down, residual
        push(Kernel::Rms,      L, true);
        push(Kernel::QmvGate,  L, false);
        push(Kernel::QmvUp,    L, true);
        push(Kernel::SiluMul,  L, true);
        push(Kernel::QmvDown,  L, true);
        push(Kernel::Residual, L, true);
    }

    // FINAL ×1: final rms, lm_head, [optional] device argmax
    push(Kernel::FinalRms,   -1, true);
    push(Kernel::QmvLmHead,  -1, true);
    push(Kernel::Argmax,     -1, true);  // I3: optional substrate; logits already produced

    return dag;
}

// Barrier flags must match the ‖ markers in build_decode_dag. Kept as a parallel
// table so encode order and barrier order can't drift.
static bool barrier_after_for(const std::vector<Dispatch>& dag, size_t i) {
    // Concurrent (no barrier) iff this dispatch and the next are in a known ‖ group.
    if (i + 1 >= dag.size()) return true;
    const Kernel a = dag[i].kind, b = dag[i + 1].kind;
    if (dag[i].layer != dag[i + 1].layer) return true;
    // qmv_q ‖ qmv_k ‖ qmv_v
    if ((a == Kernel::QmvQ && (b == Kernel::QmvK || b == Kernel::QmvV)) ||
        (a == Kernel::QmvK && b == Kernel::QmvV))
        return false;
    // qnorm ‖ knorm
    if (a == Kernel::QNorm && b == Kernel::KNorm) return false;
    // qmv_gate ‖ qmv_up
    if (a == Kernel::QmvGate && b == Kernel::QmvUp) return false;
    return true;
}

void encode_decode_step(StepEncoder& se,
                        const std::vector<Dispatch>& dag,
                        const DecodeStepPsos& psos) {
    for (size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        se.set_pso(psos[d.kind]);
        se.set_argtable(d.kind, d.ordinal);  // ordinal-keyed (unique, token-stable)
        se.dispatch(d.grid, d.tg);
        if (barrier_after_for(dag, i)) se.barrier();
    }
}

}  // namespace pie_metal_driver::raw_metal
