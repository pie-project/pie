// decode_step.cpp — build + encode the qwen3.6 per-token decode DAG (beta's lane).
//
// DAG order matches the GOLDEN kernel surface (wiki mac-golden-kernel-surface) +
// delta's coverage pass: full gated-attn (q_proj 2×-wide → QSplit, AttnGate), 4-way
// GDN in-projection (QmvIn/QmvInZ 4-bit + GdnInA/GdnInB dense bf16), GdnCore→GatedRms,
// per-class SwiGLU MLP. Launch dims come from delta's decode_dispatch.hpp helpers
// (authoritative); GdnCore's grid is beta's (gdn_core.metal {32,Vd,Vh}/{32,4,1}).
//
// Counts: 1 embed + 18 GDN×15 + 6 full-attn×20 + 2 tail(FinalRms,QmvLmHead) = 393
// raw dispatches (363 golden-tapped: QSplit/KvAppend internal, GdnCore folds into the
// gdn_core tap via GatedRms). +1 if with_argmax.
//
// Barrier modeling (beta's WAR/WAW lane): a barrier follows every dispatch except the
// few adjacent independent pairs (k‖v proj, q‖k norm, gate‖up proj) that read a common
// already-produced input and write disjoint slots.

#include "decode_step.hpp"
#include "decode_dispatch.hpp"

namespace pie_metal_driver::raw_metal {

namespace {

struct LD { Grid grid; Threadgroup tg; };

// GdnCore launch lives with beta's kernel (dispatchThreads = total threads).
// tg spans a TILE of dv (32 simdgroups) so the q/k prologue is computed once per
// threadgroup (tpit.y==0) + shared via threadgroup memory — kills the Vd-fold
// redundancy while keeping full occupancy. grid {32,Vd,Hv} / tg {32,32,1}.
LD gdncore_ld(const DecodeGeometry& g) {
    return { Grid{32, uint32_t(g.gdn_v_dim), uint32_t(g.gdn_v_heads)}, Threadgroup{32, 32, 1} };
}

}  // namespace

std::vector<Dispatch> build_decode_dag(const DecodeGeometry& g, bool with_argmax,
                                       bool fuse_residual) {
    const int q_dim  = g.n_q_heads * g.head_dim;   // 2048 (post-split query)
    const int qg_dim = 2 * q_dim;                  // 4096 (q_proj is 2×-wide [query|gate])
    const int kv_dim = g.n_kv_heads * g.head_dim;  // 512

    std::vector<Dispatch> dag;
    int ord = 0;
    auto emit = [&](Kernel k, int layer, LD ld) {
        Dispatch d; d.kind = k; d.ordinal = ord++; d.layer = layer; d.grid = ld.grid; d.tg = ld.tg;
        dag.push_back(d);
    };
    auto qmv   = [&](int N) { LD l; qmv_dispatch(N, l.grid, l.tg); return l; };
    auto rms   = [&](int row, int rows) { LD l; rms_dispatch(row, rows, l.grid, l.tg); return l; };
    auto rope  = [&](int nh) { LD l; rope_dispatch(g.rotary_dims, nh, l.grid, l.tg); return l; };
    auto resid = [&]() { LD l; residual_dispatch(g.hidden, l.grid, l.tg); return l; };

    // EMBED ×1 (4-bit dequant gather of the shared lm_head bundle).
    { LD l; embed_dispatch(g.hidden, l.grid, l.tg); emit(Kernel::EmbedGather, -1, l); }

    for (int L = 0; L < g.n_layers; ++L) {
        if (DecodeGeometry::is_full_attn(L)) {
            // Full-attn (20): attn_norm, q_proj(4096), q_split, k_proj, v_proj, q_norm,
            // k_norm, rope_q, rope_k, kv_append, sdpa, attn_gate, o_proj, attn_resid + MLP(6).
            emit(Kernel::Rms,    L, rms(g.hidden, 1));
            emit(Kernel::QmvQ,   L, qmv(qg_dim));                       // 2×-wide [query|gate]
            { LD l; q_split_dispatch(g.head_dim, g.n_q_heads, l.grid, l.tg); emit(Kernel::QSplit, L, l); }
            emit(Kernel::QmvK,   L, qmv(kv_dim));
            emit(Kernel::QmvV,   L, qmv(kv_dim));
            emit(Kernel::QNorm,  L, rms(g.head_dim, g.n_q_heads));
            emit(Kernel::KNorm,  L, rms(g.head_dim, g.n_kv_heads));
            emit(Kernel::Rope,   L, rope(g.n_q_heads));
            emit(Kernel::RopeK,  L, rope(g.n_kv_heads));
            { LD l; kv_append_dispatch(g.head_dim, g.n_kv_heads, l.grid, l.tg); emit(Kernel::KvAppend, L, l); }
            { LD l; sdpa_dispatch(g.n_q_heads, l.grid, l.tg); emit(Kernel::Sdpa, L, l); }
            { LD l; attn_gate_dispatch(g.n_q_heads, g.head_dim, l.grid, l.tg); emit(Kernel::AttnGate, L, l); }
            emit(Kernel::QmvO,   L, qmv(g.hidden));
            if (fuse_residual) dag.back().fuse_residual = true;
            else emit(Kernel::Residual, L, resid());
        } else {
            // GDN (15): attn_norm, gdn_in_qkv(6144), gdn_in_z(2048), gdn_in_a, gdn_in_b,
            // gdn_core, gated_rms, gdn_out, attn_resid + MLP(6).
            emit(Kernel::Rms,     L, rms(g.hidden, 1));
            emit(Kernel::QmvIn,   L, qmv(g.gdn_conv_dim));             // 6144, 4-bit
            emit(Kernel::QmvInZ,  L, qmv(g.gdn_v_total));              // 2048, 4-bit (gate z)
            { LD l; dense_gemv_dispatch(g.gdn_v_heads, l.grid, l.tg); emit(Kernel::GdnInA, L, l); }  // dense bf16 [16,1024]
            { LD l; dense_gemv_dispatch(g.gdn_v_heads, l.grid, l.tg); emit(Kernel::GdnInB, L, l); }
            emit(Kernel::GdnCore, L, gdncore_ld(g));                   // beta's fused 1-dispatch core
            { LD l; gated_rms_dispatch(g.gdn_v_heads, g.gdn_v_dim, l.grid, l.tg); emit(Kernel::GatedRms, L, l); }
            emit(Kernel::QmvOut,  L, qmv(g.hidden));                   // gdn_out
            if (fuse_residual) dag.back().fuse_residual = true;
            else emit(Kernel::Residual, L, resid());
        }
        // SwiGLU MLP (6, every layer): ffn_norm, gate_proj, up_proj, swiglu, down_proj, layer_out.
        emit(Kernel::FfnRms,   L, rms(g.hidden, 1));
        emit(Kernel::QmvGate,  L, qmv(g.intermediate));
        emit(Kernel::QmvUp,    L, qmv(g.intermediate));
        { LD l; silu_mul_dispatch(g.intermediate, l.grid, l.tg); emit(Kernel::SiluMul, L, l); }
        emit(Kernel::QmvDown,  L, qmv(g.hidden));
        if (fuse_residual) dag.back().fuse_residual = true;
        else emit(Kernel::LayerOut, L, resid());
    }

    // TAIL: final_norm, lm_head (logits ALWAYS produced, I3), [optional] device argmax.
    emit(Kernel::FinalRms,  -1, rms(g.hidden, 1));
    emit(Kernel::QmvLmHead, -1, qmv(g.vocab));
    if (with_argmax) {
        emit(Kernel::Argmax, -1, LD{ Grid{1024, 1, 1}, Threadgroup{1024, 1, 1} });
    }
    return dag;
}

// Barrier flags: false = this dispatch runs concurrently with the next (no barrier).
// Only adjacent independent pairs that read an already-produced common input and write
// disjoint outputs. Conservative; the gate localizes any over-/under-sync as a port bug.
static bool barrier_after(const std::vector<Dispatch>& dag, size_t i) {
    if (i + 1 >= dag.size()) return true;
    const Dispatch& a = dag[i]; const Dispatch& b = dag[i + 1];
    if (a.layer != b.layer) return true;
    if (a.kind == Kernel::QmvK    && b.kind == Kernel::QmvV) return false;  // k_proj ‖ v_proj
    if (a.kind == Kernel::QNorm   && b.kind == Kernel::KNorm) return false; // q_norm ‖ k_norm
    if (a.kind == Kernel::QmvGate && b.kind == Kernel::QmvUp) return false; // gate_proj ‖ up_proj
    if (a.kind == Kernel::QmvIn   && b.kind == Kernel::QmvInZ) return false; // gdn_in_qkv ‖ gdn_in_z
    if (a.kind == Kernel::GdnInA  && b.kind == Kernel::GdnInB) return false; // gdn_in_a ‖ gdn_in_b
    return true;
}

void encode_decode_step(StepEncoder& se,
                        const std::vector<Dispatch>& dag,
                        const DecodeStepPsos& psos,
                        bool force_barriers,
                        const StepTimingHook* timing) {
    for (size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        if (timing && timing->mark) timing->mark(static_cast<int>(i));  // boundary i
        se.set_pso(d.fuse_residual ? psos.qmv_residual : psos[d.kind]);
        se.set_argtable(d.kind, d.ordinal);  // ordinal-keyed (unique, token-stable)
        se.dispatch(d.grid, d.tg);
        if (force_barriers || barrier_after(dag, i)) se.barrier();
    }
    if (timing && timing->mark) timing->mark(static_cast<int>(dag.size()));  // final boundary
}

}  // namespace pie_metal_driver::raw_metal
