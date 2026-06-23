// scratch_schedule.cpp — derive the decode DAG's activation dataflow and linear-scan
// color the live ranges onto the SCRATCH_POOL ping-pong buffers (beta's WAR/WAW lane).
//
// The dataflow below mirrors build_decode_dag's fixed order exactly. Each logical
// activation tensor is an SSA "value"; in-place kernels (q_norm/k_norm/rope/attn_gate)
// read+write the SAME value (one buffer). We compute each value's live interval
// [first_def_ordinal, last_use_ordinal] and greedily color: two values that overlap
// (inclusive endpoints — so a same-dispatch WAR and any concurrent ‖-pair output always
// interfere) get distinct buffers. The result is hazard-free by construction.
//
// Slots delta owns (weights, GDN conv/recurrent state, KV pages, IO scalars/logits) are
// NOT modeled here — only the activation X/Out scratch this lane binds.

#include "scratch_schedule.hpp"

#include <algorithm>

namespace pie_metal_driver::raw_metal {

namespace {

// A single scratch slot a dispatch touches.
struct Use {
    int     ordinal;
    uint8_t bind_index;
    int     value;
    bool    write;
};

// bind-index constants (from decode_abi.hpp bind:: enums) for the activation slots.
namespace bi {
constexpr uint8_t EmbedOut   = 4;             // bind::Embed::Out
constexpr uint8_t RmsX = 0, RmsOut = 2;       // bind::Rms
constexpr uint8_t QmvX = 3, QmvOut = 4;       // bind::Qmv
constexpr uint8_t DenseX = 1, DenseOut = 2;   // bind::Dense
constexpr uint8_t QSplitIn = 0, QSplitQ = 1, QSplitGate = 2;  // bind::QSplit
constexpr uint8_t RopeQ = 0, RopeK = 1, RopeQOut = 2, RopeKOut = 3;  // bind::Rope
constexpr uint8_t SdpaQ = 0, SdpaOut = 3;     // bind::Sdpa (K/V from KV region)
constexpr uint8_t AttnGateAttn = 0, AttnGateGate = 1;  // bind::AttnGate (in-place)
constexpr uint8_t KvAppendK = 0, KvAppendV = 1;        // bind::KvAppend (out -> KV pages)
constexpr uint8_t GdnMixed = 0, GdnCoreOut = 3, GdnAGate = 8, GdnBGate = 9;  // bind::GdnCore
constexpr uint8_t GatedRmsX = 0, GatedRmsZ = 1, GatedRmsOut = 3;  // bind::GatedRms
constexpr uint8_t ResidX = 0, ResidR = 1, ResidOut = 2;  // bind::Residual
constexpr uint8_t SiluGate = 0, SiluUp = 1, SiluOut = 2;  // bind::SiluMul
}  // namespace bi

}  // namespace

ScratchSchedule build_scratch_schedule(const std::vector<Dispatch>& dag,
                                       const DecodeGeometry& g) {
    std::vector<Use> uses;
    int next_value = 0;
    auto fresh = [&]() { return next_value++; };

    // Live activation handles, threaded through the DAG order.
    int resid = -1;     // current residual stream (layer input / accumulator)
    int normed = -1;    // last Rms/FfnRms/FinalRms output feeding projections
    int q = -1, gate = -1, kk = -1, vv = -1, attn = -1;        // attn temporaries
    int mixed = -1, zg = -1, ag = -1, bg = -1, core = -1, gnorm = -1, gdnout = -1;  // GDN
    int gp = -1, up = -1, hh = -1, dn = -1;                    // mlp temporaries

    auto rd = [&](int ord, uint8_t b, int val) { uses.push_back({ord, b, val, false}); };
    auto wr = [&](int ord, uint8_t b, int val) { uses.push_back({ord, b, val, true}); };

    for (const Dispatch& d : dag) {
        const int o = d.ordinal;
        switch (d.kind) {
            case Kernel::EmbedGather:
                resid = fresh(); wr(o, bi::EmbedOut, resid);
                break;

            // norms feeding the block (input/ffn) produce `normed` from `resid`.
            case Kernel::Rms:
            case Kernel::FfnRms:
                normed = fresh(); rd(o, bi::RmsX, resid); wr(o, bi::RmsOut, normed);
                break;
            case Kernel::FinalRms:
                normed = fresh(); rd(o, bi::RmsX, resid); wr(o, bi::RmsOut, normed);
                break;

            // ── GDN block ──
            case Kernel::QmvIn:
                mixed = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, mixed);
                break;
            case Kernel::QmvInZ:
                zg = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, zg);
                break;
            case Kernel::GdnInA:
                ag = fresh(); rd(o, bi::DenseX, normed); wr(o, bi::DenseOut, ag);
                break;
            case Kernel::GdnInB:
                bg = fresh(); rd(o, bi::DenseX, normed); wr(o, bi::DenseOut, bg);
                break;
            case Kernel::GdnCore:
                core = fresh();
                rd(o, bi::GdnMixed, mixed); rd(o, bi::GdnAGate, ag); rd(o, bi::GdnBGate, bg);
                wr(o, bi::GdnCoreOut, core);
                break;
            case Kernel::GatedRms:
                gnorm = fresh();
                rd(o, bi::GatedRmsX, core); rd(o, bi::GatedRmsZ, zg); wr(o, bi::GatedRmsOut, gnorm);
                break;
            case Kernel::QmvOut:
                gdnout = fresh(); rd(o, bi::QmvX, gnorm); wr(o, bi::QmvOut, gdnout);
                break;

            // ── Full-attn block ──
            case Kernel::QmvQ:
                q = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, q);
                break;
            case Kernel::QSplit: {
                int Q = fresh(); gate = fresh();
                rd(o, bi::QSplitIn, q); wr(o, bi::QSplitQ, Q); wr(o, bi::QSplitGate, gate);
                q = Q;  // post-split query replaces the 2×-wide buffer
                break;
            }
            case Kernel::QmvK:
                kk = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, kk);
                break;
            case Kernel::QmvV:
                vv = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, vv);
                break;
            case Kernel::QNorm:  // in-place on the query heads
                rd(o, bi::RmsX, q); wr(o, bi::RmsOut, q);
                break;
            case Kernel::KNorm:  // in-place on the key heads
                rd(o, bi::RmsX, kk); wr(o, bi::RmsOut, kk);
                break;
            case Kernel::Rope:   // in-place on the query heads
                rd(o, bi::RopeQ, q); wr(o, bi::RopeQOut, q);
                break;
            case Kernel::RopeK:  // in-place on the key heads
                rd(o, bi::RopeK, kk); wr(o, bi::RopeKOut, kk);
                break;
            case Kernel::KvAppend:  // k/v -> KV pages (delta's); k/v read from scratch
                rd(o, bi::KvAppendK, kk); rd(o, bi::KvAppendV, vv);
                break;
            case Kernel::Sdpa:  // Q from scratch, K/V from KV region; out -> attn
                attn = fresh(); rd(o, bi::SdpaQ, q); wr(o, bi::SdpaOut, attn);
                break;
            case Kernel::AttnGate:  // attn *= sigmoid(gate), in-place on attn
                rd(o, bi::AttnGateAttn, attn); rd(o, bi::AttnGateGate, gate);
                wr(o, bi::AttnGateAttn, attn);
                break;
            case Kernel::QmvO:
                gdnout = fresh(); rd(o, bi::QmvX, attn); wr(o, bi::QmvOut, gdnout);
                break;

            // attn/gdn epilogue + mlp epilogue both fold into Residual.
            case Kernel::Residual: {
                int nr = fresh();
                rd(o, bi::ResidX, resid); rd(o, bi::ResidR, gdnout); wr(o, bi::ResidOut, nr);
                resid = nr;
                break;
            }

            // ── SwiGLU MLP ──
            case Kernel::QmvGate:
                gp = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, gp);
                break;
            case Kernel::QmvUp:
                up = fresh(); rd(o, bi::QmvX, normed); wr(o, bi::QmvOut, up);
                break;
            case Kernel::SiluMul:
                hh = fresh(); rd(o, bi::SiluGate, gp); rd(o, bi::SiluUp, up); wr(o, bi::SiluOut, hh);
                break;
            case Kernel::QmvDown:
                dn = fresh(); rd(o, bi::QmvX, hh); wr(o, bi::QmvOut, dn);
                break;
            case Kernel::LayerOut: {
                int nr = fresh();
                rd(o, bi::ResidX, resid); rd(o, bi::ResidR, dn); wr(o, bi::ResidOut, nr);
                resid = nr;
                break;
            }

            // tail lm_head: reads normed_final from scratch, writes logits to IO (not scratch).
            case Kernel::QmvLmHead:
                rd(o, bi::QmvX, normed);
                break;
            case Kernel::Argmax:  // logits/next-token both in IO; no scratch
                break;
        }
    }

    // Concurrency: dispatch i runs concurrently with i+1 (no barrier between) for the few
    // independent ‖-pairs (k‖v proj, q‖k norm, gate‖up proj). A buffer last-read at ordinal
    // i must NOT be rewritten by a dispatch concurrent with i, so we extend such a value's
    // interval by one ordinal to forbid that reuse (otherwise read@i races write@i+1).
    auto concurrent_after = [&](int i) -> bool {
        if (i + 1 >= (int)dag.size()) return false;
        const Dispatch& a = dag[i]; const Dispatch& b = dag[i + 1];
        if (a.layer != b.layer) return false;
        return (a.kind == Kernel::QmvK && b.kind == Kernel::QmvV) ||
               (a.kind == Kernel::QNorm && b.kind == Kernel::KNorm) ||
               (a.kind == Kernel::QmvGate && b.kind == Kernel::QmvUp);
    };

    // Live intervals per value.
    int nval = next_value;
    std::vector<int> def(nval, -1), last(nval, -1);
    for (const Use& u : uses) {
        if (def[u.value] < 0 || u.ordinal < def[u.value]) def[u.value] = u.ordinal;
        if (u.ordinal > last[u.value]) last[u.value] = u.ordinal;
    }
    for (int v = 0; v < nval; ++v)
        if (last[v] >= 0 && concurrent_after(last[v])) last[v] += 1;  // forbid concurrent reuse

    // Linear-scan coloring: values sorted by def; free a buffer once its holder's last
    // use has passed. Inclusive overlap => same-dispatch WAR + concurrent ‖-pair outputs
    // always interfere (distinct buffers), which is the hazard guarantee.
    std::vector<int> order(nval);
    for (int i = 0; i < nval; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return def[a] < def[b]; });

    std::vector<int> color(nval, -1);
    std::vector<int> buf_free_at;  // buf_free_at[b] = ordinal after which buffer b is free
    for (int v : order) {
        int chosen = -1;
        for (size_t b = 0; b < buf_free_at.size(); ++b) {
            if (buf_free_at[b] < def[v]) { chosen = (int)b; break; }  // strictly before -> no overlap
        }
        if (chosen < 0) { chosen = (int)buf_free_at.size(); buf_free_at.push_back(-1); }
        color[v] = chosen;
        buf_free_at[chosen] = last[v];
    }

    ScratchSchedule sched;
    sched.per_dispatch.resize(dag.size());
    sched.colors_used = (int)buf_free_at.size();
    for (const Use& u : uses) {
        sched.per_dispatch[u.ordinal].binds.push_back({u.bind_index, color[u.value]});
    }

    // Self-check: no two values with overlapping (concurrency-extended) live intervals may
    // share a buffer. Proves the schedule is WAR/WAW hazard-free by construction.
    sched.hazard_free = true;
    for (int a = 0; a < nval && sched.hazard_free; ++a) {
        if (def[a] < 0) continue;
        for (int b = a + 1; b < nval; ++b) {
            if (def[b] < 0) continue;
            bool overlap = std::max(def[a], def[b]) <= std::min(last[a], last[b]);
            if (overlap && color[a] == color[b]) { sched.hazard_free = false; break; }
        }
    }

    (void)g;
    return sched;
}

}  // namespace pie_metal_driver::raw_metal
