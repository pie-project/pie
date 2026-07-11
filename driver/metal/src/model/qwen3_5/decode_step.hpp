#pragma once
// decode_step.hpp — beta's executor: the per-token qwen3.6 decode-step DAG walker.
//
// Builds the exact ~322-dispatch decode DAG (decode_abi.hpp::Kernel order + the
// per-layer composition from wiki mac-raw-metal-decode-dag) and encodes it into one
// Metal-4 command buffer via alpha's StepEncoder. This is the `encode_fn` passed to
// RawMetalContext::run_step(encode_fn, ab) — run_step returns the encode-ms /
// gpu-exec-ms split (the headline number) with the double-buffered allocator overlap.
//
// Ownership seam:
//   * beta  — the DAG order + dispatch sequence here (this file).
//   * delta — the heap region layout + arg-table binding (slots per dispatch).
//   * alpha — RawMetalContext / StepEncoder / run_step wrappers.
//
// ── Arg-table key (integration note) ─────────────────────────────────────────
// `set_argtable(Kernel, key)` keys the prebuilt MTL4ArgumentTable. (Kernel, layer)
// is NOT unique: within one layer-cycle Rms recurs (input-norm + ffn-norm) and
// Residual recurs (attn/gdn epilogue + mlp epilogue), and qmv kinds recur. So we key
// every dispatch by its FLAT ORDINAL (0..N-1) — unique + stable, since the CB is
// byte-identical every token (I1). delta binds arg tables by the SAME ordinal.
// The Kernel `kind` + `layer` are retained only for charlie's <layer>.<tag>.npy dumps.

#include <functional>
#include <vector>
#include "decode_abi.hpp"
#include "mtl4_context.hpp"

namespace pie::metal {

// Number of Kernel kinds (for PSO-by-kind table sizing).
inline constexpr int kKernelKindCount = static_cast<int>(Kernel::GdnPrepSlotted) + 1;

// One emitted dispatch in the per-token DAG.
struct Dispatch {
    Kernel      kind;        // pso lookup + charlie dump tag
    int         ordinal;     // flat 0..N-1 — the arg-table key (unique, token-stable)
    int         layer;       // model layer (-1 for singletons) — for <layer>.<tag> dumps
    Grid        grid;        // launch geometry (TODO(delta): confirm exact dims)
    Threadgroup tg;
    bool        fuse_residual = false;  // QmvO/QmvOut/QmvDown: add the block residual in the
                                        // GEMV epilogue (buffer 7) → drops the following
                                        // Residual/LayerOut dispatch. PIE_FUSE_RESIDUAL.
};

// PSOs compiled once from src/kernels/*.metal, indexed by Kernel kind.
struct DecodeStepPsos {
    Pso by_kind[kKernelKindCount]{};
    Pso qmv_residual{};      // affine_qmv_fast_residual — used for fuse_residual dispatches.
    Pso&       operator[](Kernel k)       { return by_kind[static_cast<int>(k)]; }
    const Pso& operator[](Kernel k) const { return by_kind[static_cast<int>(k)]; }
};

// Build the ordered per-token DAG (~393 raw dispatches; 363 are golden-tapped) from
// the geometry, with grid/tg filled per dispatch via delta's decode_dispatch.hpp
// helpers (GdnCore's {32,Vd,Vh}/{32,4,1} is beta's, in gdn_core.metal). with_argmax
// appends the optional device-argmax substrate (I3); logits are ALWAYS produced.
std::vector<Dispatch> build_decode_dag(const DecodeGeometry& g, bool with_argmax = false,
                                       bool fuse_residual = false, bool gdn_prep = false);

// ── GPU-exec attribution hook (optimization-phase prep; off by default) ───────
// When provided to encode_decode_step, the walker emits a timestamp mark at boundary i
// (BEFORE dispatch i) and one final mark after the last dispatch — so diffing the
// resolved timestamps attributes gpu-exec-ms per dispatch (see decode_timing.hpp). The
// `mark` callback is wired by the integration to alpha's StepEncoder timestamp seam;
// null on the production path => zero marks, zero perturbation.
struct StepTimingHook {
    std::function<void(int boundary_index)> mark;
};

// Encode one decode step: walk the DAG, bind pso + arg table (by ordinal), dispatch+barrier.
// `force_barriers` (diagnostic): emit a barrier after EVERY dispatch, disabling the ‖-pair
// concurrency. If the non-determinism vanishes with force_barriers=true, the cause is a
// ‖-pair concurrency/barrier issue; if it persists, the race is elsewhere (in-kernel/state).
// `timing` (optional): when non-null, emit per-boundary timestamp marks for attribution.
void encode_decode_step(StepEncoder& se,
                        const std::vector<Dispatch>& dag,
                        const DecodeStepPsos& psos,
                        bool force_barriers = false,
                        const StepTimingHook* timing = nullptr);

}  // namespace pie::metal
