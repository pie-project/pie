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
#include "model/shared_kernels.hpp"

namespace pie::metal {

// Number of Kernel kinds (for PSO-by-kind table sizing).
// Every kind, including the families appended after qwen3.5's. This sizes the
// PSO table and the timing arrays, so pinning it to one family's last kind is
// an out-of-bounds read the moment another family appends — which is exactly
// what happened, as a segfault with nothing pointing at the cause. Anchored on
// the enum's actual last member instead.
inline constexpr int kKernelKindCount = static_cast<int>(Kernel::KindCount);

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
    int         qmm_bn = 0;   // output columns per threadgroup when this
                              // projection runs as the steel GEMM
                              // (affine_qmm_t); 0 = use the GEMV.
    int         qmm_split = 1;  // K partitions; >1 adds a reduce dispatch
    int         qmm_bm = 16;  // rows per threadgroup for that GEMM: a wider
                              // block dequantizes each weight tile once for
                              // twice the rows, which only pays once the batch
                              // has threadgroups to spare.
};

// PSOs compiled once from src/kernels/*.metal, indexed by Kernel kind.
struct DecodeStepPsos {
    Pso by_kind[kKernelKindCount]{};
    Pso qmv_residual{};      // affine_qmv_fast_residual — used for fuse_residual dispatches.
    Pso&       operator[](Kernel k)       { return by_kind[static_cast<int>(k)]; }
    const Pso& operator[](Kernel k) const { return by_kind[static_cast<int>(k)]; }
};

/// Whether this kind's weights are in the geometry's SECOND affine format.
///
/// Exactly the two routing projections -- `mlp.gate` and
/// `mlp.shared_expert_gate` -- which mlx_lm's quantization predicate
/// spares at 8 bits inside Qwen3.6-35B-A3B's 4-bit body. Asked in one
/// place so the pipeline choice and the launch shape cannot disagree
/// about which table a dispatch reads: the alternate table has only the
/// MATVEC, so a kind that reads it must not be given a GEMM.
///
/// It lives here rather than on `DecodeGeometry` because that struct is a
/// leaf `decode_abi.hpp` includes before `Kernel` exists.
inline bool qwen35_uses_alt_quant(Kernel k, const DecodeGeometry& g) {
    return g.has_alt_quant() &&
           (k == Kernel::LlRouter || k == Kernel::LlSharedGateProj);
}

/// Rows the mixture's sort produces for a batch of `n_tokens`.
///
/// One per (token, slot) pair, plus whatever padding the tile costs -- which
/// is none at M=1, where the sort is a pure grouping, and up to `tile - 1` per
/// touched expert once the batch is big enough to make the projections
/// matmuls. It is the extent of every routed projection, of the SiluMul
/// between them, of the gather that fills them and of the pool slot that holds
/// them, so it is asked in ONE place: five call sites each deriving it would
/// be five chances to disagree, and a disagreement here is a read off the end
/// of the routing rather than a wrong answer.
inline int moe_sorted_rows(const DecodeGeometry& g, int n_tokens = 1) {
    const int n = n_tokens > 0 ? n_tokens : 1;
    return shared_kernels::moe_sorted_rows(n * g.experts_per_token, g.n_experts);
}

// Build the ordered per-token DAG (~393 raw dispatches; 363 are golden-tapped) from
// the geometry, with grid/tg filled per dispatch via delta's decode_dispatch.hpp
// helpers (GdnCore's {32,Vd,Vh}/{32,4,1} is beta's, in gdn_core.metal). with_argmax
// appends the optional device-argmax substrate (I3); logits are ALWAYS produced.
std::vector<Dispatch> build_decode_dag(const DecodeGeometry& g, bool with_argmax = false,
                                       bool fuse_residual = false, bool gdn_prep = false);

// ── Concurrent runs ──────────────────────────────────────────────────────────
// A maximal run of consecutive same-layer dispatches that may execute at once:
// they read an activation their predecessors already produced and write disjoint
// scratch, so no barrier is needed between them. Returns, per dispatch, the index
// of the last dispatch of its run (== its own index when it runs alone).
//
// One derivation, three readers: the two encoders decide barriers from it, and
// the scratch allocator extends a value's live range to the end of the run its
// last use falls in, so a buffer is never rewritten by a dispatch still running
// alongside its reader. Keeping those in one place is what makes widening a run
// safe -- they cannot disagree.
std::vector<int> concurrent_run_ends(const std::vector<Dispatch>& dag);

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
