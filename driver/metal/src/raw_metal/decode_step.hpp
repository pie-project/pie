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

#include <vector>
#include "decode_abi.hpp"
#include "mtl4_context.hpp"

namespace pie_metal_driver::raw_metal {

// Number of Kernel kinds (for PSO-by-kind table sizing).
inline constexpr int kKernelKindCount = static_cast<int>(Kernel::Argmax) + 1;

// One emitted dispatch in the per-token DAG.
struct Dispatch {
    Kernel      kind;        // pso lookup + charlie dump tag
    int         ordinal;     // flat 0..N-1 — the arg-table key (unique, token-stable)
    int         layer;       // model layer (-1 for singletons) — for <layer>.<tag> dumps
    Grid        grid;        // launch geometry (TODO(delta): confirm exact dims)
    Threadgroup tg;
};

// PSOs compiled once (from delta's raw_metal/kernels/*.metal), indexed by Kernel kind.
struct DecodeStepPsos {
    Pso by_kind[kKernelKindCount]{};
    Pso&       operator[](Kernel k)       { return by_kind[static_cast<int>(k)]; }
    const Pso& operator[](Kernel k) const { return by_kind[static_cast<int>(k)]; }
};

// Default launch dims for a kernel at qwen3.6 decode shapes (M=1). delta refines the
// exact grid/tg from the MLX instantiations; GdnCore is beta-validated.
//   out_n = the qmv output width (N) for the gemv kernels.
Grid        default_grid(Kernel k, const DecodeGeometry& g, int out_n);
Threadgroup default_tg(Kernel k, const DecodeGeometry& g);

// Build the ordered per-token DAG (~322 dispatches) from the geometry.
std::vector<Dispatch> build_decode_dag(const DecodeGeometry& g);

// Encode one decode step: walk the DAG, bind pso + arg table (by ordinal), dispatch+barrier.
void encode_decode_step(StepEncoder& se,
                        const std::vector<Dispatch>& dag,
                        const DecodeStepPsos& psos);

}  // namespace pie_metal_driver::raw_metal
