#pragma once
// ── gemma4 executor: PSO table + per-step encode_fn (alpha) ──────────────────
// The gemma4 analog of beta's decode_psos + decode_step encode. Turns the pure
// `build_gemma4_dag` dispatch list (gemma4_decode_step.hpp) into a runnable
// `encode_fn` for RawMetalContext::run_step:
//
//   1. load_gemma4_psos — runtime-compile every gemma4 kernel from raw_metal/
//      kernels/*.metal and fan each PSO out to the Kernel kinds it serves. The
//      4-bit (q_bits==4) path reuses delta's proven affine_qmv for all linears +
//      4-bit embed_gather + tied lm_head (charlie's ~/models/4bit-tied gemma4);
//      the gemma-specific pointwise kernels (geglu-tanh, v_norm, ple_combine,
//      logit_softcap, layer_scalar) + sliding-window SDPA are alpha's.
//   2. encode_gemma4_step — walk the DAG, bind pso + arg table (by flat ordinal,
//      the ratified key) + launch dims per kind, dispatch + barrier. Mirrors
//      encode_decode_step exactly (proven correct at qwen3.6 argmax-264).
//
// Launch geometry reuses delta's decode_dispatch.hpp helpers for the shared kinds
// (qmv/rms/rope/sdpa/kv_append/residual/embed) + gemma-local helpers for the
// pointwise/PLE kinds. Per-model constants (rms plus_one=false, sdpa window,
// rope theta, softcap) are bound at the consts stage, NOT here — the PSOs and the
// launch dims are quant/flag-independent.

#include <string>
#include <vector>

#include "gemma4_decode_step.hpp"  // build_gemma4_dag, Gemma4Dispatch (pure)
#include "mtl4_context.hpp"

namespace pie_metal_driver::raw_metal::gemma4 {

// Number of gemma4 Kernel kinds (for the PSO-by-kind table).
inline constexpr int kGemma4KernelKindCount = static_cast<int>(Kernel::Argmax) + 1;

// PSOs compiled once (from raw_metal/kernels/*.metal), indexed by gemma4 Kernel kind.
struct Gemma4StepPsos {
    Pso by_kind[kGemma4KernelKindCount]{};
    Pso&       operator[](Kernel k)       { return by_kind[static_cast<int>(k)]; }
    const Pso& operator[](Kernel k) const { return by_kind[static_cast<int>(k)]; }
};

// Compile every gemma4 decode kernel from `kernels_dir` into `out`, indexed by kind.
// Only the 4-bit path (geom.q_bits==4) is wired (the shipped gemma4; bf16 fallback was
// retired on accuracy grounds). `with_argmax` controls the optional device-argmax PSO
// (not yet ported → left invalid). Returns false on the first compile failure (the
// failing entrypoint/file is written to `*err`).
bool load_gemma4_psos(RawMetalContext& ctx,
                      const std::string& kernels_dir,
                      const Gemma4Geometry& geom,
                      Gemma4StepPsos& out,
                      bool with_argmax = false,
                      std::string* err = nullptr);

// Resolve the launch geometry (total-threads Grid + Threadgroup) for one dispatch.
// `layer` selects the per-layer-type dims (head_dim 256/512, double-wide MLP); pass the
// dispatch's layer (-1 for layer-less kinds, which ignore it).
void gemma4_launch_dims(Kernel kind, int layer, const Gemma4Geometry& g,
                        Grid& grid, Threadgroup& tg);

// Encode one gemma4 decode step: walk the DAG, bind pso + arg table (by ordinal) +
// launch dims, dispatch + barrier. `force_barriers` emits a barrier after every
// dispatch (diagnostic); `vis` is the barrier cache-visibility (ExecutionOnly is the
// proven-correct + free default established on the qwen3.6 lane).
void encode_gemma4_step(StepEncoder& se,
                        const std::vector<Gemma4Dispatch>& dag,
                        const Gemma4StepPsos& psos,
                        const Gemma4Geometry& geom,
                        bool force_barriers = false,
                        BarrierVisibility vis = BarrierVisibility::ExecutionOnly);

}  // namespace pie_metal_driver::raw_metal::gemma4
