//===-- tile_alternatives.cuh - the tile kernels are ADDITIONS ------------===//
//
// Six CuTile kernels sit beside hand-written ones in this tree. **None of
// them replaces anything.** Each is an alternative with a `*_tile_preferred`
// predicate saying when to fire it, and each incumbent remains the fallback
// — necessarily, because the alternatives need NVRTC 13.3, 13.3 runtime
// headers and `tileiras`, and this crate loads NVRTC 13.0.88. An alternative
// that cannot be selected on the machine in front of you is not an
// alternative, it is a removal.
//
//     alternative                 incumbent                preferred when
//     ─────────────────────────────────────────────────────────────────────
//     moe_grouped_gemm_tile       moe_grouped_gemm         shape divides
//     rmsnorm_tile                rmsnorm                  always
//     rmsnorm_rasr_tile           rmsnorm_residual_add_    always
//                                 scale_rmsnorm
//     topk_softmax_tile           topk_softmax_warp_x1     rows <= 1024
//     swiglu_tile                 swiglu                   n <= 16 Mi
//     moe_fused_tile              (none — a negative)      never
//
// # This file exists to make the predicates checkable without a GPU
//
// A `constexpr bool` is a claim about a measurement, and a claim decays. The
// `static_assert`s below pin each predicate to the specific rows of the
// specific sweeps that produced it, so a bound that gets rounded, widened or
// "cleaned up" fails the compile rather than quietly firing a slower kernel.
//
// That is not hypothetical. `swiglu_tile_preferred` was first written as
// `6 * n <= 100 MB`, which reads like the measurement — and excluded the
// very point the measurement was taken at, 16 Mi elements being 100.7 MB.
// The assertion below caught it before it was committed. It is now stated in
// ELEMENTS, which is the unit the sweep actually swept.
//
// # Why not one header per predicate
//
// Each predicate lives with its kernel, where its table of numbers is. This
// file only collects the ASSERTIONS, so that "do the bounds still match the
// sweeps" is one translation unit and one answer rather than five.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlp/swiglu_tile.cuh"
#include "moe/moe_grouped_gemm_tile.cuh"
#include "moe/topk_softmax_tile.cuh"
#include "norm/rmsnorm_rasr_tile.cuh"
#include "norm/rmsnorm_tile.cuh"

namespace pie_cuda_driver::kernels {

namespace {

namespace n_ = ::pie_cuda_driver::kernels::norm::device;
namespace m_ = ::pie_cuda_driver::kernels::mlp::device;
namespace o_ = ::pie_cuda_driver::kernels::moe::device;

// rmsnorm: 1.94 vs 2.93 us at one row, 1590 vs 1612 us at 65,536. Never
// behind, so the predicate has no crossover to get wrong -- but it still has
// to keep answering true, because a later "optimisation" narrowing it would
// silently give the slower kernel back.
static_assert(n_::rmsnorm_tile_preferred(1, 4096), "measured 1.94 vs 2.93 us");
static_assert(n_::rmsnorm_tile_preferred(65536, 4096), "measured 1590 vs 1612 us");

// The fused norm pair: 2.41 vs 4.33 us at one row, 12.86 vs 24.71 at 2,048.
// Its incumbent is the most expensive kernel the norm family measures -- its
// own header records 10.79 us/call, 8% of a gemma-4-26B decode step -- so a
// bound that narrowed here would give back the largest single win in this
// set.
static_assert(n_::rmsnorm_rasr_tile_preferred(1, 2816), "measured 2.41 vs 4.33 us");
static_assert(n_::rmsnorm_rasr_tile_preferred(2048, 2816), "measured 12.86 vs 24.71 us");

// swiglu: 0.038 vs 0.057 ms at 16 Mi (100.7 MB touched); 0.303 vs 0.290 at
// 32 Mi (201 MB). The bound is the measured point, not an interpolation.
static_assert(m_::swiglu_tile_preferred(16LL << 20), "measured 0.038 vs 0.057 ms");
static_assert(!m_::swiglu_tile_preferred(32LL << 20), "measured 0.303 vs 0.290 ms");

// top-K: 3.06 vs 3.90 us at one row, 4.84 vs 4.85 at 1,024 -- the crossing
// itself, which the predicate includes because a tie costs nothing and the
// next point measured, 2,048, is 7.22 vs 6.08.
static_assert(o_::topk_softmax_tile_preferred(1), "measured 3.06 vs 3.90 us");
static_assert(o_::topk_softmax_tile_preferred(1024), "measured 4.84 vs 4.85 us");
static_assert(!o_::topk_softmax_tile_preferred(2048), "measured 7.22 vs 6.08 us");
// The same bound against the BLOCK form at 256 experts, which is the arm
// `families/moe.rs` records at 7.56 us/call and 4.9% of a Qwen3.6-35B-A3B
// step: 4.52 vs 6.23 us at one row, 12.32 vs 10.82 at 2,048.
static_assert(o_::topk_softmax_tile_preferred(128), "measured 4.65 vs 6.52 us at 256 experts");

// The grouped GEMM's predicate is a divisibility claim, not a crossover: it
// is ahead at both shapes pie fires and the conditions are the ones its own
// `static_assert`s state.
static_assert(o_::moe_grouped_gemm_tile_preferred(512, 2048), "gate_up divides");
static_assert(!o_::moe_grouped_gemm_tile_preferred(512, 2049), "K must divide");

}  // namespace

}  // namespace pie_cuda_driver::kernels
