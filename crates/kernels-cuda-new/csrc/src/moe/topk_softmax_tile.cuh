//===-- topk_softmax_tile.cuh - the router's top-K, in CuTile ------------===//
//
// An ALTERNATIVE to `topk_softmax.cuh`'s warp kernel, not a replacement for
// it. Both are carried; `topk_softmax_tile_preferred` below says which to
// fire, and it crosses at about a thousand rows. The CuTile twin of
// `topk_softmax.cuh`'s WARP kernel -- not its block form.
// That matters: the block form is the naive opponent and the warp form is
// the one that file already optimised into, with `__shfl_xor` reductions and
// no barriers, after measuring the block form at 4.39 us against a 0.54 us
// floor and 105 us of a 2.4 ms decode step.
//
//     rows    CuTile     topk_softmax_warp_x1     floor
//        1    3.05 us          3.90 us            0.65 us
//      128    3.18             4.00
//    2,048    7.21             6.07
//
// L40S sm_89, 32 experts, top-8, bf16 logits. **Expert indices are
// IDENTICAL** at every row count -- 0 mismatches out of 16,384 -- and the
// weights agree to 3e-8. That is the correctness bar this kernel has to
// clear and not a nicety: a different expert is a different model.
//
// # And against the BLOCK form, which is the case with a documented cost
//
// The warp rungs only serve experts that fit a warp's registers.
// `families/moe.rs` records what happens above that: *"Qwen3.6-35B-A3B
// routes through more than 128 experts at 7.56 us/call, 4.9% of its step"*
// — and that is the BLOCK form, with its three shared-memory reduction
// trees and ~36 barriers. A tile does not care how wide the router is.
//
//     rows      tile      topk_softmax (block form)
//        1      4.52 us          6.23 us      1.38x
//      128      4.65             6.52         1.40x
//    1,024      7.49             8.00         1.07x
//    2,048     12.32            10.82         0.88x
//
// 256 experts, top-8. Indices identical at every row count — 0 of 16,384 —
// and weights to 4.5e-8. The crossover lands in the same place as the
// 32-expert sweep, which is why one predicate covers both.
//
// So it wins 1.28x at decode and loses 1.19x at 2,048 rows, which is the
// same boundary `mlp/swiglu_tile.cuh` draws from the other direction: the
// tile advantage is a LATENCY and OCCUPANCY advantage, so it appears where
// there is not enough work to saturate the machine and disappears where
// there is. Decode is the first case.
//
// # Three traps, all of which produced plausible wrong answers
//
// **The weights are renormalised by the WINNERS' own sum**, not by the sum
// over all experts. `topk_softmax.cuh` explains why the two are the same
// number when you softmax first and renormalise after -- the partition
// function divides out -- and a version of this kernel that divided by the
// full sum computed a perfectly reasonable softmax that differed by up to
// 0.108 absolute. The expert INDICES were identical in both, so a test that
// only checked routing would have passed it.
//
// **A local array costs 6.7x.** Buffering the K winners in `int gi[TOPK]` /
// `float gv[TOPK]` and writing them after the loop measures **20.38 us**
// against 3.05 for writing each one as it is found -- with a compile-time
// `TOPK` and `#pragma unroll` on both loops, which do not help. A
// `__tile_global__` has no per-thread scratch in the usual sense; an indexed
// local array goes to local memory and every access is a round trip.
//
// **Scalar math is not free but it is not the problem either.** `__expf` is
// a `__device__` function and calling one from a `__tile_global__` is a hard
// compile error. `ct::exp` on a plain `float` compiles and, measured, costs
// nothing -- a variant doing K scalar `ct::exp` calls runs 3.06 us against
// this file's 3.05. That was the second hypothesis for the 20 us and it was
// wrong; the A/B is recorded because the wrong explanation is the one that
// would have been published.
//
// # Why `NE` and `TOPK` are compile-time
//
// `NE` is the tile width, so it has to be; it is rounded up to a power of
// two and the lanes past `num_experts` are masked out of the load, forced to
// -inf before any argmax, and zeroed in the sum. `TOPK` bounds the
// unrolled selection loop. Both are model constants and a JIT instantiates
// per shape -- `moe/moe_grouped_gemm_tile.cuh` has the whole of that
// argument, and the toolchain floor that still keeps this out of `UNITS`.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie_cuda_driver::kernels::moe::device {

namespace ct = ::cuda::tiles;

/// Whether this ALTERNATIVE should be preferred to
/// `topk_softmax.cuh`'s warp kernel.
///
/// Below about a thousand rows, which is where the machine stops being
/// latency-bound and the warp form's saturation starts to win:
///
///     rows      tile       topk_softmax_warp_x1
///        1      3.06 us          3.90 us
///       64      3.13             4.00
///      128      3.19             4.01
///      256      3.35             4.18
///      512      4.12             4.46
///    1,024      4.84             4.85     <- the crossing
///    2,048      7.22             6.08
///
/// 32 experts against `topk_softmax_warp_x1`; the 256-expert sweep against
/// the block form crosses in the same place (7.49 vs 8.00 us at 1,024 and
/// 12.32 vs 10.82 at 2,048), which is why one bound serves both.
///
/// Decode is one row per request, so
/// the decode path is comfortably inside this and the prefill path is
/// comfortably outside it -- which is why this is a predicate and not a
/// preference.
constexpr bool topk_softmax_tile_preferred(int rows)
{
    return rows <= 1024;
}

#ifndef NE
#define NE 256          // experts, a power of two >= the real count
#endif
#ifndef TOPK
#define TOPK 8
#endif
using f32xE = ct::tile<float, ct::shape<NE>>;
using i32xE = ct::tile<int, ct::shape<NE>>;
using bf16xE = ct::tile<__nv_bfloat16, ct::shape<NE>>;

/// Softmax over `num_experts` logits, then the K largest, ties to the LOWEST
/// index -- a correctness requirement, not a nicety: a different expert is a
/// different model.
__tile_global__ void topk_softmax_tile(
    const __nv_bfloat16* __restrict__ _logits,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K)
{
    auto logits = ct::assume_aligned<16>(_logits);
    const int row = static_cast<int>(ct::bid().x);
    auto cols = ct::iota<i32xE>();
    auto live = cols < num_experts;

    bf16xE raw;
    [[ using cutile : hint(1000, latency=1) ]]
    raw = ct::load_masked(logits + (long long)row * num_experts + cols, live,
                          ct::zeros<bf16xE>());
    auto x = ct::element_cast<float>(raw);
    // Dead lanes must never win an argmax.
    auto neg_inf = ct::full<f32xE>(-3.0e38f);
    x = ct::select(live, x, neg_inf);

    // The tree's warp kernel selects on the RAW logits and renormalises the K
    // winners by their OWN sum, which is the same number as softmaxing over
    // all experts and then renormalising -- the partition function divides
    // out. Matching that exactly matters: `topk_w` feeds the expert combine,
    // and a different weight is a different model. Dividing by the sum over
    // ALL experts instead, which the first version of this kernel did, is a
    // silently plausible answer that differs by up to 0.108 here.
    float m = static_cast<float>(ct::reduce_max<0>(x));

    // exp ONCE, in the tile domain. Scalar math is not available inside a
    // `__tile_global__` -- `__expf` is a `__device__` function and calling
    // one is a hard error -- and `ct::exp` on a scalar was measured at 20.7 us
    // against 2.6 for the tile form, so this is not a style preference.
    // exp is monotonic, so selecting on `ex` picks the same winners as
    // selecting on the raw logits.
    auto ex = ct::select(live, ct::exp(x - ct::full<f32xE>(m)), ct::zeros<f32xE>());

    // K rounds of argmax-with-exclusion. `reduce_max` gives the value; the
    // index is the smallest lane holding it, which `reduce_min` over a masked
    // iota gives directly -- so ties resolve LOW without a sort, which is the
    // tree's stated correctness requirement.
    //
    // The weights are renormalised by the WINNERS' own sum, not by the sum
    // over all experts: the tree's kernel does that and the partition
    // function divides out. Getting it wrong is silently plausible and was
    // measured at 0.108 absolute error, which is a different model.
    float denom = 0.f;
    auto avail = live;
    auto no = ct::full<ct::tile<bool, ct::shape<NE>>>(false);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        auto cand = ct::select(avail, ex, ct::zeros<f32xE>() - ct::full<f32xE>(1.f));
        float best = static_cast<float>(ct::reduce_max<0>(cand));
        auto at_best = ct::select(avail, cand == ct::full<f32xE>(best), no);
        int i = static_cast<int>(
            ct::reduce_min<0>(ct::select(at_best, cols, ct::full<i32xE>(NE))));
        topk_idx[(long long)row * TOPK + k] = i;
        topk_w[(long long)row * TOPK + k] = best;
        denom += best;
        avail = ct::select(cols == ct::full<i32xE>(i), no, avail);
    }
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        topk_w[(long long)row * TOPK + k] /= denom;
    }
}

}  // namespace pie_cuda_driver::kernels::moe::device
