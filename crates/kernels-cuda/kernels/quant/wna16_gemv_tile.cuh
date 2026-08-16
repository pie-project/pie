//===-- wna16_gemv_tile.cuh - the W4A16 GEMV, and it LOSES ---------------===//
//
// **This kernel is slower than the incumbent and is carried to say so with
// numbers.** It is the measured representative of the GEMV bucket -- the
// last row of `kernels/tile/alternatives.cuh`'s census that had no
// measurement behind it.
//
//     hidden    weights   tile        warp form   ratio
//        512      1.0 MB   4.02 us     3.91 us    0.97x
//      2,048      4.2      9.29        5.51       0.59x
//      8,192     16.8     29.50       14.46       0.49x
//     32,768     67.1    102.70       46.97       0.46x
//
// L40S sm_89, in_dim 4096, group 128. The opponent is the shape
// `dequant_wna16.cuh`'s `wna16_down_decode` uses: one warp per output row,
// `float4` activation loads, PRMT-style unpack, `__half2` accumulate, a
// `__shfl_xor` reduction.
//
// # FOUR formulations, all losing, and three refuted explanations
//
// This kernel is the best of four, which is worth knowing before writing a
// fifth. At hidden 8,192 against the warp form's 14.4 us:
//
//     one lane per OUTPUT element (this file)          29.4 us   0.49x
//     one lane per WORD, 8 nibbles unrolled            40.6      0.35x
//     Marlin shape: ROWS output rows per block, 2-D    37.3-74   0.19-0.39x
//     ROWS rows + weights via `partition_view`         37.3-61   0.24-0.39x
//
// Each was written to test a specific hypothesis and each hypothesis was
// refused by its own measurement:
//
//   * **read amplification.** One lane per output element fetches the word
//     holding eight nibbles eight times. Fixing that made it worse.
//   * **activation reuse.** Marlin, AWQ and TensorRT-LLM all put 16-32
//     output rows in a block so the activation vector is read once and
//     shared -- confirmed by reading their kernel descriptions, and this
//     kernel reads it per row. Doing it their way made it worse.
//   * **gather versus strided load.** `ct::load(ptr + index_tile)` is a
//     per-lane gather; `partition_view::load` is a strided load the compiler
//     can widen, and weight traffic dominates a W4A16 GEMV. Routing the
//     weights through a `partition_view` did not help either.
//
// Three plausible stories, three refusals. The measurement is the result and
// this header offers no mechanism -- the same posture a scalar `ct::exp`
// hypothesis earned earlier in this spike.
//
// # What the gap IS, which is L2
//
// The one thing that did explain something was the size sweep:
//
//     hidden     weights    tile      warp      ratio    warp GB/s
//      8,192       17 MB    29.42     14.42     0.49x      1,179
//     32,768       67       107.41    46.95     0.44x      1,427
//     65,536      134       227.03   193.50     0.85x        693
//    131,072      268       454.34   385.13     0.85x        696
//
// L40S L2 is 48 MB and HBM peak is ~864 GB/s. At 17 and 67 MB the warp form
// runs at 1.2-1.4 TB/s, which is not HBM -- it is L2 -- and it beats this
// kernel 2x. Past 134 MB it drops to ~693 GB/s, 80% of HBM peak, and the gap
// collapses to **0.85x**.
//
// So the warp form's advantage is that it keeps the weight stream in L2
// better, and **at the memory roofline the two are within 15%** -- the same
// place every other comparison in this spike ends up. The production case
// for this kernel is small (an MoE expert's down projection is a few hundred
// KB), which is exactly the L2-resident regime where the gap is widest.
//
// # It is more accurate, which is the incumbent's trade and not a defect
//
// The differing outputs -- 3,209 of 32,768, worst relative 3.46 -- are the
// WARP form's error, not this one's. It accumulates 4,096 products in
// `__half2`; this accumulates in fp32. Cancellation over that many terms in
// half precision is exactly where a relative error of 3 comes from. That is
// a deliberate trade in the incumbent and this kernel does not make it, so
// the two are not bit-comparable and the timing is the only comparison.
//
// # What it settles
//
// The census now has **no inferred rows**. Every one of its verdicts is
// backed by a kernel that was written and raced, including the two that say
// "do not bother": this one and `layout/gather_rows_tile.cuh`.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <crt/cuda_tile.h>

namespace pie::quant {

namespace ct = ::cuda::tiles;

/// Whether to prefer this ALTERNATIVE to the warp-per-row W4A16 GEMV.
///
/// No, at every size measured -- 0.97x at hidden 512 falling to 0.46x at
/// 32,768. Stated as a predicate rather than left out so the answer is on
/// the file rather than in whoever last read the timing table.
constexpr bool wna16_gemv_tile_preferred(int /*hidden*/, int /*inter*/)
{
    return false;
}

#ifndef BS
#define BS 1024
#endif
#ifndef INTER
#define INTER 4096
#endif
#ifndef GRP
#define GRP 128
#endif
using i32xBS = ct::tile<int, ct::shape<BS>>;
using f32xBS = ct::tile<float, ct::shape<BS>>;
using hxBS   = ct::tile<__half, ct::shape<BS>>;
using bfxBS  = ct::tile<__nv_bfloat16, ct::shape<BS>>;

/// One block per output row: the whole dot product is a tile reduction, so
/// there is no warp shuffle and no per-thread accumulator to reassociate.
/// The incumbent accumulates in `__half2` and this accumulates in fp32,
/// which is the one intended difference -- see the header.
__tile_global__ void wna16_gemv_tile(
    const __half* __restrict__ _act,
    const int* __restrict__ _packed,
    const __nv_bfloat16* __restrict__ _scale,
    __nv_bfloat16* __restrict__ out,
    int hidden, int /*intermediate*/, int /*group_size*/)
{
    const int h = static_cast<int>(ct::bid().x);
    if (h >= hidden) return;
    constexpr int words_per_row = INTER / 8;
    constexpr int groups_per_row = INTER / GRP;
    auto act    = ct::assume_aligned<16>(_act);
    auto packed = ct::assume_aligned<16>(_packed + (long long)h * words_per_row);
    auto scale  = ct::assume_aligned<16>(_scale + (long long)h * groups_per_row);

    constexpr int nb = (INTER + BS - 1) / BS;
    auto acc = ct::zeros<f32xBS>();
    for (auto j : ct::irange(0, nb)) {
        auto k = ct::iota<i32xBS>() + j * BS;
        auto widx = k / ct::full<i32xBS>(8);
        auto lane = k - widx * ct::full<i32xBS>(8);
        i32xBS w;
        [[ using cutile : hint(1000, latency=1) ]]
        w = ct::load(packed + widx);
        auto nib = (w >> (lane * ct::full<i32xBS>(4))) & ct::full<i32xBS>(0xF);
        auto q = ct::element_cast<float>(nib - ct::full<i32xBS>(8));
        bfxBS s;
        [[ using cutile : hint(1000, latency=1) ]]
        s = ct::load(scale + k / ct::full<i32xBS>(GRP));
        hxBS x;
        [[ using cutile : hint(1000, latency=1) ]]
        x = ct::load(act + k);
        acc = acc + q * ct::element_cast<float>(s) * ct::element_cast<float>(x);
    }
    float dot = static_cast<float>(ct::sum<0>(acc));
    out[h] = static_cast<__nv_bfloat16>(
        ct::reduce_max<0>(ct::element_cast<__nv_bfloat16>(
            ct::full<ct::tile<float, ct::shape<1>>>(dot))));
}

}  // namespace pie::quant
