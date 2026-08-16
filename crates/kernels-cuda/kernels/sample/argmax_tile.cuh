// SPDX-License-Identifier: Apache-2.0
//
// A CuTile alternative for the vocabulary argmax that **loses 4x**, the
// mechanism for why, and the census row it corrects.
//
// # The census row this file was written to check, and breaks
//
// `tile_alternatives.cuh` priced the ARGMAX bucket -- 26 kernels -- off
// `topk_softmax_tile` at 1.28-1.40x. That representative is wrong, and the
// mismatch is the exact failure this spike kept catching in itself: a ratio
// measured at one shape quoted for another.
//
//   * `topk_softmax` reduces over EXPERTS: 128 of them, the whole row
//     resident in ONE tile, one reduction total.
//   * `argmax` reduces over the VOCABULARY: 151,936 for Qwen3, a 297 KB row,
//     149 tiles, 149 reductions.
//
// A 1,187x difference in reduction width. Nothing about the first number
// predicted the second, and measured, the second goes the other way.
//
// # The measurement
//
// L40S, bf16 logits, vocab 151,936, against the tree's `argmax<bf16>` at its
// own BLOCK=256. Best of 15 x 50. Both exact against a CPU gold at every
// shape, including tails of 1 element:
//
//     rows      bytes     incumbent      tile        ratio
//        1     0.3 MB      14.03 us    57.93 us     0.24x
//        4     1.2 MB      14.61 us    59.73 us     0.24x
//       16     4.6 MB      14.62 us    59.74 us     0.24x
//       64      19 MB      14.74 us    59.82 us     0.25x
//      128      37 MB      15.08 us    59.91 us     0.25x
//      256      74 MB      18.87 us    65.39 us     0.29x
//      512     148 MB     212.10 us   211.32 us     1.00x
//
// The last row is the roofline band again -- both forms hit DRAM at 148 MB
// and the tile disadvantage vanishes along with the tile advantage. Sixth
// independent kernel to land in 134-148 MB. It converges to a tie, not a
// win.
//
// # Why it loses, priced piece by piece
//
// Three cut-down variants, same sweep, rows=1 / rows=64:
//
//     variant                            rows=1      rows=64
//     incumbent                          14.03 us    14.74 us
//     max_unmasked  (no index, no mask)   26.52 us    27.62 us
//     max_only      (no index, masked)    48.16 us    50.14 us
//     argmax_tile   (this file)           57.93 us    59.82 us
//     always_two    (rescan every chunk)  61.43 us    63.37 us
//
// Read from the bottom up, three separate facts:
//
//   1. **The conditional rescan works.** 57.93 against 61.43 -- skipping the
//      index reduction when the chunk does not improve saves 3.5 us, so the
//      branch is real and not predicated away. The design was sound.
//   2. **`load_masked` costs 45% of the kernel.** 48.16 masked against 26.52
//      unmasked, for the same arithmetic. That is 21.6 us to handle a tail
//      of at most 1,023 elements out of 151,936. Worth knowing generally:
//      an even-length fast path is not a micro-optimisation at this size.
//   3. **Neither of those is the problem.** The floor -- no index, no mask,
//      nothing but `reduce_max` over the row -- is 26.52 us against the
//      incumbent's 14.03. **1.9x behind before any of this file's own work
//      begins.**
//
// So the loss is structural. Locating it took three sweeps and two wrong
// laws, both of which were pushed before the third sweep contradicted them.
// `tile_alternatives.cuh` carries the full 2-D surface; the short version:
//
//   WITHDRAWN #1: "a long strided reduction is not a CuTile shape" -- right
//   about width mattering, wrong that width was the whole story, and fitted
//   to a sweep where the grid moved with it.
//
//   WITHDRAWN #2: "width is free, blocks per SM is the only variable" --
//   fitted to a width sweep run only at 512 blocks, the one column where
//   the width effect is absent.
//
// Both variables are real and the surface does not separate. The clean,
// unconfounded read is the low-grid column, where every cell is L2-resident
// so only the width changes:
//
//     8 blocks, tiles    1     4    16    64   256
//     ratio           1.07  1.09  1.03  0.80  0.66x
//
// and the wide row, where the grid recovers what the width lost:
//
//     64 tiles, blocks   8    64   142   284   512  1024  2048
//     ratio           0.80  0.81  0.87  0.98  1.03  1.12  1.17x
//
// Argmax at decode sits in the corner where both are against it: 149 tiles
// (past the 16-to-64-tile boundary) at 1 to 128 blocks (the left edge). The
// surface predicts 0.66-0.75x there.
//
// The rest of the gap is the second, independent penalty this kernel pays
// and a sum does not. A running sum lives in a tile across chunks; a running
// MAX cannot, because the index has to be recovered against it, so every
// chunk must collapse to a scalar. Measured at matched grid and width:
// 0.42x against 0.79x. 0.70 x 0.53 is 0.37, and the tail mask takes the rest
// of the way to 0.24x.
//
// The point of recording all three attempts: each wrong law was a single
// clean line fitted to a single slice, and each survived a review that the
// next slice failed. A ratio measured at one shape is a fact about that
// shape -- the lesson this file exists to hold, arrived at the hard way
// twice more while writing it.
//
// # Verdict: do not use this
//
// `argmax_tile_preferred` is `false` and there is no crossover to open it
// at. The 512-row tie is a tie, and a second code path that converges to
// parity at one batch size while being 4x slower at the seven below it is
// not worth having.
//
// Kept, not deleted, for the same reason `layout/gather_rows_tile.cuh` and
// `quant/wna16_gemv_tile.cuh` are kept: the next person to look at the
// ARGMAX bucket should find the measurement rather than repeat it, and the
// piece-by-piece table above says exactly which part is hopeless.
//
// # One dialect finding, free
//
// The three probes above were first written as
//
//     [[ using cutile : hint(1000, latency=1) ]]
//     auto raw = ct::load_masked(...);
//
// and nvcc says: `warning #20364-D: tile optimization hints are ignored when
// attached to variable declarations`. The hint silently does nothing in that
// form. Every load in this tree's tile kernels declares the tile first and
// assigns on the next line, which is why they get the hint -- that two-line
// shape is load-bearing and now has a compiler diagnostic behind it.
#pragma once

#include <cuda_tile.h>

#include <cuda_bf16.h>

namespace pie::sample {

namespace ct = ::cuda::tiles;

/// The tile shape. Powers of two only -- `ct::shape<2560>` is `concept is
/// false`, which is a compile error and not a slow path.
inline constexpr int ARGMAX_BS = 1024;

/// What the conditional rescan is worth, measured: 57.93 us with the branch
/// against 61.43 us running both reductions every chunk, at rows=1. The
/// branch is real -- 5.7% -- and it is nowhere near enough to matter.
inline constexpr float ARGMAX_RESCAN_SAVING_US = 3.50f;

/// The vocabulary argmax, one block per row.
///
/// `VOCAB` is a runtime value here and not a template parameter, unlike the
/// static extents this spike otherwise insists on: the tile shape is
/// `ARGMAX_BS` and static, and only the trip count is dynamic. That is the
/// cheap half. The 7-45% dynamic-extent penalty is on the TILE, and this
/// kernel does not pay it.
template <class T>
__tile_global__ void argmax_tile(
    const T* __restrict__ logits,
    int* __restrict__ out,
    int vocab)
{
    using f32xBS = ct::tile<float, ct::shape<ARGMAX_BS>>;
    using i32xBS = ct::tile<int, ct::shape<ARGMAX_BS>>;

    const int row = static_cast<int>(ct::bid().x);
    auto row_ptr = ct::assume_aligned<16>(logits + (long long)row * vocab);

    const int num_blocks = (vocab + ARGMAX_BS - 1) / ARGMAX_BS;
    const bool even = (vocab % ARGMAX_BS) == 0;

    float best_val = -3.402823466e+38f;
    int best_idx = 0;

    for (auto j_idx : ct::irange(0, num_blocks)) {
        auto cols = ct::iota<i32xBS>() + j_idx * ARGMAX_BS;

        // The tail is masked with -inf, not zero. A zero pad would make a row
        // of all-negative logits report a padding lane as its winner, and the
        // index would be past the vocabulary -- a token id that does not
        // exist. `EVEN` picks the path at compile time.
        using TxBS = ct::tile<T, ct::shape<ARGMAX_BS>>;
        auto neg_pad = ct::full<TxBS>(T(-3.0e+38f));

        TxBS raw;
        if (even) {
            [[ using cutile : hint(1000, latency=1) ]]
            raw = ct::load(row_ptr + cols);
        } else {
            [[ using cutile : hint(1000, latency=1) ]]
            raw = ct::load_masked(row_ptr + cols, cols < vocab, neg_pad);
        }
        auto v = ct::element_cast<float>(raw);

        const float m = static_cast<float>(ct::reduce_max<0>(v));

        // Strictly greater, matching the incumbent's `update_argmax`: on a
        // tie the earlier chunk keeps the answer.
        if (m > best_val) {
            best_val = m;
            // Only here, and only on a tile already in registers.
            best_idx = static_cast<int>(ct::reduce_min<0>(
                ct::select(v == ct::full<f32xBS>(m),
                           cols,
                           ct::full<i32xBS>(0x7fffffff))));
        }
    }

    // A tile kernel has no threadIdx to gate on. The whole block agrees on
    // `best_idx` -- it came out of a reduction -- so a 1-wide store of the
    // scalar is the block-uniform write, and there is exactly one of them.
    using i32x1 = ct::tile<int, ct::shape<1>>;
    ct::store(out + row + ct::iota<i32x1>(), ct::full<i32x1>(best_idx));
}

/// Declined. There is no crossover to open it at.
///
/// The sweep in this header is 0.24-0.29x from 1 row to 256, and 1.00x at
/// 512 where both forms are DRAM-bound. A tie at one batch size is not a
/// reason for a second code path, and the piece-by-piece table shows the
/// floor -- no index, no mask -- is still 1.9x behind.
///
/// Same shape as the tree's existing `moe_grouped_gemm_bf16_supported` and
/// `rmsnorm_vec8_ok`, and the same shape as the two other declined
/// alternatives, so the call site does not care which way a predicate went.
constexpr bool argmax_tile_preferred(long long /*rows*/, long long /*vocab*/) {
    return false;
}

/// The best ratio this kernel ever reached below the DRAM roof, x100.
/// Recorded so the refusal above cannot drift away from its reason.
inline constexpr int ARGMAX_TILE_BEST_RATIO_PCT = 29;

static_assert(!argmax_tile_preferred(1, 151936),
              "declined at decode: 0.24x");
static_assert(!argmax_tile_preferred(512, 151936),
              "declined at the 512-row tie too: 1.00x is not a win, and the "
              "seven rows below it are 0.24-0.29x");
static_assert(ARGMAX_TILE_BEST_RATIO_PCT < 100,
              "if this kernel ever measures ahead, the predicate above is "
              "the thing to revisit -- not this assert");
static_assert(ARGMAX_BS == 1024 && (ARGMAX_BS & (ARGMAX_BS - 1)) == 0,
              "tile extents must be powers of two; ct::shape<2560> is "
              "concept is false");

}  // namespace pie::sample
