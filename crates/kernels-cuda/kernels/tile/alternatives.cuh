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
//     rope_partial_tile           rotate_partial           never -- see below
//     rmsnorm_rasr_tile           rmsnorm_residual_add_    always
//                                 scale_rmsnorm
//     topk_softmax_tile           topk_softmax_warp_x1     rows <= 1024
//     swiglu_tile                 swiglu                   n <= 16 Mi
//     moe_fused_tile              two grouped GEMMs        blocks >= 212
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
// # Every one of them compiles on every architecture, and none of them spills
//
// All the timings in this tree are from one L40S -- sm_89, Ada. Speed on
// Hopper or Blackwell cannot be checked from here. Compilation and resource
// usage can, and registers and shared memory are the leading indicator of
// whether a tiling survives a part change. Built for every arch `tileiras`
// accepts, at the Ada-tuned parameters:
//
//     arch      moe_gemm    rmsnorm   rasr      swiglu  topk     fused
//     sm_80     72/160K     72/16     105/16    40/0    25/16    255/160K
//     sm_86     174/96K     72/16     106/16    40/0    25/16    255/96K
//     sm_89     174/96K     72/16     106/16    40/0    25/16    255/96K
//     sm_90     255/74K     71/1040   107/1040  38/0    30/1040  251/201K
//     sm_100    255/74K     61/1052   80/1052   40/0    32/1052  251/201K
//     sm_103    255/74K     61/1052   96/1052   40/0    30/1052  251/201K
//     sm_120    255/74K     71/1052   114/1052  40/0    30/1052  254/97K
//
//     REG / SHARED. Nothing FAILS, and every cell has STACK 0.
//
// The `255` on sm_90 and above is the same reading trap as the shared-memory
// figure. It is not spilling: STACK is 0 at every tiling tried there
// (32x128, 32x64, 64x64, 32x32), so 255 is the tile compiler using the whole
// per-thread budget on a part that has one, exactly as 96 KB of shared is it
// using the whole shared budget on Ada.
//
// What this does NOT say is that the Ada tunings are the right ones
// elsewhere. `kTileN`/`kTileK` were swept on one part; the resource figures
// move a long way across these rows (moe_gemm's shared goes 160K -> 96K ->
// 74K) and a sweep is the first thing to re-run on a new part. What it says
// is narrower and still worth having: **no architecture is excluded, and no
// kernel here is one register away from spilling on any of them.**
//
// # The whole tree, counted: 455 `__global__`s in seven buckets
//
// Every kernel in `kernels/` classified against the buckets that now have a
// MEASURED representative, so "what else should be a tile kernel" is a table
// rather than an opinion. Six of the seven verdicts below are backed by a
// kernel in this crate that was written, raced and checked for numerics; the
// seventh (`BITS`) is inferred and says so.
//
//     bucket    globals  measured by            verdict
//     ─────────────────────────────────────────────────────────────────────
//     ELEM         118   swiglu_tile            1.53x cached, par at
//                                               roofline
//     REDUCE        92   rmsnorm_tile,          1.5-2.0x, converging at
//                        rmsnorm_rasr_tile      roofline
//     SCAN          64   --                     no scan in cuda_tile.h
//     SEQ           56   --                     recurrence, sequential
//     COPY          35   gather_rows_tile       WASH: 0.97-1.10x,
//                                               bit-identical
//     ARGMAX        26   argmax_tile            LOSES 4x: 0.24-0.29x
//     DECODE        24   dequant_wna16_tile     1.9-3.5x, converging at
//                                               roofline
//     SORT          15   --                     sorting networks
//     REF           13   --                     the reference flashinfer is
//                                               checked against
//     GEMV           5   wna16_gemv_tile        LOSES: 0.46-0.97x
//     VENDOR         4   --                     flashinfer wrappers
//     GEMM           3   moe_grouped_gemm_tile  1.38-3.40x
//
// # One row of this table was wrong, and how
//
// ARGMAX was first priced off `topk_softmax_tile` at 1.28-1.40x, because
// both "reduce a row to a winner". They are not the same shape:
// `topk_softmax` reduces 128 experts inside ONE tile, `argmax` reduces
// 151,936 vocabulary entries across 149 of them. Measured, the second is
// **0.24x** -- the opposite sign, not a smaller number.
//
// `sample/argmax_tile.cuh` now carries the sweep and prices the loss piece
// by piece. The load-bearing part is that the FLOOR -- `reduce_max` alone,
// no index, no tail mask -- is already 1.9x behind, so no amount of work on
// the kernel recovers it. CuTile's reduction granularity is the tile, and a
// long strided reduction wants one accumulator and one tree.
//
// Recorded rather than quietly fixed. Every other row of this table has a
// measured representative; this one had a plausible one, and plausible is
// how the number got to be backwards for as long as it was.
//
// # Two variables, and the surface does not separate
//
// Chasing the argmax loss produced the one measurement here that applies to
// every row above -- and it took THREE attempts, two of which were pushed.
// Both wrong turns are kept, because each was a one-line law fitted to one
// slice of a two-dimensional surface, which is the same mistake as the
// ARGMAX row itself.
//
//   WITHDRAWN #1: "the reduction WIDTH is what costs" -- fitted to a sweep
//   that held total bytes constant, so the grid fell 8,192 -> 32 as the
//   width grew. Two variables moved at once.
//
//   WITHDRAWN #2: "no, width is free and BLOCKS PER SM is the only
//   variable" -- fitted to a width sweep run only at 512 blocks, which is
//   the one column where the width effect happens to be absent.
//
// The measurement, both variables moved independently. Controlled
// sum-reduction, tile form against the register-accumulator shape the tree
// uses everywhere, L40S, 142 SMs, 48 MB L2, exact at every cell. `*` marks
// cells whose working set exceeds L2:
//
//     tiles\blocks     8      64     142     284     512    1024    2048
//         1          1.07    1.07    1.08    1.09    1.12    1.24    1.29
//         4          1.09    1.09    1.07    1.06    1.13    1.19    1.19
//        16          1.03    1.05    1.00    1.02    1.07    0.82    0.86*
//        64          0.80    0.81    0.87    0.98    1.03*   1.12*   1.17*
//       256          0.66    0.67    0.75*   1.01*   1.12*   1.15*   1.18*
//
// Read it two ways; both are real:
//
//   * **Down the low-grid columns, width costs.** At 8 blocks: 1.07 -> 0.66x
//     from 1 to 256 tiles. This column is the clean one -- every cell is 0.0
//     to 4.0 MB, all L2-resident, so nothing but the width changes.
//   * **Along the wide rows, grid pays.** At 64 tiles: 0.80 -> 1.17x from 8
//     to 2,048 blocks, monotone, with no step at the L2 boundary between 284
//     (35 MB) and 512 (64 MB). The volume grows along this row, so it is the
//     less clean of the two reads.
//
// Neither variable alone predicts a cell, and the 16-tile row (1.03, 1.05,
// 1.00, 1.02, 1.07, 0.82, 0.86) is not monotone in either. **There is no
// one-line law here.** What there is:
//
//   * Decode lives in the top-left -- few blocks. There WIDTH decides, and
//     the boundary falls between 16 and 64 tiles: about 16 K to 64 K
//     elements per row.
//   * Prefill moves right, where a large grid recovers even wide reductions.
//
// Every measured row of the table above sits where this predicts.
// `rmsnorm_tile` is 4 tiles at 1 block -- top-left, and it wins at every
// batch size, which is why its predicate has no floor. `topk_softmax_tile`
// is 1 tile. `argmax_tile` is 149 tiles at 1-128 blocks -- bottom-left, the
// worst corner, and it loses 4x.
//
// A second, independent penalty rides on top for reductions that cannot
// accumulate element-wise. A running sum lives in a tile across chunks; a
// running max cannot, because an index has to be recovered against it, so
// every chunk collapses to a scalar. Same grid, same width: 0.42x against
// 0.79x. That is what puts argmax below even its corner of the surface.
//
// **Crossover VALUES still do not transfer between kernels**: 1.5 blocks/SM
// for `moe_fused`, 7.2 for the router, and for this reduction the crossover
// is not a block count at all but a width. Four kernels, four answers.
// Quoting one for another is what put 1.28-1.40x in the ARGMAX row, and then
// produced two withdrawn laws while trying to fix it.
//
// # The REDUCE row, enumerated instead of inferred
//
// The surface makes the ARGMAX mistake checkable rather than lucky. REDUCE
// is 92 kernels priced off `rmsnorm_tile`, and the question the surface
// raises is whether any of them reduce over an axis long enough to fall out
// of the winning corner -- the same question nobody asked about argmax.
//
// **The counting rule was a test, not this comment.** The first pass used a
// shell pipeline and gave 43, 45 or 46 for the same bucket depending on
// which regex asked; a count that moves with the pattern is not an
// enumeration. `vendor_manifest.rs::reduction_axis_counts` was the
// definition, and the numbers below are the last output it produced.
//
// It was deleted at `1a08b179a` with the rest of the vendored-tree manifest,
// and nothing re-derives them. **They are a dated observation now rather
// than a count CI keeps true** -- which is the weaker of the two things, and
// the paragraph after the table was written while they were still the
// stronger one.
//
//     axis width                      sites   surface says
//     ─────────────────────────────────────────────────────────────────
//     <= 1 tile                         161   1.07-1.29x  best corner
//       head_dim, K_d, V_d, kv_lora, half, D, H, cols, dim,
//       num_experts, num_routes, nkeys, heads_here, ...
//     3-7 tiles (hidden 2816-7168)       25   1.06-1.19x  still ahead
//     data-dependent                     11   UNKNOWN
//       ptir/tier0.cuh `len`, layout `total` and `num_tokens`
//     > 16 tiles                          2   0.66-0.75x  declined
//       sample/argmax.cuh `vocab` -- the one already caught
//     ─────────────────────────────────────────────────────────────────
//     classified                        199
//     axis not in the list above        101
//     ─────────────────────────────────────────────────────────────────
//     all strided block reductions      300
//
// The bottom three numbers WILL move -- a concurrent branch pushed this
// table from 300 to 307 within an hour of it being written, and while the
// gate lived it named the drift rather than letting the census quietly go
// stale. Nothing names it now, so the drift is silent and re-deriving them
// is no longer a one-line fix but a fixture to write a second time.
//
// The number that must NOT move quietly is `> 16 tiles`, which had its own
// assertion with its own message and lost it with the rest: a new reduction
// over a vocabulary-sized axis is a new ARGMAX row waiting to happen, and it
// should stop a build rather than be re-derived. Of everything the deleted
// fixture checked, that is the row worth restoring first.
//
// So the REDUCE verdict holds for the classified two thirds: 186 of the 188
// determinate sites are at 7 tiles or fewer, inside the corner
// `rmsnorm_tile` was measured in, and the 2 that are not are argmax and are
// already declined.
//
// **The 101 unclassified sites are named rather than folded into the safe
// bucket.** They are model dimensions -- `kv_dim`, `q_dim`, `total_heads`,
// `state_size`, `group_size`, `total_steps`, `span` -- all small in every
// configuration this tree serves. The one that can cross the boundary is
// `inter`, the FFN intermediate width, which reaches 18,944 on dense
// models: 18 tiles, past the 16-tile line. Two sites, not tiled today, and
// this is the note for whoever tiles them.
//
// **The 11 data-dependent sites cannot be settled here.** `ptir/tier0.cuh`'s
// `k_reduce_*` family is one block per row strided over a runtime `len` --
// structurally identical to argmax, and whether it lands in the good corner
// depends on the PTIR program, not on this tree. Anyone tiling those must
// take `len` in the predicate and cut it where the surface says:
//
//     len <= 16 * 1024   ->  tile form is ahead
//     len >= 64 * 1024   ->  tile form is behind at decode grids
//
// and measure the gap between, which is exactly where the 16-tile row stops
// being monotone. A predicate that ignores `len` there would be the ARGMAX
// row a third time.
//
//     worth a CuTile alternative   263   57%
//     measured and declined         40    8%   (COPY 35, GEMV 5)
//     cannot, or should not        152   33%
//
//     **No inferred rows.** Every verdict is backed by a kernel that was
//     written and raced, including the two that say do not bother.
//
// # What the census changed, which is why it was counted rather than guessed
//
// Three files were classified wrong on inspection and corrected by reading
// them:
//
//   * `ptir/tier0.cuh` is 27 kernels and was filed whole under SCAN because
//     it contains `k_scan`. It is an op-kernel library: sixteen are
//     elementwise, four are argmax-shaped, two are gathers, one is a
//     reduction, one is a matmul, and only THREE are scans.
//   * `attn/dsv4_compress.cuh` (16) and `attn/qkv_fused.cuh` (5) were filed
//     as elementwise. Both are reductions -- `qkv_fused`'s shuffles compute
//     an RMS norm, and `dsv4_compress` has fourteen block-reduction sites.
//
// That moved 45 kernels between buckets, most of them into the "worth it"
// column. A census of 455 kernels done by eye would have been wrong by about
// ten percent in the direction of whatever was expected.
//
// # And the one bucket left inferred was the one that was wrong
//
// The first version of this table had a `BITS` row of 32 kernels reading
// "PRMT and M=1 GEMV, both bandwidth-bound" -- a wash, inferred, flagged as
// inferred. Measured, `quant/dequant_wna16_tile.cuh` runs **3.5x** ahead of
// the scalar decode, bit-identical, because INT4 to bf16 is a 4x EXPANSION:
// it reads little, writes a lot, and does a shift, a mask, a subtract and
// two converts per output element.
//
// So `BITS` split three ways -- 24 DECODE kernels of the measured shape,
// 3 absmax reductions that belong in REDUCE, and 5 M=1 GEMVs.
// **The verdict flagged as unmeasured was the only one that was wrong, and
// it was wrong by 3.5x toward doing nothing.**
//
// The GEMV half was then measured too, because a table with one inferred row
// is a table with a hole. It LOSES, 0.46x to 0.97x -- so that half of the
// old `BITS` verdict was right, and the table now says so on the strength of
// `quant/wna16_gemv_tile.cuh` rather than on the strength of an argument.
//
// # The one predicate that is not about speed
//
// `rope_partial_tile_preferred` answers false while being 1.2-1.4x FASTER
// than its incumbent, because it is also more ACCURATE and that is a
// behaviour change rather than an improvement anyone here can wave through.
// `rope.cuh` uses `__sincosf`; the tile form uses `ct::sin`/`ct::cos`; at a
// rope angle of 3,867 radians the fp64 reference says the tile form is off
// by 1.5e-07 and the incumbent by 4.6e-04. Rope error feeds attention
// scores, so the incumbent's trade may be deliberate and the decision is not
// a merge.
//
// It also settles a question the census left open. ELEM was measured with
// `swiglu_tile`, which is CONTIGUOUS elementwise, and COPY showed that a
// permuted access with NO arithmetic is a wash. Rope is permuted WITH
// arithmetic and lands with ELEM, so **the arithmetic decides and the access
// pattern does not.**
//
// # The one machine fact four predicates keep rediscovering
//
// Four of the bounds below were derived independently and are the same
// thing: the tile advantage vanishes when the working set stops fitting
// cache. Collected, the sightings agree more closely than they had any right
// to:
//
//     kernel            MB touched   ratio    state
//     swiglu                   101    1.53    open
//     rmsnorm_rasr              92    1.97    open
//     rmsnorm_rasr             138    1.28    open
//     dequant_wna16             67    3.51    open
//     wna16_gemv                67    0.44    open (the other way)
//     ─────────────────────────────────────────────────────────────
//     wna16_gemv               134    0.85    closed
//     swiglu                   151    0.96    closed
//     rmsnorm_rasr             184    1.02    closed
//     dequant_wna16            268    1.01    closed
//
// Last point with a gap: 138 MB. First point converged: 134 MB. **The
// transition is at roughly 3x this part's 48 MB of L2**, across four kernels
// with completely different arithmetic intensities -- an elementwise
// activation, a three-pass reduction, a 4x expansion, and a weight-streaming
// GEMV -- and in the GEMV's case it closes a LOSS rather than a win.
//
// That is worth a name because it makes a port cheap: re-measure L2 capacity,
// not four crossovers.
//
//     constexpr long long kRooflineBytes = 3 * kL2Bytes;
//
// # ...and why the predicates below do NOT use it
//
// They are bounded at their own largest MEASURED point instead, which is
// tighter than the band: `swiglu_tile_preferred` stops at 100.7 MB and
// `dequant_wna16_tile_preferred` at 67 MB, both well inside 134. Widening
// them to a derived constant would be trading a measurement for a model, and
// this file exists because a bound that is a model gets quoted as a
// measurement. The band is an observation about the machine; the predicates
// are claims about kernels, and only one of those has been checked at 134 MB.
//
// # The saturation crossovers do NOT unify, and forcing them would be the
//   same error
//
// `moe_fused_tile_preferred` crosses at 212 blocks on a 142-SM part -- 1.5
// blocks per SM -- and `topk_softmax_tile_preferred` at about 1,024 rows,
// which is 7.2 blocks per SM. Those are not one line. They are two kernels
// running out of parallelism at different points for different reasons, and
// a single "blocks per SM" constant covering both would be a story rather
// than a measurement.
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
#include "moe/moe_fused_tile.cuh"
#include "moe/moe_grouped_gemm_tile.cuh"
#include "moe/topk_softmax_tile.cuh"
#include "norm/rmsnorm_rasr_tile.cuh"
#include "rope/rope_tile.cuh"
#include "norm/rmsnorm_tile.cuh"

namespace pie {

namespace {

namespace n_ = ::pie::norm;
namespace m_ = ::pie::mlp;
namespace o_ = ::pie::moe;

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
static_assert(n_::rmsnorm_rasr_tile_preferred(2048, 2816), "measured 12.93 vs 24.74 us");
// And at the far end, where the win has converged to 1.02x but has not
// reversed -- swept 1 to 65,536 rows after a sibling predicate turned out to
// have been published from a single point.
static_assert(n_::rmsnorm_rasr_tile_preferred(65536, 2816), "measured 2172 vs 2210 us");

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

// Fusing: 0.989 vs 0.654 ms at 106 blocks, 1.360 vs 1.315 at 212, 4.430 vs
// 10.244 at 1,696. The crossover is a BLOCK count because what runs out at
// the bottom is parallelism -- 106 blocks on a 142-SM part is under one per
// SM -- and blocks are what the machine schedules. Three earlier versions of
// that header called this a negative result from the 106-block point alone.
static_assert(!o_::moe_fused_tile_preferred(106), "measured 0.989 vs 0.654 ms");
static_assert(o_::moe_fused_tile_preferred(424), "measured 1.796 vs 2.594 ms");
static_assert(o_::moe_fused_tile_preferred(1696), "measured 4.430 vs 10.244 ms");

// The grouped GEMM's predicate is a divisibility claim, not a crossover: it
// is ahead at both shapes pie fires and the conditions are the ones its own
// `static_assert`s state.
static_assert(o_::moe_grouped_gemm_tile_preferred(512, 2048), "gate_up divides");
static_assert(!o_::moe_grouped_gemm_tile_preferred(512, 2049), "K must divide");
// Unconditional in the grid, swept from 1 block to 5,088: 3.40x, 1.38x,
// 2.96x, 2.66x, 3.16x, 3.26x, never crossing 1.0x and error 0 throughout.
static_assert(o_::moe_grouped_gemm_tile_preferred(2048, 256), "down divides too");

}  // namespace

}  // namespace pie
