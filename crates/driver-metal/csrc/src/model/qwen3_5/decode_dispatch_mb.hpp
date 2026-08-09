#pragma once
// decode_dispatch_mb.hpp — beta's M>1 launch-geometry helpers (multi-batch lane).
//
// The N>1 generalization of decode_dispatch.hpp. Pie's batch dim is N=total_tokens; the
// raw-Metal activations are token-major [N, feature], so every per-row kernel just stacks N
// rows on the batch axis. KEY INSIGHT (quantized_qmv.metal): affine_qmv_fast ALREADY batches
// over tid.x (`x += tid.x*in_vec_size; y += tid.x*out_vec_size`) — the M=1 path launches with
// 1 threadgroup in x. So an M>1 batched GEMV is just `grid.x *= N`, BIT-EXACT by construction
// (each row reproduces the M=1 accumulation) and reducing to the shipped path at N=1. No new
// GEMM kernel is needed for CORRECTNESS; the tiled affine_qmm (weight reuse across rows) is a
// PERF lever layered on AFTER the parity gate is green.
//
// Pure (no Metal types beyond Grid/Threadgroup). dispatchThreads semantics: Grid = TOTAL
// THREADS, tg = threads/threadgroup, threadgroups = ceil(Grid/tg) per axis (matches
// decode_dispatch.hpp + RawMetalContext::dispatch).

#include "decode_abi.hpp"
#include <algorithm>
#include <cstdlib>

#include "decode_dispatch.hpp"

// The shapes this file used to hold. They are the kernels' -- each is a pure
// function of the geometry, read off a `[[thread_position_in_grid]]` contract
// -- and moved to `crates/kernels-metal/kernels/<family>.h` with §7 step 3 of
// .wiki/kernel-metal-refactor.md. What stayed is what reads `DeviceTuning`:
// `qmm_bn`, `qmm_bn_unsplit` and `sdpa_should_tile` are crossovers measured on
// a machine, so they are this driver's and they hand their answer DOWN as an
// argument rather than being read from below.
#include "pie/kernels/attn.h"
#include "pie/kernels/layout.h"
#include "pie/kernels/mlp.h"
#include "pie/kernels/norm.h"
#include "pie/kernels/quant.h"
#include "pie/kernels/rope.h"
#include "decode_step_mb.hpp"  // M=1 helpers (qmv_dispatch, rms_dispatch, ...)
#include "mtl4_context.hpp"     // Grid, Threadgroup

#include "device_tuning.hpp"

namespace pie::metal {

using pie::kernels::attn::kSdpaMmaHeadDim;
using pie::kernels::attn::kSdpaQueryTile;
using pie::kernels::attn::kv_append_mb_dispatch;
using pie::kernels::attn::sdpa_paged_dispatch;
using pie::kernels::attn::sdpa_paged_mma_dispatch;
using pie::kernels::attn::sdpa_paged_tiled_dispatch;
using pie::kernels::layout::embed_mb_dispatch;
using pie::kernels::mlp::elementwise_mb_dispatch;
using pie::kernels::norm::rms_mb_dispatch;
using pie::kernels::quant::kQmmBM;
using pie::kernels::quant::kQmmBMCount;
using pie::kernels::quant::kQmmBMWide;
using pie::kernels::quant::kQmmBMs;
using pie::kernels::quant::kQmmSplitBN;
using pie::kernels::quant::kQmmSplitMaxOut;
using pie::kernels::quant::kQmmSplitMaxSplits;
using pie::kernels::quant::kQmmSplitTargetTgs;
using pie::kernels::quant::qmm_splitk_reduce_dispatch;
using pie::kernels::quant::qmm_t_dispatch;
using pie::kernels::quant::qmm_t_splitk_dispatch;
using pie::kernels::quant::qmm_t_strided_dispatch;
using pie::kernels::quant::qmv_mb_dispatch;
using pie::kernels::rope::rope_mb_dispatch;



/// Which row block a batch of `N` rows should use, as an index into `kQmmBMs`.
/// A block only pays for itself once the batch can fill it, so the rule is the
/// widest rung the batch can cover. Callers pad the grid up to that rung, which
/// is why the row count they pad to must be asked of THIS function and not of
/// `kQmmBMWide`: padding a one-row decode to the widest block would launch 64
/// rows of arithmetic to compute one.
inline int qmm_bm_index(int N) {
    int best = 0;
    for (int i = 1; i < kQmmBMCount; ++i)
        if (N >= kQmmBMs[i]) best = i;
    return best;
}

inline int qmm_bm(int N) { return kQmmBMs[qmm_bm_index(N)]; }

/// The reverse: which `kQmmBMs` rung a chosen row block is. Dispatch records
/// the block it launched, not the batch it came from, so the PSO lookup has to
/// come back the other way.
inline int qmm_bm_slot(int bm) {
    for (int i = 0; i < kQmmBMCount; ++i)
        if (kQmmBMs[i] == bm) return i;
    return 0;
}

// Output columns per threadgroup.  This GEMM is occupancy-bound, not bandwidth-
// bound: measured standalone at the model's shapes it turns in ~380 GFLOP/s at
// 16 threadgroups, ~900 at 32, ~1600 at 112 and saturates around 2.2-2.6 TFLOP/s
// past ~200, against a 389 GB/s machine it only ever reaches 11% of.  So take
// the widest tile -- wider means each x row is loaded fewer times -- that still
// leaves enough threadgroups to fill the machine, and past that prefer more
// threadgroups over a wider tile.
//
// The measured optimum for every projection in the checkpoint falls out of that
// one rule: BN=64 for lm_head (3880 tg), 32 for the GDN in-projection (192),
// 16 for everything else.  The old rule asked only whether `out_vec/64 >= 64`,
// which handed the GDN in-projection a BN=64 that measured 21% slower.
//
// BN partitions output columns only -- every element's K sum is unchanged -- so
// the choice is bit-exact whichever way it goes.

inline int qmm_bn(int out_vec, int N, int min_batch) {
    const int bm = qmm_bm(N);
    if (N < min_batch || N % bm != 0) return 0;
    // Take the WIDEST tile that divides the output, full stop.
    //
    // This used to gate on a threadgroup count, and that was right when the
    // GEMM had nothing else supplying parallelism: measured then, BN=64 lost
    // everywhere except lm_head and the occupancy rule was worth 677 -> 712
    // tok/s.  Split-K changed the premise.  The split now supplies the
    // threadgroups, so the only thing BN still decides is how many times each
    // weight tile is dequantized -- and wider is strictly fewer.
    //
    // Interleaved A/B, decode step, widest against the old 192-threadgroup
    // rule: 16 lanes 31.57ms to 37.02, 32 lanes 141.18 to 158.45.  The old
    // rule is a pessimization now.
    int best = 0;
    for (int bn : {16, 32, 64})
        if (out_vec % bn == 0) best = bn;
    return best;
}

/// The threadgroup count past which this GEMM stops caring about more of them.
///
/// Where BN=32 starts beating BN=16, in threadgroups.
///
/// NOT the machine's saturation point, which is what an earlier version of this
/// called it. The sweep below brackets the crossover between 144 threadgroups
/// (where 16 still wins) and 192 (where 32 does); the machine saturates higher
/// than that, and using the saturation number here cost up to 12% because it
/// let the choice run past 32 to 64.
///
/// Read from `DeviceTuning`, not a constant: it is the threadgroup count at
/// which a wider tile stops being worth fewer of them, and how many
/// threadgroups fill the machine is the machine's business. The value above is
/// the M1 Max's; the M4 Pro re-measurement, which lands lower for exactly that
/// reason, is in `device_tuning.hpp`.
inline int qmm_bn_crossover_tg_value() { return qmm_bn_crossover_tg(); }

/// `qmm_bn` for a family whose GEMM has no split-K behind it.
///
/// The rule above is correct *because* the split supplies threadgroups when the
/// output tiles do not. A family that dispatches no split has no such supply,
/// and taking the widest tile then starves the machine: at M=128 with BM=64 a
/// projection to 1024 columns gets `1024/64 * 128/64` = 32 threadgroups, which
/// the curve prices at a third of what the same work does at 200.
///
/// So: the narrow tile until there is enough work to fill the machine, and 32
/// after that. Never 64 -- that is the finding, not an omission. Measured with
/// `roofline_probe` on gemma-4-E2B's own projections, BM=64, GFLOP/s, best of
/// each row starred:
///
///     M     N        tg@32    BN=16     BN=32     BN=64
///     128    512        16   *2187*     1829      1054
///     128   1024        32   *3249*     3115      2234
///     128   2048        64   *3399*     3346      3333
///     128   3584       112    3595     *3904*     3098
///     128   6144       192    3829     *4302*     4012
///     192   1536       144   *3820*     3529      2386
///     192   2048       192    3694     *4082*     3103
///     192   6144       576    3919     *4457*     4162
///     448    256        56   *2727*     2603      1896
///     448   1536       336    3865     *4101*     3573
///     448   2048       448    3820     *4337*     3851
///     448   6144      1344    3986     *4569*     4275
///    1024   1536       768    4000     *4565*     4252
///    1024   2048      1024    3941     *4511*     4131
///    1024   6144      3072    4016     *4619*     4326
///
/// Sixteen columns of measurement and BN=64 is the best of none of them, which
/// is why the rule no longer reaches for it. The threshold sits in the only gap
/// the sweep leaves: 144 threadgroups still wants 16, 192 already wants 32.
///
/// Checked against the machine and not just the probe, because the MIXTURE's
/// tile rule was built the same way and the probe misled it three times over.
/// Forcing each width through a real llama-3.2-1B prefill agrees with the
/// table: at 448 rows BN=16/32/64 give 2565.8 / 2663.7 / 2578.3 tok/s and at
/// 1024 rows 2270.8 / 2349.8 / 2297.0, so the rule's 32 is the machine's 32.
/// The dense side's probe holds where the mixture's did not, and the reason is
/// the mixture's alone: its threadgroups read thirty-two experts' weights
/// where the probe reads one.
///
/// BN partitions output columns only, so this is bit-exact whichever way it
/// goes; it decides how many times a weight tile is dequantized, not what the
/// sum is.
inline int qmm_bn_unsplit(int out_vec, int N, int min_batch) {
    const int bm = qmm_bm(N);
    if (N < min_batch || N % bm != 0 || out_vec % 16 != 0) return 0;
    const int row_tiles = N / bm;
    // BN=64 was tried here and does not pay on the batched DECODE, which is
    // the only shape this function serves: Qwen3.6-27B on an M4 Pro measured
    // 42.3 / 72.4 tok/s at 8 and 16 lanes against 43.0 / 73.4 at BN=32, and
    // BN=16 -- which doubles the threadgroup count -- is 20% WORSE (33.6 /
    // 63.3). More threadgroups losing that badly is the finding: a decode fire
    // is one row tile, so BN does not change how many times a weight is
    // dequantized, and what it does change is how much of `x` each threadgroup
    // re-reads. 32 is the middle this shape wants.
    if (out_vec % 32 == 0 && (out_vec / 32) * row_tiles >= qmm_bn_crossover_tg_value())
        return 32;
    return 16;
}


// Each partition must be a whole number of BK-wide tiles AND whole quantization
// groups, or it reads into the next group's scales.
inline int qmm_split_k(int out_vec, int N, int K, int bm) {
    if (out_vec % kQmmSplitBN != 0 || bm <= 0) return 1;
    // Count the tiles the SPLIT dispatch will actually launch: `kQmmSplitBN`
    // wide and `bm` tall, which is the grid `qmm_t_splitk_dispatch` builds.
    //
    // This used to count rows in units of `kQmmBM`, the NARROWEST block, on the
    // theory that a wide block is twice as parallel and should be split half as
    // deep. That is backwards -- a wide block covers twice the rows in ONE
    // threadgroup, so it produces half the tiles and needs MORE split, not less
    // -- and the numbers that appeared to support it were measured on qwen3.5's
    // split path, which never dispatched its reduce and so was timing the
    // wrong answer. Counting honestly is worth 741 -> 870 tok/s at 32 lanes on
    // llama-1B.
    const int tiles = (out_vec / kQmmSplitBN) * ((N + bm - 1) / bm);
    static const int target = [] {
        const char* e = std::getenv("PIE_METAL_SPLIT_TGS");
        return e ? std::atoi(e) : kQmmSplitTargetTgs;
    }();
    int split = tiles > 0 ? target / tiles : 1;
    split = std::min(split, kQmmSplitMaxSplits);
    const int k_align = 64;  // group_size, and a multiple of BK=32
    split = std::min(split, K / k_align);
    while (split > 1 && K % (split * k_align) != 0) --split;
    return split < 2 ? 1 : split;
}



// `out/BN` threadgroups across the output, `M/BM` across the batch, each
// 32x2x2 = 128 threads (WM=WN=2 simdgroups), which is the shape steel's
// BlockMMA is written for.
// The prefill's batched projection. Rows are padded up to a whole BM tile: the
// scratch pool holds `max_tokens` rows and the tail rows land in ones the fire
// does not use, so the padding computes discardable values rather than needing
// a bounds-checked inner loop.
// A prompt has far more rows than a decode batch, so it can afford the wide
// row block -- and needs it for the same reason the decode does: the tile is
// dequantized once per row block, so a 512-row prompt at BM=16 unpacks every
// weight thirty-two times.
inline int qmm_strided_bm(int padded_rows) {
    static const bool off = std::getenv("PIE_METAL_NO_PREFILL_BM32") != nullptr;
    // 128 was instantiated and measured and is not here: Qwen3.6-27B prefill
    // gives 103.0 tok/s at 128 rows against 64's 104.5, and 106.5 against 106.7
    // at 512. A 128-row block is 12.8 KiB of threadgroup memory against 64's
    // 7.7, which takes a core from four resident threadgroups to two -- and
    // overlapping one threadgroup's weight read with another's MMA is the only
    // thing hiding either, per the note in `qmm_t_loaded_impl`. Halving the
    // dequantizations does not pay for it. The rung stops here.
    if (off) return kQmmBM;
    return padded_rows >= 64 ? 64 : (padded_rows >= 32 ? 32 : kQmmBM);
}

inline int qmm_strided_rows(int N, int max_rows) {
    const int bm = qmm_strided_bm(N);
    const int padded = ((N + bm - 1) / bm) * bm;
    return padded <= max_rows ? padded : 0;
}


/// Rows the BATCHED DECODE launches its dense GEMM over, for a fire of `n`.
///
/// The kernel takes no `M`. It is written for full tiles only -- see the header
/// of `quantized_qmm_t.metal` -- so the driver may select it only when
/// `M % BM == 0`, and the row count reaches it through the grid. Handing it the
/// raw fire width therefore made the GEMM reachable at EXACT MULTIPLES OF A
/// RUNG AND NOWHERE ELSE, which for a decode is almost never: measured on
/// Qwen3.6-27B, a device that affords 24 recurrent slots ran 75.6 tok/s at 16
/// lanes and 30-32 at 2, 4, 6, 8, 12, 20 and 24 -- a flat curve with one spike,
/// because 16 was the only width that divided a rung. Six times the lanes
/// bought nothing.
///
/// So pad the fire up to its rung, which is what every other caller of this
/// GEMM already does -- the prefill in `qmm_strided_rows` just above, and
/// llama's `llama_qmm_rows`. The padding is free of consequence for the same
/// reason theirs is: the scratch pool holds `max_tokens` rows token-major, so
/// rows `n .. padded-1` land in slots the fire does not read and compute
/// discardable values, rather than the kernel needing a bounds-checked inner
/// loop. A GEMM row's output depends only on its own input row, so garbage in
/// the tail cannot reach a real one.
///
/// Two guards, both of which fall back to the unpadded width (and so to the
/// matvec, since it will not divide a rung):
///   * `n < min_batch` -- padding must not be able to talk the dispatch past
///     the measured crossover. A 2-row fire padded to 16 would launch eight
///     times the arithmetic it needs.
///   * `padded > max_tokens` -- the pool is only that deep, and a wider write
///     would run into the next activation's slot.
inline int qmm_mb_rows(int n, int max_tokens, int min_batch) {
    const int rows = n < 1 ? 1 : n;
    if (rows < min_batch) return rows;
    const int bm = qmm_bm(rows);
    const int padded = ((rows + bm - 1) / bm) * bm;
    return padded <= (max_tokens < 1 ? 1 : max_tokens) ? padded : rows;
}








// Whether to tile the query rows. The tiled kernel gives a row a simdgroup
// where the per-row kernel gives it a threadgroup, so below a full tile it is
// strictly worse: at one row it would run one simdgroup of the thirty-two the
// other kernel would have used. A fire earns the tiled shape by filling a tile.
/// Whether a fire's attention should use the tiled kernel rather than the
/// per-row one.
///
/// NOT a row count. The tiled kernel walks RUNS OF EQUAL REQUEST inside each
/// 32-row tile, staging that run's keys into threadgroup memory and letting
/// only the run's own simdgroups read them -- so its whole advantage is rows
/// that share a key span, and its cost is a serial pass per run. A prefill is
/// all one run and wins outright. A fleet of decodes is the opposite shape:
/// thirty-two rows, thirty-two runs, one simdgroup live per pass and the tile
/// staged thirty-two times. Measured on llama-1B, batch 32, that is 370 tok/s
/// tiled against 728 per-row, and batch 64 is 480 against 915.
///
/// Asking only `N >= kSdpaQueryTile` could not tell those apart, because the
/// two fires have the SAME row count. The request count is what separates
/// them, and the caller already has it: `qo_indptr` is one entry per request
/// plus a terminator.
inline bool sdpa_should_tile(int rows, int requests) {
    const int r = requests > 0 ? requests : 1;
    // Not `kSdpaQueryTile`, though it is the same number on this machine. That
    // one is the tile's HEIGHT and is the simdgroup count; this is a crossover
    // and belongs to the machine. See `DeviceTuning`.
    return rows / r >= sdpa_tile_min_rows_per_request();
}





}  // namespace pie::metal
