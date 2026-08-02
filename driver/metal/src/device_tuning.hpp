// Per-device tuning constants.
//
// Every crossover in this driver was measured on one machine -- an M1 Max --
// and then written down as a `constexpr`. That was honest while there was one
// machine. On an M4 Pro the same constants are wrong in the direction that
// costs the most: the GEMM crossover sits three rows too high, so the batches
// where the GEMM already wins are still served by the GEMV.
//
// The rule this file exists to keep: a DEFAULT-CONSTRUCTED `DeviceTuning`
// reproduces the M1 numbers EXACTLY. Adding a device may never change what an
// unrecognised one does, and a machine this file has never heard of gets the
// constants that were measured rather than an extrapolation. Overrides are
// per-generation and each carries the measurement that justifies it.
//
// `benches/tune_device.py` takes that measurement. Every field below has a
// `PIE_METAL_*` override so it can be swept without a rebuild, and the script
// finds, for each one, a batch where its candidate values provably choose
// DIFFERENT paths -- which is not a detail. A threshold does nothing at
// batches that do not straddle it, so sweeping one anywhere else times the
// same work twice and reports the reassuring answer.

#pragma once

#include <cstdint>

namespace pie::metal {

/// What the driver knows about the GPU it is running on. Queried once and
/// cached; every field has a value on every path, including the one where the
/// query fails.
struct DeviceInfo {
    /// `MTLGPUFamilyApple<N>`, resolved NEWEST-FIRST: the families are
    /// cumulative, so an M4 answers `supportsFamily:` for Apple7 as well as
    /// Apple9 and an oldest-first probe would call every Apple Silicon device
    /// an Apple7. 0 when no Metal device answered.
    int apple_family = 0;
    /// From IOKit's `gpu-core-count`, which is the only place the count is
    /// published -- `MTLDevice` does not expose it. 0 when absent.
    int gpu_core_count = 0;
};

/// The tuned constants, defaulted to the M1 Max measurements.
struct DeviceTuning {
    /// The batch at which the ported steel GEMM overtakes the batched GEMV.
    ///
    /// M1 Max: 12. Measured -- pie's per-step cost beats mlx-lm's at every
    /// batch up to 8 with the GEMV and only loses above it.
    ///
    /// M4 Pro (Apple9, 20 cores): 8. Measured on gemma-4-E4B at the batch
    /// that discriminates -- concurrency 8, where 12 takes the GEMV and 8
    /// takes the GEMM -- same binary via the env override, arms alternated,
    /// quiet host: 12 -> 140.88, 136.92 tok/s; 8 -> 144.44, 143.63. Means
    /// 138.90 vs 144.04, +3.7%. (At concurrency 16 both settings take the
    /// same path and measure the same, which is the check that the difference
    /// above is the crossover and not the weather.) The M4's wider per-core
    /// matrix throughput moves the crossover down; the GEMV's advantage at
    /// small N is a memory-bound one and does not.
    ///
    /// M2 Max (Apple8, 38 cores): 8. Measured the same way -- one binary, the
    /// env override, arms alternated within each batch, three reps each -- on
    /// four DENSE checkpoints, GEMM against GEMV, tok/s:
    ///
    ///     batch        6              7              8
    ///     Llama-1B   386.9/393.0   427.8/398.5    480.3/408.5
    ///     Llama-3B   166.7/167.6   181.0/167.8    204.1/171.2
    ///     Qwen3-1.7B 270.9/283.8   300.3/290.9    339.1/296.4
    ///     gemma-4E2B 173.9/197.9   192.3/202.8    220.6/210.9
    ///
    /// Eight is the first batch where the GEMM wins on ALL FOUR (+17.6, +19.2,
    /// +14.4, +4.6%), and the last batch where it loses on any is seven --
    /// gemma-4-E2B, -5.2%. Seven would buy the llama family another 3-8% and
    /// cost gemma that, so the value is the one with no measured regression
    /// anywhere rather than the one with the highest mean.
    ///
    /// DENSE only: see `qmm_min_batch_moe`, which the same sweep left at 12.
    int qmm_min_batch = 12;

    /// The same crossover for a checkpoint whose FFN is ROUTED.
    ///
    /// A separate number because the measurement says so, not for symmetry. In
    /// a mixture the dense projections this constant governs are the attention
    /// four and the head; the FFN -- the largest weights in the layer, and on a
    /// decode the whole of the bandwidth -- is routed and takes
    /// `moe_tile_rows`' own decision instead. So the GEMM here pays its padding
    /// without the matrices that repay it.
    ///
    /// M2 Max (Apple8, 38 cores), GEMM against GEMV at the batches where the
    /// dense value would have switched, tok/s:
    ///
    ///     batch          6             7             8
    ///     Qwen3-30B    63.6/63.5    67.8/75.2    90.4/98.3
    ///     gemma-4-26B  60.1/64.3    71.7/76.6    84.8/96.3
    ///     gpt-oss-20B  96.0/101.3  102.1/104.9  107.4/107.8
    ///
    /// The GEMV wins or ties at every one of them: taking the dense value here
    /// would cost Qwen3-30B 8% and gemma-4-26B 12% at the batch it helps a
    /// dense model most. So a routed checkpoint keeps the M1 number.
    ///
    /// This agrees with what the two routed families already recorded from
    /// their own side on the M1 Max -- `gemma4_qmm_rows` measured a lowered
    /// crossover at -17%, `gptoss_qmm_rows` at -1%. Those were read then as
    /// "the inherited number holds"; they were the routed half of this split,
    /// taken before there was a dense half to compare against.
    int qmm_min_batch_moe = 12;

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile overtakes
    /// BN=16.
    ///
    /// M1 Max: 160. Measured with `roofline_probe` over fifteen (M, N) pairs
    /// and then checked against a real llama-3.2-1B prefill, which agrees --
    /// 2565.8 / 2663.7 / 2578.3 tok/s at BN 16/32/64, 448 rows. The sweep
    /// brackets the crossover between 144 threadgroups and 192, so this sits in
    /// the gap.
    ///
    /// A machine with more cores saturates later and wants this HIGHER, since
    /// the narrow tile's only advantage is that it makes more threadgroups.
    ///
    /// M4 Pro (Apple9, 20 cores): 96, which is that sentence read backwards --
    /// twenty cores against the M1 Max's thirty-two fill sooner, so the narrow
    /// tile runs out of reasons sooner. Re-swept with `roofline_probe` on this
    /// machine at BM=64, GFLOP/s, threadgroups counted as
    /// `(N_out/32) * ceil(M/64)`:
    ///
    ///     tg@32    BN=16    BN=32    delta
    ///        64     5004     4615    -7.8%   (M=128, N=1024)
    ///        96     5540     5863    +5.8%   (M=192, N=1024)
    ///       128     5829     5846    +0.3%   (M=256, N=1024)
    ///       160     5987     6411    +7.1%   (M=320, N=1024)
    ///       224     5980     6196    +3.6%   (M=448, N=1024)
    ///
    /// 160 is not conservative on this card, it is on the wrong side of a 5.8%
    /// gap. The bracket is (64, 96] and 96 is the sampled edge rather than the
    /// midpoint, which is what to write down when only one side was measured.
    ///
    /// End to end it is NOT a win, and this says so rather than implying one.
    /// Same binary, arms alternated on `PIE_METAL_QMM_BN_CROSSOVER_TG`, 160
    /// against 96: gpt-oss-20b-Q4 179.68 -> 180.03 tok/s (3 reps), gemma-4-E2B
    /// 621.60 -> 622.67 (2), Llama-3.2-3B 558.40 -> 560.85 (3), Llama-3.2-3B on
    /// a 3000-word prefill 13.09 -> 13.14 (2). Four workloads, four agreeing
    /// signs, not one of them clear of the noise. What carries the change is the
    /// kernel measurement and the sign, not any of those numbers. The probe says
    /// why the GEMM's 5.8% does not reach the step, too: these projections run
    /// at ~5% of the streaming roof, so the step is bound by weight traffic and
    /// not by how the tiles are cut.
    ///
    /// Those four workloads were not checked for divergence, though, and this
    /// crossover is decided PER PROJECTION: a batch where every width lands on
    /// the same side of all three candidates times identical code. So it was
    /// re-run on the M1 Max at a batch picked to straddle -- Llama-3.2-1B at 96
    /// rows, where 96 takes BN=32 on the 2048-wide projections and 160 and 256
    /// take BN=16, via `benches/tune_device.py`. 2050.8 / 2053.7 / 2053.0 tok/s
    /// at 160 / 96 / 256, five reps alternated, +0.1% against 1.2% noise; the
    /// control at 1024 rows, where all three choose alike, lands 0.0% apart on
    /// 0.1% noise. So the end-to-end neutrality survives a batch that could
    /// have shown a difference, which is worth more than the four that could
    /// not. Do not spend time re-testing this one end to end on a new card:
    /// sweep it with the probe, take the sign, move on.
    int qmm_bn_crossover_tg = 160;

    /// Rows an expert's run must hold before the mixture's GEMM takes a wider
    /// row tile: 32 above `moe_tile_mid_per`, 64 above `moe_tile_wide_per`.
    ///
    /// M1 Max: 12 and 88. Measured end to end over eight (rows, expert count)
    /// pairs on gpt-oss-20b and gemma-4-26B; the table is in
    /// `shared_kernels.hpp`. Measured END TO END on purpose --
    /// `roofline_probe` predicts the dense tile correctly and this one wrong
    /// three times over, because its single hot expert cannot show what
    /// thirty-two cold ones cost.
    ///
    /// These move with per-core matrix throughput the way `qmm_min_batch`
    /// does: a machine that runs a wide tile relatively faster wants both
    /// thresholds LOWER.
    int moe_tile_mid_per = 12;
    int moe_tile_wide_per = 88;

    /// Whether a dense g64/b4 projection stages its input to FP16 and feeds
    /// native FP16 simdgroup MMA instead of BF16.
    ///
    /// M1 Max: on, and it is the largest single win this driver has -- roughly
    /// 40% on the GEMM at every shape measured, which on gemma-4 was 938 ->
    /// 1298 tok/s of prefill.
    ///
    /// It exists BECAUSE of the machine. M1 and M2 have no native bfloat16
    /// matrix path and emulate it; Metal 3.1 and Apple9 (M3, M4) do have one.
    /// On those the staging pass has nothing left to buy and is a dispatch, a
    /// barrier and a buffer per projection -- so this is the one tuned field
    /// whose default may be actively WRONG on newer silicon rather than merely
    /// unmeasured, and the first thing to check on an M3 or M4.
    ///
    /// Measure it the way the crossovers are measured: `PIE_METAL_FP16_QMM=0`
    /// against the default, same binary, arms alternated, on a prefill-heavy
    /// shape where the GEMM is most of the fire.
    bool fp16_qmm = true;

    /// Rows a request must contribute before its attention takes the tiled
    /// kernel instead of the per-row one.
    ///
    /// M1 Max: 32, which is also the tile's height -- a fire earns the tiled
    /// shape by filling a tile. The two are separate numbers even so: the
    /// height is the simdgroup count and cannot move, while this is a
    /// crossover. Measured on llama-1B, where a 32-request fleet of one-row
    /// members runs 370 tok/s tiled against 728 per-row, so the threshold has
    /// to keep the fleet off it; a machine whose shuffles are cheaper relative
    /// to its FMAs wants it HIGHER, since the tiled kernel's whole advantage is
    /// that it removes reductions.
    int sdpa_tile_min_rows_per_request = 32;

    /// Rows an expert's run must hold before the mixture sorts and batches at
    /// all, rather than running the routed projections as matvecs.
    ///
    /// M1 Max: 4, a QUARTER of the narrow tile. It was 8 -- half the tile --
    /// on the model that the padding is wasted work and the break-even is
    /// where a run half fills a tile. Measured, that model is wrong in the
    /// direction that matters: a 4-bit mixture is bandwidth-bound, and what
    /// batching buys is reading each expert's slice ONCE instead of once per
    /// pair, which is worth far more than the arithmetic a half-empty tile
    /// throws away. 128-token prefill, tok/s:
    ///
    ///                        min_per=8   4      2      1
    ///     Qwen3.6-35B-A3B      220.8   275.1  275.0  265.3
    ///     gemma-4-26b-a4b      396.3   397.9    --     --
    ///     gpt-oss-20b          405.6   405.3    --     --
    ///
    /// The 35B is the row that moves because it is the widest routing here --
    /// 256 experts at top-8, so 128 rows is four rows an expert and 8 declined
    /// to batch at all. The other two were already batching at 8 and are
    /// unchanged. Below 4 the padding does start to cost.
    ///
    /// A DECODE still never batches: one row is `experts_per_token` pairs, and
    /// 8 pairs over 128 experts is nowhere near 4 rows an expert.
    int moe_batch_min_per_expert = 4;
};

/// The platform query. `device_tuning_apple.mm` on Apple, a stub in
/// `device_tuning.cpp` elsewhere. Call `device_info()` instead; this is the
/// seam, not the accessor.
DeviceInfo query_device_info();

/// The device this process is running on. Queried once on first call.
const DeviceInfo& device_info();

/// The tuning for this device. Queried once on first call.
///
/// `PIE_METAL_QMM_MIN_BATCH` overrides `qmm_min_batch` for a run, which is how
/// the crossover above was measured: a rebuild between arms is a different
/// binary, and a different binary is a different measurement. It carries the
/// ROUTED crossover with it unless `PIE_METAL_QMM_MIN_BATCH_MOE` names one --
/// a sweep that set only the dense number would otherwise measure a mixture
/// that never changed path and read the flat curve as the crossover not
/// mattering.
const DeviceTuning& device_tuning();

/// The GEMM crossover for this device. A function and not a constant because
/// it is a property of the machine; see `DeviceTuning::qmm_min_batch`.
///
/// Takes whether the checkpoint's FFN is routed, because the crossover is a
/// property of the machine AND of what the GEMM gets to cover: see
/// `DeviceTuning::qmm_min_batch_moe`. A parameter and not two functions so
/// that a caller cannot ask the question without answering it.
int qmm_min_batch(bool is_moe);

/// The unsplit GEMM's BN=16 -> BN=32 tile crossover, in threadgroups counted
/// at BN=32. See `DeviceTuning::qmm_bn_crossover_tg`.
int qmm_bn_crossover_tg();

/// The mixture's two row-tile crossovers, in rows per expert.
int moe_tile_mid_per();
int moe_tile_wide_per();

/// Whether the dense g64/b4 GEMM stages its input to FP16.
bool fp16_qmm();

/// The attention's tiled-vs-per-row crossover, in rows per request.
int sdpa_tile_min_rows_per_request();

/// The mixture's batch-vs-matvec crossover, in rows per expert.
int moe_batch_min_per_expert();

}  // namespace pie::metal
