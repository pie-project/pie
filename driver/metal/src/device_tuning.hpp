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
    int qmm_min_batch = 12;

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
    /// M1 Max: 8, i.e. half the narrow tile. Below it the sort's padding costs
    /// more than the weight re-reads it saves -- a decode routing eight pairs
    /// over a hundred and twenty-eight experts would make every tile one live
    /// row in sixteen. Moves with the same thing `qmm_min_batch` does.
    int moe_batch_min_per_expert = 8;
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
/// binary, and a different binary is a different measurement.
const DeviceTuning& device_tuning();

/// The GEMM crossover for this device. A function and not a constant because
/// it is a property of the machine; see `DeviceTuning::qmm_min_batch`.
int qmm_min_batch();

/// The unsplit GEMM's BN crossover, in threadgroups.
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
