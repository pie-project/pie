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
    /// The batch at which the ported steel GEMM overtakes the batched GEMV,
    /// for a checkpoint whose GEMM reaches the FP16 matrix path. See
    /// `qmm_min_batch_emulated` for the one that does not.
    ///
    /// M1 Max: 8. This used to read 12, from a sweep taken when the batched
    /// GEMM on this device was still emulating a bfloat matrix unit. It is
    /// not any more -- the dense projections stage their input to FP16 and
    /// use the instruction the hardware has -- and re-running the same sweep
    /// against the same binary via the env override, arms alternated twice,
    /// eight lanes, 128-token prompt, 64 decode steps, aggregate tok/s:
    ///
    ///     model                  12       8     delta
    ///     Qwen3.6-27B          23.0    32.1     +40%
    ///     gemma-4-31b          17.9    30.0     +68%
    ///     gemma-4-26b-a4b     107.3   130.1     +21%
    ///     gpt-oss-20b         108.2   116.2      +7%
    ///
    /// Every one of them prefers eight, and the two dense checkpoints prefer
    /// it by more than the FP16 wiring alone accounts for: what moved is not
    /// the GEMM's speed but which side of the crossover eight sits on. The
    /// old number was measuring a GEMM the device could not really run.
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
    /// DENSE only: see `qmm_min_batch_moe`, which the same sweep also moved.
    int qmm_min_batch = 8;

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
    /// dense model most. So a routed checkpoint kept the M1 number.
    ///
    /// That reading has since been overturned on the M1 Max, and by the same
    /// thing that overturned the dense number: those runs measured a routed
    /// GEMM emulating a bfloat matrix unit. With the routed expert GEMM on
    /// FP16 the M1 Max sweep reverses -- gemma-4-26b-a4b 107.3 -> 130.1 tok/s
    /// (+21%) and gpt-oss-20b 108.2 -> 116.2 (+7%) at eight lanes, arms
    /// alternated twice. The M2 table above stands as a measurement of the
    /// binary it was taken on; nothing has re-run it since the FP16 wiring,
    /// which is why the Apple8 entry still names its own number.
    int qmm_min_batch_moe = 8;

    /// The same two crossovers for a checkpoint whose quantization does NOT
    /// reach the FP16 matrix path -- anything but 4-bit at group 64.
    ///
    /// Twelve, which is what this driver shipped for every checkpoint before
    /// the matrix path existed, and it is the same measurement rather than a
    /// leftover: on the M1 Max at eight lanes Llama-1B at group 128 turns in
    /// 440.5 and 439.4 tok/s on the GEMV against 403.9 and 402.5 on the GEMM,
    /// arms alternated, -8.4%. The GEMM it runs is the one the old sweep
    /// measured, so it keeps the old sweep's answer.
    ///
    /// One number for dense and routed both, because nothing here has
    /// measured them apart and inventing a split would be a guess with a
    /// field to live in.
    int qmm_min_batch_emulated = 12;

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
    /// M1 Max: 32, and 64 never. These were 12 and 88, measured end to end
    /// over eight (rows, expert count) pairs, and they moved for the same
    /// reason `qmm_min_batch` did: the routed expert GEMM now gets the FP16
    /// matrix unit instead of an emulated bfloat one. A tile's ARITHMETIC got
    /// ~40% cheaper and the rows it pads did not, so the padding a wide tile
    /// wastes buys less than it used to, and the balance moves toward narrow.
    /// Re-swept on the same machine, prefill tok/s, arms alternated twice
    /// where the margin was under 2%:
    ///
    ///     per   model            rows    BM=16    BM=32    BM=64
    ///      12   26b               192   *642.4*   616.0       --
    ///      16   gptoss            128   *594.8*   551.8       --
    ///      24   gptoss            192   *642.3*   622.9       --
    ///      32   gptoss            256     654.2  *664.2*      --
    ///      64   gptoss            512        --  *721.4*      --
    ///     128   gptoss           1024        --  *725.1*   716.6
    ///     256   gptoss           2048        --  *656.1*   654.0
    ///
    /// So the mid threshold moves 12 -> 32, and the wide one has nowhere left
    /// to fire: 64 loses at 128 rows an expert and ties at 256, which is
    /// already past any batch this driver sees. It is set out of reach rather
    /// than deleted, because the field is what a device with a different
    /// matrix throughput would answer differently, and no other device has
    /// been re-measured since the routed GEMM changed underneath it.
    ///
    /// Measured END TO END on purpose -- `roofline_probe` predicts the dense
    /// tile correctly and this one wrong three times over, because its single
    /// hot expert cannot show what thirty-two cold ones cost.
    ///
    /// These move with per-core matrix throughput the way `qmm_min_batch`
    /// does: a machine that runs a wide tile relatively faster wants both
    /// thresholds LOWER.
    int moe_tile_mid_per = 32;
    int moe_tile_wide_per = 1 << 24;

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

    /// Whether a tiled prefill attention runs on the simdgroup MATRIX unit
    /// rather than the scalar path.
    ///
    /// M1 Max: on. `sdpa_paged_tiled` computes Q Kᵀ and P V as hand-walked dot
    /// products -- it was measured at 35.8% of a 2048-token gpt-oss prefill,
    /// running near 0.5 TFLOP/s while the quantized GEMM one dispatch away
    /// reaches ~5.6 on the same silicon. The arithmetic is a matmul; issuing it
    /// as one is what `sdpa_paged_mma.metal` is.
    ///
    /// This is a PREFILL switch. The predicate that earns the tiled shape at
    /// all (`sdpa_should_tile`) is unchanged and still keeps a fleet of decodes
    /// on the per-row kernel, where staging a tile per request would lose.
    ///
    /// It exists as a switch rather than a replacement because the matrix path
    /// depends on the register layout of `simdgroup_matrix<T,8,8>` -- see the
    /// header of that file -- and a machine whose layout differs would produce
    /// wrong numbers rather than slow ones. `PIE_METAL_SDPA_MMA=0` is the way
    /// back, and the greedy gate in `llama_bench` is what would catch it.
    bool sdpa_mma = true;

    /// Lanes that share one value row of the gated-delta scan.
    ///
    /// The scan is the only strictly sequential kernel in a prefill -- it walks
    /// the prompt token by token because the state it carries is the answer to
    /// the token before -- and it is the largest single item in a Qwen prefill:
    /// 59.4 ms for eighteen layers of a 32-token scan, which is a third of that
    /// fire and about half of a 128-token one, and it is the reason this family
    /// is the only one in this driver that loses a prefill to llama.cpp.
    ///
    /// The kernel is latency-bound on the two cross-lane reductions each token
    /// needs, not on arithmetic (7% of ALU peak) and not on bandwidth (staging
    /// q/k in threadgroup memory made it slower). That reads like an argument
    /// for narrowing the row: every halving costs one round of the xor tree and
    /// gives the simdgroup another independent row to interleave. It is wrong,
    /// and this constant exists to record that it is wrong. M1 Max,
    /// Qwen3.6-27B (Dk 128, Dv 128, 48 value heads), 128-token prefill tok/s:
    ///
    ///   lanes | xor rounds | floats/lane | tok/s
    ///   ------|------------|-------------|-------
    ///     16  |     4      |     24      |  92.4
    ///      8  |     3      |     48      |  85.6
    ///      4  |     2      |     96      |  92.6
    ///
    /// Eight is 7% SLOWER than sixteen and four only gets back to it. Removing
    /// a reduction round does not help because the latency it removes was
    /// already being hidden by occupancy, and the registers it costs take that
    /// occupancy away. The reductions are what the kernel WAITS on; they are
    /// not what it is short of. What it is short of is a formulation that does
    /// not walk one token at a time -- the chunked form of the delta rule,
    /// where a chunk's transitions are a matmul -- and no lane count reaches
    /// it.
    ///
    /// That conclusion was drawn from a sweep that only ever went DOWN. Going
    /// up is where the win is -- see `gdn_scan_rows`, whose table has 32 lanes
    /// beating 16 by 3.6% once the two knobs are swept together. A full
    /// simdgroup per dv row is the shortest q and k row a lane can read, and
    /// the extra xor round it costs is cheaper than the reads it removes.
    int gdn_scan_lanes = 32;

    /// Value rows one lane group of that scan walks, sharing the q and k it
    /// read for all of them.
    ///
    /// The table above says the reductions are not what the scan is short of.
    /// The arithmetic says what is: with LANES=16 a simdgroup covers two of Dv's
    /// 128 rows, so 64 of them per head each pull the head's whole q and k row
    /// every token, and one layer of a 128-token scan reads 402 MB of tensors
    /// that hold 6 MB. Narrowing LANES amortizes that and is the experiment
    /// above, which fails because `n_per_t` is Dk/LANES: halving the lanes
    /// doubles every register the scan carries. Walking more VALUE rows per
    /// lane group amortizes the same reads while q[] and k[] stay put.
    ///
    /// Same model and prompt, sweeping both axes:
    ///
    ///   lanes | rows | rows/simd | float/lane | tok/s
    ///   ------|------|-----------|------------|-------
    ///      8  |   1  |     4     |     16     |  89.3
    ///      8  |   2  |     8     |     32     |  94.6
    ///     16  |   1  |     2     |      8     |  99.4
    ///     16  |   2  |     4     |     16     | 100.9
    ///     16  |   4  |     8     |     32     |  97.0
    ///     32  |   2  |     2     |      8     | 103.8
    ///     32  |   4  |     4     |     16     | 104.5
    ///     32  |   8  |     8     |     32     | 102.7
    ///
    /// Read the table down the `float/lane` column and the two knobs separate
    /// cleanly. At a FIXED register cost -- (8,2), (16,4) and (32,8) all carry
    /// 32 floats a lane, as do (8,1), (16,2), (32,4) at 16 -- more lanes is
    /// always faster, because `Dk/LANES` is how much q and k a lane reads and
    /// nothing else changes. Along the other axis, 16 floats a lane beats both
    /// 8 and 32 at every lane count: too few rows and the reads are not
    /// amortized, too many and the occupancy that hides them is gone.
    ///
    /// So the earlier sweep's conclusion -- that the scan is not short of
    /// reduction rounds -- was right, and the direction it searched was wrong.
    /// It went down from 16 lanes, paying `Dk/LANES` registers to save an xor
    /// round, and lost. Going up pays an xor round to save registers AND
    /// traffic, and wins 3.6%.
    ///
    /// The win is still small next to the traffic removed: 32 lanes at 4 rows
    /// reads an eighth of what 16 lanes at 1 row does and is 5% faster. Two
    /// more experiments say why, and both failed:
    ///
    ///   * Software-pipelining the token loop -- read t+1's q, k, v and gates
    ///     into a second register set while t computes -- gives 96.7 tok/s at
    ///     every lane/row pair against 104.5. If the scan were waiting on
    ///     memory latency, issuing the loads an iteration early is exactly the
    ///     repair, and it is 8% WORSE. The registers cost more than the
    ///     latency they hide, which is another way of saying it was hidden.
    ///   * Staging q and k in threadgroup memory made bandwidth worse, not
    ///     better (recorded above).
    ///
    /// So the scan is not short of bandwidth, not short of reduction rounds,
    /// and not waiting on load latency. It is short of nothing an
    /// implementation of THIS recurrence can give it: 128 dependent steps at
    /// 7% of ALU peak is what a serial scan costs.
    ///
    /// Ablating the dispatch entirely -- wrong answers, right wall clock --
    /// puts the prefill at 118.1 tok/s against 104.5, so the scan is 11.5% of
    /// the fire and 11.5% is the budget any replacement has to beat. Only the
    /// chunked form, where a chunk's transition is a matmul and the token loop
    /// is gone, is a different implementation.
    int gdn_scan_rows = 4;

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
    ///
    /// THAT LAST PARAGRAPH IS THE COST, and on a serving engine it is large.
    /// It is true of a batch-one decode and stops being true of a FLEET, which
    /// is the shape Tier 3 measures and the shape a server runs. gpt-oss-20b is
    /// 32 experts at top-4, so `4 * n >= 32 * 4` first holds at exactly 32
    /// lanes -- and until it does, the largest weights in the model run as a
    /// per-row matvec. The throughput curve shows it as a step function
    /// (aggregate tok/s, 128-token prompt, M4 Pro 20-core):
    ///
    ///     lanes        8     16     24     32
    ///     default  156.0  178.6  176.2  460.7
    ///
    /// Swept on the same machine, same binary, via the env override below:
    ///
    ///     min_per      8      16      24
    ///         4    153.6   177.2   176.6
    ///         2    160.5   379.4   359.9
    ///         1    220.3   383.8   357.3
    ///
    /// At 1 gpt-oss passes every `llama_bench` gate, including the one that
    /// matters: an n-wide fleet reproduces mlx-lm's recorded continuation
    /// token for token.
    ///
    /// The blocker recorded here was that dropping it made gemma-4-26b-a4b's
    /// routed batched GEMM reachable at 16 lanes instead of 64, and that GEMM
    /// was believed wrong. It is not wrong. The evidence against it came from
    /// a fleet gate that compared the fleet to this driver's OWN one-row
    /// decode, which is a different kernel path, on a prompt whose
    /// continuation was ambiguous -- so the two greedy argmaxes parted company
    /// early and the gate read that as a defect. Forcing the routed GEMM onto
    /// every fire (`PIE_METAL_MOE_BATCH_MIN_PER_EXPERT=0`, so even a one-row
    /// decode batches) still reproduces mlx-lm's continuation exactly. The
    /// gate now checks a fleet against mlx-lm rather than against ourselves.
    ///
    /// The M1 Max agreed once it was asked. It had been left at the shipped 4
    /// because the M4 Pro's table was another machine's, and the sweep that
    /// finally ran here says the same thing louder -- aggregate tok/s,
    /// 128-token prompt, 64 decode steps, arms alternated:
    ///
    ///     model              lanes       4        1     delta
    ///     gpt-oss-20b            4   103.5    103.3       -0%
    ///     gpt-oss-20b            8   116.9    174.7      +49%
    ///     gpt-oss-20b           16   134.1    310.7     +132%
    ///     gemma-4-26b-a4b        8   128.9    130.3       +1%
    ///     gemma-4-26b-a4b       16   174.3    269.3      +55%
    ///
    /// The two neutral rows are neutral for a reason that is worth reading off
    /// the arithmetic rather than the table: `moe_should_batch` compares
    /// `n_pairs` to `n_experts * this`, and at 4 lanes gpt-oss routes 16 pairs
    /// over 32 experts while gemma-4-26b at 8 lanes routes 64 over 128. Both
    /// are below the threshold at ANY positive value, so neither arm batches
    /// and the rows are a control. Every row where the two arms actually
    /// choose differently is a large win, and none is a loss.
    int moe_batch_min_per_expert = 1;
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
///
/// `fp16_gemm` is the same rule, one level down. A crossover is where the GEMM
/// overtakes the GEMV, so it moves with the GEMM's speed, and on a device with
/// no bfloat matrix unit the GEMM's speed depends on whether the checkpoint
/// can reach the FP16 one: 4-bit at group 64 stages its input to half and uses
/// the instruction, and every other format keeps a sequence the compiler
/// writes to stand in for one. Measured on the M1 Max at eight lanes,
/// aggregate tok/s, GEMM against GEMV:
///
///     checkpoint             GEMV    GEMM   delta
///     Qwen3.6-27B (g64)      23.0    32.1    +40%
///     gemma-4-31b (g64)      17.9    30.0    +68%
///     Llama-1B (g128)       440.0   403.2     -8%
///
/// Same device, same batch, same code path, opposite signs -- so the constant
/// cannot be a property of the device alone. A checkpoint without the matrix
/// unit keeps the crossover measured for the GEMM it actually runs.
///
/// Both parameters, and neither defaulted, for the same reason: a caller that
/// could ask the question without answering it would get whichever answer the
/// signature happened to prefer.
int qmm_min_batch(bool is_moe, bool fp16_gemm);

/// Whether this checkpoint's quantization reaches the FP16 matrix path, which
/// is the `fp16_gemm` argument above. One function because four families ask
/// it and a fifth answer would be a fifth thing to keep in step.
bool fp16_gemm_format(int bits, int group);

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

/// Whether a tiled prefill attention runs on the simdgroup matrix unit.
bool sdpa_mma();
int gdn_scan_lanes();
int gdn_scan_rows();

/// The mixture's batch-vs-matvec crossover, in rows per expert.
int moe_batch_min_per_expert();

}  // namespace pie::metal
