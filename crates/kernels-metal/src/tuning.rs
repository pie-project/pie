//! Per-device tuning constants — every crossover this plane decides, and the
//! measurement that decided it.
//!
//! # The rule this module exists to keep
//!
//! A DEFAULT-CONSTRUCTED [`DeviceTuning`] reproduces the M1 Max numbers
//! EXACTLY. Adding a device may never change what an unrecognised one does,
//! and a machine this file has never heard of gets the constants that were
//! MEASURED rather than an extrapolation from the ones that were. Every
//! per-family override below carries the run that justifies it, and a family
//! with no entry inherits the default because nothing has measured it apart —
//! not because the default is thought to generalize.
//!
//! # No environment variables
//!
//! The reference driver swept these with a `PIE_METAL_*` override apiece, and
//! that mechanism does not come across: a shell here reads no environment
//! (constitution art. 9). What replaces it is [`Overrides`] — typed data, one
//! `Option` per field, filled from the boot document's own `[metal.tuning]`
//! table by `engine_metal::boot`. The sweep property that mattered is
//! preserved (the same binary, two arms, no rebuild between them); what is
//! gone is the ambient channel.
//!
//! The reference also recorded, at length, that a knob which silently reports
//! its default when you set it is worse than no knob — it spent two false
//! conclusions on an integer parser that folded `0` in with "not a number",
//! so `moe_batch_min_per_expert = 0` never once took the path the experiment
//! claimed to be testing. An `Option<u32>` cannot make that mistake: the
//! absence of a value and the value zero are different inhabitants.
//!
//! # How it reaches a call site
//!
//! One process-wide cell ([`install`], [`current`]), the way the reference
//! held one function-local `static`. The alternative — threading a
//! `&DeviceTuning` through every entry function — would put a parameter on
//! forty signatures to carry a value that is constant for the life of the
//! process, and `kernels-metal` names no device to hang it off instead.

use std::sync::OnceLock;

/// What this plane knows about the GPU it is running on.
///
/// Queried once by the shell that binds the device and installed here; every
/// field has a value on every path, including the one where nothing was
/// asked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `MTLGPUFamilyApple<N>`. **PROBE IT NEWEST-FIRST**: the families are
    /// cumulative, so an M4 answers `supportsFamily:` for Apple7 as well as
    /// Apple9, and an oldest-first probe would report every Apple Silicon GPU
    /// ever made as an Apple7 and hand all of them the M1 constants — a bug
    /// that looks exactly like this module not existing. 0 when nothing
    /// answered, which selects the defaults.
    pub apple_family: u32,

    /// GPU cores. Recorded, and read by nothing here: the crossovers below
    /// are set by per-core matrix throughput, which the FAMILY names and the
    /// count does not — see [`DeviceTuning::qmm_min_batch`]'s Apple9 entry.
    /// It is carried because the next constant to be measured may want it.
    /// 0 when absent.
    pub gpu_core_count: u32,
}

impl DeviceInfo {
    /// The family a `MTLDevice`'s own name implies, for a shell that has a
    /// name and no `supportsFamily:` probe.
    ///
    /// **A NAME IS A WEAKER ANSWER THAN THE PROBE AND IS NOT A SUBSTITUTE FOR
    /// IT.** It cannot report the core count at all — Metal does not publish
    /// one and IOKit's `gpu-core-count` is the only place it lives — and it
    /// answers 0 for any silicon minted after this table was written, which
    /// is the same answer a failed probe gives and lands on the same
    /// measured defaults. What it does get right is the axis the table
    /// actually branches on.
    #[must_use]
    pub fn of_name(name: &str) -> Self {
        let family = if name.contains("M1") {
            7
        } else if name.contains("M2") {
            8
        } else if name.contains("M3") || name.contains("M4") {
            9
        } else {
            0
        };
        Self {
            apple_family: family,
            gpu_core_count: 0,
        }
    }
}

/// The tuned constants, defaulted to the M1 Max measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceTuning {
    /// The batch at which the tiled GEMM overtakes the batched GEMV, for a
    /// checkpoint whose GEMM reaches the FP16 matrix path. See
    /// [`qmm_min_batch_emulated`](Self::qmm_min_batch_emulated) for the one
    /// that does not.
    ///
    /// M1 Max: 5.
    ///
    /// # IT IS ASKED OF THE FIRE'S ROWS, WHICH IS WHAT THE SWEEP MEASURED
    ///
    /// `linear::quant::act_x_wt` reads this against the row count the fire
    /// brought, so the table below is read exactly as it was taken: at width
    /// `N`, which arm is faster. A fire's rows are the COMPOSITION's, so the
    /// arm — and a lane's low-order bits with it — moves with the neighbours.
    /// That is the owner's ruling and not an oversight:
    ///
    /// > We do NOT need bit-level identity. If a much faster path has small
    /// > numerical drift from nondeterminism, that is obviously acceptable.
    ///
    /// A release read it against the SLOT's row capacity instead, to make the
    /// pick a load-time constant. It bought bit-stability across compositions
    /// and cost 76% of a one-lane decode on the small vehicle and 3.2x on the
    /// giant, because a shell wide enough to cross took the tile's eight-row
    /// floor at every width including one. That trade is the one the ruling
    /// reverses.
    ///
    /// **THE CURVE IS THE THING TO RE-SWEEP** when the GEMM moves — a wrong
    /// value here puts whole rungs on the slower arm.
    ///
    /// **WHAT THIS CROSSOVER ACTUALLY SEPARATES**, because the two arms are
    /// not two speeds of one kernel. `quant_qmv.metal`'s fast point walks one
    /// threadgroup per ROW (`x += tid.x * in_vec_size`), so an N-row fire
    /// reads the whole weight table N times; the tiled point reads it ONCE
    /// and does `BM` rows of arithmetic against it, at the narrowest rung
    /// `BM = 8` whether the fire brought eight rows or two. So the GEMV
    /// climbs with N and the GEMM is FLAT in it, and the crossover is where
    /// one line cuts the other — measured on gemma-4-31b, the GEMM arm moves
    /// 222.1 → 260.6 ms/fire from two lanes to sixteen while the GEMV arm
    /// moves 106.4 → 801.4.
    ///
    /// This read 12 while the batched GEMM was still emulating a bfloat
    /// matrix unit, then 8 once it stopped — arms alternated twice, eight
    /// lanes, 128-token prompt, 64 decode steps, aggregate tok/s: Qwen3.6-27B
    /// 23.0 → 32.1 (+40%), gemma-4-31b 17.9 → 30.0 (+68%), gemma-4-26b-a4b
    /// 107.3 → 130.1 (+21%), gpt-oss-20b 108.2 → 116.2 (+7%). What moved then
    /// was not the GEMM's speed but which side of the crossover eight sat on.
    ///
    /// **THE CROSSOVER IS A PROPERTY OF THE GEMM AND MOVES WHEN THE GEMM
    /// DOES**, of which the paragraph above is one instance and this is the
    /// next two: 8 → 6 when four checkpoints were finally swept a width at a
    /// time, and 6 → 5 now that [`crate::linear::quant`]'s `BM_RUNGS` has an
    /// 8 rung under its 16. A narrower row block is a cheaper GEMM at every
    /// width below sixteen, so the flat line moved DOWN and the crossing
    /// point moved left with it. Nothing about the GEMV changed either time.
    ///
    /// `throughput_probe` sweeps a rung at a time and can pin either arm
    /// through the boot document, so both curves are available at every
    /// width: ms/fire over 32 warm decode fires, the GEMV arm forced with
    /// `qmm_min_batch = 999` and the GEMM arm with `= 2`, lower is better,
    /// winner starred:
    ///
    /// ```text
    ///        qwen36-27b        gemma4-31b       gpt-oss-20b      qwen35-0.8b
    ///   N    GEMV    GEMM     GEMV    GEMM     GEMV   GEMM      GEMV   GEMM
    ///   2  *107.4*  206.2   *106.4*  222.1   *19.1*  24.8      *7.1*  10.2
    ///   3  *159.3*  218.9   *156.1*  223.0   *27.0*  30.6     *10.6*  12.7
    ///   4  *207.7*  228.0   *205.5*  223.6   *34.8*  36.4     *12.3*  12.8
    ///   5   258.7 *239.4*    254.7 *224.3*    42.7 *42.1*      16.8 *16.0*
    ///   6   304.8 *246.5*    304.0 *225.0*    50.5 *47.9*      18.3 *16.1*
    ///   7   357.2 *259.4*    354.1 *226.1*    58.4 *53.9*      21.9 *18.2*
    ///   8   405.8 *266.8*    403.8 *226.8*    66.3 *59.6*      23.3 *18.2*
    /// ```
    ///
    /// Five is the first width where the GEMM wins on ALL FOUR (+8.0, +13.6,
    /// +1.3, +4.6%) and four is the last where it loses on any — at four it
    /// loses on all four, by 9.7 / 8.8 / 4.7 / 4.0%, so unlike the sweep that
    /// set six there is no borderline here to argue about in either
    /// direction. The value is the one with no measured regression anywhere,
    /// which is the rule the M2 sweep below applied when it chose eight over
    /// seven.
    ///
    /// 6 → 5 is worth exactly the five-lane column, +8.0 / +13.6 / +1.3 /
    /// +4.6%: at every other width both values select the same kernel and
    /// measure the same. What the 8 rung is worth INSIDE the GEMM arm — the
    /// other half of the same sweep, and larger — is tabulated on
    /// `linear::quant::BM_RUNGS` rather than restated here.
    ///
    /// **AND IT IS THE MACHINE'S NUMBER AND NOT THE SHAPE'S**, which is worth
    /// writing down because the sweeps that set 8 ran on 1-3B checkpoints and
    /// the ones above run on four spanning 0.8B to 31B, hidden widths 2048 to
    /// 5120. All four cross between five and six. The arithmetic says why: the
    /// GEMV reads `bytes` per row and the GEMM reads `bytes` once for `BM`
    /// rows of arithmetic, so the crossover is a ratio of the device's
    /// bandwidth to its matrix throughput and the weight table's SIZE cancels.
    /// There is therefore no width-keyed rule here to add.
    ///
    /// M2 Max (Apple8, 38 cores) and M4 Pro (Apple9, 20 cores) both name 8 in
    /// [`of`](Self::of) rather than inheriting, and they keep naming it
    /// through this move: their sweeps ran on the 16-rung GEMM, and a machine
    /// whose curve nobody has re-drawn since the 8 rung landed gets the
    /// number that was MEASURED on it. The M2's is the first batch
    /// where the GEMM won on all four dense checkpoints it swept (Llama-1B
    /// +17.6%, Llama-3B +19.2%, Qwen3-1.7B +14.4%, gemma-4-E2B +4.6%) and the
    /// M4's is gemma-4-E4B at concurrency 8, 138.90 against 144.04 tok/s
    /// (+3.7%). Neither swept the widths between four and eight, and a
    /// default that moved under them would hand both machines a number no run
    /// on them has ever produced.
    ///
    /// DENSE only: see [`qmm_min_batch_moe`](Self::qmm_min_batch_moe).
    pub qmm_min_batch: u32,

    /// The same crossover for a checkpoint whose FFN is ROUTED.
    ///
    /// A separate number because the measurement says so, not for symmetry.
    /// In a mixture the dense projections this governs are the attention four
    /// and the head; the FFN — the largest weights in the layer — is routed
    /// and takes [`moe_tile_mid_per`](Self::moe_tile_mid_per)'s decision
    /// instead. So the GEMM here pays its padding without the matrices that
    /// repay it.
    ///
    /// M1 Max: 8, for the same reason the dense number moved — with the
    /// routed expert GEMM on FP16 the sweep reverses (gemma-4-26b-a4b
    /// 107.3 → 130.1, gpt-oss-20b 108.2 → 116.2, eight lanes). The M2 Max
    /// keeps 12: at the batches where the dense value would have switched the
    /// GEMV won or tied on every mixture measured — Qwen3-30B by 8%,
    /// gemma-4-26B by 12%, gpt-oss-20B by nothing either way — and nothing
    /// has re-run that machine since the FP16 wiring.
    ///
    /// **NO CALL SITE IN THIS TREE REACHES IT AND SO NOTHING RE-SWEPT IT**
    /// when [`qmm_min_batch`](Self::qmm_min_batch) moved to six, or again
    /// when it moved to five.
    /// `linear::quant::act_x_wt` is the only reader of
    /// [`qmm_min_batch`](Self::qmm_min_batch(routed:fp16_gemm:)) and it passes
    /// `routed: false` unconditionally, for the reason stated at that call:
    /// whether a checkpoint's FFN is routed is a fact about the MODEL, and no
    /// operand of `linear.matmul` carries it. So gpt-oss-20b's attention four
    /// and its head took the DENSE number in the sweep above and take it now,
    /// which is why that column is in that table and not this one.
    pub qmm_min_batch_moe: u32,

    /// The same crossover for a checkpoint whose quantization does NOT reach
    /// the FP16 matrix path — anything but 4-bit at group 64.
    ///
    /// Twelve, and it is the same measurement rather than a leftover: on the
    /// M1 Max at eight lanes Llama-1B at group 128 turns in 440.5 and 439.4
    /// tok/s on the GEMV against 403.9 and 402.5 on the GEMM, arms
    /// alternated, −8.4%. The GEMM it runs is the one the old sweep measured,
    /// so it keeps the old sweep's answer.
    ///
    /// One number for dense and routed both, because nothing has measured
    /// them apart and inventing a split would be a guess with a field to live
    /// in.
    pub qmm_min_batch_emulated: u32,

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile overtakes
    /// BN=16.
    ///
    /// M1 Max: 160. Measured over fifteen (M, N) pairs and checked against a
    /// real llama-3.2-1B prefill, which agrees — 2565.8 / 2663.7 / 2578.3
    /// tok/s at BN 16/32/64, 448 rows. The sweep brackets the crossover
    /// between 144 threadgroups and 192, so this sits in the gap.
    ///
    /// A machine with more cores saturates later and wants this HIGHER, since
    /// the narrow tile's only advantage is that it makes more threadgroups.
    /// The M4 Pro reads that sentence backwards: twenty cores against thirty-
    /// two fill sooner, so the narrow tile runs out of reasons sooner, and
    /// the re-sweep puts the bracket at (64, 96].
    pub qmm_bn_crossover_tg: u32,

    /// Rows an expert's run must hold before the mixture's GEMM takes the
    /// 32-row tile, and then the 64-row one.
    ///
    /// M1 Max: 32, and 64 never. These were 12 and 88, and they moved for the
    /// reason [`qmm_min_batch`](Self::qmm_min_batch) did — a tile's
    /// ARITHMETIC got ~40% cheaper on the FP16 matrix unit and the rows it
    /// pads did not, so the padding a wide tile wastes buys less than it used
    /// to. Re-swept, prefill tok/s, best starred:
    ///
    /// ```text
    ///  per   model     rows    BM=16    BM=32    BM=64
    ///   12   26b        192   *642.4*   616.0       --
    ///   16   gptoss     128   *594.8*   551.8       --
    ///   24   gptoss     192   *642.3*   622.9       --
    ///   32   gptoss     256     654.2  *664.2*      --
    ///   64   gptoss     512        --  *721.4*      --
    ///  128   gptoss    1024        --  *725.1*   716.6
    ///  256   gptoss    2048        --  *656.1*   654.0
    /// ```
    ///
    /// So the mid threshold moves 12 → 32 and the wide one has nowhere left
    /// to fire: 64 loses at 128 rows an expert and ties at 256, already past
    /// any batch this plane sees. It is set OUT OF REACH rather than deleted,
    /// because the field is what a device with different matrix throughput
    /// would answer differently and no other device has been re-measured
    /// since the routed GEMM changed underneath it.
    ///
    /// Measured end to end on purpose: a roofline probe predicts the dense
    /// tile correctly and this one wrong three times over, because its single
    /// hot expert cannot show what thirty-two cold ones cost.
    pub moe_tile_mid_per: u32,

    /// The 64-row rung's threshold — see
    /// [`moe_tile_mid_per`](Self::moe_tile_mid_per). Out of reach on the M1
    /// Max, which is the finding and not an omission.
    pub moe_tile_wide_per: u32,

    /// Whether a g64/b4 projection stages its input to FP16 and feeds native
    /// FP16 simdgroup MMA instead of BF16.
    ///
    /// M1 Max: on, and it is the largest single win recorded — roughly 40% on
    /// the GEMM at every shape measured, which on gemma-4 was 938 → 1298
    /// tok/s of prefill.
    ///
    /// It exists BECAUSE of the machine. M1 and M2 have no native bfloat16
    /// matrix path and emulate it; Metal 3.1 and Apple9 (M3, M4) do have one,
    /// and on those the staging pass has nothing left to buy and is a
    /// dispatch, a barrier and a buffer per projection. So this is the one
    /// tuned field whose default may be actively WRONG on newer silicon
    /// rather than merely unmeasured, and the first thing to check on an M3
    /// or an M4.
    pub fp16_qmm: bool,

    /// Rows a request must contribute before its attention takes the tiled
    /// kernel instead of the per-row one.
    ///
    /// M1 Max: 32, which is also the tile's height — a fire earns the tiled
    /// shape by filling a tile. The two are separate numbers even so: the
    /// height is the simdgroup count and cannot move, while this is a
    /// crossover. Measured on llama-1B, where a 32-request fleet of one-row
    /// members runs 370 tok/s tiled against 728 per-row (and at 64, 480
    /// against 915), so the threshold has to keep the fleet off it. A machine
    /// whose shuffles are cheaper relative to its FMAs wants it HIGHER, since
    /// the tiled kernel's whole advantage is that it removes reductions.
    pub sdpa_tile_min_rows_per_request: u32,

    /// Whether a tiled prefill attention runs on the simdgroup MATRIX unit
    /// rather than the scalar path.
    ///
    /// M1 Max: on. The scalar tiled kernel computes Q·Kᵀ and P·V as
    /// hand-walked dot products — measured at 35.8% of a 2048-token gpt-oss
    /// prefill, running near 0.5 TFLOP/s while the quantized GEMM one
    /// dispatch away reaches ~5.6 on the same silicon. The arithmetic is a
    /// matmul; issuing it as one is what `sdpa_paged_mma.metal` is.
    ///
    /// A PREFILL switch. The predicate that earns the tiled shape at all
    /// ([`sdpa_tile_min_rows_per_request`](Self::sdpa_tile_min_rows_per_request))
    /// is unchanged and still keeps a fleet of decodes on the per-row kernel.
    ///
    /// It is a switch rather than a replacement because the matrix path
    /// depends on the register layout of `simdgroup_matrix<T,8,8>`, and a
    /// machine whose layout differs would produce WRONG numbers rather than
    /// slow ones. Turning it off is the way back.
    pub sdpa_mma: bool,

    /// Lanes that share one value row of the gated-delta scan.
    ///
    /// M1 Max: 32. The scan is latency-bound on two cross-lane reductions per
    /// token, not on arithmetic (7% of ALU peak) and not on bandwidth
    /// (staging q/k in threadgroup memory made it slower) — which reads like
    /// an argument for NARROWING the row, and is wrong. Sweeping down from 16
    /// lanes: 16 → 92.4 tok/s, 8 → 85.6, 4 → 92.6. Removing a reduction round
    /// does not help because the latency it removes was already hidden by
    /// occupancy, and the registers it costs take that occupancy away.
    ///
    /// Going UP is where the win is: a full simdgroup per dv row is the
    /// shortest q and k row a lane can read, and the extra xor round it costs
    /// is cheaper than the reads it removes — 3.6% once swept together with
    /// [`gdn_scan_rows`](Self::gdn_scan_rows).
    pub gdn_scan_lanes: u32,

    /// Value rows one lane group of that scan walks, sharing the q and k it
    /// read for all of them.
    ///
    /// M1 Max: 4. Qwen3.6-27B, 128-token prefill, both axes swept:
    ///
    /// ```text
    ///  lanes | rows | float/lane | tok/s
    ///  ------|------|------------|-------
    ///     8  |   1  |     16     |  89.3
    ///     8  |   2  |     32     |  94.6
    ///    16  |   1  |      8     |  99.4
    ///    16  |   2  |     16     | 100.9
    ///    16  |   4  |     32     |  97.0
    ///    32  |   2  |      8     | 103.8
    ///    32  |   4  |     16     | 104.5
    ///    32  |   8  |     32     | 102.7
    /// ```
    ///
    /// Read down the `float/lane` column and the two knobs separate cleanly:
    /// at a fixed register cost more lanes is always faster, because
    /// `Dk/LANES` is how much q and k a lane reads; along the other axis 16
    /// floats a lane beats both 8 and 32 at every lane count, because too few
    /// rows do not amortize the reads and too many spend the occupancy that
    /// hides them.
    ///
    /// # THESE TWO REACHED NO LAUNCH AT ALL UNTIL `ssm_gdn_scan.metal`
    ///
    /// The table above is the reference's, taken on the reference's kernel,
    /// and for a while this tree carried it beside a scan that read neither
    /// field: `attn::ssm` launched `gated_delta_chunked` at a hardcoded
    /// `[128, v_heads, requests]` and never called `current()`. A knob wired
    /// to nothing is the exact failure this module's own header spends four
    /// paragraphs on, and what it cost was not the tuning — it was that
    /// nobody was looking at the kernel the constants described.
    ///
    /// `ssm_gdn_scan.metal` is that kernel's shape, ported onto this tree's
    /// operands, and [`crate::attn::ssm::gdn_scan_launch`] is where the two
    /// fields now compose an entry name and a grid. **ZERO ROWS IS A VALUE**
    /// and it is the control: no fold is stamped at zero, so the selection
    /// declines and `gated_delta_chunked` runs. Standalone, qwen3.6-27B's
    /// gated-delta shape, ms per layer
    /// (`what_the_gated_delta_scan_costs`):
    ///
    /// ```text
    ///   tokens          128     256     512
    ///   threadgroup    33.70   67.24  134.36
    ///   register        0.93    1.81    3.55
    /// ```
    ///
    /// Thirty-eight times, and it is a shape difference rather than a tuning
    /// one: 384 threadgroups against 48, the recurrent cell in registers
    /// rather than read-modify-written twice a token in device memory, and
    /// `simd_shuffle_xor` rather than ten threadgroup barriers a token. On a
    /// whole 512-token prefill (`throughput_probe`), tok/s:
    ///
    /// ```text
    ///                  rows = 0   rows = 4
    ///   qwen36-27b        48.0      116.9
    ///   qwen35-0.8b      385.9     2921.3
    /// ```
    ///
    /// **THE SWEEP ABOVE HAS NOT BEEN RE-RUN ON THIS KERNEL.** `(32, 4)` is
    /// taken on the reference's authority and on the shape argument being the
    /// same one; what has been measured here is `(32, 4)` against not
    /// folding, which is the comparison that was worth 2.4x. A re-sweep is
    /// the obvious next measurement and it is worth at most the few percent
    /// the reference's own table spans.
    pub gdn_scan_rows: u32,

    /// Rows an expert's run must hold before the mixture sorts and batches at
    /// all, rather than running the routed projections as matvecs.
    ///
    /// M1 Max: 2 — the sorted arm from `2 · experts` pairs up, which for
    /// gpt-oss-20b's 32 experts at top-4 is sixteen lanes.
    ///
    /// It was 8 — half a narrow tile — on the model that the padding is
    /// wasted work and the break-even is where a run half fills a tile. That
    /// model is wrong in the direction that matters: a 4-bit mixture is
    /// bandwidth-bound, and what batching buys is reading each expert's slice
    /// ONCE instead of once per pair, which is worth far more than the
    /// arithmetic a half-empty tile throws away. It then read 1 — "batch from
    /// one pair an expert up" — off an aggregate sweep whose window carried a
    /// 128-token prefill per lane beside its decodes.
    ///
    /// **THE STEP IS ON THE DECODE RUNGS, SO IT IS MEASURED THERE.**
    /// `throughput_probe` seats N lanes, fires a warm window of decodes alone,
    /// and PRINTS which arm each rung took, so the two arms can be pinned
    /// through the boot document and read off against each other rung by rung.
    /// gpt-oss-20b, ms/fire over 32 warm decode fires, the per-row arm forced
    /// with `moe_batch_min_per_expert = 3` and the sorted arm with `= 0`,
    /// lower is better, winner starred:
    ///
    /// ```text
    ///  lanes   pairs   per-row    sorted    delta
    ///      1       4    *11.4*      20.1     +77%
    ///      2       8    *19.1*      31.1     +63%
    ///      4      16    *34.6*      49.1     +42%
    ///      8      32    *61.5*      71.5     +16%
    ///     16      64     108.2   *98.5*      +10%
    /// ```
    ///
    /// The lines cross between 32 pairs and 64, so the threshold is `2 ·
    /// experts` and not `1 ·` — and 1 is a MEASURED regression rather than a
    /// neutral one, because 32 pairs is exactly where it switches the arm:
    /// 16% of a gpt-oss decode at eight lanes. Alternated in fresh processes
    /// at that rung alone, twice each: sorted 76.63 and 76.63, per-row 66.36
    /// and 66.37.
    ///
    /// **AND IT IS ONE MIXTURE'S SHAPE.** The crossover is stated in pairs an
    /// expert, so a checkpoint routed differently — more experts, a wider
    /// top-k, a different expert width — moves the LANE count this lands on
    /// without necessarily moving the pair count, and no other mixture has
    /// been swept on this machine since the routed GEMM took the FP16 matrix
    /// path. gemma-4-26b-a4b in particular owes its own table. What the
    /// earlier aggregate sweep recorded stands as a measurement of the window
    /// it took: 128 prompt tokens a lane is 512 pairs a lane at prefill, past
    /// every threshold either arm names, so the arm it separated was the
    /// decode one and the number it reported was diluted by a prefill both
    /// arms sorted.
    ///
    /// ZERO IS A VALUE, not a typo: "batch at any width" is the only way to
    /// reach the routed GEMM below its crossover, which is how a wrong-answer
    /// bug in that kernel gets bisected — and it is how the sorted column
    /// above was taken.
    pub moe_batch_min_per_expert: u32,

    /// **THE WIDEST ROW GROUP THE VECTOR POINT FOLDS INTO ONE WEIGHT FETCH.**
    ///
    /// M1 Max: 2. **ONE DISABLES THE ARM** and restores `quant_qmv.metal`'s
    /// one-row point at every width, which is the control every number below
    /// was taken against.
    ///
    /// # What the fold is, and what it is not
    ///
    /// `quant_qmv.metal`'s point walks one threadgroup per ROW, so an N-row
    /// fire fetches the whole bank N times; `quant_qmv_rows.metal` fetches it
    /// once per group of R. It was built to collapse the two-to-four-lane
    /// band toward the cost of one lane, and **it does not, because the fetch
    /// was never the bill.** `what_the_vector_point_is_bound_by` fires both
    /// points over a 72 MiB bank and then over a 4 MiB slice of it, so the
    /// second run's weights are in the 48 MiB system cache and its reads are
    /// nearly free; per row, scaled to the same bank, ms:
    ///
    /// ```text
    ///   rows            1      2      4      8     16
    ///   one-row DRAM  0.359  0.215  0.211  0.208  0.207
    ///   one-row cache 0.472  0.321  0.266  0.237  0.223
    /// ```
    ///
    /// Taking the memory away does not make it faster. The vector point is
    /// ARITHMETIC-bound — a mask, an integer convert and an FMA per four-bit
    /// code — and 365 GB/s at sixteen rows is a coincidence of this machine's
    /// balance rather than the ceiling it is running into.
    ///
    /// # What the fold IS worth
    ///
    /// The load INSTRUCTIONS and the address and scale arithmetic beside
    /// them: one pack fetch, one scale, one zero point per two rows instead
    /// of per row. `throughput_probe`, production ladder, aggregate tok/s,
    /// `qmv_rows_max` 1 against 2:
    ///
    /// ```text
    ///                       N=1    N=2    N=3    N=4    N=5..8
    ///   qwen36-27b  off     16.1   18.6   18.8   19.3   unchanged
    ///               on      16.1   20.5   18.8   21.5   unchanged
    ///   gemma4-31b  off     16.9   18.9   19.3   19.5   unchanged
    ///               on      16.9   22.0   19.3   23.3   unchanged
    /// ```
    ///
    /// N=1 and N=3 are the CONTROLS — neither folds, and both reproduce to
    /// the third digit.
    ///
    /// # WHY TWO AND NOT FOUR, WHICH IS THE PART THAT COST A SWEEP
    ///
    /// A wider fold holds more accumulators and fewer threadgroups, and both
    /// go the wrong way. On the giant, ms/fire at four lanes: one-row 207.70,
    /// R=2 **185.91**, R=4 229.02. R=4 is WORSE than not folding at all. The
    /// standalone bench agrees and says why — per row over the 72 MiB bank,
    /// R=2 runs 0.177 ms and R=4 runs 0.217 against the one-row 0.207 — so
    /// the arithmetic the fold cannot remove starts costing more, in
    /// occupancy, than the instructions it does remove.
    ///
    /// The 8 rung is stamped and is on the same slope (0.245 ms/row): it is
    /// kept reachable so the next machine can be asked, not because this one
    /// wants it.
    ///
    /// A machine with a wider register file or a cheaper unpack wants this
    /// HIGHER; a machine that fills on fewer threadgroups wants it higher
    /// too. Neither has been measured, so no family names a value.
    pub qmv_rows_max: u32,

    /// Weight packs one thread of the multi-row vector point reads per k
    /// step.
    ///
    /// M1 Max: 1, against the one-row point's 2. A thread of
    /// `quant_qmv_rows.metal` holds four output rows' packs and their factors
    /// live across the row loop, so the pack width multiplies the part of the
    /// live set the fold added. Over the 72 MiB bank at two rows, per row:
    /// **0.177 ms at one pack against 0.207 at two** — which is the whole of
    /// the fold's win, spent and recovered on one constant.
    ///
    /// It is a knob and not a constant in the shader because which side wins
    /// is a property of the machine's register file rather than of the
    /// checkpoint.
    pub qmv_rows_packs: u32,
}

impl Default for DeviceTuning {
    fn default() -> Self {
        Self {
            qmm_min_batch: 5,
            qmm_min_batch_moe: 8,
            qmm_min_batch_emulated: 12,
            qmm_bn_crossover_tg: 160,
            moe_tile_mid_per: 32,
            moe_tile_wide_per: 1 << 24,
            fp16_qmm: true,
            sdpa_tile_min_rows_per_request: 32,
            sdpa_mma: true,
            gdn_scan_lanes: 32,
            gdn_scan_rows: 4,
            moe_batch_min_per_expert: 2,
            qmv_rows_max: 2,
            qmv_rows_packs: 1,
        }
    }
}

impl DeviceTuning {
    /// The table, read at one device.
    ///
    /// A family with no arm inherits the defaults, which are the M1 Max's
    /// measurements — see the module header for why that is the rule rather
    /// than an accident.
    #[must_use]
    pub fn of(info: DeviceInfo) -> Self {
        let mut t = Self::default();
        match info.apple_family {
            // M3/M4. The DENSE crossover is eight here and is named rather
            // than inherited, because the default moved UNDER it: the M1's
            // own re-sweep found the lines cross at six, and it found that by
            // measuring the widths between four and eight, which no run on
            // THIS machine has ever sampled. What the M4 measured is
            // gemma-4-E4B at concurrency 8 — 12 against 8, 138.90 against
            // 144.04 tok/s, +3.7% — and that number says nothing about six.
            //
            // The tile crossover is the other entry: with twenty cores rather
            // than thirty-two the wide tile's smaller grid fills the machine
            // sooner, so it moves DOWN. Re-swept at BM=64,
            // GFLOP/s, threadgroups counted as `(N/32)·ceil(M/64)`:
            //
            //   tg@32    BN=16    BN=32    delta
            //      64     5004     4615    -7.8%
            //      96     5540     5863    +5.8%
            //     128     5829     5846    +0.3%
            //     160     5987     6411    +7.1%
            //     224     5980     6196    +3.6%
            //
            // The bracket is (64, 96] and 96 is the sampled edge rather than
            // the midpoint, which is what to write down when only one side
            // was measured. Applied by FAMILY and not by core count: the
            // crossover is set by per-core matrix throughput, which the
            // family names and the count does not.
            9 => {
                t.qmm_min_batch = 8;
                t.qmm_bn_crossover_tg = 96;
            }
            // M2. The DENSE crossover here is eight — the first batch where
            // the GEMM won on all four dense checkpoints this machine swept
            // (Llama-1B +17.6%, Llama-3B +19.2%, Qwen3-1.7B +14.4%,
            // gemma-4-E2B +4.6%) and the last where it lost on any is seven.
            // That sweep DID sample six, and at six the GEMV won all four
            // (386.9/393.0, 166.7/167.6, 270.9/283.8, 173.9/197.9 tok/s,
            // GEMM/GEMV) — so this machine's own evidence puts its crossover
            // where it already reads it, and the M1 moving to six is a
            // statement about the M1.
            //
            // The ROUTED one is twelve, and that is the finding rather than
            // an omission — at the same batches the GEMV still won on every
            // mixture measured. It is named HERE rather than inherited
            // because the default moved UNDER it too: the M1 reversed its own
            // routed number once the expert GEMM stopped emulating a bfloat
            // matrix unit, this machine has not been re-measured since, and
            // inheriting a default that changed for a reason this device was
            // never tested against would be a guess wearing a measurement's
            // clothes.
            8 => {
                t.qmm_min_batch = 8;
                t.qmm_min_batch_moe = 12;
            }
            _ => {}
        }
        t
    }

    /// The same table with a boot document's answers laid over it.
    #[must_use]
    pub fn with(mut self, over: &Overrides) -> Self {
        macro_rules! lay {
            ($($field:ident),+ $(,)?) => {
                $(if let Some(v) = over.$field { self.$field = v; })+
            };
        }
        lay!(
            qmm_min_batch,
            qmm_min_batch_moe,
            qmm_min_batch_emulated,
            qmm_bn_crossover_tg,
            moe_tile_mid_per,
            moe_tile_wide_per,
            fp16_qmm,
            sdpa_tile_min_rows_per_request,
            sdpa_mma,
            gdn_scan_lanes,
            gdn_scan_rows,
            moe_batch_min_per_expert,
            qmv_rows_max,
            qmv_rows_packs,
        );
        self
    }

    /// The GEMV/GEMM crossover for one projection, given whether the
    /// checkpoint's FFN is routed and whether this weight's format reaches
    /// the FP16 matrix path.
    #[must_use]
    pub const fn qmm_min_batch(&self, routed: bool, fp16_gemm: bool) -> u32 {
        if !fp16_gemm {
            return self.qmm_min_batch_emulated;
        }
        if routed {
            self.qmm_min_batch_moe
        } else {
            self.qmm_min_batch
        }
    }

    /// Whether a bank of this format reaches the FP16 matrix path — the
    /// staged-input GEMM is stamped at 4 bits, group 64, and nowhere else.
    #[must_use]
    pub const fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        self.fp16_qmm && bits == 4 && group == 64
    }
}

/// One boot document's answers: `None` is "keep what the device's table
/// said", and every value — zero included — is a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overrides {
    pub qmm_min_batch: Option<u32>,
    pub qmm_min_batch_moe: Option<u32>,
    pub qmm_min_batch_emulated: Option<u32>,
    pub qmm_bn_crossover_tg: Option<u32>,
    pub moe_tile_mid_per: Option<u32>,
    pub moe_tile_wide_per: Option<u32>,
    pub fp16_qmm: Option<bool>,
    pub sdpa_tile_min_rows_per_request: Option<u32>,
    pub sdpa_mma: Option<bool>,
    pub gdn_scan_lanes: Option<u32>,
    pub gdn_scan_rows: Option<u32>,
    pub moe_batch_min_per_expert: Option<u32>,
    pub qmv_rows_max: Option<u32>,
    pub qmv_rows_packs: Option<u32>,
}

static DEVICE: OnceLock<DeviceInfo> = OnceLock::new();
static OVERRIDES: OnceLock<Overrides> = OnceLock::new();
static RESOLVED: OnceLock<DeviceTuning> = OnceLock::new();

/// Say what device this process is running on — the shell that binds it,
/// once.
///
/// **THE TWO INPUTS ARRIVE FROM DIFFERENT PLACES AND IN NO FIXED ORDER**, and
/// that is why they are seated separately rather than combined by the caller:
/// the boot document is read at the door and the device is bound at load, so
/// whichever arrives first would otherwise decide what the other could still
/// change. They are folded at the first [`current`] instead, and the answer is
/// frozen from then on — a table that could move under a fire in flight would
/// let two dispatches of one step disagree about which kernel they are.
///
/// Answers whether this call is the one that seated it. Not calling it at all
/// is a supported state and lands on the M1 Max measurements, which is what
/// [`DeviceTuning::of`] answers for a device it does not recognize.
pub fn describe(info: DeviceInfo) -> bool {
    DEVICE.set(info).is_ok()
}

/// Lay a boot document's answers over whatever the device's table says. See
/// [`describe`] for the ordering.
pub fn override_with(over: Overrides) -> bool {
    OVERRIDES.set(over).is_ok()
}

/// The tuning every selection here reads. `Copy`, because a call site that
/// held a borrow would be holding one for the life of the process.
#[must_use]
pub fn current() -> DeviceTuning {
    *RESOLVED.get_or_init(|| {
        let info = DEVICE.get().copied().unwrap_or_default();
        let over = OVERRIDES.get().copied().unwrap_or_default();
        DeviceTuning::of(info).with(&over)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unrecognised_device_gets_the_m1_max_measurements() {
        let m1 = DeviceTuning::default();
        assert_eq!(DeviceTuning::of(DeviceInfo::default()), m1);
        assert_eq!(DeviceTuning::of(DeviceInfo::of_name("Apple M9 Ultra")), m1);
        assert_eq!(DeviceTuning::of(DeviceInfo::of_name("Apple M1 Max")), m1);
        assert_eq!(m1.qmm_min_batch, 5);
        assert_eq!(m1.qmm_bn_crossover_tg, 160);
        assert_eq!(m1.moe_batch_min_per_expert, 2);
    }

    #[test]
    fn a_family_override_moves_one_field_and_no_others() {
        let m1 = DeviceTuning::default();
        let m4 = DeviceTuning::of(DeviceInfo::of_name("Apple M4 Pro"));
        assert_eq!(m4.qmm_bn_crossover_tg, 96);
        assert_eq!(m4.qmm_min_batch_moe, m1.qmm_min_batch_moe);

        let m2 = DeviceTuning::of(DeviceInfo::of_name("Apple M2 Max"));
        assert_eq!(m2.qmm_min_batch_moe, 12);
        assert_eq!(m2.qmm_bn_crossover_tg, m1.qmm_bn_crossover_tg);

        // **THE DENSE CROSSOVER IS THE ONE FIELD ALL THREE NAME**, and the
        // two older families name the value the DEFAULT used to carry. That
        // is the module header's rule caught in the act: six is the M1 Max's
        // own re-sweep of the widths between four and eight, and neither of
        // the other machines has ever run one.
        assert_eq!(m1.qmm_min_batch, 5);
        assert_eq!(m2.qmm_min_batch, 8);
        assert_eq!(m4.qmm_min_batch, 8);
    }

    #[test]
    fn the_name_probe_reads_the_family_and_never_the_core_count() {
        assert_eq!(DeviceInfo::of_name("Apple M1 Max").apple_family, 7);
        assert_eq!(DeviceInfo::of_name("Apple M2").apple_family, 8);
        assert_eq!(DeviceInfo::of_name("Apple M3 Max").apple_family, 9);
        assert_eq!(DeviceInfo::of_name("Apple M4 Pro").apple_family, 9);
        assert_eq!(DeviceInfo::of_name("Apple M4 Pro").gpu_core_count, 0);
    }

    #[test]
    fn a_zero_override_is_a_value_and_not_a_parse_failure() {
        let t = DeviceTuning::default().with(&Overrides {
            moe_batch_min_per_expert: Some(0),
            ..Overrides::default()
        });
        assert_eq!(t.moe_batch_min_per_expert, 0);
        assert_eq!(t.qmm_min_batch, DeviceTuning::default().qmm_min_batch);
    }

    #[test]
    fn the_emulated_crossover_answers_for_any_format_but_g64_b4() {
        let t = DeviceTuning::default();
        assert!(t.fp16_gemm_format(4, 64));
        assert!(!t.fp16_gemm_format(4, 128));
        assert!(!t.fp16_gemm_format(8, 64));
        assert_eq!(t.qmm_min_batch(false, true), 5);
        assert_eq!(t.qmm_min_batch(false, false), 12);
        assert_eq!(t.qmm_min_batch(true, false), 12);
    }
}
