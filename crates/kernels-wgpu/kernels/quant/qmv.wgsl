// Affine GEMV projections, ported from `quant/qmv.comp` (itself from
// `quant/qmv.metal`).
//
// The port keeps MLX's little-endian affine packing, the four-rows-per-slot
// launch shape, and the bounds-checked K tail whose absence broke Gemma-4-31B
// at K=5376. WebGPU promises no subgroup width at all in the Baseline tier, so
// the final reduction is through workgroup memory over the 32 local-x lanes --
// the same shape Vulkan's port took, for the same reason.
//
// Two things are WGSL's alone and are the only real divergences from the
// `.comp`:
//
//   * every bf16 tensor is an `array<u32>` holding TWO values per word, low
//     half first, because WGSL has no 16-bit storage type even with `enable
//     f16`. Every index the GLSL uses against `uint16_t[]` is a half-index
//     here. See `common/bf16.inc.wgsl`.
//   * the output store is therefore a read-modify-write of a word this
//     invocation may only half own, and it goes through a CAS loop. See
//     `store_y` for which race that is and why it is not hypothetical.

//#include "common/bf16.inc.wgsl"
//#include "common/affine.inc.wgsl"

// ── Bindings ────────────────────────────────────────────────────────────────
//
// Numbered from the ROW in `src/quant.rs`, densely from zero over its
// buffer-kinded operands, with the scalars going to the uniform block instead.
// `.wiki/new-driver/vulkan.md` §3 records what happens when they are copied
// from Metal instead: Metal numbers scalars in the same run as buffers, so its
// `residual` is buffer 7 while the row puts it at 5, and a shader that
// transcribes the 7 binds a buffer the shell never wrote.

@group(0) @binding(0) var<storage, read_write> w: array<u32>;
@group(0) @binding(1) var<storage, read_write> scales: array<u32>;
@group(0) @binding(2) var<storage, read_write> biases: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<u32>;
// `atomic` for the CAS in `store_y`, not because anything here accumulates.
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;
// Declared only by the variants whose row has a sixth buffer operand:
// `affine_qmv_fast` has five and no sixth binding at all, so this cannot be
// declared unconditionally -- a module that binds past the row's buffer count
// is a pipeline-layout error, while a HOLE in the set is fine (§13).
//#if defined(PIE_RESIDUAL) || defined(PIE_BIAS)
@group(0) @binding(5) var<storage, read_write> extra: array<u32>;
//#endif

// The block is variant-shaped, and it has to be. `uniform_layout()` builds the
// buffer from what the ROW states -- `affine_qmv_fast` states two scalars -- so
// the fields here are exactly that row's scalars in that row's order, and
// `row_stride`/`m` appear only for the strided variant that reads them. The
// Vulkan file carried three more (`x_slot_stride`, `x_row_stride`,
// `slots_per_row`) that were the ROUTED row's shape, copied in and read by
// nothing.
struct Params {
    in_vec_size: i32,
    out_vec_size: i32,
//#if defined(PIE_WIDE_STRIDED)
    row_stride: i32,
    m: i32,
//#endif
}
@group(1) @binding(0) var<uniform> params: Params;

// Activation rows one workgroup carries, and the whole point of the shape.
//
// A workgroup owns FOUR output columns and reads their weights over the whole
// of K -- four rows of the packed matrix, which for a 2048-wide int4 head is
// 4 KiB.

// It owned EIGHT, as two y-slots of four, until the workgroup was narrowed to
// `@workgroup_size(32, 1, 1)`. The launchers state y in INVOCATIONS and the
// driver divides by the module's own width, so dropping the y extent doubled
// the workgroup count at the same total work without touching a launcher.
// Measured per dispatch, decode shapes: q/o 2048x2048 0.025 -> 0.021 ms
// (147 -> 179 GB/s), gate/up 8192x2048 0.063 -> 0.061, down 2048x8192
// 0.047 -> 0.044, lm head 128256x2048 0.793 -> 0.783. End to end tg128
// 104.7 -> 107.6 tok/s and tg256@2048 73.8 -> 75.5, prefill unchanged.
// A decode's k/v plane is 512 rows, which was 64 workgroups on a 20-core
// GPU -- three per core, with no second wave to hide a memory stall behind.
// This is the opposite lever to widening x to 64 lanes, which was tried and
// measured worse at every shape; the parallelism this kernel wants is MORE
// workgroups, not wider ones. One activation row per workgroup meant the lm head read that 8 KiB
// once per token: a 512-token prefill pulled the whole 64 MiB head 512 times,
// 67 GiB, and the measured 617 ms of a 1981 ms prefill is exactly that traffic
// at this device's bandwidth. Four rows to a workgroup retire four times the
// arithmetic per weight word and divide the traffic by four.
//
// FOUR and not more because the accumulators are registers: one `vec4<f32>`
// per activation row per four output columns, plus a `vec4` of activation
// sums. Eight would double that and the k loop is the whole kernel.
//
// A VARIANT AXIS and not a plain constant, because those registers are not
// free where there is nothing to spend them on. A decode fire is ONE row, and
// the four-row body reached it at 61 tok/s where the one-row body reaches 64 --
// the arithmetic is dead there and the allocation is not. `quant::qmv_fast`
// row, which is also exactly when `Rule::Qmv`'s quartered x extent differs
// from the row count.
// Row-groups of four per workgroup. `kernels_wgpu::quant::QMV_ROWREP` divides
// the grid by this, so the total work is fixed and only the DISPATCH SIZE
// moves -- which is the one thing the GB/s table above says this kernel is
// sensitive to. One is the shipped value; two and four exist to measure the
// slope from the wrong side, since making the grid bigger needs a model-crate
// join and making it smaller needs only this.
//
// # AND THE SLOPE IS STEEP. THIS IS THE MEASUREMENT THE TABLE ABOVE PREDICTED
//
// Three interleaved rounds, 200 samples, `what_a_decode_costs_at_length`,
// p50 ms of a 512-key qwen3-0.6b decode:
//
// | PIE_ROWREP | grid | round 1 | round 2 | round 3 | mean | cost |
// | --- | --- | --- | --- | --- | --- | --- |
// | 1 | rows/4 | 9.598 | 9.644 | 9.568 | 9.603 | |
// | 2 | rows/8 | 10.213 | 10.293 | 10.326 | 10.277 | **+7.0%** |
// | 4 | rows/16 | 12.116 | 12.651 | 12.577 | 12.448 | **+29.6%** |
//
// Non-overlapping at every round, and the three configurations do the
// IDENTICAL work: same codes unpacked, same bytes read, same reduction depth,
// same stores. The only thing that changed is how many workgroups the grid
// asks for. Against the 3.20 ms qmv spends in a decode that is +21% for one
// halving and +89% for two -- superlinear, so the curve is still climbing at
// the shipped grid and a DOUBLING should still pay.
//
// It took a knob that goes the wrong way to get, because the right way needs a
// quantization-aware join in the model crate.
//
// # AND THE WRONG WAY WAS THE ONLY WAY IT WORKS. SEE `PIE_ROWW`
//
// Every sentence in the paragraph above about a DOUBLING is wrong, and the
// measurement that killed it is under `PIE_ROWW`. This table is the falling
// edge of a knee that the shipped grid already sits on top of: halving the
// grid costs 7% and 30%, and doubling it, measured directly, buys nothing
// within a percent. The curve is not superlinear through the operating point;
// it is flat above it and steep below it, and reading a slope on one side as
// a prediction for the other is the mistake this file made twice in one
// afternoon -- once with the GB/s table and once with this one.
//
// It also retires a suspicion. Six probes have now said this kernel's inner
// loop is free -- ALU 2.4%, load issue 0.7%, wider workgroups worse twice --
// and this is the first one that moved the number. The kernel is fine. The
// dispatch is too small.
//
// # THE CHANGE THIS ASKS FOR, WHICH IS TWO ROWS A WORKGROUP AND NOT A KNOB
//
// `PIE_ROWREP` can only go the wrong way, because a workgroup already covers
// the smallest number of rows the code can express: four, the width of the
// `vec4<u32>` that `block_dot1` loads and the `vec4<f32>` it accumulates. Cut
// that to two and the grid DOUBLES at fixed work -- each row is still read by
// exactly one workgroup's thirty-two lanes, so the weight traffic is
// unchanged; only the activation is re-read, and it is a kilobyte the cache
// already holds.
//
// The slope above says that is worth roughly what `PIE_ROWREP = 2` costs, ~7%
// of a decode, and unlike the join in `turns.rs` it applies to EVERY fire --
// including `o_proj` and `down_proj`, which are 56 `affine_qmv_fast_residual`
// launches that no join can ever reach because they have no sibling to be
// joined to. It needs no model crate, no contract change and no new launch.
// It is the largest self-contained win left in this file and the two changes
// compose: a joined q||k||v at two rows a workgroup is a 2048-workgroup
// dispatch where today there are three of 512, 256 and 256.
//
// What it costs to write, honestly: `block_dot1`, `block_dot2` and `block_dot`
// all carry rows in a `vec4`, `reduce_store`'s tree and store both walk `r` to
// four, and `qmv_partials` is sized `4 * PIE_MT * 32`. Every one of those is
// mechanical and none of it is subtle -- except the slot arithmetic, which has
// already cost one afternoon and three gpu tests (see the `mt > 1u` note
// below) and which must be re-derived rather than pattern-matched.
//
// And one thing to MEASURE rather than assume: the prefill. A prefill fire has
// many activation rows and is already a large dispatch, so it sits at the flat
// end of the curve where doubling the grid buys nothing and doubling the
// reduction-tree count costs something. If it regresses, the row width wants
// to be a compile-time variant selected by the launcher on `vecs`, which the
// `PIE_MT` arms already show how to do.
const PIE_ROWREP = 1u;

// Output rows a workgroup covers, and THE KNOB THAT PROVED THE GRID IS DONE.
//
// Four is what `block_dot1`'s `vec4` naturally holds. Two halves it and doubles
// the grid at fixed weight traffic -- each row is still read by exactly one
// workgroup's thirty-two lanes -- which is the direction `PIE_ROWREP`'s slope
// appeared to be asking for. At two, the `vec4` still loads four rows and the
// upper pair REPEATS the lower, so the loop does twice the arithmetic and half
// of the loads are a second read of a line just touched. Both wastes are
// separately measured cheap here (ALU 2.4%, duplicate load 0.7%), which is what
// made a six-line probe worth more than the vec2 rewrite it stands in for.
//
// Three interleaved rounds, 200 samples, p50 ms:
//
// | | round 1 | round 2 | round 3 | mean |
// | --- | --- | --- | --- | --- |
// | four rows, grid rows/4 | 10.057 | 10.048 | 10.041 | 10.049 |
// | two rows, grid rows/2 | 9.894 | 9.987 | 10.015 | 9.965 |
//
// **+0.84%, and the rounds overlap.** Against a predicted 7%.
//
// The obvious objection is the probe's own doubled arithmetic, so it was
// removed: the same sweep with the cheap-unpack stub in `block_dot1` on BOTH
// arms, which shrinks the duplicated work to a mask and a convert.
//
// | | round 1 | round 2 | round 3 | mean |
// | --- | --- | --- | --- | --- |
// | four rows | 9.731 | 9.727 | 9.841 | 9.766 |
// | two rows | 9.824 | 9.797 | 9.886 | 9.836 |
//
// **-0.7%, worse in every round.** Taking the contamination out did not
// uncover a win, it reversed the sign -- so the two experiments straddle zero
// and the honest reading is that doubling this grid does NOTHING, +-1%.
//
// # WHAT THAT COSTS, WHICH IS THREE COMMITS' WORTH OF CONCLUSIONS
//
// The GB/s table above and the `PIE_ROWREP` table below both measure the grid
// getting SMALLER, and both were read as slopes through the operating point.
// They are not. There is a knee, the shipped grid sits on it, and below it the
// cost is steep while above it the curve is flat. That is the ordinary shape
// of a saturation curve and it is exactly the shape that punishes
// extrapolation.
//
// So: the vec2 rewrite this knob was written to justify is DEAD, and so is the
// "rate term" it lent the fusion rows in `driver-wgpu`'s `turns.rs`. Joining
// q||k||v is worth its launch floor and nothing more.
//
// # THE 68 GB/s HAS A NAME NOW, AND HALF OF IT IS A FIXED COST
//
// The line above used to end "still unexplained". `PIE_WGPU_STAMP` times each
// dispatch with the GPU's own clock, and a decode fires this kernel at three
// shapes whose weight banks are all a clean multiple of each other. Weights
// are 4-bit with bf16 scales AND biases every 64, so a bank is
// `out * in / 2 + out * in / 64 * 4` bytes:
//
// ```text
//   fire        out    in    bank        stamped
//   k, v       1024  1024   0.5625 MiB    16.1 us
//   q          2048  1024   1.1250 MiB    23.2 us
//   gate, up   3072  1024   1.6875 MiB    31.0 us
// ```
//
// Three points, and they are a straight line: **8.65 us + 13.24 us a MiB**.
// The middle point, which the fit does not use, lands at 23.55 against 23.2 --
// 1.5%, inside this machine's repeatability.
//
// So the marginal rate is **79.2 GB/s**, and the reason the whole-kernel figure
// reads 68 is the intercept. A qmv fire costs **8.65 us before it reads a
// byte**, which is 54% of the k projection and 12x the 0.71 us of dispatch
// turnaround `device.rs` measured -- so it is not the launch, it is inside the
// kernel. Every probe in this file changed the SLOPE and the slope was never
// the problem.
//
// AND IT RE-ARGUES THE JOIN THIS KNOB JUST FINISHED DEMOTING. Above, `q||k||v`
// is "worth its launch floor and nothing more", which at 0.71 us a launch is
// 0.04 ms a token. On the fit it is worth the two intercepts it deletes:
//
// ```text
//   q + k + v       23.2 + 16.1 + 16.1 = 55.4 us    joined 2.25 MiB -> 38.5
//   gate + up              31.0 + 31.0 = 62.0 us    joined 3.375   -> 53.3
// ```
//
// 16.9 and 8.7 us a layer, **0.72 ms a token between them, 7.4%**. Ten times
// the launch-floor story, and for a reason that has nothing to do with
// dispatch count. `turns.rs` carries the same arithmetic.
//
// # AND THEN A FOURTH POINT BROKE THE FIT, WHICH IS THE SIXTH RULE AGAIN
//
// The lm head is a qmv too -- 151936 rows over 1024, a **83.4 MiB** bank in
// one 37984-group fire -- and it had never appeared in the stamp table because
// Metal loses the last two encoders' counters. `device.rs` now spends two
// sacrificial one-workgroup dispatches to buy its number back. It reads
// **666 us**. The line above predicts 1113.
//
// ```text
//    0.5625 MiB    16.1 us     36.6 GB/s
//    1.1250 MiB    23.2 us     50.8 GB/s
//    1.6875 MiB    31.0 us     57.1 GB/s
//   83.4000 MiB   666.0 us    131.3 GB/s
// ```
//
// So there is no intercept, or not a constant one: **the rate rises with the
// bytes in the fire**, 37 GB/s to 131 across this range, and a straight line
// through three points 3x apart cannot be walked out to 50x. That is rule six
// -- never extrapolate a curve past its measured direction -- caught for the
// second time in this file, this time inside a fit rather than across one.
//
// It is BYTES and not workgroups, which is what makes it consistent with
// `PIE_ROWW` above rather than a contradiction of it. `PIE_ROWW = 2` doubles
// the grid at IDENTICAL bytes -- 256 groups to 512 -- and measured zero twice.
// If the driver were the grid it would have read the 36.6 -> 50.8 step, 39%.
// It did not. Something is amortised over the SIZE OF A FIRE, and a fire's
// grid is not its size.
//
// WHAT SURVIVES FOR THE JOIN, stated as a bound rather than a fit. A joined
// q||k||v is 2.25 MiB, which is bigger than every point in the fitted range,
// and the rate is monotone in bytes -- so it runs at AT LEAST the 57.1 GB/s
// the 1.6875 MiB fire gets:
//
// ```text
//   q + k + v   55.4 us today   joined <= 41.3 us   saves >= 14.1 us a layer
//   gate + up   62.0 us today   joined <= 62.0 us   saves >= 0
// ```
//
// **q||k||v is worth at least 0.39 ms a token and the bound is a real one.**
// `gate||up` is not proven at all: at 3.375 MiB its pessimistic bound is
// exactly break-even, and only the rate's rise decides it. Measure that one,
// do not assume it.
//
// # AND THEN THE TABLE LEARNED TO COUNT BYTES, AND THE CURVE BECAME A PLANE
//
// Every table above is keyed by entrypoint and grid, which silently averaged
// qwen3's `o` and `down` -- both `affine_qmv_fast_residual` at [1,256,1], over
// 1.125 and 1.6875 MiB -- into a single 21.9 us number that is neither.
// `device.rs::charge` now sums the launch's bound buffer lengths into the key,
// and the two came apart. So did the reason the "rate" was rising.
//
// Five points, per launch, windows 2 and 3 of a 512-key decode averaged (the
// ordering below repeats identically in all three windows; window 1 is ~10%
// faster across the board because its context is shorter):
//
// ```text
//   entrypoint    out     in     MiB      measured
//   k, v         1024   1024   0.566      15.85 us
//   q            2048   1024   1.131      22.90 us
//   gate, up     3072   1024   1.695      30.45 us
//   o    (res)   1024   2048   1.133      20.35 us
//   down (res)   1024   3072   1.697      23.15 us
// ```
//
// **Read the last two against the first three.** `down` binds the same bytes
// as `gate`/`up` -- 1.697 against 1.695 -- with a THIRD of the grid, and it is
// 24% FASTER. `o` binds q's bytes with half the grid and is 11% faster. Both
// gaps repeat in every window. So the cost is not the bytes either. What
// distinguishes the pairs is the number of OUTPUT ROWS, and fewer is cheaper
// at equal traffic.
//
// Fit `t = a + b*rows + c*MiB (+ d if residual)` on the three non-residual
// points and the residual pair's slope, and every point lands:
//
// ```text
//   a = 7.8 us      per fire
//   b = 4.64 ns     per output row
//   c = 4.96 us     per MiB   =  209 GB/s, against a 273 GB/s machine
//   d = 2.2 us      for the residual variant's extra read and write
//
//   k, v    15.35 predicted   15.85 measured   +3%
//   q       22.90             22.90             fitted
//   gate    30.45             30.45             fitted
//   o       20.36             20.35            +0.0%
//   down    23.16             23.15            +0.0%
// ```
//
// The residual pair is not fitted for `b` or `a` and lands within 0.01 us.
// That is the whole "rate rises with bytes" story dissolved: there was never a
// rate curve, there was a per-ROW term that grew alongside the bytes because
// every one of the first three points holds `in` at 1024 and varies `out`.
// Rule six, a third time, and this time the fix is a variable rather than a
// caveat.
//
// **THE DECODE'S QMV IS LATENCY-BOUND, NOT BANDWIDTH-BOUND.** 209 GB/s of the
// machine's 273 is already being had on the marginal byte; what is left is
// 7.8 us a fire plus 4.6 ns a row, and at 768 workgroups an M4 Pro's 20 cores
// are nowhere near occupied, so a row's latency is exposed instead of hidden.
// Which is also why the lm head still misses: 151936 rows predicts 1128 us and
// it reads ~800. At 37984 groups the machine IS occupied and the row term goes
// under the bandwidth. The model above is a LOW-OCCUPANCY model and must not
// be walked past the point where the GPU fills up -- rule six once more, now
// stated in advance instead of after.
//
// # WHAT THIS SAYS ABOUT THE JOIN, WHICH IS NOW A PREDICTION
//
// A join does not delete rows, only fires. So the saving is `a` per fire
// deleted, plus whatever the row term's non-linearity gives back -- and the
// model says that is nothing, so this is a floor and a ceiling at once:
//
// ```text
//   q + k + v   54.60 us today   joined 4096 rows, 2.263 MiB   38.0 us
//               saves 16.6 us a layer  =  0.46 ms a token  (4.8%)
//   gate + up   60.90 us today   joined 6144 rows, 3.390 MiB   53.1 us
//               saves  7.8 us a layer  =  0.22 ms a token  (2.3%)
// ```
//
// **0.68 ms a token, 7%, for both joins.** The bound-based estimate two
// sections up said ">= 0.39 ms" for q||k||v from a different and worse model;
// the two agree, which is the first time an estimate in this file has been
// reproduced by an independent route. `gate||up` is no longer break-even --
// under the row model it is a clear 0.22 ms, because the join deletes a whole
// 7.8 us fixed cost and the bytes were never the problem.
//
// # `b` WAS THE TAIL LADDER, AND THE @subgroup TIER TAKES MOST OF IT
//
// The paragraph this replaces guessed: "per output row, then: the tail tree,
// the scale and bias fetch, or the store." It was the tail tree, and the way
// to find out was not to reason about it.
//
// `attn.rs` had just established the mechanism on a different kernel -- on
// Metal a reduction LEVEL costs, not the adds in it, because a level is a
// barrier and a workgroup-memory round trip and a level with four live lanes
// costs what a level with thirty-two costs. `reduce_store` ends every
// workgroup with five such levels over 32 lanes. This workgroup is
// `@workgroup_size(32, 1, 1)`, which on any adapter whose subgroup is at least
// 32 wide is EXACTLY ONE SUBGROUP, so all five fold into register exchanges
// with no barrier at all. See `reduce_store` for the butterfly and why
// `lim = min(32, subgroup_size)` makes it correct at any width.
//
// Same window of the same bench, `PIE_WGPU_TIER` switching tiers inside one
// binary, per launch:
//
// ```text
//                   baseline   subgroup
//   q       2048 rows  21.0 us   17.2 us   -18%
//   gate/up 3072 rows  28.2 us   22.9 us   -19%
// ```
//
// **The saving scales with the ROWS, which is what identifies it.** q sheds
// 3.8 us over 2048 rows and gate/up 5.3 over 3072 -- 1.86 and 1.73 ns a row,
// the same number twice -- so what came out is 40% of `b` and nothing of `a`
// or `c`. Refitting, `b` falls from 4.64 ns a row to about 2.8, and the rest
// of it is still the scale and bias fetch or the store.
//
// End to end, three interleaved rounds with BOTH subgroup arms in (this one
// and the attention ladder `attn.rs` describes):
//
// ```text
//   round        1       2       3
//   subgroup   7.671   7.772   7.747   ms a token
//   baseline   9.667   9.575   9.574
// ```
//
// **1.88 ms a token, 19.5%**, of which the attention ladder measured 0.54 on
// its own, so this one is worth about 1.3.
//
// WHAT IT DOES TO THE JOIN, which is the other thing `b` was load-bearing for:
// nothing. A join deletes FIRES, so it collects `a`, and `a` did not move --
// q||k||v is still ~16 us a layer. It is a larger fraction of a smaller token
// now, which is the only change.
//
// `a` is still "pipeline state or a cold bank", still 7.8 us, and is now the
// biggest unexplained term in the smallest fire.
const PIE_ROWW = 4u;

// How many workgroups of row-groups ride the y axis before the rest go to z.
//
// `maxComputeWorkgroupsPerDimension` is 65535 and WebGPU guarantees it, so a
// grid of rows/2 stops being expressible at 131072 rows -- which the lm head
// passes on the model this file is tuned against, 151936 rows asking for
// 75968. At four rows a workgroup it fit by a factor of two and nobody had to
// know. It does not fit at two, so the row index is a two-digit number in base
// `PIE_YTILE` and `kernels_wgpu::quant::qmv_grid` writes the digits.
const PIE_YTILE = 65535u;

const PIE_MT = 2u;

// 4 output columns x `PIE_MT` activation rows x the 32 local-x lanes each
// column is split over.
var<workgroup> qmv_partials: array<f32, 4u * PIE_MT * 32u>;

// The `i`-th bf16 of a word already loaded. Not `pie_load_bf16`: that one takes
// a `ptr<storage, ...>`, which naga 30 parses and then refuses to validate
// ("which can't be passed into functions"), so it cannot appear in a module
// that has to reach a device. The conversion is still the include's.
fn qmv_bf16(word: u32, i: u32) -> f32 {
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// Write one bf16 output element.
//
// The word at `i >> 1` holds outputs `i & ~1` and `i | 1`, and this invocation
// generally owns only one of them: with an odd `out_vec_size` -- a one-column
// gate projection is a real shape -- consecutive outputs of one row land in one
// word while belonging to different y-slots, and consecutive `vec`s land in one
// word while belonging to different WORKGROUPS. A plain read-modify-write
// therefore loses whichever half landed second, and no workgroup barrier can
// fix the cross-workgroup case. The CAS is device-scoped, which is exactly the
// scope of the race; it retries on the spurious failure `...Weak` is allowed.
fn store_y(i: u32, v: f32) {
    let at = i >> 1u;
    let shift = (i & 1u) * 16u;
    let keep = ~(0xffffu << shift);
    let bits = pie_f32_to_bf16(v) << shift;
    var old = atomicLoad(&y[at]);
    loop {
        let r = atomicCompareExchangeWeak(&y[at], old, (old & keep) | bits);
        if r.exchanged {
            break;
        }
        old = r.old_value;
    }
}

fn affine_value(row: u32, k: u32) -> f32 {
    let len = u32(params.in_vec_size);
    let word = w[pie_affine_word_of(row, len, k)];
    let g = pie_affine_scale_of(row, len, k);
    return pie_affine_value(
        word,
        pie_affine_code_of(k),
        qmv_bf16(scales[g >> 1u], g),
        qmv_bf16(biases[g >> 1u], g),
    );
}

fn x_at(vec_: u32, k: u32) -> f32 {
//#if defined(PIE_WIDE_STRIDED)
    let i = vec_ * u32(params.row_stride) + k;
//#else
    let i = vec_ * u32(params.in_vec_size) + k;
//#endif
    return qmv_bf16(x[i >> 1u], i);
}

// The WORD holding activations `k` and `k + 1`, for an even `k`.
//
// `x_at` reads `x[i >> 1u]` and keeps one half, so a loop that walks `k` one at
// a time loads every word TWICE. A lane's block is sixteen activations against
// eight weight-word loads at four bits, so those redundant loads were the
// commonest instruction in the inner loop.
fn x_word(vec_: u32, k: u32) -> u32 {
//#if defined(PIE_WIDE_STRIDED)
    let i = vec_ * u32(params.row_stride) + k;
//#else
    let i = vec_ * u32(params.in_vec_size) + k;
//#endif
    return x[i >> 1u];
}

// Values one lane pulls per pass: FOUR words' worth, so 32 at four bits and 16
// at eight.
//
// It was two words'. Four halves the trip count of the K loop and doubles the
// weight loads in flight per lane, which is what a serial-dispatch GPU has to
// hide a memory stall behind. Measured per dispatch: lm head 128256x2048
// 0.783 -> 0.741 ms (188 -> 199 GB/s), gate/up 8192x2048 0.061 -> 0.058,
// down 2048x8192 0.044 -> 0.042; tg128 107.6 -> 110.2 tok/s. Prefill is
// unchanged -- it goes through `qmm_t`.
//
// FOUR AND NOT EIGHT. The scale and bias are hoisted to once per lane-block,
// which is only sound while a block cannot straddle two quantization groups
// -- `PIE_QMV_VPT <= PIE_GROUP`, and the narrowest group this file stamps is
// 32. Eight words is 64 codes at four bits and would read one scale for two
// groups' weights, which is wrong and not slow. The GLSL passed this as a runtime argument and then wrote the inner
// loop to a constant 16 with `i < values_per_thread` inside; here it is a real
// const-expression -- `PIE_BITS` is a prelude const -- so the loop bound is the
// constant and the redundant guard is gone. The OTHER guard is not redundant:
// see `dot_lane`.
// Whether the nibble fast path above applies: four bits to a code, so a
// 32-bit word is eight of them and two `unpack4x8unorm` calls cover it.
const PIE_NIBBLE_FAST = PIE_BITS == 4;
const PIE_QMV_VPT = PIE_CODES_PER_WORD * 4;

// One lane's block of `PIE_QMV_VPT` values, for FOUR rows at once.
//
// This is `qmv_fast_impl`'s inner loop as MLX writes it, and the shape of it is
// the whole point. The version this replaces called `affine_value(row, k)` once
// PER ELEMENT PER ROW, and that helper re-derived the word index, re-read the
// packed word, and re-read the scale AND the bias every time -- so a lane's
// sixteen values cost sixteen word loads and thirty-two scale loads per row,
// where they need two and one.
//
// Three things are hoisted, and each is hoisted to where the value actually
// changes:
//
// * **The scale and bias, to once per lane-block.** `PIE_QMV_VPT` divides
//   `PIE_GROUP` at every instantiation this file stamps -- sixteen into 32, 64
//   and 128 at four bits, eight into the same at eight -- so a lane's values
//   never straddle two groups and one pair covers them all.
// * **The packed word, to once per `PIE_CODES_PER_WORD` values.** That is what
//   the word IS: eight codes at four bits, four at eight.
// * **The activation, to once per value.** It was read once per value PER ROW.
//
// The four rows ride in a `vec4` rather than a loop, which is not a flourish:
// a `var` array indexed by a loop variable is scratch memory in every backend
// naga targets, so the array that would make this readable would also make it
// slower than what it replaces.
// One lane's block for four output columns and ONE activation row.
//
// The `PIE_MT` form below does the same arithmetic for four rows at a time and
// is what a prefill wants; a DECODE fire is one row, and running it through the
// four-row form spends four times the multiplies to throw three quarters of
// them away. Measured: 64.0 tok/s at ctx 512 with this form, 48.6 without it
// and 51.6 with only the reduction narrowed -- so it is the k loop's
// arithmetic that a decode cannot afford, not the tree. `reduce_store` picks
// between the two on a workgroup-uniform test, which is why neither needs a
// barrier inside it.
fn block_dot1(rows: vec4<u32>, vec_: u32, k0: u32) -> vec4<f32> {
    let n = u32(params.in_vec_size);
    let cpw = u32(PIE_CODES_PER_WORD);
    let mask = (1u << u32(PIE_BITS)) - 1u;
    let g = rows * (n / u32(PIE_GROUP)) + vec4<u32>(k0 / u32(PIE_GROUP));
    let s = vec4<f32>(
        qmv_bf16(scales[g.x >> 1u], g.x),
        qmv_bf16(scales[g.y >> 1u], g.y),
        qmv_bf16(scales[g.z >> 1u], g.z),
        qmv_bf16(scales[g.w >> 1u], g.w),
    );
    let b = vec4<f32>(
        qmv_bf16(biases[g.x >> 1u], g.x),
        qmv_bf16(biases[g.y >> 1u], g.y),
        qmv_bf16(biases[g.z >> 1u], g.z),
        qmv_bf16(biases[g.w >> 1u], g.w),
    );
    let wbase = rows * (n / cpw);
    var acc = vec4<f32>(0.0);
    var xsum = 0.0;
    let whole = k0 + u32(PIE_QMV_VPT) <= n;
    for (var jw = 0u; jw < u32(PIE_QMV_VPT) / u32(PIE_CODES_PER_WORD); jw = jw + 1u) {
        let base = k0 + jw * u32(PIE_CODES_PER_WORD);
        if base >= n {
            break;
        }
        let at = wbase + vec4<u32>(base / u32(PIE_CODES_PER_WORD));
        let word = vec4<u32>(w[at.x], w[at.y], w[at.z], w[at.w]);
        // FOUR CODES AN INSTRUCTION, for the shape that is 92% of a decode.
        //
        // A nibble loop pays a shift, a mask and an integer-to-float convert
        // per code per row -- three instructions for one number, and qmv is
        // ALU-bound: with the unpack stubbed out the kernel reads its weights
        // at 113 GB/s against this machine's ~120 GB/s roofline, so every
        // instruction removed here is time.
        //
        // # THAT ROOFLINE IS AN M4. THIS IS AN M4 PRO, AND IT IS NOT TRUE HERE
        //
        // 120 GB/s is the base M4: ten GPU cores, LPDDR5-7500 on a 128-bit
        // bus. The part this file is now tuned against is an M4 Pro -- twenty
        // cores, 273 GB/s -- so the sentence above is about a machine with 44%
        // of this one's bandwidth, and "113 against 120" is a saturation claim
        // that does not survive the move.
        //
        // Re-measured here, interleaved, `what_a_decode_costs_at_length`:
        //
        // | probe | p50 ms | worth |
        // | --- | --- | --- |
        // | (baseline) | 9.514 | |
        // | the eight `unpack4x8unorm` and their eight multiplies replaced by a
        //   mask and a convert | 9.285 | **2.4%** |
        // | every weight word loaded TWICE (unfoldable runtime index) | 9.587 |
        //   **-0.7%** |
        //
        // So on this part the unpack is 2.4% and the load ISSUE is nothing.
        // Neither binds. What the kernel actually reads is 218 MB of weights
        // across its 141 decode fires in 3.20 ms, which is **68 GB/s, a
        // quarter of what the part will do** -- and the shortfall is not in
        // this loop.
        //
        // # AND THE 68 GB/s IS AN ARTEFACT. THE BYTES ARE AT THE WALL.
        //
        // Everything below this heading divides the WHOLE kernel's time by the
        // bytes it reads and calls the quotient a bandwidth. That is only a
        // bandwidth if fetching the bytes is what the time is, and it is not.
        //
        // The `k0 = lid * PIE_QMV_VPT` above gives adjacent lanes a 16-byte
        // stride at a 4-byte load, so a 32-lane instruction spans 512 bytes and
        // uses 128, and 273/4 is 68 -- the measured number, exactly. That is a
        // coalescing defect with a matching fingerprint, and the "every weight
        // word loaded TWICE costs 0.7%" probe below CANNOT rule it out: a
        // repeated load is answered by the cache and says nothing about the
        // footprint the first one pulled in.
        //
        // So cut the footprint instead. `w[at.x]` four times in place of
        // `w[at.{x,y,z,w}]` -- the four rows of the block all read the first
        // row's words -- is a FOUR-fold cut in distinct weight bytes at an
        // identical issue count, identical arithmetic and identical bindings.
        // Wrong answers, so it is a probe. Two rounds against baselines of
        // 7.440 and 7.451 ms:
        //
        // | probe | p50 ms | p50 ms | saved |
        // | --- | --- | --- | --- |
        // | one row's words for all four | 6.490 | 6.527 | **0.94 ms** |
        //
        // Linear in the footprint, a 4x cut that saves 0.94 ms means the whole
        // fetch is 1.25 ms of qmv's measured 3.91. And the weight banks are
        // 0.352 GB a token -- 252 MiB of layers plus an 83.8 MiB lm head -- so
        //
        //     0.352 GB / 1.25 ms  =  282 GB/s
        //
        // against this part's 273. **qmv reads its weights at the machine's
        // wall.** There is no coalescing defect, the 16-byte stride costs
        // nothing because the four words a lane wants are consecutive and the
        // second, third and fourth `jw` trips hit lines the first pulled in,
        // and the 68 GB/s never was a rate the memory system achieved.
        //
        // What it leaves is sharper than what it removes: **2.66 ms of qmv is
        // not traffic at all**, 13.6 us across its 196 fires, in a kernel
        // whose bytes are already saturated. Every remaining idea about this
        // loop's LOADS is dead -- issue, coalescing, and now footprint -- and
        // the whole of the gap is in what a fire costs before and around them.
        //
        // # WHAT THE 3.91 ms IS, LINE BY LINE, AND 61% OF IT IS STILL OPEN
        //
        // `turns.rs` measures this kernel family at 3.91 ms of a 7.45 ms token
        // by deleting its fires. Every stream in a fire has now been priced by
        // a probe that changes ONE of them and leaves issue count, bindings
        // and grid alone. Two rounds each, against baselines of 7.440/7.451:
        //
        // | what the probe cuts | p50 ms | worth |
        // | --- | --- | --- |
        // | weight footprint 4x (one row's words for four) | 6.490 6.527 | **1.25 ms** |
        // | 6 of 8 scale/bias loads, footprint 4x | 7.290 7.302 | **0.15 ms** |
        // | activation footprint 32x (`x[(i>>1) & 3]`) | 7.461 7.449 | **tie** |
        // | 124 of 128 dead `qmv_partials` stores | 7.494 | **tie** |
        //
        // ```text
        //   weight traffic, at 282 GB/s          1.25 ms   32%
        //   scales and biases                    0.15 ms    4%
        //   the 4-bit unpack (2.4%, above)       0.09 ms    2%
        //   activation loads                     0
        //   the workgroup-memory tail            0
        //                                      ---------  ----
        //   unaccounted                          2.42 ms   61%
        // ```
        //
        // THE ACTIVATION IS FREE and that is worth stating, because each of a
        // fire's 256 workgroups reads the whole 2 KiB slice, so a fire touches
        // as many activation bytes as weight bytes. Cutting the footprint 32x
        // moved nothing: the same 2 KiB read by every workgroup is a cache hit
        // and always was.
        //
        // THE DEAD STORES ARE FREE, WHICH IS THE SURPRISING ONE. Under the
        // subgroup tier the butterfly leaves every lane of a block holding the
        // block's sum, and the final read walks one lane a block -- so 31 of
        // every 32 stores write a value nothing reads. Guarding them with
        // `lid % lim == 0` is correct at both tiers (300 tests pass, baseline
        // and subgroup) and strictly less work, and three interleaved rounds
        // read 7.554/7.458/7.471 against 7.520/7.457/7.471. A tie, so it was
        // reverted rather than shipped for tidiness. Same lesson the split
        // attention's merge learned from the other side: **rearranging what
        // touches workgroup memory buys nothing here, because the touching was
        // never the price.**
        //
        // So six probes into the loop and two into the tail have found 38% of
        // this kernel, and none of the remaining 2.42 ms -- 12.3 us across 196
        // fires -- is a load, an unpack, a store or the grid. `PIE_ROWW` and
        // `PIE_ROWREP` below say the grid is on a knee, flat above and steep
        // below. What is left is what a fire costs to BE, and the only lever
        // this file has on that is having fewer of them.
        //
        // It is in how little each fire is. Take the lm head out and 140 fires
        // move 140 MB, 513 us of data at the part's rate, in 2.9 ms: 20.8 us a
        // fire where the bytes are 1.8 us for a k/v plane and the launch floor
        // is 6. The small projections are LATENCY-bound, not bandwidth-bound,
        // and a k/v fire is 256 workgroups of 32 invocations -- 8192 threads
        // for a twenty-core GPU, each doing one 32-code block and then a
        // five-level tree.
        //
        // That reprices `fuse-qkv-quant` in `turns.rs`. Joining q, k and v is
        // not worth two launch floors (0.34 ms); it makes ONE dispatch of 1024
        // workgroups where there were three of 512, 256 and 256, and the
        // measurement above says the dispatch size is what this kernel is
        // short of. Same for gate and up.
        //
        // What is NOT worth trying, because it is measured: anything that makes
        // this loop's arithmetic cheaper, and anything that merges its loads.
        // The second is the surprise -- a lane reads four words per row four
        // bytes at a time, sixteen-byte strided across the group, which looks
        // exactly like a coalescing defect -- and issuing every one of them
        // twice costs 0.7%, so folding them into one `vec4<u32>` load can buy
        // at most that.
        //
        // `unpack4x8unorm` is Metal's `unpack_unorm4x8_to_float` and SPIR-V's
        // `UnpackUnorm4x8`: one instruction for four bytes. Masking the word
        // to `0x0f0f0f0f` puts codes 0, 2, 4 and 6 alone in their bytes and
        // shifting first puts 1, 3, 5 and 7 there, so two unpacks cover the
        // word.
        //
        // **The `255.0` is exact.** The builtin divides each byte by 255 and
        // this multiplies it back; for a byte below 256 both roundings are
        // below half an ulp of a value whose integer neighbours are 2^-8 apart
        // in relative terms, so the round trip is the identity. The codes are
        // therefore the same numbers the loop below produces, multiplied into
        // `acc` in the same order, which is what the cross-backend parity walk
        // compares.
        if PIE_NIBBLE_FAST && whole {
            let lo = vec4<u32>(0x0f0f0f0fu);
            let e0 = unpack4x8unorm(word.x & lo.x) * 255.0;
            let e1 = unpack4x8unorm(word.y & lo.y) * 255.0;
            let e2 = unpack4x8unorm(word.z & lo.z) * 255.0;
            let e3 = unpack4x8unorm(word.w & lo.w) * 255.0;
            let sh = (word >> vec4<u32>(4u)) & lo;
            let o0 = unpack4x8unorm(sh.x) * 255.0;
            let o1 = unpack4x8unorm(sh.y) * 255.0;
            let o2 = unpack4x8unorm(sh.z) * 255.0;
            let o3 = unpack4x8unorm(sh.w) * 255.0;
            let w0 = x_word(vec_, base);
            let w1 = x_word(vec_, base + 2u);
            let w2 = x_word(vec_, base + 4u);
            let w3 = x_word(vec_, base + 6u);
            let x0 = pie_bf16_to_f32(w0 & 0xffffu);
            let x1 = pie_bf16_to_f32(w0 >> 16u);
            let x2 = pie_bf16_to_f32(w1 & 0xffffu);
            let x3 = pie_bf16_to_f32(w1 >> 16u);
            let x4 = pie_bf16_to_f32(w2 & 0xffffu);
            let x5 = pie_bf16_to_f32(w2 >> 16u);
            let x6 = pie_bf16_to_f32(w3 & 0xffffu);
            let x7 = pie_bf16_to_f32(w3 >> 16u);
            xsum = xsum + x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            acc = acc + x0 * vec4<f32>(e0.x, e1.x, e2.x, e3.x);
            acc = acc + x1 * vec4<f32>(o0.x, o1.x, o2.x, o3.x);
            acc = acc + x2 * vec4<f32>(e0.y, e1.y, e2.y, e3.y);
            acc = acc + x3 * vec4<f32>(o0.y, o1.y, o2.y, o3.y);
            acc = acc + x4 * vec4<f32>(e0.z, e1.z, e2.z, e3.z);
            acc = acc + x5 * vec4<f32>(o0.z, o1.z, o2.z, o3.z);
            acc = acc + x6 * vec4<f32>(e0.w, e1.w, e2.w, e3.w);
            acc = acc + x7 * vec4<f32>(o0.w, o1.w, o2.w, o3.w);
            continue;
        }
        // TWO CODES A TRIP, because both read the same activation word.
        // `base` is a multiple of `PIE_CODES_PER_WORD` and `in_vec_size` is
        // even, so `k` is even here and the word holds `k` in its low half and
        // `k + 1` in its high one. The order the products reach `acc` is the
        // order it was -- c, then c+1 -- so this is the same sum.
        for (var c = 0u; c < u32(PIE_CODES_PER_WORD); c = c + 2u) {
            if !whole && base + c >= n {
                break;
            }
            let xw = x_word(vec_, base + c);
            let xv0 = pie_bf16_to_f32(xw & 0xffffu);
            xsum = xsum + xv0;
            let code0 = (word >> vec4<u32>(u32(PIE_BITS) * c)) & vec4<u32>(mask);
            acc = acc + xv0 * vec4<f32>(code0);
            if !whole && base + c + 1u >= n {
                break;
            }
            let xv1 = pie_bf16_to_f32(xw >> 16u);
            xsum = xsum + xv1;
            let code1 = (word >> vec4<u32>(u32(PIE_BITS) * (c + 1u))) & vec4<u32>(mask);
            acc = acc + xv1 * vec4<f32>(code1);
        }
    }
    return s * acc + b * vec4<f32>(xsum);
}

// WHAT IS LEFT IN HERE, MEASURED, SO THE NEXT PASS STARTS FROM A NUMBER
//
// On an M4 Pro with Llama-3.2-1B this file is most of a decoded token's GPU
// time. The weights it must read are 1.22G parameters at four bits plus
// scales and biases, about 0.67 GB a token.
//
// # The roofline this was measured against was WRONG, and the record is fixed
//
// An earlier version of this note put the machine's peak at 120 GB/s -- the
// base M4's -- and concluded from ~93 GB/s that the kernel was at its
// roofline and worth a few percent at most. The adapter is an M4 PRO, which
// moves 273 GB/s. Anything that reasoned from "we are at the roofline" was
// reasoning from the wrong number.
//
// What replaces it, from `how_long_a_decodes_kernels_take` at this model's own
// shapes. `wgs` is the workgroup count, which is the output height over eight;
// `work` is the dispatch time less the 0.009 ms an 8x64 FLOOR dispatch costs,
// which is what an empty kernel costs and is not this file's to spend:
//
//   shape                 wgs      ms    work      GB/s of work
//   512x2048    k/v        64   0.013   0.004      147
//   2048x2048   q/o       256   0.025   0.016      147
//   8192x2048   gate/up  1024   0.063   0.054      175
//   2048x8192   down      256   0.047   0.038      248
//   128256x2048 head    16032   0.793   0.784      188
//
// Everything but the head fits an M4 Pro's system cache when dispatched two
// hundred times over, so 248 is a cache number; the head at 147.8 MB is the
// only row that is honestly DRAM, and it reads at 188 GB/s -- 69% of peak.
//
// So THE BODY IS NOT THE PROBLEM at any shape. What the table actually shows
// is the 0.009 ms floor: it is a third of a k/v projection and 36% of a q or
// o. A token fires this kernel 113 times and 115 other kernels besides, and
// 228 x 0.009 ms is 2.0 ms of a 8.4 ms token -- a quarter of the GPU, spent
// on dispatch and not on arithmetic. The lever that is left is FEWER AND
// BIGGER dispatches, which is fusion in the authored trace, not WGSL.
//
// Per eight codes and four rows -- 32 multiply-adds -- the fast path above
// spends 32 fused multiply-adds and about as many other instructions again:
// eight `unpack4x8unorm`, eight multiplies by 255, eight bf16 widenings, and
// EIGHT `xsum` ADDS.
//
// That last eighth is the one that is not inherent. `xsum` is a sum of
// ACTIVATIONS over the lane's k block; it does not depend on the row, and yet
// every one of the 512 four-row groups in a 2048-wide projection recomputes
// the same value from the same words. A pre-pass over `x` writing one partial
// sum per `PIE_QMV_VPT` block -- 128 floats for a 2048-wide vector -- would
// turn eight adds into one load here, and summing each block in the order this
// loop does keeps the answer bit-identical.
//
// It is not done because it is not kernel-local: it needs a routine, a buffer
// the driver stages per fire, and a place in the authored trace. Recorded with
// its size (about an eighth of the arithmetic above the transfer, so a few
// percent of a token) so that it is weighed rather than guessed at.
//
// Things that were tried against this loop and MEASURED WORSE, so that they
// are not tried again: one word per lane and four words per lane
// (`PIE_QMV_VPT`), `extractBits` instead of shift-and-mask (naga emits
// clamping guards), folding the `255.0` into the scale (a wash, and it changes
// the rounding), and the `unpack4x8unorm` path in `block_dot` below, where the
// extra live registers cost more than the instructions save.
//
// THE NIBBLE PATH IN `block_dot` WAS TRIED A SECOND TIME, structured the way
// `block_dot2` is -- unpack the eight weight vectors once, then walk the four
// activation rows ONE AT A TIME, so no more than one accumulator and eight
// activation floats are live beside them. That is the exact discipline that
// made `block_dot2` a 26% win at two rows, and at four rows it loses anyway:
//
//   streams        1       4       8      16
//   before       114.0   194.8   250.1   217.5 tok/s
//   after        109.8   137.7   167.1   187.5 tok/s
//
// Two rows is the shape where the unpack pays and four is not, so the entry
// above stands and the arm split is the answer, not a rewrite of `block_dot`.
//
// AND NOTE THE ONE-STREAM COLUMN. `mt == 1` does not go through `block_dot`
// at all, and it still fell 114.0 -> 109.8. The arms share an entrypoint, so
// the register allocation is the WORST arm's, and a fat arm taxes every fire
// that never enters it. That is the reason `block_dot2` is written lean, and
// it is the first thing to weigh before adding a third arm here.
//
// 64 K-LANES INSTEAD OF 32 was tried a second time, on the theory that a
// 512-row plane gives only 64 workgroups and 4096 invocations for a whole GPU,
// and that the table above was reading occupancy. It is not. Doubling the x
// extent to `@workgroup_size(64, 2, 1)` -- same grid, twice the invocations,
// one more level of tree, 8 KiB of partials instead of 4 -- measured worse at
// EVERY shape, including the head, where occupancy was never in question:
//
//   512x2048     45 ->  35 GB/s      8192x2048    149 -> 123 GB/s
//   2048x2048    96 ->  83 GB/s    128256x2048    186 -> 146 GB/s
//
// (Whole-dispatch figures, floor included, which is why they are lower than
// the `work` column above.) A wider workgroup buys memory requests in flight
// and pays for them in resident workgroups and reduction depth, and on this
// adapter the second is dearer. The first attempt at this is recorded in the
// list above; it is repeated here with numbers so the third attempt does not
// happen.
//
// # THE LEFT COLUMN OF THAT TABLE IS THE ANSWER TO THE WHOLE KERNEL
//
// It was written to dismiss occupancy and it was read that way for a year.
// Read the BASELINE column instead, against its dispatch size -- a workgroup
// covers four output rows, so the grid is rows/4:
//
// | plane | workgroups | GB/s |
// | --- | --- | --- |
// | 512x2048 | 128 | 45 |
// | 2048x2048 | 512 | 96 |
// | 8192x2048 | 2048 | 149 |
// | 128256x2048 | 32064 | 186 |
//
// **Monotone in the grid, over 4x from end to end, and nowhere near flat by
// the last row.** That is not a property of the arithmetic -- every row runs
// the identical loop -- it is the kernel saying it cannot fill a twenty-core
// GPU until it is given tens of thousands of workgroups.
//
// READ THE `PIE_ROWW` NOTE BEFORE BELIEVING THE PARAGRAPH BELOW. Every row of
// this table has a different amount of WORK as well as a different grid, so
// "nowhere near flat" is a statement about two variables at once. Holding the
// work fixed and doubling the grid -- which `PIE_ROWW` does -- moves nothing.
// The plans drawn from this table survive as descriptions and not as levers.
//
// Now place the decode's real fires on that curve. qwen3-0.6b, hidden 1024:
//
// | fire | rows | workgroups | expected |
// | --- | --- | --- | --- |
// | k, v | 1024 | 256 | between 45 and 96 |
// | q | 2048 | 512 | ~96 |
// | gate, up | 3072 | 768 | ~100 |
// | lm head | 151936 | 37984 | ~186 |
//
// The measured whole-kernel figure, 218 MB in 3.20 ms, is **68 GB/s** -- and
// the fires are weighted toward the top of that table. The curve predicted the
// number. qmv is not slow; it is being asked in pieces too small to be fast,
// and the two probes in `block_dot1`'s note (ALU 2.4%, load issue free) are
// what that looks like from inside the loop.
//
// So the lever is the GRID, and there are only two ways to grow it. Fewer
// output rows per workgroup keeps the invocation count and doubles the groups
// -- untried, and the only remaining intra-kernel move, since the two attempts
// above prove wider is worse. Or JOIN THE FIRES, which is `turns.rs`'s fusion
// table: q||k||v turns 512 + 256 + 256 into 1024 and gate||up turns 768 + 768
// into 1536, both a step up this curve on top of the launches they delete.

// One lane's block, for four output columns AND `PIE_MT` activation rows.
//
// The four sums ride in named fields rather than an array for the reason the
// four columns ride in a `vec4`: an array indexed by a loop variable is scratch
// memory in every backend naga targets, and the k loop is where it would live.
struct BlockM {
    a0: vec4<f32>,
    a1: vec4<f32>,
    a2: vec4<f32>,
    a3: vec4<f32>,
}

// One lane's block for four output columns and EXACTLY TWO activation rows.
//
// # Why a second special case, when `block_dot` already handles two
//
// It handles two by computing four. `reduce_store` clamps `vecs` to the last
// live row, so a two-row fire runs row 1 three times and discards two of
// them -- and it takes the slow unpack besides, because only `block_dot1`
// has the nibble path. Measured on an M4 Pro, Llama-3.2-1B, two concurrent
// conversations: 12.99 ms a step against 7.52 ms for one conversation, where
// the weights read is the same read. Four rows of arithmetic on three
// instructions a code is exactly the 1.73x that is.
//
// # What this arm bought
//
// M4 Pro, Llama-3.2-1B q4, both sides built and measured back to back on the
// same thermal state (the machine drifted a few percent warmer across the
// day, so only same-sitting pairs mean anything):
//
// ```text
//                          before    after
//   2 streams, aggregate   133.7     168.6 tok/s    +26%
//   2 streams, per-stream   68.7      90.2 tok/s
//   8 streams, aggregate      -      205.2 tok/s
//   1 stream,  tg128       111.1     113.9 tok/s    unchanged (mt == 1)
//   1 stream,  pp512       900.7     902.2 tok/s    unchanged
// ```
//
// The one-row and prefill paths do not come through here and did not move,
// which is the check that the extra arm cost the entrypoint no registers.
//
// Two rows is the shape a SERVED deployment actually decodes in -- the
// second conversation is the common case, not the fourth -- and it is the
// one shape where the nibble path is affordable. The list at the foot of
// this file records `unpack4x8unorm` in `block_dot` as measured worse, and
// that stands: the FOUR-row form keeps four accumulators and four rows of
// activations live across the unpack, and the register pressure costs more
// than the instructions save. This form keeps two accumulators and walks one
// row's activations at a time, so the eight unpacked vectors are the only
// thing held across.
//
// # THE ORDER IS `block_dot1`'s AND THE ANSWER IS STILL NOT THE SAME
//
// The products reach each accumulator k ascending, exactly as they do in
// `block_dot1`. This used to conclude from that "a row summed here and the
// same row summed alone agree bit for bit", and it is FALSE: order is not
// the whole of the arithmetic. This body holds twice the accumulators and
// twice the activations live across the unpack, so the backend contracts a
// different subset of its `a + b * c` into fused multiply-adds, and a fused
// product rounds once where an unfused one rounds twice.
//
// Measured at two bf16 ulps on about five outputs in a hundred thousand, and
// only at a projection as wide as an lm head -- 0 of 8192, 1 of 16384, 7 of
// 151936, a flat rate rather than a threshold. It matters because
// `reduce_store` picks this arm by ROW COUNT: an even batch is summed here
// throughout and the tail of an odd one is summed by `block_dot1`, so a
// conversation's logits depend on how many OTHER conversations were batched
// with it. Twenty-eight layers turn five in a hundred thousand into 139561 of
// 151936 logits differing by 0.79% of the row's peak, same argmax.
//
// `driver-wgpu`'s `a_seats_answer_does_not_depend_on_how_many_seats_fired_
// with_it` is that fact with the parity sweep that found it, and its
// `BATCH_SPREAD` is the bound. There is no kernel-level test to point at:
// `kernels-wgpu` has no `tests/` any more.
//
// Exactness is available and was measured: calling `block_dot1` twice here
// makes every row count agree bit for bit and costs 13-20% of the matvec at
// m = 8 and above (`how_long_a_decodes_kernels_take`, 1024x1024 on an M4 Pro:
// 0.034 ms against 0.041 at m = 8, 1.636 against 1.856 at m = 512). It was
// not taken, because the matvec is where a batched decode spends its time and
// two ulps is not an error -- but it is the remedy if a deployment ever needs
// its answers not to depend on its own batching.
struct BlockM2 {
    a0: vec4<f32>,
    a1: vec4<f32>,
}

fn block_dot2(rows: vec4<u32>, vecs: vec2<u32>, k0: u32) -> BlockM2 {
    let n = u32(params.in_vec_size);
    let cpw = u32(PIE_CODES_PER_WORD);
    let mask = (1u << u32(PIE_BITS)) - 1u;
    let g = rows * (n / u32(PIE_GROUP)) + vec4<u32>(k0 / u32(PIE_GROUP));
    let s = vec4<f32>(
        qmv_bf16(scales[g.x >> 1u], g.x),
        qmv_bf16(scales[g.y >> 1u], g.y),
        qmv_bf16(scales[g.z >> 1u], g.z),
        qmv_bf16(scales[g.w >> 1u], g.w),
    );
    let b = vec4<f32>(
        qmv_bf16(biases[g.x >> 1u], g.x),
        qmv_bf16(biases[g.y >> 1u], g.y),
        qmv_bf16(biases[g.z >> 1u], g.z),
        qmv_bf16(biases[g.w >> 1u], g.w),
    );
    let wbase = rows * (n / cpw);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var xsum = vec2<f32>(0.0);
    let whole = k0 + u32(PIE_QMV_VPT) <= n;
    for (var jw = 0u; jw < u32(PIE_QMV_VPT) / u32(PIE_CODES_PER_WORD); jw = jw + 1u) {
        let base = k0 + jw * u32(PIE_CODES_PER_WORD);
        if base >= n {
            break;
        }
        let at = wbase + vec4<u32>(base / u32(PIE_CODES_PER_WORD));
        let word = vec4<u32>(w[at.x], w[at.y], w[at.z], w[at.w]);
        if PIE_NIBBLE_FAST && whole {
            let lo = vec4<u32>(0x0f0f0f0fu);
            let e0 = unpack4x8unorm(word.x & lo.x) * 255.0;
            let e1 = unpack4x8unorm(word.y & lo.y) * 255.0;
            let e2 = unpack4x8unorm(word.z & lo.z) * 255.0;
            let e3 = unpack4x8unorm(word.w & lo.w) * 255.0;
            let sh = (word >> vec4<u32>(4u)) & lo;
            let o0 = unpack4x8unorm(sh.x) * 255.0;
            let o1 = unpack4x8unorm(sh.y) * 255.0;
            let o2 = unpack4x8unorm(sh.z) * 255.0;
            let o3 = unpack4x8unorm(sh.w) * 255.0;
            // ONE ROW AT A TIME across the unpack, which is the whole
            // difference from the four-row attempt: eight activation floats
            // are live here, not thirty-two.
            let p0 = x_word(vecs.x, base);
            let p1 = x_word(vecs.x, base + 2u);
            let p2 = x_word(vecs.x, base + 4u);
            let p3 = x_word(vecs.x, base + 6u);
            let u0 = pie_bf16_to_f32(p0 & 0xffffu);
            let u1 = pie_bf16_to_f32(p0 >> 16u);
            let u2 = pie_bf16_to_f32(p1 & 0xffffu);
            let u3 = pie_bf16_to_f32(p1 >> 16u);
            let u4 = pie_bf16_to_f32(p2 & 0xffffu);
            let u5 = pie_bf16_to_f32(p2 >> 16u);
            let u6 = pie_bf16_to_f32(p3 & 0xffffu);
            let u7 = pie_bf16_to_f32(p3 >> 16u);
            xsum.x = xsum.x + u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7;
            a0 = a0 + u0 * vec4<f32>(e0.x, e1.x, e2.x, e3.x);
            a0 = a0 + u1 * vec4<f32>(o0.x, o1.x, o2.x, o3.x);
            a0 = a0 + u2 * vec4<f32>(e0.y, e1.y, e2.y, e3.y);
            a0 = a0 + u3 * vec4<f32>(o0.y, o1.y, o2.y, o3.y);
            a0 = a0 + u4 * vec4<f32>(e0.z, e1.z, e2.z, e3.z);
            a0 = a0 + u5 * vec4<f32>(o0.z, o1.z, o2.z, o3.z);
            a0 = a0 + u6 * vec4<f32>(e0.w, e1.w, e2.w, e3.w);
            a0 = a0 + u7 * vec4<f32>(o0.w, o1.w, o2.w, o3.w);
            let q0 = x_word(vecs.y, base);
            let q1 = x_word(vecs.y, base + 2u);
            let q2 = x_word(vecs.y, base + 4u);
            let q3 = x_word(vecs.y, base + 6u);
            let v0 = pie_bf16_to_f32(q0 & 0xffffu);
            let v1 = pie_bf16_to_f32(q0 >> 16u);
            let v2 = pie_bf16_to_f32(q1 & 0xffffu);
            let v3 = pie_bf16_to_f32(q1 >> 16u);
            let v4 = pie_bf16_to_f32(q2 & 0xffffu);
            let v5 = pie_bf16_to_f32(q2 >> 16u);
            let v6 = pie_bf16_to_f32(q3 & 0xffffu);
            let v7 = pie_bf16_to_f32(q3 >> 16u);
            xsum.y = xsum.y + v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
            a1 = a1 + v0 * vec4<f32>(e0.x, e1.x, e2.x, e3.x);
            a1 = a1 + v1 * vec4<f32>(o0.x, o1.x, o2.x, o3.x);
            a1 = a1 + v2 * vec4<f32>(e0.y, e1.y, e2.y, e3.y);
            a1 = a1 + v3 * vec4<f32>(o0.y, o1.y, o2.y, o3.y);
            a1 = a1 + v4 * vec4<f32>(e0.z, e1.z, e2.z, e3.z);
            a1 = a1 + v5 * vec4<f32>(o0.z, o1.z, o2.z, o3.z);
            a1 = a1 + v6 * vec4<f32>(e0.w, e1.w, e2.w, e3.w);
            a1 = a1 + v7 * vec4<f32>(o0.w, o1.w, o2.w, o3.w);
            continue;
        }
        for (var c = 0u; c < u32(PIE_CODES_PER_WORD); c = c + 2u) {
            if !whole && base + c >= n {
                break;
            }
            let k = base + c;
            let xw = vec2<u32>(x_word(vecs.x, k), x_word(vecs.y, k));
            let xv0 = vec2<f32>(
                pie_bf16_to_f32(xw.x & 0xffffu),
                pie_bf16_to_f32(xw.y & 0xffffu),
            );
            xsum = xsum + xv0;
            let code0 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * c)) & vec4<u32>(mask));
            a0 = a0 + xv0.x * code0;
            a1 = a1 + xv0.y * code0;
            if !whole && k + 1u >= n {
                break;
            }
            let xv1 = vec2<f32>(
                pie_bf16_to_f32(xw.x >> 16u),
                pie_bf16_to_f32(xw.y >> 16u),
            );
            xsum = xsum + xv1;
            let code1 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * (c + 1u))) & vec4<u32>(mask));
            a0 = a0 + xv1.x * code1;
            a1 = a1 + xv1.y * code1;
        }
    }
    return BlockM2(
        s * a0 + b * vec4<f32>(xsum.x),
        s * a1 + b * vec4<f32>(xsum.y),
    );
}

fn block_dot(rows: vec4<u32>, vecs: vec4<u32>, k0: u32) -> BlockM {
    let n = u32(params.in_vec_size);
    let cpw = u32(PIE_CODES_PER_WORD);
    let mask = (1u << u32(PIE_BITS)) - 1u;
    let g = rows * (n / u32(PIE_GROUP)) + vec4<u32>(k0 / u32(PIE_GROUP));
    let s = vec4<f32>(
        qmv_bf16(scales[g.x >> 1u], g.x),
        qmv_bf16(scales[g.y >> 1u], g.y),
        qmv_bf16(scales[g.z >> 1u], g.z),
        qmv_bf16(scales[g.w >> 1u], g.w),
    );
    let b = vec4<f32>(
        qmv_bf16(biases[g.x >> 1u], g.x),
        qmv_bf16(biases[g.y >> 1u], g.y),
        qmv_bf16(biases[g.z >> 1u], g.z),
        qmv_bf16(biases[g.w >> 1u], g.w),
    );
    let wbase = rows * (n / cpw);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var xsum = vec4<f32>(0.0);
    // CONST BOUNDS, both of them. Written with a `let` copy of the same
    // constant this ran 22% SLOWER than the loop it replaced, measured
    // isolated at [4096, 4096] -- a bound naga cannot fold is a loop it cannot
    // unroll, and an unrolled sixteen-step inner loop is most of what this
    // kernel is.
    //
    // The tail is hoisted out of the inner loop for the same reason: a block
    // wholly inside K takes the unguarded path, and only the last one per row
    // pays a comparison per value.
    let whole = k0 + u32(PIE_QMV_VPT) <= n;
    for (var jw = 0u; jw < u32(PIE_QMV_VPT) / u32(PIE_CODES_PER_WORD); jw = jw + 1u) {
        let base = k0 + jw * u32(PIE_CODES_PER_WORD);
        // The K tail, guarded on the WORD as well as the value: a row is
        // `n / cpw` words and reading one past the last row's last word is off
        // the end of the buffer, not merely off the end of a row.
        if base >= n {
            break;
        }
        let at = wbase + vec4<u32>(base / u32(PIE_CODES_PER_WORD));
        let word = vec4<u32>(w[at.x], w[at.y], w[at.z], w[at.w]);
        // TWO CODES A TRIP. Each activation word holds `k` and `k + 1`, so
        // walking k one at a time loaded every one of them twice -- and here
        // that is FOUR redundant loads a value, one per activation row. `base`
        // is a multiple of `PIE_CODES_PER_WORD` and `in_vec_size` is even, so
        // `k` is even and the low half is `k`. Same products, same order.
        for (var c = 0u; c < u32(PIE_CODES_PER_WORD); c = c + 2u) {
            if !whole && base + c >= n {
                break;
            }
            let k = base + c;
            // The four activation rows this workgroup carries, at one k. The
            // word above is read ONCE for all four, which is the saving.
            let xw = vec4<u32>(
                x_word(vecs.x, k),
                x_word(vecs.y, k),
                x_word(vecs.z, k),
                x_word(vecs.w, k),
            );
            let xv0 = vec4<f32>(
                pie_bf16_to_f32(xw.x & 0xffffu),
                pie_bf16_to_f32(xw.y & 0xffffu),
                pie_bf16_to_f32(xw.z & 0xffffu),
                pie_bf16_to_f32(xw.w & 0xffffu),
            );
            xsum = xsum + xv0;
            let code0 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * c)) & vec4<u32>(mask));
            a0 = a0 + xv0.x * code0;
            a1 = a1 + xv0.y * code0;
            a2 = a2 + xv0.z * code0;
            a3 = a3 + xv0.w * code0;
            if !whole && k + 1u >= n {
                break;
            }
            let xv1 = vec4<f32>(
                pie_bf16_to_f32(xw.x >> 16u),
                pie_bf16_to_f32(xw.y >> 16u),
                pie_bf16_to_f32(xw.z >> 16u),
                pie_bf16_to_f32(xw.w >> 16u),
            );
            xsum = xsum + xv1;
            let code1 = vec4<f32>((word >> vec4<u32>(u32(PIE_BITS) * (c + 1u))) & vec4<u32>(mask));
            a0 = a0 + xv1.x * code1;
            a1 = a1 + xv1.y * code1;
            a2 = a2 + xv1.z * code1;
            a3 = a3 + xv1.w * code1;
        }
    }
    // `sum_i x_i * (s * code_i + b)` regrouped: the bias rides the plain sum of
    // the activations, which is why `xsum` is worth carrying. Per activation
    // row, and in the same order the one-row form summed it.
    return BlockM(
        s * a0 + b * vec4<f32>(xsum.x),
        s * a1 + b * vec4<f32>(xsum.y),
        s * a2 + b * vec4<f32>(xsum.z),
        s * a3 + b * vec4<f32>(xsum.w),
    );
}

//#if !defined(PIE_WIDE_STRIDED)
fn reduce_store(lid: u32, ly: u32, wg: vec3<u32>, out0: u32, sg: u32) {
    let vec0 = wg.x * PIE_MT;

    // How many activation rows the fire actually bound.
    //
    // `affine_qmv_fast` states two scalars -- `in_vec_size` and
    // `out_vec_size` -- and neither is the batch. It does not have to be:
    // `binding::extent` binds an arena operand as exactly the rectangle the
    // launch covers, `rows * width * bytes`, so the bound length of `x` IS the
    // row count times the width, and `arrayLength` reads it back. Deriving it
    // here rather than adding a scalar keeps the row's operand list and every
    // other backend's uniform block alone.
    let m = max((arrayLength(&x) * 2u) / u32(params.in_vec_size), 1u);

    // Every lane runs all four columns and all `PIE_MT` rows and stores all
    // the partials, out of range or not, because the `barrier()` below is
    // next: an early return in front of a workgroup barrier is a HANG in WGSL,
    // not a wrong number. Out-of-range indices are CLAMPED rather than skipped
    // -- it keeps the lanes on the same trip count and the values are dropped
    // at the store, which already asks.
    let last = max(u32(params.out_vec_size), 1u) - 1u;
    // At `PIE_ROWW == 2` the upper pair REPEATS the lower one rather than
    // running off into the next workgroup's rows: `wbase` and `g` are the only
    // things `block_dot1` derives from this, so a repeat costs two redundant
    // word loads of an address the first pair just read and two redundant
    // unpacks, and produces lanes `.z`/`.w` that the tree below never walks.
    // Clamping to `last` instead would be wrong, not slow -- it would add row
    // `last`'s products into a partial the store then attributes to `out0`.
    var want = vec4<u32>(out0, out0 + 1u, out0 + 2u, out0 + 3u);
    if PIE_ROWW == 2u {
        want = vec4<u32>(out0, out0 + 1u, out0, out0 + 1u);
    }
    let rows = min(want, vec4<u32>(last));
    let vecs = min(
        vec4<u32>(vec0, vec0 + 1u, vec0 + 2u, vec0 + 3u),
        vec4<u32>(m - 1u),
    );
    // Rows this workgroup actually has, which is `PIE_MT` everywhere but the
    // last group of a ragged batch -- and ONE for every decode, where the fire
    // is a single row. Workgroup-uniform, so the loops it bounds may hold a
    // barrier; the k loop above cannot use it, because an unrolled `vec4` is
    // what makes that loop fast.
    // `max(m, vec0) - vec0` and not `m - vec0`: the subtraction is unsigned,
    // and a grid that dispatches MORE x groups than the batch needs -- which
    // is legal, and which every launcher in this tree used to do before
    // `quarters()` -- would wrap to four billion and walk the whole of both
    // loops below off the end. Zero is the right answer for a workgroup with
    // no rows, and it leaves every barrier here in uniform control flow.
    let mt = min(PIE_MT, max(m, vec0) - vec0);
    var t0 = vec4<f32>(0.0);
    var t1 = vec4<f32>(0.0);
    var t2 = vec4<f32>(0.0);
    var t3 = vec4<f32>(0.0);
    var k0 = lid * u32(PIE_QMV_VPT);
    // WORKGROUP-UNIFORM, and there is no barrier in either arm -- the one this
    // function has is below, after both have joined. The ragged tail of a
    // batch can still be a single row, and it takes the cheap body too.
    if mt == 1u {
        while k0 < u32(params.in_vec_size) {
            t0 = t0 + block_dot1(rows, vecs.x, k0);
            k0 = k0 + u32(PIE_QMV_VPT) * 32u;
        }
    } else if mt == 2u {
        while k0 < u32(params.in_vec_size) {
            let part = block_dot2(rows, vecs.xy, k0);
            t0 = t0 + part.a0;
            t1 = t1 + part.a1;
            k0 = k0 + u32(PIE_QMV_VPT) * 32u;
        }
    } else {
        while k0 < u32(params.in_vec_size) {
            let part = block_dot(rows, vecs, k0);
            t0 = t0 + part.a0;
            t1 = t1 + part.a1;
            t2 = t2 + part.a2;
            t3 = t3 + part.a3;
            k0 = k0 + u32(PIE_QMV_VPT) * 32u;
        }
    }
    // Slot `(r, mi)` of this y-slot is `((ly * 4 + r) * PIE_MT + mi)`. All
    // sixteen are written whether or not the group has four live rows: the
    // spare ones hold the zeros they were accumulated from, and the tree and
    // the store below walk only `mt` of them.
    let slot0 = 0u;
//#if defined(PIE_SUBGROUP)
    // FOLD THE THIRTY-TWO LANES HERE, IN REGISTERS, so the ladder below never
    // runs. The workgroup is `@workgroup_size(32, 1, 1)`, which on any adapter
    // whose subgroup is at least 32 wide is ONE subgroup, so a butterfly over
    // `off < lim` reaches every lane that holds a partial and no lane that
    // does not -- `lane ^ off` cannot leave the aligned `lim`-lane block.
    //
    // Why bother, when the ladder is five levels run once per workgroup rather
    // than once per key: because a LEVEL is what costs, not the adds in it.
    // `attn.rs` and `sdpa_paged.wgsl` establish that on this machine by
    // deleting three levels of a 63-add tree and getting a third of a kernel
    // back. Five levels here are five barriers and five workgroup-memory round
    // trips for every one of a fire's hundreds of workgroups, and this is the
    // per-output-row cost `PIE_ROWW`'s table could name but not explain.
    //
    // Every lane of a block leaves holding that block's sum, so the stores
    // below stay unconditional -- each lane writes its own slot and the
    // block's first slot is the one the final read walks, at stride `lim`. At
    // `lim == 32` that is one slot, one barrier, and no tree at all.
    let lim = min(32u, sg);
    for (var off = 1u; off < lim; off = off << 1u) {
        t0 = t0 + subgroupShuffleXor(t0, off);
        t1 = t1 + subgroupShuffleXor(t1, off);
        t2 = t2 + subgroupShuffleXor(t2, off);
        t3 = t3 + subgroupShuffleXor(t3, off);
    }
//#endif
    qmv_partials[(slot0 + 0u * PIE_MT + 0u) * 32u + lid] = t0.x;
    qmv_partials[(slot0 + 1u * PIE_MT + 0u) * 32u + lid] = t0.y;
    qmv_partials[(slot0 + 2u * PIE_MT + 0u) * 32u + lid] = t0.z;
    qmv_partials[(slot0 + 3u * PIE_MT + 0u) * 32u + lid] = t0.w;
    // The other three rows' slots, written only by a group that HAS them. A
    // decode writes four partials here, which is what it wrote before `PIE_MT`
    // existed; writing all sixteen cost it 3 tok/s of 64 and bought nothing.
    // ONE SLOT INDEX PER `mi`, GUARDED BY `mt` AND NOT BY `mt > 1` ALONE.
    // These used to be twelve stores under a single `if mt > 1u`, which is
    // right only while `PIE_MT` is 4: slot `(r, mi)` is `r * PIE_MT + mi`, so
    // an `mi` at or past `PIE_MT` is not out of bounds -- it is ROW `r + 1`'s
    // slot, written with row `r`'s partial. At `PIE_MT = 2` that silently
    // corrupted every other output column and cost three gpu tests and an
    // afternoon.
    if mt > 1u {
        qmv_partials[(slot0 + 0u * PIE_MT + 1u) * 32u + lid] = t1.x;
        qmv_partials[(slot0 + 1u * PIE_MT + 1u) * 32u + lid] = t1.y;
        qmv_partials[(slot0 + 2u * PIE_MT + 1u) * 32u + lid] = t1.z;
        qmv_partials[(slot0 + 3u * PIE_MT + 1u) * 32u + lid] = t1.w;
    }
    if mt > 2u {
        qmv_partials[(slot0 + 0u * PIE_MT + 2u) * 32u + lid] = t2.x;
        qmv_partials[(slot0 + 1u * PIE_MT + 2u) * 32u + lid] = t2.y;
        qmv_partials[(slot0 + 2u * PIE_MT + 2u) * 32u + lid] = t2.z;
        qmv_partials[(slot0 + 3u * PIE_MT + 2u) * 32u + lid] = t2.w;
    }
    if mt > 3u {
        qmv_partials[(slot0 + 0u * PIE_MT + 3u) * 32u + lid] = t3.x;
        qmv_partials[(slot0 + 1u * PIE_MT + 3u) * 32u + lid] = t3.y;
        qmv_partials[(slot0 + 2u * PIE_MT + 3u) * 32u + lid] = t3.z;
        qmv_partials[(slot0 + 3u * PIE_MT + 3u) * 32u + lid] = t3.w;
    }
    workgroupBarrier();

    // A TREE, where this used to be lane zero adding thirty-two partials for
    // each of four rows -- a hundred and twenty-eight serial adds with
    // thirty-one lanes of the y-slot watching. Five halvings do the same work
    // in five steps.
    //
    // The barrier is OUTSIDE the `if`, which is not a style choice: WGSL
    // requires `workgroupBarrier` in uniform control flow, and a barrier
    // reached by half a workgroup is undefined rather than slow.
//#if !defined(PIE_SUBGROUP)
    for (var half = 16u; half > 0u; half = half >> 1u) {
        if lid < half {
            for (var r = 0u; r < PIE_ROWW; r = r + 1u) {
                for (var mi = 0u; mi < mt; mi = mi + 1u) {
                    let at = (slot0 + r * PIE_MT + mi) * 32u + lid;
                    qmv_partials[at] = qmv_partials[at] + qmv_partials[at + half];
                }
            }
        }
        workgroupBarrier();
    }
//#endif

    if lid == 0u {
        for (var r = 0u; r < PIE_ROWW; r = r + 1u) {
            let row = out0 + r;
            if row >= u32(params.out_vec_size) {
                continue;
            }
            for (var mi = 0u; mi < mt; mi = mi + 1u) {
                let vec_ = vec0 + mi;
//#if defined(PIE_SUBGROUP)
                var sum0 = 0.0;
                for (var g = 0u; g < 32u; g = g + lim) {
                    sum0 = sum0 + qmv_partials[(slot0 + r * PIE_MT + mi) * 32u + g];
                }
//#else
                let sum0 = qmv_partials[(slot0 + r * PIE_MT + mi) * 32u];
//#endif
                var sum = sum0;
//#if defined(PIE_BIAS)
                // Unconditional, where the GLSL asked `if (tail)`: every
                // variant that defines PIE_BIAS also defines PIE_TAIL, so the
                // runtime flag could only ever be true.
                sum = sum + qmv_bf16(extra[row >> 1u], row);
//#endif
//#if defined(PIE_RESIDUAL)
                let at = vec_ * u32(params.out_vec_size) + row;
                // Rounded to bf16 BEFORE the add, which is not an accident:
                // it makes this bit-identical to the two-kernel path (project,
                // then `residual_add`) that the fused variant replaces.
                let q = pie_bf16_to_f32(pie_f32_to_bf16(sum));
                store_y(at, q + qmv_bf16(extra[at >> 1u], at));
//#else
                store_y(vec_ * u32(params.out_vec_size) + row, sum);
//#endif
            }
        }
    }
}
//#endif

//#if defined(PIE_WIDE_STRIDED)
// One row per lane over the whole of K, four lanes per y-slot. `PIE_K_LANES` is
// in the entrypoint name and read by nothing: Metal splits K over eight lanes
// and finishes with `simd_shuffle_down`, and neither Vulkan's port nor this one
// does -- the name records the shape the launcher picked, and the body is a
// serial K loop that gets the same answer with 4 of 32 lanes busy. There is no
// barrier in here, which is why the early returns below are safe.
fn wide_strided(lid: vec3<u32>, wg: vec3<u32>) {
    let row_slot = lid.x & 3u;
    if lid.x >= 4u {
        return;
    }
    let row = wg.y * 4u + row_slot;
    let vec0 = wg.x * u32(PIE_VEC);
    if row >= u32(params.out_vec_size) {
        return;
    }
    for (var v = 0u; v < u32(PIE_VEC); v = v + 1u) {
        let vec_ = vec0 + v;
        if vec_ < u32(params.m) {
            var acc = 0.0;
            for (var k = 0u; k < u32(params.in_vec_size); k = k + 1u) {
                acc = acc + x_at(vec_, k) * affine_value(row, k);
            }
            store_y(vec_ * u32(params.row_stride) + row, acc);
        }
    }
}
//#endif

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
//#if defined(PIE_SUBGROUP)
    @builtin(subgroup_size) sg: u32,
//#endif
) {
//#if !defined(PIE_SUBGROUP)
    let sg = 32u;
//#endif
//#if defined(PIE_WIDE_STRIDED)
    wide_strided(lid, wg);
//#else
    // PIE_ROWREP row-groups per workgroup, so the grid is 1/PIE_ROWREP of the
    // rows/4 it would otherwise be. A probe knob: see `PIE_ROWREP`.
    for (var rep = 0u; rep < PIE_ROWREP; rep = rep + 1u) {
        // The partials are reused between reps and the previous one's tree
        // read them, so the rewrite below has to wait. Uniform: the loop bound
        // is a constant.
        workgroupBarrier();
        let yg = wg.y + wg.z * PIE_YTILE;
        reduce_store(lid.x, lid.y, wg, (yg * PIE_ROWREP + rep) * PIE_ROWW, sg);
    }
//#endif
}

// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_RESIDUAL=1
// THE SAME TWO WITH THE TAIL LADDER TAKEN OUT, at the one quantization these
// checkpoints use. `serve::pick` takes these when the adapter has `SUBGROUP`
// and falls back to the lines above when it does not.
//
// `gs_64_b_4` only: it is what every MLX 4-bit checkpoint in the bench binds,
// it is 52% of a decode on its own, and a tier variant costs a pipeline. Mint
// the others the day a bench measures one.
// pie:instantiate affine_qmv_fast_bfloat16_gs_64_b_4 @subgroup PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_fast_residual_bfloat16_gs_64_b_4 @subgroup PIE_GROUP=64 PIE_BITS=4 PIE_RESIDUAL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_tail_bias_bfloat16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIAS=1 PIE_TAIL=1
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8 PIE_GROUP=64 PIE_BITS=4 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
// pie:instantiate affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8 PIE_GROUP=64 PIE_BITS=8 PIE_WIDE_STRIDED=1 PIE_VEC=4 PIE_K_LANES=8
