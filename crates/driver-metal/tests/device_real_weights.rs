//! **A real checkpoint's weights, through the generic executor, and what came
//! out.**
//!
//! `device_text_fire.rs` proves the fire executes against sentinels;
//! `device_checkpoint_names.rs` proves every name binds against a checkpoint.
//! Neither looks at a NUMBER, and the gap between them is where a driver hides
//! its worst defects: a fire that runs to completion over correctly-addressed
//! weights and computes nonsense is indistinguishable from a working one
//! unless somebody reads the output.
//!
//! So this reads the output. Not against a reference — that is the accuracy
//! gate's job and it wants one — but against the three failure modes that
//! account for most of the distance:
//!
//!   * **all zeros.** A projection told its extents are zero no-ops; a weight
//!     bound to an unwritten arena slot contributes nothing. Both leave the
//!     residual stream exactly as the embedding left it, or empty.
//!   * **non-finite.** A norm handed a zero epsilon divides by the root of the
//!     mean square alone; a NaN anywhere spreads to everything downstream
//!     within one layer.
//!   * **degenerate.** Every row identical means the per-token axis is not
//!     reaching the kernels — a launch whose grid collapsed, or a gather
//!     reading token 0 for every lane.
//!
//! None of those three is subtle and all three are invisible without a read.
//! Passing here is not correctness; it is the floor beneath which correctness
//! cannot be discussed.
//!
//! # Running it
//!
//! ```text
//! PIE_METAL_SMOKE_CHECKPOINT=<an MLX snapshot dir with a config.json> \
//!   cargo test -p driver-metal --features metal-4 --test device_real_weights \
//!   -- --include-ignored --test-threads=1
//! ```
//!
//! Both flags are load-bearing. `--test-threads=1` because twelve tests each
//! mapping an 18 GB checkpoint at once is a SIGKILL, not a slowdown.
//!
//! `--include-ignored` because every test here is `#[ignore]`d, and that is a
//! correction rather than a convenience. Each one used to open by reading the
//! environment variable and `return`ing with an `eprintln!` when it was
//! unset -- and libtest swallows a PASSING test's stderr. So the suite
//! reported `ok. 12 passed` in 0.00s, CI never sets the variable, and the
//! strongest gate in the crate reported twelve passes of nothing for as long
//! as it has existed. A test that reports the same result whether or not it
//! ran is not a test. `ignored` is the one word libtest has for "this did not
//! run", so the suite says it.
//!
//! Running it for the first time found what it was built to find: a real
//! gemma-4-31b decode leaves nineteen of twenty-two arena regions unwritten
//! across 1255 non-empty dispatches, and fills three with ~1e27. The same
//! twelve are green on a real llama, MLX token-for-token agreement included.
//! See `.wiki/driver/real-metal-north-star.md` §15.
//!
//! It found six defects in its first afternoon, and the last two are the ones
//! that argue for the file:
//!
//!   1. **No barrier between dispatches.** Metal does not order two dispatches
//!      in one compute encoder and the executor's loop emitted none. Three
//!      runs of one fire gave widest activations of 11.7, 23.1 and 4.5e12 --
//!      TWO OF THE THREE looked entirely plausible.
//!   2. **The readout's dtype.** The text said `F32`, `affine_qmv_fast` writes
//!      bfloat, and the logits came back exactly half zero.
//!   3. **Unzeroed arena and KV pool.** A fresh Metal buffer is usually zero
//!      and nothing promises it, so an attention read past what a fire wrote
//!      attended to whatever the allocator last held.
//!   4. **The single-row gather.** `embed_gather_4bit` reads `id[0]` and
//!      writes one row whatever grid it is handed, and the text picked it by
//!      CLASS -- but a decode of four requests is four rows. One readout lane
//!      of four held anything, and NOTHING ELSE WAS WRONG: every launch stated
//!      four rows, every grid covered them, and every other kernel read the
//!      row where the grid put it.
//!   5. **Contiguous attention over a paged pool.** The text chose by CLASS
//!      where the POOL's layout decides, so a decode walked
//!      `[page, token, head, dim]` with `sdpa_vector_decode`'s arithmetic.
//!   6. **`v_new` bound to nothing.** `dispatch::reorder` defaulted a row's
//!      output count to ONE, and `kv_append` names no `Out` -- it writes the
//!      POOL. So the last INPUT was taken for an output and `In(1)` had
//!      nothing to resolve to. The K pages filled, the V pages were zero in
//!      every layer, and the attention that read them answered zero without
//!      failing. The widest activation went 1.1 -> 14.75 when it was fixed,
//!      which is the difference between a residual stream and a rumour of
//!      one.
//!
//! Every statement in the first layer writes both rows now, attention
//! included. What is NOT established is that any of the numbers is the right
//! number -- that still wants a reference, and this gate is the floor beneath
//! it rather than a substitute for it.
//!
//! Three measurements track what is left, each pinned so it can only improve:
//! declared outputs nothing fills (**0**, was 5), readout lanes that hold
//! anything (**4** of 4, was 1), and the arena's non-zero share (**99%**, was
//! 26%).
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT`, the same variable the other
//! checkpoint tests take. Run against
//! `mlx-community/Llama-3.2-1B-Instruct-4bit`.
//!
//! # gpt-oss-20b: it agrees with MLX
//!
//! Measured 2026-08-16 against `mlx-community/gpt-oss-20b-MXFP4-Q4`, which
//! became runnable here the day `stage_plan_weights` stopped holding the
//! model twice (12.1 GB peak; the old path wanted about twice that and this
//! machine has 32 GB).
//!
//! `a_real_checkpoints_weights_produce_finite_varied_activations` passes:
//! **0 NaN and 0 inf** over 2,004,992 arena bytes in thirty regions, and
//! `one_token_at_position_zero_agrees_with_mlx` passes against the
//! `REFERENCES` row below -- argmax 11, span `[-8.25, 10.1875]`, where MLX
//! says token 11 at 10.173439 over `(-8.259116, 10.173439)`.
//!
//! This paragraph read "it loads now, and it NaNs" for six days and named
//! 909,207 NaNs downstream of the router, with the SwiGLU's `limit`/`alpha`
//! and the expert bank's padded row count as the two candidates to check
//! first. Neither was ever investigated under that heading. What closed it
//! was a sweep of the quantized kernels that found seventeen of them reading
//! whatever the previous dispatch had left in a slot the encoder never
//! wrote -- `0fc54bedb`, which was not looking for this and did not know it
//! had fixed it, because nobody had run this gate since.
//!
//! **That is the lesson worth keeping, and it is about the rig and not the
//! arithmetic:** a failure recorded in prose and not in a suite that runs is
//! a failure nobody learns has been fixed. The six days are not the cost of
//! the bug; they are the cost of the only record of it being a comment.
//!
//! # What is still open on gpt-oss: the routed prefill has no GEMM
//!
//! `attention_is_a_minority_of_a_long_prefill` **fails** on it, and the
//! profile beneath it says why in one line: at n=2048,
//! `mxfp4_qmv_routed_bias` takes **11,469 ms of 12,277 -- 93.4%** of the
//! fire. Every other statement in the model has a batched form and takes
//! its rows in one dispatch; the dense projections run
//! `affine_qmm_t_...bm_32_bn_32` at 290 ms for the same rows. The routed
//! leg has only the matvec, so a 2048-row prefill dispatches it 2048 times.
//!
//! So this is not an attention problem and the test's name is now the
//! misleading part: attention IS a minority (256 ms, 2.1%). The assertion it
//! makes still holds -- the failure is real and the number is the routed
//! GEMM's absence, which is a kernel that does not exist rather than a
//! parameter that is wrong.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use driver_metal::bind::encode::Pipelines;
use driver_metal::device::{Allocation, Context};
use driver_metal::layout::kv::Shape;
use driver_metal::layout::region::Region as _;
use driver_metal::lowering::dispatch::Geometry;
use driver_metal::lowering::executor::{FireTable, Resolver, Slice};
use driver_metal::lowering::frame::{Step, lower_step};
use driver_metal::lowering::resolve::{Names, Store};
use driver_metal::pools::kv::Pool;
use driver_metal::program::Compiler;
use driver_metal::weights::load::load;
use model::catalog::MetalBinding;
use model_ir::trace::{FireClass, ForwardPlan};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn snapshot() -> Option<PathBuf> {
    std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT").map(PathBuf::from)
}

/// Two bf16 ulps at `v`, which is what "a tolerance about the FORMAT" means
/// once the readout is not llama's.
///
/// The tolerance here was `0.05` — about two ulps at SIX, which is where
/// llama-3.2's top logit sits. gemma-4's top logit sits at THIRTY, where one
/// ulp is `0.125`, so the same absolute number is a fifth of an ulp and
/// demands of a bf16 readout a precision bf16 does not have. A constant
/// tolerance is a tolerance calibrated on one checkpoint's magnitudes.
///
/// bf16 keeps 8 significand bits, so the gap between neighbours at `v` is
/// `2^(exponent(v) - 7)`.
fn bf16_slack(v: f32) -> f32 {
    if v == 0.0 {
        return f32::EPSILON;
    }
    2.0 * (v.abs().log2().floor() - 7.0).exp2()
}

/// One hand-taken MLX measurement: which checkpoint, which token, and the
/// answer MLX gave for it at position zero.
///
/// A table rather than a constant, because "the reference" was never one
/// thing. Every entry is a MEASUREMENT someone took once, and the checkpoint
/// it was taken on is part of it -- so the honest structure is a row per
/// checkpoint and a lookup, not a constant plus a skip.
struct Reference {
    /// Matched against `PIE_METAL_SMOKE_CHECKPOINT`'s path.
    model: &'static str,
    /// The one token the fire posts. A family's own, not a shared constant:
    /// llama-3.2 begins at 128000 and gemma-4 at 2.
    bos: u32,
    /// MLX's top five, in order, with the logits it gave them.
    ///
    /// **TAKEN IN f32**, by `model.set_dtype(mx.float32)` before the forward,
    /// and this is not a detail. gpt-oss-20b's row was first taken in MLX's
    /// default dtype for that checkpoint, which is bf16, and the gate then
    /// read this driver as wrong at rank five: MLX said 279 and the driver
    /// said 198.
    ///
    /// MLX's OWN top five moves under the change:
    ///
    /// ```text
    ///   bf16  11 9.9375   1 9.875    7 9.25     13 9.25    279 9.125
    ///   f32   11 10.1734  1 9.8654   7 9.5159   12 9.1921  198 9.1736
    /// ```
    ///
    /// and the driver's answer -- 11 10.1875, 1 9.9375, 7 9.5, 198 9.25,
    /// 13 9.1875 -- is the f32 column rounded to bf16, everywhere, to within
    /// an ulp. Its span `[-8.25, 10.1875]` is `bf16(-8.259116)` and
    /// `bf16(10.173439)` EXACTLY.
    ///
    /// So the disagreement was two bf16 roundings of one f32 answer being
    /// compared to each other. A reference is the value both sides
    /// approximate, and comparing one approximation to another measures the
    /// two roundings and nothing else -- with a bf16 tolerance on top, which
    /// then has to cover both.
    ///
    /// f32 also makes the tolerance mean what its comment says it means: the
    /// driver's readout is bf16, so `bf16_slack` is a statement about the
    /// DRIVER'S format against a reference that has none.
    top: [(usize, f32); 5],
    /// The SIXTH logit, where someone has taken it.
    ///
    /// Not for comparing -- for knowing how deep the top-five claim goes. A
    /// set claim over five needs the fifth separated from the sixth by more
    /// than the readout can resolve, and only this number says whether it is.
    /// gpt-oss's fifth and sixth are 9.1736 and 9.1306, two thirds of a bf16
    /// ulp apart: which of them a bf16 readout ranks fifth is a property of
    /// the rounding.
    next: Option<f32>,
    /// The whole distribution's span, because five agreeing logits at the top
    /// is consistent with a distribution that is wrong everywhere else.
    span: (f32, f32),
    /// Three tokens well DOWN the distribution, with MLX's logit for each.
    ///
    /// The span was carrying this weight and cannot: "five agreeing logits at
    /// the top is consistent with a distribution that is wrong everywhere
    /// else" is exactly right, and the answer to it is a reading from
    /// somewhere else -- not the two most extreme values, which on a capped
    /// readout are the two the cap has erased. gemma's floor is ONE token
    /// (1852, at -27.75; the next is -19.75), and it sits where `tanh`'s
    /// slope is 0.14.
    ///
    /// Ranks 100, 1000 and 10000 out of a 262144-token vocabulary, far from
    /// the cap, where a bf16 ulp is a bf16 ulp and a disagreement is a
    /// disagreement.
    mid: [(usize, f32); 3],
    /// The widest activation MLX's OWN forward reaches on this checkpoint.
    ///
    /// A saturation bound is a measurement too, and this one was a constant:
    /// `1e3`, with the comment "a llama-1B decode measures its widest
    /// activation under one, so the bound sits orders of magnitude out".
    /// That is true of llama-1B and says nothing about anything else.
    ///
    /// gpt-oss-20b's residual stream reaches 42,752 in MLX -- measured, by
    /// walking its own layers and taking `max(abs(h))` after each. It is at
    /// 111 after layer four and at 42,496 after layer six, which is the
    /// massive-activation channel this family is known for and not a
    /// saturation. A driver reading 12,672 there was reported as "saturation
    /// or silence rather than a forward pass" by a bound forty times too
    /// tight.
    ///
    /// llama-3.2-1B's is 410.5, and this field first held `1.0` because that
    /// was what the old comment SAID -- "measures its widest activation under
    /// one" -- transcribed instead of measured. The driver's arena tops out
    /// at 410 on the same fire, one bf16 ulp from MLX's 410.5, so the number
    /// the comment gave was wrong by a factor of four hundred and the
    /// checkpoint agrees with MLX far more sharply than either said.
    ///
    /// So the number rides the checkpoint, like every other number here --
    /// and `None` where nobody has taken it, which is not the same as a
    /// small one.
    widest: Option<f32>,
    /// `final_logit_softcapping`, or zero for a readout that has none.
    ///
    /// A LOGIT AT THE CAP IS A LOGIT THE CAP HAS ERASED. `cap * tanh(x/cap)`
    /// has derivative `1 - (v/cap)^2`, which at gemma's top logit -- `v` of
    /// 29.875 against a cap of 30 -- is 0.008: a hundred and twenty units of
    /// pre-cap logit arrive as one unit of post-cap value. Comparing two
    /// implementations there measures `tanh`'s asymptote and not the fire, so
    /// the value is not compared and the TOKEN still is.
    cap: f32,
    /// The fraction of a PRE-CAP logit that this row's own ROUTING moves,
    /// and zero for a row that does not route.
    ///
    /// A top-k router is a DISCRETE function of a continuous score. gemma-4's
    /// picks 8 of 128, and the eighth and ninth scores are routinely closer
    /// together than bf16 can separate, so two implementations that agree on
    /// every score to within a rounding still send some token through a
    /// different expert -- once per layer is enough, and there are thirty.
    ///
    /// This is not a bound guessed to make a gate pass. It was MEASURED, by
    /// perturbing MLX'S OWN router logits by 0.2% and rerunning its own
    /// forward, which is a change far smaller than the difference between two
    /// correct kernels:
    ///
    /// ```text
    ///   token            3643    31836  236958    6367   77902   40308
    ///   MLX + 0.2%     0.9749   0.9694  0.9587  0.9612  0.9273  0.9398
    ///   this driver    0.9628   0.9633  0.9455  0.9538  0.9234  0.9326
    /// ```
    ///
    /// -- pre-cap, against MLX's unperturbed self. The driver sits INSIDE the
    /// band MLX reaches by disagreeing with itself, and 1% of noise lands in
    /// the same place as 0.2%: the routing has already flipped everything a
    /// near-tie can flip, so the displacement saturates rather than growing.
    ///
    /// What does NOT move is the ORDER. Across seeds the argmax stays 3643
    /// and the separated top three stay `{3643, 1082, 31836}`; only ranks
    /// four and five trade with a sixth token that was already inside a bf16
    /// ulp of them. So this row is gated exactly there -- the set and the
    /// argmax held to the letter, the VALUES held to what routing leaves.
    ///
    /// A dense row carries zero here and is unaffected. gemma-4-31b's top
    /// five still have to land within a bf16 ulp of MLX, and do.
    routing: f32,
}

/// Taken by hand with `mlx_lm.utils.load`, one forward over `[[bos]]`, the
/// last row of the logits read as f32.
///
/// GEMMA'S TOP LOGIT IS EXACTLY 30.0, which is `final_logit_softcapping`.
/// MLX saturates it too, so the saturation this crate kept re-reading as a
/// symptom is the model's own design and not a defect -- `tanh` reaches its
/// asymptote in bf16 long before the cap is loose.
const REFERENCES: &[Reference] = &[
    Reference {
        model: "Llama-3.2-1B-Instruct-4bit",
        bos: 128_000,
        top: [
            (16309, 6.406_25),
            (2, 5.949_219),
            (1757, 5.859_375),
            (791, 5.781_25),
            (475, 5.601_562),
        ],
        next: Some(5.427_544),
        span: (-4.613, 6.406),
        mid: [(111_912, 3.595_703), (34917, 2.455_078), (22631, 1.400_391)],
        widest: Some(410.5),
        cap: 0.0,
        // Dense. Nothing here routes.
        routing: 0.0,
    },
    // gpt-oss-20b, and the first routed checkpoint in the table: every
    // number above was measured on a DENSE mlp, so nothing here had ever
    // held a mixture's answer to account. Taken the same way -- `mlx_lm`
    // load, one forward over `[[bos]]`, the last logit row as f32.
    //
    // Its `bos` is 199998, which is not 128000 and not 2: `o200k_harmony`
    // numbers its specials at the END of a 201088-token vocabulary.
    //
    // No softcap. The top five sit within a fifth of a logit of each other
    // and four of the five are punctuation and articles, which is what an
    // unconditioned start-of-text row looks like when nothing has been read
    // yet -- so the MID ranks carry most of the weight here.
    Reference {
        model: "gpt-oss-20b-MXFP4-Q4",
        bos: 199_998,
        top: [
            (11, 10.173_439),
            (1, 9.865_355),
            (7, 9.515_914),
            (12, 9.192_102),
            (198, 9.173_553),
        ],
        next: Some(9.130_61),
        span: (-8.259_116, 10.173_439),
        mid: [(39985, 6.580_502), (43065, 4.287_484), (10303, 1.586_527)],
        widest: Some(42_752.0),
        cap: 0.0,
        // gpt-oss ROUTES -- 4 of 32 -- and still lands within a bf16 ulp of
        // MLX at every rank this gate reads. So a mixture does not get the
        // band for being a mixture: it gets it for having near-ties, and
        // thirty-two experts chosen four at a time do not produce them the
        // way a hundred and twenty-eight chosen eight do. Zero, and it holds.
        routing: 0.0,
    },
    Reference {
        model: "gemma-4-31b-it-4bit",
        bos: 2,
        top: [
            (236_773, 30.0),
            (236_798, 29.875),
            (236_780, 29.875),
            (236_799, 29.75),
            (236_814, 29.625),
        ],
        next: None,
        span: (-27.75, 30.0),
        mid: [(10541, 22.5), (18373, 19.25), (223_439, 14.75)],
        // Not measured. It has been passing under the loose bound below,
        // which says only that it is under a thousand.
        widest: None,
        cap: 30.0,
        // Dense gemma. Held to the ulp.
        routing: 0.0,
    },
    // The first MIXTURE gemma in the table, and the row this driver refused
    // to serve until the join was written. Its logits sit well under the cap
    // -- top 21.625 against 30 -- so unlike the dense 31b nothing here is
    // erased by `tanh` and the values are compared directly.
    //
    // Ranks five and six TIE at 20.5, which the depth scan above resolves by
    // claiming only the top four. A table that asserted all five would be
    // asking this driver to break a tie MLX did not.
    Reference {
        model: "gemma-4-26b-a4b-it-4bit",
        bos: 2,
        top: [
            (3643, 21.625),
            (1082, 21.0),
            (236_958, 20.75),
            (31836, 20.75),
            (197, 20.5),
        ],
        next: Some(20.5),
        span: (-11.125, 21.625),
        mid: [(142_507, 10.25), (165_767, 8.375), (196_731, 5.812_5)],
        widest: None,
        cap: 30.0,
        // 8 of 128, and the near-ties that come with it. See the field.
        // Measured displacement is 7.7% at its worst; this is that, rounded
        // up to a tenth, and NOT the smallest number that passes.
        routing: 0.10,
    },
];

/// The reference for whichever checkpoint the runner has, or `None` with a
/// SKIP naming what is missing.
fn reference_for(snapshot: &Path) -> Option<&'static Reference> {
    let path = snapshot.to_string_lossy();
    let found = REFERENCES.iter().find(|r| path.contains(r.model));
    if found.is_none() {
        eprintln!(
            "SKIP: no MLX reference has been measured for `{}`. Taking one is \
             a `mlx_lm` load, one forward over `[[bos]]`, and a row in \
             `REFERENCES`.",
            snapshot.display()
        );
    }
    found
}

/// Whether a hardcoded reference was measured on THIS checkpoint.
///
/// A reference is a MEASUREMENT: someone ran MLX once, by hand, over one
/// snapshot, and copied the numbers in. `PIE_METAL_SMOKE_CHECKPOINT` names
/// whichever checkpoint the runner happens to have, and the two are unrelated
/// -- so a test that states llama-3.2-1B's top five and runs against
/// gemma-4-31b asserts "MLX says token 16309" about a number MLX was never
/// asked for. It fails, which is worse than not running: it names a defect in
/// the driver for a difference that is entirely in the rig, and it does so in
/// the same report as the failures that ARE real.
///
/// So the reference states its checkpoint and the fire is skipped otherwise.
/// SKIPPED and not passed: nothing was checked, and a suite that says
/// "12 passed" about a checkpoint it never compared is the same lie in the
/// other direction. Taking the reference for a second checkpoint is a
/// half-hour of `mlx_lm` and a second table beside the first.
fn reference_taken_on(snapshot: &Path, model: &str) -> bool {
    let taken = snapshot.to_string_lossy().contains(model);
    if !taken {
        eprintln!(
            "SKIP: the reference in this test was measured on `{model}` and \
             PIE_METAL_SMOKE_CHECKPOINT is `{}`. Comparing a different \
             checkpoint against it would report the rig as a driver defect.",
            snapshot.display()
        );
    }
    taken
}

/// WHICH MODEL a snapshot is, at what affine point, and in what shape.
///
/// Eleven tests below open the same checkpoint and every one of them used to
/// spell the same four lines: normalize `config.json` into a `pie.model/1`
/// descriptor, parse the descriptor back into a private `ModelFacts`, project
/// THAT into a `DecodeGeometry`, and hand the descriptor to the loader as
/// well. Four steps, three intermediate documents, and eleven copies of the
/// sequence — so a change to any of it was eleven edits and the first one
/// missed was a test comparing a real GPU's output against the wrong shape.
///
/// It is one step now and it is worth naming what that step IS: the
/// checkpoint's TENSORS pick a `model::catalog` row, and everything else is a
/// projection of the row. No document is believed at all now, the
/// quantization included.
///
/// It used to be believed for exactly that, on the argument that a row
/// genuinely cannot state it "since `mlx-community` publishes the same
/// weights at 4 bits group 64 and at 8 bits group 32 and the two pack to
/// shapes no extent distinguishes". The row still cannot state it — it is
/// the checkpoint's and not the model's — but the second half is false, and
/// twice: `scales` is `[rows, cols / group]`, so those two differ by 2x
/// there, and the packed `weight` differs by 2x again. `LoadPlan` performs
/// both divisions per tensor and `Loaded::affine_point` reads the answer,
/// which is why the load moved ABOVE the projection here. `gpt-oss-20b`
/// declares `g32/b4` and holds not one tensor at it.
///
/// A refusal here PANICS rather than skipping. These are the A/B tests: they
/// are the only place a real Metal device's numbers are compared against
/// anything, and a skip that prints to stderr is how that comparison quietly
/// stops happening.
fn served(
    context: &Context,
    snapshot: &Path,
) -> Option<(
    &'static dyn model::catalog::Variant,
    driver_metal::batch::DecodeGeometry,
    driver_metal::weights::load::Loaded,
)> {
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(snapshot)
        .unwrap_or_else(|e| panic!("{} did not read as a checkpoint: {e:?}", snapshot.display()));
    let row = model::catalog::identify(&meta, &model::catalog::Override::None)
        .unwrap_or_else(|e| panic!("{}: {e}", snapshot.display()));
    // The DRIVER'S OWN pre-staging gate, asked here for the same reason
    // `serve/load.rs` asks it before staging: everything below this line
    // reads seventeen gigabytes off the disk and onto a device.
    //
    // A panic and not a SKIP, which is the opposite of `reference_taken_on`
    // just above, and the difference is whose fault it is. A missing MLX
    // reference is a gap in the rig — nobody measured that checkpoint — and
    // skipping states that honestly. A row this build states no Metal text
    // for is a MISCONFIGURED RUNNER: `load_model` would refuse the same
    // snapshot at the same question, so there is no version of this suite
    // that checks anything on it, and quietly printing SKIP twelve times
    // would read as "the device tests ran".
    //
    // `REFERENCES` carries `gemma-4-31b-it-4bit`, and this paragraph used to
    // say that row refused Metal because its text was `gemma4_cuda`. It does
    // not, and the row says so itself: "does NOT refuse: this build reads V
    // out of the K projection, and the 31b reproduces MLX's logits exactly
    // doing it". `LlamaLikeMetalFacts::v_from_k` is that read. Measured, not
    // argued — pointed at the snapshot, this gate passes and the load gets
    // as far as the arena.
    //
    // Where it stops is the THIRD case, handled at the `load` below: the
    // checkpoint is 17,269,186,048 resident bytes and this machine's ceiling
    // is 10,319,036,416. That is neither a gap in the rig nor a
    // misconfigured runner — it is the device, and the driver refusing it by
    // name is the behaviour under test.
    driver_metal::model::binding::serves(row).unwrap_or_else(|e| {
        panic!(
            "`{}` is `{}`, and this build states no Metal text for it: {e}. \
             `load_model` refuses this snapshot before staging, so nothing \
             below could run against it — point PIE_METAL_SMOKE_CHECKPOINT at \
             a checkpoint this build serves.",
            snapshot.display(),
            row.id()
        )
    });
    let config =
        match model_loader::checkpoint::read::read_meta(&meta, model::encoding::CONFIG_OBJECT) {
            Ok(Some(bytes)) => String::from_utf8(bytes).expect("the embedded config is utf8"),
            _ => std::fs::read_to_string(snapshot.join("config.json"))
                .unwrap_or_else(|e| panic!("{}/config.json: {e}", snapshot.display())),
        };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .unwrap_or_else(|e| panic!("{}: no encoding in the config: {e}", snapshot.display()));
    let deployment = row
        .deployment(model::catalog::Deployed::single())
        .unwrap_or_else(|e| panic!("`{}` does not deploy: {e}", row.id()));
    let loaded = match load(context, snapshot, row, &encoding) {
        Ok(loaded) => loaded,
        // THE THIRD CASE, and the doctrine above knew only two. This one
        // is not a gap in the rig and not a misconfigured runner: it is
        // the DRIVER BEHAVING, and the rig reading a pass as a failure.
        //
        // `weights::stage::fits_on_this_gpu` compares the plan's resident
        // bytes against the device's ceiling and refuses BEFORE staging,
        // naming both numbers, precisely so the kernel is not left to
        // decide. gemma-4-31b-it-4bit is 17,269,186,048 bytes and this
        // machine's ceiling is 10,319,036,416. Nothing about that is a
        // defect in anything, and panicking on it accuses `driver-metal`
        // of the one thing it got right.
        //
        // Narrow on purpose: `what` is the arena's, so every other
        // `Create` still panics. And loud on purpose -- the fear the
        // doctrine above is written against is twelve quiet SKIPs
        // reading as "the device tests ran", so this one says whose
        // limit it is and by how much.
        Err(driver_metal::Error::Create {
            what: "weight arena",
            message,
        }) => {
            eprintln!(
                "SKIP: THIS DEVICE cannot hold `{}` -- not a driver defect \
                 and not a missing measurement. {message}",
                snapshot.display(),
            );
            return None;
        }
        Err(e) => panic!("the checkpoint loads: {e:?}"),
    };
    // THE SECOND PRE-STAGING GATE, and a panic for the reason the first one
    // is: `load_model` refuses this snapshot at this exact question, so
    // there is no version of this suite that checks anything on it.
    //
    // It is asked AFTER staging rather than before, because it is the only
    // one of the two that the bytes answer and the documents cannot.
    let quant = loaded.affine_point(row.id()).unwrap_or_else(|e| {
        panic!(
            "`{}` is `{}`: {e}. `load_model` refuses this snapshot at the same \
             question, so nothing below could run against it — point \
             PIE_METAL_SMOKE_CHECKPOINT at a checkpoint this build serves.",
            snapshot.display(),
            row.id()
        )
    });
    let dg = driver_metal::batch::geometry_from_deployment(&deployment, row.load_shape(), quant)
        .unwrap_or_else(|e| panic!("`{}` projects no decodable geometry: {}", row.id(), e.0));
    Some((row, dg, loaded))
}

/// Every arena region the lowering states, each byte in exactly one of them.
///
/// # Why this is not `lowered.args` filtered
///
/// Two things about `Arg::Arena` make the stated tuple the wrong span to read.
///
/// **`width * bytes` is one ROW.** A decode's value is `rows` of them, so a
/// reader that takes the stated width looks at the FIRST TOKEN's slice and
/// calls it the region. It then cannot tell "nothing wrote this" from "the
/// write landed at the wrong offset inside it" -- and on gemma-4-31b it
/// cannot see the failure at all, because the embedding gather's output is
/// three rows of zero and one row of NaN and the NaN is not the first row.
///
/// **An arena REUSES offsets.** That is what an arena is for: the same `at`
/// carries a different tensor at different points in the schedule, so one
/// start appears several times with different widths, and a stated span can
/// run past where the next region begins. A reader that walks the list in
/// order reads those bytes once per descriptor. The arithmetic says so
/// plainly -- gemma-4-31b's arena holds 2,496,512 bf16 values and the census
/// reported 2,673,664 of them.
///
/// So: distinct starts, each given the bytes up to the next one. Where
/// descriptors at one start disagree about element width the widest wins,
/// because reading an f32 as bf16 hands back the low sixteen bits of every
/// value and those bit patterns are NaN about as often as not.
///
/// `skip` names starts to leave out AFTER the boundaries are computed, which
/// is how the NaN detector drops integer values without moving the edges of
/// the float regions beside them.
fn arena_regions(
    lowered: &model_compiler::lower::Lowered,
    arena_len: usize,
    skip: &[usize],
) -> Vec<(usize, usize, usize)> {
    let mut by_start: BTreeMap<usize, usize> = BTreeMap::new();
    for a in &lowered.args {
        if let model_compiler::lower::Arg::Arena { at, bytes, .. } = a {
            let e = by_start.entry(*at).or_insert(*bytes as usize);
            *e = (*e).max(*bytes as usize);
        }
    }
    let starts: Vec<usize> = by_start.keys().copied().collect();
    let mut out = Vec::with_capacity(starts.len());
    for (i, at) in starts.iter().enumerate() {
        let end = starts
            .get(i + 1)
            .copied()
            .unwrap_or(arena_len)
            .min(arena_len);
        if *at >= end || skip.contains(at) {
            continue;
        }
        out.push((*at, end - at, by_start[at]));
    }
    out
}

/// How many values a region of `len` bytes holds at `element` bytes each.
fn len_in_elements(len: usize, element: usize) -> usize {
    len.checked_div(element).unwrap_or(0)
}

/// What a run of the whole arena found.
#[derive(Debug, Default)]
struct Census {
    finite_nonzero: usize,
    zero: usize,
    nan: usize,
    inf: usize,
    /// The widest magnitude seen, which says whether anything saturated.
    max_abs: f32,
}

/// Count what is in `bytes`, read at `element` bytes per value.
///
/// The element width is NOT a constant over an arena, and assuming it was is
/// the first thing this gate got wrong about itself. 89% of a llama-1B
/// decode's arena is the readout, which is `DType::F32`; the rest is the
/// residual stream, which is bf16. Reading the f32 half as bf16 reports the
/// LOW sixteen bits of every logit as a number, which came out as 5.8e11 and
/// looked exactly like saturation.
///
/// `Arg::Arena` states `bytes` per element for precisely this reason -- its
/// own doc says a driver that windows a rectangle needs the stride and that
/// every hand windowing in the CUDA executor multiplied by two -- so the
/// census asks the lowering rather than guessing.
fn census(bytes: &[u8], element: usize) -> Census {
    let mut c = Census::default();
    for chunk in bytes.chunks_exact(element) {
        let v = if element == 4 {
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            // A bf16 is the TOP half of an f32, so widening is a shift.
            f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
        };
        if v.is_nan() {
            c.nan += 1;
        } else if v.is_infinite() {
            c.inf += 1;
        } else if v == 0.0 {
            c.zero += 1;
        } else {
            c.finite_nonzero += 1;
            c.max_abs = c.max_abs.max(v.abs());
        }
    }
    c
}

/// The pool this checkpoint's own geometry states, with `pages` per layer.
///
/// **Read off `DecodeGeometry`, exactly as `serve::load` reads it.** Every one
/// of the ten fires in this file used to build its own `Shape` with
/// `global_head_dim: 0, global_kv_heads: 0, full_attn_every: 0` -- which is
/// "one attention shape for the whole stack", true of every family here but
/// gemma-4. gemma-4 states a SECOND shape for its full-attention layers (head
/// dim 512 against the sliding layers' 256, a quarter the KV heads) and one
/// layer in six is full, so a pool laid out at 256 hands `sdpa_..._d_512` a
/// layer half as wide as it reads. Layers 5 and 11 survive it; layer 17 is far
/// enough in that the over-read leaves the allocation, and the fire's first
/// NaN was at statement 358 of layer 17 -- in the rig, not in the driver.
///
/// A gate that builds its own geometry does not test the driver's. This asks
/// the same source, so a deployment whose shape this file has never seen is
/// laid out right without a line being added here.
fn pool_shape(dg: &driver_metal::batch::DecodeGeometry, pages: u32) -> Shape {
    Shape {
        layers: dg.n_layers,
        kv_heads: dg.n_kv_heads,
        head_dim: dg.head_dim,
        page_size: 16,
        pages,
        element_bytes: 2,
        global_head_dim: dg.global_head_dim,
        global_kv_heads: dg.global_kv_heads,
        full_attn_every: dg.full_attn_every,
    }
}

/// What THIS load observed, exactly as `serve/load.rs` observes it.
///
/// Eleven gates below used to call `model::text::facts_from(&dg, |t| …)` and
/// get back twenty-nine model facts rebuilt from nine tensor probes. The
/// probes are gone: a checkpoint states its identity ONCE, when
/// `catalog::identify` reads its tensors in [`served`], and everything the
/// text needs after that is the row's to state.
///
/// What is left is genuinely the load's: the affine point the bytes are
/// packed at, and whether the expert bank arrived still in MXFP4. A gate that
/// re-derived either of those would be testing its own arithmetic — this asks
/// the driver's own [`observed`](driver_metal::model::binding::observed), so a
/// binding that changes shape breaks here the same way it breaks in `serve`.
fn observed(
    dg: &driver_metal::batch::DecodeGeometry,
    loaded: &driver_metal::weights::load::Loaded,
) -> MetalBinding {
    driver_metal::model::binding::observed(
        dg.quant,
        |t| loaded.affine_point_of(t),
        |t| loaded.mxfp4.contains(t),
    )
}

/// The row's own Metal text, through the one door the driver uses.
///
/// This was `llama_like_metal(&facts, &metal, class)` — a family's forward
/// function called directly, with facts this file had assembled. Two things
/// were wrong with it and both were load-bearing. It named a FAMILY, so a
/// checkpoint of any other family got llama's text and the gate below then
/// compared a real GPU's numbers against the wrong program. And it went
/// around `Variant::trace`, so the text under test was never the text
/// `serve/launch.rs` runs — which is the only text worth pointing a real
/// device at.
///
/// A refusal PANICS here for the same reason [`served`] panics: these gates
/// are the only place this workspace compares Metal's arithmetic against a
/// reference, and a silent skip is how that comparison stops happening.
fn text(
    row: &'static dyn model::catalog::Variant,
    class: FireClass,
    binding: &MetalBinding,
) -> ForwardPlan {
    driver_metal::model::binding::text(row, class, binding).unwrap_or_else(|e| {
        panic!(
            "`{}` states no metal {class:?} text: {e}. `served` gates on the \
             decode text, so reaching this means the row answers one fire \
             class and not the other",
            row.id()
        )
    })
}

/// The dispatch geometry every fire below runs under.
///
/// Eleven copies of this literal used to read `facts.q_heads`, `facts.head_dim`
/// and four more off the rebuilt facts. They read the DEPLOYMENT's projection
/// now, which is where those numbers were always from — `facts_from` copied
/// them out of the same `DecodeGeometry` this takes — so nothing about the
/// dispatch moved, only the number of documents it is read from.
fn dispatch_geometry(dg: &driver_metal::batch::DecodeGeometry, binding: &MetalBinding) -> Geometry {
    Geometry {
        q_heads: dg.n_q_heads,
        kv_heads: dg.n_kv_heads,
        head_dim: dg.head_dim,
        rotary_dims: dg.head_dim,
        n_experts: dg.n_experts,
        experts_per_token: dg.experts_per_token,
        // The quantization axes come from the BINDING, not the pool: they are
        // facts about the checkpoint on disk, and `serve` reads them from the
        // same place.
        group: binding.quant_group,
        bits: binding.quant_bits,
        // The five below are the axes a row states TWICE, and they are read
        // from the same two documents `serve/launch.rs` reads them from. A
        // literal here would be a third answer: this file points a real GPU
        // at the driver, so a geometry it assembles differently is a
        // comparison against a program `serve` never runs.
        global_head_dim: dg.global_head_dim,
        global_kv_heads: dg.global_kv_heads,
        full_attn_every: dg.full_attn_every,
        router_group: binding.router_quant_group,
        router_bits: binding.router_quant_bits,
    }
}

/// The checkpoint's weights, the fire's tables, and the pool's geometry.
struct Live<'a> {
    store: Store<'a>,
    tables: &'a driver_metal::bind::tables::Staged,
    shape: Shape,
    pages: &'a dyn Fn(u16, bool) -> Option<Slice>,
}

impl Resolver for Live<'_> {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        self.store.weight(name)
    }
    fn named(&mut self, value: model_ir::trace::ValueId) -> Option<Slice> {
        self.store.named(value)
    }
    fn kv(&mut self, layer: u16, values: bool) -> Option<Slice> {
        (self.pages)(layer, values)
    }
    fn fire(&mut self, which: FireTable) -> Option<Slice> {
        self.tables.at(which)
    }
    fn pool(&mut self, which: FireTable) -> Option<u32> {
        Some(match which {
            FireTable::KvHeadStride => self.shape.head_dim,
            FireTable::KvSeqStride => self.shape.kv_heads * self.shape.head_dim,
            FireTable::KvPageSize => self.shape.page_size,
            _ => return None,
        })
    }
}

/// How many dispatches the fire plans, and how many have an empty grid.
fn plan_count(
    lowered: &model_compiler::lower::Lowered,
    dg: &driver_metal::batch::DecodeGeometry,
    binding: &MetalBinding,
    live: &mut Live<'_>,
) -> String {
    let dispatches = driver_metal::lowering::dispatch::plan(
        lowered,
        driver_metal::lowering::executor::Frame {
            arena: Slice {
                address: 0x1_0000_0000,
                bytes: 1 << 30,
            },
        },
        dispatch_geometry(dg, binding),
        live,
    )
    .expect("the fire plans");
    let empty = dispatches
        .iter()
        .filter(|d| d.grid.contains(&0) || d.threadgroup.contains(&0))
        .count();
    format!("{} ({empty} with an empty grid)", dispatches.len())
}

/// The fire's own tables, staged into one region exactly as the engine seam
/// stages them.
///
/// FOUR DIFFERENT tokens at four different positions, which is what makes the
/// per-token checks able to fail at all. A zeroed region for every table was
/// the first draft and it decodes token 0 at position 0 on every lane -- a
/// legitimate input, and a degenerate one that says nothing about whether the
/// per-token axis works.
fn stage_tables(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
    freqs: &[f32],
) -> driver_metal::bind::tables::Staged {
    let n = step.token_ids.len() as u32;
    // FROM `qo_indptr`, because that is where the step says which rows belong
    // to which request -- and the three tables that answer "whose row is
    // this?" used to answer it without reading it.
    //
    // `req_of_token` was `0..n`: every token its own request, whatever the
    // step said. `position_ids` was `0..n`: fire-global, so a second request's
    // first token was rotated as though it were the fifth token of the first.
    // `kv_page_indptr` was `0..=n`: one page per TOKEN rather than per
    // request, so no request had a second page to attend to.
    //
    // Together they made every batched fire look like a batch of one-token
    // requests, which is a legitimate step and the one step that cannot
    // disagree with anything. `a_request_prefills_the_same_way_beside_another_one`
    // is the gate that exists to catch a request leaking into another's rows,
    // and against these tables it was comparing two fires that both said
    // there was nothing to leak between.
    let requests = step.qo_indptr.len().saturating_sub(1);
    let mut req_of_token: Vec<u32> = Vec::with_capacity(n as usize);
    let mut positions: Vec<u32> = Vec::with_capacity(n as usize);
    for r in 0..requests {
        let (from, to) = (step.qo_indptr[r], step.qo_indptr[r + 1]);
        for row in from..to {
            req_of_token.push(r as u32);
            positions.push(row - from);
        }
    }
    if req_of_token.len() != n as usize {
        // A step whose CSR does not cover its rows -- the legacy single-request
        // spelling. One request, counted from zero.
        req_of_token = vec![0; n as usize];
        positions = (0..n).collect();
    }
    // ONE PAGE PER REQUEST, in request order, wide enough for the rows it
    // holds. `page_size` is the pool's, and every step here is far inside it.
    let pages_per_request = 1u32;
    let each: Vec<u32> = (0..(requests as u32 * pages_per_request).max(n)).collect();
    let indptr: Vec<u32> = (0..=requests as u32)
        .map(|r| r * pages_per_request)
        .collect();
    let kv_write_page: Vec<u32> = req_of_token.clone();
    let w_off: Vec<u32> = positions.iter().map(|p| p % page_size.max(1)).collect();
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    driver_metal::bind::tables::stage(
        context,
        // A pool that lives exactly as long as the `Staged` does -- the lease
        // keeps it alive, and a helper that stages once wants no more.
        &driver_metal::fire::Scratch::new(),
        driver_metal::bind::tables::Frame {
            token_ids: step.token_ids,
            position_ids: &positions,
            req_of_token: &req_of_token,
            kv_page_indices: &each,
            kv_page_indptr: &indptr,
            kv_write_page: &kv_write_page,
            kv_write_offset: &w_off,
            rope_frequencies: &inv_freq,
            // The FIRE's rows, exactly as `serve::launch` stages them: the
            // wire's numbers are request-local and `row_gather` indexes the
            // stream. A rig that passed them through would be testing a
            // staging the driver does not do.
            sampling_indices: &driver_metal::lowering::frame::sampled_rows(step)
                .expect("the readout table places its rows"),
        },
    )
    .expect("the tables stage")
}

#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_real_checkpoints_weights_produce_finite_varied_activations() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // Four lanes, one token each: the decode a scheduler posts.
    let step = Step {
        token_ids: &[128_000, 9906, 1917, 128_001],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 0, 0, 0],
        sampling_indptr: &[0, 1, 2, 3, 4],
        ..Step::default()
    };
    let plan = text(row, FireClass::Decode, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 64);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };

    let freqs = driver_metal::model::rope::table(&dg);
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    let geometry = dispatch_geometry(&dg, &binding);
    let (timing, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry,
        &mut live,
    )
    .expect("the fire runs against real weights");

    assert!(
        timing.encode > std::time::Duration::ZERO,
        "nothing was encoded"
    );
    assert!(
        live.store.missed().is_empty(),
        "the fire asked for {} name(s) the checkpoint does not answer, so the \
         census below would be about sentinels: {:?}",
        live.store.missed().len(),
        live.store.missed()
    );

    // Did the KV pool get anything? An attention that reads a pool nothing
    // wrote answers zero and looks exactly like an attention that is broken.
    for l in 0..2.min(shape.layers) {
        let layer = pool.layer(l).expect("a layer");
        let n = shape.layer_bytes_at(0) as usize;
        // SAFETY: the command buffer retired.
        let (k, v) = unsafe {
            (
                core::slice::from_raw_parts(
                    layer
                        .k
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
                core::slice::from_raw_parts(
                    layer
                        .v
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
            )
        };
        // The first row of each, as numbers. A byte count says the pool was
        // written; it cannot say WHICH tensor landed there, and "the attention
        // answers with K" is exactly the question of which.
        let head = |r: &[u8]| {
            r.chunks_exact(2)
                .take(6)
                .map(|c| {
                    let x = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
                    format!("{x:.6}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!(
            "  kv layer {l}: {} of {n} K bytes non-zero, {} V\n    K[0..6] [{}]\n    V[0..6] [{}]",
            k.iter().filter(|&&b| b != 0).count(),
            v.iter().filter(|&&b| b != 0).count(),
            head(k),
            head(v),
        );
    }

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // so nothing is writing the arena.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // Every arena region the lowering states, censused at ITS element width.
    // Regions rather than the whole buffer: an arena is mixed-dtype and one
    // census over all of it is meaningful only for the dtype that happens to
    // dominate.
    let regions = arena_regions(&lowered, read.len(), &[]);

    // Which statement each arena offset belongs to, and whether it is that
    // statement's OUTPUT. A region nothing wrote is diagnosable only if the
    // report says which launch was supposed to write it.
    //
    // EVERY distinct writer, not the first: regions are reused across layers,
    // so the bytes read below are the LAST write while a first-writer-wins map
    // names the first. For a stack of identical layers the two agree and the
    // list is one entry. Where they do not agree — a region an epilogue reuses,
    // or one two different kernels take turns on — naming only the first is a
    // report that points diagnosis at the wrong kernel, which is worse than
    // naming none.
    let mut writers: HashMap<usize, Vec<String>> = HashMap::new();
    for launch in &lowered.launches {
        let symbol = &lowered.kernels[launch.kernel as usize];
        let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
        // The trace states inputs, then OUTPUTS, then weights, and the
        // ROUTINE says how many of the widthed operands are results. A region
        // that is only ever an INPUT is one nothing was ever supposed to
        // write.
        let results = results_of(symbol);
        let widthed: Vec<&model_compiler::lower::Arg> = args
            .iter()
            .filter(|a| !matches!(a, model_compiler::lower::Arg::Weight(_)))
            .collect();
        let split = widthed.len().saturating_sub(results);
        for arg in widthed.iter().skip(split) {
            if let model_compiler::lower::Arg::Arena { at, .. } = arg {
                let seen = writers.entry(*at).or_default();
                if !seen.iter().any(|s| s == symbol) {
                    seen.push(symbol.clone());
                }
            }
        }
    }

    {
        let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for l in &lowered.launches {
            *hist.entry(l.rows.end - l.rows.start).or_default() += 1;
        }
        eprintln!("launch rows histogram: {hist:?}");
    }
    eprintln!(
        "{} launch(es) -> {} dispatch(es)",
        lowered.launches.len(),
        plan_count(&lowered, &dg, &binding, &mut live)
    );

    let mut c = Census::default();
    let mut unwritten: Vec<String> = Vec::new();
    let mut widest_by_element: Vec<(usize, f32)> = Vec::new();
    for (at, len, element) in &regions {
        let end = (at + len).min(read.len());
        if *at >= end {
            continue;
        }
        let r = census(&read[*at..end], *element);
        c.finite_nonzero += r.finite_nonzero;
        c.zero += r.zero;
        c.nan += r.nan;
        c.inf += r.inf;
        widest_by_element.push((*element, r.max_abs));
        c.max_abs = c.max_abs.max(r.max_abs);
        // "Nothing wrote this" and "this was written with NaN" are opposite
        // findings and `finite_nonzero == 0` is true of BOTH -- a region of
        // nothing but NaN has no finite non-zero value in it. The arena is
        // memset to zero on the host before a dispatch is encoded
        // (`fire::run`), so a NaN in it is a WRITE, and labelling one as an
        // absent write points diagnosis at binding when the answer is
        // arithmetic. Read the zeros, not the finite values.
        let untouched = r.zero == len_in_elements(*len, *element);
        // The writer is named for EVERY region and not only the empty ones.
        // "Nothing wrote this" is one question the map answers; "what wrote
        // the widest value in the arena" is the other, and it is the one a
        // saturation asks. Naming the kernel only on failure means the report
        // is silent exactly when every region was written and one of them was
        // written wrong.
        let who = writers.get(at).map_or_else(
            || "no launch writes it — read-only".to_string(),
            |w| w.join(", "),
        );
        eprintln!(
            "  @{at:>8} {len:>8} B x{element}: {:>7} nz, {:>7} zero, {:>7} NaN, max |v| = {:<12} {who}{}",
            r.finite_nonzero,
            r.zero,
            r.nan,
            r.max_abs,
            if untouched {
                "   <- NOTHING WROTE THIS"
            } else if r.finite_nonzero == 0 && r.nan > 0 {
                "   <- WRITTEN, ALL NaN"
            } else {
                ""
            }
        );
    }
    let widest = |e: usize| {
        widest_by_element
            .iter()
            .filter(|(el, _)| *el == e)
            .map(|(_, v)| *v)
            .fold(0.0f32, f32::max)
    };
    eprintln!(
        "arena {} B in {} region(s): {} finite non-zero, {} zero, {} NaN, {} inf; \
         widest |v| = {} (bf16 {}, f32 {})",
        read.len(),
        regions.len(),
        c.finite_nonzero,
        c.zero,
        c.nan,
        c.inf,
        c.max_abs,
        widest(2),
        widest(4),
    );

    // ── the three failure modes ──
    assert_eq!(
        c.nan, 0,
        "the fire produced {} NaN(s). A NaN anywhere spreads to everything \
         downstream within one layer, so this is not a rounding question.",
        c.nan
    );
    assert_eq!(
        c.inf, 0,
        "the fire produced {} infinity(ies), which is what a norm handed a \
         zero epsilon does to a near-zero row.",
        c.inf
    );
    // A ZERO IS CAPACITY, NOT SILENCE, so the arena-wide ratio is gone.
    //
    // This asserted `finite_nonzero > zero * 10`, calibrated on llama-1B:
    // 648205 non-zero to 8179 zero, 99% of the arena holding a value. It read
    // as "a near-empty arena is a fire that ran and did not compute", and it
    // did catch the single-row gather.
    //
    // But an arena region is sized for the WIDEST fire the plan admits, and
    // this one is a decode of a single row. gemma-4-31b measures 617740
    // non-zero to 1878772 zero, and the two vocabulary regions alone
    // contribute 1572864 of those zeros -- each is 1048576 elements holding
    // one 262144-wide row. Every one of those zeros is a row this fire does
    // not have. llama passed only because its arena happens to be sized close
    // to its decode; the ratio was measuring the plan's shape and reading it
    // as the fire's health.
    //
    // The question the ratio was reaching for is asked exactly below, and
    // asked better: EVERY REGION A LAUNCH DECLARES AS ITS OUTPUT MUST HOLD A
    // VALUE. That is per-region, so a region's capacity cannot drown a
    // region's silence, and it names the launch that owed the write.

    // MAGNITUDES, and the bounds are loose on purpose: what is being caught is
    // saturation, not inaccuracy. A llama-1B decode measures its widest
    // activation under one and its widest logit around 0.08 -- both small,
    // both finite -- and the bounds sit orders of magnitude out so a real
    // drift trips them and a different checkpoint does not.
    //
    // The CEILING is this checkpoint's own, when someone has measured it.
    // `1e3` is llama-1B's, and a constant is what made it read gpt-oss's
    // massive-activation channel -- 42,752 in MLX itself -- as a defect.
    // Four times the measurement, because what is being caught is
    // saturation and not the last bf16 ulp.
    let ceiling = reference_for(&snapshot)
        .and_then(|r| r.widest)
        .map_or(1e3, |w| w * 4.0);
    assert!(
        c.max_abs > 1e-4 && c.max_abs < ceiling,
        "the widest value anywhere is {}, against a ceiling of {ceiling}, \
         which is saturation or silence rather than a forward pass.",
        c.max_abs
    );

    // The READOUT, by name rather than by dtype: it is the widest region the
    // text states, because a vocabulary is wider than anything else in a
    // decode.
    let readout = regions
        .iter()
        .max_by_key(|(_, len, _)| *len)
        .copied()
        .expect("the text states a readout");
    let (at, len, element) = readout;
    let end = (at + len).min(read.len());
    // Row ZERO of it, not all four: three of the four are empty and the lane
    // count below is what tracks that. What this asks is whether the readout
    // that DID run produced a distribution.
    //
    // Exactly half zero would mean something else entirely -- a kernel writing
    // bf16 into a slot sized for f32 -- and that is a defect this gate found
    // and closed on its first run.
    let lane_bytes = (end - at) / 4;
    let r = census(&read[at..at + lane_bytes], element);
    assert!(
        r.finite_nonzero > r.zero,
        "the readout's first lane is {} zero to {} non-zero. Half zero is a \
         dtype disagreement; mostly zero is a readout that did not run.",
        r.zero,
        r.finite_nonzero
    );
    assert!(
        r.max_abs > 1e-4,
        "every logit is under 1e-4, so the readout accumulated nothing."
    );

    // ── the regions a launch declares and does not fill ──
    //
    // ZERO, down from FIVE. All five were the same defect the lane count below
    // names: the text picked the single-row `embed_gather_4bit`, so every lane
    // but the first was zero from statement zero onward, and the branches only
    // those lanes fed never held anything.
    //
    // The NUMBER is what made it findable. It said "five regions", the writer
    // attribution said which statements, and a prefix bisection
    // (`the_second_lane_stops_somewhere_and_this_says_where`) put the stop at
    // statement 0 -- three steps, each narrowing, none of them a guess.
    // ── the declared outputs, censused as RECTANGLES ──
    //
    // Not as the regions above: `arena_regions` cuts the arena at every
    // offset any argument names, so a statement's output rectangle is cut
    // wherever another statement addresses part of it, and neither half is a
    // thing the plan declared. gemma's @65536 came out as a 10240 element
    // region holding nothing followed by an 11264 element region holding
    // everything, and the rectangle both belong to is one 21504 element
    // output that was written.
    //
    // Offsets are REUSED -- an arena slot takes a new tenant once its value
    // is dead -- so the question a single read at the end of the fire can
    // answer is not "was this ever written" but *"did the LAST statement to
    // declare these bytes as its output leave them zero?"*. That one is
    // answerable and is the one that means something: whatever wrote earlier
    // has been overwritten by design, and the final tenant is the value the
    // fire actually carries out.
    // Ownership is per BYTE, not per offset. Output rectangles overlap --
    // gemma's @174080 spans 65536 bytes and swallows both @184320 and
    // @217088 -- because the arena hands a slot to a new tenant the moment
    // the old value dies. So "the last statement to declare this offset"
    // still is not the last statement to WRITE these bytes. Paint ownership
    // in launch order, later wins, and ask each launch only about the bytes
    // it still owns when the fire retires.
    let mut owner: Vec<u32> = vec![u32::MAX; read.len()];
    let mut rects: Vec<(usize, usize, String)> = Vec::new();
    for (n, launch) in lowered.launches.iter().enumerate() {
        let symbol = &lowered.kernels[launch.kernel as usize];
        let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
        // A kernel that writes no ARENA rectangle still writes: the last
        // widthed operands of `kv_append_paged` are the KV pool, not the
        // arena, and the `Arg::Arena` filter below is what drops them.
        let results = results_of(symbol);
        let widthed: Vec<&model_compiler::lower::Arg> = args
            .iter()
            .filter(|a| !matches!(a, model_compiler::lower::Arg::Weight(_)))
            .collect();
        let split = widthed.len().saturating_sub(results);
        let rows = (launch.rows.end - launch.rows.start).max(1) as usize;
        for arg in widthed.iter().skip(split) {
            if let model_compiler::lower::Arg::Arena { at, width, bytes } = arg {
                let end = (at + *width as usize * rows * *bytes as usize).min(read.len());
                if *at >= end {
                    continue;
                }
                let id = rects.len() as u32;
                rects.push((
                    *bytes as usize,
                    *at,
                    format!("[{n}] {symbol} @{at} w{width} x{rows} rows"),
                ));
                owner[*at..end].fill(id);
            }
        }
    }
    let mut owned: Vec<Vec<u8>> = vec![Vec::new(); rects.len()];
    let mut span: Vec<(usize, usize)> = vec![(usize::MAX, 0); rects.len()];
    for (i, byte) in read.iter().enumerate() {
        if owner[i] != u32::MAX {
            let id = owner[i] as usize;
            owned[id].push(*byte);
            span[id].0 = span[id].0.min(i);
            span[id].1 = i + 1;
        }
    }
    for (id, (element, at, who)) in rects.iter().enumerate() {
        let mine = &owned[id];
        if mine.len() < *element {
            continue;
        }
        let r = census(mine, *element);
        if r.zero == len_in_elements(mine.len(), *element) {
            unwritten.push(format!(
                "  {who}: all {} of its surviving elements are zero, \
                 at element {}..{} of its own rectangle",
                mine.len() / element,
                (span[id].0 - at) / element,
                (span[id].1 - at) / element
            ));
        }
    }

    // A REGION IS NOT A RECTANGLE, and this list has to be read knowing it.
    //
    // `arena_regions` cuts the arena at every offset any argument names, so a
    // statement's output rectangle is cut wherever another statement happens
    // to address part of it. gemma-4-31b's @65536 is such a cut: every launch
    // that names the offset states `w5376` over `rows 0..4` -- one 21504
    // element rectangle -- and the census reports it as a 10240 element
    // region holding nothing followed by an 11264 element region holding
    // everything. The rectangle is partly written; neither half is a thing
    // the plan ever declared.
    //
    // So an entry here is one of two findings and does not yet say which:
    // a statement that wrote nothing, or the unwritten part of a statement
    // that wrote something. The `touched by` line is what tells them apart --
    // it prints the width and row span every launch states for the offset,
    // and a region whose length is not `width * rows * element` is a cut
    // rather than a rectangle.
    eprintln!("{} declared output(s) nothing filled", unwritten.len());
    assert!(
        unwritten.is_empty(),
        "{} statement(s) declare an output nothing filled. A statement whose \
         output stays zero is a branch of the forward pass that computes \
         nothing.\n{}",
        unwritten.len(),
        unwritten.join("\n")
    );

    // ── THE PER-TOKEN AXIS ──
    //
    // Four lanes decoded four different tokens, so the readout should hold
    // four different rows. It holds ONE: 128256 of 513024 values non-zero,
    // which is exactly one row of a 128256-wide vocabulary, and rows one
    // through three are zero all the way through.
    //
    // Measured 2026-08-10, and it is the largest remaining gap between this
    // executor and a model that answers. Nothing about it is a grid: every
    // launch states `rows 0..4`, `qmv_mb` puts the row on `grid.x` and
    // `qmv_fast_impl` reads it there (`y += tid.x * out_vec_size`), and the
    // dispatches come out `[128, 512, 1]` over `[32, 2, 1]` -- four
    // threadgroups on x, one per row. All 227 launches plan and none has an
    // empty grid.
    //
    // So the arithmetic is right and the rows still do not appear, which
    // means the next thing to look at is what the FIRST statement writes:
    // every later row being zero is what a gather that emitted one row looks
    // like four launches downstream. Reading between dispatches is the
    // instrument that settles it and this file does not have one yet.
    //
    // Pinned at one, and the number to want is four.
    let lanes = {
        let row = &read[at..end];
        let stride = row.len() / 4;
        (0..4)
            .filter(|i| {
                row[i * stride..(i + 1) * stride]
                    .chunks_exact(element)
                    .any(|c| c.iter().any(|&b| b != 0))
            })
            .count()
    };
    eprintln!("{lanes} of 4 readout lane(s) hold anything");
    assert_eq!(
        lanes, 4,
        "the per-token axis lost a lane: {lanes} of four readout rows hold \
         anything. A fire that answers one token for four is the failure this \
         gate exists to catch, because every magnitude check passes through it."
    );
}

/// **Where the second lane stops.**
///
/// The instrument the test above says it lacks: run the first `n` dispatches
/// of the fire and read the arena, for every `n`, and report the first prefix
/// after which no arena region holds anything in its second row.
///
/// A bisection rather than a guess. "Every later row is zero" is true of a
/// gather that emitted one row and of a projection that did, and four
/// launches downstream they look identical -- so the only thing that
/// distinguishes them is running fewer launches.
///
/// It found the single-row gather at statement 0 and, once that was fixed,
/// `sdpa_paged_decode` writing NEITHER row while every statement around it
/// writes both. That second finding is still open.
///
/// A report, not an assertion: what it prints is a map, and a map that fails
/// the build is a map nobody reads.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_second_lane_stops_somewhere_and_this_says_where() {
    bisect(FireClass::Decode);
}

/// The same walk over the PREFILL lane, which states a different half of the
/// kernel table: `affine_qmm_t` where a decode states `affine_qmv_fast`, and
/// a causal attention over a prefix where a decode has one key.
///
/// SIXTEEN tokens, because `Rule::Qmm` refuses a row count its tile does not
/// divide and `QMM_BMS` starts there. MLX's stages for the same prefix, for
/// comparison against what this prints:
///
///   embed 0.361, L0 attn_norm 2.207, L0 q_proj 1.320
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_prefill_lane_too() {
    bisect(FireClass::Prefill);
}

fn bisect(class: FireClass) {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // The CHECKPOINT'S OWN token, when a reference names one. Posting
    // llama-3.2's 128000 to gemma-4 is posting a token gemma does not have
    // -- it is inside gemma's 262144 vocabulary, so nothing rejects it, and
    // the trail is then a trail of a fire nobody would run. Every value in it
    // is unanswerable against a reference taken on a real token.
    let bos = reference_for(&snapshot).map_or(128_000, |r| r.bos);
    let four = [bos, bos + 1, bos + 2, bos + 3];
    let two = [bos, bos + 1];

    // A decode posts four independent lanes; a prefill posts one sequence.
    let decode = class == FireClass::Decode;
    let step = if decode {
        Step {
            token_ids: &four,
            qo_indptr: &[0, 1, 2, 3, 4],
            sampling_indices: &[0, 0, 0, 0],
            sampling_indptr: &[0, 1, 2, 3, 4],
            ..Step::default()
        }
    } else {
        // TWO rows, which the GEMM's tile does not divide -- the guard's
        // GEMV arm is what serves them.
        Step {
            token_ids: &two,
            qo_indptr: &[0, 2],
            sampling_indices: &[0],
            sampling_indptr: &[0, 1],
            ..Step::default()
        }
    };
    let plan = text(row, class, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 64);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    let staged = if decode {
        stage_tables(&context, &step, shape.page_size, &freqs)
    } else {
        stage_prefill(&context, &step, shape.page_size, &freqs)
    };

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let geometry = dispatch_geometry(&dg, &binding);

    // Every launch's OUTPUT rectangle, so a prefix can be judged by what its
    // last statement was supposed to write rather than by the whole arena.
    let outs: Vec<(usize, usize, usize, String)> = lowered
        .launches
        .iter()
        .map(|l| {
            let symbol = lowered.kernels[l.kernel as usize].clone();
            let args = &lowered.args[l.args.start as usize..l.args.end as usize];
            let last = args
                .iter()
                .rev()
                .find_map(|a| match a {
                    model_compiler::lower::Arg::Arena { at, width, bytes } => {
                        Some((*at, *width as usize, *bytes as usize))
                    }
                    _ => None,
                })
                .unwrap_or((0, 0, 0));
            (last.0, last.1, last.2, symbol)
        })
        .collect();

    // The prefixes worth running: ONE LAYER's worth, read off the plan.
    //
    // This was `1..=12`, with a comment calling twelve "one layer's worth".
    // Twelve is llama's layer. gemma-4's is twenty-one, and its MLP -- the
    // half of a block a per-row defect is as likely to sit in as the
    // attention -- begins at statement fifteen, so the dump stopped three
    // statements before the arithmetic that actually broke on the 31b and
    // reported the attention as though that were the whole block.
    //
    // A layer boundary is stated: every launch carries the layers it runs
    // over, so the first statement belonging to layer 1 ends layer 0. No
    // family constant, and a family whose block is longer still gets its
    // whole block.
    //
    // HOW MANY blocks, because a stack's layers are not all the same block.
    // gemma-4 alternates: five sliding layers then a full-attention one, and
    // the full one is a different SHAPE -- twice the head width, a quarter of
    // the KV heads, and V taken from the K projection rather than projected.
    // Layer 0 is sliding, so a walk that stops at one layer has never run the
    // other kind at all, and every defect that lives there reads as
    // accumulation. Default one, because one is what a first look wants.
    let layers: u16 = std::env::var("PIE_METAL_BISECT_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let one_layer = lowered
        .launches
        .iter()
        .position(|l| l.layers.start >= layers)
        .unwrap_or(lowered.launches.len());
    let mut first_bad: Option<(usize, String)> = None;
    for n in 1..=one_layer.min(lowered.launches.len()) {
        let arena = Allocation::new(
            &context,
            (lowered.arena_bytes as u64).max(1),
            "bisect arena",
        )
        .expect("an arena");
        // SAFETY: freshly allocated.
        unsafe { arena.zero(0, arena.len()).expect("it zeroes") };
        let dispatches = driver_metal::lowering::dispatch::plan(
            &lowered,
            driver_metal::lowering::executor::Frame {
                arena: Slice {
                    address: arena.gpu_address(),
                    bytes: arena.len(),
                },
            },
            geometry,
            &mut live,
        )
        .expect("the fire plans");
        let prefix = &dispatches[..n];
        let prepared = driver_metal::fire::run::prepare(&context, &lowered, prefix)
            .expect("the prefix prepares");
        pipelines
            .ensure(&context, &compiler, prefix)
            .expect("the pipelines compile");
        let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
        stepper
            .run(|encoder| {
                driver_metal::bind::encode::encode(
                    encoder,
                    &prepared.table,
                    &pipelines,
                    &prepared.params,
                    prefix,
                )
            })
            .expect("the prefix runs");

        let mut read = vec![0u8; arena.len() as usize];
        // SAFETY: the command buffer retired.
        unsafe {
            let raw = core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            );
            read.copy_from_slice(raw);
        }

        // The nth statement's own output, row 0 against row 1.
        let (at, width, element, symbol) = &outs[n - 1];
        let row = width * element;
        let live_row = |i: usize| {
            let (a, b) = (at + i * row, (at + (i + 1) * row).min(read.len()));
            a < b && read[a..b].iter().any(|&x| x != 0)
        };
        let (r0, r1) = (live_row(0), live_row(1));
        // The magnitude too, because "it wrote something" and "it wrote the
        // right something" are different questions and the second is the one a
        // reference can answer. MLX's numbers for the same snapshot at
        // position zero, for comparison:
        //
        //   embed 0.361, attn_norm 2.207, q_proj 1.320, v 0.413,
        //   o_proj 0.114, after attn 0.312, L0 out 20.03, L1 out 408.75
        //
        // WHERE in the row, not just how large. A magnitude alone cannot tell
        // a tail from a whole row: a kernel whose grid drops its last partial
        // group and one whose arithmetic is wrong everywhere report the same
        // number, and the element index separates them in one line.
        let (widest, widest_at) = {
            let (a, b) = (*at, (at + row).min(read.len()));
            read[a..b]
                .chunks_exact(*element)
                .enumerate()
                .map(|(i, c)| {
                    let v = if *element == 4 {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs()
                    } else {
                        f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).abs()
                    };
                    (v, i)
                })
                .fold((0.0f32, 0usize), |acc, x| if x.0 > acc.0 { x } else { acc })
        };
        // WHICH SIX, because a row is not six elements long. A kernel that
        // writes the front of every row and nothing else -- a grid short of
        // its width, a threadgroup short of its simdgroups -- prints a
        // perfect front and a correct maximum if the maximum happens to sit
        // there, and gemma-4's does. `PIE_METAL_BISECT_AT` moves the window.
        let window_at: usize = std::env::var("PIE_METAL_BISECT_AT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let head: Vec<String> = read[*at..(at + row).min(read.len())]
            .chunks_exact(*element)
            .skip(window_at)
            .take(6)
            .map(|c| {
                let v = if *element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                };
                format!("{v:.6}")
            })
            .collect();
        eprintln!(
            "  [{:>2}] L{:<2} {symbol:<44} @{at} w{width} g{:?}/{:?} row0 {} row1 {} max|v| {widest:.5}@{widest_at} [{}]",
            n - 1,
            lowered.launches[n - 1].layers.start,
            prefix[n - 1].grid,
            prefix[n - 1].threadgroup,
            if r0 { "yes" } else { "NO " },
            if r1 { "yes" } else { "NO " },
            head.join(", "),
        );
        if r0 && !r1 && first_bad.is_none() {
            first_bad = Some((n - 1, symbol.clone()));
        }
    }

    // The pool, after the whole prefix: which tensor actually landed where.
    {
        let layer = pool.layer(0).expect("a layer");
        let n = shape.layer_bytes_at(0) as usize;
        // SAFETY: the command buffers retired.
        let (k, v) = unsafe {
            (
                core::slice::from_raw_parts(
                    layer
                        .k
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
                core::slice::from_raw_parts(
                    layer
                        .v
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
            )
        };
        let head = |r: &[u8]| {
            r.chunks_exact(2)
                .take(6)
                .map(|c| {
                    let x = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
                    format!("{x:.6}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!("  pool K[0..6] [{}]", head(k));
        eprintln!("  pool V[0..6] [{}]", head(v));
    }

    match &first_bad {
        Some((i, symbol)) => eprintln!(
            "\nThe second lane stops at statement {i}, `{symbol}`: it wrote row 0 \
             and not row 1."
        ),
        None => eprintln!("\nEvery statement in the first layer wrote both rows."),
    }
}

/// Which statement first writes a NaN, over the WHOLE fire.
///
/// # Why this is a search and not a walk
///
/// [`bisect`] re-runs a prefix per statement, which is fine for the twelve
/// that make one layer and quadratic for the four hundred that make a fire.
/// This binary-searches instead: the shortest prefix whose arena holds a NaN
/// is the statement that made it, and that is ~9 runs for a 24-layer model
/// rather than ~480.
///
/// The claim it can make is narrow and worth being exact about. A NaN in the
/// arena after `n` statements and none after `n-1` means statement `n` WROTE
/// one -- it does not say the arithmetic in statement `n` is wrong, because a
/// kernel handed a bad operand produces a bad answer honestly. What it does
/// is turn "somewhere in a 20B model" into one symbol and one layer, and
/// everything after that is reading.
///
/// Prints and passes. A checkpoint with no NaN says so and this is a no-op;
/// making it an assertion would fail every green run for the one model that
/// is not.
///
/// # The step is the GATE's step, and that is load-bearing
///
/// This fired ONE token while
/// `a_real_checkpoints_weights_produce_finite_varied_activations` -- the gate
/// whose failure it exists to locate -- fires FOUR. On gemma-4-31b that was
/// the difference between "the whole fire is NaN-free" and 766,208 NaNs, and
/// this test reported the first, in a run whose only purpose was to explain
/// the second. A locator that fires a different program than the gate cannot
/// locate the gate's failure; it can only be confidently silent.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_first_statement_that_writes_a_nan_says_which_one_it_is() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    let step = Step {
        token_ids: &[128_000, 9906, 1917, 128_001],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 0, 0, 0],
        sampling_indptr: &[0, 1, 2, 3, 4],
        ..Step::default()
    };
    let plan = text(row, FireClass::Decode, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");
    let geometry = dispatch_geometry(&dg, &binding);

    let shape = pool_shape(&dg, 64);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let freqs = driver_metal::model::rope::table(&dg);
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);
    let named = HashMap::new();

    // Which arena spans hold FLOATS, off the text's own declared dtypes.
    // `Arg::Arena` carries a byte WIDTH, which cannot tell an i32 from an f32,
    // and the plan can.
    let int_offsets: std::collections::HashSet<usize> = plan
        .values
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v.dtype, model_ir::trace::DType::I32))
        .filter_map(|(id, _)| lowered.value_offset.get(id).copied())
        .collect();
    // The SAME reader the census uses. This used to take the stated
    // `width * bytes`, which is one row -- so the detector looked at the
    // first token's slice of each value and reported "no statement writes a
    // NaN" for a fire whose very first statement, the embedding gather,
    // writes three rows of zero and one row of NaN. The census said 1,028,352
    // NaN and this said none, in the same file, about the same arena. One of
    // the two readings had to be wrong and it was this one.
    let skip: Vec<usize> = int_offsets.iter().copied().collect();
    let float_spans = arena_regions(&lowered, lowered.arena_bytes as usize, &skip);
    eprintln!(
        "{} float span(s), {} integer value(s) excluded",
        float_spans.len(),
        int_offsets.len()
    );

    // Run the first `n` statements and say whether the arena holds a NaN.
    let mut probe = |n: usize| -> (Option<(usize, usize, usize)>, f32) {
        let arena = Allocation::new(
            &context,
            (lowered.arena_bytes as u64).max(1),
            "nan search arena",
        )
        .expect("an arena");
        // SAFETY: freshly allocated. Zeroed so an unwritten slot reads as a
        // zero and not as whatever the allocator had -- a stale NaN would
        // otherwise be attributed to whichever statement ran last.
        unsafe { arena.zero(0, arena.len()).expect("it zeroes") };
        let pages = |layer: u16, values: bool| {
            pool.layer(u32::from(layer)).map(|l| Slice {
                address: if values {
                    l.v.gpu_address()
                } else {
                    l.k.gpu_address()
                },
                bytes: shape.layer_bytes_at(0),
            })
        };
        let mut live = Live {
            store: Store::new(Names::mlx(), &loaded.tensors, &named),
            tables: &staged,
            shape,
            pages: &pages,
        };
        let dispatches = driver_metal::lowering::dispatch::plan(
            &lowered,
            driver_metal::lowering::executor::Frame {
                arena: Slice {
                    address: arena.gpu_address(),
                    bytes: arena.len(),
                },
            },
            geometry,
            &mut live,
        )
        .expect("the fire plans");
        let prefix = &dispatches[..n.min(dispatches.len())];
        let prepared = driver_metal::fire::run::prepare(&context, &lowered, prefix)
            .expect("the prefix prepares");
        pipelines
            .ensure(&context, &compiler, prefix)
            .expect("the pipelines compile");
        let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
        stepper
            .run(|encoder| {
                driver_metal::bind::encode::encode(
                    encoder,
                    &prepared.table,
                    &pipelines,
                    &prepared.params,
                    prefix,
                )
            })
            .expect("the prefix runs");
        // SAFETY: the command buffer retired.
        let raw = unsafe {
            core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            )
        };
        // FLOAT regions only. A routed FFN's arena is half INDEX buffers --
        // `route_sort` writes a permutation, a per-row expert, a per-tile
        // expert and an inverse, all integers -- and an index read as a float
        // is a NaN whenever its top bits happen to be an all-ones exponent.
        // `-1`, the sentinel a padded tile carries, is 0xFFFFFFFF, which is
        // exactly that. So a detector that reads the whole arena as floats
        // reports the sort as the first NaN of every mixture, every time, and
        // says nothing.
        //
        // The widest finite magnitude comes back with the answer, because the
        // two ways to arrive at a NaN look identical in a yes/no and want
        // opposite investigations: a value that GREW until bf16 could not
        // hold it (a missing or doubled scale, compounding per layer) and a
        // value that was finite and became NaN in one step (a bad read).
        //
        // WHERE the NaN is, not only that there is one. "The arena holds a
        // NaN" names a fire, and the arena is every live value at once; the
        // statement that first holds one is not always the statement that
        // WROTE one, because a plan reuses offsets and the detector reads
        // them all. The byte offset says which value, and the element index
        // within it says whether the damage is a tail or the whole row --
        // the same distinction the per-statement dump had to make before it
        // could tell an out-of-bounds read from bad arithmetic.
        let mut nan = None;
        let mut widest = 0.0f32;
        for &(at, len, element) in &float_spans {
            for (i, c) in raw[at..(at + len).min(raw.len())]
                .chunks_exact(element)
                .enumerate()
            {
                let v = if element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                };
                if v.is_nan() {
                    if nan.is_none() {
                        nan = Some((at, i, len / element));
                    }
                } else if v.abs() > widest {
                    widest = v.abs();
                }
            }
        }
        (nan, widest)
    };
    let total = lowered.launches.len();

    // THE MAGNITUDE TRAIL RUNS EITHER WAY, and it used to be reachable only
    // through a NaN: the walk below is the sharpest instrument in this file
    // and `return`ing here handed nothing at all to a fire that is finite and
    // wrong -- which is the fire a driver spends most of its life being.
    //
    // Per LAYER rather than per statement, because that is the axis a
    // reference can be taken on: MLX will hand back `max(abs(h))` after each
    // of its own blocks for the cost of a loop, and a driver whose layer four
    // reads 111 against MLX's 111 and whose layer six reads 4e2 against MLX's
    // 4e4 has named the block to open. A per-statement trail cannot be
    // compared to anything outside this process.
    eprintln!("  widest finite |v| after each layer:");
    let last_of_layer = {
        let mut ends: Vec<(u16, usize)> = Vec::new();
        for (n, l) in lowered.launches.iter().enumerate() {
            let layer = l.layers.end.saturating_sub(1);
            match ends.last_mut() {
                Some((prev, at)) if *prev == layer => *at = n + 1,
                _ => ends.push((layer, n + 1)),
            }
        }
        ends
    };
    for (layer, at) in &last_of_layer {
        let (nan, widest) = probe(*at);
        eprintln!(
            "    through layer {layer:>3} (statement {at:>4}): {widest:>12.4e}{}",
            if nan.is_some() { "  (holds a NaN)" } else { "" }
        );
    }

    let mut nan_after = |n: usize| -> bool { probe(n).0.is_some() };

    if !nan_after(total) {
        eprintln!("the whole fire ({total} statements) is NaN-free");
        return;
    }
    // The smallest prefix that has one -- COARSE SWEEP first, then bisect
    // inside the interval that flipped.
    //
    // A plain bisection over `nan_after` assumes the predicate is monotonic:
    // that once the arena holds a NaN it holds one forever. An arena REUSES
    // offsets, which is what it is for, so a later statement writing finite
    // values over a region that held a NaN takes it back out and the
    // predicate flips false again. Bisecting a predicate that is not
    // monotonic returns SOME boundary, not the FIRST one, and nothing in the
    // output distinguishes the two.
    //
    // A linear scan would be exact and is 1255 GPU runs. A sweep of `STEPS`
    // evenly spaced probes finds the first coarse interval that flips, and
    // bisecting inside that interval is exact within it -- so the answer is
    // wrong only if a NaN appears AND is fully overwritten inside one stride.
    // That is a far weaker assumption than "never overwritten", it costs
    // about thirty probes rather than eleven, and the stride is reported so
    // the reader can judge it.
    const STEPS: usize = 32;
    let stride = total.div_ceil(STEPS).max(1);
    let mut lo = 0usize;
    let mut hi = total;
    let mut swept = false;
    let mut at = stride.min(total);
    while at <= total {
        if nan_after(at) {
            lo = at - stride.min(at);
            hi = at;
            swept = true;
            break;
        }
        if at == total {
            break;
        }
        at = (at + stride).min(total);
    }
    if !swept {
        // Every coarse probe was clean and the full fire is not: the NaN
        // lives and dies inside one stride. Say so rather than bisect a
        // predicate already known to be false at both ends of every interval.
        eprintln!(
            "the full fire holds a NaN but none of the {STEPS} probes every {stride} statements does -- it is written and overwritten inside one stride, which a prefix search cannot narrow further"
        );
        return;
    }
    eprintln!("swept every {stride} statements; the flip is in {lo}..{hi}");
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if nan_after(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let launch = &lowered.launches[hi - 1];
    let symbol = &lowered.kernels[launch.kernel as usize];
    // Named by its WEIGHTS, not just its symbol. `affine_qmv_fast` is thirteen
    // different projections in one layer, and a symbol says which arithmetic
    // ran, never which tensor it ran over -- the same distinction the fine
    // magnitude walk below already had to make before it could name up_proj.
    let operands: Vec<&str> = lowered.args[launch.args.start as usize..launch.args.end as usize]
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Weight(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    // WHERE it landed, and how much of the value is gone. A NaN at element
    // zero of a value and a NaN only in its tail are different defects: the
    // first is arithmetic that was wrong from the start, the second is a read
    // that ran off the end. The statement name cannot tell them apart.
    let site = match probe(hi).0 {
        Some((at, i, of)) => format!(
            "\n  first NaN at arena byte {at}, element {i} of {of}{}",
            lowered
                .value_offset
                .iter()
                .position(|o| *o == at)
                .map_or(String::new(), |v| format!(" (value {v})"))
        ),
        None => String::new(),
    };
    eprintln!(
        "\nthe first NaN appears at statement {} of {total}: `{symbol}`, layer {:?}, rows {:?}\n  over {operands:?}{site}",
        hi - 1,
        launch.layers,
        launch.rows
    );
    // How wide the arena's finite values were on the way in. A NaN at the end
    // of a climb is a scale that compounds; a NaN one step after a settled
    // magnitude is a read. The statement alone cannot tell them apart.
    // From the START, not just the last few: an infinity found six statements
    // back says only that the answer is further back still. On gemma-4-31b
    // the arena is already infinite at statement 69, so the NaN at 74 is a
    // CONSEQUENCE -- inf minus inf, or inf times zero -- and the question is
    // where the magnitudes left bf16's range, which is a different statement
    // and possibly a different layer.
    //
    // Swept coarsely and then walked one statement at a time, for the same
    // reason the NaN search is: fourteen points over 1255 statements name an
    // interval of ninety, and an interval is not a kernel. The interval to
    // walk is the one whose magnitude RATIO is largest, which needs no
    // threshold -- "sane" is not a number this driver knows, and a reuse that
    // takes the magnitude back down has a ratio below one and so cannot be
    // mistaken for the climb.
    eprintln!("  widest finite |v| from the start:");
    let mut trail: Vec<(usize, f32, bool)> = Vec::new();
    let points = 14usize;
    let step = (hi.max(1)).div_ceil(points).max(1);
    let mut at = 0usize;
    loop {
        let (nan, widest) = probe(at);
        eprintln!(
            "    after {at:>4}: {widest:>12.4e}{}",
            if nan.is_some() { "  (holds a NaN)" } else { "" }
        );
        trail.push((at, widest, nan.is_some()));
        if at >= hi - 1 {
            break;
        }
        at = (at + step).min(hi - 1);
    }
    // The steepest climb, by ratio rather than by difference: a jump from
    // 1e3 to 5e28 and one from 5e28 to 1e29 differ by about the same amount
    // and are not the same event.
    //
    // A RATIO NEEDS TWO MAGNITUDES, and neither zero nor an infinity is one.
    // The first draft ordered those two by the value reached, which made
    // every interval ending at `inf` the maximum -- so on gemma-4-31b it
    // named 72..74, the three statements before the NaN, when the trail
    // printed directly above it showed the arena going 1.0e3 to 5.1e28
    // between 12 and 18. It picked the END of the climb over its start,
    // which is the exact mistake the whole magnitude trail exists to stop
    // the bisection making.
    //
    // So: intervals with two finite non-zero endpoints are ranked; the rest
    // are skipped, because the trail above already shows them. If every
    // interval is skipped -- the arena's first write is already infinite --
    // the first one reaching a non-finite value is walked instead, since
    // then there is no climb to find and the question is which statement
    // wrote it.
    let ranked = trail
        .windows(2)
        .filter(|w| w[0].1 > 0.0 && w[0].1.is_finite() && w[1].1.is_finite())
        .max_by(|a, b| (a[1].1 / a[0].1).total_cmp(&(b[1].1 / b[0].1)))
        .map(|w| (w[0].0, w[1].0));
    let steepest = ranked.or_else(|| {
        trail
            .windows(2)
            .find(|w| !w[1].1.is_finite())
            .map(|w| (w[0].0, w[1].0))
    });
    if let Some((from, to)) = steepest {
        eprintln!(
            "  the steepest climb is {from}..{to}; every statement in it, \
             with what wrote:"
        );
        let mut previous = f32::NAN;
        for n in from..=to {
            let (nan, widest) = probe(n);
            let nan = nan.is_some();
            let symbol = if n == 0 {
                "<the arena as staged>"
            } else {
                &lowered.kernels[lowered.launches[n - 1].kernel as usize]
            };
            let layer = if n == 0 {
                String::new()
            } else {
                format!(" layer {:?}", lowered.launches[n - 1].layers)
            };
            // The multiplier this one statement applied, which is the number
            // being looked for -- the trail above only says the interval.
            let factor = if previous.is_finite() && previous > 0.0 {
                format!("  x{:.3e}", widest / previous)
            } else {
                String::new()
            };
            // The WEIGHTS it was handed, which is the question a symbol
            // cannot answer: statements 16 and 17 of gemma-4-31b are the
            // same kernel over the same shape, and only one of them leaves
            // bf16's range. What differs is the tensor.
            let weights: Vec<&str> = if n == 0 {
                Vec::new()
            } else {
                let a = &lowered.launches[n - 1].args;
                lowered.args[a.start as usize..a.end as usize]
                    .iter()
                    .filter_map(|arg| match arg {
                        model_compiler::lower::Arg::Weight(name) => Some(name.as_str()),
                        _ => None,
                    })
                    .collect()
            };
            eprintln!(
                "    after {n:>4}: {widest:>12.4e}{}{factor}   `{symbol}`{layer} {}",
                if nan { "  (holds a NaN)" } else { "" },
                weights.join(" ")
            );
            previous = widest;
        }
    }
    // Its neighbours, because a symbol alone does not say what it was handed.
    //
    // The WHOLE layer, not four statements either side. A NaN inside an
    // attention block is a question about the block -- which shape its sdpa
    // took, whether a projection was suppressed, which norms ran -- and four
    // statements show the arithmetic without the structure. The bound comes
    // from the plan's own layer numbering, so a family whose layer is 21
    // statements gets 21 and llama's 12 gets 12.
    let layer = lowered.launches[hi - 1].layers.start;
    let first = lowered
        .launches
        .iter()
        .position(|l| l.layers.start >= layer)
        .unwrap_or(hi - 1);
    let last = lowered
        .launches
        .iter()
        .rposition(|l| l.layers.start <= layer)
        .map_or(hi, |i| i + 1);
    eprintln!("\n  every statement of layer {layer}:");
    for i in first..last.min(total) {
        let l = &lowered.launches[i];
        // The RECTANGLES as well as the names. A NaN that begins exactly at
        // one operand's width is a slot too small for what was written into
        // it, and that is visible only if the widths are on the page next to
        // the statement that wrote them.
        let mut names = Vec::new();
        for a in &lowered.args[l.args.start as usize..l.args.end as usize] {
            match a {
                model_compiler::lower::Arg::Weight(name) => names.push(name.clone()),
                model_compiler::lower::Arg::Arena { at, width, .. } => {
                    names.push(format!("@{at}w{width}"));
                }
                _ => {}
            }
        }
        eprintln!(
            "  [{i:3}]{} {:<40} rows {:?} {}",
            if i == hi - 1 { " <-" } else { "   " },
            lowered.kernels[l.kernel as usize],
            l.rows,
            names.join(" ")
        );
    }
}

/// **The first number held to a reference.**
///
/// One token at position ZERO, and the position is chosen rather than
/// convenient: rope is the identity there (cos 0 = 1, sin 0 = 0), so
/// llama-3.2's rope SCALING -- which this text does not state -- cannot make
/// the two implementations disagree, and attention attends to exactly one key,
/// its own. What is left is every piece of arithmetic that is not
/// position-dependent: the gather, five norms a layer, q/k/v/o, the gated MLP,
/// the final norm and the readout.
///
/// The reference is MLX itself, run over the same snapshot with
/// `mx.quantized_matmul` -- the same affine codec the checkpoint was written
/// with, so a disagreement is about the DRIVER and not about who read the
/// 4-bit format correctly. Its answer for `<|begin_of_text|>` (128000) is
/// argmax **16309** with logits spanning [-4.61, 6.41].
///
/// **It agrees.** Same argmax, the same top five in the same order, every
/// logit within bf16 of MLX's, and the same span. The driver's readout is bf16
/// where MLX accumulates wider, so the tolerance is a statement about the
/// FORMAT rather than slack for a wrong answer.
///
/// Getting here cost one more defect, and it was two statements into the fire:
/// `RmsParams::w_stride` is the distance between consecutive CHANNELS of the
/// gain vector -- `ws[w_stride * i]`, one for a contiguous row, and
/// `rms.metal`'s own header says so. The statement passed the AXIS. Every norm
/// read `w[2048 * i]`, strode out of the gain vector on its second channel,
/// and multiplied by whatever followed it in the checkpoint. Channel 1 came
/// out -0.016 where MLX says +0.052: the wrong sign, from the wrong tensor.
///
/// It survived everything. The fire ran, every statement wrote every row, no
/// NaN, no infinity, 99% of the arena non-zero, and the logits were a
/// plausible-looking near-uniform distribution over 128256 tokens. Only a
/// reference could see it, which is the argument for having one.
///
/// # Why position zero and not position one
///
/// Not caution -- a KNOWN gap, and stating where it is beats pretending it is
/// not there. llama-3.2's config carries
/// `rope_scaling: {rope_type: llama3, factor: 32, low_freq_factor: 1,
/// high_freq_factor: 4, original_max_position_embeddings: 8192}`, and the text
/// passes `dsl::metal::rope` a bare theta. So the driver's rotation is the
/// unscaled one and a comparison at any position but zero would be measuring
/// that rather than the executor.
///
/// The shader for it already exists: `rope_neox_freqs_decode` takes
/// `inv_freq` as a device buffer rather than deriving frequencies from a base,
/// which is exactly the shape llama-3's rescaling wants. What is missing is
/// the table -- a load-time derivation from the config, so a `Source` beside
/// the fire tables rather than anything a text can state. That is the next
/// thing this file should be pointed at.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn one_token_at_position_zero_agrees_with_mlx() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Some(reference) = reference_for(&snapshot) else {
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // ONE request, ONE token, position zero.
    let step = Step {
        token_ids: &[reference.bos],
        qo_indptr: &[0, 1],
        sampling_indices: &[0],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let plan = text(row, FireClass::Decode, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 16);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        dispatch_geometry(&dg, &binding),
        &mut live,
    )
    .expect("the fire runs");

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before the call returned.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // The readout: the widest region the text states, because a vocabulary is
    // wider than anything else in a decode.
    // The text's OWN statement of where its answer is, not a guess at it.
    // This used to take the widest arena region, which was right by
    // luck: the gemma text holds TWO vocabulary-wide buffers, because
    // the logit softcap is out of place, and the tie-break picked the
    // capped one.
    let (at, width, element) = {
        let r = lowered.readout.expect("the text states an exit seam");
        (r.at, r.vocab as usize, r.bytes as usize)
    };
    let vocab = width;
    let logits: Vec<f32> = read[at..at + vocab * element]
        .chunks_exact(element)
        .map(|c| {
            if element == 4 {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            } else {
                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
            }
        })
        .collect();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let (lo, hi) = logits
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    eprintln!(
        "argmax {} argmin {} over {vocab} logits, span [{lo}, {hi}]",
        order[0],
        order[order.len() - 1]
    );
    for (i, &t) in order.iter().take(5).enumerate() {
        eprintln!("  top{i}: {t} logit {:.6}", logits[t]);
    }
    // AND THE TAIL, printed before anything is asserted. A top-five mismatch
    // and a whole distribution shifted are different defects, and the top five
    // alone cannot tell them apart: five tokens out of 201 088 is a statement
    // about the peak. The deep ranks say whether the rest of the vocabulary
    // moved with it.
    for r in [100usize, 1000, 10000] {
        if let Some(&t) = order.get(r) {
            eprintln!("  rank{r}: {t} logit {:.6}", logits[t]);
        }
    }

    // MLX's answer for the same token over the same snapshot, top five, in
    // order, with the logits it gave them.
    //
    // The tokens must match exactly; the logits are compared with a bf16
    // tolerance because the driver's readout IS bf16 -- 8 mantissa bits, so
    // about 0.4% near six -- where MLX accumulates wider. A tolerance is
    // therefore a statement about the FORMAT and not slack for a wrong answer.
    // A TIE HAS NO ORDER, and this used to demand one. MLX gives 236798 and
    // 236780 the same 29.875 and so does this driver; which of the two a sort
    // puts first is a property of the sort. Asserting rank-for-rank reported
    // that as "MLX says 236798 and this says 236780" -- a defect in neither.
    //
    // The set is what a tied top five states, and it is not weaker: five
    // tokens out of 262144 in any order is the same claim about the fire.
    //
    // AND IT RUNS ONLY AS DEEP AS THE READOUT CAN RESOLVE, which is the rule
    // the argmax below already lives by -- applied here because the boundary
    // of a SET is a comparison too. "The top five are these five" says the
    // fifth logit outranks the sixth; where those two are closer together
    // than a bf16 ulp, a bf16 readout ranking them either way is reporting
    // its rounding.
    //
    // gpt-oss-20b at position zero, in f32: 9.1921, 9.1736, 9.1306 at ranks
    // four, five and six -- three tokens inside one ulp. This gate read the
    // driver's 198-over-13 as a defect and there is no fire that could have
    // satisfied it. Ranks one to three are 0.31, 0.35 and 0.32 apart, all of
    // them resolvable, and that is the claim this checkpoint supports.
    //
    // llama-3.2-1B loses nothing: its fifth and sixth are 5.6016 and 5.4275,
    // 2.8 ulps apart, so `depth` is the full five.
    // A prefix of length `m` claims `top[m-1]` outranks whatever is below it,
    // so `m` is founded exactly when that ONE gap is resolvable. The scan
    // stops at the first gap that is not: everything deeper is behind it.
    let depth = {
        let n = reference.top.len();
        let mut d = 0;
        for m in 1..=n {
            // Below the last entry is the sixth logit, which only `next` has.
            let Some(below) = (if m < n {
                Some(reference.top[m].1)
            } else {
                reference.next
            }) else {
                break;
            };
            if reference.top[m - 1].1 - below <= bf16_slack(below) {
                break;
            }
            d = m;
        }
        d
    };
    if depth == 0 {
        // MLX's own top two are inside an ulp of each other. There is no
        // ranking to hold this driver to, and the per-token logits below are
        // the whole of what can be asked.
        eprintln!("SKIP the set: MLX's top two are closer than bf16 resolves");
    }
    let mut want_set: Vec<usize> = reference.top[..depth].iter().map(|(t, _)| *t).collect();
    let mut got_set: Vec<usize> = order[..depth].to_vec();
    want_set.sort_unstable();
    got_set.sort_unstable();
    eprintln!(
        "the top {depth} of {} are resolvable in bf16",
        reference.top.len()
    );
    assert_eq!(
        got_set, want_set,
        "MLX's top {depth} are {want_set:?} and this driver's are {got_set:?}. At \
         position zero rope is the identity and attention has one key, so \
         nothing position-dependent can explain a difference."
    );

    // The ARGMAX exactly, whenever MLX's own top two are further apart than
    // the readout can resolve -- where they are not, the tie above is the
    // whole of what can be claimed.
    if (reference.top[0].1 - reference.top[1].1).abs() > bf16_slack(reference.top[0].1) {
        assert_eq!(
            order[0], reference.top[0].0,
            "MLX's argmax is {} and this driver's is {}.",
            reference.top[0].0, order[0]
        );
    }

    // The value, with the tolerance the CAP dictates rather than one the
    // readout alone would.
    //
    // `cap * tanh(x/cap)` has slope `1 - (v/cap)^2`, so a post-cap ulp stands
    // for `ulp / slope` of pre-cap logit. At gemma's 29.625 against a cap of
    // 30 that slope is 0.025: forty units of pre-cap logit arrive as one unit
    // of post-cap value, and two implementations agreeing there is a
    // statement about `tanh`'s asymptote and not about either fire.
    //
    // So the tolerance carries the slope, and the PRE-CAP value each side
    // implies is printed beside it -- the assertion says what can be claimed
    // and the report keeps what was measured. gemma-4-31b at position zero:
    // every one of the top five is within a bf16 ulp POST-cap, and the
    // pre-cap logits they imply differ by up to 13%, which is the reading
    // this gate cannot settle and a pre-cap comparison would.
    for (want, logit) in &reference.top {
        let mine = logits[*want];
        let slack = if reference.cap > 0.0 {
            let slope = 1.0 - (logit / reference.cap).powi(2);
            bf16_slack(*logit) / slope.max(f32::EPSILON)
        } else {
            bf16_slack(*logit)
        };
        if reference.cap > 0.0 {
            let pre = |v: f32| reference.cap * (v / reference.cap).atanh();
            eprintln!(
                "  token {want}: MLX {logit} / this {mine}  (pre-cap {:.1} / {:.1})",
                pre(*logit),
                pre(mine)
            );
        }
        // Plus what ROUTING moves, for a row that routes. Stated pre-cap --
        // that is where it was measured and where it means anything -- and
        // carried back through the same slope the bf16 term crosses.
        let allowed = if reference.routing > 0.0 && reference.cap > 0.0 {
            let pre = reference.cap * (logit / reference.cap).atanh();
            let slope = 1.0 - (logit / reference.cap).powi(2);
            slack + reference.routing * pre.abs() * slope
        } else {
            slack
        };
        assert!(
            (mine - logit).abs() <= allowed,
            "token {want}: MLX logit {logit}, this {mine} — further apart than \
             bf16, the cap's slope and this row's routing explain ({allowed})."
        );
    }

    // The SPAN, because five agreeing logits at the top is consistent with a
    // distribution that is wrong everywhere else.
    let (want_lo, want_hi) = reference.span;
    eprintln!("span [{lo}, {hi}] against MLX's [{want_lo}, {want_hi}]");

    // THE READING FROM SOMEWHERE ELSE. Three tokens down the distribution,
    // where the cap does not reach and a bf16 ulp is the whole tolerance.
    //
    // This replaces asserting on the span's two ends, which for gemma are the
    // two values the cap has erased -- and one of which (the floor) is a
    // single outlier token whose pre-cap logit is -48.7 where this driver's
    // is -27.9. That disagreement is real and is recorded in the north star;
    // it is not what the span was ever asked to establish, and a gate that
    // reports it as "the distribution is wrong" while ranks 100, 1000 and
    // 10000 agree to a bf16 ulp is reporting the wrong thing.
    for (token, want) in &reference.mid {
        let mine = logits[*token];
        if reference.cap > 0.0 {
            let pre = |v: f32| reference.cap * (v / reference.cap).atanh();
            eprintln!(
                "  mid {token}: MLX {want} / this {mine}  (pre-cap {:.2} / {:.2}, ratio {:.4})",
                pre(*want),
                pre(mine),
                pre(mine) / pre(*want)
            );
        }
    }
    // AT THE DISTRIBUTION'S SCALE, not at each value's own.
    //
    // A logit is a dot product over `hidden` terms, and its error is set by
    // the size of those terms rather than by the size of what they sum to.
    // llama's rank-10000 logit is 1.40; two ulps OF 1.40 is 0.016, and the
    // same accumulation that lands 6.41 to within 0.03 cannot land 1.40 to
    // within 0.016. Measured, it lands 1.4375 -- 2.4 ulps of itself and half
    // an ulp of the readout's widest value, which is the number that says
    // what the accumulation could do.
    let scale = bf16_slack(want_lo.abs().max(want_hi.abs()));
    for (token, want) in &reference.mid {
        let mine = logits[*token];
        // Down here the cap does not reach, so the routing band is the plain
        // fraction of the value with no slope to cross.
        let scale = scale + reference.routing * want.abs();
        assert!(
            (mine - want).abs() <= scale,
            "token {token}, well down the distribution: MLX says {want} and \
             this says {mine}, further apart than {scale}. The cap does not \
             reach here, so this is the whole distribution disagreeing and \
             not its erased extremes."
        );
    }
}

/// **A two-token PREFILL, held to the same reference.**
///
/// Everything the position-zero gate could not reach: rope at a position that
/// rotates, attention over a prefix rather than one key, and the M>1 lane's
/// own symbols — a prefill states `affine_qmm_t` where a decode states
/// `affine_qmv_fast`, so this is a different half of the kernel table.
///
/// The readout is the LAST token's, which is what a prefill produces and what
/// a sampler wants. MLX's answer for `[128000, 9906]` at position 1 is argmax
/// **0** with the distribution spanning [-5.42, 18.56].
///
/// **It agrees.** Same argmax, same top five in order, span [-5.41, 18.63]
/// against MLX's [-5.42, 18.56]. So the M>1 lane is held to a reference too,
/// and between them the two gates cover both halves of the kernel table.
///
/// Three things had to land for it, and each was invisible to the other gate:
///
///   * the projection GUARD, because `qmm_t.metal` needs `M % BM == 0` and two
///     rows tile nothing — which took `region_out` on the arms and
///     `Lowering::region_outs` under them;
///   * the ROW GATHER, because a prefill's stream is one row per TOKEN and its
///     readout one per REQUEST. Without it the readout read row 0 and answered
///     the FIRST token's distribution — exactly right, for a question nobody
///     asked;
///   * `Source::RequestCount` as `Ty::InPacked`, because how many rows to
///     gather is the fire's number and it is a FIELD of a packed struct rather
///     than an operand.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_two_token_prefill_agrees_with_mlx() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    // The reference below is llama-3.2-1B-Instruct-4bit's, taken by hand.
    if !reference_taken_on(&snapshot, "Llama-3.2-1B-Instruct-4bit") {
        return;
    }
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // ONE request, TWO tokens: a prefill.
    //
    // `sampling_indices: &[1]` — the LAST token's, which is what a prefill
    // produces and what a sampler wants. Asking for index 0 returns position
    // zero's distribution, and it returns it EXACTLY: 16309 at 6.40625,
    // matching the decode gate and MLX. Worth knowing, because it means the
    // readout gather is right and the difference below is the sequence.
    let step = Step {
        token_ids: &[128_000, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let plan = text(row, FireClass::Prefill, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 16);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    // Both tokens are ONE request's, so `req_of_token` is all zeros and both
    // land in that request's first page at their own offsets. `stage_tables`
    // states one request per token, which is a decode's shape — so the tables
    // here are the prefill's own.
    let staged = stage_prefill(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        dispatch_geometry(&dg, &binding),
        &mut live,
    )
    .expect("the prefill runs");

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before the call returned.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // The text's OWN statement of where its answer is, not a guess at it.
    // This used to take the widest arena region, which was right by
    // luck: the gemma text holds TWO vocabulary-wide buffers, because
    // the logit softcap is out of place, and the tie-break picked the
    // capped one.
    let (at, width, element) = {
        let r = lowered.readout.expect("the text states an exit seam");
        (r.at, r.vocab as usize, r.bytes as usize)
    };
    let logits: Vec<f32> = read[at..at + width * element]
        .chunks_exact(element)
        .map(|c| {
            if element == 4 {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            } else {
                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let (lo, hi) = logits
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    eprintln!("prefill argmax {} span [{lo}, {hi}]", order[0]);
    for (i, &t) in order.iter().take(5).enumerate() {
        eprintln!("  top{i}: {t} logit {:.6}", logits[t]);
    }

    const MLX: [(usize, f32); 5] = [
        (0, 18.562_5),
        (11, 18.234_375),
        (5127, 17.937_5),
        (1070, 17.468_75),
        (323, 17.296_875),
    ];
    for (i, (want, logit)) in MLX.iter().enumerate() {
        assert_eq!(
            order[i], *want,
            "rank {i}: MLX says token {want} and this says {}",
            order[i]
        );
        assert!(
            (logits[order[i]] - logit).abs() < 0.2,
            "token {want}: MLX logit {logit}, this {} — further apart than bf16 \
             explains at this magnitude.",
            logits[order[i]]
        );
    }
    assert!(
        (hi - 18.5625).abs() < 0.2 && (lo + 5.422).abs() < 0.2,
        "the distribution spans [{lo}, {hi}] where MLX spans [-5.422, 18.563]."
    );
}

/// **The rotation reaches every row of a prefill, and not only the first.**
///
/// This needs no reference, which is the point — it is a gate a laptop with
/// the checkpoint can run when nothing has captured MLX for the case.
///
/// Rope at position ZERO is the identity: `theta = scale * 0 * inv_freq` is
/// zero, so `cos` is one and `sin` is zero and the pair comes back unchanged.
/// So a prefill of the SAME token twice writes two K rows that are the same
/// projection of the same embedding, and the ONLY thing that can separate them
/// is the rotation. Row 0 must be the raw projection and row 1 must be it
/// turned by one position. Two IDENTICAL rows therefore say exactly one thing:
/// the rotation never reached row 1.
///
/// Which is what a single-row kernel over a multi-row grid does. `Rule::Rope`
/// dispatches `[rotary_dims/2, q_heads, rows]`, and a kernel that declares
/// `uint2 pos [[thread_position_in_grid]]` is never handed `pos.z` — every row
/// of the grid computes row 0's index, races on row 0's memory, and leaves
/// rows 1.. untouched.
///
/// A logit comparison cannot be trusted to catch this on its own: it reads one
/// distribution at the end of twenty-eight layers, and the constants it checks
/// have to come from somewhere. This reads the rotation's own output.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_prefill_rotates_its_second_row() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // The SAME token twice. Same embedding, same projection, so the two K rows
    // leave the matmul bit-identical and only the rotation can part them.
    let step = Step {
        token_ids: &[9906, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let plan = text(row, FireClass::Prefill, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 16);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    // This is the FREQS lane's fire; the base lane's twin below is the other.
    // A checkpoint whose ladder is not rescaled never reaches this kernel, so
    // running it would prove nothing about the shader under test.
    //
    // SKIPPED rather than asserted. The assertion read the right fact and drew
    // the wrong conclusion from it: "this deployment takes the other lane" is
    // a statement about the CHECKPOINT, and a suite that reports it as a
    // driver failure puts it in the same list as the failures that are real.
    // gemma-4 takes the base lane for its sliding layers, and said FAILED for
    // it beside the NaN that mattered.
    //
    // The fact used to be read off a `LlamaLikeMetalFacts` this file rebuilt
    // from the tensors; it is read off the DEPLOYMENT's ladder now, which is
    // the same number one seam earlier and no longer a second opinion about
    // the checkpoint that could disagree with the text's.
    if dg.rope_rescale.is_none() {
        eprintln!(
            "SKIP: this checkpoint does not rescale its ladder, so the BASE \
             lane is the one it takes -- see the twin below."
        );
        return;
    }
    let staged = stage_prefill(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        dispatch_geometry(&dg, &binding),
        &mut live,
    )
    .expect("the prefill runs");

    // Layer zero's keys. `stage_prefill` writes row r into page zero at slot r,
    // and the layer is `[pages, page_size, kv_heads * head_dim]`.
    let layer = pool.layer(0).expect("layer zero is pooled");
    let row_bytes = shape.row_bytes_at(0) as usize;
    let mut keys = vec![0u8; row_bytes * 2];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // and the pool's K region is at least two rows wide.
    unsafe {
        let raw = core::slice::from_raw_parts(
            layer
                .k
                .host_span(0, keys.len() as u64)
                .expect("the pages are addressable")
                .as_ptr()
                .cast_const(),
            keys.len(),
        );
        keys.copy_from_slice(raw);
    }
    let (row0, row1) = keys.split_at(row_bytes);

    assert!(
        row0.iter().any(|b| *b != 0),
        "row zero's key is all zeros, so nothing was written and the rest of \
         this gate would pass for the wrong reason"
    );
    assert!(
        row0 != row1,
        "the two K rows are byte-identical after a prefill of the same token \
         twice. Rope at position zero is the identity, so row one should be \
         the same projection turned by one position — identical rows mean the \
         rotation never reached row one."
    );
}

/// **The same, on the ladder a deployment that does NOT rescale takes.**
///
/// The gate above and this one look identical and are not: they differ only in
/// `rope_freq_table`, which is the whole of what parts `neox_freqs_mb` from
/// `neox_mb`. Running both is what separated the two defects that produced the
/// same symptom.
///
/// The lane choice was one of them — the rescaled branch named its DECODE
/// symbol whatever the fire was. The other was underneath both lanes:
/// `Rule::Rope` took its head axis from the fire's `q_heads` while the rotation
/// is stated once per tensor, so k's launch covered thirty-two heads of an
/// eight-head buffer and `neox_mb` strided its rows by q's width. Fixing the
/// lane alone leaves this failing; fixing the axis alone leaves the one above
/// failing. Neither is visible from a single lane, so both lanes stay.
///
/// **How the lane is chosen changed, and it had to.** This gate used to force
/// its lane with `metal.rope_freq_table = false` — it took the facts the
/// driver had rebuilt from the tensors, overwrote one of them, and traced the
/// text from the result. That knob does not exist any more: the ladder is the
/// ROW's statement, reached through `Variant::trace`, and a test that could
/// still flip it would be testing a text no deployment can ask for. So the
/// lane is chosen by WHICH snapshot `PIE_METAL_SMOKE_CHECKPOINT` names, and
/// this gate skips the snapshots that take the other one. `qwen3-0.6b` does
/// not rescale and lands here; `llama-3.2-1b` does and lands in the twin
/// above. Both are pinned live by `catalog_coverage.rs`, so neither lane can
/// quietly stop being reachable — and running the suite against one snapshot
/// now covers one lane, which is why the twins report their skip loudly.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_prefill_rotates_its_second_row_on_the_base_ladder() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // The SAME token twice. Same embedding, same projection, so the two K rows
    // leave the matmul bit-identical and only the rotation can part them.
    let step = Step {
        token_ids: &[9906, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    // This is the BASE lane's fire; the twin above is the FREQS one. See this
    // gate's doc for why the lane is the snapshot's to choose and not this
    // test's to force.
    if dg.rope_rescale.is_some() {
        eprintln!(
            "SKIP: this checkpoint rescales its ladder, so the FREQS \
             lane is the one it takes -- see the twin above."
        );
        return;
    }
    let plan = text(row, FireClass::Prefill, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 16);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    let staged = stage_prefill(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        dispatch_geometry(&dg, &binding),
        &mut live,
    )
    .expect("the prefill runs");

    // Layer zero's keys. `stage_prefill` writes row r into page zero at slot r,
    // and the layer is `[pages, page_size, kv_heads * head_dim]`.
    let layer = pool.layer(0).expect("layer zero is pooled");
    let row_bytes = shape.row_bytes_at(0) as usize;
    let mut keys = vec![0u8; row_bytes * 2];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // and the pool's K region is at least two rows wide.
    unsafe {
        let raw = core::slice::from_raw_parts(
            layer
                .k
                .host_span(0, keys.len() as u64)
                .expect("the pages are addressable")
                .as_ptr()
                .cast_const(),
            keys.len(),
        );
        keys.copy_from_slice(raw);
    }
    let (row0, row1) = keys.split_at(row_bytes);

    assert!(
        row0.iter().any(|b| *b != 0),
        "row zero's key is all zeros, so nothing was written and the rest of \
         this gate would pass for the wrong reason"
    );
    assert!(
        row0 != row1,
        "the two K rows are byte-identical after a prefill of the same token \
         twice. Rope at position zero is the identity, so row one should be \
         the same projection turned by one position — identical rows mean the \
         rotation never reached row one."
    );
}

/// One request's tables: every token belongs to request zero and lands in that
/// request's page at its own offset.
///
/// `stage_tables` states one request PER TOKEN, which is a decode's shape. A
/// prefill is the other one, and getting it wrong makes every token its own
/// sequence — which attends to nothing and is exactly the answer a broken
/// attention gives.
fn stage_prefill(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
    freqs: &[f32],
) -> driver_metal::bind::tables::Staged {
    let n = step.token_ids.len() as u32;
    // FROM `qo_indptr`. This staged EVERY prefill as one request: positions
    // `0..n` counted across the whole fire, `req_of_token` all zero, a single
    // page, and `kv_page_indptr = [0, 1]` saying "one request" whatever the
    // step said.
    //
    // For a one-request prefill that is exactly right, which is why it
    // survived. For a TWO-request prefill it is a different fire than the one
    // asked for: the two requests are concatenated into one sequence, the
    // second one's first token is rotated as though it were the third token
    // of the first, and both write their keys into the same page where causal
    // attention then lets the second read the first's.
    //
    // `a_request_prefills_the_same_way_beside_another_one` is the gate for
    // precisely that leak, and it was staging the leak itself.
    let requests = step.qo_indptr.len().saturating_sub(1);
    let mut req_of_token: Vec<u32> = Vec::with_capacity(n as usize);
    let mut positions: Vec<u32> = Vec::with_capacity(n as usize);
    for r in 0..requests {
        let (from, to) = (step.qo_indptr[r], step.qo_indptr[r + 1]);
        for row in from..to {
            req_of_token.push(r as u32);
            positions.push(row - from);
        }
    }
    if req_of_token.len() != n as usize {
        req_of_token = vec![0; n as usize];
        positions = (0..n).collect();
    }
    // ONE PAGE PER REQUEST, so a request's keys are somewhere another request
    // is not looking. Each request's rows are far inside one page here; a
    // longer prefill would need a run of them, and `pages_for` is where that
    // would go.
    let requests = requests.max(1) as u32;
    let each: Vec<u32> = (0..requests).collect();
    let indptr: Vec<u32> = (0..=requests).collect();
    let w_off: Vec<u32> = positions.iter().map(|p| p % page_size.max(1)).collect();
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    driver_metal::bind::tables::stage(
        context,
        // A pool that lives exactly as long as the `Staged` does -- the lease
        // keeps it alive, and a helper that stages once wants no more.
        &driver_metal::fire::Scratch::new(),
        driver_metal::bind::tables::Frame {
            token_ids: step.token_ids,
            position_ids: &positions,
            req_of_token: &req_of_token,
            kv_page_indices: &each,
            kv_page_indptr: &indptr,
            kv_write_page: &req_of_token,
            kv_write_offset: &w_off,
            rope_frequencies: &inv_freq,
            // The FIRE's rows, exactly as `serve::launch` stages them: the
            // wire's numbers are request-local and `row_gather` indexes the
            // stream. A rig that passed them through would be testing a
            // staging the driver does not do.
            sampling_indices: &driver_metal::lowering::frame::sampled_rows(step)
                .expect("the readout table places its rows"),
        },
    )
    .expect("the tables stage")
}

/// **A generation, token for token, against MLX.**
///
/// The standard `device_smoke.rs` holds the retiring path to — decode a
/// sequence and compare every token — through the generic executor instead.
/// It is the last thing between `batch/dispatch_llama.rs` and the bin.
///
/// One prefill of `[BOS, "Hello"]` then three decodes, each reading the KV the
/// last one wrote. That carryover is what a single-fire gate cannot reach: an
/// append that lands one row off, a page index that does not advance, a stride
/// that is right for the first token and wrong for the second — none of them
/// show until a second fire reads what a first one wrote.
///
/// MLX's greedy continuation from the same prompt is `0, 358, 2846, 12304`,
/// computed by recomputing the WHOLE prefix at every step so that nothing is
/// carried on the reference side. A KV bug here cannot hide in a shared
/// assumption.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_generation_agrees_with_mlx_token_for_token() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    // The reference below is llama-3.2-1B-Instruct-4bit's, taken by hand.
    if !reference_taken_on(&snapshot, "Llama-3.2-1B-Instruct-4bit") {
        return;
    }
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    // ONE pool for the whole generation. That is the point: every fire after
    // the first reads what its predecessors wrote.
    let shape = pool_shape(&dg, 16);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();

    const MLX: [u32; 4] = [0, 358, 2846, 12304];
    let mut seq: Vec<u32> = vec![128_000, 9906];
    let mut got: Vec<u32> = Vec::new();

    for (turn, &want) in MLX.iter().enumerate() {
        // The first fire is the PREFILL of the prompt; every one after is a
        // decode of the last token at its own position.
        let (tokens, first): (Vec<u32>, u32) = if turn == 0 {
            (seq.clone(), 0)
        } else {
            (vec![*seq.last().expect("a sequence")], seq.len() as u32 - 1)
        };
        let n = tokens.len() as u32;
        let positions: Vec<u32> = (first..first + n).collect();
        let class = if n > 1 {
            FireClass::Prefill
        } else {
            FireClass::Decode
        };

        let step = Step {
            token_ids: &tokens,
            qo_indptr: &[0, n],
            sampling_indices: &[n - 1],
            sampling_indptr: &[0, 1],
            ..Step::default()
        };
        let plan = text(row, class, &binding);
        let lowered = lower_step(&plan, &step).expect("the step lowers");

        // One request, one page list. The write destinations advance with the
        // POSITION, which is what makes each fire land after the last.
        let zeros: Vec<u32> = vec![0; n as usize];
        let w_off: Vec<u32> = positions.iter().map(|p| p % shape.page_size).collect();
        let staged = driver_metal::bind::tables::stage(
            &context,
            &driver_metal::fire::Scratch::new(),
            driver_metal::bind::tables::Frame {
                token_ids: &tokens,
                position_ids: &positions,
                req_of_token: &zeros,
                kv_page_indices: &[0],
                kv_page_indptr: &[0, 1],
                kv_write_page: &zeros,
                kv_write_offset: &w_off,
                rope_frequencies: &inv_freq,
                sampling_indices: &[n - 1],
            },
        )
        .expect("the tables stage");

        let named = HashMap::new();
        let mut live = Live {
            store: Store::new(Names::mlx(), &loaded.tensors, &named),
            tables: &staged,
            shape,
            pages: &pages,
        };
        let (_, arena) = driver_metal::fire::run::run_keeping_arena(
            &context,
            &compiler,
            &mut pipelines,
            &lowered,
            dispatch_geometry(&dg, &binding),
            &mut live,
        )
        .expect("the fire runs");

        let mut read = vec![0u8; arena.len() as usize];
        // SAFETY: the command buffer retired before the call returned.
        unsafe {
            let raw = core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            );
            read.copy_from_slice(raw);
        }

        let (at, width, element) = {
            // The text's OWN statement of where its answer is, not a guess at it.
            // This used to take the widest arena region, which was right by luck:
            // the gemma text holds TWO vocabulary-wide buffers, because the logit
            // softcap is out of place, and the tie-break picked the capped one.
            let r = lowered.readout.expect("the text states an exit seam");
            (r.at, r.vocab as usize, r.bytes as usize)
        };
        let logits: Vec<f32> = read[at..at + width * element]
            .chunks_exact(element)
            .map(|c| {
                if element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                }
            })
            .collect();
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .expect("a readout has an argmax");

        eprintln!("turn {turn}: {next} (MLX {want})");
        got.push(next);
        seq.push(next);
    }

    assert_eq!(
        got, MLX,
        "the generation diverged from MLX. A first token that agrees and a \
         second that does not is the KV carryover, which no single-fire gate \
         reaches."
    );
}

/// The REPLAYED path, over a real checkpoint, against the encoded one.
///
/// # The gap this closes
///
/// `submit` serves by replaying a recorded indirect command buffer — 424
/// dispatches issued as one `executeCommandsInBuffer`, 311× cheaper than
/// encoding them. `device_icb.rs` proves the two paths agree byte-for-byte,
/// but on **sentinel weights**: one region answering every name. The tests
/// above prove the ENCODED path agrees with MLX, but they call
/// `run_keeping_arena`, which encodes.
///
/// So each half was covered and nothing spanned both. What a real checkpoint
/// adds is **address diversity**: the weight arena, the fire tables, the
/// activation arena, the scalars, and two spans per layer for the KV pool.
/// Every one of those addresses becomes a `setKernelBuffer` on a recorded
/// command through `Regions::resolve`, and a resolution to the wrong span is
/// a kernel reading another layer's cache — silently.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_replayed_fire_over_real_weights_agrees_with_the_encoded_one() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    let step = Step {
        token_ids: &[128_000],
        qo_indptr: &[0, 1],
        sampling_indices: &[0],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let plan = text(row, FireClass::Decode, &binding);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = pool_shape(&dg, 16);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(&dg);
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);
    let geometry = dispatch_geometry(&dg, &binding);
    let (at, vocab, element) = {
        let r = lowered.readout.expect("the text states an exit seam");
        (r.at, r.vocab as usize, r.bytes as usize)
    };
    let logits_of = |bytes: &[u8]| -> Vec<f32> {
        bytes[at..at + vocab * element]
            .chunks_exact(element)
            .map(|c| {
                if element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                }
            })
            .collect()
    };

    // ── The ENCODED path, which the tests above hold to MLX. ──
    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry,
        &mut live,
    )
    .expect("the encoded fire runs");
    // SAFETY: the command buffer retired before the call returned.
    let encoded_arena = unsafe {
        core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        )
    }
    .to_vec();
    let encoded = logits_of(&encoded_arena);

    // ── The REPLAYED path, which is what serves. ──
    //
    // Every region a fire's operands may land in, registered so the recording
    // can turn an address back into a buffer. This is the list a seam builds:
    // the weights, the pool's layers, the tables — and `submit` adds the
    // arena and the scalars it leases.
    let mut regions = driver_metal::device::Regions::new();
    for region in &loaded.regions {
        regions.add(region);
    }
    for l in 0..shape.layers {
        if let Some(layer) = pool.layer(l) {
            layer.k.register(&mut regions);
            layer.v.register(&mut regions);
        }
    }
    regions.add(staged.region());
    regions.set_null(staged.region());

    let mut recordings = driver_metal::fire::Recordings::new();
    let scratch = driver_metal::fire::Scratch::new();
    let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let replayed_arena;
    let replayed = {
        let mut machine = driver_metal::fire::run::Machine {
            context: &context,
            compiler: &compiler,
            pipelines: &mut pipelines,
            stepper: &mut stepper,
            scratch: &scratch,
            regions: &mut regions,
            recordings: Some(&mut recordings),
        };
        let fire = driver_metal::fire::run::submit(&mut machine, &lowered, geometry, &mut live)
            .expect("the replayed fire commits");
        machine
            .stepper
            .wait_for(fire.value)
            .expect("the replayed fire retires");
        // SAFETY: waited for above.
        replayed_arena = unsafe {
            core::slice::from_raw_parts(
                fire.arena.contents().as_ptr().cast_const().cast::<u8>(),
                fire.arena.len() as usize,
            )
        }
        .to_vec();
        logits_of(&replayed_arena)
    };
    // WHICH STATEMENT the two paths part at, not which logits.
    //
    // The readout is the last value a fire writes, so comparing it says only
    // that something upstream differs -- the same blindness the NaN detector
    // had before it reported the arena offset. The arenas are the whole
    // computation, and the first byte they disagree on lands in one value,
    // which one launch wrote.
    if encoded_arena != replayed_arena {
        let at = encoded_arena
            .iter()
            .zip(&replayed_arena)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let value = lowered
            .value_offset
            .iter()
            .enumerate()
            .filter(|(_, o)| **o <= at)
            .max_by_key(|(_, o)| **o);
        let wrote = value.and_then(|(v, o)| {
            lowered
                .launches
                .iter()
                .position(|l| {
                    lowered.args[l.args.start as usize..l.args.end as usize]
                        .iter()
                        .any(|a| matches!(a, model_compiler::lower::Arg::Arena { at, .. } if at == o))
                })
                .map(|n| (v, *o, n, lowered.kernels[lowered.launches[n].kernel as usize].clone()))
        });
        eprintln!(
            "  the arenas first differ at byte {at} of {}; {:?}",
            encoded_arena.len(),
            wrote.map(|(v, o, n, k)| format!("value {v} @{o}, first bound by statement {n} `{k}`"))
        );
    }

    // The fire was RECORDED, not silently encoded. `submit` falls back when a
    // recording cannot be made -- right for serving, useless here, and
    // otherwise this compares the encode path with itself. Falsified: emptying
    // the region registry makes this fail with `left: 0`.
    assert_eq!(
        recordings.recorded(),
        1,
        "the fire was not recorded, so this proved nothing"
    );

    let mut order: Vec<usize> = (0..encoded.len()).collect();
    order.sort_by(|&a, &b| encoded[b].total_cmp(&encoded[a]));
    let mut replayed_order: Vec<usize> = (0..replayed.len()).collect();
    replayed_order.sort_by(|&a, &b| replayed[b].total_cmp(&replayed[a]));
    eprintln!(
        "encoded argmax {} ({:.6}), replayed argmax {} ({:.6}) over {vocab} logits",
        order[0], encoded[order[0]], replayed_order[0], replayed[replayed_order[0]]
    );
    // HOW they differ, not only that they do. An equality assertion over a
    // quarter of a million logits prints two lists and says nothing: one
    // operand bound to the wrong buffer and a recording that resolved nothing
    // both fail it, and the counts tell them apart -- a handful of differing
    // logits is an operand, all of them NaN is a resolution.
    let differ = encoded
        .iter()
        .zip(&replayed)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if differ > 0 {
        eprintln!(
            "  {differ} of {vocab} logits differ; encoded holds {} NaN, replayed {}",
            encoded.iter().filter(|v| v.is_nan()).count(),
            replayed.iter().filter(|v| v.is_nan()).count()
        );
    }

    // BIT-IDENTICAL, not within a tolerance. The two paths issue the same
    // kernels over the same buffers in the same order; the only difference is
    // who tells the GPU about them. Anything but equality is a recording that
    // bound something else.
    assert_eq!(
        encoded, replayed,
        "the replayed fire computes different logits from the encoded one -- \
         and the encoded one is what the MLX gates above hold to a reference"
    );
    assert!(
        encoded.iter().any(|v| *v != 0.0 && v.is_finite()),
        "both paths produced nothing usable, so the comparison proved nothing"
    );
}

/// Two requests' tables for a batched PREFILL: request r owns page r and its
/// tokens are positioned from zero inside it.
///
/// This is the third staging shape in this file and the one no gate had.
/// `stage_tables` states one request PER TOKEN (a decode fleet) and
/// `stage_prefill` states ONE request holding every token. A served frame is
/// routinely neither: several requests, each with several tokens.
fn stage_prefill_fleet(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
    freqs: &[f32],
) -> driver_metal::bind::tables::Staged {
    let bounds = step.qo_indptr;
    let requests = bounds.len() as u32 - 1;
    let page_size = page_size.max(1);
    let mut positions = Vec::new();
    let mut req_of_token = Vec::new();
    let mut write_page = Vec::new();
    let mut write_offset = Vec::new();
    // As MANY pages as the request's tokens need, and the count is the whole
    // point. This staged ONE page per request and wrapped the offset with
    // `p % page_size`, so a request longer than a page wrote its seventeenth
    // token over its first one's keys and then attended over sixteen slots of
    // a sequence it thought was sixty-four. Every gate in this file prefills
    // five tokens or fewer, so nothing had ever crossed a page — which made
    // the harness, not the driver, the thing that could not answer a long
    // prompt.
    let mut page_indices: Vec<u32> = Vec::new();
    let mut page_indptr: Vec<u32> = vec![0];
    for r in 0..requests {
        let (lo, hi) = (bounds[r as usize], bounds[r as usize + 1]);
        let first = page_indices.len() as u32;
        let pages = (hi - lo).div_ceil(page_size);
        for p in 0..(hi - lo) {
            positions.push(p);
            req_of_token.push(r);
            write_page.push(first + p / page_size);
            write_offset.push(p % page_size);
        }
        // Distinct pages per request, so nothing one request writes lands in
        // another's window.
        page_indices.extend(first..first + pages);
        page_indptr.push(page_indices.len() as u32);
    }
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    driver_metal::bind::tables::stage(
        context,
        // A pool that lives exactly as long as the `Staged` does -- the lease
        // keeps it alive, and a helper that stages once wants no more.
        &driver_metal::fire::Scratch::new(),
        driver_metal::bind::tables::Frame {
            token_ids: step.token_ids,
            position_ids: &positions,
            req_of_token: &req_of_token,
            kv_page_indices: &page_indices,
            kv_page_indptr: &page_indptr,
            kv_write_page: &write_page,
            kv_write_offset: &write_offset,
            rope_frequencies: &inv_freq,
            // The FIRE's rows, exactly as `serve::launch` stages them: the
            // wire's numbers are request-local and `row_gather` indexes the
            // stream. A rig that passed them through would be testing a
            // staging the driver does not do.
            sampling_indices: &driver_metal::lowering::frame::sampled_rows(step)
                .expect("the readout table places its rows"),
        },
    )
    .expect("the tables stage")
}

/// Read one sampled row's logits out of a retired arena.
fn logits_at(read: &[u8], lowered: &model_compiler::lower::Lowered, row: usize) -> Vec<f32> {
    let r = lowered.readout.expect("the text states an exit seam");
    let (width, element) = (r.vocab as usize, r.bytes as usize);
    let at = r.at + row * width * element;
    read[at..at + width * element]
        .chunks_exact(element)
        .map(|c| {
            if element == 4 {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            } else {
                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
            }
        })
        .collect()
}

/// Everything a prefill needs that does not change between fires.
struct Rig<'a> {
    context: &'a Context,
    compiler: &'a Compiler,
    loaded: &'a driver_metal::weights::load::Loaded,
    row: &'static dyn model::catalog::Variant,
    binding: &'a MetalBinding,
    dg: &'a driver_metal::batch::DecodeGeometry,
}

/// Run one prefill fire and hand back every sampled row's logits.
///
/// Staged as a FLEET whatever the request count, which costs nothing: at one
/// request `stage_prefill_fleet` produces exactly `stage_prefill`'s tables —
/// positions from zero, request zero, page zero. So the comparison below runs
/// both fires through one staging path and a difference between them cannot be
/// the harness.
/// Which storage a pool under test is built on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    /// One allocation per layer side, sized once.
    Fixed,
    /// A sparse address space per layer side, committed to the same size.
    Elastic,
}

fn prefill_logits(rig: &Rig<'_>, pipelines: &mut Pipelines, step: &Step<'_>) -> Vec<Vec<f32>> {
    prefill_logits_on(rig, pipelines, step, Storage::Fixed)
}

fn prefill_logits_on(
    rig: &Rig<'_>,
    pipelines: &mut Pipelines,
    step: &Step<'_>,
    storage: Storage,
) -> Vec<Vec<f32>> {
    let Rig {
        context,
        compiler,
        loaded,
        row,
        binding,
        dg,
    } = *rig;
    let plan = text(row, FireClass::Prefill, binding);
    let lowered = lower_step(&plan, step).expect("the step lowers");

    let shape = pool_shape(dg, 16);
    // The arena outlives the pool: an elastic buffer charges its tiles back
    // on drop, and dropping the arena first would leave nothing to charge.
    let arena_for_pool = driver_metal::device::Arena::new(1024 * 1024 * 1024, 0);
    let pool = match storage {
        Storage::Fixed => Pool::allocate(context, shape).expect("a pool"),
        Storage::Elastic => {
            let mut stepper = driver_metal::device::Stepper::new(context).expect("stepper");
            Pool::allocate_elastic(context, &mut stepper, &arena_for_pool, shape)
                .expect("an elastic pool")
        }
    };
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::table(dg);
    let staged = stage_prefill_fleet(context, step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        context,
        compiler,
        pipelines,
        &lowered,
        dispatch_geometry(dg, binding),
        &mut live,
    )
    .expect("the prefill runs");

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before the call returned.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }
    (0..step.sampling_indices.len())
        .map(|row| logits_at(&read, &lowered, row))
        .collect()
}

/// **A request's answer does not depend on what shares its fire.**
///
/// `device_smoke.rs`'s tombstone names this as one of two claims no current
/// gate makes, and it is the one a served frame exercises constantly: the
/// engine batches whatever is ready. Every device gate here runs either ONE
/// request holding every token or one request PER token — a decode fleet.
/// Several requests each holding several tokens, which is the shape of a
/// batched prefill, was never run.
///
/// The check needs no reference. The same prompt is prefilled alone and then
/// again beside a second, longer, unrelated request, and the two answers must
/// agree bit for bit. Anything that leaks between requests — a position that
/// counts from the fire rather than the request, a mask that lets row two see
/// row one's sequence, a page index that does not advance — moves the first
/// request's distribution, and the fire's own arithmetic is the only thing
/// that could have moved it.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_request_prefills_the_same_way_beside_another_one() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    let rig = Rig {
        context: &context,
        compiler: &compiler,
        loaded: &loaded,
        row,
        binding: &binding,
        dg: &dg,
    };

    let alone = Step {
        token_ids: &[128_000, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let solo = prefill_logits(&rig, &mut pipelines, &alone);

    // The same two tokens, then a THIRD-and-fourth of someone else's. The
    // second request is deliberately a different length and different tokens,
    // so nothing about it can be mistaken for the first's.
    let together = Step {
        token_ids: &[128_000, 9906, 128_000, 3923, 374],
        qo_indptr: &[0, 2, 5],
        sampling_indices: &[1, 2],
        sampling_indptr: &[0, 1, 2],
        ..Step::default()
    };
    let batched = prefill_logits(&rig, &mut pipelines, &together);

    // The SECOND request run alone. This is the sensitive direction and the
    // reason the comparison is not one-sided: attention is causal, so nothing
    // the second request does can reach the first one's rows even if the fire
    // were staged as a single sequence. The second request is where a leak
    // shows — it is the one that could read the first's keys, count its
    // positions from the fire instead of from itself, or land in its page.
    let second_alone = Step {
        token_ids: &[128_000, 3923, 374],
        qo_indptr: &[0, 3],
        sampling_indices: &[2],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let solo_b = prefill_logits(&rig, &mut pipelines, &second_alone);

    assert_eq!(batched.len(), 2, "the fire samples both requests");
    // WHICH KIND of nothing, because zeros and NaNs are different bugs: an
    // all-zero row is a lane that never ran, an all-NaN row is one that ran
    // on bad operands, and "produced distributions at all" named neither.
    let describe = |row: &[f32]| {
        let nan = row.iter().filter(|v| v.is_nan()).count();
        let zero = row.iter().filter(|v| **v == 0.0).count();
        format!("{} logits: {nan} NaN, {zero} zero", row.len())
    };
    assert!(
        solo_b[0].iter().any(|v| v.is_finite() && *v != 0.0),
        "the three-token solo prefill produced no distribution -- {}; \
         the two-token one beside it is {}",
        describe(&solo_b[0]),
        describe(&solo[0])
    );
    for (which, alone, batched) in [
        ("first", &solo[0], &batched[0]),
        ("second", &solo_b[0], &batched[1]),
    ] {
        let worst = alone
            .iter()
            .zip(batched)
            .enumerate()
            .max_by(|x, y| (x.1.0 - x.1.1).abs().total_cmp(&(y.1.0 - y.1.1).abs()))
            .expect("a vocabulary");
        assert_eq!(
            alone, batched,
            "the {which} request's distribution moved when the other joined \
             its fire. Widest disagreement at token {}: alone {}, batched {}.",
            worst.0, worst.1.0, worst.1.1
        );
    }
}

/// **The batched GEMM answers what the matvec answers.**
///
/// The one claim no gate in this file made, and the gap is structural rather
/// than accidental. The text guards the tiled GEMM with
/// `TokensMultipleOf(qmm_tile.0)` — thirty-two — and every correctness oracle
/// here prefills ONE or TWO tokens. So `affine_qmm_t` was never once compared
/// against anything: the arm that runs on every real prompt was covered only
/// by `attention_is_a_minority_of_a_long_prefill`, which measures TIME.
///
/// What lived in that gap: the driver derived the GEMM's `(bm, bn)` tile from
/// the fire's geometry while `model-compiler` compiled the entrypoint for the
/// tile in its NAME, and at a 512-row prefill the two said `(64, 64)` and
/// `(32, 32)`. `affine_qmm_t` is a template on `BM`/`BN` — `y_row = tid.y *
/// BM` — so the grid is threadgroups times the COMPILED tile, and a
/// threadgroup count computed for a 64-wide tile covers a QUARTER of the
/// output. Every long prefill's projections were three quarters whatever the
/// arena held, through sixteen layers, and every gate here passed.
///
/// **This needs no reference**, which is why it can exist at all: the two arms
/// of the guard are two spellings of one matrix product, so the same tokens
/// through both must give the same distribution. `qmm_multi_batch` is the
/// load-time fact that chooses between them (`LlamaLikeMetalFacts`), so
/// flipping it on the binding is exactly the guard's own choice made by hand.
///
/// SIXTY-FOUR tokens and not thirty-two, and the difference is the point: at
/// 32 rows the derivation and the name agreed on `bm` and differed only on
/// `bn`, so half the output was written. At 64 they differ on both.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_batched_gemm_answers_what_the_matvec_answers() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let observed = observed(&dg, &loaded);
    // The matvec side is the SAME binding with one fact flipped, so a
    // difference between the two answers cannot be a different deployment.
    let unbatched = MetalBinding {
        qmm_multi_batch: false,
        ..observed
    };

    // A run of distinct real tokens rather than one repeated: a projection
    // that wrote only its first tile would still agree with itself on a
    // constant sequence.
    let tokens: Vec<u32> = (0..64u32).map(|i| 1000 + i * 37).collect();
    let step = Step {
        token_ids: &tokens,
        qo_indptr: &[0, 64],
        sampling_indices: &[1, 63],
        sampling_indptr: &[0, 2],
        ..Step::default()
    };

    fn rig<'a>(
        context: &'a Context,
        compiler: &'a Compiler,
        loaded: &'a driver_metal::weights::load::Loaded,
        row: &'static dyn model::catalog::Variant,
        binding: &'a MetalBinding,
        dg: &'a driver_metal::batch::DecodeGeometry,
    ) -> Rig<'a> {
        Rig {
            context,
            compiler,
            loaded,
            row,
            binding,
            dg,
        }
    }
    // ONE pipeline cache for both fires, so a difference between them cannot
    // be a shader compiled two ways.
    let gemm = prefill_logits(
        &rig(&context, &compiler, &loaded, row, &observed, &dg),
        &mut pipelines,
        &step,
    );
    let matvec = prefill_logits(
        &rig(&context, &compiler, &loaded, row, &unbatched, &dg),
        &mut pipelines,
        &step,
    );

    // Without this the test passes by taking the matvec twice: a deployment
    // whose `qmm_multi_batch` were false would compare nothing.
    assert!(
        observed.qmm_multi_batch,
        "this checkpoint states `qmm_multi_batch: false`, so both sides ran \
         the matvec and nothing was compared"
    );
    assert!(
        matvec[0].iter().any(|v| v.is_finite() && *v != 0.0),
        "the matvec side produced no distribution at all"
    );

    // The CONTROL, and the ARBITER. Two tokens is below the guard, so both
    // bindings resolve to the matvec and the two fires differ only in a guard
    // neither takes -- their answers must agree bit for bit, and a failure
    // here is the harness rather than either kernel.
    //
    // It is also the reference the 64-row fires are measured against, and it
    // needs no MLX: attention is CAUSAL, so row 1 of a sixty-four token
    // prefill sees tokens 0 and 1 and nothing else. Its distribution is the
    // two-token prefill's, whichever projection kernel produced it.
    //
    // That is what found the defect under this one. `stage_prefill_fleet`
    // staged ONE page per request and wrapped the write offset, so at 64
    // tokens over a 16-token page the seventeenth token overwrote the first's
    // keys -- and row 1 came back 15 logits away from its own two-token
    // answer on BOTH sides. Every gate in this file prefills five tokens or
    // fewer; the harness had never crossed a page.
    let short = Step {
        token_ids: &tokens[..2],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let a = prefill_logits(
        &rig(&context, &compiler, &loaded, row, &observed, &dg),
        &mut pipelines,
        &short,
    );
    let b = prefill_logits(
        &rig(&context, &compiler, &loaded, row, &unbatched, &dg),
        &mut pipelines,
        &short,
    );
    assert_eq!(
        a[0], b[0],
        "below the guard both bindings take the matvec, so this comparison is \
         of one program with itself -- and it disagreed"
    );

    let worst_of = |x: &[f32], y: &[f32]| -> (usize, f32, f32) {
        let (at, (p, q)) = x
            .iter()
            .zip(y)
            .enumerate()
            .max_by(|m, n| (m.1.0 - m.1.1).abs().total_cmp(&(n.1.0 - n.1.1).abs()))
            .expect("a vocabulary");
        (at, *p, *q)
    };
    // The scale the arithmetic ran at, which is the distribution's and not any
    // one logit's: sixteen layers of bf16 rounding accumulate proportionally
    // to the activations, so a tolerance read off a logit near zero demands a
    // precision no part of the network has.
    let span = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let slack = 2.0 * bf16_slack(span(&matvec[1]));

    // The matvec is the SAME kernel in both fires, so causality is exact
    // there -- and this is the direction that fails hardest when the KV
    // paging is wrong.
    let (at, p, q) = worst_of(&a[0], &matvec[0]);
    assert_eq!(
        a[0], matvec[0],
        "row 1 of a 64-token prefill is not row 1 of a 2-token prefill under \
         the SAME kernel. Widest gap at token {at}: short {p}, long {q}. \
         Attention is causal, so nothing after row 1 may reach it."
    );
    // The GEMM's own causality, to a tolerance, because it is a different
    // kernel summing the same row in a different order.
    let (at, p, q) = worst_of(&a[0], &gemm[0]);
    assert!(
        (p - q).abs() <= slack,
        "row 1 under the GEMM is {q} at token {at} where the two-token \
         prefill says {p}, further than {slack}. Causality does not depend on \
         which projection kernel ran."
    );

    // NOT bit for bit: the two kernels sum a row in different orders, so the
    // last bits differ legitimately. A quarter-written projection does not
    // land within a couple of bf16 ulps of anything -- when the tile was
    // derived rather than read off the name, this gap was 14.
    let (at, g, m) = worst_of(&gemm[1], &matvec[1]);
    assert!(
        (g - m).abs() <= slack,
        "the GEMM and the matvec disagree at token {at}: GEMM {g}, matvec {m}, \
         which is further apart than {slack}. They are two spellings of one \
         matrix product."
    );

    // The ORDER too, because a distribution can stay within tolerance
    // pointwise and still rank differently at the top, which is the only part
    // a sampler reads.
    let top5 = |v: &[f32]| {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&x, &y| v[y].total_cmp(&v[x]));
        order[..5].to_vec()
    };
    assert_eq!(
        top5(&gemm[1]),
        top5(&matvec[1]),
        "the two arms rank the top of the distribution differently"
    );
}

/// **An elastic pool answers exactly as a fixed one does.**
///
/// The point of elastic KV is that a pool can be resized without every
/// address bound into an argument table moving. The point of THIS gate is
/// that the change is free before anyone resizes anything: the pages are in
/// placement heaps behind a sparse buffer instead of in one allocation, and
/// nothing above them may be able to tell.
///
/// A weaker check — that the fire runs, or that the activations are finite —
/// would pass over a pool whose rows landed a page apart, because attention
/// over the wrong keys is still finite. So the comparison is bit-for-bit
/// against the same fire on a fixed pool: same weights, same tokens, same
/// staging, one storage difference. Real weights matter here for the same
/// reason they matter to the rope gates — a synthetic pool of zeros gives the
/// same answer whatever it is read through.
///
/// **What it does not reach.** Five tokens over a page size of sixteen touch
/// the first page of each layer and no other, so a commit that covered only
/// the front of every buffer would pass this — measured, by halving it. What
/// fails it is a pool with nothing mapped, which is the shape the mistake
/// takes when a commit is skipped rather than shortened. The per-page
/// arithmetic is gated in `device_elastic.rs` instead, where a span past what
/// is mapped is refused rather than served.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn an_elastic_pool_answers_exactly_as_a_fixed_one_does() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);

    let rig = Rig {
        context: &context,
        compiler: &compiler,
        loaded: &loaded,
        row,
        binding: &binding,
        dg: &dg,
    };

    // Two requests of different lengths, so the fire reads pages belonging to
    // more than one sequence: a storage seam that only showed up past the
    // first request's rows would survive a single-sequence check.
    let step = Step {
        token_ids: &[128_000, 9906, 128_000, 3923, 374],
        qo_indptr: &[0, 2, 5],
        sampling_indices: &[1, 2],
        sampling_indptr: &[0, 1, 2],
        ..Step::default()
    };

    let fixed = prefill_logits_on(&rig, &mut pipelines, &step, Storage::Fixed);
    let elastic = prefill_logits_on(&rig, &mut pipelines, &step, Storage::Elastic);

    assert_eq!(
        fixed.len(),
        elastic.len(),
        "the same fire sampled a different number of rows"
    );
    for (row, (want, got)) in fixed.iter().zip(&elastic).enumerate() {
        assert_eq!(
            want.len(),
            got.len(),
            "row {row}: the two pools produced different vocabulary widths"
        );
        let differ = want
            .iter()
            .zip(got)
            .enumerate()
            .find(|(_, (a, b))| a.to_bits() != b.to_bits());
        assert!(
            differ.is_none(),
            "row {row}: an elastic pool changed the answer. \
             {differ:?} — the pages are the only thing that differs between \
             these two runs, so a logit that moved means the fire read \
             different bytes: heaps that do not sit where the sparse buffer \
             says, a commit that did not reach every layer, or a zero that \
             did not clear what a fixed allocation happened to have clear"
        );
    }
}

/// **Attention is a minority of a long prefill, because the prefill tiles.**
///
/// Every other device gate here asks what a fire ANSWERS. This one asks what
/// it COSTS, and it is here because the answer decided which kernel the text
/// names. `dsl::metal::sdpa` branches on `multi_batch` to a different shader,
/// and a branch taken on someone else's measurement is a branch taken on
/// faith: `sdpa_paged_tiled.metal`'s header measures the 30B checkpoint, and
/// nothing in this repository had ever measured a Metal prefill.
///
/// **The estimator.** Prefill time is `a + b*n + c*n^2` — a constant to
/// stage the fire, a linear term for everything that touches each token
/// once, and a quadratic one for attention, where every query row reads
/// every key before it. Three geometrically spaced points annihilate the
/// affine part exactly:
///
/// ```text
///   t(4m) - 3*t(2m) + 2*t(m) = 6*c*m^2
/// ```
///
/// because `a` cancels as `1 - 3 + 2` and `b*n` as `4 - 6 + 2`. So `c` falls
/// out of three timings with no fitting and no assumption about `a` or `b`,
/// which is the point: whatever the constant costs, it cannot contaminate
/// the number this gate reads.
///
/// **What it measured** (Llama-3.2-1B-Instruct-4bit, M-series, release,
/// best of six per point, milliseconds):
///
/// ```text
///     n     decode kernel   tiled kernel   mma kernel
///   512         406.0           347.9         325.6
///  1024         953.4           741.3         662.7
///  2048        2486.8          1684.2        1377.6
///   c       2.788e-4        9.930e-5      2.589e-5   ms per token^2
///   quadratic term at n=2048:
///              1169 ms (47%)    417 ms (25%)  109 ms (8%)
/// ```
///
/// The linear term barely moved (634 -> 610 us/token) and the constant not
/// at all, which is the check on the check: only the attention kernel
/// changed, so only the quadratic coefficient should have. It fell by 64%,
/// and the saving grows as `n^2` — 58 ms at 512, 212 at 1024, 803 at 2048,
/// almost exactly four times per doubling.
///
/// **What it measures on a mixture of experts** (gpt-oss-20b-MXFP4-Q4, same
/// machine, same estimator, milliseconds):
///
/// ```text
///     n     decode kernel   tiled kernel   mma kernel
///   512        3686.7          3559.9        3503.9
///  1024        7580.2          7147.9        6994.3
///  2048       16030.1         14583.8       14051.9
///   c       4.215e-4        1.654e-4      4.883e-5   ms per token^2
///   quadratic term at n=2048:
///             1768 ms (11%)    694 ms (5%)   205 ms (1%)
/// ```
///
/// The kernel is the same 2.5x cheaper on the quadratic term, on a second
/// family and on the sink variant — which is the only place
/// `sdpa_paged_tiled_sink` has ever been timed. But the SHARE is 11%
/// against 5% and the threshold below is 40%, so on THIS checkpoint the
/// CLAIM holds whichever kernel is wired: routing every token to four of 32
/// experts across 24 layers is per-token work that swamps attention at
/// every length this gate can afford to run — a 2048-token prefill costs
/// 14.6 s here against the 1B's 1.7 s. A
/// gate that cannot fail is not a gate — it reports a wiring it never
/// tested — so the sensitivity is asserted as well as the claim, and this
/// checkpoint does not pass. It fires one of the two refusals: the
/// sensitivity assert on a quiet machine, and on a busy one the negative
/// coefficient above, because 5% of the total is at this machine's noise
/// floor. Both say the same thing, which is that a mixture of experts is
/// the wrong instrument for this question.
///
/// It also reproduces, independently and on a different checkpoint, the
/// claim the shader's own header makes: the quadratic term is a large
/// minority of an unsized prefill. The header said 39% at n=2048 on the 30B;
/// this says 47% on the 1B. Two machines, two models, one shape.
///
/// **The claim.** Attention costs less than 40% of a 2048-token prefill.
/// Measured on the dense checkpoint at 8% with the mma kernel the DSL now
/// prefers at `_d_64`, 25% with the tiled one behind it, and 47% with the
/// row-by-row decode kernel both replace -- so the threshold sits above
/// every wired state and below the unwired one, and names which is in.
/// Unwiring the tiled branch fires this gate; making attention cheaper still
/// only widens the margin. The ratio is what is asserted and not the
/// milliseconds, because the ratio is the part that does not depend on how
/// fast the machine underneath is: the falsifying run above was repeated on
/// a contended machine where every timing inflated by half — 603/1461/3924
/// against 406/953/2487 — and it still read 51% against 47%. The
/// milliseconds moved and the ratio did not.
///
/// **On measuring three things that must be compared.** The estimator
/// DIFFERENCES its three points, so it amplifies anything that lands on one
/// of them alone: 250 ms of contention on `t(m)` moves the numerator by
/// 500 and on `t(2m)` by 750. Measuring all six repetitions of one length
/// before starting the next puts a contention episode entirely inside one
/// point, and that is how this gate first behaved — on a busy machine it
/// returned 103% and 11% on consecutive runs of the SAME wiring, a false
/// red and a false green from one binary in one minute. Firing one round of
/// all three lengths instead spreads an episode across all three, where
/// `1 - 3 + 2` cancels most of it; the same three runs then read 25%, 25%,
/// 24%. The arrangement is the measurement.
#[test]
#[ignore = "needs a real checkpoint and a device, and times a fire"]
fn attention_is_a_minority_of_a_long_prefill() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());
    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);
    let freqs = driver_metal::model::rope::table(&dg);

    // Geometrically spaced, which is what the estimator above needs. The
    // longest is 2048 because that is the length the shader's header argues
    // about, and short enough that the pool below is a few hundred megabytes.
    const M: u32 = 512;
    const ROUNDS: usize = 6;
    let plan = text(row, FireClass::Prefill, &binding);
    let named = HashMap::new();

    // Everything that is not the fire, done once and outside the timing
    // loop below: that is what lets the loop interleave the three lengths.
    let prepared: Vec<_> = [M, 2 * M, 4 * M]
        .into_iter()
        .map(|n| {
            // Arbitrary ids inside the vocabulary. Attention's cost does not
            // depend on WHICH tokens, only on how many, and this gate reads
            // no logits at all.
            let tokens: Vec<u32> = (0..n).map(|i| 1000 + i % 5000).collect();
            let step = Step {
                token_ids: &tokens,
                qo_indptr: &[0, n],
                sampling_indices: &[n - 1],
                sampling_indptr: &[0, 1],
                ..Step::default()
            };
            let lowered = lower_step(&plan, &step).expect("the step lowers");
            // Four pages of slack past what the tokens need, so the last page
            // being partial cannot round the allocation short.
            let shape = pool_shape(&dg, n.div_ceil(16) + 4);
            let pool = Pool::allocate(&context, shape).expect("a pool");
            let staged = stage_prefill_fleet(&context, &step, shape.page_size, &freqs);
            (n, lowered, shape, pool, staged)
        })
        .collect();

    // INTERLEAVED, and that is the whole point of the arrangement. Measuring
    // all six of one length before starting the next puts a contention
    // episode entirely inside one point, and the estimator DIFFERENCES the
    // three points, so it amplifies exactly that: 250 ms landing on `t(m)`
    // moves the numerator by 500, and on `t(2m)` by 750. Firing one round of
    // all three lengths spreads an episode across all three instead, where
    // `1 - 3 + 2` cancels most of it. Measured on a contended machine, the
    // sequential arrangement returned 103% and 11% on consecutive runs of
    // the SAME wiring.
    //
    // The BEST of six and not the mean. Timing noise on a shared GPU is
    // one-sided — a slow run means something else was resident, a fast one
    // cannot mean the work was skipped — so the minimum is the estimate and
    // the spread is contention. The first round also builds every pipeline
    // these texts name, which is why it is not the only one.
    let mut ms = [f64::MAX; 3];
    let mut worst = [0f64; 3];
    for _ in 0..ROUNDS {
        for (slot, (_, lowered, shape, pool, staged)) in prepared.iter().enumerate() {
            let pages = |layer: u16, values: bool| {
                pool.layer(u32::from(layer)).map(|l| Slice {
                    address: if values {
                        l.v.gpu_address()
                    } else {
                        l.k.gpu_address()
                    },
                    bytes: shape.layer_bytes_at(0),
                })
            };
            let mut live = Live {
                store: Store::new(Names::mlx(), &loaded.tensors, &named),
                tables: staged,
                shape: *shape,
                pages: &pages,
            };
            let started = std::time::Instant::now();
            let (_, _arena) = driver_metal::fire::run::run_keeping_arena(
                &context,
                &compiler,
                &mut pipelines,
                lowered,
                dispatch_geometry(&dg, &binding),
                &mut live,
            )
            .expect("the prefill runs");
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            ms[slot] = ms[slot].min(elapsed);
            worst[slot] = worst[slot].max(elapsed);
        }
    }
    for (slot, (n, ..)) in prepared.iter().enumerate() {
        // The spread is printed because it is how a reader tells a real
        // regression from a busy machine.
        println!(
            "prefill n={n}: {:.1} ms (worst of {ROUNDS}: {:.1} ms)",
            ms[slot], worst[slot]
        );
    }

    let quadratic_per_token_squared =
        (ms[2] - 3.0 * ms[1] + 2.0 * ms[0]) / (6.0 * f64::from(M) * f64::from(M));
    let longest = f64::from(4 * M);
    let attention = quadratic_per_token_squared * longest * longest;
    let share = attention / ms[2];
    println!(
        "c={quadratic_per_token_squared:.3e} ms/token^2, attention at n={longest}: \
         {attention:.0} ms of {:.0} ms ({:.0}%)",
        ms[2],
        share * 100.0
    );

    // A NEGATIVE coefficient means the three points did not resolve a
    // quadratic at all — the machine was too noisy, or this build got
    // something cheaper than attention. Either way the gate has measured
    // nothing and says so rather than passing on a number it does not
    // believe.
    assert!(
        quadratic_per_token_squared > 0.0,
        "no quadratic term resolved from {ms:?} ms; \
         the timings are too noisy to say anything about attention"
    );
    // How much dearer the row-by-row kernel's quadratic term is than the one
    // the DSL actually wires, which at `_d_64` is the matrix-unit form and
    // not the tiled one. Measured twice: 10.8x on Llama-3.2-1B (2.788e-4
    // against 2.589e-5) and 8.6x on gpt-oss-20b (4.215e-4 against 4.883e-5).
    // The smaller is taken, so the sensitivity below is the conservative one.
    //
    // It tracks the WIRED kernel by construction. Were the mma preference
    // removed and the tiled form left to serve 64-wide heads, the ratio
    // would fall to 2.5x-2.7x and this constant would be wrong in the safe
    // direction -- it would refuse to answer on checkpoints it could still
    // have judged, rather than passing one it could not.
    const DECODE_OVER_WIRED: f64 = 8.6;
    // A checkpoint whose per-token work swamps attention passes the claim
    // below whichever kernel is wired, and a gate that cannot fail is worse
    // than no gate: it reports a wiring it never tested. So the gate asks
    // first whether it COULD have failed — whether attention at the dearer
    // kernel's cost would have crossed the threshold — and refuses to
    // answer where it could not, exactly as it refuses a negative
    // coefficient above.
    assert!(
        share * DECODE_OVER_WIRED >= 0.40,
        "this checkpoint cannot tell the two attention kernels apart: at \
         {:.0}% of a {longest:.0}-token prefill, attention would still be \
         under the 40% threshold at {DECODE_OVER_WIRED}x the cost. Its \
         per-token work swamps the quadratic term — run this gate on a \
         dense checkpoint (Llama-3.2-1B measured 8% on the wired mma kernel \
         against 47% on the decode kernel it replaces) where \
         the threshold sits between the two states.",
        share * 100.0
    );
    assert!(
        share < 0.40,
        "attention is {:.0}% of a {longest:.0}-token prefill ({attention:.0} ms of {:.0} ms). \
         The mma kernel measured 8%, the tiled one 25% and the row-by-row \
         decode kernel 47%, so this \
         reads as a prefill that is naming `sdpa_paged_decode` again.",
        share * 100.0,
        ms[2]
    );
}

/// Where a long prefill's time actually goes, by kernel symbol.
///
/// Not a gate — a measurement, and the tool that attributes the gap against
/// mlx-lm. It exploits the fact that these kernels branch on no operand
/// value: a subset of a fire encoded on its own does the same arithmetic on
/// whatever bytes the arena happens to hold, so its time is the time those
/// dispatches cost inside the whole. The subsets therefore sum to roughly
/// the whole, and the residual is what the barriers and the ramp cost.
#[test]
#[ignore = "a measurement, not a gate"]
fn where_a_long_prefill_spends_its_time() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());
    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);
    let freqs = driver_metal::model::rope::table(&dg);
    let plan = text(row, FireClass::Prefill, &binding);
    let named = HashMap::new();

    const N: u32 = 2048;
    const ROUNDS: usize = 5;
    let tokens: Vec<u32> = (0..N).map(|i| 1000 + i % 5000).collect();
    let step = Step {
        token_ids: &tokens,
        qo_indptr: &[0, N],
        sampling_indices: &[N - 1],
        sampling_indptr: &[0, 1],
        ..Step::default()
    };
    let whole = lower_step(&plan, &step).expect("the step lowers");
    let shape = pool_shape(&dg, N.div_ceil(16) + 4);
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let staged = stage_prefill_fleet(&context, &step, shape.page_size, &freqs);

    // Wall and host encode. There is no third number: `Timing::gpu` is an
    // `Option` this path never fills, and reading it as a `f64::MAX` sentinel
    // printed 1.8e308 where a millisecond count belonged.
    let mut best = |lowered: &model_compiler::lower::Lowered| -> (f64, f64) {
        let mut ms = f64::MAX;
        let mut enc = f64::MAX;
        for _ in 0..ROUNDS {
            let pages = |layer: u16, values: bool| {
                pool.layer(u32::from(layer)).map(|l| Slice {
                    address: if values {
                        l.v.gpu_address()
                    } else {
                        l.k.gpu_address()
                    },
                    bytes: shape.layer_bytes_at(0),
                })
            };
            let mut live = Live {
                store: Store::new(Names::mlx(), &loaded.tensors, &named),
                tables: &staged,
                shape,
                pages: &pages,
            };
            let started = std::time::Instant::now();
            let (timing, _arena) = driver_metal::fire::run::run_keeping_arena(
                &context,
                &compiler,
                &mut pipelines,
                lowered,
                dispatch_geometry(&dg, &binding),
                &mut live,
            )
            .expect("the prefill runs");
            ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
            enc = enc.min(timing.encode.as_secs_f64() * 1e3);
        }
        (ms, enc)
    };

    let (total, total_encode) = best(&whole);
    println!("\nwhole fire: {total:.1} ms wall, {total_encode:.1} ms host encode");
    let mut rows: Vec<(f64, usize, String)> = Vec::new();
    for (index, symbol) in whole.kernels.iter().enumerate() {
        let mut only = whole.clone();
        only.launches.retain(|l| usize::from(l.kernel) == index);
        let count = only.launches.len();
        if count == 0 {
            continue;
        }
        let first = &only.launches[0];
        let shape = format!("rows {:?} layers {:?}", first.rows, first.layers);
        let (wall, enc) = best(&only);
        rows.push((
            wall,
            count,
            format!("{symbol}  [{shape}] wall {wall:.1} enc {enc:.1}"),
        ));
    }
    // And one layer's statements one at a time, which is what compares
    // against another runtime's per-projection numbers.
    println!("\nlayer 0, statement by statement (wall minus the ~9.5 ms encode floor):");
    let layer0: Vec<usize> = whole
        .launches
        .iter()
        .enumerate()
        .filter(|(_, l)| l.layers == (0..1) && l.rows == (0..N))
        .map(|(i, _)| i)
        .take(20)
        .collect();
    let mut slowest = (0.0_f64, 0_usize);
    for i in layer0 {
        let mut one = whole.clone();
        let kept = one.launches[i].clone();
        one.launches = vec![kept];
        let (wall, enc) = best(&one);
        if wall - enc > slowest.0 {
            slowest = (wall - enc, i);
        }
        println!(
            "  {:7.2} ms  {}",
            wall - enc,
            whole.kernels[whole.launches[i].kernel as usize]
        );
    }

    // THE SAME LAUNCH, REPEATED. One statement alone in a fire pays whatever a
    // fire pays once -- a submission, a first touch of every page its operands
    // name, a cold cache. Copying that one launch k times and fitting the line
    // separates that constant from the marginal cost of one more dispatch,
    // which is the number that compares against another runtime's per-op time.
    let (alone, which) = slowest;
    println!(
        "\nthe slowest layer-0 statement ({}) repeated in one fire:",
        whole.kernels[whole.launches[which].kernel as usize]
    );
    let mut marginal = Vec::new();
    for k in [1_usize, 2, 4, 8] {
        let mut many = whole.clone();
        many.launches = vec![whole.launches[which].clone(); k];
        let (wall, enc) = best(&many);
        marginal.push((k, wall - enc));
        println!(
            "  x{k:<3} {:8.2} ms  ({:6.2} ms each)",
            wall - enc,
            (wall - enc) / k as f64
        );
    }
    if let (Some(&(k1, t1)), Some(&(k2, t2))) = (marginal.first(), marginal.last()) {
        println!(
            "  marginal {:.2} ms a dispatch, fixed {:.2} ms a fire (alone it read {alone:.2})",
            (t2 - t1) / (k2 - k1) as f64,
            t1 - (t2 - t1) / (k2 - k1) as f64,
        );
    }

    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let attributed: f64 = rows.iter().map(|r| r.0).sum();
    println!("\nprefill n={N}: {total:.1} ms total, {attributed:.1} ms attributed");
    for (ms, count, symbol) in &rows {
        println!(
            "  {ms:8.2} ms  {:5.1}%  x{count:<4} {symbol}",
            ms / total * 100.0
        );
    }
    println!(
        "  {:8.2} ms  {:5.1}%        (unattributed: barriers, ramp, encode)",
        total - attributed,
        (total - attributed) / total * 100.0
    );
}

/// What a DECODE costs, at the context lengths a served request lives at.
///
/// Every throughput gate in this file fires one long prompt, which is the half
/// of serving that saturates the machine. A decode saturates nothing: it reads
/// the whole weight set to produce one row, so it is bandwidth against a fixed
/// per-fire host cost, and neither of those shows up in a prefill's profile.
///
/// Measured the way a server pays for it -- `submit` on a reused `Stepper`,
/// through a reused `Lowerings` and a reused `Scratch`, with the lowering
/// lookup and the table staging inside the timed region because a served step
/// does both.
///
/// # What mlx-lm costs for the same thing
///
/// **256 tok/s at 128 of context and 238 at 1024**, on this machine and this
/// checkpoint, measured to mean the same thing this does: a `make_prompt_cache`
/// filled to the stated length, then 64 single-token steps of
/// `model(y[None], cache=cache)` + `argmax` + `mx.eval`, timed as a whole.
///
/// That is not the number `mlx_lm.generate` prints. Its `generation_tps` is
/// **310 and 270**, and the difference is not kernels: `stream_generate` runs
/// `mx.async_eval` and encodes the next step while the current one is on the
/// GPU, so its host cost is hidden behind its device cost. Ours is not --
/// `what_a_decode_costs_at_length` prints the host share, and subtracting it
/// lands almost exactly on mlx-lm's pipelined figure. The gap named there is a
/// serving-loop one, and closing it means encoding step N+1 before step N's
/// sampled token has come back to the host, which means the token ids table
/// filled on the DEVICE. Nothing here does that yet.
///
/// Comparing against 310 would flatter mlx-lm's kernels with a scheduling
/// trick, and comparing against `generate`'s wall clock would flatter ours
/// with mlx-lm's detokenizer. Both are avoided by timing the same loop.
///
/// Where this has been, on Llama-3.2-1B-Instruct-4bit over an M1 Max:
///
///   |                              | @128 | @1024 |
///   | fresh tables allocation      |  184 |   169 |
///   | tables leased from `Scratch` |  203 |   170 |
///   | lowering cached              |  218 |   194 |
///   | barriers made conditional    |  244 |   207 |
///   | attention walks pages        | 256-270 |  235 |
///   | *mlx-lm, same loop*          |  256 |   238 |
///
/// The short-context figure moves run to run because a 3.7 ms step with a
/// 0.5 ms host share is close enough to the scheduler's noise to see it; the
/// `best` this prints is steady at 3.68 ms.
///
/// `PIE_METAL_DECODE_REPLAY=1` runs the same decodes with a `Recordings` on
/// the side, and what it measures does not agree with the claim the path was
/// built on. `.wiki/driver/graph-metal.md` §5② has encoding at **76.4 % of a
/// decode**; here the whole host side is 1.1-1.3 ms of a 4.9 ms step, and
/// replaying removes 0.18 ms of it while the wall gets *worse*:
///
///   at 128 of context | encoded | replayed
///   wall              | 4.93 ms | 5.24 ms
///   host              | 1.30 ms | 1.12 ms
///
/// So a Metal 4 `execute_commands` is not free on the device side, and this
/// model's 250-dispatch decode is not encode-bound. The number that motivated
/// recording came off a 424-dispatch `llama_like` decode in the C++ shell,
/// which is a different fire. Recording stays -- it is right for a bigger
/// graph and it is what keeps the address discipline honest -- but nothing
/// here should be read as it paying for itself yet.
///
/// It also only records DECODES: recording the 128-token prefill that builds
/// the context faults the queue outright (`MTL4CommandQueueErrorDomain error
/// 1`, on the first fire), which is a defect this test names and does not
/// chase.
#[test]
#[ignore = "needs a checkpoint and a device"]
fn what_a_decode_costs_at_length() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let replay = std::env::var("PIE_METAL_DECODE_REPLAY").is_ok_and(|v| !v.is_empty());
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());
    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);
    let geometry = dispatch_geometry(&dg, &binding);
    let freqs = driver_metal::model::rope::table(&dg);
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    let named = HashMap::new();

    const STEPS: u32 = 32;
    for ctx in [128_u32, 1024] {
        let shape = pool_shape(&dg, (ctx + STEPS).div_ceil(16) + 4);
        let pool = Pool::allocate(&context, shape).expect("a pool");
        let pages = |layer: u16, values: bool| {
            pool.layer(u32::from(layer)).map(|l| Slice {
                address: if values {
                    l.v.gpu_address()
                } else {
                    l.k.gpu_address()
                },
                bytes: shape.layer_bytes_at(0),
            })
        };
        let mut regions = driver_metal::device::Regions::new();
        for region in &loaded.regions {
            regions.add(region);
        }
        for l in 0..shape.layers {
            if let Some(layer) = pool.layer(l) {
                layer.k.register(&mut regions);
                layer.v.register(&mut regions);
            }
        }
        let mut recordings = driver_metal::fire::Recordings::new();
        let mut lowerings = driver_metal::lowering::cached::Lowerings::new();
        let scratch = driver_metal::fire::Scratch::new();
        let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");

        // One fire, from the position it starts at and how many rows it
        // carries. The context below and every decode after take this road.
        let fire = |first: u32,
                    count: u32,
                    lowerings: &mut driver_metal::lowering::cached::Lowerings,
                    pipelines: &mut Pipelines,
                    regions: &mut driver_metal::device::Regions,
                    recordings: &mut driver_metal::fire::Recordings,
                    stepper: &mut driver_metal::device::Stepper|
         -> (f64, f64) {
            let started = std::time::Instant::now();
            let tokens: Vec<u32> = (0..count).map(|i| 1000 + (first + i) % 5000).collect();
            let positions: Vec<u32> = (first..first + count).collect();
            let step = Step {
                token_ids: &tokens,
                qo_indptr: &[0, count],
                sampling_indices: &[count - 1],
                sampling_indptr: &[0, 1],
                ..Step::default()
            };
            let class = if count > 1 {
                FireClass::Prefill
            } else {
                FireClass::Decode
            };
            // Through the CACHE, because that is what serves: `serve::launch`
            // holds a `Lowerings` and a decode's graph is a constant of the
            // deployment. Timed inside the region anyway -- a hit is not free,
            // it is `rows_of` plus a hash.
            let lowered = lowerings
                .for_step(class, &step, || {
                    Ok::<_, std::convert::Infallible>(text(row, class, &binding))
                })
                .expect("the step lowers");
            let held = (first + count).div_ceil(shape.page_size);
            let page_list: Vec<u32> = (0..held).collect();
            let zeros: Vec<u32> = vec![0; count as usize];
            let w_page: Vec<u32> = positions.iter().map(|p| p / shape.page_size).collect();
            let w_off: Vec<u32> = positions.iter().map(|p| p % shape.page_size).collect();
            let staged = driver_metal::bind::tables::stage(
                &context,
                &scratch,
                driver_metal::bind::tables::Frame {
                    token_ids: &tokens,
                    position_ids: &positions,
                    req_of_token: &zeros,
                    kv_page_indices: &page_list,
                    kv_page_indptr: &[0, held],
                    kv_write_page: &w_page,
                    kv_write_offset: &w_off,
                    rope_frequencies: &inv_freq,
                    sampling_indices: &[count - 1],
                },
            )
            .expect("the tables stage");
            regions.add(staged.region());
            regions.set_null(staged.region());
            let mut live = Live {
                store: Store::new(Names::mlx(), &loaded.tensors, &named),
                tables: &staged,
                shape,
                pages: &pages,
            };
            let mut machine = driver_metal::fire::run::Machine {
                context: &context,
                compiler: &compiler,
                pipelines,
                stepper,
                scratch: &scratch,
                regions,
                recordings: (replay && count == 1).then_some(recordings),
            };
            let submitted =
                driver_metal::fire::run::submit(&mut machine, &lowered, geometry, &mut live)
                    .expect("the fire commits");
            let encoded = started.elapsed().as_secs_f64() * 1e3;
            machine
                .stepper
                .wait_for(submitted.value)
                .expect("the fire retires");
            (started.elapsed().as_secs_f64() * 1e3, encoded)
        };

        // The context, then the decodes that read it. The first decode is
        // discarded: it is the one that compiles the decode pipelines and, in
        // replay, records them, and a server pays that once per shape.
        fire(
            0,
            ctx,
            &mut lowerings,
            &mut pipelines,
            &mut regions,
            &mut recordings,
            &mut stepper,
        );
        fire(
            ctx,
            1,
            &mut lowerings,
            &mut pipelines,
            &mut regions,
            &mut recordings,
            &mut stepper,
        );
        let (mut walls, mut encodes) = (Vec::new(), Vec::new());
        for i in 1..STEPS {
            let (wall, encode) = fire(
                ctx + i,
                1,
                &mut lowerings,
                &mut pipelines,
                &mut regions,
                &mut recordings,
                &mut stepper,
            );
            walls.push(wall);
            encodes.push(encode);
        }
        walls.sort_by(f64::total_cmp);
        encodes.sort_by(f64::total_cmp);
        let median = walls[walls.len() / 2];
        println!(
            "decode at {ctx} of context: {median:.2} ms a token ({:.0} tok/s), \
             host {:.2} ms of it, best {:.2} worst {:.2}, {} recordings",
            1e3 / median,
            encodes[encodes.len() / 2],
            walls[0],
            walls[walls.len() - 1],
            recordings.recorded(),
        );
    }
}

/// Where a DECODE spends its time, per symbol, at two context lengths.
///
/// `where_a_long_prefill_spends_its_time` answers this for the other half of
/// serving, and the two profiles have almost nothing in common: a prefill is
/// the matrix unit against 2048 rows, a decode is the memory system against
/// one. What made this worth writing is that `what_a_decode_costs_at_length`
/// measured the gap to mlx-lm GROWING with context -- 0.95 ms more per token
/// between 128 and 1024, against mlx-lm's 0.32 -- which says the cost is in
/// whatever reads the context, not in the projections that do not.
///
/// Each symbol is timed as the whole SET of its launches (all sixteen layers'
/// worth), repeated, so what prints is that symbol's contribution to one
/// decode rather than one dispatch of it. The repeat is not optional: a lone
/// statement in a fire pays a submission and a first touch that a statement
/// in the middle of 250 does not, and that constant is larger than most of
/// the rows below.
#[test]
#[ignore = "needs a checkpoint and a device"]
fn where_a_decode_spends_its_time() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());
    let Some((row, dg, loaded)) = served(&context, &snapshot) else {
        return;
    };
    let binding = observed(&dg, &loaded);
    let geometry = dispatch_geometry(&dg, &binding);
    let freqs = driver_metal::model::rope::table(&dg);
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    let named = HashMap::new();
    let scratch = driver_metal::fire::Scratch::new();
    const ROUNDS: usize = 7;

    for ctx in [128_u32, 1024] {
        let shape = pool_shape(&dg, ctx.div_ceil(16) + 4);
        let pool = Pool::allocate(&context, shape).expect("a pool");
        let held = (ctx + 1).div_ceil(shape.page_size);
        let page_list: Vec<u32> = (0..held).collect();
        let staged = driver_metal::bind::tables::stage(
            &context,
            &scratch,
            driver_metal::bind::tables::Frame {
                token_ids: &[1234],
                position_ids: &[ctx],
                req_of_token: &[0],
                kv_page_indices: &page_list,
                kv_page_indptr: &[0, held],
                kv_write_page: &[ctx / shape.page_size],
                kv_write_offset: &[ctx % shape.page_size],
                rope_frequencies: &inv_freq,
                sampling_indices: &[0],
            },
        )
        .expect("the tables stage");
        let step = Step {
            token_ids: &[1234],
            qo_indptr: &[0, 1],
            sampling_indices: &[0],
            sampling_indptr: &[0, 1],
            ..Step::default()
        };
        let plan = text(row, FireClass::Decode, &binding);
        let whole = lower_step(&plan, &step).expect("the step lowers");

        let mut best = |lowered: &model_compiler::lower::Lowered| -> (f64, f64) {
            let (mut ms, mut enc) = (f64::MAX, f64::MAX);
            for _ in 0..ROUNDS {
                let pages = |layer: u16, values: bool| {
                    pool.layer(u32::from(layer)).map(|l| Slice {
                        address: if values {
                            l.v.gpu_address()
                        } else {
                            l.k.gpu_address()
                        },
                        bytes: shape.layer_bytes_at(0),
                    })
                };
                let mut live = Live {
                    store: Store::new(Names::mlx(), &loaded.tensors, &named),
                    tables: &staged,
                    shape,
                    pages: &pages,
                };
                let started = std::time::Instant::now();
                let (timing, _arena) = driver_metal::fire::run::run_keeping_arena(
                    &context,
                    &compiler,
                    &mut pipelines,
                    lowered,
                    geometry,
                    &mut live,
                )
                .expect("the decode runs");
                ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
                enc = enc.min(timing.encode.as_secs_f64() * 1e3);
            }
            (ms, enc)
        };

        let (total, encode) = best(&whole);
        println!(
            "\ndecode at {ctx} of context: {total:.2} ms wall, {encode:.2} ms host encode, \
             {} launches over {} symbols",
            whole.launches.len(),
            whole.kernels.len(),
        );
        let mut rows: Vec<(f64, usize, String)> = Vec::new();
        for (index, symbol) in whole.kernels.iter().enumerate() {
            let mut only = whole.clone();
            only.launches.retain(|l| usize::from(l.kernel) == index);
            let count = only.launches.len();
            if count == 0 {
                continue;
            }
            // THE SAME SET, EIGHT TIMES. `(t8 - t1) / 7` is what one decode
            // pays for this symbol; `t1 - marginal` is the per-fire constant
            // it was measured through, and printing it is what keeps the
            // first column from being read as the second.
            let mut many = whole.clone();
            many.launches = std::iter::repeat_n(only.launches.clone(), 8)
                .flatten()
                .collect();
            let (one, _) = best(&only);
            let (eight, _) = best(&many);
            let marginal = (eight - one) / 7.0;
            rows.push((
                marginal,
                count,
                format!("{symbol}  x{count}  {marginal:6.3} ms  (alone {one:5.2})"),
            ));
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        let sum: f64 = rows.iter().map(|r| r.0).sum();
        for (_, _, line) in &rows {
            println!("  {line}");
        }
        println!("  ---- {sum:.2} ms of marginal against a {total:.2} ms fire");
    }
}

/// How many of a launch's widthed operands the kernel WRITES.
///
/// The trace states inputs, then outputs, then weights, so this is the length
/// of the tail that is a result -- which is the split
/// `arena_regions`'s writers and the rectangle census below both need.
///
/// It read the row's `operands` column and counted `Source::Out(i)`. Every
/// Metal family has retired its rows, and the answer moved to where it was
/// always derivable: a routine spells a written buffer `BufMut` or `F32sMut`
/// and a read one `Buf`, `I32s`, `U32s`, `U8s`, `F32s`, so the WRITE COUNT is
/// the count of mutable arguments in the signature. `driver-metal`'s
/// `directed` reads the same fact for `Touches`, which is the encoder's
/// hazard analysis -- so an answer that drifted here would have shown up
/// there as a missing barrier first.
///
/// A symbol nothing resolves falls back to ONE, unchanged: an unknown
/// statement that wrote nothing is far more likely to be one this could not
/// name than one that truly has no result, and attributing zero writers to it
/// makes the region it filled look unwritten.
fn results_of(symbol: &str) -> usize {
    let Some((routine, _)) = driver_metal::lowering::routine::crossed(symbol) else {
        return 1;
    };
    routine
        .args
        .iter()
        .filter(|(ty, _)| matches!(ty, kernels::Ty::BufMut | kernels::Ty::F32sMut))
        .count()
}
