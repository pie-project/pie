//! One fire after another, over the same cache.
//!
//! Everything below this is per-fire; what lives here is what can only be
//! wrong ACROSS fires: a conversation's pages must be its own from its first
//! fire to its last ([`Book`] answers that), its positions must continue
//! rather than restart, and a deployment needs BOTH plans — a text traced at
//! `FireClass::Prefill` states tiled GEMMs where `FireClass::Decode` states
//! matrix-vector products. Pipelines outlive the per-fire lowering because
//! `wgpu::Device::create_shader_module` runs a whole shader compiler. A step
//! does not sample, and does not keep the arena.

use model_compiler::lower::{Fire as LowerFire, Uncovered};
use model_ir::trace::ForwardPlan;

use crate::device::{Device, Pipelines};
use crate::dispatch::Geometry;
use crate::pages::{Book, Unhoused};
use crate::resources::{Frame, Model, Pool, Request, Unstageable, Weights};
use crate::serve::{Fire, Fired, Logits, Modules, Unfired, Unread, fire, logits};
use kernels_wgpu::Capability;

/// What one conversation wants out of one fire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    /// The conversation, by the id its [`Book`] seat is under.
    pub who: u64,
    /// The tokens to append, in order.
    ///
    /// One for a decode, many for a prefill; both produce a lowering from the
    /// same code path, which is what makes a mixed batch expressible. A turn
    /// of four tokens produces FOUR distributions and only the last has seen
    /// the whole prompt — [`Step::readout_of`] says which one that is.
    pub tokens: Vec<u32>,
}

/// What a deployment keeps between fires.
///
/// One struct rather than three arguments, so that a caller cannot pass a book
/// from one deployment and a pool from another. The pipeline cache is
/// deliberately NOT here: it belongs to the device, and two deployments on one
/// device should share it.
pub struct Held<'a> {
    /// Who owns which page.
    pub book: &'a mut Book,
    /// The cache and the per-fire tables.
    pub pool: &'a mut Pool,
    /// The gated DeltaNet's carry, for a deployment that opened one.
    ///
    /// Travels with the pool and the book because it belongs to the same thing
    /// they do — one deployment's per-layer state.
    pub recurrent: Option<&'a crate::resources::RecurrentPool>,
    /// The checkpoint.
    pub weights: &'a Weights,
    /// Lowerings already derived, by fire shape.
    ///
    /// Held beside the pool and the book because a lowering is the graph of
    /// ONE text: a cache from another deployment would run the other model's
    /// graph over these weights.
    pub lowerings: &'a mut crate::lowering::cached::Lowerings,
}

/// Why a step did not run.
///
/// Every layer's refusal, kept apart rather than flattened into one string: the
/// caller's next move differs completely between "the cache is full" (evict, or
/// queue this conversation) and "this adapter cannot build that kernel" (a
/// hardware fact, and no amount of waiting fixes it).
#[derive(Debug)]
pub enum Unstepped {
    /// The cache had no room for a conversation's growth.
    Unhoused(Unhoused),
    /// The turns did not make a stageable fire.
    Unstageable(Unstageable),
    /// The plan does not cover these rows.
    Uncovered(Uncovered),
    /// The device refused.
    Failed(crate::device::Failed),
    /// The fire did not run.
    Unfired(Unfired),
    /// The fire ran and its answer could not be read.
    Unread(Unread),
    /// A step of no turns.
    ///
    /// Refused rather than answered with an empty [`Logits`], because a lowering
    /// over zero rows is not a thing this crate has ever produced and a server
    /// asking for one has lost track of its own queue.
    Nothing,
    /// A request named a recurrent slot past the end of the pool.
    NoSlot {
        /// The slot the frame named.
        slot: u32,
        /// How many the pool holds.
        slots: u32,
    },
}

impl std::fmt::Display for Unstepped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unhoused(e) => write!(f, "{e}"),
            Self::Unstageable(e) => write!(f, "{e}"),
            Self::Uncovered(e) => write!(f, "{e:?}"),
            Self::Failed(e) => write!(f, "{e}"),
            Self::Unfired(e) => write!(f, "{e}"),
            Self::Unread(e) => write!(f, "{e}"),
            Self::Nothing => write!(f, "a step of no turns"),
            Self::NoSlot { slot, slots } => write!(
                f,
                "a request's carry is in recurrent slot {slot} and the pool holds {slots}"
            ),
        }
    }
}

impl std::error::Error for Unstepped {}

/// What one step did.
#[derive(Debug)]
pub struct Step {
    /// One distribution per row of the fire, in the order the rows were fired.
    ///
    /// **Not the order the turns were given.** A frame reorders rows by its
    /// seriation, so [`Step::readout_of`] is what turns a turn index back into a
    /// row of this.
    pub logits: Logits,
    /// The rectangles this step ran and how many submissions it took.
    pub fired: Fired,
    /// The rows this step fired over.
    pub rows: usize,
    /// Which row of [`Step::logits`] is each turn's answer.
    ///
    /// Every row samples — see [`Serving::step`] — so a prefill of four tokens
    /// produces four distributions, and the frame reorders rows, so the row
    /// order is not the turn order even for one-token turns.
    /// `logits.row(step.readout_of[i])` is turn `i`'s distribution.
    ///
    /// A turn naming SEVERAL readout rows has all of them in
    /// [`Step::readouts_of`]; this stays its LAST.
    pub readout_of: Vec<usize>,
    /// Every row turn `i` reads out, in fire-row order.
    ///
    /// One entry per turn, each the whole span rather than a single row: a
    /// decode names one, a speculative verifier names one per drafted token.
    /// Built from the frame's own `sampling_indptr`, so it is the placement
    /// after seriation and not the plan's numbering.
    pub readouts_of: Vec<Vec<usize>>,
    /// The position of each row, in the order the rows were fired.
    ///
    /// Here so a caller can read the pool's `Positions` table back and check
    /// that the tables the DEVICE holds are this step's. Without it the only
    /// evidence that a step restages is that it calls `stage`, and a control
    /// that staged only on a conversation's first step did not fire.
    pub positions: Vec<u32>,
    /// The whole ARENA this fire computed in, when one was asked for.
    ///
    /// Empty unless [`Serving::keep_arena`] is set, because a fire's arena is
    /// megabytes and a readback is a device stall.
    ///
    /// # Why a driver exposes this at all
    ///
    /// A fire's activations are unobservable from outside it. `fire_over`
    /// allocates the arena, fires, reads the readout's own range and drops it
    /// inside one function, so a caller that wants to know WHERE a computation
    /// first went wrong has only the last value in the chain to look at — and
    /// "the logits are NaN" is the same sentence for a bad embedding and a bad
    /// lm head.
    ///
    /// That is not hypothetical. `tests/hybrid_probe.rs` spent a long
    /// investigation bisecting an odd-row NaN by ZEROING WEIGHTS, one
    /// subsystem at a time, because switching a branch off and re-firing was
    /// the only instrument available. It got as far as one MLP block and
    /// stopped there, and the thing it could not do was read the value between
    /// two kernels.
    ///
    /// `Lowered::value_offset` says where every traced value lives, so with
    /// the bytes beside it a caller can name the first one that is not finite.
    /// That is the difference between "somewhere in these five kernels" and a
    /// kernel.
    pub arena: Vec<u8>,
    /// Pipelines the cache held after this step.
    ///
    /// Here so that "the second step builds nothing" is a number a test can
    /// compare rather than a claim about timing.
    pub pipelines: usize,
}

/// A model on a device, fired repeatedly.
///
/// Holds the things that outlive a fire and borrows nothing that does not: the
/// plan, the geometry and the tier are the deployment, and the book, pool,
/// weights and pipeline cache are passed to each step because they are mutated
/// by it and a server may well own them elsewhere.
#[derive(Clone, Copy)]
pub struct Serving<'a> {
    /// The text a one-row step lowers.
    pub plan: &'a ForwardPlan,
    /// The text a step of more than one row lowers.
    ///
    /// **Two plans and not one, because the class is not cosmetic.** A text
    /// traced at `FireClass::Decode` states `affine_qmv_fast` for its
    /// projections; traced at `FireClass::Prefill` it states the tiled GEMM.
    /// A deployment carrying only the decode plan answers a sixty-four-token
    /// prompt with sixty-four matrix-vector products: correct, and the wrong
    /// kernel.
    ///
    /// Not an `Option` with a fallback, which would make the slow path the
    /// quiet default. Tracing a plan is host-side and cheap.
    ///
    /// The PLAN switches at two rows; the KERNEL does not. `model`'s
    /// `TokensMultipleOf(tile)` guard takes the GEMM only when the tile
    /// DIVIDES the row count, so a two-row step lowers to the matvec out of
    /// the prefill text. The GEMM costs the same for one row as for
    /// thirty-two — it computes a whole tile either way.
    ///
    /// **The crossover is between 16 and 32 rows, not near 150.** The 150
    /// came from `how_long_a_decodes_kernels_take`'s switch sweep while it
    /// was gridding the matvec by neither of the launcher's two rules — `m`
    /// row groups instead of `ceil(m / PIE_MT)`, over half the output
    /// columns. Re-measured with `quant::qmv_grid`'s own rule, a 1024x1024
    /// affine plane at gs 64 / 4 bits:
    ///
    /// ```text
    ///   m     qmv       qmm_t     tiled / matvec
    ///   1     0.024 ms  0.078 ms  3.29
    ///   2     0.026     0.077     2.93
    ///   4     0.026     0.073     2.75
    ///   8     0.040     0.078     1.95
    ///   16    0.063     0.077     1.22
    ///   32    0.110     0.078     0.71
    ///   128   0.385     0.185     0.48
    ///   512   1.497     0.664     0.44
    /// ```
    ///
    /// Which leaves the guard well placed by luck AT THE BOTTOM: `bm = 32` is
    /// about where the GEMM starts winning, so every SMALL batch it refuses is
    /// a batch the matvec should have had anyway.
    ///
    /// # AND BADLY PLACED EVERYWHERE ELSE -- THE PREFILL CLIFF
    ///
    /// That sentence used to stop at "anyway", and read as a defence of the
    /// guard. It is only a statement about counts BELOW the tile.
    /// `TokensMultipleOf` refuses every count the tile does not divide, which
    /// is thirty-one lengths in thirty-two AT EVERY SIZE, and a 500-token
    /// prompt is not a batch the matvec should have had.
    ///
    /// Measured on this deployment, llama-3.2-1B q4 on an M4 Pro, `pp` at the
    /// stated prompt length:
    ///
    /// ```text
    ///   256   1256.8 tok/s      496    529.2 tok/s
    ///   480   1236.9            500    528.5
    ///   512   1238.4            504    528.5
    ///   1024  1112.3            511    527.2
    /// ```
    ///
    /// A hard cliff on `n % 32`, worth 2.34x, and it is the whole fire and not
    /// a tail chunk: the guest chunks a prompt to `max_embed_length` (4096
    /// here), so 500 tokens arrive as ONE fire of 500 rows and every
    /// projection in it takes the matvec arm. The isolated bench says the same
    /// ratio from the other side -- at m = 512 on a 1024x1024 plane, qmv is
    /// 1.637 ms against the GEMM's 0.664, or 2.47x -- which is close enough to
    /// the 2.34x measured here to identify the path with no profiler.
    ///
    /// ## What fixing it needs, which is why this is a comment and not a patch
    ///
    /// Three things, and the first two are worthless without the third:
    ///
    /// 1. A fact for "this backend's GEMM tolerates a row count its tile does
    ///    not divide", so the text can state `TokensGT(tile - 1)` where the
    ///    backend has said so. It cannot be unconditional: `kernels-metal` and
    ///    `kernels-vulkan` have the same contract and no such tolerance.
    /// 2. `Rule::Qmm` rounding the row axis up rather than refusing with
    ///    `Ungeometric::PartialTile`.
    /// 3. THE KERNEL ACTUALLY TOLERATING IT. This is the work. `qmm_t.wgsl`
    ///    has no `M` argument at all -- `Params` is variant-shaped and carries
    ///    `k` and `n` -- so `write_out` cannot bound its rows, and rounding
    ///    the grid up today does what `forward/mod.rs` says it was measured
    ///    doing: "a finite wrong answer plus a tile's worth of overrun into
    ///    the next value".
    ///
    /// Doing 1 and 2 without 3 is not a slow model, it is a wrong one, which
    /// is why the order matters more than the size.
    ///
    /// ## 1 AND 2 WERE TRIED. THEY ARE WRONG *AND* SLOWER -- DO NOT REPEAT
    ///
    /// Measured, not reasoned: `geometry::eval`'s `PartialTile` refusal was
    /// disabled and all three `GuardPred::TokensMultipleOf(tile)` in
    /// `llama_like/forward/mod.rs` were relaxed to `TokensGT(tile - 1)`, so
    /// the GEMM ran with its grid rounded up and its stores unbounded.
    ///
    /// WRONG, as the contract says. Sixteen greedy tokens after a 496-token
    /// prompt of one repeated id:
    ///
    /// ```text
    ///   refusing (matvec)  [1109 x11, 323, 315, 323, 315, 323]
    ///   rounded up (GEMM)  [92652, 11, 755, 11, 11, 198, 52, 11, ...]
    /// ```
    ///
    /// The matvec continues the repetition, which is the answer; the rounded
    /// GEMM is unrelated text, not a near miss.
    ///
    /// AND SLOWER, which was NOT predicted and is the more useful half:
    ///
    /// ```text
    ///          refusing   rounded up
    ///   pp512    1239.4      1239.4     (multiple: same path, control)
    ///   pp480    1241.1      1241.1     (multiple: same path, control)
    ///   pp500     528.5       136.1     3.9x SLOWER
    ///   pp496     529.2        97.1     5.4x SLOWER
    /// ```
    ///
    /// So the partial-tile GEMM is not a fast answer waiting behind a safety
    /// check. At 496 rows it ran at about a TWELFTH of the 1240 tok/s the
    /// same kernel reaches at 512 -- far past what sixteen wasted rows in
    /// five hundred can explain, and the wasted fraction cannot explain 496
    /// being slower than 500 either.
    ///
    /// ### The denormal hypothesis was wrong, and that is good news
    ///
    /// The guess was that the overhang rows stage whatever lies past the
    /// activation, so the FMA pipeline runs on denormals and NaNs, which
    /// Apple's GPU does not do at rate -- and therefore that a partial-tile
    /// GEMM would need the STAGING loop zeroed and not merely its stores
    /// bounded.
    ///
    /// `kernels-wgpu`'s `how_long_a_decodes_kernels_take` tests exactly that:
    /// one launch, `m = 496` padded to 512, `(bm, bn) = (32, 64)`, `n = k =
    /// 2048`, with ONLY the content of the sixteen overhang rows changed.
    ///
    /// ```text
    ///   tail=zeros      1.690 ms      tail=NaN        1.691 ms
    ///   tail=ones       1.692 ms      tail=infinity   1.693 ms
    ///   tail=denormal   1.690 ms
    /// ```
    ///
    /// Three thousandths of a millisecond across the whole range, at 2.54
    /// TFLOP/s -- the same rate this tile reaches on aligned shapes. The
    /// overhang costs its arithmetic and NOTHING else, whatever is in it.
    ///
    /// So the twelvefold end-to-end slowdown is NOT the GEMM. It is a
    /// downstream consequence of the CORRUPTION: the unbounded stores run a
    /// tile's worth of rows into the next value in the arena, and something
    /// that reads that value afterwards is what got slow. Which means the
    /// slowness and the wrongness are one bug and not two, and bounding the
    /// stores is expected to fix both.
    ///
    /// That flips the recommendation. The fix is the three parts above --
    /// there is no fourth -- and part 3 is worth doing.
    ///
    /// ### PART 3 IS DONE
    ///
    /// `Params` now ends with `m`, passed at all nineteen `qmm_t.wgsl` fire
    /// sites in `kernels-wgpu::quant` (each already computed it via
    /// `ctx.ask::<i32, keys::Rows>()` for the grid and simply dropped it),
    /// and `write_out` returns on `row >= params.m`. No grid arithmetic
    /// changed: `qmm_grid` always rounded up with `div_ceil`.
    ///
    /// It is proved, not asserted.
    /// `kernels-wgpu`'s `a_tiled_gemm_agrees_over_every_tile_shape_and_
    /// quantization_point` fires m = 33 over nine tiles and six codecs and
    /// now checks the overhang still holds its sentinel. Remove the `row`
    /// term and it reports 705 of 705 overhang values written, so the check
    /// measures the guard and not the weather.
    ///
    /// And with parts 1 and 2 stubbed out by hand on top of it, the cliff
    /// goes away:
    ///
    /// ```text
    ///            refusing   unbounded GEMM   bounded GEMM
    ///   pp512      1238.1           1239.4         1238.1
    ///   pp496       529.2             97.1         1187.2
    /// ```
    ///
    /// 2.24x, and the twelvefold slowdown was indeed the corruption.
    ///
    /// ### ALL THREE PARTS ARE NOW IN, AND THE CLIFF IS GONE
    ///
    /// `MetalBinding::qmm_partial_rows` is the fact, `true` at this
    /// backend's seam alone; the projections' guard reads
    /// `GuardPred::TokensGT(tile - 1)` where it is set; and `Rule::Qmm`
    /// above no longer refuses. Measured on the shipped build:
    ///
    /// ```text
    ///   pp480   1237.5        pp496   1208.6   (was 529.2)
    ///   pp512   1232.9        pp500   1217.4   (was 528.5)
    ///   pp1024  1099.6        pp495   1213.1
    /// ```
    ///
    /// Flat. `pp2048` is 894.7 and `tg128` 113.0, both unmoved.
    ///
    /// ### WHAT THE CORRECTNESS EVIDENCE ACTUALLY IS
    ///
    /// In the kernel, it is exact: `a_tiled_gemm_agrees_over_every_tile_
    /// shape_and_quantization_point` fires m = 33 across nine tiles and six
    /// codecs against a host sum, and now also proves the overhang keeps its
    /// sentinel.
    ///
    /// End to end it is a CONTROL and not a match, and the distinction is
    /// worth writing down because the obvious reading of the raw numbers is
    /// that this change broke the model. A live probe at 496 tokens answers
    /// differently under the GEMM than under the matvec. That looks damning
    /// until the same comparison is run at 512, where BOTH paths are legal:
    ///
    /// ```text
    ///   512  matvec  [220, 16, 13, 15, 13, 15, ...]
    ///   512  GEMM    [5113, 64, 11, 264, 198, 1494, ...]
    /// ```
    ///
    /// They disagree completely at a length with no partial tile in it. The
    /// two families agree only to about 5% of the row's peak -- the tolerance
    /// `the_tiled_gemm_answers_the_way_the_vector_kernel_does` asserts -- and
    /// a greedy argmax over a random-token prompt has no margin, so the
    /// families part on the first token whatever the row count is.
    ///
    /// Against that control the partial-tile numbers are what a CORRECT
    /// implementation gives: the GEMM answers at 495 and 512 are identical,
    /// 496 and 500 stay in the same token vocabulary, and the matvec answers
    /// at 496 and 512 are identical to each other. Each family is
    /// self-consistent across the modulus; only the families differ.
    ///
    /// `the_tiled_gemm_answers_the_way_the_vector_kernel_does_at_a_partial_
    /// tile` is the test that would settle it directly -- a real logit row at
    /// 495 tokens, one family against the other, under the 5% claim. It SKIPS
    /// on this machine for want of a bf16 checkpoint its loader can quantize.
    /// Run it where one exists; it is the last thing owed on this change. Note the matvec is FLAT to four rows —
    /// one workgroup carries `PIE_MT = 4` of them and reads the weights once
    /// — and rises with the row groups after.
    ///
    /// RETIRED WITH `kernels-wgpu`'s TEST TREE. That name is a record of a
    /// measurement now, not a live proof: the crate lost `tests/` and every
    /// in-file `mod tests` when the three shader planes moved their numbers to
    /// the fire that reads them, and nothing in this workspace re-runs it. What
    /// it reported is still why the sentence above says what it says; what is
    /// gone is the thing that would notice if it stopped being true.
    pub prefill: &'a ForwardPlan,
    /// The model's shape, for the launch rules that need it.
    pub geometry: Geometry,
    /// The tier every pipeline is built at.
    pub tier: Capability,
    /// Record only the first `n` rectangles of each fire.
    ///
    /// A DIAGNOSTIC and the third instrument this backend's numerical hunts
    /// have needed: it makes the arena's end state the state AT `n`, which is
    /// the only way to see a value between two kernels. See
    /// [`crate::serve::Fire::prefix`].
    pub prefix: Option<usize>,
    /// Submit every dispatch on its own command buffer, with a device wait
    /// between them, instead of recording them all into one pass.
    ///
    /// Ruinously slow and a DIAGNOSTIC. `crate::serve::Fire::one_at_a_time`
    /// has always existed and nothing could reach it, so the claim its own doc
    /// makes — that `wgpu` inserts the barriers and a disagreement between the
    /// two paths therefore cannot be a missing one — had never been tested
    /// against a real text.
    pub one_at_a_time: bool,
    /// Read the whole arena back after each fire and hand it to the caller on
    /// [`Step::arena`].
    ///
    /// Off by default and a DIAGNOSTIC: the arena is megabytes, the readback
    /// is a device stall, and a server has no use for it. See [`Step::arena`]
    /// for what it is for.
    pub keep_arena: bool,
}

impl Serving<'_> {
    /// Fire one step over `turns` and read what it computed.
    ///
    /// The order is the whole content of this function, and every pair of
    /// adjacent lines below is a defect somebody could introduce:
    ///
    /// - grow before framing, or the frame describes pages the book has not
    ///   handed out;
    /// - frame before lowering, because the lowering's row count, arena size
    ///   and readout all come from the frame's seriation;
    /// - stage before firing, or the fire reads the PREVIOUS step's tables;
    /// - read before the arena is dropped.
    ///
    /// # Errors
    ///
    /// [`Unstepped`]. **A refused growth leaves the book untouched.** A
    /// failure after the growth does NOT roll it back: the pages are the
    /// conversation's and the history is still on the device.
    pub fn step<M: Modules>(
        &self,
        device: &Device,
        pipelines: &mut Pipelines,
        modules: &M,
        held: &mut Held<'_>,
        turns: &[Turn],
    ) -> Result<Step, Unstepped> {
        if turns.is_empty() {
            return Err(Unstepped::Nothing);
        }
        // Grown in one pass so that a refusal partway leaves the earlier turns
        // seated -- see the note above about why that is not rolled back.
        let mut requests = Vec::with_capacity(turns.len());
        for turn in turns {
            requests.push(
                held.book
                    .grow(turn.who, turn.tokens.len())
                    .map_err(Unstepped::Unhoused)?,
            );
        }
        let tokens: Vec<&[u32]> = turns.iter().map(|t| t.tokens.as_slice()).collect();
        self.fire_over(device, pipelines, modules, held, &requests, &tokens)
    }

    /// One step over requests the CALLER allocated, touching no book.
    ///
    /// The engine's scheduler owns eviction, prefix sharing and the copy plans,
    /// and hands a driver a page CSR naming physical pages it chose.
    /// [`Serving::step`]'s allocator is the wrong one for that caller: two
    /// allocators handing out page 7 is not an error anybody sees, because
    /// attention then reads another conversation's keys and the model answers
    /// fluently. So this exists in order that [`crate::frames`] never touches
    /// [`crate::pages::Book`]; it is literally the same fire as `step`.
    ///
    /// `tokens` is parallel to `requests`: one slice of token ids per request.
    ///
    /// # Errors
    ///
    /// Every [`Unstepped`] except [`Unstepped::Unhoused`], which is the book's
    /// and this path has no book.
    pub fn over<M: Modules>(
        &self,
        device: &Device,
        pipelines: &mut Pipelines,
        modules: &M,
        held: &mut Held<'_>,
        requests: &[Request],
        tokens: &[&[u32]],
    ) -> Result<Step, Unstepped> {
        if requests.is_empty() {
            return Err(Unstepped::Nothing);
        }
        self.fire_over(device, pipelines, modules, held, requests, tokens)
    }

    /// The body both paths share.
    ///
    /// Private, and one function rather than two, because the failure the split
    /// would invite is precise: a fix to the sampling-index workaround below
    /// applied to one path and not the other would make an engine-served fire
    /// and a book-served fire read different rows of the same cache, and both
    /// would answer.
    fn fire_over<M: Modules>(
        &self,
        device: &Device,
        pipelines: &mut Pipelines,
        modules: &M,
        held: &mut Held<'_>,
        requests: &[Request],
        tokens: &[&[u32]],
    ) -> Result<Step, Unstepped> {
        let shape = held.pool.shape();
        let frame = Frame::of(shape, requests).map_err(Unstepped::Unstageable)?;
        // A SLOT THIS POOL DOES NOT HOLD, refused rather than dispatched.
        //
        // `wgpu` clamps an out-of-bounds storage read instead of trapping, so
        // a slot past the slab resolves to some other conversation's carry and
        // the fire answers fluently with it. That is the failure this whole
        // table was found by, one level down: the difference between a state
        // nobody wrote and a state somebody else did is invisible in the
        // output.
        if let Some(pool) = held.recurrent.as_ref()
            && let Some(&slot) = frame
                .recurrent_slots
                .iter()
                .find(|&&s| s >= pool.shape().slots)
        {
            return Err(Unstepped::NoSlot {
                slot,
                slots: pool.shape().slots,
            });
        }
        // THE ROWS THE FRAME SAYS SAMPLE, and no longer every row.
        //
        // This forced `samples: true` on all of them, for a reason that was
        // real: the texts spell their epilogue as plain `OpKind::Launch` ops
        // rather than `OpKind::LmHead`, so `Lowerer::epilogue` never runs and
        // the generic path emits the gather over `0..n_requests` while its
        // INPUT spans the whole token stream. `binding::extent` measured every
        // operand by the launch's rectangle, so that input bound one row, WGSL
        // clamped the rest to zero, and the head projected zeros.
        //
        // `Lowered::arg_rows` is the missing statement -- each operand's own
        // row count, beside the launch's -- and `binding::bind` widens a
        // binding to it. The gather reads the stream again, and the head runs
        // over the REQUESTS: a 512-token prefill's readout falls from 512
        // distributions to one, which is 316.9 ms of lm head and a 296.8 MB
        // readback that nobody wanted.
        //
        // What this costs instead is a remap. `Step::readout_of` and
        // `Step::readouts_of` named FIRE rows, and the logits are now one row
        // per SAMPLED row, so both are put through `frame.sampling_indices`
        // below.
        let rows = frame.seriation();
        // One row is a decode, more than one is a prefill -- including a BATCH
        // of one-token turns, since the row count is what the kernel choice
        // depends on.
        //
        // ONE binding, two uses, deliberately: `prefill` picks the text AND is
        // half the cache key, and a cache keyed on a different rule than the
        // one that chose the plan would serve one text's graph for the other.
        // # What a batched decode costs, measured
        //
        // This one-line decision is also the batching policy, and it was
        // never measured against a real batch until it was. Timed on an M4
        // Pro, Llama-3.2-1B at 4 bits, n concurrent conversations of a
        // one-token prompt so that every step below is a DECODE and not a
        // prefill chunk -- which is the distinction the first version of this
        // note got wrong, by histogramming steps without separating them:
        //
        // | conversations | launches | ms a step | ms a token |
        // |---|---|---|---|
        // | 1 | 244 | 7.52 | 7.52 |
        // | 2 | 244 | 12.99 | 6.50 |
        // | 7 | 228 | 22.12 | 3.16 |
        // | 8 | 228 | 23.39 | 2.92 |
        //
        // **THE SECOND CONVERSATION VERY NEARLY COSTS A WHOLE STEP**, 7.52 ms
        // to 12.99, a 1.16x improvement per token. A decode reads the whole
        // 4-bit weight set to serve one token and that read is the floor, so
        // a second row should ride it nearly free. The served consequence,
        // with a KV cache large enough that nothing queues: aggregate
        // throughput saturates near 230 tok/s however many conversations are
        // offered, against 117 for one.
        //
        // ## Where it goes, which is NOT this line
        //
        // The step's own phases say the fire: `low=0 stage=54us fire=12623us`.
        // And the two-row fire dispatches the SAME 244 launches at the SAME
        // grids as the one-row fire -- `affine_qmv_fast` at `[1, 2048, 1]`
        // either way -- because `quant/qmv.wgsl` carries `PIE_MT = 4`
        // activation rows inside one workgroup. Nothing got bigger. The work
        // inside each workgroup did.
        //
        // Two things happen at once when `mt` leaves one, both in
        // `reduce_store`:
        //
        // * The lane drops from `block_dot1` to `block_dot`, and only
        //   `block_dot1` has the `unpack4x8unorm` nibble path. Two rows pay
        //   three instructions a code where one row pays about one.
        // * `block_dot` computes all FOUR row slots regardless. `vecs` is
        //   clamped to the last live row, so a two-row fire multiplies row 1
        //   three times and throws two of them away.
        //
        // So a 2-row decode does 4 rows of arithmetic on the slow unpack, and
        // 1.73x of a one-row step is what that predicts.
        //
        // ## Which text an n-row fire actually takes
        //
        // Not what the line below reads like, and worth stating because it
        // was got wrong twice. `prefill` here selects the PLAN; the kernel is
        // then chosen inside it by `GuardPred::TokensMultipleOf(bm)`, which
        // admits the tiled GEMM only when the tile DIVIDES the row count. No
        // batch a scheduler gathers divides 32, so THE GEMM ARM NEVER RUNS IN
        // SERVING and every step here is matvec however many rows it carries.
        //
        // Dumped at 8 concurrent conversations, 228 launches:
        //
        // ```text
        //   32 x affine_qmv_fast          grid=[2, 128, 1]
        //   16 x affine_qmv_fast          grid=[2, 512, 1]
        //   32 x affine_qmv_fast          grid=[2, 2048, 1]
        //    1 x affine_qmv_fast          grid=[2, 32064, 1]
        //   32 x affine_qmv_fast_residual grid=[2, 512, 1]
        //   16 x kv_append_paged          grid=[1, 8, 8]
        //   16 x neox_freqs_mb            grid=[1, 8, 8]
        //   16 x neox_freqs_mb            grid=[1, 32, 8]
        //   33 x rms_single_row           grid=[8, 1, 1]
        //   16 x sdpa_paged_decode        grid=[32, 8, 1]
        //   16 x silu_mul                 grid=[256, 1, 1]
        // ```
        //
        // Every grid carries the rows: qmv in x, at `ceil(rows / PIE_MT)`;
        // rope and the KV append in z; the norm and `silu_mul` in x. No
        // `affine_qmm_t` appears at all.
        //
        // The `2` in the qmv grids above is `ceil(8 / PIE_MT)` at the
        // `PIE_MT = 4` this was dumped under. It reads `4` now that
        // `PIE_MT` is 2 -- the same rows over twice the workgroups, which
        // is the whole of that change. Nothing else in the table moves.
        //
        // 228 and not 244 because the SPLIT is off, not because the text
        // changed: `rows * q_heads` is 256 against `PIE_SPLIT_BELOW = 128`,
        // so the one-pass `sdpa_paged_decode` replaces the split/merge pair
        // and saves a launch a layer. A one-row fire states the pair.
        //
        // ## HOW TO DUMP THIS WITHOUT BEING LIED TO
        //
        // `probe::dump` is ONE-SHOT and fires on the first launch list at or
        // over `PIE_WGPU_DUMP`. The first such list is whatever the server
        // happened to gather, and the first reading of this table was a
        // ONE-ROW fire read as an eight-row one -- every qmv grid began `1`,
        // which is exactly what "the grid does not carry the rows" looks
        // like. Read the ROW COUNT off `neox_freqs_mb`'s z before believing
        // any of it, and if it is not the count you asked for, dump repeatedly
        // and take a later fire.
        //
        // ## What NOT to do about it
        //
        // Not "put the nibble path in `block_dot`": qmv.wgsl's own list of
        // things measured worse already has that entry, for the four-row form,
        // where the live registers cost more than the instructions save.
        //
        // ## What was done about it, and what is left
        //
        // The untried thing was an `mt == 2` arm with a two-row block dot --
        // half the live registers of the rejected experiment, and no wasted
        // slots. `qmv.wgsl` now has one (`block_dot2`), and it took the two-
        // stream aggregate from 133.7 to 168.6 tok/s, +26%, with one-stream
        // tg128 and pp512 unmoved. Its table lives there.
        //
        // The same discipline applied to `block_dot`'s four rows LOSES, at
        // every concurrency and on the one-row path too; that is measured and
        // recorded there as well. So rows 3 and 4 keep the four-slot form,
        // and the next launch-count lever is fusion, not this kernel.
        //
        // ## Which fusion is NOT available, and why it is not one commit
        //
        // Both dense projection joins are closed to this deployment, and it
        // is worth writing down which door each is behind so the next reader
        // does not open two of the three and conclude it is close.
        //
        // `q‖k‖v` is closed on PURPOSE and should stay closed. The text used
        // to branch on a `qkv_fused` fact and the fact cost a checkpoint:
        // once `driver-metal` stopped building `LlamaLikeFacts` itself, the
        // CATALOG row's answer reached the text, that row is CUDA's, it says
        // `true` on all eight llama-3 rows, and llama-3.2-1B died on
        // `Unbound { symbol: "affine_qmv_fast_bfloat16_gs_64_b_4", why:
        // UnknownWeight("layer.0.qkv") }`. `forward/mod.rs` states the three
        // projections unconditionally now and says all of that in place.
        //
        // `gate‖up` is closed by THREE independent gates, and a change that
        // clears fewer than all three gets a slower model, not a broken one,
        // which is the failure mode that wastes an afternoon:
        //
        //   1. `builder.rs::dense_fused_projection_joins` returns early on
        //      `Projections::InPlace`, and `compile_load_plan` authors every
        //      MLX load with it.
        //   2. `fused_join_candidate` takes a part only if
        //      `is_raw(&raw.encoding, DType::BF16)`. THIS one is the wall.
        //      An affine-u4 bank is a u32 weight plane with `.scales` and
        //      `.biases` siblings, so the candidate is `None` before the
        //      policy is ever consulted -- the join has no notion of
        //      concatenating three planes per part, and giving it one is the
        //      real work. (It is SOUND to concatenate them here, since both
        //      banks are row-sharded and share a group size, but sound is
        //      not written.)
        //   3. `forward/mod.rs` ASSERTS `!metal.gate_up_fused`: there is no
        //      packed arm in the Metal-side text, because `silu_mul` takes
        //      gate and up as two buffers and nothing splits a packed bank.
        //      A wgpu `silu_mul` reading one bank at an offset is the easy
        //      part of this.
        //
        // So it is a model-crate change (a quantization-aware join), a
        // shared-text change (the packed arm, currently asserted against),
        // and a kernel -- for 16 launches of 244. Worth doing, not worth
        // starting without the isolated bench wired up to confirm it, and
        // NOT reachable from inside `driver-wgpu` alone, which is what the
        // earlier "the next lever is fusion" line failed to say.
        //
        // # THE BENCH THIS ASKED FOR NOW EXISTS, AND HERE IS THE PRICE LIST
        //
        // `what_a_decode_costs_at_length` steps `Fire::prefix` across the
        // whole plan and reads a straight line: a 512-key qwen3-0.6b decode on
        // an Apple M4 Pro is 480 rectangles and 9.665 ms, against 0.276 ms for
        // a fire that records none of them. **19.6 microseconds a rectangle,
        // and the number does not vary with what the rectangle computes** --
        // every block of thirty costs 0.45 to 0.75 ms wherever it sits, and no
        // rectangle in a layer stands out. The same test's prefill sibling is
        // 984 microseconds a rectangle, which is what a rectangle doing work
        // looks like.
        //
        // So a fused pair keeps its arithmetic and saves only the FLOOR, which
        // the split-K note prices at ~13 us of that 19.6. At 28 layers:
        //
        // ```text
        //   fusion            rects/layer   removed    saved      of 9.665 ms
        //   q||k||v                     2        56   0.73 ms           7.5 %
        //   gate||up                    1        28   0.36 ms           3.8 %
        //   rope q + rope k             2        56   0.73 ms           7.5 %
        //   rmsnorm into its qmv        2        56   0.73 ms           7.5 %
        //   ------------------------------------------------------------------
        //   all four                    7       196   2.55 ms          26.4 %
        // ```
        //
        // # THE LIST IS MISSING A ROW, AND IT IS THE BIGGEST ONE
        //
        // `PIE_WGPU_DUMP` censuses the decode's 480 launches (the table is in
        // `what_a_decode_costs_at_length`) and `rms_single_row` is **113** of
        // them -- 23.5% of the token. The row above counts 56: the pre-attn
        // and pre-mlp norms. The other 57 are one final norm and, per layer,
        // the two PER-HEAD qk-norms this model has:
        //
        // ```text
        //   fusion                     rects/layer   removed    saved     of 9.83 ms
        //   q-norm + k-norm into rope            2        56   0.73 ms         7.5 %
        // ```
        //
        // # THAT ROW IS NOW LANDED, AND IT MEASURED 0.26 ms
        //
        // `kernels-wgpu`'s `norm::rms_rope` ships and the deployment states
        // `fused_qk_rope: true`. Ten interleaved runs put it at **0.26 ms,
        // 2.7%, ~104.3 -> ~107.2 tok/s** -- against the 0.34 ms the marginal
        // 6 us predicted and the 0.73 ms the 13 us above predicted. The
        // marginal price is the one that was right, which is the first
        // end-to-end evidence for any of these estimates and the reason the
        // three remaining rows should be believed at ~0.34 ms each rather
        // than ~0.73.
        //
        // The full table, including why the fused column is bimodal and why
        // two interleaved pairs would have reported no effect at all, is in
        // `what_a_decode_costs_at_length`.
        //
        // # AND THE q||k||v ROW IS WORTH MORE THAN A FLOOR, FOR A SECOND REASON
        //
        // Every row above prices a fusion as launches removed times a floor,
        // because the fused pair "keeps its arithmetic". For the three norm and
        // rope rows that is right. For `q||k||v` and `gate||up` it is not, and
        // `qmv.wgsl` now carries the measurement that says so.
        //
        // qmv's 141 decode fires read 218 MB in 3.20 ms -- **68 GB/s of this
        // part's 273**. Removing most of the nibble-unpack ALU buys 2.4% and
        // issuing every weight load twice costs 0.7%, so neither the
        // arithmetic nor the load count is what holds it at a quarter rate.
        // Drop the lm head and the remaining 140 fires are 20.8 us each for
        // 1.8 to 5.5 us of bytes over a 6 us floor: they are too SMALL to fill
        // the GPU, 256 workgroups of 32 invocations for twenty cores.
        //
        // Joining q, k and v does not just delete two launches. It turns three
        // dispatches of 512, 256 and 256 workgroups into one of 1024 -- it
        // fixes the thing the fires are actually short of. The 0.34 ms in the
        // table is therefore a FLOOR on that row, not an estimate of it, and
        // the same holds for `gate||up` (768 + 768 -> 1536). Neither the
        // per-head-norm row nor the rope row gets this bonus; they were
        // already one-workgroup fires where only the floor was ever at stake,
        // which is exactly why `rms_rope` landed at the marginal 0.26 ms and
        // not a penny more.
        //
        // And `qmv.wgsl` can put a number on the bonus, because it has held a
        // measured GB/s-against-dispatch-size curve since it was written and
        // nobody read it that way: 128 workgroups 45 GB/s, 512 -> 96, 2048 ->
        // 149, 32064 -> 186. Stepping q||k||v from the 256-512 band to 1024,
        // and gate||up from 768 to 1536, is worth roughly 1.4x on the 143 MB
        // those five fires read across 28 layers:
        //
        // ```text
        //   fusion       launches   floor      rate        together   of 9.5 ms
        //   q||k||v            56   0.34 ms    ~0.38 ms    ~0.7 ms        7.4 %
        //   gate||up           28   0.17 ms    ~0.30 ms    ~0.5 ms        5.0 %
        // ```
        //
        // That is the largest single item left in the decode, it is twice what
        // the floor-only table said, and both rows are blocked in the same
        // place -- `builder.rs::dense_fused_projection_joins` returning early,
        // for this backend at its very first line, because MLX checkpoints
        // bind under `Projections::InPlace`.
        //
        // # AND THE RATE TERM IS NOW MEASURED, NOT EXTRAPOLATED
        //
        // `qmv.wgsl` grew a `PIE_ROWREP` knob that makes each workgroup walk
        // more row-groups and divides the grid to match: identical codes,
        // identical bytes, identical reduction, only the dispatch size moves.
        // Three interleaved rounds at 200 samples:
        //
        // ```text
        //   PIE_ROWREP   grid       p50 mean    against 1
        //   1            rows/4     9.603 ms
        //   2            rows/8    10.277 ms      +7.0 %
        //   4            rows/16   12.448 ms     +29.6 %
        // ```
        //
        // Halving qmv's grid costs 21% OF QMV; quartering it costs 89%. The
        // curve is superlinear and still climbing at the shipped grid, so the
        // doubling a join would buy is real. The rate column above is the
        // conservative end of that slope.
        //
        // # TWO THINGS THAT LOOK LIKE SHORTCUTS TO THAT GRID AND ARE NOT
        //
        // SPLIT-K IN `qmv` ITSELF needs no model crate at all: give each
        // output row two workgroups over half of K each and merge. It doubles
        // the grid, which is the measured lever, and it still loses. The slope
        // says a doubling is worth about 0.55 ms spread over 141 fires -- 3.9
        // us apiece -- and the merge is a dispatch, which is 6. It loses by
        // half before the merge kernel does any work, and it loses at EVERY
        // split factor because the merge count tracks the fire count. That is
        // why the join is the only way up this curve: it is the only change
        // that grows the grid while REMOVING launches.
        //
        // AND THE JOIN SAVES ONE LAUNCH A LAYER, NOT TWO. `model_dsl::metal`'s
        // `split_qkv` is a `launch_with_params`, not a view: the packed bank
        // has to be deinterleaved into three buffers before `rms_rope` and
        // `kv_append_paged` can read them. So three qmv fires become one qmv
        // and one split, and 0.17 ms of the 0.34 ms floor saving goes straight
        // back. The rate term survives whole, which is why the row is still
        // worth ~0.5 ms.
        //
        // # AND THEN THE RATE TERM WAS MEASURED DIRECTLY AND IT IS ZERO
        //
        // Everything above about a rate bonus comes from reading two curves
        // that only ever measured the grid getting SMALLER. `qmv.wgsl`'s
        // `PIE_ROWW` measures it getting bigger: two output rows a workgroup
        // instead of four, same weight traffic, grid doubled. Three interleaved
        // rounds put it at +0.84% with the rounds overlapping, and the same
        // sweep with the loop's arithmetic stubbed on both arms -- which is the
        // only thing the probe adds -- puts it at -0.7%, worse in every round.
        // The two straddle zero.
        //
        // There is a knee and the shipped grid is on top of it. Below, the cost
        // is steep (halving is 7%, quartering 30%); above, the curve is flat.
        //
        // So strike the rate column. `q||k||v` is worth its launch floor and
        // `split_qkv` takes half of that back, which leaves ~0.17 ms and 1.8%;
        // `gate||up` has no split to pay and is worth ~0.17 ms too. Both still
        // want the model-crate join, and neither is now the largest item in the
        // decode -- they are ordinary launch-floor savings like the norm rows,
        // and they should be planned as such.
        //
        // The thing that looked unexplained -- qmv reading its weights at 68
        // GB/s of 273 with the ALU, the load issue and the dispatch size all
        // measuring free -- is answered in `device.rs` above the dispatch
        // loop, and the answer is that wgpu-hal's Metal backend only ever
        // builds a `MTLDispatchTypeSerial` compute encoder. Independent
        // dispatches cannot overlap on this backend at any resource layout, so
        // a kernel whose fires are small spends its time between them and no
        // knob inside the kernel can reach it.
        //
        // # THE HOST'S SHARE IS 18% OF A DECODE, AND HERE IS ALL OF IT
        //
        // Every budget above this line is a GPU budget: a kernel's share, or a
        // launch's. What none of them says is what the CPU spends, and the
        // "unattributed ~0.92 ms" in the table below was a subtraction and not
        // a measurement. It can be measured directly -- skip EVERY kernel and
        // the token that remains has no dispatches in it at all:
        //
        // ```text
        //   PIE_WGPU_SKIP=affine_qmm,affine_qmv,embed_gather,kv_append,rms,
        //                 row_gather,sdpa,silu_mul,rope,softmax,argmax,cast,
        //                 copy,add,mul
        // ```
        //
        // **1.123 ms, three rounds inside a tenth of each other, 15% of a
        // 7.446 ms token, and not one workgroup launched.** `device.rs`'s skip
        // deletes `set_pipeline`, `set_bind_group` and `dispatch_workgroups`
        // and nothing else, so everything the host does survives it.
        //
        // The `[probe]` line decomposes that floor -- `encode` is the top of
        // `fire` to just before `submit`, `gpu` is `submit` to after the wait,
        // `read` is the mapped readback -- and the difference between their sum
        // and the bench's own median is what `Shell::step` spends outside
        // `fire` altogether:
        //
        // ```text
        //   shadow + bind groups, 424 fires        0.266 ms   3.6%
        //   recording 424 dispatches               0.234      3.1%   (by
        //                                                     difference:
        //                                                     shipped encode
        //                                                     is 0.500)
        //   submit + wait, an EMPTY command buffer 0.354      4.8%
        //   the readback, 303872 B of logits       0.005      0.1%
        //   `step` outside `fire`                 ~0.490      6.6%   (WRONG.
        //                                                     Corrected below:
        //                                                     it is 0.14, and
        //                                                     the rest is in
        //                                                     `serve::fire`.)
        //   ------------------------------------------------------
        //   the host                              ~1.35      18%
        //   the GPU                               ~6.1       82%
        // ```
        //
        // Four things follow.
        //
        // **The readback is nothing.** 297 KiB of bf16 logits cross back every
        // token in 5 microseconds. The `answer`/`staged` path is not a target
        // and a device-side sampler would buy nothing here.
        //
        // **The wait is not a scheduler sleep.** `wait()` blocks in
        // `poll(PollType::Wait)`, which sleeps on a condvar, and the obvious
        // suspicion is wake-up latency. Replacing it with a `PollType::Poll`
        // spin over the same deadline reads 1.161/1.151/1.140 against
        // 1.123/1.125/1.128 interleaved -- WORSE, three times out of three.
        // 0.354 ms is what a Metal command buffer costs to commit and retire,
        // and this crate cannot spend less of it.
        //
        // **THE BIND CACHE MISSES 30% OF THE TIME AND IT DOES NOT MATTER.**
        // Counted: 10489 storage misses in 35164 calls, and the missing fires
        // are exactly the three that touch the KV cache -- `rms_rope` (56 a
        // token), `kv_append` (28) and `sdpa_paged_decode_split` (28), 112 of
        // 424 -- because their bindings carry a RANGE that moves as the
        // context grows, so the content key is new every token. The uniform
        // cache, by contrast, hits essentially always (20 misses, then none).
        //
        // This looks like a fix and is not one. Dropping `offset` and `size`
        // from the key makes every call a hit -- wrong answers, a probe only --
        // and buys **0.011 ms**, 0.15% of a token, interleaved three rounds.
        // `wgpu`'s `create_bind_group` on Metal builds a small Rust value and
        // no driver object; it is the KEY that costs, not the miss. Do not
        // rebind the KV pages whole to chase this.
        //
        // **And what is left has no lever.** The four survivors are 0.23-0.49
        // ms each. Two of them -- the bind groups and the recording -- are
        // strictly per dispatch, at (0.266 + 0.234) / 424 = **1.18 us of host
        // per fire**, which is the number a fusion actually deletes and which
        // says again what the join section below says: 28 fewer fires is 0.033
        // ms. The submit is a hardware round trip.
        //
        // # WHERE THE HOST ACTUALLY SPENDS IT, AND THE CORRECTION THAT COST
        //
        // The row above that says `step` outside `fire` is 0.490 ms is WRONG,
        // and it was wrong because it was a DIFFERENCE -- the host floor minus
        // the `[probe]` counters -- and the probe's own counters are inflated
        // by the probe. Measured directly instead, with a temporary timer
        // module reading the same floor config (every kernel skipped, so the
        // GPU does nothing and only the host is on the clock):
        //
        // ```text
        //   `serve::fire` before it calls `record`  0.551 ms   7.4% of a token
        //     of which `dispatch::plan_all`         0.503      6.8%
        //   `record`'s run construction             0.030
        //   `run_all_reading`, ALL of it            0.43       (bind + encode +
        //                                                      submit + wait +
        //                                                      read)
        //   `step` outside `fire`                   0.144      (stage 0.049,
        //                                                      tail 0.053,
        //                                                      arena 0.009,
        //                                                      ids 0.003,
        //                                                      run build 0.030)
        //   ------------------------------------------------------
        //   the host                                1.11-1.12  15%
        // ```
        //
        // Two things this settles.
        //
        // **`step` outside `fire` is 0.14 ms, not 0.49, and it has no lever.**
        // The biggest thing in it went already -- `lowering::cached` took
        // 0.765 ms off it in an earlier sitting, and `Lowerings` is confirmed
        // a real cache here: 0.966 ms on the first step of a run and 0.000 on
        // every one after. What is left is five items of 3-53 microseconds.
        // Stop looking here.
        //
        // **THE HOST'S ONE BIG ITEM IS `dispatch::plan_all`, AT 0.503 MS.**
        // That is 91% of `serve::fire`'s pre-`record` half, 45% of the whole
        // host floor, and 6.8% of a shipped token -- the single largest host
        // cost in a decode, and it had never been priced. It runs once per
        // launch per step and re-derives, every token, a dispatch geometry
        // that the plan already fixed. `plan_all` itself is four lines; it
        // delegates to `lowering::routine::plan`, and that is where the 0.49
        // lives.
        //
        // One piece of it is already off. `routine::armed`, the symbol fork
        // `plan_all` runs first, called `kernels_wgpu::routines()`, which
        // COLLECTS the registry into a fresh `Vec`, and then linear-scanned it
        // comparing names -- 424 allocations and 424 scans a token to answer a
        // question whose answer is a constant. It is a `OnceLock` map now and
        // `plan_all` reads 0.503 against 0.518 interleaved. **0.015 ms**: real,
        // and small, which is itself the finding -- the registry is short and
        // the allocation is cheap, so the cost is NOT the symbol fork and the
        // remaining 0.49 is the planning proper.
        //
        // # `plan_all` CUT APART, AND THE COUNTING PASS TAKEN OFF IT
        //
        // `plan_all` is four lines; `lowering::routine::plan` is where its
        // 0.49 lives, and timed inside that function, per token:
        //
        // ```text
        //   the COUNTING bind pass  0.097 ms   27% of `plan`
        //   the real bind pass      0.070
        //   `stating`, the body     0.073
        //   the `bounds` walk       0.035      (every buffer resolved)
        //   the `out` walk          0.034
        //   preamble + tail         ~0.05
        //   ---------------------------------
        //   `routine::plan`         0.359
        //   `plan_all` around it    0.503      (the rest is `hold::facts`,
        //                                      `armed` and `widths`)
        // ```
        //
        // NOTHING HERE DOMINATES. Six items of 0.03-0.10 ms, which is why this
        // side never showed up in a profile: it is the same small walk run 424
        // times. But one of the six is pure duplicate work, and it is off now.
        //
        // **THE COUNTING PASS PRODUCED ONE INTEGER.** `plan` binds twice -- the
        // note at its head says why, and the reason is real: a `F32sMut` cannot
        // tell a statement's result from a GDN recurrent slab, so the binder
        // walks an undivided statement first purely to COUNT the asks. But
        // `asked_results()` returns a `usize`, and the walk that produces it
        // reads only the routine's SIGNATURE under `facts` -- no buffer, no row
        // range, no step. It was 0.097 ms a token, 27% of `plan` and 1.3% of a
        // shipped token, to recompute a constant 424 times.
        //
        // Memoised on `(routine.name, facts, args.len(), scalars.len())` in a
        // thread-local map. The two lengths are not needed by the argument
        // above; they are in the key so a statement of a different shape
        // cannot borrow another's count.
        //
        // ```text
        //   the counting pass       0.097  ->  0.034     (the LOOKUP, now)
        //   `routine::plan`         0.359  ->  0.303
        //   the host floor          1.123  ->  1.037-1.048
        //   a shipped decode        7.408-7.446 -> 7.329-7.377
        // ```
        //
        // Three readings, all three below every baseline reading, and the
        // token delta equals the floor delta -- which is the check that says
        // a host saving reached the token at all. 134.3 -> 136.4 tok/s.
        //
        // The residue is the LOOKUP: hashing an eleven-field `Facts` 424 times
        // costs 0.034 ms, a third of the pass it replaced. If this side is
        // worked again, that is a key to make cheaper, not a pass to bring
        // back.
        //
        // # THE ONE FIRE THAT REACHES 188 GB/s, AND THE 196 THAT REACH 65
        //
        // `qmv` is 3.88 ms and 52% of the token and it had only ever been
        // priced WHOLE. Priced by GRID instead -- a temporary
        // `PIE_WGPU_SKIPBIG=<rows>` in `run_all_reading` that drops the
        // dispatch of any fire whose `groups[1]` is at least that, so the host
        // is untouched and only the GPU moves -- four thresholds cut it into
        // its four shapes. Base 7.348; 20000 reads 6.909, 700 reads 5.652,
        // 500 reads 5.147, 200 reads 3.469.
        //
        // The bytes are the model's, not a guess: 4-bit weights with `gs = 64`
        // bf16 scales, hidden 1024, intermediate 3072, 28 layers, vocab
        // 151936.
        //
        // ```text
        //   grid rows  fires    bytes     ms     GB/s   us/fire  what
        //      37984      1    82.7 MB   0.439   188     439     the LM head
        //        768     56    93.6      1.257    74      22     gate + up
        //        512     28    31.2      0.505    62      18     q
        //        256    112   109.2      1.678    65      15     k, v, o, down
        //   ------------------------------------------------------
        //              197   316.7 MB   3.879    82
        // ```
        //
        // **THE LM HEAD IS 70% OF THIS MACHINE'S PEAK.** 82.7 MB in 0.439 ms
        // is 188 GB/s against the M4 Pro's 273, and it is ONE fire of 37984
        // workgroups. It is the most efficient thing in the token by a factor
        // of three, it was never looked at because it looked like an obvious
        // target -- 31% of all `qmv` grid rows -- and there is nothing in it
        // to win. Do not go back to it.
        //
        // **AND IT IS THE CALIBRATION EVERYTHING ELSE IS NOW MEASURED
        // AGAINST.** The same kernel, on the same weights, in the same token,
        // gets 62-74 GB/s when the grid is 256-768 rows. If the 196 layer
        // fires ran at the LM head's rate they would take 1.24 ms instead of
        // 3.44. **That gap is 2.2 ms, 30% of a decoded token, and it is the
        // largest single item in this project.**
        //
        // It is not a flat dispatch fee. The excess per fire is 15/18/22 us
        // and RISES with the grid, and the rate rises with it too -- 65, 62,
        // 74, 188 -- which is the shape of a ramp, not a toll. Two other
        // kernels say the same: `silu_mul` fires 28 times for 0.040 ms (1.4 us
        // a fire) and `kv_append` 28 times for 0.072 (2.6), so a dispatch that
        // moves little costs almost nothing. What costs is a fire that WANTS
        // bandwidth and does not have enough workgroups in flight to keep the
        // memory system busy through its own start and drain. 256 workgroups
        // over 20 cores is thirteen apiece.
        //
        // # THE SAME FAMILY WITHOUT AN INSTRUMENT, AND THE STAMP IS 32% HIGH
        //
        // `serving.rs`'s census fits the qmv family from `PIE_WGPU_STAMP`
        // counters and reaches `7.8 us + 4.64 ns/row + 4.96 us/MiB (+2.2 if
        // residual)`. Deletion reaches the same five shapes with no instrument
        // on the clock at all -- `PIE_WGPU_SKIPBIG` for the grid cuts,
        // `PIE_WGPU_SKIP=affine_qmv_fast_residual` to lift `o` and `down` out
        // of the 256-row row they share with `k` and `v`:
        //
        // ```text
        //   fire            fires    ms     us/fire   STAMPED
        //   k, v              56    0.510      9.1     15.85
        //   q                 28    0.505     18.0     22.90
        //   gate, up          56    1.257     22.4     30.45
        //   o, down (res)     56    1.168     20.9     21.75 (mean)
        //   the lm head        1    0.439    439       666
        //   ------------------------------------------
        //                    197    3.879
        // ```
        //
        // **THE RIGHT-HAND COLUMN SUMS TO 5.118 ms AND qmv IS 3.88.** The
        // left-hand column sums to 3.879. One of these two instruments agrees
        // with the whole it is a decomposition of, and it is not the stamp:
        // stamping writes a timestamped pass per launch and charges it 424
        // times, this file already says a `[cost]` row is a share and never a
        // duration, and here is that bias measured -- **32%, and not evenly
        // spread.** The residual pair lands within 4%; `k`/`v`, the smallest
        // fire, is charged 74% over. A fixed per-launch instrument overstates
        // small fires most, which is exactly the direction that flatters a
        // join.
        //
        // **AND THAT REVERSES WHICH JOIN TO DO FIRST.** Fitting the tail of
        // the clean column, `gate` to the lm head, gives `13.8 us + 2.80
        // ns/row`, and a join is a fire deleted at constant rows:
        //
        // ```text
        //   gate||up   2 x 22.4 = 44.8 -> one 6144-row fire, 31.0
        //              13.8 us a layer  = 0.386 ms   NO SPLIT KERNEL
        //   q||k||v    18.0 + 9.1 + 9.1 = 36.2 -> one 4096-row fire, 25.3
        //              10.9 us a layer  = 0.306 ms, less `split_qkv`'s own
        //              launch, so 0.14-0.23 net
        // ```
        //
        // The stamped fit ordered these the other way -- q||k||v 16.6 us a
        // layer against gate||up's 7.8 -- because it charged `k` and `v` 15.85
        // us each when they cost 9.1, and a three-fire join collects two of
        // those inflated intercepts. **gate||up is the bigger of the two, it
        // is the one with no split kernel behind it, and it should be done
        // first.** Both estimates of the pair still land near 0.5-0.6 ms,
        // which is where the stamped 0.68 was after its own give-back
        // correction; the total was roughly right and the ORDER was not.
        //
        // # AND THE OBVIOUS WAY TO CHECK THAT NUMBER MEASURES THE CACHE
        //
        // `gate||up` turns two 3072-row fires into one 6144-row fire, and the
        // only anchor above 3072 rows is the lm head at 151936, so the 31.0 us
        // it predicts is an extrapolation across a fifty-fold gap. The cheap
        // check looks perfect: cap the lm head's grid with a temporary
        // `PIE_WGPU_GRIDCAP=<workgroups>` and the SAME BANK yields a fire of
        // any size, reading real weights, in the real token. It answers
        //
        // ```text
        //   cap        bytes    ms over cap=1
        //        1       0.0     0.000
        //      768       1.9    -0.033      <- FASTER than one workgroup
        //     6144      14.9     0.010      <- 1400 GB/s
        //    12288      29.9     0.071      <-  400 GB/s
        //    24576      59.8     0.258      <-  230 GB/s
        //    37984      92.4     0.436      <-  190 GB/s, the shipped fire
        // ```
        //
        // **A CAPPED FIRE ALWAYS READS THE SAME FIRST BYTES OF THE BANK**, so
        // at 15 and 30 MB it is reading the system level cache and reporting
        // impossible rates, and only the last two rows are DRAM. The probe
        // measures residency, not the curve. Do not repeat it; a cap that
        // moved the OFFSET as well as the count would be needed, and the
        // uniform block does not carry one.
        //
        // It does confirm the shipped lm head at 190 GB/s from a second
        // direction, and it bounds the join from below in the only way it can:
        // between 3072 and 6144 rows the marginal cost is somewhere between
        // the 2.80 ns/row the tail fit gives and the 4.30 ns/row the measured
        // 2048->3072 step gives, so `gate||up` saves 13.8 us a layer at best
        // and 9.2 at worst -- **0.26 to 0.39 ms, 3.5% to 5.3% of a token** --
        // and no reading of any curve here makes it worth less than that.
        //
        // # AND IT REFITS THE COST MODEL, WHICH MOVES THE JOINS AGAIN
        //
        // The fit further down this file -- **7.15 us + 7.73 us a MiB**, a
        // marginal 135.7 GB/s -- was drawn from three points that span 0.56 to
        // 1.69 MiB. Extended to the LM head it predicts 617 us and the fire
        // measures 439: **40% high**, which is what fitting a slope over a
        // three-fold span and reading it over an eighty-fold one does.
        //
        // Refitted on the two ENDS, 0.93 MiB at 15.0 us and 78.9 MiB at 439:
        //
        //     9.94 us + 5.44 us a MiB   (193 GB/s marginal)
        //
        // It under-predicts the two middle shapes by 2-4 us, so the true curve
        // is not a line; but the INTERCEPT is the load-bearing term and it
        // moved 7.15 -> 9.94, up 39%. **196 qmv fires x 9.94 us is 1.95 ms,
        // 26% of a decoded token, spent before a weight byte is read.**
        //
        // The joins, re-priced on that intercept:
        //
        // ```text
        //   q||k||v    3 fires -> 1 qmv + 1 split_qkv
        //              2 x 9.94 - ~5 = ~15 us a layer = 0.42 ms, 5.7%
        //   gate||up   2 fires -> 1, no split
        //              9.94 us a layer            = 0.28 ms, 3.8%
        //   -------------------------------------------------------
        //                                          0.70 ms, 9.5%
        // ```
        //
        // Against 0.49 ms on the old fit. And the blast radius is now known to
        // have a THIRD gate on it, cheaper than the two this file already
        // names: `qwen_3` does not call `dense_fused_projection_joins` at all.
        // The other two stand -- `Projections::InPlace` returns early, and so
        // does a runtime quantization scheme -- but the second is arguable
        // rather than fundamental: this deployment quantizes with `gs = 64`
        // along K, every output row is quantized independently of every other,
        // and a join CONCATENATES OUTPUT ROWS. Nothing about it crosses a
        // quantization group.
        //
        // **THIS RE-PRICES FUSION BY A FACTOR OF TEN.** The host section above
        // costs a fire at 1.18 us and concludes that deleting 28 of them buys
        // 0.033 ms. That is the HOST's share and it stands, but the GPU's
        // share of a `qmv` fire is 15-22 us, so 28 fewer `qmv` fires is
        // 0.4-0.6 ms, not 0.033. Every join argument in this file that was
        // settled on the 1.18 us number was settled against the wrong number.
        // Joining `gate` and `up` -- 56 fires into 28, and they share an input
        // and a grid -- is the first thing to re-open.
        //
        // What is left of `plan_all` is 0.42 ms of five small walks, and the
        // precedent for taking them wholesale is `lowering::cached` -- the same
        // shape of question, a pure function of the plan recomputed per step,
        // which paid 0.765 ms. The obstacle is that `plan`'s OUTPUT is not
        // step-invariant: the `bounds` walk resolves buffers, and the KV
        // ranges move as the context grows. The other five walks are
        // structure, and structure does not move.
        //
        // # THE ATTENTION, TAKEN APART THE SAME WAY
        //
        // `sdpa_paged` is 1.838 ms and 24.7% of the token, and until now that
        // was all that was known about it. The same uniform-false early return
        // that emptied `qmv` was put at the top of `sdpa_paged.wgsl`'s
        // `decode_row`, which empties the split arm and leaves the merge
        // running. Plain medians, three rounds, no instrument:
        //
        // ```text
        //   split body       1.458 ms   19.6%   28 fires   52.1 us each
        //   split launch     0.270       3.6%   28          9.6 us each
        //   merge, all of it 0.110       1.5%   28          3.9 us each
        // ```
        //
        // The merge is closed -- 1.5% of a token buys nothing worth a change.
        // The 9.6 us launch is the largest per-fire dispatch cost in the
        // decode, but it is not anomalous: this arm launches 128 workgroups of
        // 256 threads, the biggest grid in the token, and at 75 ns a workgroup
        // it is paying exactly `qmv`'s rate. The ramp is per workgroup, and
        // both axes of this grid -- `PIE_SPLITS` and the query head count --
        // are swept shut.
        //
        // So the whole of the opportunity is 1.458 ms of body, and the file
        // that owns it now carries the analysis: it is a cache hit throughout,
        // it is not bandwidth, and the only lever that has ever moved it is
        // issuing fewer and wider loads per lane.
        //
        // AND THAT LEVER IS GONE TOO. Six deletion probes into the key loop,
        // five of them free: the V fetch is HALF of every load in the body and
        // returns nothing, the five butterfly shuffles a key return nothing,
        // both `exp()`s return nothing, and remapping a lane's channels from
        // strided to contiguous -- the shape a wide load would need -- is a
        // dead heat over three interleaved rounds. Only the page-table load
        // answers, at 0.197 ms, because it is the one DEPENDENT load in the
        // body; four ways of collecting it (register cache, workgroup staging
        // with and without a fallback branch, and a one-block prefetch) all
        // lose. Widening a lane to a 16-byte vector costs 0.55 ms, on the same
        // residency cliff `qmv` fell off.
        //
        // A kernel whose every part is free and whose whole is 1.458 ms is
        // bound at the loop and not at any instruction in it. The knob that
        // changes how much is in flight is `PIE_SPLITS`, and it has been swept
        // four times and says eight.
        //
        // # THE LAUNCH COSTS 1.085 ms A TOKEN, AND THE JOIN STILL DOES NOT PAY
        //
        // Measured in `quant/qmv.wgsl`, not here, but it settles the question
        // this file has been circling for four sittings. Plain interleaved
        // decode medians, no `PIE_WGPU_PROBE` and no `PIE_WGPU_STAMP`: 7.425 ms
        // shipped, 4.626 ms with qmv's body deleted but every fire still
        // encoded and every workgroup still launched, 3.541 ms with the fires
        // never encoded. So 1.085 ms of a token is spent by 196 qmv fires
        // before they read a byte.
        //
        // That is NOT 5.54 us a dispatch, which is what dividing it by 196
        // says and what an earlier revision of this comment claimed. Deleting
        // 28 small-grid fires directly -- `PIE_WGPU_SKIP=silu_mul` -- returns
        // 1.4 us each, and `kv_append` returns 2.6 us each. Re-running the
        // empty build with the qmv grid cut eightfold returns 0.406 ms of the
        // 1.085. Most of the launch cost is the GRID, and the fixed residue is
        // ~2-3 us a fire.
        //
        // A join leaves the grid alone -- 512 + 256 + 256 workgroups become
        // 1024 -- so it can only collect the residue, and `dsl::ops::split_qkv`
        // is itself a launch, so q||k||v nets one fire a layer rather than two.
        // 28 fires at ~2-3 us is under 1% of a token, against changes in
        // `model`, `model-dsl` and three kernel crates. The ~8% below, fitted
        // from a stamped table, is an artefact of that instrument: probing and
        // stamping inflate a decode median from ~7.4 ms to ~13.8 ms -- one
        // timestamped pass per launch, charged 424 times -- so a `[cost]` row
        // is a SHARE, never a duration, and any absolute microsecond quoted
        // from one is high.
        //
        // # THE WHOLE TABLE, RE-TAKEN, AND EVERY NUMBER BELOW IS STALE BY 30%
        //
        // Everything under the next heading was stamped when a token was 8.3 to
        // 9.8 ms. It is 6.08 ms of GPU now -- the subgroup ladders, the fused
        // `rms_rope` and the attention's block rewrite all landed since -- so
        // the per-launch costs it quotes are 30% high and the SHARES it draws
        // from them are wrong in both directions. Re-taken with
        // `PIE_WGPU_PROBE=1 PIE_WGPU_STAMP=50`, reading the window that
        // contains only decode fires (`inside=6.082ms/fire`, `span=6.364`, so
        // turnaround is 0.282 ms a fire over 424 launches -- **0.67 us a
        // launch**, which reproduces the 0.71 below on a token a third
        // smaller):
        //
        // ```text
        //   kernel                        each      x/token     ms      share
        //   sdpa_paged_decode_split      49.5 us       28      1.386    22.8%
        //   affine_qmv_fast   [1,768,1]  20.2          56      1.129    18.5%
        //   affine_qmv_fast   [1,256,1]  11.5          56      0.643    10.6%
        //   affine_qmv_residual (down)   18.4          28      0.514     8.4%
        //   affine_qmv_fast  lm head    457.0           1      0.457     7.4%
        //   affine_qmv_fast   [1,512,1]  15.3          28      0.429     7.1%
        //   affine_qmv_residual (o)      15.3          28      0.429     7.0%
        //   rms_single_row                6.8          57      0.386     6.4%
        //   rms_rope          [1,16,1]    7.1          28      0.200     3.2%
        //   rms_rope          [1,8,1]     6.6          28      0.186     3.0%
        //   sdpa_paged_decode_merge       5.1          28      0.143     2.4%
        //   kv_append_paged               3.6          28      0.100     1.8%
        //   silu_mul                      3.1          28      0.086     1.3%
        //                                            ----     -----
        //                                             424     6.088
        // ```
        //
        // qmv is 3.60 ms and 59%, attention 1.53 and 25%, the three norms 0.77
        // and 13%. Nothing else in the decode is 2%.
        //
        // AND THE qmv FIT MOVED, WHICH IS THE PART THAT CHANGES A DECISION.
        // The same three shapes hold 0.5625, 1.125 and 1.6875 MiB of weight
        // bank, and the two outer points now give
        //
        //     7.15 us + 7.73 us a MiB
        //
        // with the middle point landing at 15.85 against 15.3 measured. The
        // marginal rate has gone 79.2 GB/s -> **135.7 GB/s** and the intercept
        // 8.65 -> 7.15 us. So the kernels got faster at the bytes and barely
        // faster at the floor, and the floor is now a LARGER fraction: 196 qmv
        // fires x 7.15 us is 1.40 ms, **23% of the token**, against 45% when
        // the intercept was first fitted.
        //
        // WHAT THAT SAYS ABOUT THE TWO JOINS, in today's numbers:
        //
        // ```text
        //   q||k||v    3 fires -> 1 qmv + 1 split_qkv
        //              saves two intercepts, pays a small kernel back
        //              2 x 7.15 - ~4 = ~10 us a layer = 0.29 ms, 4.8%
        //   gate||up   2 fires -> 1, no split (silu_mul reads one bank
        //              at an offset)
        //              7.15 us a layer = 0.20 ms, 3.3%
        // ```
        //
        // **~0.49 ms and 8% for both**, which is smaller than the ~1.2 ms the
        // rate-bonus paragraphs above claimed and larger than the 0.04 ms the
        // turnaround-only correction below leaves. Both of those were right
        // about their own term and wrong about which term dominates; the
        // intercept does, it is inside the kernel, and a join is still the only
        // change that deletes one.
        //
        // AND `rms_single_row` STAYS DEAD for the same reason it always was --
        // 0.386 ms is the whole prize and the five qmv fires that would absorb
        // it are 3.14 ms -- but note it is now 6.8 us a fire against a 3.1 us
        // `silu_mul`, so about half of it is the same floor everything else
        // pays and not its arithmetic. One workgroup on twenty cores.
        //
        // # AND THEN THE 4.6 us TURNED OUT NOT TO BE TURNAROUND
        //
        // EVERYTHING BELOW THIS HEADING IS PRICED AT 4.6 us A LAUNCH AND THAT
        // NUMBER IS WRONG AS A LAUNCH PRICE. `PIE_WGPU_STAMP` times each
        // dispatch with the GPU's own clock; over a 424-launch decode the GPU
        // is inside a kernel for 8.56 ms of the 8.86 ms it is busy, so
        // turnaround is **0.71 us a launch**, measured with one encoder per
        // dispatch where the shipped path uses one for the whole fire. The
        // 4.6 us the `rms_rope` fusion bought was the KERNEL: `rms_single_row`
        // costs 7.9 us of GPU and the fusion deleted 56 of them.
        //
        // So the rule for this whole list changes shape. A fold is worth what
        // its DELETED KERNEL costs, minus what it adds to the kernel that
        // absorbs it, plus 0.71 us. The launch count barely matters. Measured
        // costs, from `serving.rs`'s table:
        //
        // ```text
        //   kv_append_paged      4.3 us  x 28  = 0.12 ms
        //   rms_single_row       7.9 us  x 57  = 0.45 ms
        //   silu_mul             3.3 us  x 28  = 0.09 ms
        //   sdpa_decode_merge    6.4 us  x 28  = 0.18 ms
        // ```
        //
        // Which KILLS `rms_single_row into its qmv`. The prize is 0.45 ms, but
        // the five qmv fires a layer that read its output must each redo the
        // row sum at ~+26% ALU, and the 768-group qmv costs 31.0 us -- +26% is
        // +8.1 us, MORE than the 7.9 us saved, on each of the five. The fold
        // makes the token slower. It was ranked last here for having a quarter
        // of its prize eaten; the real figure is more than all of it.
        //
        // And it RE-ARGUES `q||k||v` on completely different grounds. As a
        // launch-count saving it is now worth 56 x 0.71 us = 0.04 ms and is
        // not worth the model-crate work. What makes it worth doing is that
        // qmv's fires are NOT linear in their rows:
        //
        // ```text
        //   [1, 256, 1]   16.1 us      [1, 512, 1]   23.2 us
        //   [1, 768, 1]   31.0 us
        // ```
        //
        // Three times the rows for twice the time -- and the three shapes fit a
        // STRAIGHT LINE in the bytes their weight banks hold. At 4-bit with
        // bf16 scales and biases every 64 those banks are 0.5625, 1.125 and
        // 1.6875 MiB, and the three stamped times are
        //
        //     8.65 us + 13.24 us a MiB
        //
        // with the middle point, which the fit does not use, landing at 23.55
        // against 23.2. So the marginal rate is 79.2 GB/s and **a qmv fire
        // costs 8.65 us before it reads a byte** -- 12x the 0.71 us of
        // turnaround, so it is inside the kernel and not the dispatch.
        //
        // ## AND THE INTERCEPT IS NOT A qmv FACT. IT IS 45% OF THE TOKEN.
        //
        // The 8.65 us above is fitted from ONE kernel's three shapes and reads
        // like a property of `affine_qmv_fast`. It is not. Three kernels with
        // nothing in common measure the same constant:
        //
        // ```text
        //   affine_qmv_fast   fitted intercept, three bank sizes     8.65 us
        //   rms_single_row    its WHOLE cost, grid [1, 1, 1], 2 KiB  7.9  us
        //   sdpa_..._split    residue with the 64-key loop gutted    ~9    us
        // ```
        //
        // A row reduction over 2 KiB in one workgroup and a 16x8 attention
        // grid with an empty body have no shared arithmetic, no shared traffic
        // and no shared occupancy. What they share is being a DISPATCH, and
        // `device.rs` names the mechanism above `run_all`: wgpu-hal 30 opens
        // every Metal compute encoder with `computeCommandEncoder()`, which is
        // `MTLDispatchTypeSerial`, so each dispatch waits for the previous one
        // to COMPLETE and pays a full drain and refill whatever it computes.
        //
        // Count the fires and the term stops being a footnote. Normalising
        // `PIE_WGPU_STAMP=200`'s table by `kv_append_paged`, which fires
        // exactly once a layer:
        //
        // ```text
        //   qmv (five rows)        195   rms_single_row          57
        //   sdpa split + merge      56   rms_rope (two rows)      57
        //   kv_append               28   silu_mul                 26
        //                                                    ------
        //                                                    ~421 a token
        // ```
        //
        // **421 fires x 8 us = 3.4 ms of a 7.465 ms token.** The average fire
        // is 17.7 us and 8 of that is charged before it reads a byte. That is
        // the largest single item in this decode by a factor of two over the
        // attention's whole key loop, and no knob inside any kernel reaches
        // it -- six probes into the split kernel and three into qmv came back
        // free for this reason and not by coincidence.
        //
        // Two consequences for this list.
        //
        // **The rule "a fold is worth its deleted kernel's cost" has a floor
        // of 8 us, and every candidate here is near it.** `rms_single_row` at
        // 7.9, `silu_mul` at 3.3 and `sdpa_decode_merge` at 6.4 are not four
        // different prices; they are one constant plus a little work. So a
        // fold that deletes one fire a layer is worth about 28 x 8 us = 0.22
        // ms and NO candidate can be worth much more, which is why the
        // repriced joins, the rope||append fold and the norm rows all land
        // between 0.15 and 0.45 ms. The list is flat. Rank it by cost of
        // implementation, not by prize.
        //
        // **And the only change that beats the whole list is not in this
        // repo.** llama.cpp opens its encoder `MTLDispatchTypeConcurrent` and
        // places `memoryBarrierWithScope` where it needs one; `wgpu` exposes
        // no way to ask for that. Reaching it means patching wgpu-hal and
        // taking over the hazard tracking wgpu currently gets for free from
        // Metal's ordering -- and most consecutive pairs in this plan ARE
        // dependent, so the win is only over the genuinely independent runs
        // (q, k, v; gate, up), which is the same handful of fires the joins
        // delete anyway. It is a real option and it is a judgement about
        // dependencies, not a measurement. Do not start it without deciding
        // that first.
        //
        // ## AND THEN A FIRE WAS ACTUALLY DELETED, AND 8 us IS WRONG
        //
        // Everything under the heading above is inference. Three routes to
        // ~8 us and not one of them removed a dispatch: a straight-line fit
        // through three qmv bank sizes, a kernel small enough to be ASSUMED
        // all overhead, and the residue of an attention kernel with its loop
        // gutted. Rule 7 of the measurement discipline says a profiler's
        // shares must be checked against a probe before they are spent, and
        // that heading spent them -- 3.4 ms and 45% of a token -- on the same
        // day it restated the rule.
        //
        // `device.rs`'s `PIE_WGPU_SKIP` deletes a kernel's fires outright:
        // no dispatch, no pipeline switch, no binds, which is what a fold
        // deletes. Wrong answers, so it is a probe. Three interleaved rounds
        // at 200 samples, each against its own baseline (7.446, 7.545,
        // 7.443):
        //
        // ```text
        //   kernel               fires   saved ms         mean    us/fire
        //   rms_single_row         57   .403 .539 .409   0.450     7.9
        //   sdpa_decode_merge      28   .159 .291 .149   0.200     7.1
        //   rms_rope               57   .317 .382 .339   0.346     6.1
        //   kv_append_paged        28   .051 .180 .100   0.110     3.9
        //   silu_mul               26   .084 .081 .060   0.075     2.9
        //                         ---                    -----
        //   all five at once      196                    1.075
        // ```
        //
        // The singles sum to 1.181 against 1.075 measured together, so the
        // effects are ADDITIVE within 10%, and the joint figure is one large
        // well-resolved measurement -- 14.4% of a token -- rather than five
        // that each sit near the 1.7% repeatability. Read the singles for
        // ranking and the 1.075 for the total.
        //
        // **196 fires for 1.075 ms is 5.5 us each, not 8.** And the spread is
        // the finding, because it is 2.7x wide and a dispatch drain cannot
        // have one:
        //
        //   * `rms_single_row` reads 2 KiB in ONE workgroup and costs 7.9 us.
        //   * `kv_append_paged` writes into a 64 MiB pool and costs 3.9.
        //   * `silu_mul` is 2.9.
        //
        // Not traffic, then, and not grid width either -- `merge` at [16,1,1]
        // is dearer than `silu_mul` at [12,1,1]. What separates them is
        // whether the kernel REDUCES. `rms_single_row`, `sdpa_decode_merge`
        // and `rms_rope` all fold a row and cost 6 to 8 us; `kv_append_paged`
        // and `silu_mul` are elementwise and cost 3 to 4. A reduction in a
        // single workgroup is a barriered tree whose depth is latency nothing
        // else in the fire can hide.
        //
        // So the model is a ~3 us floor plus 3 to 5 us for a reduction, and
        // the 8 us "constant" was two reduction kernels agreeing with each
        // other. `qmv`'s 8.65 us intercept belongs to qmv, `rms_single_row`'s
        // 7.9 belongs to a row sum, and the split attention's ~9 belongs to
        // its merge tail -- three kernels, three costs, no shared floor.
        //
        // The 45% claim is withdrawn. 196 of ~421 fires carry 1.075 ms; if
        // the other 225 were the same the whole per-fire term would be ~2.3
        // ms, and those 225 are qmv and attention, which do the work. Call it
        // under a third of the token and stop quoting a number nobody
        // removed a fire to get.
        //
        // WHAT IT DOES TO THIS LIST, which is why the probe was written. Every
        // row's prize is now an UPPER BOUND that was measured rather than
        // modelled -- upper, because the absorbing kernel takes the work on:
        //
        // ```text
        //   rms_single_row into its qmv       <= 0.450   already lost, below
        //   merge into the split kernel       <= 0.200   needs a device sync
        //   q-norm||k-norm in one rms_rope    <= 0.173   half of 57 fires
        //   rope k + kv_append                <= 0.110   was quoted at 0.12
        //   silu_mul into its qmv             <= 0.075
        // ```
        //
        // Not one of them reaches the repriced joins' 0.34 to 0.45 ms, so the
        // joins keep the top of the list -- on their own merits this time,
        // against bounds instead of against a constant.
        //
        // ## AND THE SAME KNOB SETTLES THE SHARES, WHICH WERE WRONG BOTH WAYS
        //
        // `[cost]`'s table has ranked this decode since the instrument was
        // built, and its shares have been argued about for five sittings:
        // stamped, `sdpa_paged_decode_split` reads 35-39% and the five qmv
        // rows sum to about 47%. `sdpa_paged.wgsl` then corrected the
        // attention to 19% from three indirect routes. Skipping the fires
        // answers it directly. Two rounds, baselines 7.440 and 7.451:
        //
        // ```text
        //   skipped                 ms        saved     share    fires
        //   affine_qmv_fast     3.552 3.528   3.91 ms   52.5%    ~196
        //   sdpa_..._split      5.700 5.768   1.71 ms   23.0%      28
        //   the five above          --        1.08 ms   14.4%     196
        //                                     -------   -----
        //                                     6.70 ms   90.0%
        // ```
        //
        // The two rounds agree to 0.06 ms on a 3.9 ms effect, so these are
        // the firmest numbers in this file.
        //
        // **Both stamp shares were wrong, and in the directions the surcharge
        // theory predicts.** Apple has no `TIMESTAMP_QUERY_INSIDE_PASSES`, so
        // stamping gives each fire its own compute pass and a pass costs in
        // proportion to the state its launch establishes -- which inflates
        // WIDE grids and leaves narrow ones nearly alone. The attention has
        // the widest grid in a decode (16 x 8 x 256) and stamped 35-39%
        // against a true 23%. qmv has the narrowest and stamped 47% against a
        // true 52.5%. A theory that only ever explained one direction has now
        // been checked in the other.
        //
        // The earlier 19% correction was right to disbelieve the stamp and a
        // little low: 1.71 ms, not 1.4. Its three routes bounded the loop and
        // the pieces it could name, and it had no way to see what it had not
        // thought to price. That is the difference between summing parts and
        // removing the whole.
        //
        // **AND IT MOVES THE PROGRAMME. qmv IS MORE THAN HALF THE DECODE.**
        // 3.91 ms in one kernel family. Its weight banks are 0.352 GB a token
        // -- 252 MiB of layers plus an 83.8 MiB lm head -- which is 1.29 ms at
        // this machine's 273 GB/s, so qmv runs at **THREE TIMES its own
        // bandwidth floor**, and the fitted marginal rate of 79.2 GB/s says
        // the same thing from the slope. Nothing else in the decode is that
        // far from a wall it could reach. The fold list above is worth about
        // 1 ms in total and every row of it needs another crate; closing half
        // of qmv's gap is worth more than all of them and is entirely inside
        // `qmv.wgsl`.
        //
        // What is already ruled out there, so nobody starts at the top:
        // `PIE_ROWW` (rows a workgroup, the grid getting BIGGER) straddles
        // zero over three rounds, dispatch size is free, and load issue is
        // free. What has NOT been priced is the 4-bit unpack and the affine
        // dequant on the critical path of every 64 weights.
        //
        // A join deletes intercepts, and that is the whole prize:
        //
        // ```text
        //   q + k + v    23.2 + 16.1 + 16.1 = 55.4 us   joined 2.25  MiB -> 38.5
        //   gate + up           31.0 + 31.0 = 62.0 us   joined 3.375 MiB -> 53.3
        // ```
        //
        // 16.9 and 8.7 us a layer, 0.72 ms a token between them -- EXCEPT that
        // a fourth point later broke the straight line. The lm head is a qmv
        // over an 83.4 MiB bank and it reads 666 us where the fit says 1113,
        // so there is no constant intercept: the rate rises with the bytes in
        // a fire, 36.6 -> 50.8 -> 57.1 -> 131.3 GB/s across the four. See
        // `qmv.wgsl`'s `PIE_ROWW` note for why that is consistent with the
        // grid being measured free.
        //
        // Restated as a BOUND, which is what survives. The rate is monotone in
        // bytes and a joined fire is bigger than every fitted point, so it runs
        // at no worse than the 57.1 GB/s of the largest one:
        //
        // ```text
        //   q + k + v   55.4 us today   joined <= 41.3 us   saves >= 14.1 us a layer
        //   gate + up   62.0 us today   joined <= 62.0 us   saves >= 0
        // ```
        //
        // # AND THEN THE COST TABLE LEARNED TO COUNT BYTES, AND BOTH ARE PROVEN
        //
        // The table above is keyed by entrypoint and grid, which averaged `o`
        // and `down` -- both `_residual` at [1,256,1], over 1.125 and 1.6875
        // MiB -- into one number. `device.rs::charge` now folds the launch's
        // bound bytes into the key, they came apart, and the pair settled the
        // whole question. `down` binds the SAME bytes as `gate`/`up` with a
        // THIRD of the grid and is 24% faster; `o` binds q's bytes with half
        // the grid and is 11% faster. Repeats in every window.
        //
        // So the axis is neither the grid nor the bytes: it is OUTPUT ROWS.
        // `qmv.wgsl` carries the fit; it is
        //
        //   t = 7.8 us + 4.64 ns/row + 4.96 us/MiB (+2.2 us if residual)
        //
        // and it lands the residual pair -- which is not fitted for the first
        // two terms -- within 0.01 us. 4.96 us/MiB is 209 GB/s on a 273 GB/s
        // machine, so **the decode's qmv is latency-bound, not bandwidth-bound**,
        // and what is left is a fixed cost a fire plus a cost a row.
        //
        // A JOIN DELETES FIRES, NOT ROWS. So it collects `a` per fire deleted,
        // exactly, and this stops being a bound:
        //
        // ```text
        //   q + k + v   54.6 us today   joined 4096 rows 2.263 MiB -> 38.0
        //               saves 16.6 us a layer  =  0.46 ms a token  4.8%
        //   gate + up   60.9 us today   joined 6144 rows 3.390 MiB -> 53.1
        //               saves  7.8 us a layer  =  0.22 ms a token  2.3%
        // ```
        //
        // **0.68 ms a token, 7%, for the two joins together.** The bound above
        // said ">= 0.39" for q||k||v from a worse model and the row model says
        // 0.46 -- the first estimate in this file to be reproduced by an
        // independent route. And `gate||up` is no longer break-even: it was
        // never about bytes, it is about deleting a 7.8 us fixed cost.
        //
        // Both are now measured facts about the kernel, so the remaining work
        // is entirely in the model crate -- see the blockers below.
        //
        // ## AND 0.68 IS THE KERNEL'S NUMBER, NOT THE PIPELINE'S. HALVE IT.
        //
        // This figure has been quoted as the plan's largest remaining item for
        // four sittings and it is the wrong one to rank with, for two reasons
        // that are both already written down ABOVE this heading and were never
        // carried down into the arithmetic here.
        //
        // **It does not pay for `split_qkv`.** 54.6 -> 38.0 us is three qmv
        // fires becoming one. The pipeline needs the packed bank taken apart
        // again before `rms_rope` and `kv_append_paged` can read it, and
        // `model_dsl::metal`'s `split_qkv` is a `launch_with_params`, not a
        // view -- so it is three fires becoming TWO, and "AND THE JOIN SAVES
        // ONE LAUNCH A LAYER, NOT TWO" above prices the returned half at 0.17
        // ms. `gate||up` has no split to pay and keeps its 0.22.
        //
        // **And the token it is a percentage OF has moved.** 0.68 ms was 9%
        // of the 7.7 ms token it was measured against. The decode is 7.49 ms
        // now, and the honest figure -- 0.22 + 0.22, or the 0.17 + 0.17 that
        // the same section reaches from the launch floor instead of the row
        // model -- is 0.34 to 0.45 ms, which is 4.5% to 6%.
        //
        // Two independent routes to the same halving, so the range is real
        // and its top is 6%. That does not make the joins worthless; it makes
        // them an ordinary launch-floor row, the size of two or three of the
        // norm folds below, bought at the price of a change spanning the model
        // crate, the shared forward text and a kernel. Rank them there. The
        // section above already said so in prose -- "they are ordinary
        // launch-floor savings like the norm rows, and they should be planned
        // as such" -- and this is that sentence done as arithmetic, because a
        // number left standing in a table outranks a paragraph every time.
        //
        // The one thing above this heading that survives untouched is the
        // `rope k + kv_append` fold, because it deletes a whole kernel and
        // adds nothing: 0.12 ms, all of it kept.
        //
        // Nothing below is deleted, because the ordering of the REDUNDANCY
        // column is still right and it is the reasoning that matters. Read it
        // with 0.71 us substituted for 4.6 and the arithmetic redone.
        //
        // # WHICH MAKES THIS LIST THE WHOLE REMAINING PLAN, SO RANK IT
        //
        // At 4.6 us a launch, and split by whether the fold does REDUNDANT
        // WORK -- a fusion that only moves where a value is written is free,
        // and one that folds a REDUCTION into a wide consumer pays for it once
        // per workgroup of that consumer:
        //
        // ```text
        //   fold                          launches   saved    redundancy
        //   rope k + kv_append (k half)         28   0.13 ms   none: a store target
        //   v projection + kv_append (v)         0   0.00 ms   none, but see below
        //   q||k||v                             56   0.26 ms   none; needs the model crate
        //   gate||up                            28   0.13 ms   none; needs the model crate
        //   silu_mul into down                  28   0.13 ms   ~25% more ALU in `down`
        //   rms_single_row into its qmv         56   0.26 ms   ~26% more ALU in five fires
        // ```
        //
        // THE FIRST TWO ARE ONE ITEM, not two. `kv_append_paged` appends k AND
        // v in one launch, so folding only the k half into `rms_rope` leaves
        // the launch standing for v. The pair has to move together: `rms_rope`
        // writes its roped k straight into the paged pool, and the v
        // projection's `affine_qmv_fast` writes its output there instead of
        // into the arena. Both are changes of DESTINATION -- no value is
        // recomputed, no arithmetic is added, and the bf16 rounding the
        // append does today is the rounding the store already does. That makes
        // it the cheapest 0.13 ms on this list and the one to do first, at the
        // cost of two kernel variants that take a page table.
        //
        // THE LAST TWO ARE NOT FREE and should be priced before they are
        // written. `rms_single_row` is a reduction over the whole row; five
        // qmv fires a layer read its output, and each of their 256-to-768
        // workgroups would have to redo the sum. Against a 4.6 us launch the
        // redundant reduction is roughly 0.06 ms of the 0.26 ms saved -- still
        // positive, but a quarter of the prize, and it is the kind of estimate
        // that has been wrong twice in this file today. Measure it with a
        // throwaway variant before committing to the fixtures.
        //
        // The other 0.17 ms is reachable and is a separate, later change:
        // nothing downstream actually needs three BUFFERS, it needs three
        // RANGES. `rms_rope` already takes q and k as two operands and
        // `kv_append_paged` takes k and v; if an operand could be bound as a
        // sub-extent of the fused rectangle the split kernel would have no
        // reason to exist. `binding::extent` binds `rows * width * bytes` from
        // a rectangle's start, so the missing piece is an offset, and the
        // missing piece before that is a DSL value that is a view rather than
        // a launch. Worth stating here because a reader who lands on
        // `split_qkv` will assume the deinterleave is inherent, and it is not.
        //
        // They are the cheapest arithmetic in the model -- a 16- and an
        // 8-workgroup normalise -- fired as two separate rectangles a layer
        // immediately before the rope that reads them, which is two more
        // rectangles a layer than the arithmetic deserves. That makes the
        // whole norm question worth 112 launches rather than 56, and it makes
        // rope the natural place to put them, since rope already runs twice a
        // layer over exactly those two tensors.
        //
        // Also checked, so nobody re-checks it: the attention's split and
        // merge are NOT a collapsible pair. One launch instead of two is 26%
        // slower, and `kernels-wgpu::attn`'s `splits` carries the sweep and
        // the reason.
        //
        // 99.8 tok/s becomes about 135. That is the honest size of the whole
        // fusion program on this machine, and no single one of them is worth
        // an afternoon on its own -- which is the thing the list above could
        // not say before there was a bench.
        //
        // **AND THAT ARITHMETIC IS TOO GENEROUS.** The ~13 us a removed
        // rectangle above comes from dividing a decode by its launch count,
        // which assumes every launch costs the same. Priced MARGINALLY
        // instead -- duplicate a routine's fire and read the difference --
        // a launch that computes little costs **6.0 us** (56 extra per-head
        // norms: 9.828 -> 10.162 ms) while a `qmv` costs **22.7** (141
        // extra: -> 13.025 ms), and the gap is this model streaming 2.5 MiB
        // of weights per qmv, which is 13.8 us of irreducible bandwidth.
        // The table is in `what_a_decode_costs_at_length`.
        //
        // At 6 us, the 196 launches above are 1.18 ms and the whole program
        // lands near **115 tok/s**, not 135. It remains the largest lever
        // this backend has and it is still worth doing; it is not a route to
        // llama.cpp's 259, and nothing in this list is.
        //
        // **AND IT IS NO LONGER THE LARGEST LEVER.** Two reduction ladders
        // came out of `sdpa_paged` and `qmv` -- no launch was deleted -- and
        // the token went 9.64 -> 7.730 ms, 19.5%, which is more than this
        // whole list. So the denominator under every figure above has moved
        // and the ranking has changed: at 7.730 ms the two joins are the
        // MEASURED 0.68 ms of the section below, ~9% of a token, landing near
        // 141 tok/s. The flat 6 us was also the wrong shape, because a join
        // deletes fires and not rows, so it collects the per-fire term
        // exactly -- see the byte-keyed cost model in `qmv.wgsl`.
        //
        // And not this threshold. Moving it only trades one bad regime for
        // the other: below 4 rows a step is qmv with three quarters of its
        // slots idle, and at 4 and above it is a GEMM tiled 32x32 with seven
        // eighths of a tile idle. A served deployment lives in the gap.
        let prefill = frame.rows() > 1;
        let plan = if prefill { self.prefill } else { self.plan };
        // Derived on the first step of a shape and kept: `lower` is a pure
        // function of the plan, the rows and the fire flag, and this driver
        // was paying 0.765 ms of every decode to recompute a constant. See
        // `lowering::cached`.
        let low = held
            .lowerings
            .get(
                plan,
                crate::lowering::cached::Shape {
                    prefill,
                    rows: rows.clone(),
                },
                LowerFire {
                    captures_across_splits: false,
                },
            )
            .map_err(Unstepped::Uncovered)?;

        held.pool.stage(device, &frame).map_err(Unstepped::Failed)?;
        // The table and the lowering say the same thing again: `Pool::stage`
        // writes `SamplingIndices` from the frame, and the frame's seriation
        // above is what the lowering was handed. The identity table that used
        // to be written over it belonged to the every-row workaround.
        // The tokens, put where the FRAME says each row is rather than in the
        // order the turns arrived. A step that wrote them in turn order would
        // feed every conversation somebody else's token whenever the seriation
        // reordered anything, and the answer would still look like text.
        let ids = place(tokens, &frame.request_of_token);
        held.pool
            .state(device, crate::binding::FireTable::TokenIds, &ids)
            .map_err(Unstepped::Failed)?;

        // FROM THE POOL. The allocation is handed back when `held` drops at
        // the end of this function, which is what keeps the bind-group cache
        // keyed on it warm from one token to the next.
        let held_arena = device
            .arena(low.arena_bytes as u64)
            .map_err(Unstepped::Failed)?;
        let arena = held_arena.buffer();
        // The plan's runtime streams, so a text's `positions` binds the
        // table this step just staged rather than the seam stand-in. Per
        // step because the decode and prefill plans each mint their own ids.
        let streams = crate::runtime::Streams::of(plan);
        let model = Model {
            weights: held.weights,
            pool: held.pool,
            recurrent: held.recurrent,
            runtime: &streams,
        };
        let ran = fire(
            device,
            pipelines,
            modules,
            low,
            Fire {
                arena: crate::binding::Arena {
                    buffer: arena,
                    bytes: low.arena_bytes as u64,
                },
                resolver: &model,
                geometry: self.geometry,
                tier: self.tier,
                one_at_a_time: self.one_at_a_time,
                prefix: self.prefix,
            },
        );
        // The arena is handed back to the device's pool when this function
        // returns, on both paths, which is what `driver-vulkan`'s explicit
        // `device.free(arena)` on both sides of the `?` is standing in for.
        // Nothing here can leak it, and nothing here can release it early
        // either -- the read below still names it.
        let (fired, readout) = ran.map_err(Unstepped::Unfired)?;
        let logits = logits(device, arena, low, &readout).map_err(Unstepped::Unread)?;
        // AFTER the fire and before the arena is dropped, which is the only
        // window it exists in.
        let kept = if self.keep_arena {
            device
                .read_at(arena, 0, low.arena_bytes as u64)
                .map_err(Unstepped::Failed)?
        } else {
            Vec::new()
        };
        Ok(Step {
            logits,
            arena: kept,
            fired,
            rows: frame.rows(),
            positions: frame.positions.clone(),
            readout_of: last_row_of(tokens.len(), &frame.request_of_token)
                .into_iter()
                .map(|row| logit_row_of(&frame, row))
                .collect(),
            readouts_of: readouts_of(tokens.len(), &frame),
            pipelines: pipelines.built(),
        })
    }
}

/// Every row each turn reads out, by turn order.
///
/// The frame's `sampling_indptr` is the CSR and `sampling_indices` the rows,
/// both already in fire-row placement, so this is a regrouping and not a
/// derivation -- a second derivation would be a second chance to disagree
/// with `Frame::of` about whose rows are whose.
///
/// A turn the frame has no boundary for gets an empty span rather than a
/// guess. It cannot happen for a turn a step grew.
#[must_use]
fn readouts_of(turns: usize, frame: &Frame) -> Vec<Vec<usize>> {
    (0..turns)
        .map(|t| {
            match (
                frame.sampling_indptr.get(t),
                frame.sampling_indptr.get(t + 1),
            ) {
                // The CSR POSITIONS and not the row values they hold: the
                // gather compacts the stream into one output row per sampled
                // row, in `sampling_indices` order, so position `j` of that
                // list is logit row `j`.
                (Some(&lo), Some(&hi)) if hi >= lo => (lo as usize..hi as usize)
                    .filter(|&j| j < frame.sampling_indices.len())
                    .collect(),
                _ => Vec::new(),
            }
        })
        .collect()
}

/// Which LOGIT row a fire row's distribution landed in.
///
/// The epilogue gather compacts the stream into one row per sampled row, in
/// `sampling_indices` order, so a fire row's distribution is at that row's
/// POSITION in the list. A row the frame does not sample has no distribution
/// and gets [`NO_ROW`], which `frames::serve` already refuses by name.
#[must_use]
fn logit_row_of(frame: &Frame, row: usize) -> usize {
    if row == NO_ROW {
        return NO_ROW;
    }
    frame
        .sampling_indices
        .iter()
        .position(|&at| at as usize == row)
        .unwrap_or(NO_ROW)
}

/// A row index no fire has: what a turn that contributed no rows gets, so
/// that reading it is a refusal rather than another turn's answer.
pub const NO_ROW: usize = usize::MAX;

/// The last fire row each turn contributed, or [`NO_ROW`] for a turn that
/// contributed none.
///
/// Not row zero for an ownerless turn: `frames::serve` reads this as the
/// fallback when a request's read-out span is empty, so a zero would hand
/// that turn the FIRST turn's distribution and return its token. [`NO_ROW`]
/// is not a row, so the bound check in `serve` refuses it by name. No real
/// plan reaches it — a probe over the curated suite counted zero ownerless
/// turns.
#[must_use]
fn last_row_of(turns: usize, request_of_token: &[u32]) -> Vec<usize> {
    let mut last = vec![NO_ROW; turns];
    for (t, which) in request_of_token.iter().enumerate() {
        if let Some(slot) = last.get_mut(*which as usize) {
            *slot = t;
        }
    }
    last
}

/// Lay the turns' tokens out in the frame's row order.
///
/// Separate and pure because the device loop cannot reach it: a version that
/// gave every row its turn's FIRST token passes every one-token step
/// unchanged, so the claim has to be stated where a multi-token turn can be.
///
/// `request_of_token` is the frame's, so a row belongs to `turns[which]`; the
/// running count per turn is what makes the second row of a conversation its
/// second token. A row past a turn's tokens gets zero rather than panicking,
/// because aborting a server on an arithmetic slip it could survive is worse.
#[must_use]
fn place(tokens: &[&[u32]], request_of_token: &[u32]) -> Vec<u32> {
    let mut taken = vec![0usize; tokens.len()];
    let mut ids = vec![0u32; request_of_token.len()];
    for (id, which) in ids.iter_mut().zip(request_of_token) {
        let which = *which as usize;
        let Some(turn) = tokens.get(which) else {
            continue;
        };
        *id = turn.get(taken[which]).copied().unwrap_or(0);
        taken[which] += 1;
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(who: u64, tokens: &[u32]) -> Turn {
        Turn {
            who,
            tokens: tokens.to_vec(),
        }
    }

    /// What `place` takes, since it serves the engine's path too.
    ///
    /// The engine's frames carry no [`Turn`] — their requests are already the
    /// scheduler's — so the shared body speaks in token slices.
    fn tokens_of(turns: &[Turn]) -> Vec<&[u32]> {
        turns.iter().map(|t| t.tokens.as_slice()).collect()
    }

    /// A conversation's second row gets its second token.
    #[test]
    fn each_row_gets_its_own_turns_next_token() {
        let turns = [turn(1, &[11, 12, 13]), turn(2, &[21, 22])];
        assert_eq!(
            place(&tokens_of(&turns), &[0, 0, 0, 1, 1]),
            vec![11, 12, 13, 21, 22],
            "in order, the placement is the concatenation"
        );
    }

    /// And it still does when the frame interleaves them.
    ///
    /// The defect this prevents: a step writing tokens in TURN order feeds every
    /// conversation somebody else's token whenever the seriation reorders
    /// anything, and the answer still looks like text.
    #[test]
    fn an_interleaved_frame_does_not_hand_a_turn_another_turns_token() {
        let turns = [turn(1, &[11, 12, 13]), turn(2, &[21, 22])];
        assert_eq!(
            place(&tokens_of(&turns), &[1, 0, 1, 0, 0]),
            vec![21, 11, 22, 12, 13]
        );
    }

    /// A turn's answer is its LAST row, not its first.
    ///
    /// The defect: a prefill of four tokens produces four distributions and only
    /// the fourth has seen the whole prompt. A caller handed the first samples
    /// from a model that read one token.
    #[test]
    fn a_turns_readout_is_the_last_row_it_contributed() {
        assert_eq!(last_row_of(2, &[0, 0, 0, 1]), vec![2, 3]);
        assert_eq!(
            last_row_of(2, &[1, 0, 1, 0, 0]),
            vec![4, 2],
            "and it follows the frame's order, not the turns'"
        );
        // A turn that contributed no rows. Answering `0` handed it the FIRST
        // turn's row, which `frames::serve` reads as its fallback.
        assert_eq!(last_row_of(3, &[0, 1]), vec![0, 1, NO_ROW]);
    }

    /// A row past its turn's tokens is zero and not a panic.
    #[test]
    fn a_row_with_no_token_left_is_zero() {
        let turns = [turn(1, &[11])];
        assert_eq!(place(&tokens_of(&turns), &[0, 0, 0]), vec![11, 0, 0]);
        assert_eq!(
            place(&tokens_of(&turns), &[0, 7]),
            vec![11, 0],
            "and so is an unknown turn"
        );
    }
}
