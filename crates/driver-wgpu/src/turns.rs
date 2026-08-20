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
        let model = Model {
            weights: held.weights,
            pool: held.pool,
            recurrent: held.recurrent,
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
