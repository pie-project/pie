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
    /// the prefill text. Measured, the GEMM costs the same for one row as
    /// for thirty-two — it computes a whole tile either way — and the
    /// crossover against the matvec is near 150 rows on both this backend
    /// and `driver-vulkan`.
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

        let arena = device
            .zeroed(low.arena_bytes as u64)
            .map_err(Unstepped::Failed)?;
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
                    buffer: &arena,
                    bytes: low.arena_bytes as u64,
                },
                resolver: &model,
                geometry: self.geometry,
                tier: self.tier,
                one_at_a_time: self.one_at_a_time,
                prefix: self.prefix,
            },
        );
        // The arena is dropped when this function returns, on both paths, which
        // is what `driver-vulkan`'s explicit `device.free(arena)` on both sides
        // of the `?` is standing in for. Nothing here can leak it, and nothing
        // here can free it early either -- the read below still names it.
        let (fired, readout) = ran.map_err(Unstepped::Unfired)?;
        let logits = logits(device, &arena, low, &readout).map_err(Unstepped::Unread)?;
        // AFTER the fire and before the arena is dropped, which is the only
        // window it exists in.
        let kept = if self.keep_arena {
            device
                .read_at(&arena, 0, low.arena_bytes as u64)
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
