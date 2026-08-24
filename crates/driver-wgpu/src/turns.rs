//! One fire after another, over the same cache: whose row is whose.
//!
//! Everything below this is per-fire; what lives here is what can only be
//! wrong ACROSS fires. A conversation's pages must be its own from its first
//! fire to its last (`crate::pages::Book` answers that) and its positions must
//! continue rather than restart -- and, at the level this module still speaks
//! at, a ROW must belong to the turn that asked for it. A row handed the wrong
//! turn's token, or a turn handed another turn's distribution, is a server
//! that answers fluently for the wrong conversation, and nothing downstream
//! can tell.
//!
//! So what is here is [`Turn`] and [`Held`] (what a caller states), [`Step`]
//! (what comes back), and the four pure row helpers -- `readouts_of`,
//! `logit_row_of`, `last_row_of`, `place` -- that decide the placement.
//! `Serving`, which lowered a plan and fired it, STOOD in the middle of this
//! file, and `Unstepped`, the ways a step refused, STOOD below it; both are
//! recorded where they were.
//!
//! A step does not sample, and does not keep the arena.

use crate::pages::Book;
use crate::resources::{Frame, Pool};
use crate::serve::{Fired, Logits};

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
    pub weights: &'a crate::resources::Weights,
    // `lowerings: &mut lowering::cached::Lowerings` STOOD HERE -- the
    // lowerings already derived, by fire shape. It was held beside the pool
    // and the book for the same reason they travel together: a lowering is the
    // graph of ONE text, and a cache from another deployment would run the
    // other model's graph over these weights. `lowering::cached` is deleted
    // with the `Lowered` it kept.
}

// `pub enum Unstepped` STOOD HERE -- why a step did not run, with every
// layer's refusal kept apart rather than flattened into one string, because
// the caller's next move differs completely between "the cache is full"
// (evict, or queue this conversation) and "this adapter cannot build that
// kernel" (a hardware fact, and no amount of waiting fixes it).
//
// Seven variants and `Serving::step` raised all seven: `Unhoused` from the
// book, `Unstageable` from the frame, `Uncovered` from the lowering, `Failed`
// from the device, `Unfired` and `Unread` from the fire, `NoSlot` for a
// request naming a recurrent slot past the end of the pool, and `Nothing` for
// a step of NO TURNS -- refused rather than answered with an empty `Logits`,
// because a lowering over zero rows is not a thing this crate ever produced
// and a server asking for one has lost track of its own queue.
//
// It went with `Serving`, and `frames::Unlaunched::Unstepped` -- the one
// variant that carried it -- went with it.

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
    /// Every row samples, so a prefill of four tokens produces four
    /// distributions, and the frame reorders rows — so the row order is not
    /// the turn order even for one-token turns.
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
    /// Empty unless the caller asked to keep it, because a fire's arena is
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
    /// A plan that says where every traced value LIVES is what makes these
    /// bytes readable: with an offset per value a caller can name the first
    /// one that is not finite, which is the difference between "somewhere in
    /// these five kernels" and a kernel. `Lowered::value_offset` was that
    /// table and is deleted; whoever lays out an arena next owes it.
    pub arena: Vec<u8>,
    /// Pipelines the cache held after this step.
    ///
    /// Here so that "the second step builds nothing" is a number a test can
    /// compare rather than a claim about timing.
    pub pipelines: usize,
}

// `struct Serving` AND ITS 1,525-LINE `impl` STOOD HERE.
//
// It was the per-step driver: given a device, a pipeline cache, a module
// store, a `Pool`, a `Book`, a `Weights` and the turns, it built the fire's
// `Frame`, lowered the right one of the two `ForwardPlan`s for that fire's row
// count, fired every rectangle through `serve::fire`, read the logits back
// with `serve::logits`, and handed each turn its own distribution.
//
// The lowering is the whole of why it is gone: `Serving` named
// `model_compiler::lower::{Fire, Uncovered}` directly and its step was
// "lower, then fire". `model_compiler::lower` is deleted, `serve::fire` and
// `serve::logits` with it, and there is no way to keep a step that is defined
// as a walk over a `Lowered`.
//
// WHAT IS KEPT, AND WHY IT IS KEPT HERE. Everything above and below this note
// is the part of a step that is about TURNS rather than about a lowering:
// [`Turn`] and [`Held`] are what a caller states, [`Step`] is what comes back,
// and the four row helpers
// below -- `readouts_of`, `logit_row_of`, `last_row_of`, `place` -- are the
// arithmetic that decides WHOSE row is whose. That arithmetic is the thing
// this module exists for and the thing that can only be wrong across fires: a
// row handed the wrong turn's token, or a turn handed another turn's
// distribution, is a server that answers fluently for the wrong conversation.
// All four are pure, all four are still tested below, and `crate::frames`
// imports [`Step`] from here.
//
// THE FOUR HELPERS BELOW ARE `pub` NOW, and were private while `Serving` was
// their one caller. They are the whole of what this module still asserts, so a
// caller assembling a step out of the parts has to be able to reach them --
// and a reader who cannot reach them cannot check a placement against them.
//
// TWO MEASURED FACTS FROM THE DELETED CODE, worth keeping because neither is
// recoverable by reading what is left:
//
// * A DEPLOYMENT NEEDS BOTH PLANS. A text traced at `FireClass::Prefill`
//   states tiled GEMMs where `FireClass::Decode` states matrix-vector
//   products, and `Serving::prefill` chose between them by the fire's row
//   count. Lowering one at the other class's rectangle is not merely slower;
//   it states the wrong kernels.
// * PIPELINES OUTLIVE THE PER-FIRE LOWERING, because
//   `wgpu::Device::create_shader_module` runs a whole shader compiler. That is
//   why the pipeline cache was a caller's object handed in rather than a step's
//   own.

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
pub fn readouts_of(turns: usize, frame: &Frame) -> Vec<Vec<usize>> {
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
pub fn logit_row_of(frame: &Frame, row: usize) -> usize {
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
pub fn last_row_of(turns: usize, request_of_token: &[u32]) -> Vec<usize> {
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
pub fn place(tokens: &[&[u32]], request_of_token: &[u32]) -> Vec<u32> {
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
