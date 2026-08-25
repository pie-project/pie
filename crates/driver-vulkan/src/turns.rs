//! What one conversation wants of one fire, and what a fire gave it back.
//!
//! # `Serving` STOOD HERE, AND SO DID THE STEP LOOP
//!
//! This module was the server: `Serving::over` held the turns against the
//! [`Book`], staged a [`Frame`](crate::resources::Frame), lowered it, carved an
//! arena, fired, read the logits back and handed out one distribution per turn.
//! It went with `model_compiler::lower`, and what is left is the VOCABULARY it
//! spoke in -- a [`Turn`], the things a deployment [`Held`] between fires, and
//! the [`Step`] a fire produced -- plus the four row arithmetics no fire can
//! get wrong twice.
//!
//! FOUR FINDINGS THE LOOP WAS BUILT ON, and each was reachable before it
//! existed. They are written down here because the next loop owes all four and
//! none of them is recoverable from the types that survive:
//!
//! 1. A conversation's pages must be its own from its first fire to its last.
//!    [`Book`] answers that; a caller that grew a request by hand did not.
//! 2. A conversation's positions must CONTINUE. Every test in this crate
//!    before the loop wrote `(0..n)` and re-fired from scratch, which is a
//!    prefill repeated, not a decode.
//! 3. A deployment needs BOTH plans. A text traced at `FireClass::Prefill`
//!    states tiled GEMMs where the same text traced at `FireClass::Decode`
//!    states matrix-vector products, and `Serving` held ONE plan until that was
//!    measured -- so a prompt was answered one row at a time by the decode
//!    kernel. The divergence starts at SIXTEEN rows, the tile height; below
//!    that the two lower identically. [`crate::walk::lane::Baked::lane`] is
//!    where a `Program` walk makes the same choice, and it makes it per fire
//!    class off the same plan, which is why it cannot make this mistake.
//! 4. The lowering was per-fire and the pipelines are not. A step that rebuilt
//!    its pipelines would be correct and unusably slow, and nothing measured
//!    it -- [`Fired::parsed`] is the number that does.
//!
//! # What a step is not
//!
//! It does not sample. [`Step::logits`] is a distribution and the token that
//! comes back next fire is the caller's to choose -- which is the same place
//! `driver-metal` stops, and not an accident: the temperature, the top-k and
//! the seed belong to a request, and a driver that owned them would own the
//! request too.
//!
//! It did not keep the arena. Each step allocated one and freed it, because the
//! arena's size depends on the row count and the row count changes every step.
//! `turns::Arenas` was the cache that fixed it and the paragraph naming its
//! numbers is below, where the module stood.

use crate::pages::Book;
use crate::resources::{Pool, Request, Unstageable, Weights};
use crate::serve::{Fired, Logits};

/// What one conversation wants out of one fire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    /// The conversation, by the id its [`Book`] seat is under.
    pub who: u64,
    /// The tokens to append, in order.
    ///
    /// One for a decode, many for a prefill. The distinction is not a mode
    /// here -- both produce a lowering from the same code path, which is what
    /// would make a mixed batch expressible at all.
    ///
    /// A prefill of many tokens and a decode of one are the same code path,
    /// and a step can mix them -- which is what a server does and what
    /// nothing in this crate could express until `Serving::step` forced every
    /// row to sample. See there for why that is needed and what it costs.
    ///
    /// The consequence a caller must not miss: a turn of four tokens produces
    /// FOUR distributions, and only the last has seen the whole prompt.
    /// [`Step::readout_of`] says which one that is.
    pub tokens: Vec<u32>,
}

/// What a deployment keeps between fires.
///
/// One struct rather than three arguments, and the grouping is the point:
/// these are exactly the things a step MUTATES or reads that outlive it, and
/// a caller holding them apart can pass a book from one deployment and a pool
/// from another. They are borrows because a server owns them -- the pipeline
/// cache is deliberately NOT here, since it belongs to the device rather than
/// to the model and two deployments on one device should share it.
pub struct Held<'a> {
    /// Who owns which page.
    pub book: &'a mut Book,
    /// The cache and the per-fire tables.
    pub pool: &'a mut Pool,
    /// The checkpoint.
    pub weights: &'a Weights,
    // `lowerings: &mut Lowerings`, `arenas: &mut Arenas` and `plans: &mut
    // replay::Plans` STOOD HERE -- the three things a deployment kept between
    // fires that were not the book, the pool or the weights. All three were
    // keyed on a `Lowered` and all three are deleted with it.
    //
    // `plans` was here rather than beside the pipelines for the reason the
    // pipelines are NOT here: a pipeline is a property of the DEVICE and two
    // deployments on one device share them, while a recorded plan is a
    // property of one deployment's lowering, arena and pool, so two
    // deployments sharing one would take turns invalidating it. Whatever
    // caches a walk's recording owes the same placement.
}

// `Arenas`, `Lowerings`, `Unstepped`, `Serving`, `state_of`, `arena_for`,
// `partial_tile`, `slice` and `join` STOOD HERE -- 1,054 lines, the whole of
// the step loop, and every one of them was about a `Lowered`.
//
// `Serving::over` WAS THE STEP: hold the turns against the book, stage a
// `Frame`, lower it (or find the lowering), carve an arena, fire, read the
// logits back and hand out one distribution per turn. `Lowerings` cached the
// derived plans by row shape -- keyed on a `Vec<Row>` as tall as the fire,
// because hashing the key is proportional to the thing being avoided while
// comparing it stops at the first difference. `state_of` hashed the buffers a
// `Model` answered with, so `replay` could tell a fire whose inputs had moved
// from one whose had not.
//
// FOUR OF THEM ARE FINDINGS AND ARE WRITTEN DOWN RATHER THAN DELETED:
//
// * `Arenas` -- a decode step allocated a device buffer for the plan's arena,
//   zeroed it, fired into it, read the logits out and freed it EVERY TOKEN,
//   and the size is a property of the lowering alone, so a conversation that
//   decodes for a thousand tokens asked for the same 326 KB a thousand times.
//   Measured per decode step on a 4090: `vkCreateBuffer` + `vkAllocateMemory`
//   0.132 ms, `vkDestroyBuffer` + `vkFreeMemory` 0.089 ms. A fifth of a
//   millisecond a token to hand the same allocation back and ask for it again.
//   It kept ONE buffer with a 16 MiB ceiling, because a shell decodes at one
//   row shape for as long as a conversation runs and a prefill's arena is
//   `rows * vocab * 4` -- 233 MB for 384 rows, a quarter of a gigabyte held
//   for a fire that has finished. Whatever carves the walk's arena owes the
//   same cache and the same ceiling.
//
// * `slice` and `join` -- the TILED PREFILL. A prompt longer than the gemm's
//   row tile was cut into whole-tile pieces, fired one at a time and rejoined,
//   because `geometry::eval` refuses a fire whose rows are not a whole number
//   of tiles by name (`PartialTile`), and `partial_tile` is what read that
//   refusal back out of a `Unstepped` to decide whether to re-cut. A piece
//   carried its own rows, its own mask and its own write targets, and
//   `a_piece_of_a_request_carries_its_rows_mask_and_write_targets` existed so
//   that the next field added to a `Request` was refused by a test rather than
//   dropped by the helper. The walk states its own rows; if a tile rule ever
//   refuses one, this is the shape of the answer.
//
// * `Unstepped` -- the two refusals kept APART rather than flattened into one
//   string, because the caller's next move differs completely between "the
//   cache is full" (evict, or queue this conversation) and "this plan names a
//   kernel you did not give me" (a build problem, and no amount of waiting
//   fixes it). `pages::Unhoused` and `resources::Unstageable` both still
//   exist and both still say exactly that; what is gone is the enum that
//   joined them to a `serve::Unfired`.

/// What one step did.
#[derive(Debug)]
pub struct Step {
    /// One distribution per turn, in the order the turns were given.
    ///
    /// **Not the order the rows were fired in.** A frame reorders rows by its
    /// seriation, so a caller reading `logits.row(i)` for turn `i` is relying
    /// on the readout being gathered back into turn order -- which is what
    /// `sampling_indices` is for.
    pub logits: Logits,
    /// The rectangles this step ran and how many submissions it took.
    pub fired: Fired,
    /// The rows this step fired over.
    pub rows: usize,
    /// Which row of [`Step::logits`] is each turn's answer.
    ///
    /// Needed rather than obvious, and for two reasons at once. Every row
    /// samples -- see `Serving::step` for why -- so a prefill of four tokens
    /// produces FOUR distributions of which a caller wants the last. And the
    /// frame reorders rows, so even for one-token turns the row order is not
    /// the turn order.
    ///
    /// `logits.row(step.readout_of[i])` is turn `i`'s distribution.
    ///
    /// A turn naming SEVERAL readout rows has all of them in
    /// [`Step::readouts_of`]; this stays its LAST, which is what a caller
    /// wanting one answer means and what every decode wants.
    pub readout_of: Vec<usize>,
    /// Every row turn `i` reads out, in WHOLE-fire row order.
    ///
    /// A decode names one and this is a one-element vector; a speculative
    /// verifier names one per drafted token.
    pub readouts_of: Vec<Vec<usize>>,
    /// The position of each row, in the order the rows were fired.
    ///
    /// Here so a caller can read the pool's `Positions` table back and check
    /// that the tables the DEVICE holds are this step's. Without it the only
    /// evidence that a step restages is that it calls `stage`, and a control
    /// that staged only on a conversation's first step did not fire.
    pub positions: Vec<u32>,
    /// Pipelines the cache held after this step.
    ///
    /// Here so that "the second step builds nothing" is a number a test can
    /// compare rather than a claim about timing.
    pub pipelines: usize,
}

/// Every row each request reads out, in WHOLE-fire row numbering.
///
/// The same walk `Frame::of` does -- requests in order, a base per request --
/// so the two cannot disagree about placement without disagreeing about the
/// row order itself.
///
/// # Why the rows are bounds-checked HERE, and not left to `Frame::of`
///
/// `Request::read` answers in the request's OWN numbering, so `base + row`
/// places an out-of-range row inside the NEXT request -- the same silent
/// cross-request read this crate chased through `envelope` and `frames`. Two
/// requests of two rows each, the first naming row 2, gives `[[2], [3]]`:
/// request 0's answer is request 1's first row, and nothing faults.
///
/// This was written first as a comment claiming no guard was needed, because
/// `over` only reaches this after `tiled`, and `tiled` builds a `Frame` per
/// sub-fire where `Frame::of` refuses the row by name with `NotItsRow`. That
/// is true of the WHOLE-fire path and false of the split one: `slice` keeps
/// only the samples landing in its own piece (`row >= from && row < to`), so
/// an out-of-range row is FILTERED OUT of every piece, each piece falls back
/// to "the last row", no `Frame::of` ever sees it -- and `over` then walks
/// the ORIGINAL requests here. The guard upstream cannot fire precisely when
/// this arithmetic is reached with a bad row.
///
/// So the check sits with the arithmetic it protects, where it is reachable
/// by a test that needs no device.
pub fn spans_of(requests: &[Request]) -> Result<Vec<Vec<usize>>, Unstageable> {
    let mut out = Vec::with_capacity(requests.len());
    let mut base = 0usize;
    for (r, request) in requests.iter().enumerate() {
        let rows = request.positions.len();
        let mut span = Vec::new();
        for row in request.read() {
            if row as usize >= rows {
                return Err(Unstageable::NotItsRow {
                    request: r,
                    row,
                    rows,
                });
            }
            span.push(base + row as usize);
        }
        out.push(span);
        base += rows;
    }
    Ok(out)
}

/// A row index no fire has: what a request that contributed no rows gets, so
/// that reading it is a refusal rather than another request's answer.
pub const NO_ROW: usize = usize::MAX;

/// The last fire row each request contributed, or [`NO_ROW`] for one that
/// contributed none.
///
/// # Why an ownerless request is not row zero
///
/// It used to be: the vector started at `0` and only requests owning a token
/// overwrote their slot, so a request with no rows kept row 0 -- the FIRST
/// request's. `frames::serve` reads this as the fallback when a read-out span
/// is empty, so such a request would have been handed another conversation's
/// distribution and returned its token. That is the `driver-metal` defect
/// `member_requests` exists to prevent, reached through a different door.
///
/// `NO_ROW` is not a row, so the bound check in `serve` refuses it by name.
/// No new code path, because none is warranted: a probe over the whole
/// curated suite counted ZERO ownerless requests, and that is a property of
/// what the ENGINE emits, so it holds for this driver too.
#[must_use]
pub fn last_row_of(requests: usize, request_of_token: &[u32]) -> Vec<usize> {
    let mut last = vec![NO_ROW; requests];
    for (t, which) in request_of_token.iter().enumerate() {
        if let Some(slot) = last.get_mut(*which as usize) {
            *slot = t;
        }
    }
    last
}

/// Lay the turns' tokens out in the frame's row order.
///
/// Separate and pure because the loop cannot reach it: every turn a step can
/// fire today is one token long -- a longer one is refused, for the arena
/// overrun [`Turn::tokens`] records -- so a version that gave every row its
/// turn's FIRST token passes the device test unchanged. That control did not
/// fire, so the claim moved to where it can be made.
///
/// `request_of_token` is the frame's, so a row belongs to `turns[which]`; the
/// running count per turn is what makes the second row of a conversation its
/// second token rather than its first.
///
/// A row past a turn's tokens gets zero rather than panicking. It cannot
/// happen -- the frame is built from the same turns -- and a driver that
/// aborted a server on an arithmetic slip it could survive is worse than one
/// that fires a wrong token.
#[must_use]
pub fn place(tokens: &[&[u32]], request_of_token: &[u32]) -> Vec<u32> {
    let mut taken = vec![0usize; tokens.len()];
    let mut ids = vec![0u32; request_of_token.len()];
    for (id, which) in ids.iter_mut().zip(request_of_token) {
        let which = *which as usize;
        let Some(mine) = tokens.get(which) else {
            continue;
        };
        *id = mine.get(taken[which]).copied().unwrap_or(0);
        taken[which] += 1;
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read-out row past its own request is refused, not renumbered.
    ///
    /// Without the check `spans_of` answers `[[2], [3]]` for this: request 0
    /// reading fire row 2, which is request 1's FIRST row. That is a request
    /// answering out of another conversation's distribution, and it is silent
    /// -- the row exists, the read succeeds, the text is fluent.
    #[test]
    fn a_read_out_row_past_its_own_request_is_refused_before_it_is_renumbered() {
        let mut mine = Request::of(vec![0, 1], vec![0]);
        mine.samples = vec![2];
        let theirs = Request::of(vec![0, 1], vec![1]);
        assert_eq!(
            spans_of(&[mine, theirs]),
            Err(Unstageable::NotItsRow {
                request: 0,
                row: 2,
                rows: 2
            }),
            "the row belongs to nobody, so it is named rather than placed"
        );
    }

    /// And the rows that ARE its own still come back in fire numbering.
    #[test]
    fn every_request_s_own_rows_are_numbered_from_its_start_in_the_fire() {
        let mut mine = Request::of(vec![0, 1, 2], vec![0]);
        mine.samples = vec![0, 2];
        let theirs = Request::of(vec![0, 1], vec![1]);
        assert_eq!(
            spans_of(&[mine, theirs]),
            Ok(vec![vec![0, 2], vec![4]]),
            "request 1 spans fire rows 3..5, so its own last row is 4"
        );
    }

    // `a_piece_of_a_request_carries_its_rows_mask_and_write_targets` and
    // `a_split_fire_filters_the_bad_row_away_instead_of_refusing_it` STOOD
    // HERE, and both were about `slice`, which went with the tiled prefill.
    //
    // The first is the reason `slice` was a named function at all: it rebuilt
    // each piece with `Request::of`, which knows only positions and pages, so
    // every field added to a `Request` since was one the walk could drop in
    // SILENCE -- and two already had been. A traced request came out of a
    // split fire looking scheduler-placed, which put beam search's two lanes
    // back on the page-sharing refusal; and its mask rows came out WHOLE, read
    // against the piece's own row numbering, which is another row's
    // allow-bytes. The test existed so the next field added was refused by a
    // test rather than dropped by a helper, and whoever splits a fire again
    // owes it back.
    //
    // The second is the measurement behind [`spans_of`]'s guard, and the guard
    // stays without it: `slice` kept only the samples landing in its own piece,
    // so an out-of-range row was FILTERED OUT of every piece rather than
    // refused, every piece fell back to "the last row", and no `Frame::of` ever
    // saw it. That is why the bounds check sits with the arithmetic in
    // `spans_of` rather than upstream -- the upstream guard cannot fire
    // precisely when the arithmetic is reached with a bad row.

    fn turn(who: u64, tokens: &[u32]) -> Turn {
        Turn {
            who,
            tokens: tokens.to_vec(),
        }
    }

    /// A conversation's second row gets its second token.
    #[test]
    fn each_row_gets_its_own_turns_next_token() {
        let turns = [turn(1, &[11, 12, 13]), turn(2, &[21, 22])];
        assert_eq!(
            place(&[&turns[0].tokens, &turns[1].tokens], &[0, 0, 0, 1, 1]),
            vec![11, 12, 13, 21, 22],
            "in order, the placement is the concatenation"
        );
    }

    /// And it still does when the frame interleaves them.
    ///
    /// The defect this prevents: a step writing tokens in TURN order feeds
    /// every conversation somebody else's token whenever the seriation
    /// reorders anything, and the answer still looks like text.
    #[test]
    fn an_interleaved_frame_does_not_hand_a_turn_another_turns_token() {
        let turns = [turn(1, &[11, 12, 13]), turn(2, &[21, 22])];
        assert_eq!(
            place(&[&turns[0].tokens, &turns[1].tokens], &[1, 0, 1, 0, 0]),
            vec![21, 11, 22, 12, 13]
        );
    }

    /// A turn's answer is its LAST row, not its first.
    ///
    /// The defect: a prefill of four tokens produces four distributions and
    /// only the fourth has seen the whole prompt. A caller handed the first
    /// samples from a model that read one token.
    #[test]
    fn a_turns_readout_is_the_last_row_it_contributed() {
        assert_eq!(last_row_of(2, &[0, 0, 0, 1]), vec![2, 3]);
        assert_eq!(
            last_row_of(2, &[1, 0, 1, 0, 0]),
            vec![4, 2],
            "and it follows the frame's order, not the turns'"
        );
        // A request that contributed no rows. This used to answer `0` -- the
        // FIRST request's row -- and `frames::serve` reads exactly this as
        // the fallback when a read-out span is empty, so it would have
        // returned another conversation's token. The test named the case and
        // pinned the wrong answer.
        assert_eq!(last_row_of(3, &[0, 1]), vec![0, 1, NO_ROW]);
    }

    /// A row past its turn's tokens is zero and not a panic.
    #[test]
    fn a_row_with_no_token_left_is_zero() {
        let turns = [turn(1, &[11])];
        assert_eq!(place(&[&turns[0].tokens], &[0, 0, 0]), vec![11, 0, 0]);
        assert_eq!(
            place(&[&turns[0].tokens], &[0, 7]),
            vec![11, 0],
            "and so is an unknown turn"
        );
    }
}
