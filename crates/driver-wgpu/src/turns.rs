//! One fire after another, over the same cache.
//!
//! Everything below this is per-fire: [`fire`] runs a lowering once, [`Frame`]
//! describes one moment, [`logits`] reads one answer. A server is none of those.
//! It is the same pool, the same weights and the same pipeline cache carried
//! across thousands of fires while conversations arrive, grow and leave — and
//! the parts that can only be wrong ACROSS fires live here because there is
//! nowhere else they could live.
//!
//! There are four of them, and each was reachable before this module existed:
//!
//! 1. A conversation's pages must be its own from its first fire to its last.
//!    [`Book`] answers that; a caller that grew a request by hand did not.
//! 2. A conversation's positions must continue. A test that writes `(0..n)` and
//!    re-fires from scratch is a prefill repeated, not a decode.
//! 3. A deployment needs BOTH plans. A text traced at `FireClass::Prefill`
//!    states tiled GEMMs where the same text traced at `FireClass::Decode`
//!    states matrix-vector products, so a deployment carrying one plan answers a
//!    prompt one row at a time with the decode kernel. The divergence starts at
//!    SIXTEEN rows, the tile height; below that the two plans lower identically.
//! 4. The lowering is per-fire and the pipelines are not. A step that rebuilt
//!    its pipelines would be correct and unusably slow — and slower here than on
//!    either sibling, because `wgpu::Device::create_shader_module` runs a whole
//!    shader compiler where `vkCreateShaderModule` copies a blob.
//!
//! # What a step is not
//!
//! It does not sample. [`Step::logits`] is a distribution and the token that
//! comes back next fire is the caller's to choose: the temperature, the top-k
//! and the seed belong to a request, and a driver that owned them would own the
//! request too.
//!
//! It does not keep the arena. Each step allocates one and drops it, because
//! `Lowered::arena_bytes` depends on the row count and the row count changes
//! every step. A server that fired the same shape repeatedly would want to keep
//! the largest and reuse it; nothing here does, and the cost is one allocation
//! per fire rather than one per token.
//!
//! [`fire`]: crate::serve::fire
//! [`logits`]: crate::serve::logits
//! [`Frame`]: crate::resources::Frame

use model_compiler::lower::{Fire as LowerFire, Uncovered, lower};
use model_compiler::trace::ForwardPlan;

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
    /// One for a decode, many for a prefill. The distinction is not a mode here
    /// — both produce a lowering from the same code path, which is what makes a
    /// mixed batch expressible at all.
    ///
    /// The consequence a caller must not miss: a turn of four tokens produces
    /// FOUR distributions, and only the last has seen the whole prompt.
    /// [`Step::readout_of`] says which one that is.
    pub tokens: Vec<u32>,
}

/// What a deployment keeps between fires.
///
/// One struct rather than three arguments, and the grouping is the point: these
/// are exactly the things a step MUTATES or reads that outlive it, and a caller
/// holding them apart can pass a book from one deployment and a pool from
/// another. They are borrows because a server owns them — the pipeline cache is
/// deliberately NOT here, since it belongs to the device rather than to the
/// model and two deployments on one device should share it.
pub struct Held<'a> {
    /// Who owns which page.
    pub book: &'a mut Book,
    /// The cache and the per-fire tables.
    pub pool: &'a mut Pool,
    /// The checkpoint.
    pub weights: &'a Weights,
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
    /// Needed rather than obvious, and for two reasons at once. Every row
    /// samples — see [`Serving::step`] for why — so a prefill of four tokens
    /// produces FOUR distributions of which a caller wants the last. And the
    /// frame reorders rows, so even for one-token turns the row order is not the
    /// turn order.
    ///
    /// `logits.row(step.readout_of[i])` is turn `i`'s distribution.
    pub readout_of: Vec<usize>,
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
    /// projections, and traced at `FireClass::Prefill` states
    /// `affine_qmm_t_..._bm_16_bn_32` instead — the tiled GEMM, the largest
    /// kernel family in the tree. A deployment carrying only the decode plan
    /// therefore answers a sixty-four-token prompt with sixty-four
    /// matrix-VECTOR products, one per row. Correct, and the wrong kernel.
    ///
    /// Not an `Option` with a fallback. A fallback would make the slow path the
    /// quiet default, which is exactly the defect this field exists to stop.
    /// Tracing a plan is host-side and cheap, so a deployment that only ever
    /// decodes pays one trace it does not use.
    pub prefill: &'a ForwardPlan,
    /// The model's shape, for the launch rules that need it.
    pub geometry: Geometry,
    /// The tier every pipeline is built at.
    pub tier: Capability,
}

impl Serving<'_> {
    /// Fire one step over `turns` and read what it computed.
    ///
    /// The order is the whole content of this function, and every pair of
    /// adjacent lines below is a defect somebody could introduce:
    ///
    /// - grow before framing, or the frame describes pages the book has not
    ///   handed out;
    /// - frame before lowering, because the lowering's row count, arena size and
    ///   readout all come from the frame's seriation;
    /// - stage before firing, or the fire reads the PREVIOUS step's tables —
    ///   which is the failure that looks like a model that has forgotten the
    ///   last token;
    /// - read before the arena is dropped.
    ///
    /// # Errors
    ///
    /// [`Unstepped`]. **A refused growth leaves the book untouched**, so a
    /// caller that evicts and retries gets the same answer it would have got had
    /// it evicted first. A failure after the growth does NOT roll it back: the
    /// pages are the conversation's, the fire that was going to use them did
    /// not run, and rolling back would lose a history that is still on the
    /// device.
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
    /// The engine's scheduler already owns eviction, prefix sharing and the
    /// copy plans that move pages between conversations, and it hands a driver
    /// a page CSR naming physical pages it chose. [`Serving::step`]'s
    /// allocator is the right one for a server built on this crate alone and
    /// the WRONG one for that caller: two allocators handing out page 7 is not
    /// an error anybody sees, because attention then reads another
    /// conversation's keys and the model answers fluently.
    ///
    /// So this exists in order that [`crate::frames`] does not have to touch
    /// [`crate::pages::Book`]. It is the same fire as `step` — literally, they
    /// share a body — differing only in who decided which pages.
    ///
    /// `tokens` is parallel to `requests`: one slice of token ids per request,
    /// in the same order.
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
        // EVERY ROW SAMPLES, and this is a workaround with a name.
        //
        // A frame's own seriation marks only the rows a request reads out, so a
        // prefill of four tokens lowers to `n_requests = 1` and an arena sized
        // for ONE row of logits. The texts, though, spell their epilogue as
        // plain `OpKind::Launch` ops rather than `OpKind::LmHead`, so
        // `Lowerer::epilogue` -- which would emit them over `0..sampled` --
        // never runs, and the generic path emits them over the whole token
        // window. The head then writes `rows * vocab` into an arena holding
        // `1 * vocab`, and `binding::extent` refuses the fire.
        //
        // Not one text's quirk: the shell every text goes through reaches
        // `lm_head` directly rather than the generic `logits()` that emits
        // `OpKind::LmHead`, so nothing this suite produces reaches
        // `Lowerer::epilogue` at all and the workaround is universal.
        //
        // The cost is real and is paid knowingly: the lm head runs over every
        // token in a prefill instead of once per request, which for a long
        // prompt is most of the fire. It is not a policy choice -- it is the only
        // shape this text lowers into consistently, and it stops being needed the
        // day the shell names its epilogue.
        let mut rows = frame.seriation();
        for row in &mut rows {
            row.samples = true;
        }
        // One row is a decode, more than one is a prefill -- including a BATCH
        // of one-token turns, which is not a slip. Four rows want a tiled GEMM
        // as much as a four-token prompt does; the row count is what the kernel
        // choice actually depends on, and the name of the class is only how the
        // text spells it.
        let plan = if frame.rows() > 1 {
            self.prefill
        } else {
            self.plan
        };
        let low = lower(
            plan,
            &rows,
            LowerFire {
                captures_across_splits: false,
            },
        )
        .map_err(Unstepped::Uncovered)?;

        held.pool.stage(device, &frame).map_err(Unstepped::Failed)?;
        // AND THE TABLE MUST SAY THE SAME THING THE LOWERING WAS TOLD.
        //
        // `Pool::stage` writes `SamplingIndices` from the frame, which names
        // only the rows a request reads out -- one entry for a prefill of four.
        // The lowering above was told every row samples, so its `row_gather`
        // reads `n_requests` entries, which is four. Left alone that is a
        // sixteen-byte read of a four-byte binding.
        //
        // On this backend that is quieter than on either sibling and not louder:
        // WGSL requires the implementation to bounds-check every access against
        // the BOUND range, so the three rows past the end read as ZERO rather
        // than as whatever the allocator put after the table. Three rows of
        // zeros gathered into the hidden state is a plausible tensor, and the
        // model that produces it says something.
        let identity: Vec<u32> = (0..frame.rows() as u32).collect();
        held.pool
            .state(
                device,
                crate::binding::FireTable::SamplingIndices,
                &identity,
            )
            .map_err(Unstepped::Failed)?;
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
        };
        let ran = fire(
            device,
            pipelines,
            modules,
            &low,
            Fire {
                arena: crate::binding::Arena {
                    buffer: &arena,
                    bytes: low.arena_bytes as u64,
                },
                resolver: &model,
                geometry: self.geometry,
                tier: self.tier,
                one_at_a_time: false,
            },
        );
        // The arena is dropped when this function returns, on both paths, which
        // is what `driver-vulkan`'s explicit `device.free(arena)` on both sides
        // of the `?` is standing in for. Nothing here can leak it, and nothing
        // here can free it early either -- the read below still names it.
        let fired = ran.map_err(Unstepped::Unfired)?;
        let logits = logits(device, &arena, &low).map_err(Unstepped::Unread)?;
        Ok(Step {
            logits,
            fired,
            rows: frame.rows(),
            positions: frame.positions.clone(),
            readout_of: last_row_of(tokens.len(), &frame.request_of_token),
            pipelines: pipelines.built(),
        })
    }
}

/// The last row each turn contributes, by row order.
///
/// Every row samples, so the readout's rows ARE the fire's rows in order, and a
/// turn's answer is the distribution of its last token — the only one that has
/// seen the whole prompt.
///
/// A turn with no row gets zero. That cannot happen for a turn a step grew, and
/// answering with a row that exists beats a panic in a server loop.
#[must_use]
fn last_row_of(turns: usize, request_of_token: &[u32]) -> Vec<usize> {
    let mut last = vec![0usize; turns];
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
/// gave every row its turn's FIRST token passes every one-token step unchanged,
/// so the claim has to be made where a multi-token turn can be stated without a
/// model.
///
/// `request_of_token` is the frame's, so a row belongs to `turns[which]`; the
/// running count per turn is what makes the second row of a conversation its
/// second token rather than its first.
///
/// A row past a turn's tokens gets zero rather than panicking. It cannot happen
/// — the frame is built from the same turns — and a driver that aborted a server
/// on an arithmetic slip it could survive is worse than one that fires a wrong
/// token.
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

    /// What `place` now takes, since it serves the engine's path too.
    ///
    /// A [`Turn`] carries WHO it belongs to and the engine's frames carry no
    /// such thing — their requests are already the scheduler's — so the shared
    /// body speaks in token slices. These tests still build turns, because the
    /// property they hold is about a conversation's tokens landing on that
    /// conversation's rows.
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
        assert_eq!(last_row_of(3, &[0, 1]), vec![0, 1, 0], "a turn with no row");
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
