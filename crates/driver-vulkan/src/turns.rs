//! One fire after another, over the same cache.
//!
//! Everything below this is per-fire: [`serve::fire`] runs a lowering once,
//! [`Frame`] describes one moment, [`logits`] reads one answer. A server is
//! none of those. It is the same pool, the same weights and the same pipeline
//! cache carried across thousands of fires while conversations arrive, grow
//! and leave -- and the parts that can only be wrong ACROSS fires live here
//! because there is nowhere else they could live.
//!
//! There are three of them, and each was reachable before this module existed:
//!
//! 1. A conversation's pages must be its own from its first fire to its last.
//!    [`Book`] answers that; a caller that grew a request by hand did not.
//! 2. A conversation's positions must continue. Every test in this crate
//!    before this one wrote `(0..n)` and re-fired from scratch, which is a
//!    prefill repeated, not a decode.
//! 3. A deployment needs BOTH plans. `llama_like_metal` traced at
//!    `FireClass::Prefill` states tiled GEMMs where the same text traced at
//!    `FireClass::Decode` states matrix-vector products, and `Serving` held
//!    one plan until that was measured -- so a prompt was answered one row at
//!    a time by the decode kernel. The divergence starts at SIXTEEN rows, the
//!    tile height; below that the two plans lower identically.
//! 4. The lowering is per-fire and the pipelines are not. A step that rebuilt
//!    its pipelines would be correct and unusably slow, and nothing measured
//!    it.
//!
//! # What a step is not
//!
//! It does not sample. [`Step::logits`] is a distribution and the token that
//! comes back next fire is the caller's to choose -- which is the same place
//! `driver-metal` stops, and not an accident: the temperature, the top-k and
//! the seed belong to a request, and a driver that owned them would own the
//! request too.
//!
//! It does not keep the arena. Each step allocates one and frees it, because
//! `Lowered::arena_bytes` depends on the row count and the row count changes
//! every step. A server that fired the same shape repeatedly would want to
//! keep the largest and reuse it; nothing here does, and the cost is one
//! allocation per fire rather than one per token.
//!
//! [`serve::fire`]: crate::serve::fire
//! [`logits`]: crate::serve::logits
//! [`Frame`]: crate::resources::Frame

use model_compiler::lower::{Fire as LowerFire, Lowered, Row, Uncovered, lower};
use model_ir::trace::ForwardPlan;

use crate::device::{Device, Pipelines};
use crate::dispatch::Geometry;
use crate::pages::{Book, Unhoused};
use crate::resources::{Frame, Model, Pool, Request, Unstageable, Weights};
use crate::serve::{Fire, Fired, Logits, Modules, Unfired, Unread, fire_reusing, logits_of};
use kernels_vulkan::Capability;

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
    /// Lowerings already computed, by the row shape that produced them.
    pub lowerings: &'a mut Lowerings,
    /// The arena buffer, kept between steps instead of reallocated per step.
    pub arenas: &'a mut Arenas,
    /// The last fire, kept so the next one need not be planned or recorded.
    ///
    /// Here rather than beside the pipelines for the reason the pipelines are
    /// NOT here: a pipeline is a property of the device and two deployments
    /// on one device share them, while a plan is a property of one
    /// deployment's lowering, arena and pool. Two deployments sharing one of
    /// these would take turns invalidating it.
    pub plans: &'a mut crate::replay::Plans,
}

/// The scratch arena a step fires into, kept between steps.
///
/// # What it is for
///
/// A decode step allocates a device buffer for the plan's arena, zeroes it,
/// fires into it, reads the logits out and frees it -- every token. The SIZE
/// of that buffer is a property of the LOWERING and nothing else:
/// `Lowered::arena_bytes` is what `model_compiler` computed from the plan, and
/// a conversation that decodes for a thousand tokens asks for the same 326 KB
/// a thousand times.
///
/// Measured, release, `tests/hostprof.rs`, per decode step on a 4090:
/// `vkCreateBuffer` plus `vkAllocateMemory` is **0.132 ms** and the matching
/// `vkDestroyBuffer` plus `vkFreeMemory` is **0.089 ms**, against a host step
/// of 1.72 ms outside `run_all` -- a denominator since retracted as mostly
/// fence wait, see this crate's module doc; the two absolute figures are
/// phase spans and stand. A fifth of a millisecond a token to hand the
/// same allocation back and ask for it again; both rows read 0.000 ms with
/// this cache in front of them, and only the `vkCmdFillBuffer` that zeroes it
/// is left.
///
/// # Why this is sound
///
/// The buffer is returned to this cache at the point the step used to free
/// it, which is AFTER `logits_of` has copied the readout out to the host --
/// and `Device::run_all` waits on a fence, so by then the queue is done with
/// it. Nothing holds a `Bound` into it: the dispatches that named it were
/// dropped with the fire.
///
/// It is handed back out only for an EXACT byte size, and `arena_for` zeroes
/// whatever it gets before the fire sees it, exactly as it did when every
/// arena was fresh. So a step cannot observe whether its arena is new --
/// which is the property that makes this a cache and not a change of
/// behaviour.
///
/// # Why one buffer and why a ceiling
///
/// One, because a shell decodes at one row shape for as long as a
/// conversation runs, so a second slot would hold a prefill's arena that the
/// next thousand steps do not want. A prefill's arena is `rows * vocab * 4`
/// -- 233 MB for 384 rows -- and keeping that between steps would be a
/// quarter of a gigabyte of VRAM held for a fire that has finished. Anything
/// over [`Arenas::KEEP`] is freed on the spot and the decode-sized buffer, if
/// one is held, stays.
#[derive(Default)]
pub struct Arenas {
    /// The one buffer, if there is one worth keeping.
    held: Option<crate::device::Buffer>,
}

impl Arenas {
    /// The largest arena worth holding between steps: 16 MiB.
    ///
    /// Well above every decode arena this fleet states -- qwen3-0.6b's is 326
    /// KB and gpt-oss-20B's is under two megabytes -- and far below the
    /// prefill arenas that must not be held.
    const KEEP: u64 = 16 << 20;

    /// The held buffer if it is EXACTLY `size` bytes, taking it out of the
    /// cache.
    ///
    /// Exact and not "at least", though a larger one would serve: an arena
    /// bound `whole` at a length past what the plan states is a descriptor
    /// range this crate did not compute, and `binding::extent`'s refusals are
    /// the check that a rectangle fits its arena. Handing out a bigger buffer
    /// would make that check pass for a plan it should refuse.
    fn take(&mut self, size: u64) -> Option<crate::device::Buffer> {
        if self.held.as_ref().is_some_and(|b| b.size() == size) {
            return self.held.take();
        }
        None
    }

    /// Give a buffer back, keeping it if it is small enough to be worth
    /// holding.
    fn give(&mut self, device: &Device, buffer: crate::device::Buffer) {
        if buffer.size() > Self::KEEP {
            device.free(buffer);
            return;
        }
        if let Some(old) = self.held.replace(buffer) {
            device.free(old);
        }
    }

    /// Free what is held.
    ///
    /// Explicit and not a `Drop`, for the reason [`Device::free`] is: the
    /// device is what destroys a buffer, and a cache that borrowed one could
    /// not be held beside it. [`crate::shell::Shell`]'s `Drop` is the caller.
    pub fn release(&mut self, device: &Device) {
        if let Some(b) = self.held.take() {
            device.free(b);
        }
    }
}

/// Lowerings kept between fires, by the row shape they were lowered for.
///
/// # Why this is sound
///
/// [`model_compiler::lower::lower`] is a pure function of a plan, a slice of
/// [`Row`], and a [`LowerFire`] — and a `Row` is six booleans and an optional
/// depth. It carries no position, no history length, no page and no token.
/// Everything that varies between two decodes of the same conversation
/// reaches the GPU through [`Pool::stage`] and the state tables, not through
/// here. So two steps whose rows compare equal have identical lowerings, and
/// the second one need not be computed.
///
/// # Why it was worth doing
///
/// Measured on a decode of the real 4-bit qwen3-0.6B, phase by phase:
/// `lower` was **1.38 ms** of an 8.9 ms step, second only to the 6.6 ms of
/// dispatches and larger than everything else combined. A server decodes with
/// the same row shape for as long as a conversation runs, so every one of
/// those milliseconds after the first was spent recomputing an answer it
/// already had.
///
/// # Why a small vector and not a map
///
/// The key is a `Vec<Row>` as tall as the fire, so hashing it is
/// proportional to the thing being avoided, while comparing it stops at the
/// first difference. A deployment sees very few distinct shapes — one per
/// batch width it actually runs — and [`Lowerings::CAP`] bounds it. Eviction
/// is oldest-first rather than least-recently-used, which for this access
/// pattern is the same thing and is one line instead of a counter per entry.
#[derive(Default)]
pub struct Lowerings {
    held: Vec<(bool, Vec<Row>, Lowered, u64)>,
    lowered: u32,
    /// The last serial handed out. See [`Lowerings::of`].
    serial: u64,
}

impl Lowerings {
    /// How many shapes to keep.
    ///
    /// Eight because a lowering is small — launches, symbols and offsets, no
    /// weights and no arena — and because a deployment that runs more than
    /// eight distinct batch widths in rotation is one whose lowerings are not
    /// its problem.
    const CAP: usize = 8;

    /// How many lowerings this has actually computed.
    ///
    /// The saving, stated as a number a test can read rather than as a wall
    /// time it would have to race. Without it the cache is unfalsifiable from
    /// outside: a broken one that recomputed every step returns exactly the
    /// same answers, only slower.
    #[must_use]
    pub fn lowered(&self) -> u32 {
        self.lowered
    }

    /// The lowering for `rows`, computed only if it is not already held.
    ///
    /// `prefill` is part of the key and not a detail: the two plans state
    /// different projection kernels for identical rows — the whole reason
    /// [`Serving::prefill`] exists — so a cache keyed on rows alone would
    /// answer a one-row fire with the prefill's GEMM or the reverse.
    fn of(
        &mut self,
        prefill: bool,
        plan: &ForwardPlan,
        rows: &[Row],
    ) -> Result<(&Lowered, u64), Uncovered> {
        if let Some(at) = self
            .held
            .iter()
            .position(|(p, k, _, _)| *p == prefill && k.as_slice() == rows)
        {
            return Ok((&self.held[at].2, self.held[at].3));
        }
        self.lowered += 1;
        let low = lower(
            plan,
            rows,
            LowerFire {
                captures_across_splits: false,
            },
        )?;
        if self.held.len() == Self::CAP {
            self.held.remove(0);
        }
        // A NUMBER PER LOWERING, AND NEVER REUSED. `crate::replay` keys a
        // recorded fire on which lowering produced it, and the obvious key --
        // the `&Lowered`'s address -- is the one that cannot be used: this is
        // a `Vec` that reallocates and evicts, so an address that named one
        // lowering can later name another, and a plan cache that believed it
        // would replay one shape's command buffer for another shape's rows.
        self.serial += 1;
        let serial = self.serial;
        self.held.push((prefill, rows.to_vec(), low, serial));
        let last = self.held.last().expect("just pushed");
        Ok((&last.2, last.3))
    }
}

/// Why a step did not run.
///
/// Every layer's refusal, kept apart rather than flattened into one string:
/// the caller's next move differs completely between "the cache is full"
/// (evict, or queue this conversation) and "this plan names a kernel you did
/// not give me" (a build problem, and no amount of waiting fixes it).
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
    /// Refused rather than answered with an empty [`Logits`], because a
    /// lowering over zero rows is not a thing this crate has ever produced and
    /// a server asking for one has lost track of its own queue.
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

/// A model on a device, fired repeatedly.
///
/// Holds the things that outlive a fire and borrows nothing that does not:
/// the plan, the geometry and the tier are the deployment, and the book, pool,
/// weights and pipeline cache are passed to each step because they are
/// mutated by it and a server may well own them elsewhere.
#[derive(Clone, Copy)]
pub struct Serving<'a> {
    /// The text a one-row step lowers.
    pub plan: &'a ForwardPlan,
    /// The text a step of more than one row lowers.
    ///
    /// **Two plans and not one, because the class is not cosmetic.** This
    /// struct held a single `plan` until it was measured: `llama_like_metal`
    /// traced at `FireClass::Decode` states `affine_qmv_fast` for its
    /// projections, and traced at `FireClass::Prefill` states
    /// `affine_qmm_t_..._bm_16_bn_32` instead -- the tiled GEMM, the largest
    /// kernel family in the tree. A deployment carrying only the decode plan
    /// therefore answered a sixty-four-token prompt with sixty-four
    /// matrix-VECTOR products, one per row. Correct, and the wrong kernel.
    ///
    /// The two plans agree on everything else, measured over qwen3-0.6B: the
    /// same 452 launches, and the same attention symbol
    /// (`sdpa_paged_decode_bfloat16_d_128`) in both, because the paged decode
    /// kernel is what this text uses at every width.
    ///
    /// Not an `Option` with a fallback. A fallback would make the slow path
    /// the quiet default, which is exactly the defect this field exists to
    /// stop. Tracing a plan is host-side and cheap, so a deployment that only
    /// ever decodes pays one trace it does not use.
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
    /// - frame before lowering, because the lowering's row count, arena size
    ///   and readout all come from the frame's seriation;
    /// - stage before firing, or the fire reads the PREVIOUS step's tables --
    ///   which is the failure that looks like a model that has forgotten the
    ///   last token;
    /// - free the arena after reading, not after firing.
    ///
    /// # Errors
    ///
    /// [`Unstepped`]. **A refused growth leaves the book untouched**, so a
    /// caller that evicts and retries gets the same answer it would have got
    /// had it evicted first. A failure after the growth does NOT roll it back:
    /// the pages are the conversation's, the fire that was going to use them
    /// did not run, and rolling back would lose a history that is still on the
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
        self.over(device, pipelines, modules, held, &requests, &tokens)
    }

    /// Fire over requests somebody else allocated pages for.
    ///
    /// # Why this exists beside [`Self::step`]
    ///
    /// There are two page allocators in this system and only one of them can
    /// be right for a given caller. [`crate::pages::Book`] is this driver's
    /// own: it decides which physical page a conversation gets, which is what
    /// a server built on this crate alone needs. The ENGINE decides for
    /// itself -- its scheduler hands down a `kv_page_indices` CSR naming
    /// physical pages it chose, because it also runs eviction, prefix sharing
    /// and the copy plans that move pages between conversations.
    ///
    /// A driver that ran the engine's frames through its own book would have
    /// two allocators disagreeing about who owns page 7, and the disagreement
    /// is silent: attention reads a page that holds another conversation's
    /// keys and returns fluent text. So the engine's path does not touch the
    /// book at all, and this is the seam where it joins.
    ///
    /// `tokens` is per REQUEST, parallel to `requests`, in the same order. It
    /// is separate from the requests because a [`Request`] states positions
    /// and pages -- where a row goes -- and says nothing about what is in it.
    ///
    /// # Errors
    ///
    /// [`Unstepped`], minus [`Unstepped::Unhoused`], which only a growth can
    /// produce.
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
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        let owned: Vec<Vec<u32>> = tokens.iter().map(|t| t.to_vec()).collect();
        let mut step = self.tiled(
            device,
            pipelines,
            modules,
            held,
            requests,
            &owned,
            0..rows,
            0,
        )?;
        // The readout is the WHOLE fire's, whatever it was split into. Each
        // sub-fire computed one for its own rows, which numbers a request's
        // last row from the sub-fire's start; a caller asked about the turns
        // it handed in.
        let mut request_of_row = Vec::with_capacity(rows);
        for (r, request) in requests.iter().enumerate() {
            request_of_row.extend(std::iter::repeat_n(
                u32::try_from(r).unwrap_or(u32::MAX),
                request.positions.len(),
            ));
        }
        step.readout_of = last_row_of(requests.len(), &request_of_row);
        // Re-derived here for the same reason `readout_of` is: `over` is the
        // only caller that knows the WHOLE fire, and a sub-fire numbers rows
        // from its own start.
        step.readouts_of = spans_of(requests).map_err(Unstepped::Unstageable)?;
        step.positions = requests
            .iter()
            .flat_map(|r| r.positions.iter().copied())
            .collect();
        Ok(step)
    }

    /// One fire, or as many tile-shaped ones as it takes.
    ///
    /// # The tile the GEMM does not have
    ///
    /// `affine_qmm_t` is compiled for row tiles of 16, 32 and 64 and nothing
    /// narrower, and it reads its tile FROM the grid: a fire of 29 rows has
    /// no grid over a 16-row tile that covers 29 rows and stops there, so
    /// `geometry::eval` refuses it by name -- `PartialTile`. Every prompt
    /// whose length is not a multiple of the tile is such a fire, which is
    /// very nearly every prompt: "The capital of France is" is 29 tokens.
    ///
    /// `device.rs` used to say a caller above this crate owed the batching.
    /// No caller does -- the engine hands down the tokens a request arrived
    /// with -- and `driver-metal` refuses the same fire the same way, so this
    /// is not a Vulkan gap but the first place the fleet walked into it.
    ///
    /// So the refusal is caught here and answered with fires the tile does
    /// cover:
    ///
    /// * `rows >= tile`: a HEAD of `rows - rows % tile` and then a TAIL of
    ///   exactly `tile` rows ending at the last row. The two OVERLAP, and the
    ///   overlap is recomputation rather than a second answer: appending a
    ///   token's KV writes the same bytes to the same slot whatever fire does
    ///   it, and a row's attention reads a history that does not depend on
    ///   which fire computed it. 29 rows is a 16-row fire and a 16-row fire,
    ///   with rows 13..16 computed twice and the second answer kept.
    /// * `rows < tile`: one row at a time, which is a decode and the path
    ///   every step of every conversation already takes.
    ///
    /// The cost is one extra fire per prefill, or `rows` fires for a prompt
    /// shorter than a tile. It is paid knowingly and it is the correctness
    /// floor: the day a narrower tile is compiled, the same code splits into
    /// smaller pieces without changing.
    ///
    /// `depth` bounds the recursion: a sub-fire may name a WIDER tile than
    /// the one that sent it here -- the text picks its tile by row count --
    /// and each split makes the pieces smaller, so the descent terminates,
    /// but a bound that does not depend on that reasoning is cheaper than
    /// trusting it.
    #[allow(clippy::too_many_arguments)]
    fn tiled<M: Modules>(
        &self,
        device: &Device,
        pipelines: &mut Pipelines,
        modules: &M,
        held: &mut Held<'_>,
        requests: &[Request],
        tokens: &[Vec<u32>],
        span: std::ops::Range<usize>,
        depth: u32,
    ) -> Result<Step, Unstepped> {
        let (cut, cuts) = slice(requests, tokens, span.clone());
        let borrowed: Vec<&[u32]> = cuts.iter().map(Vec::as_slice).collect();
        let refused = match self.once(device, pipelines, modules, held, &cut, &borrowed) {
            Ok(step) => return Ok(step),
            Err(e) => e,
        };
        let rows = span.len();
        let Some(tile) = partial_tile(&refused) else {
            return Err(refused);
        };
        if depth >= 3 || tile == 0 {
            return Err(refused);
        }
        if rows < tile {
            // One row at a time. Fired in row order, because a row's
            // attention reads the rows before it out of the cache and they
            // have to be in it.
            let mut whole: Option<Step> = None;
            for row in span.clone() {
                let one = self.tiled(
                    device,
                    pipelines,
                    modules,
                    held,
                    requests,
                    tokens,
                    row..row + 1,
                    depth + 1,
                )?;
                whole = Some(match whole {
                    None => one,
                    Some(so_far) => join(so_far, one, 0),
                });
            }
            return whole.ok_or(Unstepped::Nothing);
        }
        let head = rows - rows % tile;
        let a = self.tiled(
            device,
            pipelines,
            modules,
            held,
            requests,
            tokens,
            span.start..span.start + head,
            depth + 1,
        )?;
        if head == rows {
            return Ok(a);
        }
        let b = self.tiled(
            device,
            pipelines,
            modules,
            held,
            requests,
            tokens,
            span.end - tile..span.end,
            depth + 1,
        )?;
        // The tail fire recomputed the last `tile - (rows - head)` rows of the
        // head; its FIRST rows are those, and they are dropped here.
        Ok(join(a, b, tile - (rows - head)))
    }

    /// One fire, over exactly the requests it is given.
    #[allow(clippy::too_many_lines)]
    fn once<M: Modules>(
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
        let frame_span = crate::phase::span("step/frame");
        let shape = held.pool.shape();
        let frame = Frame::of(shape, requests).map_err(Unstepped::Unstageable)?;
        // EVERY ROW SAMPLES, and this is a workaround with a name.
        //
        // A frame's own seriation marks only the rows a request reads out, so
        // a prefill of four tokens lowers to `n_requests = 1` and an arena
        // sized for ONE row of logits. The texts, though, spell their epilogue
        // as three plain `OpKind::Launch` ops rather than `OpKind::LmHead`, so
        // `Lowerer::epilogue` -- which would emit them over `0..sampled` --
        // never runs, and the generic path emits them over the whole token
        // window. The head then writes `rows * vocab` into an arena holding
        // `1 * vocab`, and `binding::extent` refuses the fire.
        //
        // This is not one text's quirk, which was the first guess and was
        // wrong. `llama_like_metal` reaches `dsl::metal::lm_head`
        // (`llama_like/forward/mod.rs`) rather than the generic `logits()`
        // that emits `OpKind::LmHead`, and it is the metal shell that every
        // text goes through. Counted: qwen3-0.6B lowers to 452 ops with ZERO
        // `LmHead`, gpt-oss-20B to 484 ops with ZERO. So `Lowerer::epilogue`
        // is not merely unused here -- nothing this shell produces reaches it,
        // and the workaround below is universal rather than a special case.
        //
        // `lower.rs` records the same thing from the other side: the metal
        // shell "forces `samples: true` on every row and pays a prefill's head
        // over every token rather than one per request". Doing the same here
        // makes `n_requests` the row count, so the arena is sized for what the
        // launches actually write. Measured over both a prefill of four and a
        // mixed batch of one and three: the worst operand overrun is ZERO
        // bytes, the same tightest-fit-with-nothing-to-spare `tests/arena.rs`
        // finds for decodes.
        //
        // The cost is real and is paid knowingly: the lm head runs over every
        // token in a prefill instead of once per request, which for a long
        // prompt is most of the fire. It is not a policy choice -- it is the
        // only shape this text lowers into consistently, and it stops being
        // needed the day the metal shell names its epilogue.
        let mut rows = frame.seriation();
        for row in &mut rows {
            row.samples = true;
        }
        // One row is a decode, more than one is a prefill -- including a
        // BATCH of one-token turns, which is not a slip. Four rows want a
        // tiled GEMM as much as a four-token prompt does; the row count is
        // what the kernel choice actually depends on, and the name of the
        // class is only how the text spells it.
        let prefill = frame.rows() > 1;
        let plan = if prefill { self.prefill } else { self.plan };
        // Split from `held` by field, so that the borrow this lowering is
        // returned behind does not stand in the way of `held.pool` below.
        drop(frame_span);
        let lower_span = crate::phase::span("step/lower");
        let (low, serial) = held
            .lowerings
            .of(prefill, plan, &rows)
            .map_err(Unstepped::Uncovered)?;
        drop(lower_span);

        let stage_span = crate::phase::span("step/stage");
        held.pool.stage(device, &frame).map_err(Unstepped::Failed)?;
        // The flash decode's scratch, sized before anything is planned
        // because the arm reads the table's PRESENCE to decide whether it may
        // split at all. `Pool::partials` grows and never shrinks, so a steady
        // decode allocates this once and `crate::replay` keeps its key.
        {
            use crate::binding::FireNumber;
            use crate::binding::Resolve;
            let bucket = held.pool.number(FireNumber::KvHistoryBucket).unwrap_or(0);
            let rows = frame.rows() as u32;
            let splits = kernels_vulkan::attn::decode_splits(
                i32::try_from(bucket).unwrap_or(i32::MAX),
                self.geometry.q_heads.cast_signed(),
                rows.cast_signed(),
            )
            .max(1) as u64;
            let floats = splits
                * u64::from(rows)
                * u64::from(self.geometry.q_heads)
                * (u64::from(self.geometry.head_dim) + 2);
            held.pool
                .partials(device, floats)
                .map_err(Unstepped::Failed)?;
        }
        // AND THE TABLE MUST SAY THE SAME THING THE LOWERING WAS TOLD.
        //
        // `Pool::stage` writes `SamplingIndices` from the frame, which names
        // only the rows a request reads out -- one entry for a prefill of
        // four. The lowering above was told every row samples, so its
        // `row_gather` reads `n_requests` entries, which is four. Left alone
        // that is a sixteen-byte read of a four-byte buffer: `Arg::Named` has
        // no extent for `binding::extent` to check, the descriptor is bound
        // `whole`, and the validation layer does not report storage-buffer
        // overruns even with GPU-AV on -- measured. It would have gathered
        // three rows of whatever the allocator put after the table and called
        // them hidden states.
        //
        // The identity, because every row samples and the readout's rows are
        // then the fire's rows in order. `Step::readout_of` is what turns that
        // back into an answer per turn.
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

        drop(stage_span);
        let arena_span = crate::phase::span("step/arena");
        let arena = arena_for(device, held.arenas, low.arena_bytes)?;
        drop(arena_span);
        // The plan's runtime streams, so a text's `positions` binds the
        // table this step just staged rather than the seam stand-in. Per
        // step because the decode and prefill plans each mint their own ids.
        let streams = crate::runtime::Streams::of(plan);
        let model = Model {
            weights: held.weights,
            pool: held.pool,
            runtime: &streams,
        };
        let fire_span = crate::phase::span("step/fire");
        // WHAT MAKES THE LAST FIRE STILL THE RIGHT ANSWER.
        //
        // Every part of it is stated rather than inferred, and the two halves
        // come from different places for a reason `crate::replay::Key` gives
        // at length: the device counts what it allocated and freed, which
        // covers every buffer any of these 452 rectangles could bind, and the
        // caller states everything that is NOT a buffer -- which lowering,
        // and the four fire-wide numbers a routine reads off the pool, none
        // of which any allocation counter can see change.
        let key = crate::replay::Key {
            plan: serial,
            state: state_of(&model, held.weights),
            arena: arena.identity(),
            arena_bytes: low.arena_bytes as u64,
            allocations: device.allocations(),
            frees: device.frees(),
            geometry: self.geometry,
            tier: self.tier,
            align: device.min_storage_offset(),
        };
        let ran = fire_reusing(
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
                one_at_a_time: false,
            },
            held.plans,
            key,
        );
        // THE ROWS A CALLER WILL ASK ABOUT, and not every row the text
        // computed. The two differ by three orders of magnitude on a prefill:
        // "every row samples" above makes the readout as tall as the fire, so
        // a 1024-token prompt produces 155 million logits and its turn wants
        // 151,936 of them.
        //
        // Safe to narrow to exactly this because `Serving::over` recomputes
        // `readout_of` from the requests' own order, and `Frame::of` pushes a
        // row per position walking the requests in that same order -- so the
        // rows it names are the rows named here. In a SPLIT fire each piece
        // names the last row of each request it contains, and a request's
        // last row lies in the piece that contains it, so every row `over`
        // can reach was read by the sub-fire that produced it.
        drop(fire_span);
        let read_span = crate::phase::span("step/logits");
        let readout_of = last_row_of(requests.len(), &frame.request_of_token);
        // This piece's own last rows, PLUS every row its requests asked for.
        // `slice` carries `samples` through the cut for exactly this: without
        // it the frame names only last rows and a verifier's earlier rows are
        // never read back, which is a `Logits::row` of `None` four layers
        // later. A read is ONE span from the lowest wanted row to the highest
        // (see `serve::logits_of`), so widening it costs the widening and not
        // a submission -- widening it to EVERY row instead would cost 1.2 GB
        // on a 4096-row prefill, which is what that design exists to avoid.
        let mut wanted = readout_of.clone();
        wanted.extend(frame.sampling_indices.iter().map(|&r| r as usize));
        wanted.sort_unstable();
        wanted.dedup();
        let read = ran.map_err(Unstepped::Unfired).and_then(|fired| {
            match logits_of(device, &arena, low, &wanted) {
                Ok(l) => Ok((fired, l)),
                Err(e) => Err(Unstepped::Unread(e)),
            }
        });
        drop(read_span);
        let free_span = crate::phase::span("step/free");
        // Given back on both paths: a step that refused and leaked its arena
        // would run a server out of memory in exactly the situations where it
        // was already in trouble. `Arenas::give` frees anything it does not
        // keep, so this is that free plus a decode's one reuse.
        held.arenas.give(device, arena);
        drop(free_span);
        let (fired, logits) = read?;
        Ok(Step {
            logits,
            fired,
            rows: frame.rows(),
            // This piece's, left empty: `over` replaces it with the whole
            // fire's and `join` renumbers the logits to match.
            readouts_of: Vec::new(),
            positions: frame.positions.clone(),
            readout_of,
            pipelines: pipelines.built(),
        })
    }
}

/// The tile a refusal names, if it is the tile refusal.
///
/// Reads through three layers -- a step's, a fire's and a dispatch's -- and
/// that is the point: `Serving::tiled` acts on ONE condition and every other
/// refusal has to pass through it untouched. Matching the whole path by hand
/// is how a later variant that happens to carry a `tile` field stays
/// unhandled instead of quietly becoming a split.
/// Everything a fire's resolver answers that is not a buffer, as one number.
///
/// The device's allocation and free counts say that the SET of buffers has
/// not moved, which covers every weight, every KV layer and every fire table
/// -- a table this pool replaced was one `vkCreateBuffer` and one
/// `vkDestroyBuffer`, and both are counted. What they cannot see is a NUMBER:
/// `Resolve::number` answers the page size, the two cache strides and the
/// mask pitch, a routine bakes them into a push block or a scalar block, and
/// none of the four allocates anything when it changes.
///
/// So they are asked for directly, through the same resolver the fire will
/// use. Four calls, on a path that is about to plan 452 rectangles.
///
/// The weight count is in here as the cheapest statement that the checkpoint
/// is the same one; a store that swapped a tensor for another of the same
/// size would have freed one buffer and allocated another, which the device
/// counts.
fn state_of(model: &Model<'_>, weights: &Weights) -> u64 {
    use crate::binding::FireNumber;
    use crate::binding::Resolve;
    let mut state = weights.len() as u64;
    for which in [
        FireNumber::KvPageSize,
        FireNumber::KvHeadStride,
        FireNumber::KvSeqStride,
        FireNumber::AttentionMaskStride,
        // The flash decode's split count is a function of this bucket, the
        // split count is a GRID, and the grid is what `crate::replay`
        // re-submits. Leave it out and the boundary where the bucket doubles
        // replays the old grid -- half the history unattended, and only at
        // that one token. See `crate::binding::FireNumber::KvHistoryBucket`.
        FireNumber::KvHistoryBucket,
    ] {
        // A missing answer and a zero are different facts, and a routine acts
        // on the difference -- a mask pitch of zero is "no mask", where
        // `None` is "this resolver does not answer that". Folded so they
        // cannot collide.
        state = state.wrapping_mul(0x100_0000_01b3)
            ^ u64::from(model.number(which).unwrap_or(u32::MAX))
            ^ u64::from(model.number(which).is_none());
    }
    state
}

/// A zeroed arena, zeroed BY THE DEVICE.
///
/// # Why not `Device::buffer(&vec![0u8; n])`
///
/// That is what this was, and it made a zero-filled `Vec` in system memory
/// and then uploaded it. Both halves are paid in full: the host memset, and
/// then the whole arena across the bus. For a decode the arena is 326 KB and
/// nobody would notice. For a prefill it is `rows * vocab * 4` -- 384 rows of
/// qwen3-0.6b's 151,936-entry vocabulary is **233 megabytes** -- and the
/// measurement is not subtle:
///
/// | arena | host `vec!` + upload | `empty` + `vkCmdFillBuffer` |
/// |---|---|---|
/// | 326 KB (a decode) | 0.20 ms | 0.14 ms |
/// | 233 MB (a 384-row prefill) | **35.5 ms** | **1.5 ms** |
///
/// 35.5 ms was 21% of that prefill's 167 ms. The upload runs at the 10 GB/s
/// `Device::write` documents, which is the bus behaving correctly -- the
/// mistake is not slow bytes, it is sending 233 MB of zeros over a bus at all
/// when the card can fill them in place at its own memory bandwidth.
///
/// This is the same fault this crate found in `Pool::resize` and then again
/// in `Device::copy_within`, one verb further over: those two MOVED bytes the
/// device already had through the host, and this one MADE bytes on the host
/// that the device could have made itself. Nothing was ever wrong with the
/// contents, which is why no test went red for any of the three.
///
/// # Why it is still zeroed at all
///
/// `Device::empty` alone would be faster still, and the arena is written
/// before it is read by every op that reads it -- `tests/arena.rs` measures
/// the fit as exact with nothing to spare. But "every op writes before it
/// reads" is a property of the LOWERING, not of this function, and a text
/// whose ops leave a gap would then read whatever the allocator last had in
/// that memory: another conversation's logits, and fluent text. The fill
/// costs 1.5 ms on the largest arena here and buys the guarantee outright.
///
/// # Errors
///
/// [`Unstepped::Failed`], if the allocation or the fill is refused.
fn arena_for(
    device: &Device,
    arenas: &mut Arenas,
    bytes: usize,
) -> Result<crate::device::Buffer, Unstepped> {
    // Rounded up because `Device::zero` fills whole words, and an arena whose
    // size is not a multiple of four would otherwise leave its last bytes
    // holding whatever the allocation came with.
    let size = (bytes as u64).max(4).next_multiple_of(4);
    let alloc = crate::phase::span("step/arena/empty");
    // The step before this one's, when it was the same size. See [`Arenas`].
    let arena = match arenas.take(size) {
        Some(held) => held,
        None => device.empty(size).map_err(Unstepped::Failed)?,
    };
    drop(alloc);
    let _z = crate::phase::span("step/arena/zero");
    device.zero(&arena, 0, size).map_err(Unstepped::Failed)?;
    Ok(arena)
}

fn partial_tile(why: &Unstepped) -> Option<usize> {
    match why {
        Unstepped::Unfired(Unfired::Unplannable {
            why:
                crate::dispatch::Undispatchable::Ungeometric {
                    why: crate::geometry::Ungeometric::PartialTile { tile, .. },
                },
            ..
        }) => Some(*tile as usize),
        _ => None,
    }
}

/// The rows `span` of `requests`, as requests of their own.
///
/// A request keeps ALL its pages whichever of its rows are taken: the pages
/// are the conversation's history and a row attends the whole of it, so a
/// sub-fire holding only the pages its own rows write to would attend a
/// prefix of the conversation and answer fluently.
fn slice(
    requests: &[Request],
    tokens: &[Vec<u32>],
    span: std::ops::Range<usize>,
) -> (Vec<Request>, Vec<Vec<u32>>) {
    let (mut cut, mut cuts) = (Vec::new(), Vec::new());
    let mut base = 0usize;
    for (r, request) in requests.iter().enumerate() {
        let n = request.positions.len();
        let (lo, hi) = (span.start.max(base), span.end.min(base + n));
        base += n;
        if lo >= hi {
            continue;
        }
        let (from, to) = (lo - (base - n), hi - (base - n));
        let mut piece = Request::of(request.positions[from..to].to_vec(), request.pages.clone());
        // A piece of a traced request is still traced: the pages it carries
        // are the same pages the program stated, and cutting the fire into
        // sub-fires does not turn them into a scheduler's placement.
        piece.traced = request.traced;
        // And its write targets are per ROW, so they are cut with the rows.
        if !request.writes.is_empty() {
            piece.writes = request.writes
                [from.min(request.writes.len())..to.min(request.writes.len())]
                .to_vec();
        }
        // The mask rows are per ROW, so they are cut with the rows. Left whole
        // they would be read against the piece's own row numbering, which is
        // another row's allow-bytes; dropped they would silently unmask -- an
        // empty mask is not a mask of zeros, it is the causal rule alone.
        if !request.mask.is_empty() {
            piece.mask =
                request.mask[from.min(request.mask.len())..to.min(request.mask.len())].to_vec();
        }
        // The readout rows that land in THIS piece, renumbered to it.
        //
        // Left empty when none do, which `Request::read` takes as "the last
        // row" -- one extra row read back and nothing else, because `over`
        // rewrites the whole fire's readout afterwards. That is why this
        // needs no third state for "reads nothing": the only consumer of a
        // piece's readout is the READBACK, and reading one row more than
        // asked is a wider span, not a wrong answer.
        piece.samples = request
            .samples
            .iter()
            .filter_map(|&row| {
                let row = row as usize;
                (row >= from && row < to).then(|| u32::try_from(row - from).unwrap_or(u32::MAX))
            })
            .collect();
        cut.push(piece);
        cuts.push(
            tokens
                .get(r)
                .map(|t| t.get(from..to.min(t.len())).unwrap_or(&[]).to_vec())
                .unwrap_or_default(),
        );
    }
    (cut, cuts)
}

/// Two sub-fires' answers, end to end, dropping `overlap` rows off the second.
///
/// The dropped rows are rows the first fire already computed. Both answers are
/// the same numbers -- the same weights over the same history -- so which one
/// is kept is not a choice about arithmetic; the LATER one is kept because it
/// is the one whose fire wrote the cache last, and keeping the pair in that
/// order is what makes the join independent of how the split was made.
fn join(first: Step, second: Step, overlap: usize) -> Step {
    let vocab = if first.logits.vocab == 0 {
        second.logits.vocab
    } else {
        first.logits.vocab
    };
    // Both halves hold only the rows they were asked for -- see
    // `serve::Logits::read` -- so the join is over those lists and not over a
    // row count. The second fire's row `k` is the whole fire's row
    // `first.rows + k - overlap`, and its first `overlap` rows are the ones
    // the first fire already computed and already holds.
    let mut values = first.logits.values;
    let mut read = first.logits.read;
    for (at, &row) in second.logits.read.iter().enumerate() {
        if row < overlap {
            continue;
        }
        let Some(one) = second.logits.values.get(at * vocab..(at + 1) * vocab) else {
            continue;
        };
        values.extend_from_slice(one);
        read.push(first.logits.rows + row - overlap);
    }
    let kept = second.logits.rows.saturating_sub(overlap);
    Step {
        logits: Logits {
            rows: first.logits.rows + kept,
            vocab,
            values,
            read,
        },
        fired: Fired {
            dispatches: first.fired.dispatches + second.fired.dispatches,
            submissions: first.fired.submissions + second.fired.submissions,
            blocks: first.fired.blocks + second.fired.blocks,
            parsed: first.fired.parsed + second.fired.parsed,
            tiered: 0,
        },
        rows: first.rows + second.rows.saturating_sub(overlap),
        // Both rewritten by `over`, which is the only caller that knows the
        // whole fire. Stated from the pieces anyway so that a `Step` out of
        // `join` is never half-filled.
        readout_of: first.readout_of.clone(),
        readouts_of: first.readouts_of.clone(),
        positions: first
            .positions
            .iter()
            .copied()
            .chain(second.positions.iter().skip(overlap).copied())
            .collect(),
        pipelines: second.pipelines,
    }
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
fn spans_of(requests: &[Request]) -> Result<Vec<Vec<usize>>, Unstageable> {
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
fn last_row_of(requests: usize, request_of_token: &[u32]) -> Vec<usize> {
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
fn place(tokens: &[&[u32]], request_of_token: &[u32]) -> Vec<u32> {
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

    /// A piece of a request keeps everything that is stated per ROW.
    ///
    /// `slice` rebuilds each piece with `Request::of`, which knows only
    /// positions and pages. Every field added since is one this walk can drop
    /// in silence, and two already were: a traced request came out of a split
    /// fire looking scheduler-placed, which put beam search's two lanes back on
    /// the page-sharing refusal, and its mask rows came out whole -- read
    /// against the piece's own row numbering, which is another row's
    /// allow-bytes.
    #[test]
    fn a_piece_of_a_request_carries_its_rows_mask_and_write_targets() {
        let mut whole = Request::of(vec![0, 1, 2], vec![3]);
        whole.traced = true;
        whole.mask = vec![vec![1, 0], vec![1, 1], vec![0, 1]];
        whole.writes = vec![(3, 0), (3, 2), (3, 3)];

        let (cut, _) = slice(&[whole], &[vec![7, 8, 9]], 1..3);
        let piece = &cut[0];
        assert_eq!(piece.positions, [1, 2]);
        assert!(piece.traced, "the program placed these pages either way");
        assert_eq!(
            piece.mask,
            [vec![1, 1], vec![0, 1]],
            "the rows the piece carries, and only those"
        );
        assert_eq!(
            piece.writes,
            [(3, 2), (3, 3)],
            "cut with the rows, or row 0's slot would be written twice"
        );
    }

    /// `slice` drops an out-of-range row, which is why `Frame::of` cannot
    /// catch one on a split fire.
    ///
    /// This is the measurement behind `spans_of`'s guard: the piece comes out
    /// with NO samples, so `Request::read` answers "the last row", so nothing
    /// downstream is ever handed the bad row to refuse.
    #[test]
    fn a_split_fire_filters_the_bad_row_away_instead_of_refusing_it() {
        let mut mine = Request::of(vec![0, 1], vec![0]);
        mine.samples = vec![2];
        let (cut, _) = slice(&[mine], &[vec![7, 8]], 0..2);
        assert_eq!(
            cut[0].samples,
            Vec::<u32>::new(),
            "the row landed in no piece, so the piece states no read-out"
        );
        assert_eq!(
            cut[0].read(),
            vec![1],
            "and an empty table reads the last row, which faults nowhere"
        );
    }

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
