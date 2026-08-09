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

use model_compiler::lower::{Fire as LowerFire, Uncovered, lower};
use model_compiler::trace::ForwardPlan;

use crate::device::{Device, Pipelines};
use crate::dispatch::Geometry;
use crate::pages::{Book, Unhoused};
use crate::resources::{Frame, Model, Pool, Request, Unstageable, Weights};
use crate::serve::{Fire, Fired, Logits, Modules, Unfired, Unread, fire, logits};
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

        let arena = device
            .buffer(&vec![0u8; low.arena_bytes])
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
        let read =
            ran.map_err(Unstepped::Unfired)
                .and_then(|fired| match logits(device, &arena, &low) {
                    Ok(l) => Ok((fired, l)),
                    Err(e) => Err(Unstepped::Unread(e)),
                });
        // Freed on both paths: a step that refused and leaked its arena would
        // run a server out of memory in exactly the situations where it was
        // already in trouble.
        device.free(arena);
        let (fired, logits) = read?;
        Ok(Step {
            logits,
            fired,
            rows: frame.rows(),
            positions: frame.positions.clone(),
            readout_of: last_row_of(requests.len(), &frame.request_of_token),
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
        cut.push(Request::of(
            request.positions[from..to].to_vec(),
            request.pages.clone(),
        ));
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
    let mut values = first.logits.values;
    values.extend_from_slice(
        second
            .logits
            .values
            .get(overlap * vocab..)
            .unwrap_or_default(),
    );
    let kept = second.logits.rows.saturating_sub(overlap);
    Step {
        logits: Logits {
            rows: first.logits.rows + kept,
            vocab,
            values,
        },
        fired: Fired {
            dispatches: first.fired.dispatches + second.fired.dispatches,
            submissions: first.fired.submissions + second.fired.submissions,
            blocks: first.fired.blocks + second.fired.blocks,
            parsed: first.fired.parsed + second.fired.parsed,
        },
        rows: first.rows + second.rows.saturating_sub(overlap),
        // Both rewritten by `over`, which is the only caller that knows the
        // whole fire. Stated from the pieces anyway so that a `Step` out of
        // `join` is never half-filled.
        readout_of: first.readout_of.clone(),
        positions: first
            .positions
            .iter()
            .copied()
            .chain(second.positions.iter().skip(overlap).copied())
            .collect(),
        pipelines: second.pipelines,
    }
}

/// The last row each turn contributes, by row order.
///
/// Every row samples, so the readout's rows ARE the fire's rows in order, and
/// a turn's answer is the distribution of its last token -- the only one that
/// has seen the whole prompt.
///
/// A turn with no row gets zero. That cannot happen for a turn a step grew,
/// and answering with a row that exists beats a panic in a server loop.
#[must_use]
fn last_row_of(requests: usize, request_of_token: &[u32]) -> Vec<usize> {
    let mut last = vec![0usize; requests];
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
        assert_eq!(last_row_of(3, &[0, 1]), vec![0, 1, 0], "a turn with no row");
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
