//! Where a symbol's WGSL comes from, and what a fire reports having done.
//!
//! # `Fire`, `fire`, `record` and `logits` STOOD HERE
//!
//! This module was the fire path: **plan everything, then build every
//! pipeline, then record** -- three passes, in that order, over one lowered
//! plan's four hundred to four thousand rectangles. Each boundary existed
//! because of a borrow a single pass gets wrong. A dispatch whose scalars go
//! in a STORAGE buffer needs that buffer alive when the queue runs the command
//! buffer, so every block was allocated in pass one and lived until the
//! submission completed. [`crate::device::Pipelines::get`] takes `&mut self`
//! because it may
//! build, so a caller cannot hold a reference to one pipeline while asking for
//! the next -- and recording needs one reference per launch, all alive at
//! once; so pass two built every distinct module and pass three asked for them
//! through `Pipelines::peek`, which borrows immutably.
//!
//! `fire` and `record` took a `model_compiler::lower::Lowered` and a run of
//! its `Launch`es, and `logits` read the exit tensor out of the arena by the
//! `Readout` that lowering carried. All three are deleted with the type. What
//! stays is everything that was never about the lowering: where a module comes
//! from, what a fire REPORTS, and the vocabulary its refusals come back under.
//!
//! Two measurements are worth keeping out of `record`'s doc, because both were
//! expensive to get and neither is recoverable from the code:
//!
//! * On an M4 Pro, a 244-dispatch decode of Llama-3.2-1B at 4 bits, **the host
//!   is done**. Every host phase together was 0.7 ms of 8.5, and the GPU wait
//!   was 93%. Two host bugs worth twenty tok/s were found in this file within
//!   a day of first measuring it, and the right conclusion from that is not
//!   "keep looking here" -- it is that the looking now has a number attached
//!   and the number is small.
//! * **The wait is the launch COUNT, not bandwidth.** Truncating a fire to its
//!   first n launches and fitting the line over 44 points from 86 to 244 gave
//!   `wait ~= 1655 us + 23.0 us x launches`. Only 1.7 ms of a decode's wait is
//!   independent of how many launches it is spread over, and `kernels_wgpu::
//!   attn`'s split note prices a dispatch that does no extra work at ~13 us --
//!   so roughly **244 x 13 us = 3.2 ms of a 7.3 ms wait is the launch floor**.
//!   The way down is FEWER LAUNCHES, which is the model text's shape and the
//!   compiler's business. (A truncated fire computes garbage, so only the
//!   TIMES mean anything, and the first fires of a run are warm-up: fitting
//!   them in moves the slope by half.)
//!
//! # What this does not do, and the deployment concern that deletes
//!
//! **It does not load modules, and there is nothing to load.**
//! `driver-vulkan`'s counterpart takes a `Modules` store and its `shell.rs`
//! takes a `BTreeMap<String, Vec<u8>>` of SPIR-V, because SPIR-V is a build
//! product: something has to compile it, put it somewhere, and tell the driver
//! where. `driver-cuda` ships a fatbin; `driver-metal` ships a `.metallib`.
//!
//! Here `kernels_wgpu::entrypoint_source(symbol, tier)` IS the module, embedded
//! in the rlib by `include_str!`, expanded and compiled in this process by
//! `naga`. So a shell here takes no module argument at all, and there is no
//! directory to find. That is not a convenience; it is the deployment story
//! this backend exists for, and it is worth saying loudly because it is
//! invisible in a diff -- the argument that is not there.
//!
//! `driver-vulkan` now says the same thing. Its `Shell::open` lost its module
//! argument too, its build script -- whose only job was to relay
//! `DEP_PIE_KERNELS_VULKAN_SPV_DIR` -- is deleted, and its store is called
//! [`Embedded`] as well. The two arrived from opposite ends: this crate never
//! had a directory, that one had to be talked out of the one it had.
//!
//! [`Modules`] survives as a seam with exactly one implementation that
//! matters, [`Embedded`], which is what every caller uses. It is a trait
//! rather than a direct call because a test needs to inject a module the tree
//! does not have: a source table whose entries all parse cannot produce the
//! refusal.
//!
//! It does not choose a [`Capability`] per launch. One tier for the whole
//! fire, because the tier is a property of the device and picking it per
//! module would build two pipelines for the same symbol.
//!
//! # The read/write rule, which neither sibling has to handle
//!
//! Every rectangle of a real plan binds the arena both readable and writable
//! -- its input is one range and its output is another -- and **WebGPU refuses
//! a dispatch that binds one buffer both READABLE and WRITABLE**, however far
//! apart the two ranges are. Two WRITABLE bindings are fine, which is the way
//! out and the reason the shader tree declares no `var<storage, read>`. See
//! [`crate::device`]'s own section for the rule and the citation.
//!
//! [`crate::device::Device::run_all`] shadows the read side into a scratch
//! buffer for any dispatch that still needs it, so a fire is written the way a
//! fire would be written on any backend. What this file kept was the REPORT --
//! [`Fired::shadowed`] is how many copies a fire paid for -- and the number is
//! ZERO for every real plan, which is worth reporting for exactly the reason
//! the large number was: a cost nobody can see is a cost nobody fixes, and one
//! `read` declaration brings all 451 of them back.

use std::collections::BTreeMap;

use crate::baker::marks::{BufferId, Slice};
use crate::device::Failed;
use kernels_wgpu::Capability;

/// Where the WGSL for a symbol comes from.
///
/// A trait, and the default implementation is the whole answer — see the module
/// docs for why there is nothing to configure. Kept as a seam because a test
/// that checks what happens to a module `naga` will not take has no other way to
/// produce one.
pub trait Modules {
    /// The source for the point a body named, by the file it named it in.
    ///
    /// THE FIRE PATH'S LOOKUP. A body states both halves — `Fire::at` takes
    /// the file and the entrypoint — so this is a lookup in one source rather
    /// than a scan across every embedded one, and two files declaring the same
    /// entrypoint name can no longer resolve to whichever came first.
    fn at(&self, file: &str, entrypoint: &str, tier: Capability) -> Option<String>;

    /// The source for an entrypoint at a tier.
    ///
    /// `None` means this store has not got it, which for a tier above
    /// [`Capability::Baseline`] is the ordinary answer and the caller's cue to
    /// ask again at baseline.
    ///
    /// **Not the fire path.** [`Self::at`] is. This answers for a name that
    /// arrives without a file — the plan's own symbol, and the reflection a
    /// server does before any body has run.
    fn source(&self, entrypoint: &str, tier: Capability) -> Option<String>;
}

/// The shader tree in the rlib.
///
/// A unit struct because it holds nothing: the sources are `&'static str`s
/// `kernels-wgpu`'s build script wrote into the crate, and `entrypoint_source`
/// expands the variant's includes and `//#if` arms on demand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Embedded;

impl Modules for Embedded {
    fn at(&self, file: &str, entrypoint: &str, tier: Capability) -> Option<String> {
        kernels_wgpu::source::at(file, entrypoint, tier).ok()
    }

    fn source(&self, entrypoint: &str, tier: Capability) -> Option<String> {
        kernels_wgpu::entrypoint_source(entrypoint, tier).ok()
    }
}

/// The source a fire will run for `entrypoint`, and the tier it came from.
///
/// A tier is an ADDITIONAL variant of an entrypoint that already exists, so
/// most entrypoints have only a baseline one and asking for
/// [`Capability::Subgroup`] answers nothing. That is the ordinary case rather
/// than an error — `kernels-wgpu`'s own docs say a driver "reads
/// `Capability::PREFERENCE`, takes the first tier the adapter's features allow
/// and the tree has a variant for, and falls back to `Baseline`" — so the
/// fallback lives here, in one place, and hands back WHERE IT LANDED.
///
/// That second half is what the pipeline cache is keyed by. Without it a
/// deployment at `Subgroup` and one at `Baseline` would build the same baseline
/// module twice under two keys, and `Pipelines::built()` would stop being a
/// number that stops growing.
#[must_use]
pub fn pick<M: Modules>(
    modules: &M,
    entrypoint: &str,
    tier: Capability,
) -> Option<(String, Capability)> {
    // DOWN THE PREFERENCE LIST, NOT STRAIGHT TO BASELINE.
    //
    // This used to try `tier` and then jump to `Capability::Baseline`, which
    // was indistinguishable from correct for as long as there were exactly two
    // tiers in play. Adding `Capability::Matrix` above `Subgroup` made the
    // difference visible and expensive: with `Matrix` at the head of the
    // adapter's list, every symbol without a `@matrix` module -- which is all
    // of them, so far -- skipped its `@subgroup` module and landed on
    // baseline. The decode read 9.30 ms against 7.49 and the prefill 312.1
    // against 285.9, both exactly their baseline numbers, which is how the bug
    // announced itself.
    //
    // `PREFERENCE` is best-first and `Baseline` is always its last entry, so
    // filtering it by `<= tier` and taking the first hit is the same answer as
    // before wherever only two tiers exist, and the right one where more do.
    for candidate in Capability::PREFERENCE
        .into_iter()
        .filter(|candidate| *candidate <= tier)
    {
        if let Some(source) = modules.source(entrypoint, candidate) {
            return Some((source, candidate));
        }
    }
    None
}

/// What a fire did.
///
/// Returned rather than discarded because both numbers are otherwise
/// unobservable, and a caller that cannot observe them cannot tell a fire that
/// ran from one that quietly ran less.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fired {
    /// How many rectangles were recorded. One per launch the plan states.
    pub dispatches: usize,
    /// How many command buffers were submitted.
    ///
    /// One, unless the caller asked for a submission per launch — in which
    /// case it is [`Self::dispatches`], and the difference between the two
    /// numbers is the only externally visible sign of which path ran.
    ///
    /// **Counted since it was found to be a lie.** This was the literal `1`
    /// for as long as the field existed, and the queue was getting 735 for a
    /// real 452-launch decode: [`crate::device::Device::run_all`] opened a
    /// fresh encoder either side of every shadow point, and 451 of 452
    /// launches shadow something. Merging them into one encoder took a decode
    /// from 31.9 ms to 20.5 ms on an RTX 4090. A field whose whole purpose is
    /// that a caller "cannot tell a fire that ran from one that quietly ran
    /// less" was reporting a constant.
    pub submissions: usize,
    /// How many read-only operands had to be copied out of a buffer their own
    /// dispatch also writes.
    ///
    /// **Zero, for every real plan**, and it is reported rather than hidden
    /// because it was 451 of 452 rectangles until the shader tree stopped
    /// declaring `var<storage, read>`. WebGPU refuses a dispatch that binds
    /// one buffer both readable and writable; two WRITABLE bindings are one
    /// usage bit and legal. See [`crate::device`]'s module docs for the rule,
    /// the citation and the measurement — 25.1 ms to 11.2 ms per decode.
    ///
    /// So a non-zero here now means a `read` binding came back somewhere in
    /// the tree, whose only other symptom is that decoding got twice as
    /// slow.
    pub shadowed: usize,
    /// How many device-to-device copies the walk's `InOut` points forced.
    ///
    /// [`Self::shadowed`]'s sibling and NOT the same number: a shadow is this
    /// backend's answer to a WebGPU rule and should be zero, while a staged
    /// copy is what an `InOut` declaration means on every plane and is a real
    /// property of the text — one per `norm.residual_add`, per `rope.partial`,
    /// per `gate.sigmoid_mul`. Reported because it is the price of the walk
    /// minting a fresh rectangle for every result, and a text that merged its
    /// residual instead would show the difference here.
    pub staged: usize,
}

/// Why a fire did not run.
///
/// Two variants, both about the DEVICE, and the pair is the point: a launch
/// the device refused and a submission it never finished are different claims
/// about where a fire stopped, and
/// `a_submission_that_failed_names_the_submission_and_not_a_launch` is what
/// keeps them printing differently. Nothing production-side raises either
/// today -- `fire` was the constructor and left with the lowering it walked --
/// so that test is also what BUILDS them, which is why the pair survived the
/// cut and the four plan-side variants did not.
///
/// Every variant names the LAUNCH INDEX as well as the symbol. A plan states the
/// same symbol hundreds of times — `rms_single_row_bfloat16` appears once per
/// layer per norm — so a refusal that named only the symbol would not say which
/// of them, and the interesting question about a failure at rectangle 2891 of
/// 3992 is almost always what came before it.
#[derive(Debug)]
pub enum Unfired {
    // FOUR PLAN-SIDE VARIANTS STOOD HERE and left with the walk that raised
    // them: `NoModule` (no module for a symbol the plan states, at any tier),
    // `Unreadable` (a module the store holds that is not WGSL this crate can
    // dispatch), `Unplannable` (a rectangle `plan_one` could not turn into a
    // dispatch, carrying the deleted `dispatch::Undispatchable`), and
    // `Impossible` (this crate contradicting itself -- `plan_one` produced
    // `Params::Block` and `block_at` together or neither, and a pipeline built
    // one line above is one `peek` answers for; both unreachable, and both
    // stated rather than unwrapped because the consequence of the first was
    // binding a fire's scalars over an operand, which is a wrong answer and
    // not a crash).
    //
    // TWO OF THEM ARE BACK, because [`run`] is the walk's device half and it
    // raises them again -- `NoModule` for a symbol no tier of the tree carries,
    // and `Unbound` for a region naming an allocation the fire was not given.
    // `Unreadable` did not come back and cannot: a module that is not WGSL this
    // crate can dispatch is `Failed::Module`, which the pipeline cache raises
    // and `Refused` carries, so a second spelling of it would be two names for
    // one condition. `Unplannable` and `Impossible` were the lowering's.
    /// No module for a symbol the walk fired, at any tier.
    ///
    /// A PLAN-SIDE FAILURE AND NOT A DEVICE ONE, which is why it is not
    /// `Refused`: nothing was submitted and no adapter was asked. A claim body
    /// named `(file, entrypoint)` and the embedded tree has no such variant, so
    /// the disagreement is between `kernels-wgpu`'s claim table and
    /// `kernels-wgpu`'s shader tree and has nothing to do with this machine.
    NoModule {
        /// Which launch.
        at: usize,
        /// The entrypoint the body named.
        symbol: String,
        /// The file it named it in.
        file: String,
    },
    /// A region naming an allocation this fire was not given.
    ///
    /// [`run`]'s `buffers` is indexed by
    /// [`BufferId`](crate::baker::marks::BufferId) — the walk was handed those
    /// ids when its arena, its banks and its pools were built — so an id past
    /// the table is a driver that minted a region against a table it did not
    /// then pass. Refused by name rather than bound to whatever is at index
    /// zero, which is the weight arena on every fire this driver makes.
    Unbound {
        /// Which launch.
        at: usize,
        /// The entrypoint the body named.
        symbol: String,
        /// The id the region carried.
        buffer: usize,
    },
    /// The device refused this launch.
    ///
    /// A launch, and only a launch: everything that reaches here was checked
    /// before anything was submitted. A failure of the submission itself is
    /// [`Self::Undelivered`], which is a different variant precisely so that
    /// it cannot be printed as a launch index.
    ///
    /// IT NAMES THE SYMBOL TOO, which this type's own doc claimed of every
    /// variant while this one carried an index alone. The claim became worth
    /// keeping when [`run`] started raising it over a real plan: `launch 56` of
    /// 387 is a number a reader cannot act on, and the refusals that arrive
    /// here — a bind group whose entry count does not match the module's, a
    /// uniform block shorter than the struct — are all statements ABOUT a
    /// module, which has a name.
    Refused {
        /// Which launch.
        at: usize,
        /// The entrypoint it fired, or what the allocation was for when no
        /// launch had been reached yet.
        symbol: String,
        /// What failed.
        why: Failed,
    },
    /// The device never finished the submission.
    ///
    /// Not any one launch's refusal. `Device::run_all` submits the whole plan
    /// at once and waits once, so a device that errors or does not answer has
    /// named nothing smaller than the submission. `of` is how many launches
    /// were in it.
    ///
    /// This existed before it had a name, as `Refused { at: run.len() }` — an
    /// index one past the last launch — and a reader who did not know that
    /// convention read the count as the offending dispatch. See
    /// [`crate::device::Stage`] for the one that did.
    Undelivered {
        /// How many launches the submission held.
        of: usize,
        /// What failed.
        why: Failed,
    },
}

impl std::fmt::Display for Unfired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModule { at, symbol, file } => {
                write!(f, "launch {at}: no module for `{symbol}` in `{file}`")
            }
            Self::Unbound { at, symbol, buffer } => write!(
                f,
                "launch {at} `{symbol}`: buffer {buffer} is not one this fire was given"
            ),
            Self::Refused { at, symbol, why } => write!(f, "launch {at} `{symbol}`: {why}"),
            Self::Undelivered { of, why } => {
                write!(f, "the submission of {of} launches: {why}")
            }
        }
    }
}

impl std::error::Error for Unfired {}

// `pub enum Unread` STOOD HERE -- why a fire's answer could not be read. Four
// variants, and `logits` was the only thing that raised any of them:
// `NoExit` for a lowering that states no read-out at all, `PastArena` for a
// read-out range past the arena the PLAN sized (not past the buffer, which may
// legitimately be larger), `Width` for an element width that is neither two
// nor four, and `Refused` carrying the readback's own device failure rather
// than flattening it into a range error. All four went with `logits`.
//
// The rule the second and third encoded is the one to keep: a read-out's width
// is FOUR OR TWO and four is not a given -- `affine_qmv_fast` writes bf16 and a
// text's declared dtype does not change what a kernel does, so a reader that
// assumed f32 got a vocabulary exactly half zeros, which looks like a dead half
// of a tensor and is really two elements read as one.

/// A fire's distributions, off the arena and widened.
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    /// How many distributions: **one per ROW OF THE FIRE**, not one per
    /// readout.
    ///
    /// This doc said *"one per readout, so `Frame::readouts` and
    /// `Lowered::n_requests` and this are the same number"* and that was not
    /// what the code did — `logits` sets this from `exit.rows`, the fire's
    /// row count. Measured rather than argued: a 512-token prefill of one turn
    /// hands back `rows = 512` and 77,791,232 values, 296.8 MB as `f32`, of
    /// which the caller wants one row. [`crate::turns::Step::readout_of`]'s
    /// doc had it right all along — *"every row samples"*.
    ///
    /// # The optimisation this description was hiding
    ///
    /// Narrowing the readback to the rows [`crate::turns::Step::readouts_of`]
    /// names is worth ~180 ms of an 815 ms 512-row prefill in release
    /// (`driver-wgpu`'s `where_a_prefills_time_goes_across_its_plan` measures
    /// a fire that records ZERO rectangles and still pays it). llama.cpp
    /// computes prompt logits for the last token of a batch rather than all of
    /// them, and turns in 40,400 tok/s against this backend's 628.
    ///
    /// It is a semantic change to a public type, so here is its blast radius,
    /// counted rather than guessed. Seventeen call sites reach a row THROUGH
    /// `readout_of`/`readouts_of` and would keep working under a remap —
    /// including [`crate::frames`]'s own, which is the only one inside this
    /// crate's `src`. Five index by fire row directly and would each need
    /// reading: `tests/serving.rs` at the raw-`values` slice and at the
    /// `map(|&at| ...)` gather, and `tests/hybrid_probe.rs` at its
    /// `map_while` walk, its `picked` row and its span gather.
    pub rows: usize,
    /// Elements in one distribution, which is the vocabulary.
    pub vocab: usize,
    /// `rows * vocab` values, row major.
    pub values: Vec<f32>,
}

impl Logits {
    /// One distribution.
    #[must_use]
    pub fn row(&self, at: usize) -> Option<&[f32]> {
        self.values.get(at * self.vocab..(at + 1) * self.vocab)
    }
}

/// Put a walk's dispatches on the card, in the order the walk stated them.
///
/// **THE DEVICE HALF OF THE BAKER PATH**, and the one function in this crate
/// that takes what [`crate::baker::walk::Fire::walk`] produced and turns it into
/// `dispatch_workgroups`. Everything upstream of it is arithmetic: a `Program`,
/// an arena, a claim body stating an entrypoint and a grid. `driver-vulkan`'s
/// `serve::run` is the sibling and arrived first; this is the same three passes
/// over a plane whose modules are compiled here rather than at build time.
///
/// `buffers` is indexed by [`BufferId`], because a
/// [`Slice`](crate::baker::marks::Slice) names an ALLOCATION and an offset and
/// only the caller knows which allocation is which — the walk was handed those
/// ids when its arena, its banks and its pools were built. An id past the table
/// is [`Unfired::Unbound`].
///
/// # The three passes, and why they are three
///
/// **Resolve everything, then build every pipeline, then record.** Each
/// boundary is a borrow a single pass gets wrong, and both were found the hard
/// way by the fire path this replaces:
///
/// * A dispatch's scalars are the fields of one uniform block, and the block has
///   to be alive when the QUEUE runs the command buffer rather than when it is
///   recorded. So every block is written in pass one, into a vector that
///   outlives the submission.
/// * [`crate::device::Pipelines::get`] takes `&mut self` because it may build,
///   so a caller cannot hold a reference to one pipeline while asking for the
///   next — and recording needs one reference per launch, all alive at once. So
///   pass two builds every distinct module and pass three asks through
///   [`crate::device::Pipelines::peek`], which borrows immutably.
///
/// # The tier walk is per SYMBOL and the lookup is per FILE
///
/// A claim body states both halves — `Fire::at(file, entrypoint)` — so the
/// source is looked up in one file rather than scanned across every embedded
/// one, and two files declaring the same entrypoint name cannot resolve to
/// whichever came first. The walk down [`Capability::PREFERENCE`] is [`pick`]'s,
/// restated here against [`Modules::at`] because `pick` asks by entrypoint
/// alone.
///
/// # The honest null is a real allocation
///
/// `kernels_wgpu`'s `points::absent` binds a region that addresses nothing —
/// six of the sdpa arms declare a slot their point does not carry — and
/// `Encode::absent` answers it with [`crate::baker::marks::NOTHING`]. WebGPU
/// refuses a zero-sized binding, so this allocates four zeroed bytes once per
/// run and binds THOSE. A shader that reads the slot reads zeros loudly instead
/// of reading a neighbour, which is what the null meant.
///
/// # Errors
///
/// [`Unfired`], naming the launch index in every case but the submission's. A
/// failure part way through has still recorded and possibly RUN everything
/// before it: the queue has no way to undo a submitted command buffer, and this
/// does not pretend otherwise.
#[expect(
    clippy::too_many_arguments,
    reason = "a fire is a plan, a card, a module store and a read-out; naming \
              them in a struct would be one struct per call site"
)]
pub fn run<M: Modules>(
    device: &crate::device::Device,
    pipelines: &mut crate::device::Pipelines,
    modules: &M,
    buffers: &[&crate::device::Buffer],
    dispatches: &[crate::baker::dispatch::Dispatch],
    blits: &[crate::baker::walk::Blit],
    tier: Capability,
    read: Option<(BufferId, u64, u64)>,
) -> Result<(Fired, Vec<u8>), Unfired> {
    use crate::binding::Bound as Range;
    use crate::device::{Recorded, Stage, Staged};

    let align = device.min_storage_offset();
    // Allocated before anything is resolved, because `held` below hands it back
    // for every region the walk marked as addressing nothing.
    let nothing = device.zeroed(4).map_err(|why| Unfired::Refused {
        at: 0,
        symbol: "the run's null binding".to_owned(),
        why,
    })?;
    let held = |slice: Slice, at: usize, symbol: &str| -> Result<&crate::device::Buffer, Unfired> {
        if slice.is_nothing() {
            return Ok(&nothing);
        }
        buffers
            .get(slice.buffer.0 as usize)
            .copied()
            .ok_or_else(|| Unfired::Unbound {
                at,
                symbol: symbol.to_owned(),
                buffer: slice.buffer.0 as usize,
            })
    };

    // ── pass one: every region resolved and every block written ────────
    let mut bound: Vec<Vec<Range<'_, crate::device::Buffer>>> =
        Vec::with_capacity(dispatches.len());
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(dispatches.len());
    for (at, d) in dispatches.iter().enumerate() {
        let mut args = Vec::with_capacity(d.args.len());
        for a in &d.args {
            let buffer = held(a.slice, at, d.symbol)?;
            // A region that addresses nothing binds the whole four-byte null
            // rather than a zero-length range of it.
            let (offset, bytes) = if a.slice.is_nothing() {
                (0, buffer.size())
            } else {
                (a.slice.at, a.slice.bytes)
            };
            args.push(Range::within(buffer, offset, bytes, align).map_err(|why| {
                Unfired::Refused {
                    at,
                    symbol: d.symbol.to_owned(),
                    why: Failed::Wgpu(format!("a region the adapter cannot address: {why}")),
                }
            })?);
        }
        bound.push(args);
        blocks.push(d.uniform());
    }

    // The `InOut` copies, filed against the FIRST dispatch of the statement
    // that asked. A body may state more than one launch — `rope.full` is two —
    // and the operand's bytes have to be in place before the first of them
    // writes through the handle, not before the last.
    let mut staged: Vec<Vec<Staged<'_>>> = vec![Vec::new(); dispatches.len()];
    for blit in blits {
        let at = dispatches
            .iter()
            .position(|d| d.op == blit.op)
            .ok_or_else(|| Unfired::Unbound {
                at: blit.op as usize,
                symbol: "an in-place copy for a statement that planned no dispatch".to_owned(),
                buffer: blit.from.buffer.0 as usize,
            })?;
        let symbol = dispatches[at].symbol;
        staged[at].push(Staged {
            from: held(blit.from, at, symbol)?,
            at: blit.from.at,
            into: held(blit.to, at, symbol)?,
            to: blit.to.at,
            bytes: blit.bytes,
        });
    }

    // ── pass two: one pipeline per distinct module ─────────────────────
    let mut landed: BTreeMap<&'static str, Capability> = BTreeMap::new();
    for (file, symbol, _stamp) in crate::baker::dispatch::pipelines_needed(dispatches) {
        if landed.contains_key(symbol) {
            continue;
        }
        // The first launch that names it, so a refusal points at a dispatch a
        // reader can find rather than at launch zero.
        let at = dispatches
            .iter()
            .position(|d| d.symbol == symbol)
            .unwrap_or_default();
        let (source, cap) = Capability::PREFERENCE
            .into_iter()
            .filter(|candidate| *candidate <= tier)
            .find_map(|candidate| {
                modules
                    .at(file, symbol, candidate)
                    .map(|source| (source, candidate))
            })
            .ok_or_else(|| Unfired::NoModule {
                at,
                symbol: symbol.to_owned(),
                file: file.to_owned(),
            })?;
        pipelines
            .get(device, symbol, cap, &source)
            .map_err(|why| Unfired::Refused {
                at,
                symbol: symbol.to_owned(),
                why,
            })?;
        landed.insert(symbol, cap);
    }

    // ── the join between a body's run and a module's bind group ────────
    //
    // THERE IS NOTHING TO JOIN, AND THAT IS A RULE RATHER THAN AN OBSERVATION.
    // A claim body's buffer run is exactly the `@group(0)` bindings the module
    // READS, in binding order, and the bind group this driver builds covers
    // exactly the same set (`device::Pipelines::build` filters by
    // `Declared::used`). So the run and the layout are the same list and the
    // only thing left to do is hand it over — `Device::check_bindable` compares
    // the two counts and refuses a dispatch where they differ.
    //
    // IT WAS NOT TRUE UNTIL THIS MILESTONE, and the defect is worth recording
    // because nothing in the tree could see it. `attn/sdpa_paged.wgsl` declares
    // `sinks` at binding 10 for every variant and only the `PIE_WITH_SINK` ones
    // read it; `kernels_wgpu::attn`'s unsplit `decode`, its split and merge
    // arms, `tiled` (which is `attention.prefill` and `attention.masked`) and
    // `mma` all passed `points::absent` at that slot, and the split and merge
    // arms passed more of them at slots their variants do not reach either.
    // Every one of those dispatches was refused by `Failed::Bindings` — so
    // `attention.decode` and `attention.prefill` could not be fired AT ALL on
    // this plane, on any adapter, for any SKU. `tests/device_sink.rs` did not
    // see it because the arms it fires are the sink-bearing and `_lse` ones,
    // which bind their reads exactly; `tests/device_fire.rs` fires one norm.
    // What found it was walking a whole tower, which is what
    // `tests/banked_argmax.rs` now does.
    //
    // `points::absent` has no caller left in `kernels-wgpu` and is deleted.
    // `Encode::absent` stays, because the case it names is real — a binding a
    // module READS that the point states no operand for — and the day one
    // exists this is where its handle would arrive.

    // ── pass three: record ─────────────────────────────────────────────
    let mut recorded = Vec::with_capacity(dispatches.len());
    for (at, d) in dispatches.iter().enumerate() {
        let cap = landed[d.symbol];
        let pipeline = pipelines
            .peek(d.symbol, cap)
            .ok_or_else(|| Unfired::Refused {
                at,
                symbol: d.symbol.to_owned(),
                why: Failed::Wgpu(format!("built at {} and not held", cap.tag())),
            })?;
        recorded.push(Recorded {
            pipeline,
            buffers: &bound[at],
            uniform: &blocks[at],
            groups: pipeline.workgroups(d.lanes),
            staged: &staged[at],
        });
    }

    let readout = match read {
        None => None,
        Some((buffer, at, bytes)) => Some((
            *buffers
                .get(buffer.0 as usize)
                .ok_or_else(|| Unfired::Unbound {
                    at: dispatches.len().saturating_sub(1),
                    symbol: "the read-out".to_owned(),
                    buffer: buffer.0 as usize,
                })?,
            at,
            bytes,
        )),
    };

    let (ran, answer) = device
        .run_all_reading(&recorded, readout)
        .map_err(|(stage, why)| match stage {
            Stage::Launch(at) => Unfired::Refused {
                at,
                symbol: dispatches
                    .get(at)
                    .map_or_else(String::new, |d| d.symbol.to_owned()),
                why,
            },
            Stage::Submission { of } => Unfired::Undelivered { of, why },
        })?;
    Ok((
        Fired {
            dispatches: dispatches.len(),
            submissions: ran.buffers,
            shadowed: ran.shadowed,
            staged: ran.staged,
        },
        answer,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded tree answers for a real entrypoint and not for a made-up
    /// one.
    ///
    /// The whole of what `Modules` has to do, and it needs no adapter because
    /// the sources are in the rlib -- which is the claim this module's docs
    /// make and this is the check on it.
    #[test]
    fn the_embedded_tree_is_the_module_store() {
        let tree = Embedded;
        let source = tree
            .source("rms_single_row_bfloat16", Capability::Baseline)
            .expect("the tree holds the norm every text states");
        assert!(
            source.contains("@compute"),
            "the source is not a dispatchable module"
        );
        assert_eq!(tree.source("no_such_kernel", Capability::Baseline), None);
    }

    /// bf16 is widened by shift, and the two answers are not close.
    ///
    /// `0x3f80` is 1.0 as a bf16 and 16256 as an integer, so a cast and a shift
    /// disagree by four orders of magnitude -- and both are finite, which is why
    /// nothing downstream would report it.
    #[test]
    fn a_bfloat_is_widened_by_shift_and_not_by_cast() {
        let bits: u16 = 0x3f80;
        let shifted = f32::from_bits(u32::from(bits) << 16);
        assert_eq!(shifted, 1.0);
        assert_eq!(f32::from(bits), 16256.0, "the cast this must not be");
    }

    /// A submission that failed does not name a launch that does not exist.
    ///
    /// `Device::run_all` submits the whole plan and waits ONCE, so a device
    /// that errors or never answers has singled out nothing. That case used to
    /// come back as `Refused { at: run.len() }` -- an index one past the last
    /// launch -- and print as `launch 452:`. Valid indices for that fire were
    /// 0..=451, so the message named a launch that was not there.
    ///
    /// It is not a hypothetical misreading. That exact line was read in this
    /// repository as "the 452nd dispatch is slow", and written up twice, when
    /// what happened was that all 452 were submitted together and the one wait
    /// covering them ran out. The count and the index are the same number and
    /// the message could not say which it meant.
    ///
    /// So the two are different variants, and this is the check that they read
    /// differently: whatever else changes, the submission case must not print
    /// as `launch <the count>`.
    #[test]
    fn a_submission_that_failed_names_the_submission_and_not_a_launch() {
        let why = Failed::Wgpu(
            "the device did not answer: The requested Wait timed out before \
             the submission was completed."
                .to_string(),
        );
        let whole = Unfired::Undelivered {
            of: 452,
            why: why.clone(),
        }
        .to_string();
        let one = Unfired::Refused {
            at: 452,
            symbol: "rms_single_row_bfloat16".to_owned(),
            why,
        }
        .to_string();

        assert_ne!(
            whole, one,
            "the submission of 452 launches and the 452nd launch are different \
             claims about where a fire stopped, and a reader gets only the string"
        );
        assert!(
            !whole.starts_with("launch "),
            "a submission-level failure still reads as a launch index: {whole}"
        );
        assert!(
            whole.contains("submission of 452 launches"),
            "and it should say how many were in flight, which is the only true \
             thing there is to say about where it stopped: {whole}"
        );
        assert!(
            whole.contains("Wait timed out"),
            "without losing what the device said: {whole}"
        );
    }

    /// The two PLAN-SIDE refusals name the module, because a launch index of
    /// 387 is not a thing a reader can act on.
    ///
    /// Both came back with [`run`] and both are about a disagreement between
    /// `kernels-wgpu`'s claim tables and its shader tree rather than about this
    /// machine — a body naming an entrypoint no tier of the embedded tree
    /// carries, and a region naming an allocation the fire was never given. So
    /// neither is a `Refused`, which means "the device refused this launch",
    /// and neither may print as one.
    #[test]
    fn the_two_plan_side_refusals_name_the_module_and_not_only_the_launch() {
        let missing = Unfired::NoModule {
            at: 56,
            symbol: "sdpa_paged_decode_bfloat16_d_256".to_owned(),
            file: "attn/sdpa_paged.wgsl".to_owned(),
        }
        .to_string();
        assert!(
            missing.contains("sdpa_paged_decode_bfloat16_d_256")
                && missing.contains("attn/sdpa_paged.wgsl"),
            "a missing module has to say WHICH, in which file: {missing}",
        );

        let unbound = Unfired::Unbound {
            at: 56,
            symbol: "sdpa_paged_decode_bfloat16_d_256".to_owned(),
            buffer: 91,
        }
        .to_string();
        assert!(
            unbound.contains("91") && unbound.contains("sdpa_paged_decode_bfloat16_d_256"),
            "an unbound region has to say which allocation id it named: {unbound}",
        );
        assert_ne!(
            missing, unbound,
            "the two are different claims about where a fire stopped and a \
             reader gets only the string",
        );
    }

    /// The stage a failure carries is the stage `run_all` assigned it.
    ///
    /// `Stage` exists so the CALL SITE stops inferring "was this the whole
    /// submission?" from `at == run.len()`. That inference was correct and it
    /// was invisible: nothing broke if `run_all` changed which number it sent,
    /// and the only symptom would have been a wrong message. Pinning the two
    /// constructors keeps the mapping in `fire` honest.
    #[test]
    fn the_two_stages_are_a_launch_and_the_whole_submission() {
        use crate::device::Stage;
        assert_ne!(
            Stage::Launch(452),
            Stage::Submission { of: 452 },
            "the whole point is that these two are not the same value"
        );
    }
}
