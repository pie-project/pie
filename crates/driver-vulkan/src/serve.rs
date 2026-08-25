//! Where a symbol's SPIR-V comes from, what a fire reports having done, and
//! the one function that puts a walk's dispatches on the card.
//!
//! # `Fire`, `fire`, `fire_reusing`, `plan_and_fire`, `record`, `logits` and
//! `plan_routine` STOOD HERE
//!
//! This module was the legacy fire path: **plan everything, then build every
//! pipeline, then record** -- three passes, in that order, over one lowered
//! plan's four hundred to four thousand rectangles. Each boundary existed
//! because of a borrow a single pass gets wrong, and both are still true of
//! [`run`] below, which is why they are written down here rather than deleted
//! with the code that first met them.
//!
//! A dispatch whose module reads its scalars out of a storage buffer needs a
//! buffer holding those scalars, and that buffer must be alive when the QUEUE
//! runs the command buffer -- not when it is recorded. Allocating it inside a
//! recording loop and freeing it at the end of the iteration is a
//! use-after-free that the validation layer catches and a caller does not.
//!
//! [`crate::device::Pipelines::get`] takes `&mut self` because it may build,
//! so a caller cannot hold a reference to one pipeline while asking for the
//! next -- and recording needs one reference per launch, all alive at once. So
//! the second pass builds every distinct module and the third asks for them
//! through [`crate::device::Pipelines::peek`], which borrows immutably.
//!
//! What went is everything that was about the LOWERING. `fire` and
//! `plan_and_fire` took a `model_compiler::lower::Lowered` and a run of its
//! `Launch`es; `plan_routine` joined one `Launch` against one
//! `kernels_vulkan::routine::Routine`; `logits` read the exit tensor out of the
//! arena by the `Readout` that lowering carried. All of it is deleted with the
//! type. [`crate::walk::fire::Fire::walk`] is the walk now, and what it hands
//! back is a run of [`crate::baker::dispatch::Dispatch`] -- which is what
//! [`run`] takes.
//!
//! One measurement is worth keeping out of `record`'s doc, because it was
//! expensive to get and is not recoverable from the code: on a 244-dispatch
//! decode, **the host is done**. Every host phase together was 0.7 ms of 8.5
//! and the GPU wait was 93%, and the wait is the launch COUNT rather than
//! bandwidth -- truncating a fire to its first n launches and fitting the line
//! over 44 points from 86 to 244 gave `wait ~= 1655 us + 23.0 us x launches`.
//! The way down is FEWER LAUNCHES, which is the model text's shape and the
//! compiler's business.
//!
//! # What this does not do
//!
//! It does not COMPILE modules. `kernels-vulkan` runs `slangc` and embeds the
//! result in its rlib; this crate looks a symbol up in that table and never
//! sees a shader. [`Modules`] is still the seam, because the tier walk is a
//! property of the device and the device is here — [`Embedded`] is the store a
//! server uses, and a `BTreeMap<String, Vec<u8>>` satisfies it too, which is
//! what lets a test narrow the set.
//!
//! It does not decide where modules LIVE. That question is gone: it used to be
//! answered by a boot-config key naming a directory, which named a `target/`
//! tree, which is a build and not a deployment.
//!
//! It does not choose a [`Capability`] per launch. One tier for the whole
//! fire, because the tier is a property of the device and picking it per
//! module would build two pipelines for the same symbol.

use std::collections::BTreeMap;

use crate::device::Failed;
use kernels_vulkan::Capability;

/// Where the SPIR-V for a symbol comes from.
///
/// A trait and not a `&dyn Fn` so that the common case -- a map built once at
/// startup -- is the case with no wrapper in it.
///
/// # The tier is an argument, and for a long time it was not
///
/// `kernels-vulkan` compiles a tiered module beside the baseline one and
/// names it `<entrypoint>.<tag>.spv`, and every store here is keyed by FILE
/// STEM -- so the cooperative-matrix build of `affine_qmm_t_..._bm_32_bn_32`
/// is stored under `affine_qmm_t_..._bm_32_bn_32.coopmat`.
///
/// A plan never names that. A plan states the bare entrypoint, because the
/// tier is a property of the DEVICE and not of the text. So while this method
/// took only a symbol, the lookup could not reach a tiered module even in
/// principle: **all 146 cooperative-matrix modules and all 20 fp16 ones were
/// unreachable**, on every device, in production and in tests alike.
///
/// Nothing failed. `Device::tiers()` still reported `Coopmat` first, `Shell`
/// still set `tier: Coopmat`, the pipeline cache still keyed on it and the
/// seam still advertised it -- every part of the machinery agreed the tier
/// was in use except the one lookup that had to name the file. That is why
/// it survived: it looks exactly like a tier that is on, and it is only
/// visible in a benchmark that refuses to move.
///
/// It was found by measuring. A prefill of 1536 tokens cost the same at a
/// GEMM row tile of 16, 32 and 64 -- 56.1 s, 54.7 s, 54.4 s -- when 32 and 64
/// have a cooperative-matrix module and 16 deliberately does not. A tier
/// that changes nothing when you switch to it is a tier that is not running.
pub trait Modules {
    /// The module a body named, by the artifact it stated.
    ///
    /// THE FIRE PATH'S WHOLE LOOKUP. A claim body composes this name from an
    /// entrypoint and the tier
    /// [`Encode::best`](kernels_vulkan::plane::Encode::best) gave it, so by
    /// the time it arrives the choice is made and there is nothing left to
    /// decide. No walk, no rule to spell a name with, and a name that names
    /// nothing is a refusal — which is what keeps a mis-named artifact an
    /// error instead of a silently slower kernel.
    fn at(&self, file: &str) -> Option<&[u8]>;

    /// The best module for an entrypoint at `tier`, or `None`.
    ///
    /// Walks [`Capability::PREFERENCE`] from `tier` downward and takes the
    /// first the store has, so a device at `Coopmat` still gets the baseline
    /// module for an entrypoint that was never compiled with the extension --
    /// which is most of them, and is what "tiers are additive" means.
    ///
    /// **Not the fire path.** [`Self::at`] is. This survives for the plan's
    /// own questions -- whether a symbol has a module at all, and which tier
    /// answered -- which are asked of a name a plan carries rather than of one
    /// a body composed.
    fn code(&self, symbol: &str, tier: Capability) -> Option<&[u8]>;

    /// WHICH capability supplied [`Self::code`]'s answer.
    ///
    /// The same walk, reporting the tier it stopped at instead of the bytes.
    /// It exists because "the tier is on" was previously inferred rather than
    /// observed: the device test compared a tiered run against a scalar one
    /// and took a bitwise DIFFERENCE as proof the module had loaded. That is
    /// a proxy, and when it failed it said only that the two agreed, which is
    /// the one thing that is never useful to know.
    fn resolved(&self, symbol: &str, tier: Capability) -> Option<Capability>;
}

impl Modules for BTreeMap<String, Vec<u8>> {
    fn at(&self, file: &str) -> Option<&[u8]> {
        self.get(file.strip_suffix(".spv").unwrap_or(file))
            .map(Vec::as_slice)
    }

    fn code(&self, symbol: &str, tier: Capability) -> Option<&[u8]> {
        Capability::PREFERENCE
            .iter()
            .skip_while(|&&c| c != tier)
            .find_map(|&c| match c {
                Capability::Baseline => self.get(symbol),
                other => self.get(&format!("{symbol}.{}", other.tag())),
            })
            .map(Vec::as_slice)
    }

    fn resolved(&self, symbol: &str, tier: Capability) -> Option<Capability> {
        Capability::PREFERENCE
            .iter()
            .skip_while(|&&c| c != tier)
            .find(|&&c| match c {
                Capability::Baseline => self.contains_key(symbol),
                other => self.contains_key(&format!("{symbol}.{}", other.tag())),
            })
            .copied()
    }
}

/// The modules `kernels-vulkan` compiled into its rlib.
///
/// The production store, and a unit struct because it holds nothing: the words
/// are `'static` data in the binary, so "where are the kernels" has no state
/// to keep. See `kernels_vulkan::module` for why they are not a directory.
///
/// The tier walk is here rather than there for the reason the trait's own
/// documentation gives: which tier to ask for is a property of the DEVICE, and
/// `kernels-vulkan` has no device. It hands out a module for an exact tier and
/// this decides how far down to walk.
#[derive(Debug, Clone, Copy, Default)]
pub struct Embedded;

impl Modules for Embedded {
    fn at(&self, file: &str) -> Option<&[u8]> {
        kernels_vulkan::module::at(file)
    }

    fn code(&self, symbol: &str, tier: Capability) -> Option<&[u8]> {
        Capability::PREFERENCE
            .iter()
            .skip_while(|&&c| c != tier)
            .find_map(|&c| kernels_vulkan::code(symbol, c))
    }

    fn resolved(&self, symbol: &str, tier: Capability) -> Option<Capability> {
        Capability::PREFERENCE
            .iter()
            .skip_while(|&&c| c != tier)
            .find(|&&c| kernels_vulkan::code(symbol, c).is_some())
            .copied()
    }
}

/// A boxed store is a store.
///
/// `shell::Shell` held `Box<dyn Modules + Send + Sync>` so that a server could
/// serve from [`Embedded`] while the tier tests served from a map with modules
/// deliberately removed. Without this, every `M: Modules` signature in this
/// file would have to become `M: ?Sized`, which is a change to the whole seam
/// in order to say one thing about one caller.
///
/// `Shell` is deleted and this survives, because the caller it was written for
/// is the one thing about that module that is certainly coming back: whatever
/// assembles a device, a pool and a lane owns a module store it does not want
/// to be generic over.
impl<T: Modules + ?Sized> Modules for Box<T> {
    fn at(&self, file: &str) -> Option<&[u8]> {
        (**self).at(file)
    }

    fn code(&self, symbol: &str, tier: Capability) -> Option<&[u8]> {
        (**self).code(symbol, tier)
    }

    fn resolved(&self, symbol: &str, tier: Capability) -> Option<Capability> {
        (**self).resolved(symbol, tier)
    }
}

// `pub struct Fire<'a, R: Resolve>` STOOD HERE -- what a fire needed that the
// plan did not carry: the `Arena` the plan's offsets were into, the `&R` that
// answered for weights, cache and tables, the `dispatch::Geometry` the launch
// rules read the model's shape out of, the `Capability` every pipeline in the
// fire was built at, and `one_at_a_time`.
//
// THAT LAST FIELD IS THE ONE WORTH CARRYING FORWARD, because it is a finding
// and not a convenience. Vulkan gives NO ORDERING AT ALL between dispatches in
// one command buffer unless a barrier states it, so a fire that computes the
// wrong answer has two suspects -- the plan, and the ordering this crate
// imposed on it. Running the same plan one dispatch at a time separates them:
// `Device::run` waits on a fence, which is the strongest ordering Vulkan has,
// so the slow version is what the plan MEANS and the recorded one is what it
// COSTS. Ruinously slow -- a real plan is four thousand rectangles and that is
// four thousand submissions -- and worth every second of it exactly once.
//
// The successor to the hazard question is not a flag. `baker::dispatch::Touches`
// is the set of regions a dispatch WRITES, computed per statement from the
// operand marks a claim body states, and `baker::dispatch::merge` is what turns
// a run of them into the barriers between. The debug path this flag opened is
// the same walk with `Touches` widened to everything.

/// What a fire did.
///
/// Returned rather than discarded because both numbers are otherwise
/// unobservable, and a caller that cannot observe them cannot tell a fire that
/// ran from one that quietly ran less. Measured: with this function returning
/// `()`, a version that dropped the last launch of a real plan and a version
/// that ignored [`Fire::one_at_a_time`] both passed the test that exists to
/// check them, because the test compared two runs that had each lost the same
/// thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fired {
    /// How many rectangles were recorded. One per launch the plan states.
    pub dispatches: usize,
    /// How many command buffers were submitted.
    ///
    /// One, unless [`Fire::one_at_a_time`] asked otherwise -- in which case it
    /// is [`Self::dispatches`], and the difference between the two numbers is
    /// the only externally visible sign of which path ran.
    pub submissions: usize,
    /// How many rectangles put their scalars in a storage block.
    ///
    /// Reported for the same reason as the other two: it is the number this
    /// fire used to allocate a device buffer for, one each, and a caller that
    /// cannot see it cannot tell one allocation from a hundred and fourteen.
    /// Every one of them is bound out of a single buffer now, and this is
    /// what a test compares [`Device::allocations`] against.
    pub blocks: usize,
    /// How many SPIR-V modules this fire read.
    ///
    /// One per distinct symbol, not one per rectangle: a qwen3 decode states
    /// 452 rectangles over 9 symbols, and reading a module is a walk over a
    /// few thousand words. The two versions dispatch identically and one of
    /// them spent 22 milliseconds a fire doing it, so this number is the only
    /// thing that tells them apart.
    pub parsed: usize,
    /// How many of those symbols resolved ABOVE `Capability::Baseline`.
    ///
    /// The direct answer to "did this fire actually run the tier it was built
    /// at", which nothing reported before. A fire at `Coopmat` on a store
    /// with no cooperative-matrix module is not an error -- tiers are
    /// additive and most entrypoints have only a baseline build -- so the
    /// only way to tell a tier that is ON from one that is merely SELECTED is
    /// to count what it reached.
    pub tiered: usize,
}

/// Why a fire did not run.
///
/// Every variant names the LAUNCH INDEX as well as the symbol. A plan states
/// the same symbol hundreds of times -- `rms_single_row_bfloat16` appears once
/// per layer per norm -- so a refusal that named only the symbol would not say
/// which of them, and the interesting question about a failure at rectangle
/// 2891 of 3992 is almost always what came before it.
#[derive(Debug)]
pub enum Unfired {
    /// No module for a symbol the plan states.
    NoModule {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
    },
    /// A module this store holds is not SPIR-V this crate can read.
    Unreadable {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// What was wrong with the module.
        why: crate::spirv::Malformed,
    },
    /// A dispatch names a region no buffer this fire holds answers for.
    ///
    /// The walk mints a [`crate::baker::marks::Slice`] against a
    /// [`crate::baker::marks::BufferId`], and the id is an index into the
    /// table a caller hands [`run`]. An id past that table means the caller
    /// staged the fire against one set of allocations and walked it against
    /// another, and the honest answer is to name the id: falling back to
    /// buffer 0 would bind the ARENA where a weight belongs, at an offset the
    /// arena is big enough to contain, and compute a plausible number.
    Unbound {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// The id nothing answered for.
        buffer: usize,
    },
    /// The module and the claim body disagree about this launch's scalars.
    ///
    /// THE DEFECT THE WHOLE PARAMETER SIDE EXISTS TO CATCH, and it has no
    /// symptom on this backend: `robustBufferAccess` is on, so a block short of
    /// what the shader reads returns ZEROS rather than faulting.
    /// `tests/device.rs` shows a dispatch accepting one, producing 256 zeros,
    /// and the validation layer saying nothing at all.
    Unplaceable {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// What [`crate::binding::params_from`] said.
        why: crate::binding::Misplaced,
    },
    /// This crate contradicted itself.
    ///
    /// Not a caller's mistake and not a device's, and every case is stated
    /// rather than unwrapped for the same reason: the consequence of guessing
    /// is a fire's scalars bound OVER an operand -- a wrong answer, not a
    /// crash.
    Impossible {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// What did not hold.
        what: &'static str,
    },
    /// The device refused this launch.
    ///
    /// A launch, and only a launch: everything that reaches here was checked
    /// before anything was submitted. A failure of the SUBMISSION is
    /// [`Self::Undelivered`], which is a different variant precisely so that
    /// it cannot be printed as a launch index.
    Refused {
        /// Which launch.
        at: usize,
        /// What failed.
        why: Failed,
    },
    /// The device never finished the submission.
    ///
    /// Not any one launch's refusal. [`crate::device::Device::run_all`]
    /// submits the whole run at once and waits once, so a device that errors
    /// or does not answer has named nothing smaller than the submission.
    ///
    /// This existed before it had a name, as `Refused { at: run.len() }` -- an
    /// index one past the last launch -- and a reader who did not know that
    /// convention read the count as the offending dispatch.
    Undelivered {
        /// How many launches the submission held.
        of: usize,
        /// What failed.
        why: Failed,
    },
    // `Unplannable` STOOD HERE and went with the join that raised it: it
    // carried the deleted `dispatch::Undispatchable` -- a rectangle `plan_one`
    // could not turn into a dispatch. A claim body states its own operands and
    // its own grid, so there is no rectangle left to refuse; what replaces it
    // is `kernels::plane::Refusal`, which a body raises and `walk::Refused`
    // carries out of the walk BEFORE anything is recorded.
}

impl std::fmt::Display for Unfired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModule { at, symbol } => {
                write!(f, "launch {at} names `{symbol}` and no module was given")
            }
            Self::Unreadable { at, symbol, why } => {
                write!(f, "launch {at} (`{symbol}`): {why}")
            }
            Self::Unbound { at, symbol, buffer } => {
                write!(
                    f,
                    "launch {at} (`{symbol}`) names buffer {buffer}, which this fire does not hold"
                )
            }
            Self::Unplaceable { at, symbol, why } => {
                write!(f, "launch {at} (`{symbol}`): {why:?}")
            }
            Self::Impossible { at, symbol, what } => {
                write!(f, "launch {at} (`{symbol}`): {what}")
            }
            Self::Refused { at, why } => write!(f, "launch {at}: {why}"),
            Self::Undelivered { of, why } => {
                write!(f, "the submission of {of} launches: {why}")
            }
        }
    }
}

impl std::error::Error for Unfired {}

/// A fire's distributions, off the arena and widened.
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    /// How many distributions: one per readout, so [`Frame::readouts`] and
    /// [`Lowered::n_requests`] and this are the same number.
    ///
    /// [`Frame::readouts`]: crate::resources::Frame::readouts
    pub rows: usize,
    /// Elements in one distribution, which is the vocabulary.
    pub vocab: usize,
    /// `read.len() * vocab` values, row major, in `read`'s order.
    pub values: Vec<f32>,
    /// Which of the fire's rows `values` holds, ascending.
    ///
    /// # Why this is not always `0..rows`
    ///
    /// It used to be, and that is what a 1024-token prefill costs when it is:
    /// the text computes a distribution for EVERY row -- see the "every row
    /// samples" note in `turns.rs`, which is a workaround for how the epilogue
    /// is spelled and not a choice this crate makes -- so the arena holds 155
    /// million logits and a caller wants one of them. Copying and widening the
    /// other 151,935 rows was half the step.
    ///
    /// So [`logits_of`] reads a stated set instead, and this says which. A row
    /// nobody asked for is not zero and not stale: [`Logits::row`] answers
    /// `None` for it, which is the difference between a caller getting a wrong
    /// distribution and a caller getting an error.
    pub read: Vec<usize>,
}

impl Logits {
    /// One distribution, by the row number the FIRE gave it.
    ///
    /// `None` both for a row past the fire and for a row that was not read
    /// back. The two are not distinguished because a caller can do nothing
    /// different about them: both mean "this driver does not have that
    /// distribution".
    #[must_use]
    pub fn row(&self, at: usize) -> Option<&[f32]> {
        let held = self.read.binary_search(&at).ok()?;
        self.values.get(held * self.vocab..(held + 1) * self.vocab)
    }
}

// `pub enum Unread`, `logits` and `logits_of` STOOD HERE -- the last mile, and
// the four numbers they needed were stated by `model_compiler::lower::Readout`:
// a byte offset, a row count, a vocabulary and an element WIDTH.
//
// THE WIDTH IS THE FINDING, and it is why reading logits was a function rather
// than a slice of the arena. **Four is not a given.** The metal read-out is
// bf16, because `affine_qmv_fast` writes bf16 and a text's declared dtype does
// not change what a kernel does -- and a reader that assumed f32 got a
// vocabulary exactly half zeros, which looks like a dead half of a tensor and
// is really two elements read as one.
//
// `logits_of` read a STATED SET of rows rather than all of them, and that is
// the second number worth keeping. A 1024-token prefill computes a
// distribution for every row -- see the "every row samples" note in
// `crate::turns`, which is a workaround for how an epilogue is spelled and not
// a choice this crate makes -- so the arena holds 155 million logits and a
// caller wants one of them. Copying and widening the other 151,935 rows was
// half the step. [`Logits::read`] is what survives of that: the rows the
// caller asked for, so that a row nobody asked for answers `None` instead of
// answering staleness.
//
// `Unread` went with them: `NoExit` (the text states no exit), `PastArena`
// (the readout's range runs off the arena it names) and `Width` (an element
// width this crate cannot widen). All three were about a `Readout`, and there
// is no `Readout`.
//
// `push_scalars`, `Reflection`, `widen_a_gathers_source` and `plan_routine`
// STOOD BELOW THEM and are the rest of the legacy fire: packing a launch's
// scalars into the push range, caching `(geometry::Module, spirv::Declared)`
// per symbol so a fire parsed each module once, widening a gather's source
// rectangle to the row it read, and the 156-line join that turned one
// `Launch` plus one `kernels_vulkan::routine::Routine` into a recorded
// dispatch. `walk::fire` is the join now, `baker::encode::Encoder` is what a
// body pushes into, and `binding::params_from` is the packing.

/// Record a run of the walk's dispatches, submit them once and wait.
///
/// THE DEVICE HALF OF THE BAKER PATH, and the one function in this crate that
/// takes what [`crate::walk::fire::Fire::walk`] produced and puts it on a card.
/// Everything upstream of it is arithmetic: a `Program`, an arena, a claim body
/// stating an entrypoint and a grid. This is where that becomes a
/// `vkCmdDispatch`.
///
/// `buffers` is indexed by [`crate::baker::marks::BufferId`], because a `Slice`
/// names an ALLOCATION and an offset and only the caller knows which allocation
/// is which — the walk was handed those ids when its arena and its banks were
/// built. An id past the table is [`Unfired::Unbound`].
///
/// # The three passes, and why they are three
///
/// **Plan everything, then build every pipeline, then record.** Each boundary
/// is a borrow a single pass gets wrong, and both were found the hard way by
/// the fire path this replaces:
///
/// * A dispatch whose scalars go in a STORAGE BLOCK needs a buffer holding
///   them, and that buffer must be alive when the QUEUE runs the command
///   buffer, not when it is recorded. Every block of the run is written into
///   ONE buffer here, allocated after pass one and freed after the submission
///   has completed — [`crate::device::Device::run_all`] waits on a fence, which
///   is what makes that safe.
/// * [`crate::device::Pipelines::get`] takes `&mut self` because it may build,
///   so a caller cannot hold a reference to one pipeline while asking for the
///   next, and recording needs one reference per launch all alive at once. So
///   pass two builds every distinct module and pass three asks through
///   [`crate::device::Pipelines::peek`], which borrows immutably.
///
/// # The grid is the body's lanes over the module's own workgroup
///
/// A `#[claims]` body states TOTAL INVOCATIONS
/// ([`Dispatch::lanes`](crate::baker::dispatch::Dispatch::lanes)) and the module
/// declares `[numthreads]`; the division is here because only this side has read
/// the module. It is done per AXIS and only on the axes the module is actually
/// indexed by ([`crate::spirv::Declared::grid_axes`]) — a grid on the wrong axis
/// is this crate's own recorded defect, and it returned success while leaving
/// every row but the first holding the zeros its buffer was born with.
///
/// # Errors
///
/// [`Unfired`], naming the launch index in every case but the submission's. A
/// failure part way through has still recorded and possibly RUN everything
/// before it: Vulkan has no way to undo a submitted command buffer, and this
/// does not pretend otherwise.
pub fn run<M: Modules>(
    device: &crate::device::Device,
    pipelines: &mut crate::device::Pipelines,
    modules: &M,
    buffers: &[&crate::device::Buffer],
    dispatches: &[crate::baker::dispatch::Dispatch],
    tier: Capability,
) -> Result<Fired, Unfired> {
    use crate::binding::{Params, params_from};
    use crate::device::Bound;

    let align = device.min_storage_offset().max(1);

    // ── pass one ──────────────────────────────────────────────────────
    //
    // Every module read ONCE, by the file the BODY named. Spelling that name a
    // second way is what let the tier and the module disagree for the life of
    // this crate: 146 cooperative-matrix modules and 20 fp16 ones were dead on
    // every device from the first commit, and nothing failed.
    let planning = crate::phase::span("run/plan");
    let mut declared: BTreeMap<&'static str, crate::spirv::Declared> = BTreeMap::new();
    let mut bound: Vec<Vec<Bound<'_>>> = Vec::with_capacity(dispatches.len());
    let mut writes: Vec<Vec<bool>> = Vec::with_capacity(dispatches.len());
    let mut placed: Vec<Params> = Vec::with_capacity(dispatches.len());
    let mut grids: Vec<[u32; 3]> = Vec::with_capacity(dispatches.len());
    let mut block = Vec::<u8>::new();
    // `(binding slot, offset, len)` for each dispatch whose scalars are a
    // struct, and `None` for the push-block and no-scalar cases.
    let mut spans: Vec<Option<(usize, u64, u64)>> = Vec::with_capacity(dispatches.len());

    for (at, d) in dispatches.iter().enumerate() {
        if !declared.contains_key(d.file) {
            let code = modules.at(d.file).ok_or_else(|| Unfired::NoModule {
                at,
                symbol: d.symbol.to_owned(),
            })?;
            let words = crate::spirv::words(code).map_err(|why| Unfired::Unreadable {
                at,
                symbol: d.symbol.to_owned(),
                why,
            })?;
            let module = crate::spirv::declared(&words).map_err(|why| Unfired::Unreadable {
                at,
                symbol: d.symbol.to_owned(),
                why,
            })?;
            declared.insert(d.file, module);
        }
        let module = &declared[d.file];

        let mut args = Vec::with_capacity(d.args.len());
        let mut w = Vec::with_capacity(d.args.len());
        for a in &d.args {
            let held = buffers
                .get(a.slice.buffer.0 as usize)
                .ok_or_else(|| Unfired::Unbound {
                    at,
                    symbol: d.symbol.to_owned(),
                    buffer: a.slice.buffer.0 as usize,
                })?;
            args.push(
                Bound::within(held, a.slice.at, a.slice.bytes, align)
                    .map_err(|why| Unfired::Refused { at, why })?,
            );
            // THE DIRECTION IS THE ENCODER'S AND IT ALREADY SAID SO.
            // `Touches::writes` is every region a body marked `.arg_mut()`, and
            // this reads the same set back per operand. An empty `writes` is
            // read by `run_all`'s barriers as "all of them", which is safe and
            // orders every pair; stating it is what lets two reads of one
            // buffer run at once.
            //
            // CONTAINMENT AND NOT EQUALITY. `super::baker::dispatch::merge`
            // keeps one entry per `(buffer, at)` and widens it to the largest
            // extent seen, so a region that went in can come back out LONGER
            // than it was. Comparing for equality would answer `false` for it
            // -- an operand the shader writes reported as read-only, which is
            // a barrier this driver would not place. The containment test
            // cannot under-report, which is the only direction that matters.
            w.push(d.touches.writes.iter().any(|s| {
                s.buffer == a.slice.buffer
                    && s.at <= a.slice.at
                    && s.at + s.bytes >= a.slice.at + a.slice.bytes
            }));
        }

        let params = params_from(&d.params, module).map_err(|why| Unfired::Unplaceable {
            at,
            symbol: d.symbol.to_owned(),
            why,
        })?;
        spans.push(match &params {
            Params::Block { bytes, at: slot } => {
                // Each block starts where a descriptor can be built, which is
                // the DEVICE's alignment and not four bytes: they share one
                // buffer, so the padding between them is what makes the second
                // one addressable at all.
                let pad = block.len() as u64 % align;
                if pad != 0 {
                    block.resize(block.len() + (align - pad) as usize, 0);
                }
                let offset = block.len() as u64;
                block.extend_from_slice(bytes);
                Some((*slot, offset, bytes.len() as u64))
            }
            Params::Push(_) | Params::None => None,
        });
        placed.push(params);

        // ONE PER AXIS THE MODULE IS INDEXED BY, and ONE on every other. An
        // axis the shader never reads is a dimension of the grid it does not
        // walk, so dividing there would multiply the dispatch count by a number
        // nothing consumes; leaving it at one is what makes a `[n, 1, 1]` body
        // and a `[n, m, 1]` body the same code path here.
        let mut groups = [1u32; 3];
        for (axis, group) in groups.iter_mut().enumerate() {
            if module.grid_axes[axis] {
                *group = d.lanes[axis].div_ceil(module.local[axis].max(1));
            }
        }
        grids.push(groups);
        bound.push(args);
        writes.push(w);
    }
    drop(planning);

    // ONE ALLOCATION FOR EVERY BLOCK IN THE RUN. A buffer per dispatch is 114
    // allocations on a real plan for the same bytes; what a caller sees of the
    // difference is [`Fired::blocks`] against `Device::allocations`.
    let blocks = spans.iter().filter(|s| s.is_some()).count();
    let held = if block.is_empty() {
        None
    } else {
        // NAMED AGAINST THE FIRST LAUNCH THAT ASKED FOR ONE, and not against
        // launch zero. `Refused` means "the device refused THIS launch" and a
        // reader following that convention would go and look at a dispatch
        // whose scalars are push constants.
        let asked = spans.iter().position(Option::is_some).unwrap_or(0);
        Some(
            device
                .buffer(&block)
                .map_err(|why| Unfired::Refused { at: asked, why })?,
        )
    };

    let fired = submit(
        device,
        pipelines,
        modules,
        dispatches,
        &declared,
        &mut bound,
        &mut writes,
        &placed,
        &spans,
        &grids,
        held.as_ref(),
        align,
        tier,
        blocks,
    );
    if let Some(b) = held {
        // AFTER the submission and not after the recording: `run_all` waits on
        // a fence, so by here the queue is done with these bytes. Freeing on
        // the other side of it is a use-after-free the validation layer catches
        // and a caller does not.
        device.free(b);
    }
    fired
}

/// How many of a module's bindings BELOW `raw` it actually reads.
///
/// The translation between the module's own `binding = N` and the position of
/// that binding in a DENSE operand run. `declared.used` is indexed by binding
/// number and is `bindings` long, so a hole reads as `false` rather than as an
/// absence -- see [`crate::spirv::Declared::used`], whose own doc carries why
/// a Vulkan driver has to know the difference at all.
///
/// A `raw` past what the module declares answers the whole count, which the
/// caller then refuses against the run's length rather than silently seating.
fn declared_used(module: &crate::spirv::Declared, raw: usize) -> usize {
    module.used.iter().take(raw).filter(|read| **read).count()
}

/// Passes two and three of [`run`].
///
/// Split out so that the block buffer's life is one statement in the caller and
/// the free happens on both the `Ok` and the `Err` path without a `Drop` type
/// whose only job is to hold a device reference.
#[allow(clippy::too_many_arguments)]
fn submit<'b, M: Modules>(
    device: &crate::device::Device,
    pipelines: &mut crate::device::Pipelines,
    modules: &M,
    dispatches: &[crate::baker::dispatch::Dispatch],
    declared: &BTreeMap<&'static str, crate::spirv::Declared>,
    bound: &mut [Vec<crate::device::Bound<'b>>],
    writes: &mut [Vec<bool>],
    placed: &[crate::binding::Params],
    spans: &[Option<(usize, u64, u64)>],
    grids: &[[u32; 3]],
    block: Option<&'b crate::device::Buffer>,
    align: u64,
    tier: Capability,
    blocks: usize,
) -> Result<Fired, Unfired> {
    use crate::binding::Params;
    use crate::device::{Bound, Recorded};

    // ── pass two ──────────────────────────────────────────────────────
    let building = crate::phase::span("run/pipelines");
    for (at, d) in dispatches.iter().enumerate() {
        if let Some((slot, offset, len)) = spans[at] {
            let Some(module) = declared.get(d.file) else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: d.symbol.to_owned(),
                    what: "no reflection, one pass after every module was read",
                });
            };
            let Some(buffer) = block else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: d.symbol.to_owned(),
                    what: "its scalars are a block and pass one allocated no block buffer",
                });
            };
            // THE SPAN AND NOT THE WHOLE BUFFER. Every block of the run is in
            // there, so `WHOLE_SIZE` -- or a length past this one's -- would
            // let a shader reading one word too far read the NEXT rectangle's
            // scalars, which are plausible numbers.
            let region = Bound::within(buffer, offset, len, align)
                .map_err(|why| Unfired::Refused { at, why })?;
            // THE SEAT IS THE DENSE ONE, NOT THE RAW BINDING NUMBER, and the
            // two are the same number only while no module has a hole below
            // its parameter block.
            //
            // `params_from` finds the block by SIZE and answers the module's
            // own `binding = N`. This list is the body's DENSE run --
            // `Recorded::buffers` is "one range per binding the module reads,
            // less its holes", and `device::slots` re-expands it by skipping
            // the unread ones. So the position to insert at is how many
            // bindings BELOW `slot` the module actually reads.
            //
            // It is exactly `slot` for every entry point in this tree today,
            // and that is a measurement rather than a hope: every module with a
            // hole (`affine_qmm_t*`, `kv_append_paged`, `affine_qmv_routed`,
            // `qmm_splitk_reduce`, `sdpa_paged_decode_*`) takes push constants,
            // and every module with a storage block (`argmax_logits`,
            // `combine_sorted`, `route_sort`, `gdn_*`, `geglu_tanh*`,
            // `gated_rms*`, `rms_*`) declares a contiguous set. Computing it
            // rather than assuming it is what keeps the first kernel to combine
            // the two from binding a fire's scalars OVER an operand -- a wrong
            // answer that `robustBufferAccess` has nothing to say about,
            // because nothing is out of bounds.
            let seat = declared_used(module, slot);
            if seat > bound[at].len() {
                return Err(Unfired::Impossible {
                    at,
                    symbol: d.symbol.to_owned(),
                    what: "its scalar block sits at a binding past the operands the body bound",
                });
            }
            bound[at].insert(seat, region);
            // Read-only: the shader reads its parameters out of it and this
            // driver is the only thing that ever writes it, on the host, before
            // the run is submitted.
            writes[at].insert(seat, false);
        }
        let push = match &placed[at] {
            Params::Push(p) => p.len() as u32,
            Params::Block { .. } | Params::None => 0,
        };
        // ASKED ONLY WHEN THERE IS SOMETHING TO ASK FOR. `Pipelines::get`
        // builds on a miss and returns the held pipeline on a hit, so from the
        // second fire onward this loop's whole job is cache hits -- and it was
        // paying for the module bytes before every one of them. `peek` is the
        // same lookup `get` starts with and takes `&self`, so this is not a
        // second cache: it is the miss test, hoisted above the work only a miss
        // uses.
        if pipelines.peek(d.symbol, tier).is_none() {
            let code = modules.at(d.file).ok_or_else(|| Unfired::NoModule {
                at,
                symbol: d.symbol.to_owned(),
            })?;
            pipelines
                .get(device, d.symbol, code, push, bound[at].len() as u32, tier)
                .map_err(|why| Unfired::Refused { at, why })?;
        }
    }
    drop(building);

    // ── pass three ────────────────────────────────────────────────────
    let recording = crate::phase::span("run/recorded");
    let mut run = Vec::with_capacity(dispatches.len());
    for (at, d) in dispatches.iter().enumerate() {
        let Some(pipeline) = pipelines.peek(d.symbol, tier) else {
            return Err(Unfired::Impossible {
                at,
                symbol: d.symbol.to_owned(),
                what: "no pipeline, one pass after every pipeline was built",
            });
        };
        run.push(Recorded {
            symbol: d.symbol,
            pipeline,
            buffers: &bound[at],
            writes: &writes[at],
            push: match &placed[at] {
                Params::Push(p) => p,
                Params::Block { .. } | Params::None => &[],
            },
            groups: grids[at],
        });
    }
    drop(recording);

    {
        let _s = crate::phase::span("run/run_all");
        let launches = run.len();
        device.run_all(&run).map_err(|(at, why)| {
            // The submission's own failure names the SUBMISSION. `run_all`
            // reports `run.len()` for it -- an index one past the last launch
            // -- and a reader who does not know that convention reads the count
            // as the offending dispatch.
            if at >= launches {
                Unfired::Undelivered { of: launches, why }
            } else {
                Unfired::Refused { at, why }
            }
        })?;
    }

    Ok(Fired {
        dispatches: run.len(),
        submissions: 1,
        blocks,
        parsed: dispatches
            .iter()
            .map(|d| d.file)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        // A FILE WHOSE STEM IS NOT ITS ENTRYPOINT IS A TIERED BUILD.
        // `Capability::module` spells the baseline one `<entrypoint>.spv` and
        // every other `<entrypoint>.<tag>.spv`, and the BODY composed the name,
        // so this counts what the bodies REACHED rather than what the device
        // advertised. A fire at `Coopmat` over a tree with no cooperative-matrix
        // module is not an error -- tiers are additive -- so the only way to
        // tell a tier that is ON from one that is merely SELECTED is to count
        // what it reached.
        tiered: dispatches
            .iter()
            .filter(|d| {
                d.file
                    .strip_suffix(".spv")
                    .is_none_or(|stem| stem != d.symbol)
            })
            .map(|d| d.file)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
    })
}
