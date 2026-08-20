//! One fire, from a lowered plan to a submitted command buffer.
//!
//! Everything below this module answers a question about ONE rectangle: which
//! buffers, at which offsets, with which scalars, over which grid. A fire is
//! four hundred to four thousand of them, and assembling them is not a loop over
//! that answer — it is three passes that must happen in order, and the order is
//! not a matter of taste.
//!
//! # Why three passes
//!
//! **Plan everything, then build every pipeline, then record.** Each boundary
//! exists because of a borrow or a lifetime that a single pass gets wrong.
//!
//! A dispatch whose row puts its scalars in a STORAGE buffer needs a buffer
//! holding those scalars, and that buffer must be alive when the queue runs the
//! command buffer. On `wgpu` that is not the use-after-free it is on Vulkan —
//! a `wgpu::BindGroup` holds its resources alive on its own — but a buffer
//! dropped inside the recording loop is still a buffer nothing can read back or
//! restage, so every block is allocated in the first pass and lives until the
//! submission completes.
//!
//! [`Pipelines::get`] takes `&mut self` because it may build, so a caller cannot
//! hold a reference to one pipeline while asking for the next. Recording needs
//! one reference per launch, all alive at once. So the second pass builds every
//! distinct module and the third asks for them through [`Pipelines::peek`],
//! which borrows immutably.
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
//! `naga`. So **[`crate::shell::Shell::open`] takes no module argument at
//! all**, and there is no directory to find. That is not a convenience; it is
//! the deployment story this backend exists for, and it is worth saying loudly
//! because it is invisible in a diff — the argument that is not there.
//!
//! `driver-vulkan` now says the same thing. Its `Shell::open` lost its module
//! argument too, its build script — whose only job was to relay
//! `DEP_PIE_KERNELS_VULKAN_SPV_DIR` — is deleted, and its store is called
//! [`Embedded`] as well. The two arrived from opposite ends: this crate never
//! had a directory, that one had to be talked out of the one it had.
//!
//! [`Modules`] survives as a seam with exactly one implementation that matters,
//! [`Embedded`], which is what every caller uses. It is a trait rather than a
//! direct call because a test needs to inject a module the tree does not have:
//! `a_module_that_is_not_wgsl_is_a_named_refusal_and_not_a_panic` cannot be
//! written against a source table whose entries all parse.
//!
//! It does not choose a [`Capability`] per launch. One tier for the whole fire,
//! because the tier is a property of the device and picking it per module would
//! build two pipelines for the same symbol.
//!
//! # What it does that neither sibling has to
//!
//! Every rectangle of a real plan binds the arena both readable and writable —
//! its input is one range and its output is another — and **WebGPU refuses a
//! dispatch that binds one buffer both READABLE and WRITABLE**, however far
//! apart the two ranges are. Two WRITABLE bindings are fine, which is the way
//! out and the reason the shader tree declares no `var<storage, read>`. See
//! [`crate::device`]'s own section for the rule and the citation.
//!
//! Nothing in this file handles it, and that is deliberate:
//! [`crate::device::Device::run_all`] shadows the read side into a scratch
//! buffer for any dispatch that still needs it, so a fire is written the way a
//! fire would be written on any backend. What this file does is REPORT it —
//! [`Fired::shadowed`] is how many copies a fire paid for — and the number is
//! now ZERO for every real plan, which is worth reporting for exactly the
//! reason the large number was: a cost nobody can see is a cost nobody
//! fixes, and one `read` declaration brings all 451 of them back.

use model_compiler::lower::Lowered;

use crate::binding::{Arena, ParamSlot, Params, Resolve};
use crate::device::{Buffer, Device, Failed, Pipelines, Recorded};
use crate::dispatch::{Built, Geometry, Sources, Undispatchable};
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
    if let Some(source) = modules.source(entrypoint, tier) {
        return Some((source, tier));
    }
    if tier == Capability::Baseline {
        return None;
    }
    modules
        .source(entrypoint, Capability::Baseline)
        .map(|source| (source, Capability::Baseline))
}

/// What a fire needs that the plan does not carry.
#[derive(Debug)]
pub struct Fire<'a, R: Resolve> {
    /// The buffer the plan's offsets are into.
    pub arena: Arena<'a, R::Buffer>,
    /// What answers for weights, cache and tables.
    pub resolver: &'a R,
    /// The model's shape, for the launch rules that need it.
    pub geometry: Geometry,
    /// The tier every pipeline in this fire is built at.
    pub tier: Capability,
    /// Submit every dispatch on its own command buffer instead of recording them
    /// all into one compute pass.
    ///
    /// The debug path, and it is a real one rather than a courtesy — though for
    /// a different reason than its Vulkan counterpart, and the difference is
    /// worth stating because it decides what a disagreement MEANS.
    ///
    /// There, the two paths differ in whether a barrier was written: Vulkan
    /// gives no ordering between dispatches in one command buffer, so the slow
    /// path is what the plan means and the recorded one is what the shell's
    /// barriers make of it. Here `wgpu` inserts the barrier itself, at every
    /// encoding granularity, so the two paths are ordered IDENTICALLY and a
    /// disagreement cannot be a missing barrier. It would be a bug in `wgpu`'s
    /// tracker, or a hazard through memory `wgpu` cannot see — neither of which
    /// this driver can produce, since every buffer it touches goes through a
    /// bind group.
    ///
    /// So the test that runs a real plan both ways is not checking this shell's
    /// synchronisation. It is checking that claim, which is the one thing in the
    /// device half taken from reading somebody else's source rather than from
    /// running something.
    ///
    /// Ruinously slow: a real plan is four thousand rectangles and this is four
    /// thousand submissions with a device wait on each. Not a setting to leave
    /// on.
    pub one_at_a_time: bool,
    /// Record only the FIRST `n` rectangles of the plan, or all of them.
    ///
    /// # What a truncated fire is for
    ///
    /// It is the only way to see an intermediate value on this backend. A
    /// fire's arena is allocated, written by every rectangle in turn, read for
    /// its readout and dropped inside one function, so the whole of what is
    /// observable is the LAST thing written to each offset — and once a
    /// computation has gone wrong, everything after it has gone wrong too.
    ///
    /// Stopping after `n` makes the arena's end state the state AT `n`. A
    /// caller that walks `n` finds the rectangle a value first went wrong at,
    /// which is a question no readback of a whole fire can answer.
    ///
    /// The answer is a MEASUREMENT and not a fire: a truncated plan computes
    /// nothing anybody wants, and its readout is whatever the arena held.
    pub prefix: Option<usize>,
}

// Derived `Clone` and `Copy` would bound `R: Copy`, and a resolver is a pool or
// a weight store -- neither is copyable and neither needs to be, because this
// holds a REFERENCE to one. Written out so that `Fire { one_at_a_time: true,
// ..what }` works, which is the natural way to ask for the same fire the slow
// way.
impl<R: Resolve> Clone for Fire<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Resolve> Copy for Fire<'_, R> {}

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
    /// One, unless [`Fire::one_at_a_time`] asked otherwise — in which case it is
    /// [`Self::dispatches`], and the difference between the two numbers is the
    /// only externally visible sign of which path ran.
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
}

/// Why a fire did not run.
///
/// Every variant names the LAUNCH INDEX as well as the symbol. A plan states the
/// same symbol hundreds of times — `rms_single_row_bfloat16` appears once per
/// layer per norm — so a refusal that named only the symbol would not say which
/// of them, and the interesting question about a failure at rectangle 2891 of
/// 3992 is almost always what came before it.
#[derive(Debug)]
pub enum Unfired {
    /// No module for a symbol the plan states, at any tier.
    NoModule {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
    },
    /// A module this store holds is not WGSL this crate can dispatch.
    Unreadable {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// What was wrong with the module.
        why: crate::reflect::Unreadable,
    },
    /// A rectangle this crate cannot turn into a dispatch.
    Unplannable {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// What the planner said.
        why: Undispatchable,
    },
    /// This crate contradicted itself.
    ///
    /// Not a caller's mistake and not a device's: `plan_one` produces
    /// `Params::Block` and `block_at` together or neither, and a pipeline built
    /// one line above is a pipeline `peek` answers for. Both are unreachable,
    /// and both are stated rather than unwrapped because the consequence of the
    /// first one is binding a fire's scalars over an operand — a wrong answer,
    /// not a crash.
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
    /// before anything was submitted. A failure of the submission itself is
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
            Self::NoModule { at, symbol } => {
                write!(f, "launch {at} names `{symbol}` and no module has it")
            }
            Self::Unreadable { at, symbol, why } => write!(f, "launch {at} (`{symbol}`): {why}"),
            Self::Unplannable { at, symbol, why } => write!(f, "launch {at} (`{symbol}`): {why}"),
            Self::Impossible { at, symbol, what } => write!(f, "launch {at} (`{symbol}`): {what}"),
            Self::Refused { at, why } => write!(f, "launch {at}: {why}"),
            Self::Undelivered { of, why } => {
                write!(f, "the submission of {of} launches: {why}")
            }
        }
    }
}

impl std::error::Error for Unfired {}

/// Plan, build, record and submit a whole lowering, and wait for it.
///
/// # Errors
///
/// [`Unfired`], naming the launch index in every case but one — a failure of
/// the submission itself names the submission, because it has not singled out
/// a launch. A failure part way
/// through the RECORDING has submitted nothing — [`Device::run_all`] checks
/// every dispatch before it encodes any — so unless
/// [`Fire::one_at_a_time`] is set, a refusal means the device did nothing at
/// all. That is a stronger promise than the Vulkan sibling can make and it is
/// worth relying on: a caller retrying a refused fire is retrying from the same
/// state.
pub fn fire<R: Resolve<Buffer = Buffer>, M: Modules>(
    device: &Device,
    pipelines: &mut Pipelines,
    modules: &M,
    lowered: &Lowered,
    what: Fire<'_, R>,
) -> Result<(Fired, Vec<u8>), Unfired> {
    let Fire {
        arena,
        resolver,
        geometry,
        tier,
        one_at_a_time,
        prefix,
    } = what;
    let launches = match prefix {
        Some(n) => &lowered.launches[..n.min(lowered.launches.len())],
        None => &lowered.launches[..],
    };

    // Pass one. Nothing here touches the device except to allocate the scalar
    // blocks, which must outlive the submission.
    let mut planned = Vec::with_capacity(lowered.launches.len());
    let mut blocks: Vec<Option<Buffer>> = Vec::with_capacity(lowered.launches.len());
    // Every DISTINCT module first, expanded and reflected once each.
    //
    // # Why this is a pass of its own
    //
    // Because it used to be inside the loop below, once per LAUNCH, and that
    // was 95% of a decode. A Qwen3-0.6B step fires 452 launches over eleven
    // distinct symbols, and each one was expanding the WGSL from the embedded
    // tree -- includes spliced, `//#if` arms resolved, defines substituted --
    // and then handing the result to `naga` for a full parse. Measured on a
    // real step: `pick` 700 ms, `reflect::declared` 227 ms, everything else in
    // the fire including the GPU 31 ms.
    //
    // The comment that used to sit here said it was "still not where the time
    // goes, because `create_shader_module` does the same parse plus a whole
    // backend translation and is cached". Both halves are true and the
    // conclusion does not follow: the cached thing is paid ONCE PER SYMBOL and
    // this was paid once per LAUNCH, so the comparison was between eleven of
    // one and four hundred and fifty-two of the other.
    //
    // The lifetime worry in that comment was also unfounded. `entrypoint_source`
    // returns an owned `String`, so this map borrows nothing from `modules`.
    let mut seen: std::collections::BTreeMap<&str, crate::device::Read> =
        std::collections::BTreeMap::new();
    // The same sources, shareable. `seen` owns one `String` per distinct
    // symbol and the planning loop needs one per DISPATCH; this is the
    // difference between an `Arc` bump and a copy of the expanded WGSL.
    let mut arcs: std::collections::BTreeMap<&str, std::sync::Arc<str>> =
        std::collections::BTreeMap::new();
    for (at, launch) in launches.iter().enumerate() {
        let symbol = lowered.kernels[launch.kernel as usize].as_str();
        if seen.contains_key(symbol) {
            continue;
        }
        // ...and once per SHELL, not once per step: the expansion and the
        // parse are functions of the key, and a server fires the same nine or
        // ten symbols for the life of a model. Per step this was 13 ms of a
        // 58 ms decode.
        if let Some(hit) = pipelines.module(symbol, tier) {
            let hit = crate::device::Read {
                source: hit.source.clone(),
                tier: hit.tier,
                declared: hit.declared.clone(),
            };
            arcs.insert(symbol, std::sync::Arc::from(hit.source.as_str()));
            seen.insert(symbol, hit);
            continue;
        }
        let Some((source, at_tier)) = pick(modules, symbol, tier) else {
            return Err(Unfired::NoModule {
                at,
                symbol: symbol.to_owned(),
            });
        };
        let declared = match crate::reflect::declared(&source) {
            Ok(d) => d,
            Err(why) => {
                return Err(Unfired::Unreadable {
                    at,
                    symbol: symbol.to_owned(),
                    why,
                });
            }
        };
        pipelines.remember(
            symbol,
            tier,
            crate::device::Read {
                source: source.clone(),
                tier: at_tier,
                declared: declared.clone(),
            },
        );
        arcs.insert(symbol, std::sync::Arc::from(source.as_str()));
        seen.insert(
            symbol,
            crate::device::Read {
                source,
                tier: at_tier,
                declared,
            },
        );
    }

    // The points a BODY fires that no launch names. Kept apart from `seen`
    // so that map stays borrowed for the whole loop: these need a source and
    // a tier to build a pipeline from and nothing else, their block being
    // reflected and memoized by `reflect::point`, which is where
    // `routine::plan` binds against it.
    let mut extra: std::collections::BTreeMap<&str, (std::sync::Arc<str>, Capability)> =
        std::collections::BTreeMap::new();
    for (at, launch) in launches.iter().enumerate() {
        let symbol = lowered.kernels[launch.kernel as usize].as_str();
        // Present by construction: the pass above inserted every symbol this
        // loop can name, and refused the fire if it could not.
        // BORROWED, and the source shared. `seen` is not touched again in
        // this loop -- a point a body fires goes in `extra` -- so the declared
        // block needs no copy, and the source is an `Arc` bump rather than a
        // copy of the expanded WGSL.
        let read = &seen[symbol];
        let (at_tier, declared) = (read.tier, &read.declared);
        let source = std::sync::Arc::clone(&arcs[symbol]);
        let planned_all = crate::dispatch::plan_all(
            lowered,
            launch,
            Built {
                module: crate::geometry::Module::loaded(symbol, declared),
                declared,
            },
            Sources {
                arena,
                resolver,
                min_offset: device.min_storage_offset(),
            },
            geometry,
        );
        let ds = match planned_all {
            Ok(ds) => ds,
            Err(why) => {
                return Err(Unfired::Unplannable {
                    at,
                    symbol: symbol.to_owned(),
                    why,
                });
            }
        };
        for d in ds {
            // A ROUTINE MAY STATE MORE THAN ONE. `attn`'s split decode is two
            // passes over one statement -- slices, then the merge that
            // finishes them -- and the second names a point the LOWERING does
            // not carry, so its module is read here rather than in pass one.
            //
            // The read is cached per shell exactly as pass one's is: `pick`
            // and `reflect::declared` are a WGSL expansion and a full `naga`
            // parse, and paying them per launch is what pass one exists to
            // stop.
            let (src, tier_of) = if d.symbol == symbol {
                (std::sync::Arc::clone(&source), at_tier)
            } else if let Some((src, point_tier)) = extra.get(d.symbol) {
                (std::sync::Arc::clone(src), *point_tier)
            } else {
                // ONCE PER SHELL, through the same cache pass one uses, and
                // this is not a nicety. `pick` expands the WGSL from the
                // embedded tree -- includes spliced, `//#if` arms resolved --
                // and pass one exists because doing that per launch measured
                // 700 ms of a 58 ms step. A body-fired point that missed the
                // cache would pay it per FIRE: measured at two picks a token,
                // it cost 113.1 -> 93.6 tok/s on Llama-3.2-1B at 512 context,
                // which is most of a millisecond and a half for two strings.
                let (src, point_tier) = match pipelines.module(d.symbol, tier) {
                    Some(hit) => (hit.source.clone(), hit.tier),
                    None => {
                        let Some((src, point_tier)) = pick(modules, d.symbol, tier) else {
                            return Err(Unfired::NoModule {
                                at,
                                symbol: d.symbol.to_owned(),
                            });
                        };
                        let point_declared = match crate::reflect::declared(&src) {
                            Ok(one) => one,
                            Err(why) => {
                                return Err(Unfired::Unreadable {
                                    at,
                                    symbol: d.symbol.to_owned(),
                                    why,
                                });
                            }
                        };
                        pipelines.remember(
                            d.symbol,
                            tier,
                            crate::device::Read {
                                source: src.clone(),
                                tier: point_tier,
                                declared: point_declared,
                            },
                        );
                        (src, point_tier)
                    }
                };
                let src: std::sync::Arc<str> = std::sync::Arc::from(src.as_str());
                extra.insert(d.symbol, (std::sync::Arc::clone(&src), point_tier));
                (src, point_tier)
            };
            // Only a STORAGE block gets a buffer here. The uniform case is the
            // ordinary one and its buffer is the device's to make, per dispatch, in
            // `Device::run_all` -- which can own it because that call waits, so the
            // block is alive for exactly as long as the queue needs it and no
            // longer. Vulkan cannot do that: its push constants and its parameter
            // blocks are two different mechanisms and only one of them is a buffer.
            match &d.params {
                Params::Block {
                    bytes,
                    at: ParamSlot::Storage(_),
                } => match device.buffer(bytes) {
                    Ok(b) => blocks.push(Some(b)),
                    Err(why) => return Err(Unfired::Refused { at, why }),
                },
                _ => blocks.push(None),
            }
            planned.push((d.symbol.to_owned(), tier_of, src, d));
        }
    }

    // THE ANSWER, ASKED FOR WITH THE WORK. `Lowered::readout` states where the
    // distributions land, and it states it BEFORE the fire -- so the copy can
    // ride the fire's own command buffer instead of costing a second queue
    // round trip, which measured 0.67 ms of a 11.8 ms decoded token. A range
    // that does not fit the arena the lowering sized is deliberately NOT asked
    // for: `logits` is where that is checked and named, and it falls back to
    // its own `read_at` for the refusal.
    let asked = lowered
        .readout
        .filter(|exit| {
            matches!(exit.bytes, 2 | 4)
                && exit
                    .at
                    .saturating_add(exit.rows as usize * exit.vocab as usize * exit.bytes as usize)
                    <= lowered.arena_bytes
        })
        .map(|exit| {
            (
                arena.buffer,
                exit.at as u64,
                u64::from(exit.rows) * u64::from(exit.vocab) * u64::from(exit.bytes),
            )
        });
    record(device, pipelines, &planned, &blocks, one_at_a_time, asked)
}

/// Build every pipeline, record every dispatch, submit once and wait.
///
/// # Where a step's time goes
///
/// Measured on an RTX 4090, qwen3-0.6B, a one-row decode of 452 launches,
/// release, after the lowering cache, the single command buffer and the
/// removal of the shadow copies:
///
/// | | ms |
/// |---|---|
/// | `plan_one` x 452 | 1.5 |
/// | `check_bindable` x 452 | 0.08 |
/// | **bind groups x 452** | **2.05** |
/// | encoding | 1.01 |
/// | submit | 0.50 |
/// | **the GPU wait** | **~7** |
/// | logits readback | 0.35 |
/// | **whole step** | **12.7** |
///
/// ## What is left, and where it is
///
/// The wait is 452 dispatches, not bandwidth: this model's staged weights are
/// 335 MB and a 4090 reads that in well under a millisecond, so ~15 us a
/// dispatch is the dispatch. **The way down from here is FEWER LAUNCHES**,
/// which is the model text's shape and the compiler's business.
///
/// That sentence was in this doc before any of the above was fixed, against a
/// table measured on llvmpipe that put ~36 ms in `run_all` — and it was wrong
/// then, because three quarters of that was host work this file's callers
/// were doing over and over. It is worth keeping the retraction beside the
/// claim: "nothing more to find here" is the conclusion that stops the
/// looking, and it should be the last one reached rather than the first.
///
/// The one host item still worth having is the bind groups, 2.05 ms of 452
/// `create_bind_group` calls plus a uniform buffer each. They are a function
/// of the lowering, which is now cached, and of the ARENA, which is still a
/// fresh allocation every step — so caching them means giving the arena a
/// lifetime first.
///
/// # That was tried anyway, and here is the hit rate
///
/// A cache in `Device`, one entry per dispatch POSITION, keyed on the resolved
/// binding list — buffer, offset and length per slot, which folds the shadow
/// scratch in — plus the pipeline and the uniform bytes. It hit **0 of 10,000
/// dispatches**, and the misses say exactly why:
///
/// ```text
/// BINDMISS at 118: pipeline true slots true uniform true resolved false
///    slot 3: buffer false off 31488 vs 31488 len 2048 vs 2048
/// ```
///
/// Decode against decode, everything matches but the HANDLE: same pipeline,
/// same slots, same uniform block, same offsets, same lengths, different
/// allocation. The geometry is already perfectly stable and the arena is not,
/// which is the sentence above measured rather than reasoned.
///
/// So the order is fixed: give the arena a lifetime and the cache follows for
/// free, at 100% instead of 0%. And the prize is the 2.05 ms named above — not
/// the tens of milliseconds a decode's per-launch average suggests, which is
/// the mistake this note existed to prevent and which was made again anyway.
///
/// # The same step on an M4 Pro, and why the list above no longer applies
///
/// Llama-3.2-1B at 4 bits, a one-row decode of 244 dispatches, release, after
/// everything above and after the split decode. Timed by bracketing the
/// phases of `Device::run_all_reading` and of `Turns::step`:
///
/// | | us |
/// |---|---|
/// | `lowering::cached` lookup | 0.4 |
/// | `Pool::stage` | 65 |
/// | plan, pass one and two | ~350 |
/// | `check_bindable` + shadows | 11 |
/// | **bind groups x 244** | **85** |
/// | encoding | 7 |
/// | submit | 165 |
/// | **the GPU wait** | **7900** |
/// | logits readback and widening | 50 |
/// | **whole step** | **8500** |
///
/// So the bind groups are 85 us here, not the 2.05 ms the 4090 table names,
/// and the arena-lifetime cache the note above spends a section arguing for
/// would be worth **one percent of one percent** of a token. That plan is
/// retired: the arena came from a pool in the meantime and Metal's descriptor
/// sets are cheap, and either one alone would have been enough.
///
/// THE HOST IS DONE. Every host phase together is 0.7 ms of 8.5, and the wait
/// is 93%. This is worth stating flatly because two host bugs worth twenty
/// tok/s were found in this file within a day of measuring it -- the right
/// conclusion from that is not "keep looking here", it is "the looking now
/// has a number attached and the number is small".
///
/// The wait is 7.9 ms over 244 dispatches, 32 us each, against a dispatch
/// floor measured at ~13 us in `kernels_wgpu::attn`'s split note. Roughly
/// FORTY PERCENT of a decoded token is the launch floor rather than
/// arithmetic, and the lever is the one the section above already names:
/// fewer launches. The decode's fifteen dispatches a layer are two RMS norms
/// on a ONE-workgroup grid, two rope fires, a KV append over eight
/// workgroups, the split and its merge, a `silu_mul`, and seven `qmv`s --
/// q, k, v, o, gate, up, down. Three of those seven are one fused matmul
/// each in every other serving stack.
///
/// ## How much of the wait is the count, measured rather than argued
///
/// [`Fire::prefix`] already truncates a fire to its first n launches, so the
/// wait can be measured AS A FUNCTION OF THE COUNT without changing a kernel:
/// step the prefix by four across a run of decodes and fit the line. Over the
/// settled half of such a run, 44 points from 86 launches to 244:
///
/// ```text
/// wait ~= 1655 us + 23.0 us x launches      (7280 us at 244)
/// ```
///
/// Only 1.7 ms of a decode's wait is independent of how many launches it is
/// spread over. The slope is not all floor -- a later launch does real work
/// too -- but the floor's own share is known independently: the split-K note
/// in `kernels_wgpu::attn` measures a dispatch that does NO extra work at
/// ~13 us. So roughly 13 of the 23 is floor and 10 is arithmetic, and
/// **244 x 13 us = 3.2 ms of a 7.3 ms wait is the launch floor.**
///
/// Two cautions on the method, because it is cheap enough to be re-run wrong.
/// A truncated fire computes garbage, so the KV cache diverges and only the
/// TIMES mean anything. And the first fires of a run are warm-up: fitting
/// them in moves the slope by half.
///
/// The same sweep prices the tail directly. From 240 launches to 244 -- the
/// final norm, the row gather and the lm head over 32064 workgroups -- is
/// about 300 us, which retires the reading that `PIE_WGPU_PROBE_EACH` gives:
/// that mode's submit-and-wait per launch put the lm head at 3 ms, ten times
/// over, because it charges every launch a queue round trip. Per-launch
/// probing ranks launches. It does not price them.
fn record(
    device: &Device,
    pipelines: &mut Pipelines,
    planned: &[(
        String,
        Capability,
        // SHARED, not cloned. A launch's source is the whole expanded WGSL --
        // tens of kilobytes -- and this list has one entry per DISPATCH, so a
        // `String` here was a memcpy of the text per dispatch per step. Two
        // dispatches of one point share the one buffer.
        std::sync::Arc<str>,
        crate::dispatch::Dispatch<'_, Buffer>,
    )],
    blocks: &[Option<Buffer>],
    one_at_a_time: bool,
    // The range to copy out in the fire's OWN command buffer, as
    // `(buffer, offset, length)`. See `fire` for why it is asked for here
    // rather than read afterwards.
    asked: Option<(&Buffer, u64, u64)>,
) -> Result<(Fired, Vec<u8>), Unfired> {
    // Pass two: every distinct module gets a pipeline, so that pass three can
    // hold a reference to all of them at once.
    let mut buffers = Vec::with_capacity(planned.len());
    for (at, ((symbol, tier, source, d), block)) in planned.iter().zip(blocks).enumerate() {
        let mut b = d.buffers.clone();
        if let Some(buf) = block {
            let Some(slot) = d.block_at else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.clone(),
                    what: "its scalars are a storage block and the planner named no slot for it",
                });
            };
            if slot > b.len() {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.clone(),
                    what: "the planner put the scalar block past the module's own bindings",
                });
            }
            b.insert(slot, crate::binding::Bound::whole(buf));
        }
        pipelines
            .get(device, symbol, *tier, source)
            .map_err(|why| Unfired::Refused { at, why })?;
        buffers.push(b);
    }

    // Pass three.
    let mut run = Vec::with_capacity(planned.len());
    for ((symbol, tier, _, d), b) in planned.iter().zip(&buffers) {
        let Some(pipeline) = pipelines.peek(symbol, *tier) else {
            return Err(Unfired::Impossible {
                // A real launch index, not the sentinel `Undelivered` replaced:
                // `run` is being built, so its length IS the index of the entry
                // this iteration would have pushed.
                at: run.len(),
                symbol: symbol.clone(),
                what: "no pipeline, one pass after every pipeline was built",
            });
        };
        run.push(Recorded {
            pipeline,
            buffers: b,
            // Only the UNIFORM case rides here. A storage block is a buffer and
            // is already in `b`, which is the simplification this backend's ABI
            // buys -- see `binding::ParamSlot`.
            uniform: match &d.params {
                Params::Block {
                    bytes,
                    at: ParamSlot::Uniform,
                } => bytes.as_slice(),
                // A storage block does not exclude a uniform one: the GDN
                // prefill pair takes `GdnCoreParams` as a buffer and its two
                // rectangle scalars as a `@group(1)` block. `Dispatch::uniform`
                // is empty for everything else.
                _ => d.uniform.as_slice(),
            },
            groups: d.groups,
        });
    }

    if one_at_a_time {
        let mut shadowed = 0;
        let mut submissions = 0;
        for (at, r) in run.iter().enumerate() {
            let ran = device
                .run_all(std::slice::from_ref(r))
                // A submission of ONE launch does single that launch out, so
                // the stage carries nothing this loop does not already know.
                .map_err(|(_, why)| Unfired::Refused { at, why })?;
            shadowed += ran.shadowed;
            submissions += ran.buffers;
        }
        // One submission per launch already, so there is no fire-wide command
        // buffer to fold the readout copy into; the caller reads for itself.
        return Ok((
            Fired {
                dispatches: run.len(),
                submissions,
                shadowed,
            },
            Vec::new(),
        ));
    }
    let (ran, read) = device
        .run_all_reading(&run, asked)
        .map_err(|(stage, why)| match stage {
            crate::device::Stage::Launch(at) => Unfired::Refused { at, why },
            crate::device::Stage::Submission { of } => Unfired::Undelivered { of, why },
        })?;
    Ok((Fired {
        dispatches: run.len(),
        // COUNTED, not assumed. This was `1` for as long as the field existed
        // and the queue was getting 735 for a real decode -- see
        // `crate::device::Ran`.
        submissions: ran.buffers,
        shadowed: ran.shadowed,
    }, read))
}

/// A fire's distributions, off the arena and widened.
#[derive(Clone, Debug, PartialEq)]
pub struct Logits {
    /// How many distributions: **one per ROW OF THE FIRE**, not one per
    /// readout.
    ///
    /// This doc said *"one per readout, so `Frame::readouts` and
    /// `Lowered::n_requests` and this are the same number"* and that was not
    /// what the code did — [`logits`] sets this from `exit.rows`, the fire's
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

/// Why a fire's logits could not be read.
#[derive(Clone, Debug, PartialEq)]
pub enum Unread {
    /// The text states no exit.
    ///
    /// Not an error about this fire: a text that computes something other than a
    /// distribution is a legitimate text, and the caller asked the wrong question
    /// of it.
    NoExit,
    /// The readout's range runs off the arena it names.
    PastArena {
        /// Where it starts.
        at: usize,
        /// How many bytes it wants.
        extent: usize,
        /// How many the arena has.
        arena: usize,
    },
    /// The exit's elements are a width this crate cannot widen.
    Width(
        /// Bytes per element, as the lowering states it.
        u32,
    ),
    /// The device would not give the bytes back.
    Refused(
        /// What it said.
        Failed,
    ),
}

impl std::fmt::Display for Unread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoExit => write!(f, "this text states no exit, so it has no logits"),
            Self::PastArena { at, extent, arena } => {
                write!(f, "{extent} bytes at {at} run off an arena of {arena}")
            }
            Self::Width(b) => write!(
                f,
                "an exit of {b}-byte elements is not one this crate reads"
            ),
            Self::Refused(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Unread {}

/// Read a fire's logits back off its arena.
///
/// The last mile, and the four numbers it needs are stated by the lowering:
/// [`model_compiler::lower::Readout`] gives a byte offset, a row count, a
/// vocabulary and an element WIDTH.
///
/// That width is the whole reason this is a function rather than a slice of the
/// arena. **Four is not a given.** The read-out can be bf16, because
/// `affine_qmv_fast` writes bf16 and a text's declared dtype does not change
/// what a kernel does — and a reader that assumed f32 got a vocabulary exactly
/// half zeros, which looks like a dead half of a tensor and is really two
/// elements read as one. That defect is recorded in `Readout::bytes`' own doc,
/// which is where this got it from, and it is checked here rather than assumed.
///
/// **bf16 is widened by SHIFT and not by cast.** A bf16 is the top sixteen bits
/// of an f32, so `u16 -> u32 << 16 -> f32::from_bits` is a reinterpretation;
/// `v as f32` is a numeric conversion and would turn the bit pattern `0x3f80`
/// into 16256.0 where it means 1.0. Nothing downstream could tell — both are
/// finite floats.
///
/// Unlike its Vulkan counterpart this reads only the readout's own range rather
/// than the whole arena, because [`Device::read_at`] takes an offset where
/// `Device::read` there does not.
///
/// # Errors
///
/// [`Unread`], with the readback's own failure carried through
/// [`Unread::Refused`] rather than flattened into a range error.
pub fn logits(
    device: &Device,
    arena: &Buffer,
    lowered: &Lowered,
    already: &[u8],
) -> Result<Logits, Unread> {
    let exit = lowered.readout.ok_or(Unread::NoExit)?;
    let rows = exit.rows as usize;
    let vocab = exit.vocab as usize;
    let extent = rows * vocab * exit.bytes as usize;
    // Against the arena the LOWERING sized, not against the buffer -- a caller
    // that allocated a larger arena than the plan asked for would otherwise be
    // told a range past the plan's own end was fine.
    if exit.at.saturating_add(extent) > lowered.arena_bytes {
        return Err(Unread::PastArena {
            at: exit.at,
            extent,
            arena: lowered.arena_bytes,
        });
    }
    if !matches!(exit.bytes, 2 | 4) {
        return Err(Unread::Width(exit.bytes));
    }
    // What the fire already copied out, where it did. `Fired::readout` is the
    // same range read one submission earlier; an empty one means the fire had
    // no command buffer to fold it into, and then this reads for itself.
    let read;
    let bytes: &[u8] = if already.len() == extent {
        already
    } else {
        read = device
            .read_at(arena, exit.at as u64, extent as u64)
            .map_err(Unread::Refused)?;
        &read
    };
    let values = match exit.bytes {
        2 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    };
    Ok(Logits {
        rows,
        vocab,
        values,
    })
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
        let one = Unfired::Refused { at: 452, why }.to_string();

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
