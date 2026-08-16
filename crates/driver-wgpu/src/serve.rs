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
//! `naga`. So **[`crate::shell::Shell::open`] takes no module argument at all**,
//! there is no directory to find, no `OUT_DIR` to relay, and no build script in
//! this crate where its sibling has one whose only job is to pass
//! `DEP_PIE_KERNELS_VULKAN_SPV_DIR` along. That is not a convenience; it is the
//! deployment story this backend exists for, and it is worth saying loudly
//! because it is invisible in a diff — the argument that is not there.
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
    /// The source for an entrypoint at a tier.
    ///
    /// `None` means this store has not got it, which for a tier above
    /// [`Capability::Baseline`] is the ordinary answer and the caller's cue to
    /// ask again at baseline.
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
) -> Result<Fired, Unfired> {
    let Fire {
        arena,
        resolver,
        geometry,
        tier,
        one_at_a_time,
    } = what;

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
    for (at, launch) in lowered.launches.iter().enumerate() {
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
                sig: hit.sig,
            };
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
                sig: crate::dispatch::row_of(kernels_wgpu::KERNELS, symbol),
            },
        );
        seen.insert(
            symbol,
            crate::device::Read {
                source,
                tier: at_tier,
                declared,
                sig: crate::dispatch::row_of(kernels_wgpu::KERNELS, symbol),
            },
        );
    }

    for (at, launch) in lowered.launches.iter().enumerate() {
        let symbol = lowered.kernels[launch.kernel as usize].as_str();
        // Present by construction: the pass above inserted every symbol this
        // loop can name, and refused the fire if it could not.
        let read = &seen[symbol];
        let (source, at_tier, declared, sig) =
            (read.source.clone(), read.tier, &read.declared, read.sig);
        let planned_one = crate::dispatch::plan_one(
            lowered,
            launch,
            kernels_wgpu::KERNELS,
            Built {
                module: crate::geometry::Module::loaded(symbol, declared),
                declared,
                // Once per SYMBOL, not once per launch: `sig_in` walks the
                // table twice and this loop runs 452 times over ten symbols.
                sig,
            },
            Sources {
                arena,
                resolver,
                min_offset: device.min_storage_offset(),
            },
            geometry,
        );
        let d = match planned_one {
            Ok(d) => d,
            Err(why) => {
                return Err(Unfired::Unplannable {
                    at,
                    symbol: symbol.to_owned(),
                    why,
                });
            }
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
        planned.push((symbol.to_owned(), at_tier, source, d));
    }

    record(device, pipelines, &planned, &blocks, one_at_a_time)
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
fn record(
    device: &Device,
    pipelines: &mut Pipelines,
    planned: &[(
        String,
        Capability,
        String,
        crate::dispatch::Dispatch<'_, Buffer>,
    )],
    blocks: &[Option<Buffer>],
    one_at_a_time: bool,
) -> Result<Fired, Unfired> {
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
                _ => &[],
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
        return Ok(Fired {
            dispatches: run.len(),
            submissions,
            shadowed,
        });
    }
    let ran = device.run_all(&run).map_err(|(stage, why)| match stage {
        crate::device::Stage::Launch(at) => Unfired::Refused { at, why },
        crate::device::Stage::Submission { of } => Unfired::Undelivered { of, why },
    })?;
    Ok(Fired {
        dispatches: run.len(),
        // COUNTED, not assumed. This was `1` for as long as the field existed
        // and the queue was getting 735 for a real decode -- see
        // `crate::device::Ran`.
        submissions: ran.buffers,
        shadowed: ran.shadowed,
    })
}

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
pub fn logits(device: &Device, arena: &Buffer, lowered: &Lowered) -> Result<Logits, Unread> {
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
    let bytes = device
        .read_at(arena, exit.at as u64, extent as u64)
        .map_err(Unread::Refused)?;
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
