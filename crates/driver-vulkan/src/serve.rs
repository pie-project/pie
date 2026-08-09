//! One fire, from a lowered plan to a submitted command buffer.
//!
//! Everything below this module answers a question about ONE rectangle: which
//! buffers, at which offsets, with which scalars, over which grid. A fire is
//! four hundred to four thousand of them, and assembling them is not a loop
//! over that answer -- it is three passes that must happen in order, and the
//! order is not a matter of taste.
//!
//! # Why three passes
//!
//! **Plan everything, then build every pipeline, then record.** Each boundary
//! exists because of a borrow or a lifetime that a single pass gets wrong.
//!
//! A dispatch whose module reads its scalars out of a storage buffer needs a
//! buffer holding those scalars, and that buffer must be alive when the queue
//! runs the command buffer -- not when it is recorded. Allocating it inside a
//! recording loop and freeing it at the end of the iteration is a
//! use-after-free that the validation layer catches and a caller does not, so
//! every block is allocated in the first pass and freed after the submission
//! completes.
//!
//! [`Pipelines::get`] takes `&mut self` because it may build, so a caller
//! cannot hold a reference to one pipeline while asking for the next.
//! Recording needs one reference per launch, all alive at once. So the second
//! pass builds every distinct module and the third asks for them through
//! [`Pipelines::peek`], which borrows immutably.
//!
//! This module exists because that shape was worked out inside a test, which
//! is the wrong place for it: a test that assembles a fire is testing its own
//! assembly, and the next caller writes it again and gets one of the two
//! boundaries wrong.
//!
//! # What this does not do
//!
//! It does not load modules. `kernels-vulkan` compiles them and a server
//! decides where they live; a driver that read a directory would be a driver
//! with an opinion about deployment. [`Modules`] is the seam, and a
//! `BTreeMap<String, Vec<u8>>` satisfies it.
//!
//! It does not choose a [`Capability`] per launch. One tier for the whole
//! fire, because the tier is a property of the device and picking it per
//! module would build two pipelines for the same symbol.

use std::collections::BTreeMap;

use model_compiler::lower::Lowered;

use crate::binding::{Arena, Params, Resolve};
use crate::device::{Bound, Device, Failed, Pipelines, Recorded};
use crate::dispatch::{Built, Geometry, Sources, Undispatchable};
use kernels_vulkan::Capability;

/// Where the SPIR-V for a symbol comes from.
///
/// A trait and not a `&dyn Fn` so that the common case -- a map built once at
/// startup -- is the case with no wrapper in it.
pub trait Modules {
    /// The module for an entrypoint, or `None` if this store has not got it.
    fn code(&self, symbol: &str) -> Option<&[u8]>;
}

impl Modules for BTreeMap<String, Vec<u8>> {
    fn code(&self, symbol: &str) -> Option<&[u8]> {
        self.get(symbol).map(Vec::as_slice)
    }
}

/// What a fire needs that the plan does not carry.
#[derive(Debug)]
pub struct Fire<'a, R: Resolve> {
    /// The buffer the plan's offsets are into.
    pub arena: Arena<'a>,
    /// What answers for weights, cache and tables.
    pub resolver: &'a R,
    /// The model's shape, for the launch rules that need it.
    pub geometry: Geometry,
    /// The tier every pipeline in this fire is built at.
    pub tier: Capability,
    /// Submit every dispatch on its own fence instead of recording them all
    /// into one command buffer.
    ///
    /// The debug path, and it is a real one rather than a courtesy. Vulkan
    /// gives no ordering at all between dispatches in one command buffer
    /// unless a barrier states it, so a fire that computes the wrong answer
    /// has two suspects: the plan, and the ordering this crate imposed on it.
    /// Running the same plan one dispatch at a time separates them --
    /// [`Device::run`] waits on a fence, which is the strongest ordering
    /// Vulkan has, so this is what the plan MEANS and the recorded version is
    /// what it COSTS.
    ///
    /// Ruinously slow: a real plan is four thousand rectangles and this is
    /// four thousand submissions. Not a setting to leave on.
    pub one_at_a_time: bool,
}

// Derived `Clone` and `Copy` would bound `R: Copy`, and a resolver is a pool
// or a weight store -- neither is copyable and neither needs to be, because
// this holds a REFERENCE to one. Written out so that `Fire { one_at_a_time:
// true, ..what }` works, which is the natural way to ask for the same fire the
// slow way.
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
    /// `Params::Block` and `block_at` together or neither, and a pipeline
    /// built one line above is a pipeline `peek` answers for. Both are
    /// unreachable, and both are stated rather than unwrapped because the
    /// consequence of the first one is binding a fire's scalars over an
    /// operand -- a wrong answer, not a crash.
    Impossible {
        /// Which launch.
        at: usize,
        /// The symbol it names.
        symbol: String,
        /// What did not hold.
        what: &'static str,
    },
    /// A Vulkan call failed.
    Refused {
        /// Which launch, or the length of the plan if it was the submission.
        at: usize,
        /// What failed.
        why: Failed,
    },
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
            Self::Unplannable { at, symbol, why } => {
                write!(f, "launch {at} (`{symbol}`): {why}")
            }
            Self::Impossible { at, symbol, what } => {
                write!(f, "launch {at} (`{symbol}`): {what}")
            }
            Self::Refused { at, why } => write!(f, "launch {at}: {why}"),
        }
    }
}

impl std::error::Error for Unfired {}

/// Plan, build, record and submit a whole lowering, and wait for it.
///
/// The blocks this allocates are freed before it returns, and they are freed
/// AFTER the submission has completed rather than after it was recorded --
/// [`Device::run_all`] waits on a fence, which is what makes that safe. A
/// version of this that returned before the queue finished would have to hand
/// the blocks back to the caller, and a caller that dropped them on the wrong
/// side of the fence would be freeing memory the GPU is reading.
///
/// # Errors
///
/// [`Unfired`], naming the launch index in every case. A failure part way
/// through has still recorded and possibly RUN everything before it: Vulkan
/// has no way to undo a submitted command buffer, and this does not pretend
/// otherwise.
pub fn fire<R: Resolve, M: Modules>(
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
    let mut blocks: Vec<Option<crate::device::Buffer>> = Vec::with_capacity(planned.capacity());
    let give_back = |device: &Device, blocks: Vec<Option<crate::device::Buffer>>| {
        for b in blocks.into_iter().flatten() {
            device.free(b);
        }
    };
    for (at, launch) in lowered.launches.iter().enumerate() {
        let symbol = lowered.kernels[launch.kernel as usize].as_str();
        let Some(code) = modules.code(symbol) else {
            give_back(device, blocks);
            return Err(Unfired::NoModule {
                at,
                symbol: symbol.to_owned(),
            });
        };
        // Read per launch rather than cached per symbol. Measured on a real
        // plan of 3992 rectangles over 19 distinct symbols: reading the
        // module is a walk over a few thousand words and does not show
        // against the pipeline builds. Caching it would be a `BTreeMap` whose
        // entries borrow from `modules`, which is a lifetime this signature
        // does not need to carry.
        let declared = match crate::spirv::words(code).and_then(|w| crate::spirv::declared(&w)) {
            Ok(d) => d,
            Err(why) => {
                give_back(device, blocks);
                return Err(Unfired::Unreadable {
                    at,
                    symbol: symbol.to_owned(),
                    why,
                });
            }
        };
        let planned_one = crate::dispatch::plan_one(
            lowered,
            launch,
            kernels_vulkan::KERNELS,
            Built {
                module: crate::geometry::Module::named(
                    symbol,
                    [declared.local[0], declared.local[1], declared.local[2]],
                ),
                declared: &declared,
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
                give_back(device, blocks);
                return Err(Unfired::Unplannable {
                    at,
                    symbol: symbol.to_owned(),
                    why,
                });
            }
        };
        match &d.params {
            Params::Block { bytes, .. } => match device.buffer(bytes) {
                Ok(b) => blocks.push(Some(b)),
                Err(why) => {
                    give_back(device, blocks);
                    return Err(Unfired::Refused { at, why });
                }
            },
            _ => blocks.push(None),
        }
        planned.push((symbol.to_owned(), d));
    }

    // Passes two and three are a separate function so that every borrow of
    // `blocks` -- and `Bound::whole` takes one per scalar block -- ends before
    // the frees below. Written inline, the borrow checker is right to refuse:
    // the recorded buffers name that memory, and freeing it while they do is
    // the exact defect this shape exists to prevent.
    //
    // Measured, so this is not a guess: moving the `give_back` below to before
    // this line does not compile, and the message is `borrow of moved value:
    // blocks`. The one ordering rule in this module that Vulkan will not
    // enforce is the one Rust does.
    let outcome = record(
        device,
        pipelines,
        modules,
        &planned,
        &blocks,
        tier,
        one_at_a_time,
    );
    // After the fence and not before: `run_all` waits, so by here the queue
    // is done with every block. This is the whole reason the blocks are owned
    // by this function rather than returned.
    give_back(device, blocks);
    outcome
}

/// Build every pipeline, record every dispatch, submit once and wait.
fn record<M: Modules>(
    device: &Device,
    pipelines: &mut Pipelines,
    modules: &M,
    planned: &[(String, crate::dispatch::Dispatch<'_>)],
    blocks: &[Option<crate::device::Buffer>],
    tier: Capability,
    one_at_a_time: bool,
) -> Result<Fired, Unfired> {
    // Pass two: every distinct module gets a pipeline, so that pass three can
    // hold a reference to all of them at once.
    let mut buffers = Vec::with_capacity(planned.len());
    for (at, ((symbol, d), block)) in planned.iter().zip(blocks).enumerate() {
        let mut b = d.buffers.clone();
        if let Some(buf) = block {
            let Some(slot) = d.block_at else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.clone(),
                    what: "its scalars are a block and the planner named no slot for it",
                });
            };
            b.insert(slot, Bound::whole(buf));
        }
        let push = match &d.params {
            Params::Push(p) => p.len() as u32,
            _ => 0,
        };
        // `modules.code` answered for this symbol in pass one, so it answers
        // here.
        let code = modules.code(symbol).unwrap_or_default();
        pipelines
            .get(device, symbol, code, push, b.len() as u32, tier)
            .map_err(|why| Unfired::Refused { at, why })?;
        buffers.push(b);
    }

    // Pass three.
    let mut run = Vec::with_capacity(planned.len());
    for ((symbol, d), b) in planned.iter().zip(&buffers) {
        let Some(pipeline) = pipelines.peek(symbol, tier) else {
            return Err(Unfired::Impossible {
                at: run.len(),
                symbol: symbol.clone(),
                what: "no pipeline, one pass after every pipeline was built",
            });
        };
        run.push(Recorded {
            pipeline,
            buffers: b,
            push: match &d.params {
                Params::Push(p) => p,
                _ => &[],
            },
            groups: d.groups,
        });
    }

    if one_at_a_time {
        for (at, r) in run.iter().enumerate() {
            device
                .run(r.pipeline, r.buffers, r.push, r.groups)
                .map_err(|why| Unfired::Refused { at, why })?;
        }
        return Ok(Fired {
            dispatches: run.len(),
            submissions: run.len(),
        });
    }
    device
        .run_all(&run)
        .map_err(|(at, why)| Unfired::Refused { at, why })?;
    Ok(Fired {
        dispatches: run.len(),
        submissions: 1,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unread {
    /// The text states no exit.
    ///
    /// Not an error about this fire: a text that computes something other than
    /// a distribution is a legitimate text, and the caller asked the wrong
    /// question of it.
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
    Width(u32),
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
/// That width is the whole reason this is a function rather than a slice of
/// the arena. **Four is not a given.** The metal read-out is bf16, because
/// `affine_qmv_fast` writes bf16 and a text's declared dtype does not change
/// what a kernel does -- and a reader that assumed f32 got a vocabulary
/// exactly half zeros, which looks like a dead half of a tensor and is really
/// two elements read as one. That defect is recorded in `Readout::bytes`'
/// own doc, which is where this got it from, and it is checked here rather
/// than assumed.
///
/// This reads the WHOLE arena and slices it. `Device::read` has no offset, and
/// an arena is tens of megabytes against a vocabulary of a few hundred
/// kilobytes, so this is wasteful and is not on any path that matters: a fire
/// is read once.
///
/// # Errors
///
/// [`Unread`], and [`Failed`] is folded into it only through
/// [`Unread::PastArena`] -- the read itself is checked before it is asked for.
pub fn logits(
    device: &Device,
    arena: &crate::device::Buffer,
    lowered: &Lowered,
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
    let whole = device.read(arena).map_err(|_| Unread::PastArena {
        at: exit.at,
        extent,
        arena: lowered.arena_bytes,
    })?;
    let bytes = whole
        .get(exit.at..exit.at + extent)
        .ok_or(Unread::PastArena {
            at: exit.at,
            extent,
            arena: whole.len(),
        })?;
    let values = match exit.bytes {
        // bf16 is the TOP half of an f32, so widening is a shift and not a
        // conversion. Written as bits rather than through a cast because
        // `u16 as f32` is a numeric conversion and this is a reinterpretation.
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
