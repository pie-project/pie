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
use crate::dispatch::{Geometry, Undispatchable};
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
    /// The best module for an entrypoint at `tier`, or `None`.
    ///
    /// Walks [`Capability::PREFERENCE`] from `tier` downward and takes the
    /// first the store has, so a device at `Coopmat` still gets the baseline
    /// module for an entrypoint that was never compiled with the extension --
    /// which is most of them, and is what "tiers are additive" means.
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
/// The block buffer this allocates is freed before it returns, and it is freed
/// AFTER the submission has completed rather than after it was recorded --
/// [`Device::run_all`] waits on a fence, which is what makes that safe. A
/// version of this that returned before the queue finished would have to hand
/// it back to the caller, and a caller that dropped it on the wrong side of
/// the fence would be freeing memory the GPU is reading.
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

    // Pass one. Nothing here touches the device at all: the scalar blocks are
    // gathered into ONE host buffer, allocated once after the walk.
    //
    // They used to be a buffer each. Measured on this card, a real decode of
    // 452 rectangles states 114 of them and allocating them cost 30
    // milliseconds -- 260 microseconds an allocation, which is what
    // `vkCreateBuffer` plus `vkAllocateMemory` plus a map and an unmap costs
    // when it is asked for 40 bytes. That was more than the recording of the
    // whole fire.
    //
    // It was also a cliff rather than a slope: `maxMemoryAllocationCount` is
    // a real limit -- 4096 on many devices -- and a fire is free to state
    // more rectangles than that.
    //
    // The same decode's 114 blocks are 3624 bytes gathered here, and the one
    // allocation that holds them costs 200 to 450 microseconds. The padding
    // below is what the gathering costs: the spans are aligned to what the
    // device addresses a storage buffer from, so 3624 bytes of scalars take
    // rather more room than that. Room is not what was scarce.
    let mut planned = Vec::with_capacity(lowered.launches.len());
    // A `read` MAP STOOD HERE, keyed by symbol, holding the module's
    // `Declared` beside the table row -- two walks over something whose size
    // is the tree's and not the fire's, cached so that neither happened once
    // per LAUNCH. The row half has no table to come from, and the `Declared`
    // half is `Reflection`'s job: it caches by ENTRYPOINT, which is the finer
    // key of the two, because a routine composes entrypoints its plan symbol
    // does not name.
    //
    // `parsed` and `tiered` counters stood here too, incremented at this
    // map's cache miss. Both are derived from `reflection.seen` now, at the
    // `Fired` below, which says why.

    let mut spans: Vec<Option<(u64, u64)>> = Vec::with_capacity(planned.capacity());
    let mut scalars: Vec<u8> = Vec::new();
    let give_back = |device: &Device, block: Option<crate::device::Buffer>| {
        if let Some(b) = block {
            device.free(b);
        }
    };
    // The reflection every path shares. On the table path `read` below keys
    // it by the PLAN's symbol; a routine names its own entrypoint, which the
    // plan need not carry at all, so the routine path needs a cache that can
    // still miss. Both walk the same SPIR-V exactly once per name.
    let reflection = Reflection::new(modules, tier);
    for (at, launch) in lowered.launches.iter().enumerate() {
        let symbol = lowered.kernels[launch.kernel as usize].as_str();
        // The ROUTINE path, when the symbol's family has arms.
        //
        // The registry does its OWN lookup, against the entrypoint stems it
        // states, rather than resolving the symbol through
        // `kernels::sig_in(KERNELS, ..)`. That is what lets a family's rows be
        // deleted the moment its arms are written: a fork that read the table
        // would make a crossed family unreachable the instant its rows went,
        // which is the circularity `.wiki/kernel-x/refactor-bigplan.md` §7
        // leaves between Stage 3 ("that family's `kernel!` rows deleted") and
        // Stage 5 ("the arm registry becomes the lookup").
        if let Some((routine, arm)) = crate::arm::arm_for(symbol) {
            // The two refusals a MISSING module gets, said here so that
            // crossing a family does not change the shape of the answer.
            //
            // `Reflect::of` collapses "the store holds nothing under this
            // name" and "it holds something that is not SPIR-V" into `None`,
            // and a body meeting that `None` refuses with `Undeclared` -- a
            // refusal that reads as "the shader does not bind what the body
            // states", which is a different fault with a different fix. The
            // table path has always distinguished them and a caller acts on
            // the distinction: `NoModule` means build the module, `Unreadable`
            // means the bytes under it are damaged.
            //
            // Asked of the PLAN's symbol, which is the name the store is keyed
            // by and the name the refusal must carry. A body that composes a
            // different entrypoint still meets `Undeclared`, correctly: that
            // is a body naming a module this build did not produce, which is
            // the body's fault and not the store's.
            let Some(code) = modules.code(symbol, tier) else {
                return Err(Unfired::NoModule {
                    at,
                    symbol: symbol.to_owned(),
                });
            };
            if let Err(why) = crate::spirv::words(code).and_then(|w| crate::spirv::declared(&w)) {
                return Err(Unfired::Unreadable {
                    at,
                    symbol: symbol.to_owned(),
                    why,
                });
            }
            let made = plan_routine(
                lowered,
                launch,
                symbol,
                routine,
                arm,
                arena,
                resolver,
                geometry,
                &reflection,
                device.min_storage_offset(),
            )
            .map_err(|why| Unfired::Unplannable {
                at,
                symbol: symbol.to_owned(),
                why,
            })?;
            for d in made {
                push_scalars(device, &d, &mut spans, &mut scalars);
                planned.push(d);
            }
            continue;
        }
        // THE TABLE PATH STOOD HERE.
        //
        // Nine hundred lines of it across `serve` and `dispatch`: read the
        // module, resolve `kernels::sig_in(kernels_vulkan::KERNELS, symbol)`,
        // hand the row's `Rule` and `operands` to `dispatch::plan_one`, and
        // let a declarative column say which of a launch's arguments was a
        // buffer, which a scalar, and how wide a grid to launch.
        //
        // `kernels_vulkan::KERNELS` has no rows. Every one of the 481
        // entrypoints this build produces is reached by an arm above --
        // `arm::every_entrypoint_a_plan_can_name_finds_an_arm` is the sweep
        // that says so, over `kernels_vulkan::entrypoints()` rather than over
        // a list this file keeps. So the fallback could only ever be taken by
        // a symbol a PLAN named that this driver does not build, and for that
        // symbol the table path's own answer was `Undispatchable::Unknown`.
        //
        // It says it directly now, which is the only behaviour change: a
        // symbol with no arm is refused by name instead of being carried
        // through a module read and a row lookup to arrive at the same word.
        // The read is skipped too, so an unknown name no longer costs a
        // SPIR-V walk to be told it is unknown.
        return Err(Unfired::Unplannable {
            at,
            symbol: symbol.to_owned(),
            why: crate::dispatch::Undispatchable::Unknown {
                symbol: symbol.to_owned(),
            },
        });
    }

    // One allocation, or none when no rectangle in this fire states a block.
    let block = if scalars.is_empty() {
        None
    } else {
        match device.buffer(&scalars) {
            Ok(b) => Some(b),
            Err(why) => return Err(Unfired::Refused { at: 0, why }),
        }
    };

    // Passes two and three are a separate function so that every borrow of
    // the block buffer -- and `Bound::at` takes one per scalar block -- ends
    // before the free below. Written inline, the borrow checker is right to
    // refuse: the recorded buffers name that memory, and freeing it while
    // they do is the exact defect this shape exists to prevent.
    //
    // Measured, so this is not a guess: moving the `give_back` below to
    // before this line does not compile. The one ordering rule in this module
    // that Vulkan will not enforce is the one Rust does.
    let outcome = record(
        device,
        pipelines,
        modules,
        &planned,
        Blocks {
            block: block.as_ref(),
            spans: &spans,
        },
        tier,
        one_at_a_time,
    )
    // Filled in here because `record` never sees pass one. See `Fired`.
    .map(|f| Fired {
        // Plus the routine path's own reads. Its cache is keyed by
        // ENTRYPOINT rather than by plan symbol -- a body spells its own, and
        // a two-pass reduction names two for one statement -- but a miss is a
        // miss, and the invariant this number exists for is "one walk of the
        // SPIR-V per module, not one per rectangle".
        parsed: reflection.seen.borrow().len(),
        // The tier, for the entrypoints the ROUTINE path reached.
        //
        // `tiered` was incremented only in the table path's cache miss,
        // because that was once the only path, and it went to ZERO the moment
        // the last family crossed. That is not a lost statistic:
        // `the_default_tile_reaches_the_tier_in_production` reads this number
        // to check that a real prefill REACHES the cooperative-matrix build,
        // and a zero is indistinguishable from a driver that silently serves
        // baseline everywhere -- which is the exact defect the test exists
        // for, so the check had gone blind rather than false.
        //
        // Counted off the reflection's keys, beside `parsed` and for the same
        // reason: those are the entrypoints that were really dispatched, which
        // a routine composes and the plan's symbol need not name.
        tiered: reflection
            .seen
            .borrow()
            .keys()
            .filter(|e| {
                modules
                    .resolved(e, tier)
                    .is_some_and(|c| c != Capability::Baseline)
            })
            .count(),
        ..f
    });
    // After the fence and not before: `run_all` waits, so by here the queue
    // is done with every block in it. This is the whole reason the buffer is
    // owned by this function rather than returned.
    give_back(device, block);
    outcome
}

/// Every scalar block of one fire, in one buffer.
///
/// A pair rather than two arguments because they are one thing: a span is
/// meaningless without the buffer it indexes, and a buffer whose spans came
/// from a different fire would bind whatever is at those offsets -- which are
/// scalars, so it would be plausible numbers rather than a fault.
struct Blocks<'a> {
    /// `None` when no rectangle in the fire states a block.
    block: Option<&'a crate::device::Buffer>,
    /// Offset and length per planned rectangle, `None` where it pushes.
    spans: &'a [Option<(u64, u64)>],
}

/// Build every pipeline, record every dispatch, submit once and wait.
fn record<M: Modules>(
    device: &Device,
    pipelines: &mut Pipelines,
    modules: &M,
    planned: &[crate::dispatch::Dispatch<'_>],
    scalars: Blocks<'_>,
    tier: Capability,
    one_at_a_time: bool,
) -> Result<Fired, Unfired> {
    // Pass two: every distinct module gets a pipeline, so that pass three can
    // hold a reference to all of them at once.
    let mut buffers = Vec::with_capacity(planned.len());
    // The write masks, kept beside the buffers because pass two can INSERT a
    // slot into a launch's bindings -- the scalar block -- and a mask that
    // still described the plan's list would be off by one from there on.
    let mut writes: Vec<Vec<bool>> = Vec::with_capacity(planned.len());
    let Blocks { block, spans } = scalars;
    let blocks = spans.iter().filter(|s| s.is_some()).count();
    for (at, (d, span)) in planned.iter().zip(spans).enumerate() {
        let symbol = d.symbol.as_ref();
        let mut b = d.buffers.clone();
        let mut w = d.writes.clone();
        if let Some((offset, len)) = *span {
            let Some(slot) = d.block_at else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.to_owned(),
                    what: "its scalars are a block and the planner named no slot for it",
                });
            };
            let Some(buf) = block else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.to_owned(),
                    what: "its scalars name a span of a block buffer that was not allocated",
                });
            };
            // The span and not the whole buffer. Every block of the fire is
            // in there, so `WHOLE_SIZE` -- or a length past this one's --
            // would let a shader reading one word too far read the NEXT
            // rectangle's scalars, which are plausible numbers.
            let bound =
                Bound::at(device, buf, offset, len).map_err(|why| Unfired::Refused { at, why })?;
            b.insert(slot, bound);
            // Read-only: the shader reads its parameters out of it and this
            // driver is the only thing that ever writes it, on the host,
            // before the fire is submitted.
            w.insert(slot, false);
        }
        let push = match &d.params {
            Params::Push(p) => p.len() as u32,
            _ => 0,
        };
        // `modules.code` answered for this symbol in pass one, so it answers
        // here.
        let code = modules.code(symbol, tier).unwrap_or_default();
        pipelines
            .get(device, symbol, code, push, b.len() as u32, tier)
            .map_err(|why| Unfired::Refused { at, why })?;
        buffers.push(b);
        writes.push(w);
    }

    // Pass three.
    let mut run = Vec::with_capacity(planned.len());
    for ((d, b), w) in planned.iter().zip(&buffers).zip(&writes) {
        let symbol = d.symbol.as_ref();
        let Some(pipeline) = pipelines.peek(symbol, tier) else {
            return Err(Unfired::Impossible {
                at: run.len(),
                symbol: symbol.to_owned(),
                what: "no pipeline, one pass after every pipeline was built",
            });
        };
        run.push(Recorded {
            symbol,
            pipeline,
            buffers: b,
            writes: w,
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
            blocks,
            // Pass one's, and this function is passes two and three. The
            // caller fills it.
            parsed: 0,
            tiered: 0,
        });
    }
    device
        .run_all(&run)
        .map_err(|(at, why)| Unfired::Refused { at, why })?;
    Ok(Fired {
        dispatches: run.len(),
        submissions: 1,
        blocks,
        parsed: 0,
        tiered: 0,
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
/// Every row of the readout. [`logits_of`] is the one a server should call:
/// this is for callers -- tests, mostly -- that want the whole rectangle and
/// know it is small.
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
    let rows = lowered.readout.ok_or(Unread::NoExit)?.rows as usize;
    logits_of(device, arena, lowered, &(0..rows).collect::<Vec<_>>())
}

/// The rows `want` of a fire's distributions, off the arena and widened.
///
/// `want` is taken as fire row numbers, and rows outside the readout are
/// ignored rather than refused: a caller states the rows it will ask about,
/// and a row the fire did not produce is a question [`Logits::row`] already
/// answers with `None`.
///
/// # What it reads
///
/// ONE contiguous span, from the first wanted row to the last, and then only
/// the wanted rows are widened out of it. A span rather than a row each
/// because a read is a DMA with a fence -- about 300 us of fixed cost -- so
/// eight scattered rows would be eight submissions, and a decode's rows are
/// every row anyway. A prefill's single readout is its last row, which makes
/// the span one row of a rectangle with a thousand.
///
/// # Errors
///
/// [`Unread`], and [`Failed`] is folded into it only through
/// [`Unread::PastArena`] -- the read itself is checked before it is asked for.
pub fn logits_of(
    device: &Device,
    arena: &crate::device::Buffer,
    lowered: &Lowered,
    want: &[usize],
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
    // The rows a caller stated, and one span of them.
    //
    // `read` is what came back and `values` is in its order, so a row nobody
    // asked for is absent rather than wrong -- see `Logits::read`.
    let mut read: Vec<usize> = want.iter().copied().filter(|&r| r < rows).collect();
    read.sort_unstable();
    read.dedup();
    let (Some(&first), Some(&last)) = (read.first(), read.last()) else {
        return Ok(Logits {
            rows,
            vocab,
            values: Vec::new(),
            read,
        });
    };
    let width = vocab * exit.bytes as usize;
    let span = (last + 1 - first) * width;
    // The READOUT and not the arena.
    //
    // Reading the whole buffer and slicing it was 334 megabytes for a
    // 1024-token prefill where the logits are 311 of them -- and of THOSE, a
    // caller wants one row. `Device::read_at` is also where the staged copy
    // is, so this line is what puts a fire's answer on the copy engine
    // instead of on an uncached PCIe read.
    let bytes = device
        .read_at(arena, (exit.at + first * width) as u64, span as u64)
        .map_err(|_| Unread::PastArena {
            at: exit.at,
            extent,
            arena: lowered.arena_bytes,
        })?;
    let mut values = Vec::with_capacity(read.len() * vocab);
    for &row in &read {
        let at = (row - first) * width;
        let one = bytes.get(at..at + width).ok_or(Unread::PastArena {
            at: exit.at,
            extent,
            arena: lowered.arena_bytes,
        })?;
        match exit.bytes {
            // bf16 is the TOP half of an f32, so widening is a shift and not
            // a conversion. Written as bits rather than through a cast
            // because `u16 as f32` is a numeric conversion and this is a
            // reinterpretation.
            2 => values.extend(
                one.chunks_exact(2)
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)),
            ),
            _ => values.extend(
                one.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
        }
    }
    Ok(Logits {
        rows,
        vocab,
        values,
        read,
    })
}

/// Gather one dispatch's scalar block into the fire's single staging run.
///
/// Factored out because both planning paths need it and neither may skip it:
/// `spans` is indexed by DISPATCH, so a path that pushed a `Dispatch` without
/// pushing a span would shift every block after it onto the wrong offsets --
/// which binds a real buffer of a plausible size and computes a number.
///
/// The alignment is what the device will address a storage buffer from,
/// because these are bound as sub-ranges and `Bound::at` refuses any other
/// offset -- rightly, since an unaligned descriptor is invalid rather than
/// slow.
fn push_scalars(
    device: &Device,
    d: &crate::dispatch::Dispatch<'_>,
    spans: &mut Vec<Option<(u64, u64)>>,
    scalars: &mut Vec<u8>,
) {
    match &d.params {
        Params::Block { bytes, .. } => {
            let align = device.min_storage_offset();
            let pad = scalars.len() as u64 % align;
            if pad != 0 {
                scalars.resize(scalars.len() + (align - pad) as usize, 0);
            }
            spans.push(Some((scalars.len() as u64, bytes.len() as u64)));
            scalars.extend_from_slice(bytes);
        }
        Params::Push(_) | Params::None => spans.push(None),
    }
}

/// The module cache the routine path reflects through.
///
/// One walk of the SPIR-V per ENTRYPOINT, which is the same guarantee the
/// table path's `read` map gives and for the same measured reason: reading a
/// module is a few thousand words, and doing it once per launch cost 22
/// milliseconds of a 24-millisecond pass.
///
/// Keyed by entrypoint rather than by plan symbol because a routine spells its
/// own -- a body that instantiates an axis builds `affine_qmm_t_bf16_gs_128_b_4`
/// itself, and a two-pass reduction names two entrypoints for one statement.
/// A miss is cached as a miss, so a routine naming a module this build did not
/// produce costs one lookup rather than one parse per launch.
type Reflected = (crate::geometry::Module, std::rc::Rc<crate::spirv::Declared>);

/// A module store's SPIR-V, read at most once per entrypoint.
///
/// See [`Reflected`]: this is the cache that guarantee lives in, and it is
/// public so a walk with no GPU can hand one to [`plan_routine`].
pub struct Reflection<'m, M: Modules> {
    modules: &'m M,
    tier: Capability,
    seen: core::cell::RefCell<BTreeMap<String, Option<Reflected>>>,
}

impl<'m, M: Modules> Reflection<'m, M> {
    /// A reflection over `modules`, reading each entrypoint at most once.
    #[must_use]
    pub fn new(modules: &'m M, tier: Capability) -> Self {
        Self {
            modules,
            tier,
            seen: core::cell::RefCell::new(BTreeMap::new()),
        }
    }
}

impl<M: Modules> crate::encode::Reflect for Reflection<'_, M> {
    fn of(&self, entrypoint: &str) -> Option<Reflected> {
        if let Some(held) = self.seen.borrow().get(entrypoint) {
            return held.clone();
        }
        let made = self
            .modules
            .code(entrypoint, self.tier)
            .and_then(|code| {
                crate::spirv::words(code)
                    .and_then(|w| crate::spirv::declared(&w))
                    .ok()
            })
            .map(|d| {
                (
                    crate::geometry::Module::named(
                        entrypoint,
                        [d.local[0], d.local[1], d.local[2]],
                    ),
                    std::rc::Rc::new(d),
                )
            });
        self.seen
            .borrow_mut()
            .insert(entrypoint.to_owned(), made.clone());
        made
    }
}

/// The three routines whose INPUT is taller than their rectangle.
///
/// `.wiki/kernel-x/vulkan-refactor.md` §10, and it is a real wrong-answer path
/// rather than a tidiness: `binding::extent` sizes every arena operand by
/// `launch.rows.end - launch.rows.start`. That is right for every elementwise
/// and per-row kernel in the tree, and false for exactly the kernels whose
/// input and output have different row counts.
///
/// `layout::row_gather` is the one a test caught. A prefill of six tokens
/// serving four requests states a rectangle of FOUR rows -- the readouts --
/// and reads the rows the sampling table names, which are token indices and
/// run to five. The source was bound to four rows, so the read of row 5 fell
/// off the descriptor range and came back as whatever `robustBufferAccess`
/// gives, which is zeros, SILENTLY: the validation layer does not report
/// storage-buffer overruns and the fire completed.
///
/// `moe::route_gather` and `moe::combine_sorted` have the same shape -- the
/// sorted extent against the token count -- and their rectangle is the taller
/// of the two, so neither has been seen to lose data. They are named here
/// because being right by accident is not a property to rely on.
///
/// The proper fix is a field: `Arg::Arena` states its own rows and `extent`
/// stops asking the launch. That is a `model-compiler` schema change with
/// construction sites in every backend, and this is the interim: bind the
/// source from its own offset to the END of the plan's arena.
///
/// A LOOSER bound and not a wrong one. The arena is the plan's own
/// allocation, so nothing outside it becomes readable; what is lost is the
/// tight range check on one operand of three kernels. A range that is too
/// SHORT is the dangerous direction -- it answers zeros and reports success,
/// which is what this exists to stop.
///
/// # Errors
///
/// [`Undispatchable::Operand`] if the widened range is one the device cannot
/// address from, which is the refusal the original binding would have given.
fn widen_a_gathers_source<'a>(
    bound: &mut [crate::device::Bound<'a>],
    routine: &str,
    lowered: &Lowered,
    launch: &model_compiler::lower::Launch,
    arena: Arena<'a>,
    min_offset: u64,
) -> Result<(), Undispatchable> {
    if !matches!(routine, "row_gather" | "route_gather" | "combine_sorted") {
        return Ok(());
    }
    let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
    // Operand 0 in all three: the activation the gather reads. The results
    // and the permutations are in the rectangle's own space and stay as bound.
    let Some(model_compiler::lower::Arg::Arena { at, .. }) = args.first() else {
        return Ok(());
    };
    let at = *at as u64;
    let Some(rest) = arena.bytes.checked_sub(at) else {
        return Ok(());
    };
    let Some(slot) = bound.first_mut() else {
        return Ok(());
    };
    if rest <= slot.len() {
        return Ok(());
    }
    *slot = crate::device::Bound::within(arena.buffer, at, rest, min_offset).map_err(|why| {
        Undispatchable::Operand {
            at: 0,
            why: crate::binding::Unbindable::Unaddressable(why),
        }
    })?;
    Ok(())
}

/// One launch of a crossed routine, as the dispatches its body asked for.
///
/// The ARM resolves the statement's operands into handles and states the
/// routine's argument list; the BODY states the rectangle and the entrypoint.
/// Neither half can state the other's, which is the point: an arm that
/// computed a grid would put back the second opinion the refactor removes, and
/// a body that reached for an operand would need to know a trace.
///
/// # Errors
///
/// [`Undispatchable::Operand`] for an operand the statement does not carry,
/// and [`Undispatchable::Refused`] for anything the routine or the encoder
/// would not launch -- an empty extent, an entrypoint this build did not
/// produce, an argument list the module's bindings do not fit.
///
/// `min_offset` is the device's storage-buffer alignment. It is a NUMBER
/// rather than a `&Device` so that a walk with no GPU in it can ask this
/// question: `tests/arena.rs` puts every rectangle of six real texts through
/// here, which is the coverage that used to run against `KERNELS` and has
/// nowhere else to go now that the table is empty.
#[allow(clippy::too_many_arguments)]
pub fn plan_routine<'a, R: Resolve, M: Modules>(
    lowered: &Lowered,
    launch: &model_compiler::lower::Launch,
    symbol: &str,
    routine: &'static kernels_vulkan::routine::Routine,
    arm: crate::arm::Arm,
    arena: Arena<'a>,
    resolver: &'a R,
    geometry: Geometry,
    reflection: &Reflection<'_, M>,
    min_offset: u64,
) -> Result<Vec<crate::dispatch::Dispatch<'a>>, Undispatchable> {
    // A conditional rectangle's guard was NOT answered by the lowering, and
    // this walk has no way to answer it -- recording every arm would run
    // every arm. The table path refuses here for the same reason.
    if launch.cond != model_compiler::lower::Launch::NO_COND {
        return Err(Undispatchable::Conditional {
            symbol: symbol.to_owned(),
            cond: launch.cond,
        });
    }
    let mut bound = crate::binding::bind(lowered, launch, arena, resolver, min_offset)
        .map_err(|(at, why)| Undispatchable::Operand { at, why })?;
    widen_a_gathers_source(&mut bound, routine.name, lowered, launch, arena, min_offset)?;

    // How many of the widthed operands are RESULTS. Mostly the count of
    // `BufMut` in the routine's own signature, and `traced_results` says which
    // two routines that count is wrong for and why.
    let results = crate::arm::traced_results(routine);
    let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
    let (ins, outs, weights) = crate::arm::split(args, results);
    let params: Vec<Option<u32>> = (launch.params.start as usize..launch.params.end as usize)
        .map(|at| lowered.params.get(at).copied())
        .collect();

    let widths: Vec<u32> = args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Arena { width, .. }
            | model_compiler::lower::Arg::Named { width, .. } => Some(*width),
            model_compiler::lower::Arg::Weight(_) => None,
        })
        .collect();
    let (group, bits) = crate::arm::affine_of(symbol).unwrap_or((0, 0));
    let facts = crate::arm::Facts {
        rows: launch.rows.end - launch.rows.start,
        // The last widthed operand is the launch's last OUTPUT, which is what
        // sizes the rectangle; the first is its first input. The same two
        // numbers `dispatch::dims_of` reads, off the same list.
        width: widths.last().copied().unwrap_or(0),
        in_width: widths.first().copied().unwrap_or(0),
        q_heads: geometry.q_heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        rotary_dims: geometry.rotary_dims,
        n_experts: geometry.n_experts,
        experts_per_token: geometry.experts_per_token,
        group,
        bits,
        tile: crate::arm::tile_of(symbol),
        layer: launch.layers.start,
        // The readouts this fire serves, which is the PLAN's number and not
        // the sampling table's length.
        //
        // Reading it off the table instead was tried and is wrong: that
        // buffer is a fire-wide resource sized once, so its length answers
        // how many indices it can HOLD, not how many this fire gathers. On a
        // 32-row readout it answered 1, and `row_gather` wrote one row and
        // reported success -- caught only because the two matmul kernels then
        // disagreed about 3 of 32 tokens.
        //
        // `.wiki/kernel-x/vulkan-refactor.md` §10 -- that `binding::extent`
        // sizes every arena operand by `launch.rows`, which is false for a
        // gather whose OUTPUT is in request space and whose INPUT is in token
        // space -- is a real defect and is NOT this number's to fix. It is
        // fixed where the extent is computed, and until then this stays
        // exactly what `kernels::Source::RequestCount` resolved to.
        requests: lowered.n_requests,
    };

    let mut handles = crate::arm::Handles::new(&bound, &ins, &outs, &weights, &params, resolver);
    let values = arm(&mut handles, facts).map_err(|why| Undispatchable::Refused {
        symbol: symbol.to_owned(),
        why,
    })?;
    let taken = handles.bound().to_vec();
    let staged = handles.staged();
    let encoder = crate::encode::Encoder::new(reflection, &taken, &staged, launch.op);
    (routine.body)(&encoder, &values).map_err(|why| Undispatchable::Refused {
        symbol: symbol.to_owned(),
        why,
    })?;
    Ok(encoder.finish())
}

#[cfg(test)]
mod tests {
    use super::Modules;
    use kernels_vulkan::Capability;
    use std::collections::BTreeMap;

    fn store() -> BTreeMap<String, Vec<u8>> {
        let mut m = BTreeMap::new();
        m.insert("qmm".to_owned(), vec![1]);
        m.insert("qmm.coopmat".to_owned(), vec![2]);
        m.insert("qmm.fp16".to_owned(), vec![3]);
        m.insert("rope".to_owned(), vec![4]);
        m
    }

    /// `resolved` names the tier `code` actually answered from.
    ///
    /// The pair has to agree or the count built on it is decoration: one says
    /// which bytes, the other says where they came from, and a test that
    /// asserted only the second could pass while the first fell back.
    #[test]
    fn the_tier_reported_is_the_tier_the_bytes_came_from() {
        let m = store();
        assert_eq!(
            m.resolved("qmm", Capability::Coopmat),
            Some(Capability::Coopmat)
        );
        assert_eq!(
            m.resolved("rope", Capability::Coopmat),
            Some(Capability::Baseline),
            "no coopmat build, so the tier that answered is the baseline"
        );
        assert_eq!(m.resolved("nothing", Capability::Coopmat), None);
        for symbol in ["qmm", "rope"] {
            for tier in Capability::PREFERENCE {
                assert_eq!(
                    m.resolved(symbol, tier).is_some(),
                    m.code(symbol, tier).is_some(),
                    "{symbol} at {tier:?}: the two walks disagree about whether \
                     there is an answer at all"
                );
            }
        }
    }

    /// A lower tier never REPORTS a higher one, for the same reason it never
    /// reads its module.
    #[test]
    fn a_lower_tier_never_reports_a_higher_ones_module() {
        let m = store();
        assert_eq!(
            m.resolved("qmm", Capability::Baseline),
            Some(Capability::Baseline)
        );
        assert_eq!(
            m.resolved("qmm", Capability::Fp16),
            Some(Capability::Fp16),
            "the walk starts AT the tier, so fp16 stops at its own build"
        );
    }

    /// A tiered module is reachable, which for a long time it was not.
    ///
    /// `kernels-vulkan` names a tiered build `<entrypoint>.<tag>.spv` and
    /// every store here is keyed by file stem, so the cooperative-matrix
    /// build of `qmm` is under `qmm.coopmat`. A plan states `qmm`. While the
    /// lookup took only a symbol it could not reach the tiered file at all,
    /// and all 146 coopmat modules were dead on every device.
    ///
    /// Nothing failed while they were: the device still REPORTED the tier,
    /// the shell still set it and the pipeline cache still keyed on it. This
    /// is the assertion that says the file is what changes, because it is the
    /// only part that was ever wrong.
    #[test]
    fn a_tier_selects_the_module_compiled_for_it() {
        let m = store();
        assert_eq!(m.code("qmm", Capability::Coopmat), Some(&[2u8][..]));
        assert_eq!(m.code("qmm", Capability::Fp16), Some(&[3u8][..]));
        assert_eq!(m.code("qmm", Capability::Baseline), Some(&[1u8][..]));
    }

    /// A tier the entrypoint was not compiled at falls back, rather than
    /// refusing.
    ///
    /// This is what "tiers are additive" means and it is most of the tree:
    /// `rope` has no coopmat build and never will, and a device at the top
    /// tier must still get its baseline module. A lookup that returned `None`
    /// here would refuse to fire an entire model on the best hardware.
    #[test]
    fn a_tier_without_a_module_falls_back_to_the_one_below() {
        let m = store();
        assert_eq!(m.code("rope", Capability::Coopmat), Some(&[4u8][..]));
        assert_eq!(m.code("rope", Capability::Fp16), Some(&[4u8][..]));
    }

    /// A tier never reaches ABOVE itself.
    ///
    /// A baseline device asking for `qmm` must get the baseline module even
    /// though a coopmat one is sitting in the same map -- the store is a
    /// directory listing and says nothing about what this device can load.
    /// Handing it `qmm.coopmat` is a module that names a capability the
    /// device did not enable, which is a validation error and not a slow
    /// answer.
    #[test]
    fn a_lower_tier_never_reaches_a_higher_ones_module() {
        let m = store();
        assert_eq!(m.code("qmm", Capability::Baseline), Some(&[1u8][..]));
        let only_high: BTreeMap<String, Vec<u8>> =
            [("qmm.coopmat".to_owned(), vec![2])].into_iter().collect();
        assert_eq!(only_high.code("qmm", Capability::Baseline), None);
        assert_eq!(only_high.code("qmm", Capability::Fp16), None);
    }
}
