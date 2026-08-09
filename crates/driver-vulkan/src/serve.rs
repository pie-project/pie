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
    let mut read: std::collections::BTreeMap<&str, crate::spirv::Declared> =
        std::collections::BTreeMap::new();
    // Counted where the parse happens rather than taken from `read.len()` at
    // the end. The map's size is the number of distinct symbols whether the
    // cache is consulted or not -- measured: a mutation that dropped every
    // entry before looking it up reported the same number and passed the test
    // that exists to catch it. This counts the walk.
    let mut parsed = 0usize;
    let mut spans: Vec<Option<(u64, u64)>> = Vec::with_capacity(planned.capacity());
    let mut scalars: Vec<u8> = Vec::new();
    let give_back = |device: &Device, block: Option<crate::device::Buffer>| {
        if let Some(b) = block {
            device.free(b);
        }
    };
    for (at, launch) in lowered.launches.iter().enumerate() {
        let symbol = lowered.kernels[launch.kernel as usize].as_str();
        let Some(code) = modules.code(symbol, tier) else {
            return Err(Unfired::NoModule {
                at,
                symbol: symbol.to_owned(),
            });
        };
        // Read once per SYMBOL, not once per launch.
        //
        // This was the other way round, and the note here said that reading a
        // module is a walk over a few thousand words which does not show
        // against the pipeline builds. That was measured, and it stopped
        // being true when the pipelines started being cached: a fire whose
        // pipelines are all built spends its time here instead. Measured on a
        // qwen3 decode, 452 rectangles over 9 distinct symbols, with the
        // parse timed separately -- 22 milliseconds of the 24 this pass cost
        // were the same nine modules being read four hundred and fifty-two
        // times.
        //
        // The map is keyed by a borrow of the plan's own symbol, which is why
        // it costs no lifetime the signature did not already have: `Declared`
        // is owned, and `plan_one` borrows it only for the call.
        let declared = match read.entry(symbol) {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                parsed += 1;
                match crate::spirv::words(code).and_then(|w| crate::spirv::declared(&w)) {
                    Ok(d) => e.insert(d),
                    Err(why) => {
                        return Err(Unfired::Unreadable {
                            at,
                            symbol: symbol.to_owned(),
                            why,
                        });
                    }
                }
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
                declared,
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
        match &d.params {
            Params::Block { bytes, .. } => {
                // Aligned to what the device will address a storage buffer
                // from, because these are bound as sub-ranges and
                // `Bound::at` refuses any other offset -- rightly, since an
                // unaligned descriptor is invalid rather than slow.
                let align = device.min_storage_offset();
                let pad = scalars.len() as u64 % align;
                if pad != 0 {
                    scalars.resize(scalars.len() + (align - pad) as usize, 0);
                }
                spans.push(Some((scalars.len() as u64, bytes.len() as u64)));
                scalars.extend_from_slice(bytes);
            }
            _ => spans.push(None),
        }
        planned.push((symbol.to_owned(), d));
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
    .map(|f| Fired { parsed, ..f });
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
    planned: &[(String, crate::dispatch::Dispatch<'_>)],
    scalars: Blocks<'_>,
    tier: Capability,
    one_at_a_time: bool,
) -> Result<Fired, Unfired> {
    // Pass two: every distinct module gets a pipeline, so that pass three can
    // hold a reference to all of them at once.
    let mut buffers = Vec::with_capacity(planned.len());
    let Blocks { block, spans } = scalars;
    let blocks = spans.iter().filter(|s| s.is_some()).count();
    for (at, ((symbol, d), span)) in planned.iter().zip(spans).enumerate() {
        let mut b = d.buffers.clone();
        if let Some((offset, len)) = *span {
            let Some(slot) = d.block_at else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.clone(),
                    what: "its scalars are a block and the planner named no slot for it",
                });
            };
            let Some(buf) = block else {
                return Err(Unfired::Impossible {
                    at,
                    symbol: symbol.clone(),
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
            blocks,
            // Pass one's, and this function is passes two and three. The
            // caller fills it.
            parsed: 0,
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
