//! Turning a lowering's operands into descriptor ranges.
//!
//! [`model_compiler::lower`] hands a driver a flat list of [`Arg`]s and says
//! *"bind these"*. On Metal that is nearly free: `setBuffer:offset:` takes an
//! address and a byte offset and no length at all, so `driver-metal`'s binder
//! resolves each operand to its base plus its offset, reports the rest of the
//! arena as the extent, and is done. The extent is not load-bearing there
//! because nothing reads it.
//!
//! Vulkan has no such call. A storage descriptor is a buffer, an offset AND a
//! range, and all three are checked. So this module has to answer a question
//! Metal never asked: **how many bytes is this operand?**
//!
//! # The extent is in the plan, and it is exact
//!
//! It is `rows × width × bytes`: the launch states its row count, and the
//! operand states its row width and its element size. Measured over every
//! arena operand six real texts produce in both fire classes -- 14324 of them
//! -- every extent lands inside the arena, and the tightest fit has **zero**
//! bytes to spare. An operand ends exactly where the arena does.
//!
//! That zero is the useful part. It says the formula is not a lower bound that
//! happens to be safe, it is the real extent: if it were an under-estimate the
//! slack would never reach zero, and if it were an over-estimate it would have
//! run past. `tests/arena.rs` keeps that number.
//!
//! # Why the range matters more here than a length usually does
//!
//! The arena is ONE buffer holding every activation. `VK_WHOLE_SIZE` on an
//! operand at offset `at` therefore covers every tensor allocated after it,
//! and a kernel indexing past its own rows would read or write a neighbour
//! silently -- `tests/device.rs` demonstrates exactly that, and demonstrates
//! that a real range confines it instead, because `robustBufferAccess` is on.
//! So the range is not bookkeeping. It is the only thing standing between a
//! stray index and another tensor's bytes.
//!
//! # What this module does not do
//!
//! It resolves operands. It does not build the parameter side of the call:
//! seven of the reachable symbols want their scalars as push constants and six
//! want them as a buffer of their own, which `tests/arena.rs` measures and a
//! later layer will act on.

use model_compiler::lower::{Arg, Launch, Lowered};
use model_compiler::trace::ValueId;

use crate::device::{Bound, Buffer};

/// The frame's arena: one buffer, every activation.
#[derive(Clone, Copy, Debug)]
pub struct Arena<'a> {
    /// The buffer the offsets are into.
    pub buffer: &'a Buffer,
    /// How many bytes of it the plan was allowed to place into.
    ///
    /// Stated separately from the buffer's own size because a driver may hold
    /// a larger arena than a given fire needs, and the question an operand
    /// asks is whether it fits the PLAN's arena. A buffer big enough to
    /// contain a mistake still contains one.
    pub bytes: u64,
}

/// Where the operands this crate cannot resolve come from.
///
/// Two of the three [`Arg`] kinds name something the plan does not hold: a
/// weight by its trace name and a seam value by its id. Both are the driver's
/// own tables, so both are asked for rather than looked up.
///
/// Takes `&self` rather than `&mut self` -- unlike the Metal one, which can
/// return a copied address -- because a Vulkan binding borrows the buffer it
/// names, and the borrow has to outlive the call that produced it.
pub trait Resolve {
    /// The buffer holding a weight, by the name the trace states.
    fn weight(&self, name: &str) -> Option<&Buffer>;
    /// The buffer holding a seam value the backend binds by name.
    fn named(&self, value: ValueId) -> Option<&Buffer>;

    /// The KV cache for one layer, keys or values.
    ///
    /// STATE, not an operand: no traced value stands for it, so no plan
    /// mentions it and no arena holds it. `kv_append_paged` names both and
    /// the paged attentions read both.
    ///
    /// Defaulted to `None` so that a resolver serving a text without paged
    /// attention does not have to state a method it will never be asked for.
    /// The refusal that produces is [`Unbindable::NoDriverResource`], which
    /// names what was missing.
    fn kv(&self, _layer: u16, _values: bool) -> Option<&Buffer> {
        None
    }

    /// One of the fire's own numbers.
    ///
    /// A pool's shape, not a statement's scalar. A text that stated its page
    /// size would be right for one deployment and silently wrong for the
    /// next, so the row names the number and the driver answers it.
    fn number(&self, _which: FireNumber) -> Option<u32> {
        None
    }

    /// One of the fire's own tables.
    ///
    /// Also state. The kernel row names WHICH; this forwards the name and
    /// never reads what it means, which is what keeps the driver from having
    /// an opinion about a table's contents.
    fn table(&self, _which: FireTable) -> Option<&Buffer> {
        None
    }
}

/// The fire-wide numbers a kernel row may name.
///
/// Scalars rather than buffers, so these are appended to the parameter run
/// where the row places them -- not given a descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FireNumber {
    /// How many rows one KV page holds.
    KvPageSize,
    /// The stride between heads in the cache.
    KvHeadStride,
    /// The stride between positions in the cache.
    KvSeqStride,
}

/// The fire-wide tables a kernel row may name.
///
/// Transcribed from `driver-metal`, which reached the same list by the same
/// route: these are the [`kernels::Source`] variants that are neither an
/// operand of the statement nor a scalar, so nothing but the driver can
/// supply them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FireTable {
    /// The token id at each row.
    TokenIds,
    /// The position of each row in its sequence.
    Positions,
    /// Which request each row belongs to.
    RequestOfToken,
    /// The pages each request's KV occupies.
    KvPageIndices,
    /// Where each request's run starts in [`Self::KvPageIndices`].
    KvPageIndptr,
    /// The attention mask.
    AttentionMask,
    /// Whether the mask is in force.
    AttentionMaskEnabled,
    /// The page each row's KV is written to.
    KvWritePage,
    /// The offset within that page.
    KvWriteOffset,
    /// The rope frequency table.
    RopeFrequencies,
    /// Which rows the readout samples.
    SamplingIndices,
}

impl std::fmt::Display for Unbindable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PastArena { at, extent, arena } => {
                write!(f, "{extent} bytes at {at} runs off an arena of {arena}")
            }
            Self::UnknownWeight(name) => write!(f, "no buffer holds the weight `{name}`"),
            Self::UnknownNamed(value) => write!(f, "no buffer holds the named value {value}"),
            Self::Constant(name) => {
                write!(
                    f,
                    "`{name}` is a dispatch constant riding a weight slot, not a buffer"
                )
            }
            Self::NoOperand => write!(f, "the row names an operand this statement does not state"),
            Self::NoKvCache { layer, values } => write!(
                f,
                "no {} cache for layer {layer}",
                if *values { "value" } else { "key" }
            ),
            Self::NoDriverResource(what) => write!(f, "the driver holds no {what:?} table"),
            Self::Unaddressable(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Unbindable {}

/// The marker a constant rides the weight-name slot under.
///
/// Transcribed from `driver-metal`: `dsl::cuda::scalar_mul` puts a scalar in
/// an operand slot so the launch's arity holds, and states that no binder
/// looks for it. Metal binds a zero-length region and calls that honest.
///
/// This backend cannot. A descriptor of range zero is invalid in Vulkan --
/// `Bound::within` refuses it as an overrun -- so a slot under this prefix is one
/// the caller must fill some other way, and saying so is the only correct
/// answer. It is [`Unbindable::Constant`] rather than a silent empty range.
const SCALE_PREFIX: &str = "scale.";

/// Why an operand could not become a descriptor range.
#[derive(Clone, Debug, PartialEq)]
pub enum Unbindable {
    /// An arena operand's rectangle runs past the arena the plan sized.
    ///
    /// Never seen in a real lowering -- and the tightest real operand ends
    /// exactly AT the end, so a plan one row wider would land here.
    PastArena {
        /// Where in the arena the operand starts.
        at: usize,
        /// The rectangle it wanted: `rows × width × bytes`.
        extent: u64,
        /// What the plan said the arena holds.
        arena: u64,
    },
    /// The plan names a weight the resolver does not hold.
    UnknownWeight(
        /// The name the plan stated.
        String,
    ),
    /// The plan names a seam value the resolver does not bind.
    UnknownNamed(
        /// The value id the plan stated.
        ValueId,
    ),
    /// The slot holds a dispatch constant, not a buffer.
    ///
    /// See `SCALE_PREFIX`. The refusal is the finding: there is no range
    /// that means "nothing" in Vulkan.
    Constant(
        /// The constant's name, with the `scale.` prefix removed.
        String,
    ),
    /// The row names an operand of the statement that the statement does not
    /// have.
    ///
    /// A row states the shape of every deployment its kernel serves, so a
    /// statement reaching only part of that shape is ordinary -- but a slot
    /// left holding nothing is a descriptor the shader reads, so it is a
    /// refusal rather than a gap. `Source::Unbound` is how a row says "gap".
    NoOperand,
    /// The row names the KV cache and the resolver does not hold one.
    NoKvCache {
        /// Which layer was asked for.
        layer: u16,
        /// Values rather than keys.
        values: bool,
    },
    /// The row names a fire table the resolver does not hold.
    NoDriverResource(
        /// Which table.
        FireTable,
    ),
    /// The device cannot address the offset or extent this operand needs.
    ///
    /// Carries what [`Bound::within`] said, so an alignment failure and a
    /// length failure stay distinguishable at the point they are reported.
    Unaddressable(
        /// What the range check said.
        crate::device::Failed,
    ),
}

/// How many bytes one operand of this launch covers.
///
/// `None` for the operand kinds whose extent is not the plan's to state: a
/// weight is as big as its tensor and a seam value is as big as the backend
/// made it, and in both cases the resolver's buffer already says so.
#[must_use]
pub fn extent(arg: &Arg, launch: &Launch) -> Option<u64> {
    match arg {
        Arg::Arena { width, bytes, .. } => {
            let rows = u64::from(launch.rows.end - launch.rows.start);
            // Saturating rather than wrapping: a plan with an absurd width
            // should produce a refusal below, not a small number that binds.
            Some(
                rows.saturating_mul(u64::from(*width))
                    .saturating_mul(u64::from(*bytes)),
            )
        }
        Arg::Named { .. } | Arg::Weight(_) => None,
    }
}

/// Resolve one operand into a descriptor range.
///
/// `min_offset` is the device's `minStorageBufferOffsetAlignment`, which the
/// range is checked against here rather than at dispatch: an operand that
/// cannot be addressed is a fact about the plan and the device together, and
/// the earliest place both are known is here.
///
/// # Errors
///
/// [`Unbindable`], naming which of the rules could not be applied.
pub fn resolve<'a, R: Resolve>(
    arg: &Arg,
    launch: &Launch,
    arena: Arena<'a>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Bound<'a>, Unbindable> {
    match arg {
        Arg::Arena { at, .. } => {
            let extent = extent(arg, launch).expect("an arena operand states its rectangle");
            let at64 = *at as u64;
            // Checked before `Bound::within` so the refusal names the ARENA. The
            // buffer may well be larger than the plan's arena, in which case
            // `Bound::within` would accept a range the plan had no right to.
            if at64.saturating_add(extent) > arena.bytes {
                return Err(Unbindable::PastArena {
                    at: *at,
                    extent,
                    arena: arena.bytes,
                });
            }
            Bound::within(arena.buffer, at64, extent, min_offset).map_err(Unbindable::Unaddressable)
        }
        Arg::Named { value, .. } => resolver
            .named(*value)
            .map(Bound::whole)
            .ok_or(Unbindable::UnknownNamed(*value)),
        Arg::Weight(name) => {
            if let Some(rest) = name.strip_prefix(SCALE_PREFIX) {
                return Err(Unbindable::Constant(rest.to_owned()));
            }
            resolver
                .weight(name)
                .map(Bound::whole)
                .ok_or_else(|| Unbindable::UnknownWeight(name.clone()))
        }
    }
}

/// Resolve every operand of one launch, in the order the plan states them.
///
/// # Errors
///
/// The first [`Unbindable`] any operand produces, and the index of the operand
/// that produced it. Nothing partial comes back: a dispatch with some ranges
/// resolved is a dispatch that would read whatever the descriptor set happened
/// to hold in the others, which on a reused set is the previous launch's
/// operand and not garbage -- so it would look plausible.
pub fn bind<'a, R: Resolve>(
    lowered: &Lowered,
    launch: &Launch,
    arena: Arena<'a>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Vec<Bound<'a>>, (usize, Unbindable)> {
    let span = launch.args.start as usize..launch.args.end as usize;
    let mut bound = Vec::with_capacity(span.len());
    for (i, arg) in lowered.args[span].iter().enumerate() {
        bound.push(resolve(arg, launch, arena, resolver, min_offset).map_err(|e| (i, e))?);
    }
    Ok(bound)
}

/// What one of a module's binding slots holds.
///
/// The middle term the positional binder did not have. A plan states its
/// operands in TRACE order -- inputs, then outputs, then weights -- and a
/// shader binds them in the order its kernel row states, and those are not
/// the same order. `rms_single_row`'s row is `In(0), Weight(0), Out(0),
/// params`, and `norm/rms.comp` decorates exactly that; the trace hands over
/// `In(0), Out(0), Weight(0)`.
#[derive(Clone, Debug, PartialEq)]
pub enum Slot<'a> {
    /// A range of a buffer: an operand, a weight, or a driver resource.
    Buffer(Bound<'a>),
    /// The slot the row reserves for this launch's scalars.
    ///
    /// The caller allocates and fills it -- see [`params`] -- because a
    /// buffer needs a device and this module is arithmetic.
    Params,
    /// A slot the row states and nothing fills.
    ///
    /// `Source::Unbound`. `kv_append_paged` has six, kept so that the rest of
    /// its row stays at the positions a shared ring ABI put them; the module
    /// leaves them as descriptor holes and nothing reads them.
    Nothing,
}

/// Where each of a launch's binding slots gets what it holds.
///
/// This is the step the first version of this file did not have, and its
/// absence was silent: binding a plan's operands positionally agrees with the
/// row for 1094 of the 3992 rectangles this tree had when it was measured
/// and disagrees for **2898**,
/// across twelve symbols. Every one of those dispatched, and every one bound
/// a real buffer of a plausible size to the wrong slot.
///
/// `rms_single_row` is the clearest: the shader is `0=x, 1=w, 2=out`, the row
/// says `In(0), Weight(0), Out(0)`, and the trace says `In(0), Out(0),
/// Weight(0)`. Positionally the norm reads its own output as the weight and
/// writes the weight buffer.
///
/// # Errors
///
/// The first [`Unbindable`] any slot produces, with the SLOT's index -- not
/// the plan operand's, since a refusal points at a descriptor.
pub fn reorder<'a, R: Resolve>(
    sig: &kernels::KernelSig,
    lowered: &Lowered,
    launch: &Launch,
    arena: Arena<'a>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Vec<Slot<'a>>, (usize, Unbindable)> {
    // A row that states no operands has never told anyone an order, so the
    // trace's is the only one there is.
    if sig.operands.is_empty() {
        return Ok(bind(lowered, launch, arena, resolver, min_offset)?
            .into_iter()
            .map(Slot::Buffer)
            .collect());
    }

    let span = launch.args.start as usize..launch.args.end as usize;
    let args = &lowered.args[span];

    // The trace's three runs. `Launch::args` states inputs in operand order,
    // then outputs, then the weights -- so the weights are the ones that ARE
    // `Arg::Weight`, and the split between inputs and outputs is by count.
    let widthed: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| !matches!(a, Arg::Weight(_)))
        .map(|(i, _)| i)
        .collect();
    let weights: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, Arg::Weight(_)))
        .map(|(i, _)| i)
        .collect();
    // How many of the widthed run are outputs: one past the highest `Out(i)`
    // the row names. Clamped to what the trace actually handed over, because
    // a row may state an output a given statement does not produce.
    let results = sig
        .operands
        .iter()
        .filter_map(|o| match o.source {
            kernels::Source::Out(i) => Some(usize::from(i) + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .min(widthed.len());
    let (ins, outs) = widthed.split_at(widthed.len() - results);

    let layer = launch.layers.start;
    let mut slots = Vec::with_capacity(sig.operands.len());
    for (slot, operand) in sig.operands.iter().enumerate() {
        let one = |at: Option<&usize>| -> Result<Slot<'a>, Unbindable> {
            let at = at.ok_or(Unbindable::NoOperand)?;
            resolve(&args[*at], launch, arena, resolver, min_offset).map(Slot::Buffer)
        };
        let held = |b: Option<&'a Buffer>, what: FireTable| -> Result<Slot<'a>, Unbindable> {
            b.map(Bound::whole)
                .map(Slot::Buffer)
                .ok_or(Unbindable::NoDriverResource(what))
        };
        let got = match operand.source {
            kernels::Source::In(i) => one(ins.get(i as usize)),
            kernels::Source::Out(i) => one(outs.get(i as usize)),
            kernels::Source::Weight(i) => one(weights.get(i as usize)),
            // The KV cache is per-LAYER state. The layer span of a rectangle
            // is always one wide, so its start is the layer.
            kernels::Source::KvKeys => resolver
                .kv(layer, false)
                .map(Bound::whole)
                .map(Slot::Buffer)
                .ok_or(Unbindable::NoKvCache {
                    layer,
                    values: false,
                }),
            kernels::Source::KvValues => resolver
                .kv(layer, true)
                .map(Bound::whole)
                .map(Slot::Buffer)
                .ok_or(Unbindable::NoKvCache {
                    layer,
                    values: true,
                }),
            kernels::Source::TokenIds => {
                held(resolver.table(FireTable::TokenIds), FireTable::TokenIds)
            }
            kernels::Source::Positions => {
                held(resolver.table(FireTable::Positions), FireTable::Positions)
            }
            kernels::Source::RequestOfToken => held(
                resolver.table(FireTable::RequestOfToken),
                FireTable::RequestOfToken,
            ),
            kernels::Source::KvPageIndices => held(
                resolver.table(FireTable::KvPageIndices),
                FireTable::KvPageIndices,
            ),
            kernels::Source::KvPageIndptr => held(
                resolver.table(FireTable::KvPageIndptr),
                FireTable::KvPageIndptr,
            ),
            kernels::Source::AttentionMask => held(
                resolver.table(FireTable::AttentionMask),
                FireTable::AttentionMask,
            ),
            kernels::Source::AttentionMaskEnabled => held(
                resolver.table(FireTable::AttentionMaskEnabled),
                FireTable::AttentionMaskEnabled,
            ),
            kernels::Source::KvWritePage => held(
                resolver.table(FireTable::KvWritePage),
                FireTable::KvWritePage,
            ),
            kernels::Source::KvWriteOffset => held(
                resolver.table(FireTable::KvWriteOffset),
                FireTable::KvWriteOffset,
            ),
            kernels::Source::RopeFrequencies => held(
                resolver.table(FireTable::RopeFrequencies),
                FireTable::RopeFrequencies,
            ),
            kernels::Source::SamplingIndices => held(
                resolver.table(FireTable::SamplingIndices),
                FireTable::SamplingIndices,
            ),
            // Slots the row states and nothing binds. `Unbound` is a gap kept
            // on purpose; everything else here is a SCALAR, which does not
            // come out of the operand list at all -- it rides the parameter
            // block at whatever slot the row placed it, and `params` decides
            // where that is.
            kernels::Source::Unbound => Ok(Slot::Nothing),
            _ => Ok(Slot::Params),
        };
        slots.push(got.map_err(|e| (slot, e))?);
    }
    Ok(slots)
}

/// Where a launch's scalars go.
///
/// A plan states its operands and its scalars separately, which is already the
/// shape Vulkan wants -- descriptors on one side, a push block on the other.
/// But only half the reachable kernels take them that way. `tests/arena.rs`
/// measures the split over every symbol three real texts launch: six take
/// their scalars as a push block, and six take them as a plain struct in a
/// storage buffer of their own. Seven, once descriptor holes are subtracted:
/// `affine_qmv_routed` reads as short of a buffer until the slot no shader
/// reads is discounted, and then it is an ordinary push-block kernel.
///
/// Which one a kernel wants is not a naming convention and not a list to keep
/// up to date. It is read off the compiled module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Params {
    /// The scalars are the module's push-constant block.
    ///
    /// The bytes are laid out at the offsets the SHADER declares, not packed
    /// end to end, because those are not always the same thing and the
    /// difference is silent -- `crate::lowering::pack` exists for that reason
    /// and this is the same rule applied to a plan.
    Push(Vec<u8>),
    /// The scalars are a struct in a storage buffer, at this binding.
    ///
    /// The caller allocates it, writes these bytes, and binds it at `at`. It
    /// is the caller's because a buffer needs a device and this module is
    /// arithmetic; keeping it that way is what lets the split be decided on a
    /// machine with no GPU.
    Block {
        /// The struct's bytes, exactly as long as the shader's block.
        bytes: Vec<u8>,
        /// Which binding to put it at.
        at: usize,
    },
    /// The module declares neither, and the launch states no scalars.
    None,
}

/// Why a launch's scalars could not be placed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Misplaced {
    /// The module wants its scalars somewhere, and the two somewheres
    /// disagree about how many there are.
    ///
    /// This is the defect the whole parameter side exists to catch, and it has
    /// no symptom on this backend: `robustBufferAccess` is on, so a block
    /// short of what the shader reads returns ZEROS rather than faulting.
    /// `tests/device.rs` shows a dispatch accepting one, producing 256 zeros,
    /// and the validation layer saying nothing at all.
    Count {
        /// How many scalars the plan states.
        stated: usize,
        /// How many the module's push block holds.
        push: usize,
        /// The sizes of every knowable block the module declares, in bytes.
        blocks: Vec<u32>,
    },
}

/// Place one launch's scalars the way its module wants them.
///
/// # Errors
///
/// [`Misplaced::Count`] when neither of the module's two shapes can hold what
/// the plan states.
pub fn params(
    lowered: &Lowered,
    launch: &Launch,
    declared: &crate::spirv::Declared,
) -> Result<Params, Misplaced> {
    let stated = &lowered.params[launch.params.start as usize..launch.params.end as usize];
    params_from(stated, declared)
}

/// The scalars one launch hands one module, in the order the ROW places them.
///
/// The statement's own run is a flat list the row indexes into, and a row may
/// index into only part of it: `neox_mb` states `ParamF32(0), ParamF32(1),
/// Param(2)` against a plan that carries four, and its module has room for
/// three. Taking the run whole was refusing it as one scalar too many.
///
/// A row also interleaves numbers the DRIVER resolves. `kv_append_paged` is
/// `Param(0), KvPageSize, Param(1)` -- the page size is a property of the
/// pool, not of the statement, and it lands between the two scalars the
/// statement did carry, not after them. Both paged decodes do the same, which
/// is exactly the one word each was short of.
///
/// # Errors
///
/// [`Misplaced`], as [`params`] -- with the run this built, so the count in
/// the refusal is the count the module would have been given.
pub fn scalars<R: Resolve>(
    sig: &kernels::KernelSig,
    lowered: &Lowered,
    launch: &Launch,
    declared: &crate::spirv::Declared,
    resolver: &R,
) -> Result<Params, Misplaced> {
    let stated = &lowered.params[launch.params.start as usize..launch.params.end as usize];
    if sig.operands.is_empty() {
        return params_from(stated, declared);
    }
    let mut run: Vec<u32> = Vec::new();
    for operand in sig.operands {
        // A field of the preceding packed struct: the driver's number, added
        // to the run the struct covers. `row_gather` is the only one --
        // `Param(0)` there is a `Buf`, which is how a row says "the rest of
        // this run is a struct in a buffer".
        if operand.ty == kernels::Ty::InPacked {
            run.push(match operand.source {
                kernels::Source::RequestCount => lowered.n_requests,
                _ => 0,
            });
            continue;
        }
        let number = match operand.source {
            kernels::Source::KvPageSize => Some(FireNumber::KvPageSize),
            kernels::Source::KvHeadStride => Some(FireNumber::KvHeadStride),
            kernels::Source::KvSeqStride => Some(FireNumber::KvSeqStride),
            _ => None,
        };
        if let Some(want) = number {
            // Zero rather than a refusal, matching `driver-metal`: a pool
            // that has not been built yet has no page size, and the caller
            // that has not built one is not dispatching against it either.
            //
            // What lands here was unwatched for a long time and looked
            // watched. `Pool` answering correctly is checked in `resources`;
            // the shader addressing correctly is checked in `tests/device.rs`
            // -- which hand-writes its push constants and so never comes
            // through here. Between them this line could hand attention a
            // constant and every test stayed green. `tests/arena.rs`'s
            // `every_row_naming_a_pool_number_is_handed_that_number_and_not_another`
            // now pins the whole run for every row in the table that names
            // one, which is also what makes SWAPPING the two strides a
            // failure -- the defect with the most plausible output, since
            // both numbers are present either way.
            run.push(resolver.number(want).unwrap_or(0));
            continue;
        }
        match operand.source {
            kernels::Source::Param(i) | kernels::Source::ParamF32(i) => {
                // A pointer where a scalar could be is how a row says "the
                // rest of this run is a struct, and it starts here". So it
                // takes the whole tail, not one word: `rms_single_row`'s row
                // is one `Param(0)` against five scalars, and picking the
                // single word at index 0 refused it as four too many.
                if matches!(operand.ty, kernels::Ty::Buf | kernels::Ty::BufMut) {
                    run.extend_from_slice(stated.get(usize::from(i)..).unwrap_or(&[]));
                } else {
                    // The `unwrap_or` is unreachable across every row and
                    // text measured here -- replacing the zero with a one
                    // changes nothing, and asserting inside it never fires.
                    // Kept because the alternative is an index panic on a
                    // statement that carried fewer scalars than its row
                    // indexes, which is a plan defect this crate would rather
                    // report as a short run than as a crash.
                    run.push(stated.get(usize::from(i)).copied().unwrap_or(0));
                }
            }
            _ => {}
        }
    }
    params_from(&run, declared)
}

fn params_from(stated: &[u32], declared: &crate::spirv::Declared) -> Result<Params, Misplaced> {
    // Asked in this order because push is the stronger claim: it accounts for
    // every descriptor as well as every scalar, and a module that declares a
    // push block of the right size is not also hiding a parameter buffer.
    if declared.push_offsets.len() == stated.len() {
        if stated.is_empty() {
            return Ok(Params::None);
        }
        // Sized from the block's own extent rather than from four bytes per
        // scalar: `vkCmdPushConstants` takes a size, and a block with a gap
        // in it needs the gap written or the range does not cover the members
        // after it.
        let end = declared
            .push_offsets
            .iter()
            .map(|o| *o as usize + 4)
            .max()
            .unwrap_or(0);
        let mut bytes = vec![0u8; end];
        for (word, offset) in stated.iter().zip(&declared.push_offsets) {
            let at = *offset as usize;
            bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        return Ok(Params::Push(bytes));
    }

    // Found by SIZE and not by position. Looking for it at the binding one
    // past the operand count is the obvious guess and is wrong for two of the
    // six: `combine_sorted` binds its 12-byte block at 3 of 5 and `route_sort`
    // its 28-byte block at 4 of 6, each with an operand after it. Where a
    // parameter block sits is the kernel's own ABI.
    let want = stated.len() as u32 * 4;
    if want > 0
        && let Some(at) = declared.block_bytes.iter().position(|b| *b == Some(want))
    {
        let mut bytes = Vec::with_capacity(want as usize);
        for word in stated {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        return Ok(Params::Block { bytes, at });
    }

    if stated.is_empty() && declared.push_offsets.is_empty() {
        return Ok(Params::None);
    }

    Err(Misplaced::Count {
        stated: stated.len(),
        push: declared.push_offsets.len(),
        blocks: declared.block_bytes.iter().flatten().copied().collect(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A `Buffer` this test can name without a device.
    ///
    /// Binding never dereferences the handle -- it produces offsets and
    /// lengths against it -- so a zeroed one is enough to ask every question
    /// in this module, and asking them without a GPU is the point.
    fn buffer(bytes: u64) -> Buffer {
        Buffer::placeholder(bytes)
    }

    #[derive(Default)]
    struct Store {
        weights: BTreeMap<String, Buffer>,
        named: BTreeMap<ValueId, Buffer>,
    }

    impl Resolve for Store {
        fn weight(&self, name: &str) -> Option<&Buffer> {
            self.weights.get(name)
        }
        fn named(&self, value: ValueId) -> Option<&Buffer> {
            self.named.get(&value)
        }
    }

    /// A launch over `rows` rows with `n` operands starting at zero.
    fn launch(rows: u32, n: u32) -> Launch {
        Launch {
            kernel: 0,
            rows: 0..rows,
            layers: 0..1,
            op: 0,
            args: 0..n,
            params: 0..0,
            peel: None,
            cond: 0,
        }
    }

    #[test]
    fn an_arena_operands_extent_is_its_rectangle_and_not_one_row() {
        let arg = Arg::Arena {
            at: 0,
            width: 128,
            bytes: 2,
        };
        // The distinction Metal never had to draw: one row is 256 bytes, and
        // a 64-row prefill launch of the same operand covers 64 times that.
        assert_eq!(extent(&arg, &launch(1, 1)), Some(256));
        assert_eq!(extent(&arg, &launch(64, 1)), Some(64 * 256));
    }

    #[test]
    fn a_weight_and_a_seam_value_do_not_get_their_extent_from_the_plan() {
        assert_eq!(extent(&Arg::Weight("w".into()), &launch(1, 1)), None);
        assert_eq!(
            extent(&Arg::Named { value: 0, width: 8 }, &launch(1, 1)),
            None
        );
    }

    #[test]
    fn an_arena_operand_is_bound_at_its_offset_for_its_rectangle() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let arg = Arg::Arena {
            at: 512,
            width: 64,
            bytes: 2,
        };
        let store = Store::default();
        let b = resolve(&arg, &launch(4, 1), arena, &store, 16).expect("bindable");
        assert_eq!(b.offset(), 512);
        // Four rows of 64 elements at 2 bytes -- NOT the 512 one row would
        // give, and not the rest of the arena `VK_WHOLE_SIZE` would give.
        assert_eq!(b.len(), 512);
    }

    #[test]
    fn an_operand_whose_rectangle_runs_past_the_arena_is_refused() {
        let buf = buffer(1 << 20);
        // The buffer is a megabyte; the PLAN said the arena is 1024 bytes.
        // The refusal has to come from the plan's number, or a driver holding
        // a generously sized arena would accept an operand that addresses
        // another fire's bytes.
        let arena = Arena {
            buffer: &buf,
            bytes: 1024,
        };
        let arg = Arg::Arena {
            at: 768,
            width: 64,
            bytes: 2,
        };
        let err =
            resolve(&arg, &launch(4, 1), arena, &Store::default(), 16).expect_err("runs past");
        assert_eq!(
            err,
            Unbindable::PastArena {
                at: 768,
                extent: 512,
                arena: 1024
            }
        );
    }

    #[test]
    fn an_operand_the_device_cannot_address_from_is_refused_as_such() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        // 260 is inside the arena and a multiple of 4, so the plan and Metal
        // are both content. It is not a multiple of 256.
        let arg = Arg::Arena {
            at: 260,
            width: 64,
            bytes: 2,
        };
        let err =
            resolve(&arg, &launch(1, 1), arena, &Store::default(), 256).expect_err("misaligned");
        assert!(
            matches!(
                err,
                Unbindable::Unaddressable(crate::device::Failed::Unaligned { .. })
            ),
            "an offset the device cannot use is not the same refusal as one the \
             plan oversized: {err:?}"
        );
    }

    #[test]
    fn a_weight_and_a_seam_value_come_from_the_resolver_whole() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let mut store = Store::default();
        store.weights.insert("layer.3.q_proj".into(), buffer(4096));
        store.named.insert(7, buffer(64));

        let w = resolve(
            &Arg::Weight("layer.3.q_proj".into()),
            &launch(1, 1),
            arena,
            &store,
            16,
        )
        .expect("held");
        // Whole, because the plan does not state a weight's extent and the
        // tensor's own size is the right answer.
        assert_eq!((w.offset(), w.len()), (0, 4096));

        let n = resolve(
            &Arg::Named { value: 7, width: 8 },
            &launch(1, 1),
            arena,
            &store,
            16,
        )
        .expect("bound");
        assert_eq!((n.offset(), n.len()), (0, 64));
    }

    #[test]
    fn a_name_the_resolver_does_not_hold_is_named_in_the_refusal() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let store = Store::default();
        assert_eq!(
            resolve(
                &Arg::Weight("layer.3.q_proj".into()),
                &launch(1, 1),
                arena,
                &store,
                16
            )
            .expect_err("not held"),
            Unbindable::UnknownWeight("layer.3.q_proj".into())
        );
        assert_eq!(
            resolve(
                &Arg::Named { value: 7, width: 8 },
                &launch(1, 1),
                arena,
                &store,
                16
            )
            .expect_err("not bound"),
            Unbindable::UnknownNamed(7)
        );
    }

    /// The one place this backend must refuse where Metal proceeds.
    #[test]
    fn a_slot_holding_a_dispatch_constant_cannot_be_a_range_on_this_backend() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        // Metal binds `{address: 0, bytes: 0}` here and calls it honest,
        // because a zero-length Metal binding is legal and unread. Vulkan has
        // no such descriptor, so the honest answer is a refusal that says
        // which constant the caller still owes.
        assert_eq!(
            resolve(
                &Arg::Weight("scale.rope_theta".into()),
                &launch(1, 1),
                arena,
                &Store::default(),
                16
            )
            .expect_err("a constant is not a range"),
            Unbindable::Constant("rope_theta".into())
        );
    }

    #[test]
    fn a_launch_binds_whole_or_not_at_all() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let mut store = Store::default();
        store.weights.insert("held".into(), buffer(64));
        let lowered = lowered(vec![
            Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            },
            Arg::Weight("held".into()),
            Arg::Weight("absent".into()),
        ]);

        let (i, err) = bind(&lowered, &launch(1, 3), arena, &store, 16).expect_err("one is absent");
        // The index is reported because a refusal that only names the weight
        // cannot say WHICH operand slot the dispatch was going to leave stale.
        assert_eq!(i, 2);
        assert_eq!(err, Unbindable::UnknownWeight("absent".into()));

        // The same launch minus the absent operand binds all of what remains.
        assert_eq!(
            bind(&lowered, &launch(1, 2), arena, &store, 16)
                .expect("both held")
                .len(),
            2
        );
    }

    #[test]
    fn operands_come_back_in_the_order_the_plan_states_them() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        // Three arena operands at distinguishable offsets. Descriptor slots
        // are positional, so an order this crate rearranged would hand every
        // kernel its inputs shuffled and no test of one operand would notice.
        let lowered = lowered(
            [1024usize, 256, 2048]
                .into_iter()
                .map(|at| Arg::Arena {
                    at,
                    width: 8,
                    bytes: 2,
                })
                .collect(),
        );
        let store = Store::default();
        let bound = bind(&lowered, &launch(1, 3), arena, &store, 16).expect("bindable");
        assert_eq!(
            bound.iter().map(Bound::offset).collect::<Vec<_>>(),
            [1024, 256, 2048]
        );
    }

    /// A module declaring a push block at the given offsets and blocks of the
    /// given sizes.
    fn declared(push: &[u32], blocks: &[Option<u32>]) -> crate::spirv::Declared {
        crate::spirv::Declared {
            local: [1, 1, 1],
            bindings: blocks.len() as u32,
            used: vec![true; blocks.len()],
            reads_workgroup_count: false,
            grid_axes: [true, false, false],
            push_offsets: push.to_vec(),
            block_bytes: blocks.to_vec(),
        }
    }

    /// A `Lowered` whose only content is the scalars under test.
    fn with_params(words: Vec<u32>) -> Lowered {
        let mut low = lowered(Vec::new());
        low.params = words;
        low
    }

    fn scalar_launch(n: u32) -> Launch {
        let mut l = launch(1, 0);
        l.params = 0..n;
        l
    }

    #[test]
    fn scalars_a_push_block_holds_are_written_at_the_offsets_the_shader_declares() {
        let low = with_params(vec![7, 9]);
        // The gap is the point. Packed end to end these would be at 0 and 4;
        // the shader says 0 and 8, and a driver that packed them would hand
        // the second member's value to whatever sits at 4.
        let got = params(&low, &scalar_launch(2), &declared(&[0, 8], &[None])).expect("placed");
        assert_eq!(got, Params::Push(vec![7, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]));
    }

    #[test]
    fn scalars_a_block_holds_go_to_the_binding_whose_size_matches() {
        let low = with_params(vec![1, 2, 3]);
        // Three scalars is twelve bytes, and the module declares a 12-byte
        // block at binding 1 -- with an operand after it, which is why the
        // position cannot be derived from the operand count.
        let got = params(
            &low,
            &scalar_launch(3),
            &declared(&[], &[None, Some(12), None]),
        )
        .expect("placed");
        assert_eq!(
            got,
            Params::Block {
                bytes: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
                at: 1
            }
        );
    }

    #[test]
    fn a_launch_with_no_scalars_and_a_module_with_no_block_place_nothing() {
        let low = with_params(Vec::new());
        assert_eq!(
            params(&low, &scalar_launch(0), &declared(&[], &[None])).expect("placed"),
            Params::None
        );
    }

    #[test]
    fn scalars_neither_shape_can_hold_are_refused_rather_than_truncated() {
        let low = with_params(vec![1, 2, 3, 4]);
        // Four scalars; the push block holds two and the only sized block is
        // twelve bytes. Writing two and leaving the shader to read four is
        // the defect with no symptom -- `robustBufferAccess` returns zeros --
        // so it has to be a refusal.
        let err = params(
            &low,
            &scalar_launch(4),
            &declared(&[0, 4], &[None, Some(12)]),
        )
        .expect_err("neither fits");
        assert_eq!(
            err,
            Misplaced::Count {
                stated: 4,
                push: 2,
                blocks: vec![12]
            }
        );
    }

    #[test]
    fn a_push_block_with_a_hole_after_its_last_member_is_not_written_short() {
        let low = with_params(vec![5]);
        // One member at offset 12: the range `vkCmdPushConstants` gets must
        // reach 16, not 4, or the write does not cover the member at all.
        let got = params(&low, &scalar_launch(1), &declared(&[12], &[None])).expect("placed");
        assert_eq!(
            got,
            Params::Push(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0])
        );
    }

    /// A `Lowered` holding nothing but the operands under test.
    fn lowered(args: Vec<Arg>) -> Lowered {
        Lowered {
            launches: Vec::new(),
            kernels: Vec::new(),
            rectangles: 0,
            arena_bytes: 0,
            value_offset: Vec::new(),
            value_owner: Vec::new(),
            epilogue_gather: usize::MAX,
            epilogue_norm: usize::MAX,
            args,
            structural: Vec::new(),
            residue: Vec::new(),
            params: Vec::new(),
            n_requests: 0,
            conds: Vec::new(),
            readout: None,
        }
    }
}

/// The refusals [`descriptors`] produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unlayoutable {
    /// The row states more slots than the module declares, and the excess is
    /// not all unbound.
    ///
    /// A row states every deployment its kernel serves, so a row longer than
    /// one module is ordinary -- but only where the tail is `Unbound`. A
    /// buffer past the end is a buffer nothing can hold.
    Overlong {
        /// How many slots the row states.
        stated: usize,
        /// How many the module declares.
        module: u32,
    },
    /// The scalar block's slot is not where the row puts its parameters.
    ///
    /// The module says which binding is the block; the row says which operand
    /// is `Param(0)`. Both are read off separately, so their disagreement is
    /// a finding rather than a fact -- and it means the shader would read its
    /// scalars out of an operand.
    BlockElsewhere {
        /// Where the SPIR-V puts the block.
        module: usize,
        /// Where the row's parameters land.
        row: usize,
    },
    /// The row leaves a slot unbound that the module decorates and reads.
    ///
    /// Measured to be exactly the descriptor holes, on both the modules that
    /// have any: `affine_qmv_routed` 1 and 1, `kv_append_paged` 6 and 6. So a
    /// mismatch means one of the two readings is wrong.
    Unfilled {
        /// The slot the row leaves empty.
        at: usize,
    },
}

impl core::fmt::Display for Unlayoutable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Overlong { stated, module } => {
                write!(
                    f,
                    "the row states {stated} slots and the module declares {module}"
                )
            }
            Self::BlockElsewhere { module, row } => {
                write!(
                    f,
                    "the module's block is binding {module} and the row's is {row}"
                )
            }
            Self::Unfilled { at } => write!(f, "binding {at} is read and nothing fills it"),
        }
    }
}

impl core::error::Error for Unlayoutable {}

/// Cut a row's slots down to the module's bindings.
///
/// [`reorder`] answers "what does slot *k* of the ROW hold"; this answers
/// "what does binding *k* of the MODULE hold", and they differ in two
/// measured ways.
///
/// A scalar occupies a row slot whether or not it occupies a descriptor. Six
/// of this tree's reachable symbols read their parameters as a struct in a
/// storage buffer and that buffer IS a binding; seven push theirs, and their
/// scalar slots take no descriptor at all, so everything after them moves
/// down. `route_sort`'s row is `In0,Out0,Out1,Out2,P0,Out3` and its block is
/// binding 4, which is where the row's `P0` lands -- read off the SPIR-V and
/// off the table separately, and checked here against each other.
///
/// And a row may be longer than a module. `router_topk` and
/// `sdpa_paged_decode` both end in `Unbound`, one slot past a layout that
/// does not declare it.
///
/// # Errors
///
/// [`Unlayoutable`], which names which of the two readings disagreed.
pub fn descriptors<'a>(
    slots: Vec<Slot<'a>>,
    placed: &Params,
    declared: &crate::spirv::Declared,
) -> Result<Vec<Slot<'a>>, Unlayoutable> {
    let block = match placed {
        Params::Block { at, .. } => Some(*at),
        Params::Push(_) | Params::None => None,
    };
    let mut out = Vec::with_capacity(declared.bindings as usize);
    let mut seen = false;
    for slot in slots {
        match slot {
            // Only the FIRST scalar slot can be the block: a row states its
            // parameters as a run, and a run has one head. The rest are more
            // of the same struct, or numbers the driver appends.
            Slot::Params if block.is_some() && !seen => {
                seen = true;
                out.push(Slot::Params);
            }
            Slot::Params => {}
            other => out.push(other),
        }
    }

    // The tail past the layout, which has to be nothing at all.
    while out.len() > declared.bindings as usize {
        match out.pop() {
            Some(Slot::Nothing) => {}
            _ => {
                return Err(Unlayoutable::Overlong {
                    stated: out.len() + 1,
                    module: declared.bindings,
                });
            }
        }
    }

    if let (Some(at), Some(row)) = (block, out.iter().position(|s| matches!(s, Slot::Params)))
        && at != row
    {
        return Err(Unlayoutable::BlockElsewhere { module: at, row });
    }

    // Every empty slot has to be one the module never reads. This is the
    // check that pins the two readings together: measured, the unbound slots
    // and the descriptor holes are the same slots, on both modules that have
    // either.
    for (at, slot) in out.iter().enumerate() {
        // `used.get` is never out of range for any module here -- measured,
        // by asserting inside the fallback and seeing nothing fire. `false`
        // is still the right default: a slot the reflection did not describe
        // is one this crate cannot claim the shader reads.
        // `used.get` is never out of range for any module here -- measured,
        // by asserting inside the fallback and seeing nothing fire. `false`
        // is still the right default: a slot the reflection did not describe
        // is one this crate cannot claim the shader reads.
        if matches!(slot, Slot::Nothing) && declared.used.get(at).copied().unwrap_or(false) {
            return Err(Unlayoutable::Unfilled { at });
        }
    }
    Ok(out)
}
