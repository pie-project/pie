//! Turning a lowering's operands into bind-group entries.
//!
//! [`model_compiler::lower`] hands a driver a flat list of [`Arg`]s and says
//! *"bind these"*. On Metal that is nearly free: `setBuffer:offset:` takes an
//! address and a byte offset and no length at all, so `driver-metal`'s binder
//! resolves each operand to its base plus its offset, reports the rest of the
//! arena as the extent, and is done. The extent is not load-bearing there
//! because nothing reads it.
//!
//! WebGPU has no such call. A `BufferBinding` is a buffer, an offset AND a
//! size, and all three are validated. So this module has to answer a question
//! Metal never asked: **how many bytes is this operand?**
//!
//! # The extent is in the plan, and it is exact
//!
//! It is `rows × width × bytes`: the launch states its row count, and the
//! operand states its row width and its element size. Measured on the Vulkan
//! side over every arena operand six real texts produce in both fire classes
//! -- 14324 of them -- every extent lands inside the arena, and the tightest
//! fit has **zero** bytes to spare. An operand ends exactly where the arena
//! does.
//!
//! That zero is the useful part. It says the formula is not a lower bound that
//! happens to be safe, it is the real extent: if it were an under-estimate the
//! slack would never reach zero, and if it were an over-estimate it would have
//! run past.
//!
//! # Why the size matters more here than a length usually does
//!
//! The arena is ONE buffer holding every activation. Binding the whole of it
//! for an operand at offset `at` therefore covers every tensor allocated
//! after it, and a kernel indexing past its own rows would read or write a
//! neighbour. WGSL requires an implementation to bounds-check every access
//! against the BOUND range, so a real size confines a stray index -- and
//! confines it to a zero rather than to a fault, which is why an operand bound
//! too LONG is a silent wrong answer and not a crash.
//!
//! # The buffer type is not this module's, and how that is arranged
//!
//! `driver-vulkan`'s binder names `crate::device::Buffer` directly, so its
//! whole arithmetic depends on a module full of device handles. This one does
//! not name a buffer type at all: [`Resolve`] carries an associated
//! [`Resolve::Buffer`], bounded by [`Allocation`], which asks a buffer for the
//! only thing binding needs to know -- how many bytes it holds.
//!
//! **The device half implements [`Allocation`] for its own buffer and names it
//! as `Resolve::Buffer`.** That is the whole seam, and it was chosen over a
//! newtype declared here because a newtype would either lose the handle -- and
//! then the device half has to carry a side table from newtype to real buffer
//! -- or hold one, and then this module is not portable after all. An
//! associated type costs a generic parameter on five functions and buys a
//! binder whose every offset, extent and refusal is checkable with
//! [`Placeholder`] on a machine with no adapter.
//!
//! # What this module does not do
//!
//! It resolves operands and places scalars. It does not create a bind group,
//! allocate the uniform buffer or write it: those need a device, and the whole
//! point of the split is that the arithmetic does not.

use model_compiler::lower::{Arg, Launch, Lowered};
use model_compiler::trace::ValueId;

/// What binding needs to know about a device allocation.
///
/// One method, because one number is all the arithmetic uses: an offset is
/// checked against a length and a length is checked against a size. The
/// handle, the usage flags, the memory it lives in and the queue that wrote it
/// are all the device half's business.
///
/// `PartialEq` is a supertrait rather than a method, and it is doing real
/// work: two [`Bound`]s are the same range when they name the same MEMORY, and
/// a caller holding two clones of one buffer handle must find its two
/// identical ranges equal. The device half decides what that means for its own
/// type -- for a `wgpu::Buffer` it is the underlying resource, and a derived
/// `PartialEq` over a struct wrapping one is the ordinary answer.
pub trait Allocation: PartialEq {
    /// Bytes the allocation holds.
    fn size(&self) -> u64;
}

/// A buffer that is a size and nothing else.
///
/// Binding produces offsets and lengths AGAINST a buffer and never touches the
/// allocation, so a size is enough to ask every question in this module -- and
/// asking them without an adapter is the point, since the machines that change
/// a lowering are not the machines that run a fire.
///
/// Public rather than test-only because `tests/` and the device half's own
/// unit tests both want it, and a driver that needed a GPU to check its arena
/// arithmetic would only ever check it where it was already working.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placeholder(
    /// The size it reports.
    pub u64,
);

impl Allocation for Placeholder {
    fn size(&self) -> u64 {
        self.0
    }
}

/// The frame's arena: one buffer, every activation.
#[derive(Debug)]
pub struct Arena<'a, B> {
    /// The buffer the offsets are into.
    pub buffer: &'a B,
    /// How many bytes of it the plan was allowed to place into.
    ///
    /// Stated separately from the buffer's own size because a driver may hold
    /// a larger arena than a given fire needs, and the question an operand
    /// asks is whether it fits the PLAN's arena. A buffer big enough to
    /// contain a mistake still contains one.
    pub bytes: u64,
}

// Hand-written rather than derived: `derive(Clone)` would add a `B: Clone`
// bound, and an arena is a REFERENCE to a buffer -- copying it never copies
// the allocation, so the bound would be a requirement on the device half for
// nothing.
impl<B> Clone for Arena<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Arena<'_, B> {}

/// What one bind-group entry addresses: a buffer, and which part of it.
///
/// # Why this is not `&B`
///
/// A driver does not allocate a buffer per tensor. It allocates an arena and
/// hands out offsets into it, because a fire's activations are hundreds of
/// values whose lifetimes nest and whose sizes are known together. WebGPU's
/// `min_storage_buffer_offset_alignment` says how coarsely those offsets may
/// fall. So the arena model needs a type that carries the offset and can be
/// REFUSED, rather than a reference that cannot.
///
/// The extent travels with the address for the reason `driver-metal`'s binder
/// gives: an arena reused across fires can be smaller than the new one needs,
/// and an operand whose length lives in a neighbouring field is a bound two
/// call sites have to agree about.
#[derive(Debug)]
pub struct Bound<'a, B> {
    buffer: &'a B,
    offset: u64,
    len: u64,
}

impl<B> Clone for Bound<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Bound<'_, B> {}

/// Two bounds are the same range when they name the same MEMORY, not the same
/// `&B`.
///
/// Written out rather than derived because the derived version would compare
/// the reference, and a caller holding two handles to one allocation would
/// find its two identical ranges unequal. What a test asks of a dispatch is
/// where it points, and the answer is the buffer, the offset and the length --
/// which is why [`Allocation`] requires `PartialEq` of the buffer itself.
impl<B: Allocation> PartialEq for Bound<'_, B> {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer && self.offset == other.offset && self.len == other.len
    }
}

impl<B: Allocation + Eq> Eq for Bound<'_, B> {}

/// Why a range could not be addressed.
///
/// Its own type rather than a device one, because both conditions are
/// arithmetic: an alignment and a bound. `driver-vulkan` puts the same two
/// cases in `device::Failed` next to a dozen conditions that need a device,
/// which is why its binder cannot be compiled without one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unaddressable {
    /// A sub-range starts at an offset the device will not address from.
    ///
    /// `min_storage_buffer_offset_alignment` is a hardware granularity, not a
    /// preference: `wgpu` refuses a `BufferBinding` whose offset it does not
    /// divide, and refuses it by validation rather than by returning an
    /// error, so a driver that let one through gets a panic from inside the
    /// encoder with nothing about the launch in it.
    ///
    /// The offset is REFUSED and never rounded. Rounding down aliases the
    /// previous tensor and rounding up drops the first rows of this one --
    /// both produce numbers, and the plan's own offsets are all multiples of
    /// 256 already, so a misaligned one is a defect in the lowering rather
    /// than a granularity to accommodate.
    Unaligned {
        /// Where the range wanted to start.
        offset: u64,
        /// What the device requires it to divide.
        alignment: u64,
    },
    /// The range leaves the buffer, or is empty.
    ///
    /// A zero-length range is this rather than a variant of its own: `wgpu`
    /// refuses a zero-sized binding, and it is also always the same defect --
    /// a width computed from a shape that came out empty -- so the numbers
    /// that produced it are the useful part of the message.
    Overrun {
        /// Where the range starts.
        offset: u64,
        /// How long it wanted to be.
        len: u64,
        /// What the buffer holds.
        size: u64,
    },
}

impl core::fmt::Display for Unaddressable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unaligned { offset, alignment } => {
                write!(f, "offset {offset} is not a multiple of {alignment}")
            }
            Self::Overrun { offset, len, size } => {
                write!(f, "{len} bytes at {offset} does not fit a buffer of {size}")
            }
        }
    }
}

impl core::error::Error for Unaddressable {}

impl<'a, B: Allocation> Bound<'a, B> {
    /// The whole buffer.
    ///
    /// Offset zero, which every alignment divides, so this cannot be refused
    /// and does not need the device to say so.
    pub fn whole(buffer: &'a B) -> Self {
        Self {
            buffer,
            offset: 0,
            len: buffer.size(),
        }
    }

    /// `len` bytes at `offset`, against a stated alignment.
    ///
    /// The alignment is passed in rather than read off a device for the reason
    /// [`crate::facts::of`] takes a number: binding a plan is arithmetic over
    /// offsets and extents, and the machines that CHANGE a plan -- the ones
    /// running `model-compiler`'s tests -- have no adapter to ask.
    ///
    /// It is also how a check can be made against the SPECIFICATION's
    /// [`crate::facts::GUARANTEED_STORAGE_ALIGNMENT`] rather than the local
    /// card's, which is the difference between knowing a plan binds here and
    /// knowing it binds anywhere -- including in a browser, which is the
    /// deployment this backend exists for.
    ///
    /// # Errors
    ///
    /// [`Unaddressable`], naming which of the two rules it broke.
    pub fn within(
        buffer: &'a B,
        offset: u64,
        len: u64,
        alignment: u64,
    ) -> Result<Self, Unaddressable> {
        // `max(1)` because the guarantee is a promise from an implementation,
        // and dividing by a zero one would panic where refusing is the whole
        // job. Every offset is a multiple of 1, so an implementation that
        // reports nothing constrains nothing.
        let alignment = alignment.max(1);
        if !offset.is_multiple_of(alignment) {
            return Err(Unaddressable::Unaligned { offset, alignment });
        }
        // Checked rather than added: an offset near `u64::MAX` would wrap to a
        // small sum and pass a bound it is nowhere near.
        let end = offset.checked_add(len);
        if len == 0 || end.is_none_or(|e| e > buffer.size()) {
            return Err(Unaddressable::Overrun {
                offset,
                len,
                size: buffer.size(),
            });
        }
        Ok(Self {
            buffer,
            offset,
            len,
        })
    }

    /// Where the range starts in its buffer.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Bytes the range covers.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Is the range empty? Never, by construction -- both constructors refuse
    /// it -- but clippy asks for it beside `len` and a caller may prefer to
    /// ask.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer the range lives in.
    #[must_use]
    pub fn buffer(&self) -> &'a B {
        self.buffer
    }
}

/// Where the operands this crate cannot resolve come from.
///
/// Two of the three [`Arg`] kinds name something the plan does not hold: a
/// weight by its trace name and a seam value by its id. Both are the driver's
/// own tables, so both are asked for rather than looked up.
///
/// Takes `&self` rather than `&mut self` -- unlike the Metal one, which can
/// return a copied address -- because a binding borrows the buffer it names,
/// and the borrow has to outlive the call that produced it.
///
/// # What the device half implements
///
/// ```ignore
/// impl driver_wgpu::binding::Allocation for device::Buffer {
///     fn size(&self) -> u64 { self.size }
/// }
///
/// impl driver_wgpu::binding::Resolve for Store {
///     type Buffer = device::Buffer;
///     fn weight(&self, name: &str) -> Option<&device::Buffer> { .. }
///     fn named(&self, value: ValueId) -> Option<&device::Buffer> { .. }
///     // `kv`, `number` and `table` are defaulted; a resolver serving a text
///     // without paged attention need not state them.
/// }
/// ```
pub trait Resolve {
    /// Whatever the device half calls an allocation.
    ///
    /// The seam: this module never names a buffer type of its own, so the
    /// device half chooses, and everything here is arithmetic over
    /// [`Allocation::size`].
    type Buffer: Allocation;

    /// The buffer holding a weight, by the name the trace states.
    fn weight(&self, name: &str) -> Option<&Self::Buffer>;

    /// The buffer holding a seam value the backend binds by name.
    fn named(&self, value: ValueId) -> Option<&Self::Buffer>;

    /// The KV cache for one layer, keys or values.
    ///
    /// STATE, not an operand: no traced value stands for it, so no plan
    /// mentions it and no arena holds it. `kv_append_paged` names both and the
    /// paged attentions read both.
    ///
    /// Defaulted to `None` so that a resolver serving a text without paged
    /// attention does not have to state a method it will never be asked for.
    /// The refusal that produces is [`Unbindable::NoDriverResource`], which
    /// names what was missing.
    fn kv(&self, _layer: u16, _values: bool) -> Option<&Self::Buffer> {
        None
    }

    /// One of the fire's own numbers.
    ///
    /// A pool's shape, not a statement's scalar. A text that stated its page
    /// size would be right for one deployment and silently wrong for the next,
    /// so the row names the number and the driver answers it.
    fn number(&self, _which: FireNumber) -> Option<u32> {
        None
    }

    /// One of the fire's own tables.
    ///
    /// Also state. The kernel row names WHICH; this forwards the name and
    /// never reads what it means, which is what keeps the driver from having
    /// an opinion about a table's contents.
    fn table(&self, _which: FireTable) -> Option<&Self::Buffer> {
        None
    }
}

/// The fire-wide numbers a kernel row may name.
///
/// Scalars rather than buffers, so these are appended to the parameter run
/// where the row places them -- not given a bind-group entry.
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
/// operand of the statement nor a scalar, so nothing but the driver can supply
/// them.
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

/// The marker a constant rides the weight-name slot under.
///
/// Transcribed from `driver-metal`: `dsl::cuda::scalar_mul` puts a scalar in an
/// operand slot so the launch's arity holds, and states that no binder looks
/// for it. Metal binds a zero-length region and calls that honest.
///
/// This backend cannot. `wgpu` refuses a zero-sized `BufferBinding` --
/// [`Bound::within`] refuses it as an overrun -- so a slot under this prefix
/// is one the caller must fill some other way, and saying so is the only
/// correct answer. It is [`Unbindable::Constant`] rather than a silent empty
/// range.
const SCALE_PREFIX: &str = "scale.";

/// Why an operand could not become a bind-group entry.
#[derive(Clone, Debug, PartialEq)]
pub enum Unbindable {
    /// A seam value whose rectangle is larger than the stand-in buffer.
    PastSeam {
        /// The value the plan named.
        value: ValueId,
        /// Bytes its rectangle covers.
        extent: u64,
        /// Bytes the stand-in holds.
        seam: u64,
    },
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
    /// See `SCALE_PREFIX`. The refusal is the finding: there is no binding
    /// that means "nothing" in WebGPU.
    Constant(
        /// The constant's name, with the `scale.` prefix removed.
        String,
    ),
    /// The row names an operand of the statement that the statement does not
    /// have.
    ///
    /// A row states the shape of every deployment its kernel serves, so a
    /// statement reaching only part of that shape is ordinary -- but a slot
    /// left holding nothing is an entry the shader reads, so it is a refusal
    /// rather than a gap. `Source::Unbound` is how a row says "gap".
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
        Unaddressable,
    ),
}

impl std::fmt::Display for Unbindable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PastSeam {
                value,
                extent,
                seam,
            } => write!(
                f,
                "seam value {value:?} covers {extent} bytes and the stand-in holds {seam}"
            ),
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

/// How many bytes one operand of this launch covers.
///
/// `None` only for a WEIGHT, whose extent is its tensor's and which the plan
/// does not carry. Everything else states a rectangle and an element width,
/// so everything else can be measured.
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
        // A seam value states the same three things an arena operand does --
        // rows from the launch, `width`, and `bytes` -- so it is measured the
        // same way. It answered `None` here until `Arg::Named` carried
        // `bytes`, which is what let a seam value bind a whole buffer without
        // anyone asking whether the rectangle fit in it.
        Arg::Named { width, bytes, .. } => {
            let rows = u64::from(launch.rows.end - launch.rows.start);
            Some(
                rows.saturating_mul(u64::from(*width))
                    .saturating_mul(u64::from(*bytes)),
            )
        }
        Arg::Weight(_) => None,
    }
}

/// Resolve one operand into a bind-group range.
///
/// `min_offset` is the device's `min_storage_buffer_offset_alignment`, which
/// the range is checked against here rather than at encode time: an operand
/// that cannot be addressed is a fact about the plan and the device together,
/// and the earliest place both are known is here.
///
/// # Errors
///
/// [`Unbindable`], naming which of the rules could not be applied.
pub fn resolve<'a, R: Resolve>(
    arg: &Arg,
    launch: &Launch,
    arena: Arena<'a, R::Buffer>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Bound<'a, R::Buffer>, Unbindable> {
    match arg {
        Arg::Arena { at, .. } => {
            let extent = extent(arg, launch).expect("an arena operand states its rectangle");
            let at64 = *at as u64;
            // Checked before `Bound::within` so the refusal names the ARENA.
            // The buffer may well be larger than the plan's arena, in which
            // case `Bound::within` would accept a range the plan had no right
            // to.
            if at64.saturating_add(extent) > arena.bytes {
                return Err(Unbindable::PastArena {
                    at: *at,
                    extent,
                    arena: arena.bytes,
                });
            }
            Bound::within(arena.buffer, at64, extent, min_offset).map_err(Unbindable::Unaddressable)
        }
        Arg::Named { value, .. } => {
            let held = resolver
                .named(*value)
                .ok_or(Unbindable::UnknownNamed(*value))?;
            // THE STAND-IN IS A SIZE, AND IT WAS NEVER CHECKED.
            //
            // `Deployment::seam` is documented as "the stand-in buffer's size,
            // which bounds the largest scalar block a fire can stage" -- a
            // bound nothing enforced. `Bound::whole` binds the buffer however
            // small it is, and WGSL bounds-checks every access against the
            // BOUND range, so a rectangle larger than the seam reads ZEROS
            // past the end. That is the same silent answer `Source::OutWidth`
            // produced, arriving by a different door: a plausible tensor, a
            // fire that succeeds, and a model that says something.
            //
            // The seam is 4 MiB by default and the values that ride it are
            // small, so this refuses nothing the engine builds today. It is
            // the deployment knob a caller may lower.
            if let Some(extent) = extent(arg, launch) {
                if extent > held.size() {
                    return Err(Unbindable::PastSeam {
                        value: *value,
                        extent,
                        seam: held.size(),
                    });
                }
            }
            Ok(Bound::whole(held))
        }
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
/// resolved is a dispatch that would read whatever the bind group happened to
/// hold in the others, which on a reused group is the previous launch's
/// operand and not garbage -- so it would look plausible.
pub fn bind<'a, R: Resolve>(
    lowered: &Lowered,
    launch: &Launch,
    arena: Arena<'a, R::Buffer>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Vec<Bound<'a, R::Buffer>>, (usize, Unbindable)> {
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
/// shader binds them in the order its kernel row states, and those are not the
/// same order. `rms_single_row`'s row is `x, w, out, params`, and
/// `norm/rms.wgsl` declares exactly that; the trace hands over `In(0), Out(0),
/// Weight(0)`.
#[derive(Debug)]
pub enum Slot<'a, B> {
    /// A range of a buffer: an operand, a weight, or a driver resource.
    Buffer(Bound<'a, B>),
    /// The slot the row reserves for this launch's scalars.
    ///
    /// The caller allocates and fills it -- see [`Params`] -- because a buffer
    /// needs a device and this module is arithmetic.
    Params,
    /// A slot the row states and nothing fills.
    ///
    /// `Source::Unbound`. `kv_append_paged` keeps several, so that the rest of
    /// its row stays at the positions a shared ring ABI put them; the module
    /// declares no global for them and nothing reads them.
    Nothing,
}

impl<B> Clone for Slot<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Slot<'_, B> {}

impl<B: Allocation> PartialEq for Slot<'_, B> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Buffer(a), Self::Buffer(b)) => a == b,
            (Self::Params, Self::Params) | (Self::Nothing, Self::Nothing) => true,
            _ => false,
        }
    }
}

/// The three runs `Launch::args` is laid out in, as indices into that slice.
///
/// `Launch::args` states inputs in operand order, then outputs, then the
/// weights the statement names -- so the weights are the ones that ARE
/// [`Arg::Weight`], and the split between inputs and outputs is by COUNT,
/// which only the row knows.
///
/// A type rather than three lines inside [`reorder`], and the reason is
/// `driver-vulkan`'s: [`scalars`] needs the identical split to answer
/// [`kernels::Source::OutWidth`], and two copies of this arithmetic would be
/// two chances to disagree about which arg is an output. A disagreement there
/// is a kernel handed the wrong pointer or the wrong row pitch, and both of
/// those compute rather than fail.
struct Runs {
    /// Indices of the inputs, in operand order.
    ins: Vec<usize>,
    /// Indices of the outputs, in operand order.
    outs: Vec<usize>,
    /// Indices of the named weights, in the order the trace states them.
    weights: Vec<usize>,
}

fn runs(sig: &kernels::KernelSig, args: &[Arg]) -> Runs {
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
    // the row names. Clamped to what the trace actually handed over, because a
    // row may state an output a given statement does not produce.
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
    Runs {
        ins: ins.to_vec(),
        outs: outs.to_vec(),
        weights,
    }
}

/// Whether an operand crosses as a device allocation rather than as a value.
///
/// The same question `kernels_wgpu::bindings` asks, asked here because
/// [`scalars`] has to know whether an operand it did not recognise owes the
/// scalar run a word or nothing at all. Read off the KIND and not off a list
/// of source names, so a row that grows a new buffer source cannot land in the
/// uniform block by omission.
fn is_buffer_kind(ty: kernels::Ty) -> bool {
    use kernels::Ty;
    matches!(
        ty,
        Ty::BufMut
            | Ty::Buf
            | Ty::I32s
            | Ty::I64s
            | Ty::U32s
            | Ty::U8s
            | Ty::F32sMut
            | Ty::F32s
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut
            | Ty::U16s
            | Ty::U16sMut
            | Ty::I8s
            | Ty::BufArray
            | Ty::BufArrayMut
            | Ty::BufArrayOut
            | Ty::BufArrayOutMut
            | Ty::U8Array
            | Ty::I32Array
    )
}

/// One operand's row width, as the plan states it.
///
/// `None` for a weight, whose extent the plan does not carry -- the resolver's
/// buffer is what knows how big a tensor is, and a scalar derived from it
/// would be a number this layer invented.
fn arg_width(arg: &Arg) -> Option<u32> {
    match arg {
        Arg::Arena { width, .. } | Arg::Named { width, .. } => Some(*width),
        Arg::Weight(_) => None,
    }
}

/// Where each of a launch's binding slots gets what it holds.
///
/// This is the step the first version of the Vulkan binder did not have, and
/// its absence was silent: binding a plan's operands positionally agrees with
/// the row for 1094 of the 3992 rectangles that tree had when it was measured
/// and disagrees for **2898**, across twelve symbols. Every one of those
/// dispatched, and every one bound a real buffer of a plausible size to the
/// wrong slot.
///
/// It is if anything quieter here. Every operand in this table is a storage
/// buffer, and a `wgpu` bind group is validated against its LAYOUT -- which
/// says "storage buffer" at every one of those entries -- so a shuffled set is
/// accepted by the strictest thing in the stack.
///
/// `rms_single_row` is the clearest case: the shader is `0=x, 1=w, 2=out`, the
/// row says `In(0), Weight(0), Out(0)`, and the trace says `In(0), Out(0),
/// Weight(0)`. Positionally the norm reads its own output as the weight and
/// writes the weight buffer.
///
/// # The unstated rows are launchable, and this is where that is carried
///
/// A row that states no operands has never told anyone an order, so the
/// trace's is the only one there is -- and `driver-metal` binds exactly that.
/// The 56 unstated rows are 292 entrypoints including `affine_qmm_t`,
/// `sdpa_paged_tiled`, `gdn_core` and `argmax_logits`, which is most of what a
/// model runs, so treating them as unlaunchable would be treating the backend
/// as unusable. The fallback below is not a convenience; it is the majority
/// path.
///
/// # Errors
///
/// The first [`Unbindable`] any slot produces, with the SLOT's index -- not
/// the plan operand's, since a refusal points at a bind-group entry.
pub fn reorder<'a, R: Resolve>(
    sig: &kernels::KernelSig,
    lowered: &Lowered,
    launch: &Launch,
    arena: Arena<'a, R::Buffer>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Vec<Slot<'a, R::Buffer>>, (usize, Unbindable)> {
    if sig.operands.is_empty() {
        return Ok(bind(lowered, launch, arena, resolver, min_offset)?
            .into_iter()
            .map(Slot::Buffer)
            .collect());
    }

    let span = launch.args.start as usize..launch.args.end as usize;
    let args = &lowered.args[span];
    let Runs { ins, outs, weights } = runs(sig, args);
    let (ins, outs, weights) = (&ins[..], &outs[..], &weights[..]);

    let layer = launch.layers.start;
    let mut slots = Vec::with_capacity(sig.operands.len());
    for (slot, operand) in sig.operands.iter().enumerate() {
        let one = |at: Option<&usize>| -> Result<Slot<'a, R::Buffer>, Unbindable> {
            let at = at.ok_or(Unbindable::NoOperand)?;
            resolve(&args[*at], launch, arena, resolver, min_offset).map(Slot::Buffer)
        };
        let held = |b: Option<&'a R::Buffer>,
                    what: FireTable|
         -> Result<Slot<'a, R::Buffer>, Unbindable> {
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
            // come out of the operand list at all -- it rides a parameter
            // buffer at whatever slot the row placed it, and `scalars` decides
            // where that is.
            kernels::Source::Unbound => Ok(Slot::Nothing),
            // EVERY remaining source, spelled out, and the reason is a defect
            // this file has already made once. `_ => Ok(Slot::Params)` sent
            // anything unrecognised to the parameter buffer -- a scalar the
            // TEXT states -- and a source that means a fact about the FIRE
            // then arrives as whatever the text happened to put there.
            // `Source::OutWidth` did exactly that at the sister site in
            // `scalars`, where a `_ => {}` produced a silent zero: a width of
            // zero is a grid of nothing, and the dispatch returned success
            // over untouched memory.
            //
            // The `LaunchRule` match in `geometry.rs` is enumerated for the
            // same reason and says so; this one now is too, so a variant added
            // to `Source` tomorrow stops this build instead of being read as a
            // parameter. That property is NOT free elsewhere:
            // `driver-vulkan/src/binding.rs:683` catches all and answers
            // `None`, so a new source lands there silently.
            kernels::Source::WeightNamed
            | kernels::Source::WeightNamed2
            | kernels::Source::WeightSuffix(_)
            | kernels::Source::Param(_)
            | kernels::Source::ParamF32(_)
            | kernels::Source::KvHeadStride
            | kernels::Source::KvSeqStride
            | kernels::Source::KvPageSize
            | kernels::Source::KvLayerView
            | kernels::Source::KvLayerField(_)
            | kernels::Source::RequestCount
            | kernels::Source::Rows
            | kernels::Source::ResultOrRegion(_)
            | kernels::Source::Aux(_)
            | kernels::Source::OutRows(_)
            | kernels::Source::InRows(_)
            | kernels::Source::OutWidth(_)
            | kernels::Source::InWidth(_)
            | kernels::Source::OutElements(_)
            | kernels::Source::InElements(_)
            | kernels::Source::InDim(_, _)
            | kernels::Source::OutDim(_, _)
            | kernels::Source::Attn(_)
            | kernels::Source::AttnWindow
            | kernels::Source::AttnPlan(_)
            | kernels::Source::AttnNonZero(_)
            | kernels::Source::Gdn(_)
            | kernels::Source::GdnSlab(_)
            | kernels::Source::CtxByLayer(_)
            | kernels::Source::Ctx(_)
            | kernels::Source::CtxNonZero(_)
            | kernels::Source::RoutesOfParam(_)
            | kernels::Source::Lit(_)
            | kernels::Source::Width(_)
            | kernels::Source::Mul(_, _)
            | kernels::Source::Sub(_, _)
            | kernels::Source::Div(_, _)
            | kernels::Source::Isqrt(_)
            | kernels::Source::Ne(_, _)
            | kernels::Source::Or(_, _)
            | kernels::Source::IfPresent(_, _, _)
            | kernels::Source::PerHeadDim
            | kernels::Source::NamedScale
            | kernels::Source::RotaryWidth
            | kernels::Source::LayerScale
            | kernels::Source::Beta => Ok(Slot::Params),
        };
        slots.push(got.map_err(|e| (slot, e))?);
    }
    Ok(slots)
}

/// Which bind-group slot a launch's parameter buffer goes in.
///
/// Both cases are ONE mechanism -- allocate a buffer, write these bytes, bind
/// it there -- which is what makes [`Params`] a single carrying variant where
/// `driver-vulkan` needs two. There, `Params::Push` goes through
/// `vkCmdPushConstants` and `Params::Block` through a descriptor: two API
/// calls, so two variants. WebGPU has no push constants at all, so the only
/// question left is WHICH slot, and a slot is data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamSlot {
    /// `@group(1) @binding(0)`: the uniform block every scalar operand of the
    /// row is a field of.
    ///
    /// The ordinary case. `kernels_wgpu::uniform_layout` gives the offsets,
    /// which are the SHADER's and not a packing -- an eight-byte stride after
    /// a four-byte count starts at 8, and the block rounds to 16.
    Uniform,
    /// `@group(0) @binding(n)`: a storage buffer the row names as a `Buf`
    /// operand whose source is a `Param`.
    ///
    /// A row says "the rest of this run is a STRUCT, and it starts here" by
    /// giving a `Param` operand a buffer kind. `rms_single_row`'s `params:
    /// Buf` is the case, and `norm/rms.wgsl` declares `RmsParams` at
    /// `@group(0) @binding(3)` to receive it -- a struct is a struct, and
    /// moving it into the uniform block would be changing the kernel's ABI
    /// from the driver.
    ///
    /// `driver-vulkan` finds this slot by SIZE, scanning the module's bindings
    /// for a block whose byte count matches the scalar run, because the shader
    /// is the only place that answer exists. Here the ROW answers it:
    /// `kernels_wgpu::bindings` states which `@group(0)` entry each operand
    /// takes, so the placement is read off the table and the reflection is a
    /// check on it rather than the source of it.
    Storage(
        /// The `@group(0)` binding number.
        u32,
    ),
}

/// Where a launch's scalars go, and the bytes to put there.
///
/// ONE carrying variant where `driver-vulkan` has two; see [`ParamSlot`] for
/// why the distinction it draws does not exist on this backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Params {
    /// A buffer of these bytes, bound at this slot.
    ///
    /// The caller allocates it, writes the bytes and binds it, because a
    /// buffer needs a device and this module is arithmetic -- which is what
    /// lets the placement be decided on a machine with no adapter.
    Block {
        /// The bytes, at the offsets the layout states rather than packed end
        /// to end.
        bytes: Vec<u8>,
        /// Where the buffer goes.
        at: ParamSlot,
    },
    /// The module declares neither, and the launch states no scalars.
    None,
}

/// Why a launch's scalars could not be placed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Misplaced {
    /// The module wants its scalars somewhere, and the two somewheres disagree
    /// about how many there are.
    ///
    /// This is the defect the whole parameter side exists to catch, and it has
    /// no symptom on this backend: WGSL requires every access to be
    /// bounds-checked, so a block short of what the shader reads returns ZEROS
    /// rather than faulting. A missing `logits_pitch` is not garbage, it is a
    /// plausible number that no layer and no assertion will object to.
    Count {
        /// How many scalars the plan states.
        stated: usize,
        /// How many fields the module's uniform block holds.
        uniform: usize,
        /// The sizes of every knowable `@group(0)` block the module declares,
        /// in bytes.
        blocks: Vec<u32>,
    },
    /// The row names a scalar this driver cannot work out.
    ///
    /// A NAMED refusal where there used to be a zero, and the change is the
    /// point. [`kernels::Source`] has a family of DERIVED scalars -- the width
    /// of a result, the row count of an operand, an element product -- that no
    /// entry of `Lowered::params` holds: the driver computes them from the
    /// launch. A `_ => 0` arm for those is the worst possible default, because
    /// zero is a plausible number: a width of zero is a grid of nothing, which
    /// dispatches nothing, reads back as the zeros the buffer was born with,
    /// and completes successfully. `.wiki/new-driver/vulkan.md` §9 and §12 are
    /// about exactly that failure.
    ///
    /// It was not hypothetical. `kernels-metal` grew a hundredth row --
    /// `add_bias`, the Qwen-2 attention biases that were being served as
    /// fluent wrong text because no kernel added them -- and its `width`
    /// operand is [`kernels::Source::OutWidth`], which this module resolved as
    /// zero. `dispatch.rs`'s stated-entrypoint sweep is what caught it, one
    /// row after the row landed.
    ///
    /// Reachable only by a row naming a source outside
    /// [`Self::Count`]'s handled set; the sweep in this module's own tests
    /// pins that no row in `kernels-wgpu`'s table does today.
    Unresolved {
        /// Which operand of the row, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
        /// The [`kernels::Source`] variant, rendered -- `Source` is not `Eq`,
        /// so it cannot be carried whole in a type that is.
        source: String,
    },
    /// The row addresses the KV cache CONTIGUOUSLY, and this driver's pool is
    /// paged.
    ///
    /// [`kernels::Source::KvHeadStride`] and [`kernels::Source::KvSeqStride`]
    /// appear on exactly the rows that walk the cache with two strides and no
    /// page table -- `kv_append`, `sdpa_vector_decode`, `sdpa_vector_decode_
    /// swa`. `attn/kv_write.wgsl` shows the pair side by side: the paged
    /// writer takes `page_size` and `n_kv_heads`, the contiguous one takes the
    /// strides and computes `h * head_stride + pos * seq_stride + d`.
    ///
    /// `resources::Shape` allocates `[page, token, head, dim]` for every fire
    /// this driver runs, so that expression is right only while the fire's
    /// pages happen to be physically consecutive from zero -- true of one
    /// freshly-allocated sequence and false of the second one. It reads real
    /// memory at every step and attends to the WRONG TOKENS: nothing faults,
    /// nothing is out of bounds, and the text stays fluent.
    ///
    /// `crates/model` reached the same conclusion from the other side and
    /// stopped emitting these rows (*"no contiguous attention over a paged
    /// pool"*). This is the same fact said where it is a fact -- the pool's
    /// layout is the DRIVER's, and a text is not the last thing that can be
    /// wrong about it.
    ///
    /// The refusal is blanket rather than conditional on the translation being
    /// the identity. A driver that served these rows *sometimes* would be
    /// correct on the first request of a fresh cache and wrong afterwards,
    /// which is the worst way to be wrong.
    Contiguous {
        /// Which operand of the row, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
    },
}

impl core::fmt::Display for Misplaced {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Count {
                stated,
                uniform,
                blocks,
            } => write!(
                f,
                "{stated} scalars stated; the uniform block holds {uniform} and \
                 the sized storage blocks are {blocks:?}"
            ),
            Self::Unresolved { at, name, source } => write!(
                f,
                "operand {at} (`{name}`) is sourced from {source}, which this \
                 driver does not know how to work out"
            ),
            Self::Contiguous { at, name } => write!(
                f,
                "operand {at} (`{name}`) is a contiguous KV stride, and this \
                 driver's pool is paged: the row would read real memory at \
                 the wrong tokens"
            ),
        }
    }
}

impl core::error::Error for Misplaced {}

/// Place one launch's scalars the way its module wants them.
///
/// For a row that states no operands, which has no layout of its own to place
/// them by. [`scalars`] is what a stated row goes through.
///
/// # Errors
///
/// [`Misplaced::Count`] when neither of the module's two shapes can hold what
/// the plan states.
pub fn params(
    lowered: &Lowered,
    launch: &Launch,
    declared: &crate::reflect::Declared,
) -> Result<Params, Misplaced> {
    let stated = &lowered.params[launch.params.start as usize..launch.params.end as usize];
    params_from(stated, declared)
}

/// The scalars one launch hands one module, in the order the ROW places them.
///
/// The statement's own run is a flat list the row indexes into, and a row may
/// index into only part of it: `neox_mb` states `ParamF32(0), ParamF32(1),
/// Param(2)` against a plan that carries four. Taking the run whole was
/// refusing it as one scalar too many.
///
/// A row also interleaves numbers the DRIVER resolves. `kv_append_paged` is
/// `Param(0), KvPageSize, Param(1)` -- the page size is a property of the pool,
/// not of the statement, and it lands between the two scalars the statement did
/// carry, not after them. Both paged decodes do the same, which is exactly the
/// one word each was short of.
///
/// # Errors
///
/// [`Misplaced`], as [`params`] -- with the run this built, so the count in the
/// refusal is the count the module would have been given.
pub fn scalars<R: Resolve>(
    sig: &kernels::KernelSig,
    lowered: &Lowered,
    launch: &Launch,
    declared: &crate::reflect::Declared,
    resolver: &R,
) -> Result<Params, Misplaced> {
    let stated = &lowered.params[launch.params.start as usize..launch.params.end as usize];
    if sig.operands.is_empty() {
        return params_from(stated, declared);
    }
    // Which `@group(0)` entry a `Param` operand of buffer kind takes, if the
    // row has one. Read off `kernels_wgpu::bindings` rather than counted here,
    // so the driver and a test asking the same question ask it of one place.
    let bindings = kernels_wgpu::bindings(sig);
    let span = launch.args.start as usize..launch.args.end as usize;
    let args = &lowered.args[span];
    let split = runs(sig, args);
    // The derived scalars, in one place. Each is computed from the LAUNCH --
    // its row span and its operands' stated widths -- because no entry of
    // `Lowered::params` holds them: a bias vector's length is the projection's
    // width, and the trace already said that when it sized the output, so a
    // row that needs the pitch names where to READ it rather than which scalar
    // slot holds it.
    //
    // `None` is a refusal below and never a zero. Which of these a row may
    // name is not this module's guess: `every_source_a_stated_row_names_is_one
    // _this_driver_can_work_out` walks the table and says.
    let derived = |source: kernels::Source| -> Option<u32> {
        let rows = launch.rows.end - launch.rows.start;
        let width =
            |run: &[usize], i: u8| run.get(usize::from(i)).and_then(|at| arg_width(&args[*at]));
        match source {
            kernels::Source::RequestCount => Some(lowered.n_requests),
            // The trailing-dims product of one value -- what a row of it is
            // worth in elements, which is exactly `Arg`'s stated width.
            kernels::Source::OutWidth(i) => width(&split.outs, i),
            kernels::Source::InWidth(i) => width(&split.ins, i),
            // Rows times that width: the ELEMENT count a flat launcher takes
            // where a row-shaped one takes both. Saturating, because a plan
            // with an absurd width should reach the refusals below rather than
            // wrap to a small number that binds.
            kernels::Source::OutElements(i) => {
                width(&split.outs, i).map(|w| w.saturating_mul(rows))
            }
            kernels::Source::InElements(i) => width(&split.ins, i).map(|w| w.saturating_mul(rows)),
            // NOT the fire's row count, and that is why they are refused
            // rather than answered with `rows`. `OutRows` is a value's LEADING
            // extent: `Rows` for a token-shaped value, a load-time constant
            // for a fixed one, and for the MoE aligned path the padded
            // block-major count, which is neither. A driver that answered the
            // fire's rows would be right for most values and silently wrong
            // for the ones the source exists to distinguish -- and the plan
            // does not carry the number, so the honest answer is that this
            // layer cannot supply it.
            //
            // No row in `kernels-wgpu`'s table names one. If one does, this is
            // the line that has to grow a real answer.
            kernels::Source::OutRows(_) | kernels::Source::InRows(_) | kernels::Source::Rows => {
                None
            }
            _ => None,
        }
    };

    let mut struct_at = None;
    let mut run: Vec<u32> = Vec::new();
    for (at, operand) in sig.operands.iter().enumerate() {
        // A field of the preceding packed struct: the driver's number, added
        // to the run the struct covers. `row_gather` is the only one --
        // `Param(0)` there is a `Buf`, which is how a row says "the rest of
        // this run is a struct in a buffer".
        if operand.ty == kernels::Ty::InPacked {
            run.push(
                derived(operand.source).ok_or_else(|| Misplaced::Unresolved {
                    at,
                    name: operand.name,
                    source: format!("{:?}", operand.source),
                })?,
            );
            continue;
        }
        // The two contiguous strides are refused rather than answered. The
        // pool is `[page, token, head, dim]`, so `Shape::number` CAN produce a
        // head stride and a sequence stride for it -- and handing them to a
        // kernel that walks the cache without a page table is what makes the
        // launch succeed against the wrong tokens. See `Misplaced::Contiguous`.
        if matches!(
            operand.source,
            kernels::Source::KvHeadStride | kernels::Source::KvSeqStride
        ) {
            return Err(Misplaced::Contiguous {
                at,
                name: operand.name,
            });
        }
        let number = match operand.source {
            kernels::Source::KvPageSize => Some(FireNumber::KvPageSize),
            _ => None,
        };
        if let Some(want) = number {
            // Zero rather than a refusal, matching `driver-metal`: a pool that
            // has not been built yet has no page size, and the caller that has
            // not built one is not dispatching against it either. Unlike the
            // derived family above, this one is a question the RESOLVER
            // answers and a caller that has not built a pool is not
            // dispatching against it.
            run.push(resolver.number(want).unwrap_or(0));
            continue;
        }
        match operand.source {
            kernels::Source::Param(i) | kernels::Source::ParamF32(i) => {
                // A buffer where a scalar could be is how a row says "the rest
                // of this run is a struct, and it starts here". So it takes
                // the whole tail, not one word: `rms_single_row`'s row is one
                // `Param(0)` against five scalars, and picking the single word
                // at index 0 refused it as four too many.
                if matches!(operand.ty, kernels::Ty::Buf | kernels::Ty::BufMut) {
                    if let Some(kernels_wgpu::Binding::Storage(n)) = bindings.get(at) {
                        struct_at.get_or_insert(*n);
                    }
                    run.extend_from_slice(stated.get(usize::from(i)..).unwrap_or(&[]));
                } else {
                    // The `unwrap_or` is unreachable across every row and text
                    // the Vulkan port measured. Kept because the alternative
                    // is an index panic on a statement that carried fewer
                    // scalars than its row indexes, which is a plan defect
                    // this crate would rather report as a short run than as a
                    // crash.
                    run.push(stated.get(usize::from(i)).copied().unwrap_or(0));
                }
            }
            // Every other source is either a BUFFER -- which `reorder` placed
            // and which contributes no scalar -- or a derived number. The
            // difference is decided by the operand's KIND rather than by a
            // list of source names, so a row that grows a new buffer source
            // does not land in the scalar run by omission and a row that grows
            // a new derived one does not land in it as a zero.
            other => {
                if !is_buffer_kind(operand.ty) {
                    run.push(derived(other).ok_or_else(|| Misplaced::Unresolved {
                        at,
                        name: operand.name,
                        source: format!("{other:?}"),
                    })?);
                }
            }
        }
    }

    // The row's own answer, when it gave one, before the module's. A stated
    // row that names a `Buf` param has told the driver exactly which entry the
    // struct is, and the size-matching below is a fallback for the rows that
    // did not.
    if let Some(at) = struct_at {
        if run.is_empty() {
            return Ok(Params::None);
        }
        return Ok(Params::Block {
            bytes: words(&run),
            at: ParamSlot::Storage(at),
        });
    }
    params_from(&run, declared)
}

/// A run of `u32` as little-endian bytes.
fn words(run: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(run.len() * 4);
    for word in run {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Place a run of scalar words by what the MODULE declares.
///
/// The reflection's answer rather than the row's, which is what an unstated row
/// needs -- and what a stated row falls back to when it names no `Buf` param.
fn params_from(stated: &[u32], declared: &crate::reflect::Declared) -> Result<Params, Misplaced> {
    // Asked in this order because the uniform block is the stronger claim: it
    // is the ABI's own home for a scalar run, and a module that declares one of
    // the right width is not also hiding a parameter struct.
    if declared.uniform_offsets.len() == stated.len() {
        if stated.is_empty() {
            return Ok(Params::None);
        }
        // Sized from the block the SHADER declares, not from four bytes per
        // scalar. `wgpu` refuses a uniform binding smaller than the struct, so
        // a run written end to end would be refused outright for a block with
        // a gap in it -- and, worse, would place every field after the gap at
        // the wrong offset if it happened to be big enough.
        let mut bytes = vec![0u8; declared.uniform_bytes as usize];
        for (word, offset) in stated.iter().zip(&declared.uniform_offsets) {
            let at = *offset as usize;
            // Four bytes, whatever the field's width. A `vec2<u32>` member --
            // which is how `kernels-wgpu` spells a 64-bit stride, since WGSL
            // has no 64-bit integer -- gets its LOW word written and its high
            // word left zero, which is the right answer for every stride and
            // extent a plan states in a `u32`. A run that carried a real
            // 64-bit value would have to arrive as two words, because
            // `Lowered::params` is a `Vec<u32>` and has nowhere to put one.
            if at + 4 <= bytes.len() {
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        return Ok(Params::Block {
            bytes,
            at: ParamSlot::Uniform,
        });
    }

    // Found by SIZE and not by position, for the reason `driver-vulkan`
    // measured: where a parameter struct sits is the kernel's own ABI, and two
    // of its six had an operand AFTER the block.
    let want = stated.len() as u32 * 4;
    if want > 0
        && let Some(at) = declared.block_bytes.iter().position(|b| *b == Some(want))
    {
        return Ok(Params::Block {
            bytes: words(stated),
            at: ParamSlot::Storage(at as u32),
        });
    }

    if stated.is_empty() && declared.uniform_offsets.is_empty() {
        return Ok(Params::None);
    }

    Err(Misplaced::Count {
        stated: stated.len(),
        uniform: declared.uniform_offsets.len(),
        blocks: declared.block_bytes.iter().flatten().copied().collect(),
    })
}

/// The refusals [`descriptors`] produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unlayoutable {
    /// The row states more slots than the module declares, and the excess is
    /// not all unbound.
    ///
    /// A row states every deployment its kernel serves, so a row longer than
    /// one module is ordinary -- but only where the tail is `Unbound`. A buffer
    /// past the end is a buffer nothing can hold.
    Overlong {
        /// How many slots the row states.
        stated: usize,
        /// How many the module declares.
        module: u32,
    },
    /// The parameter struct's slot is not where the row puts its parameters.
    ///
    /// The module says which `@group(0)` binding is the struct; the row says
    /// which operand is the `Buf` param. Both are read off separately, so
    /// their disagreement is a finding rather than a fact -- and it means the
    /// shader would read its scalars out of an operand.
    BlockElsewhere {
        /// Where the reflection puts the struct.
        module: usize,
        /// Where the row's parameters land.
        row: usize,
    },
    /// The row leaves a slot unbound that the module declares and reads.
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

/// Cut a row's slots down to the module's `@group(0)` bindings.
///
/// [`reorder`] answers "what does slot *k* of the ROW hold"; this answers "what
/// does binding *k* of the module's storage group hold", and they differ in two
/// measured ways.
///
/// A scalar occupies a row slot whether or not it occupies a bind-group entry.
/// A row whose scalars ride the uniform block takes NO `@group(0)` entry for
/// them, so everything after them moves down; a row whose `Param` operand is a
/// `Buf` does take one, and it is a binding the plan never mentions.
///
/// And a row may be longer than a module: several rows end in `Unbound`, one
/// slot past a layout that does not declare it.
///
/// # Errors
///
/// [`Unlayoutable`], which names which of the two readings disagreed.
pub fn descriptors<'a, B: Allocation>(
    slots: Vec<Slot<'a, B>>,
    placed: &Params,
    declared: &crate::reflect::Declared,
) -> Result<Vec<Slot<'a, B>>, Unlayoutable> {
    let block = match placed {
        Params::Block {
            at: ParamSlot::Storage(at),
            ..
        } => Some(*at as usize),
        Params::Block {
            at: ParamSlot::Uniform,
            ..
        }
        | Params::None => None,
    };
    let mut out = Vec::with_capacity(declared.bindings as usize);
    let mut seen = false;
    for slot in slots {
        match slot {
            // Only the FIRST scalar slot can be the struct: a row states its
            // parameters as a run, and a run has one head. The rest are more of
            // the same struct, or numbers the driver appends.
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

    // Every empty slot has to be one the module never reads. This is the check
    // that pins the two readings together: the row's `Unbound` slots and the
    // module's unread bindings have to be the same slots.
    for (at, slot) in out.iter().enumerate() {
        // `false` is the right default for a slot the reflection did not
        // describe: it is one this crate cannot claim the shader reads.
        if matches!(slot, Slot::Nothing) && declared.used.get(at).copied().unwrap_or(false) {
            return Err(Unlayoutable::Unfilled { at });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A buffer this test can name without an adapter.
    fn buffer(bytes: u64) -> Placeholder {
        Placeholder(bytes)
    }

    #[derive(Default)]
    struct Store {
        weights: BTreeMap<String, Placeholder>,
        named: BTreeMap<ValueId, Placeholder>,
    }

    impl Resolve for Store {
        type Buffer = Placeholder;
        fn weight(&self, name: &str) -> Option<&Placeholder> {
            self.weights.get(name)
        }
        fn named(&self, value: ValueId) -> Option<&Placeholder> {
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

    /// A seam value of the stated element width, at value id zero.
    fn seam(width: u32, bytes: u32) -> Arg {
        named(0, width, bytes)
    }

    /// A seam value, by id.
    fn named(value: u32, width: u32, bytes: u32) -> Arg {
        Arg::Named {
            value,
            width,
            bytes,
        }
    }

    #[test]
    fn an_arena_operands_extent_is_its_rectangle_and_not_one_row() {
        let arg = Arg::Arena {
            at: 0,
            width: 128,
            bytes: 2,
        };
        // The distinction Metal never had to draw: one row is 256 bytes, and a
        // 64-row prefill launch of the same operand covers 64 times that.
        assert_eq!(extent(&arg, &launch(1, 1)), Some(256));
        assert_eq!(extent(&arg, &launch(64, 1)), Some(64 * 256));
    }

    /// A weight has no rectangle; a SEAM VALUE has one, and states it.
    ///
    /// This asserted that neither did, and the seam half was wrong. A weight
    /// genuinely has none -- its extent is the tensor's, which the plan does
    /// not carry -- but `Arg::Named` states the same three things
    /// `Arg::Arena` does. Answering `None` there is what let a seam value
    /// bind a stand-in buffer without anyone asking whether the rectangle
    /// fits in it, and WGSL bounds-checking then reads ZEROS past the end.
    ///
    /// The four-byte case is asserted because it is the one a driver has
    /// MEASURED, and because it is the case a plausible shortcut gets wrong.
    /// `Arg::Named` carried no `bytes` until this change, so the obvious
    /// reading was "two, like every activation" -- and `driver-vulkan`
    /// records the real one as a four-row gather over a one-entry `u32`
    /// table, *"a sixteen-byte read of a four-byte buffer"*. A two-byte
    /// assumption calls that rectangle eight bytes, fits it in four, and
    /// reports nothing.
    #[test]
    fn a_weight_has_no_rectangle_and_a_seam_value_does() {
        assert_eq!(extent(&Arg::Weight("w".into()), &launch(1, 1)), None);
        assert_eq!(
            extent(&seam(8, 2), &launch(1, 1)),
            Some(16),
            "one row of eight bf16 elements"
        );
        assert_eq!(
            extent(&seam(8, 2), &launch(64, 1)),
            Some(64 * 16),
            "the rows are the launch's, as for an arena operand"
        );
        assert_eq!(
            extent(&seam(1, 4), &launch(4, 1)),
            Some(16),
            "the vulkan case: four u32 rows are sixteen bytes, not eight"
        );
    }

    /// A seam value larger than the stand-in is refused, not bound short.
    ///
    /// `Deployment::seam` is documented as "the stand-in buffer's size, which
    /// bounds the largest scalar block a fire can stage" -- and nothing
    /// enforced the bound. `Bound::whole` binds the buffer however small it
    /// is, and a rectangle past its end reads zeros: a plausible tensor, a
    /// fire that succeeds, and a model that says something.
    ///
    /// The control is the first assertion: a value that FITS still binds
    /// whole, so this is about the size and not about the path.
    #[test]
    fn a_seam_value_larger_than_the_stand_in_is_refused() {
        let stand_in = buffer(256);
        let store = Store {
            named: [(0u32, stand_in)].into_iter().collect(),
            ..Store::default()
        };
        let arena_buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &arena_buf,
            bytes: 1 << 20,
        };
        // 8 elements x 2 bytes x 16 rows = 256, exactly what the stand-in
        // holds.
        let fits = seam(8, 2);
        assert!(
            resolve(&fits, &launch(16, 1), arena, &store, 1).is_ok(),
            "a rectangle the stand-in holds must still bind whole"
        );
        // One row more is one row past it.
        let over = resolve(&fits, &launch(17, 1), arena, &store, 1)
            .expect_err("a rectangle past the stand-in is not bound short");
        assert!(
            matches!(over, Unbindable::PastSeam { .. }),
            "the refusal names the seam: {over:?}"
        );
        // And the width is READ, not assumed: the same rectangle in four-byte
        // elements is twice the bytes and does not fit. This is the assertion
        // that fails if `extent` goes back to multiplying by two.
        let wide = resolve(&seam(8, 4), &launch(16, 1), arena, &store, 1)
            .expect_err("four-byte elements are twice the bytes");
        assert!(
            matches!(wide, Unbindable::PastSeam { .. }),
            "the element width is read from the plan: {wide:?}"
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
        let b = resolve(&arg, &launch(4, 1), arena, &store, 256).expect("bindable");
        assert_eq!(b.offset(), 512);
        // Four rows of 64 elements at 2 bytes -- NOT the 512 one row would
        // give, and not the rest of the arena a whole-buffer binding would.
        assert_eq!(b.len(), 512);
        assert!(!b.is_empty());
        assert_eq!(b.buffer(), &buf);
    }

    #[test]
    fn an_operand_whose_rectangle_runs_past_the_arena_is_refused() {
        let buf = buffer(1 << 20);
        // The buffer is a megabyte; the PLAN said the arena is 1024 bytes. The
        // refusal has to come from the plan's number, or a driver holding a
        // generously sized arena would accept an operand that addresses another
        // fire's bytes.
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
            resolve(&arg, &launch(4, 1), arena, &Store::default(), 256).expect_err("runs past");
        assert_eq!(
            err,
            Unbindable::PastArena {
                at: 768,
                extent: 512,
                arena: 1024
            }
        );
    }

    /// A misaligned offset is refused BY NAME and never rounded.
    ///
    /// 260 is inside the arena and a multiple of 4, so the plan is content and
    /// Metal would be too. WebGPU's guaranteed
    /// `min_storage_buffer_offset_alignment` is 256, and `wgpu` answers a
    /// binding that does not divide it with a validation failure inside the
    /// encoder -- which is a panic naming a number, not a launch.
    #[test]
    fn an_operand_the_device_cannot_address_from_is_refused_as_such() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let arg = Arg::Arena {
            at: 260,
            width: 64,
            bytes: 2,
        };
        let err = resolve(
            &arg,
            &launch(1, 1),
            arena,
            &Store::default(),
            u64::from(crate::facts::GUARANTEED_STORAGE_ALIGNMENT),
        )
        .expect_err("misaligned");
        assert_eq!(
            err,
            Unbindable::Unaddressable(Unaddressable::Unaligned {
                offset: 260,
                alignment: 256
            }),
            "an offset the device cannot use is not the same refusal as one the \
             plan oversized"
        );
    }

    /// A zero-length range is an overrun and not an empty binding.
    #[test]
    fn a_zero_length_range_is_refused_because_webgpu_has_no_empty_binding() {
        let buf = buffer(1024);
        assert_eq!(
            Bound::within(&buf, 0, 0, 256),
            Err(Unaddressable::Overrun {
                offset: 0,
                len: 0,
                size: 1024
            })
        );
    }

    /// An offset near the top of the range does not wrap into the buffer.
    #[test]
    fn an_offset_that_would_wrap_is_refused_rather_than_summed() {
        let buf = buffer(1024);
        assert!(matches!(
            Bound::within(&buf, u64::MAX - 255, 512, 256),
            Err(Unaddressable::Overrun { .. })
        ));
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
            256,
        )
        .expect("held");
        // Whole, because the plan does not state a weight's extent and the
        // tensor's own size is the right answer.
        assert_eq!((w.offset(), w.len()), (0, 4096));

        let n = resolve(&named(7, 8, 2), &launch(1, 1), arena, &store, 256).expect("bound");
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
                256
            )
            .expect_err("not held"),
            Unbindable::UnknownWeight("layer.3.q_proj".into())
        );
        assert_eq!(
            resolve(&named(7, 8, 2), &launch(1, 1), arena, &store, 256).expect_err("not bound"),
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
        // because a zero-length Metal binding is legal and unread. WebGPU has
        // no such binding, so the honest answer is a refusal that says which
        // constant the caller still owes.
        assert_eq!(
            resolve(
                &Arg::Weight("scale.rope_theta".into()),
                &launch(1, 1),
                arena,
                &Store::default(),
                256
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

        let (i, err) =
            bind(&lowered, &launch(1, 3), arena, &store, 256).expect_err("one is absent");
        // The index is reported because a refusal that only names the weight
        // cannot say WHICH slot the dispatch was going to leave stale.
        assert_eq!(i, 2);
        assert_eq!(err, Unbindable::UnknownWeight("absent".into()));

        // The same launch minus the absent operand binds all of what remains.
        assert_eq!(
            bind(&lowered, &launch(1, 2), arena, &store, 256)
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
        // Three arena operands at distinguishable offsets, all multiples of the
        // guaranteed 256. Bind-group entries are positional, so an order this
        // crate rearranged would hand every kernel its inputs shuffled and no
        // test of one operand would notice.
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
        let bound = bind(&lowered, &launch(1, 3), arena, &store, 256).expect("bindable");
        assert_eq!(
            bound.iter().map(Bound::offset).collect::<Vec<_>>(),
            [1024, 256, 2048]
        );
    }

    /// A module declaring a uniform block at the given offsets and `@group(0)`
    /// blocks of the given sizes.
    fn declared(uniform: &[u32], blocks: &[Option<u32>]) -> crate::reflect::Declared {
        crate::reflect::Declared {
            local: [1, 1, 1],
            bindings: blocks.len() as u32,
            used: vec![true; blocks.len()],
            reads_workgroup_count: false,
            grid_axes: [true, false, false],
            uniform_offsets: uniform.to_vec(),
            uniform_bytes: uniform
                .iter()
                .map(|o| (o + 4).next_multiple_of(16))
                .max()
                .unwrap_or(0),
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
    fn scalars_the_uniform_block_holds_go_at_the_offsets_the_shader_declares() {
        let low = with_params(vec![7, 9]);
        // The gap is the point. Packed end to end these would be at 0 and 4;
        // the shader says 0 and 8 -- which is what a `vec2<u32>` after an `i32`
        // gives -- and a driver that packed them would hand the second member's
        // value to whatever sits at 4.
        let got = params(&low, &scalar_launch(2), &declared(&[0, 8], &[None])).expect("placed");
        assert_eq!(
            got,
            Params::Block {
                bytes: vec![7, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0],
                at: ParamSlot::Uniform,
            },
            "and the block is 16 bytes, not 12: the uniform address space \
             rounds a host-shareable struct to 16 and `wgpu` refuses a binding \
             that is not a multiple of it"
        );
    }

    #[test]
    fn scalars_a_storage_struct_holds_go_to_the_binding_whose_size_matches() {
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
                at: ParamSlot::Storage(1),
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
        // Four scalars; the uniform block holds two and the only sized storage
        // block is twelve bytes. Writing two and leaving the shader to read
        // four is the defect with no symptom -- WGSL's bounds checking returns
        // zeros -- so it has to be a refusal.
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
                uniform: 2,
                blocks: vec![12]
            }
        );
    }

    /// The block is sized by the SHADER's struct, not by the field count.
    ///
    /// The WebGPU direction of the check, and the opposite of Vulkan's
    /// `layout-10069`: there a shader that over-declares its push block against
    /// the pipeline's range is the validation error, so the module must not ask
    /// for more than the layout promised. Here a uniform binding must be at
    /// LEAST the struct's size, so the shell must not offer less -- and a
    /// member at offset 12 needs a 16-byte buffer, not a 4-byte one.
    #[test]
    fn a_block_with_a_hole_before_its_member_is_not_written_short() {
        let low = with_params(vec![5]);
        let got = params(&low, &scalar_launch(1), &declared(&[12], &[None])).expect("placed");
        assert_eq!(
            got,
            Params::Block {
                bytes: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                at: ParamSlot::Uniform,
            }
        );
    }

    /// A contiguous-cache row is refused; the paged one beside it is not.
    ///
    /// `attn/kv_write.wgsl` holds both writers. The paged entry point takes
    /// `page_size` and `n_kv_heads` and walks a page table; the contiguous one
    /// takes `k_head_stride`/`k_seq_stride` and computes `h * head_stride +
    /// pos * seq_stride + d`. `resources::Shape` allocates `[page, token,
    /// head, dim]` for every fire, so `Shape::number` CAN produce both strides
    /// -- and a launch given them reads real memory at the wrong tokens the
    /// moment a sequence's pages are not consecutive from zero. Nothing
    /// faults, nothing is out of bounds, and the text stays fluent.
    ///
    /// `crates/model` stopped emitting these rows (*"no contiguous attention
    /// over a paged pool"*), which guards the texts that exist. This is the
    /// driver saying it too, because the pool's layout is the driver's fact.
    ///
    /// The paged writer is the control: it is the SAME shader file and the
    /// same kind of row, so a refusal that fired on the file, on `Source::
    /// KvPageSize`, or on scalars generally would fail here.
    #[test]
    fn the_contiguous_kv_writer_is_refused_and_the_paged_one_is_not() {
        let store = Store::default();
        let placed = |name: &str| {
            let sig = kernels::sig_in(kernels_wgpu::KERNELS, name).expect("stated");
            let fields = kernels_wgpu::uniform_layout(sig);
            let offsets: Vec<u32> = fields.iter().map(|f| f.offset).collect();
            let low = with_params((0..fields.len() as u32).map(|n| n + 1).collect());
            scalars(
                sig,
                &low,
                &scalar_launch(fields.len() as u32),
                &declared(&offsets, &[None; 4]),
                &store,
            )
        };
        assert!(
            placed("kv_append_paged").is_ok(),
            "the paged writer still places its scalars"
        );
        let why = placed("kv_append").expect_err("the contiguous writer is refused");
        assert!(
            matches!(
                why,
                Misplaced::Contiguous {
                    name: "k_head_stride",
                    ..
                }
            ),
            "the refusal names the operand: {why}"
        );
        assert!(format!("{why}").contains("paged"), "and says why: {why}");
    }

    /// A row that names a `Buf` param says where its struct goes; the module
    /// is not consulted.
    ///
    /// The simplification the ABI buys. `driver-vulkan` has to find this by
    /// scanning the reflection for a block of matching size, because a shader
    /// is the only place that answer exists there. `kernels_wgpu::bindings`
    /// states it, so a row that names its params buffer is answered from the
    /// table and the reflection stays a check.
    #[test]
    fn a_row_that_names_its_params_buffer_places_it_from_the_row() {
        let sig = kernels::sig_in(kernels_wgpu::KERNELS, "rms_single_row")
            .expect("the table states the row norm");
        // `x, w, out, params` -- the struct is the fourth operand and so the
        // fourth `@group(0)` entry.
        let low = with_params(vec![1, 2, 3, 4, 5]);
        let store = Store::default();
        let got = scalars(
            sig,
            &low,
            &scalar_launch(5),
            // Deliberately a module that declares NOTHING useful: no uniform
            // block, no sized storage block. If the placement came from the
            // reflection this would refuse.
            &declared(&[], &[None, None, None, None]),
            &store,
        )
        .expect("the row places it");
        assert_eq!(
            got,
            Params::Block {
                bytes: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0],
                at: ParamSlot::Storage(3),
            }
        );
    }

    /// A row's scalar operands ride the uniform block, and take no entry.
    ///
    /// The specimen is `kv_append_paged` and used to be `kv_append`, which is
    /// now refused outright: its scalars are the contiguous strides, and this
    /// driver's pool is paged. The paged writer has three scalars of its own
    /// (`head_dim`, `page_size`, `n_kv_heads`), so it asks the same question
    /// about a row this driver can actually serve.
    #[test]
    fn a_rows_scalar_operands_are_placed_in_the_uniform_block() {
        let sig = kernels::sig_in(kernels_wgpu::KERNELS, "kv_append_paged").expect("stated");
        let fields = kernels_wgpu::uniform_layout(sig);
        assert!(
            !fields.is_empty(),
            "this row was chosen for having scalars; it now has none"
        );
        let offsets: Vec<u32> = fields.iter().map(|f| f.offset).collect();
        let low = with_params((0..fields.len() as u32).map(|n| n + 1).collect());
        let store = Store::default();
        let got = scalars(
            sig,
            &low,
            &scalar_launch(fields.len() as u32),
            &declared(&offsets, &[None; 4]),
            &store,
        )
        .expect("placed");
        let Params::Block { bytes, at } = got else {
            panic!("a row with scalars places them somewhere");
        };
        assert_eq!(at, ParamSlot::Uniform);
        assert_eq!(bytes.len() % 16, 0);
    }

    /// A uniform block takes no `@group(0)` entry, and a struct does.
    #[test]
    fn a_uniform_block_is_not_a_storage_binding_and_a_struct_is() {
        let buf = buffer(4096);
        let slots = vec![
            Slot::Buffer(Bound::whole(&buf)),
            Slot::<Placeholder>::Params,
            Slot::Buffer(Bound::whole(&buf)),
        ];
        // The uniform case: the scalar slot disappears, and what follows it
        // moves down.
        let laid = descriptors(
            slots.clone(),
            &Params::Block {
                bytes: vec![0; 16],
                at: ParamSlot::Uniform,
            },
            &declared(&[0], &[None, None]),
        )
        .expect("laid out");
        assert_eq!(laid.len(), 2);
        assert!(laid.iter().all(|s| matches!(s, Slot::Buffer(_))));

        // The storage case: the scalar slot stays and IS binding 1.
        let laid = descriptors(
            slots,
            &Params::Block {
                bytes: vec![0; 12],
                at: ParamSlot::Storage(1),
            },
            &declared(&[], &[None, Some(12), None]),
        )
        .expect("laid out");
        assert_eq!(laid.len(), 3);
        assert!(matches!(laid[1], Slot::Params));
    }

    /// The two readings of where a params struct sits have to agree.
    #[test]
    fn a_struct_the_module_puts_elsewhere_is_a_refusal_and_not_a_guess() {
        let buf = buffer(4096);
        let slots = vec![
            Slot::<Placeholder>::Params,
            Slot::Buffer(Bound::whole(&buf)),
            Slot::Buffer(Bound::whole(&buf)),
        ];
        assert_eq!(
            descriptors(
                slots,
                &Params::Block {
                    bytes: vec![0; 12],
                    at: ParamSlot::Storage(2),
                },
                &declared(&[], &[None, None, Some(12)]),
            ),
            Err(Unlayoutable::BlockElsewhere { module: 2, row: 0 })
        );
    }

    /// A row longer than its module is fine only where the tail is nothing.
    #[test]
    fn a_row_that_overruns_its_module_with_a_buffer_is_refused() {
        let buf = buffer(4096);
        // Two real buffers against a one-binding module.
        assert_eq!(
            descriptors(
                vec![
                    Slot::Buffer(Bound::whole(&buf)),
                    Slot::Buffer(Bound::whole(&buf))
                ],
                &Params::None,
                &declared(&[], &[None]),
            ),
            Err(Unlayoutable::Overlong {
                stated: 2,
                module: 1
            })
        );
        // And an `Unbound` tail past the same module is dropped, not refused.
        let laid = descriptors(
            vec![Slot::Buffer(Bound::whole(&buf)), Slot::Nothing],
            &Params::None,
            &declared(&[], &[None]),
        )
        .expect("the tail is nothing");
        assert_eq!(laid.len(), 1);
    }

    /// A slot the row leaves empty that the shader READS is a refusal.
    #[test]
    fn an_unfilled_slot_the_module_reads_is_refused() {
        let buf = buffer(4096);
        let mut d = declared(&[], &[None, None]);
        d.used = vec![true, true];
        assert_eq!(
            descriptors(
                vec![Slot::Buffer(Bound::whole(&buf)), Slot::Nothing],
                &Params::None,
                &d,
            ),
            Err(Unlayoutable::Unfilled { at: 1 })
        );
        // And the same slot is fine when the module declares it and never
        // reads it, which is exactly the hole `Declared::used` exists to name.
        d.used = vec![true, false];
        assert!(
            descriptors(
                vec![Slot::Buffer(Bound::whole(&buf)), Slot::Nothing],
                &Params::None,
                &d,
            )
            .is_ok()
        );
    }

    /// No stated row wants BOTH a params struct and a uniform block.
    ///
    /// The assumption [`scalars`] rests on, pinned so that the row which
    /// breaks it is a failure here rather than a launch whose uniform fields
    /// were silently dropped. A row that named a `Buf` param AND scalar
    /// operands would need two parameter buffers, and [`Params`] carries one;
    /// the honest fix would be a second field, not a guess about which the
    /// shader really reads.
    ///
    /// It is not a hypothetical shape -- `norm/rms.wgsl` declares a
    /// `RmsParams` STORAGE struct and a `Strided` UNIFORM block in the same
    /// file -- it is only that the row which states operands
    /// (`rms_single_row`) is not the variant that has both. The strided ones
    /// state no operands at all.
    #[test]
    fn no_stated_row_wants_a_params_struct_and_a_uniform_block_at_once() {
        let mut stated = 0;
        for sig in kernels_wgpu::KERNELS {
            if sig.operands.is_empty() {
                continue;
            }
            stated += 1;
            let struct_param = sig.operands.iter().any(|o| {
                matches!(
                    o.source,
                    kernels::Source::Param(_) | kernels::Source::ParamF32(_)
                ) && matches!(o.ty, kernels::Ty::Buf | kernels::Ty::BufMut)
            });
            assert!(
                !struct_param || kernels_wgpu::uniform_layout(sig).is_empty(),
                "`{}` names a `Buf` param AND {} uniform fields; `Params` \
                 carries one buffer and `scalars` would drop the block",
                sig.symbol,
                kernels_wgpu::uniform_layout(sig).len()
            );
        }
        assert!(stated >= 40, "only {stated} rows state operands");
    }

    /// A derived scalar is the launch's own number, not a zero and not a guess.
    ///
    /// `add_bias` is the row this is written for, and it arrived as a defect
    /// rather than as a feature: `kernels-metal` grew a hundredth row for the
    /// Qwen-2 attention biases -- which had been served as fluent WRONG TEXT,
    /// because no kernel added them -- and its `width` operand is
    /// [`kernels::Source::OutWidth`], a source this module used to fall
    /// through to zero.
    ///
    /// A zero is the worst possible answer here and not a neutral one. The
    /// width is a row pitch: at zero the shader adds the bias to element 0 of
    /// every row, or launches a grid of nothing, and either way it returns
    /// success over a buffer that still holds the projection it was supposed
    /// to bias. Nothing reports it.
    ///
    /// So the assertion is on the VALUE. `add_bias`'s output is 4096 wide, and
    /// 4096 is what the uniform block has to carry.
    #[test]
    fn a_derived_width_is_the_launchs_own_width() {
        let sig = kernels::sig_in(kernels_wgpu::KERNELS, "add_bias").expect("upstream's 100th row");
        assert!(
            sig.operands
                .iter()
                .any(|o| matches!(o.source, kernels::Source::OutWidth(_))),
            "this row was chosen for naming a derived width; it no longer does"
        );

        // `out: BufMut <- Out(0)`, `bias: Buf <- Weight(0)`, `width: I32 <-
        // OutWidth(0)`. The trace hands over the output and then the weight.
        let lowered = {
            let mut low = lowered(vec![
                Arg::Arena {
                    at: 0,
                    width: 4096,
                    bytes: 2,
                },
                Arg::Weight("bias".into()),
            ]);
            low.params = Vec::new();
            low
        };
        let mut launch = launch(8, 2);
        launch.params = 0..0;

        let d = declared(&[0], &[None, None]);
        let got = scalars(sig, &lowered, &launch, &d, &Store::default()).expect("resolves");
        let Params::Block { bytes, at } = got else {
            panic!("a row with a scalar places it somewhere");
        };
        assert_eq!(at, ParamSlot::Uniform);
        assert_eq!(
            &bytes[0..4],
            &4096u32.to_le_bytes(),
            "the width the plan gave the output, not a zero and not the fire's"
        );
    }

    /// Every source a stated row names is one this driver can work out.
    ///
    /// The assertion that would have caught `add_bias` the moment it landed,
    /// instead of at the moment somebody ran a Qwen-2. It walks all 44 stated
    /// rows and asks [`scalars`] for each, and a source that reaches the
    /// fallthrough now comes back as [`Misplaced::Unresolved`] naming itself
    /// rather than as a zero nobody sees.
    ///
    /// The plan it builds is deliberately shaped from the ROW -- one arena
    /// operand per input and output, one weight per weight -- because the
    /// derived sources are answered from the launch's args, so a plan that did
    /// not match the row would make the sweep pass for the wrong reason.
    ///
    /// # What it cannot catch, and what does
    ///
    /// It walks the sources STATED ROWS NAME. A variant added to
    /// `kernels::Source` that no row uses yet is invisible to it -- and the
    /// day a row does use it, this test would be finding out at the same
    /// moment a fire does. The guard for that is the COMPILER: `slot_of`
    /// enumerates every variant of `Source` individually, so a new one stops
    /// this build. That is deliberate and it is not free elsewhere;
    /// `driver-vulkan/src/binding.rs` catches all and answers `None`, so a new
    /// source lands there silently.
    #[test]
    fn every_source_a_stated_row_names_is_one_this_driver_can_work_out() {
        let mut checked = 0;
        let mut unresolved = Vec::new();
        for sig in kernels_wgpu::KERNELS {
            if sig.operands.is_empty() {
                continue;
            }
            let run = |pick: fn(kernels::Source) -> Option<usize>| {
                sig.operands
                    .iter()
                    .filter_map(|o| pick(o.source))
                    .max()
                    .map_or(0, |i| i + 1)
            };
            let ins = run(|s| match s {
                kernels::Source::In(i) => Some(i as usize),
                _ => None,
            });
            let outs = run(|s| match s {
                kernels::Source::Out(i) => Some(i as usize),
                _ => None,
            });
            let weights = run(|s| match s {
                kernels::Source::Weight(i) => Some(i as usize),
                _ => None,
            });
            let args: Vec<Arg> = (0..ins + outs)
                .map(|i| Arg::Arena {
                    at: i * 4096,
                    width: 4096,
                    bytes: 2,
                })
                .chain((0..weights).map(|i| Arg::Weight(format!("w{i}"))))
                .collect();
            let mut low = lowered(args);
            low.params = vec![7; 16];
            low.n_requests = 3;
            let mut l = launch(8, (ins + outs + weights) as u32);
            l.params = 0..16;

            // A module wide enough that the placement never fails for a reason
            // that is not the one under test: this asks whether every source
            // RESOLVES, and `Misplaced::Count` is a different question that
            // `every_stated_entrypoint_plans_or_is_refused_by_a_named_disagreement`
            // asks against the real modules.
            let d = declared(&[0, 4, 8, 12], &[None; 16]);
            match scalars(sig, &low, &l, &d, &Store::default()) {
                Err(Misplaced::Unresolved { at, name, source }) => {
                    unresolved.push(format!(
                        "{}: operand {at} (`{name}`) <- {source}",
                        sig.symbol
                    ));
                }
                _ => checked += 1,
            }
        }
        assert!(
            unresolved.is_empty(),
            "{} stated operands name a source this driver cannot work out:\n  {}",
            unresolved.len(),
            unresolved.join("\n  ")
        );
        assert_eq!(
            checked, 44,
            "44 rows state operands; if that moved, upstream added or filled in \
             a row and this sweep should have been the thing that noticed"
        );
    }

    /// And a source it cannot work out is REFUSED, by name.
    ///
    /// The control for the sweep above, which would pass just as well if
    /// `scalars` had gone back to answering zero. `OutRows` is the case with
    /// teeth: it is a real variant of the fleet's vocabulary, no row here names
    /// it, and it is deliberately NOT derived -- a value's leading extent is
    /// the fire's rows only for a token-shaped value, and answering `rows` for
    /// the MoE aligned path would be a plausible number in place of a padded
    /// block-major count.
    #[test]
    fn a_source_this_driver_cannot_work_out_is_refused_rather_than_zeroed() {
        let base = kernels::sig_in(kernels_wgpu::KERNELS, "add_bias").expect("stated");
        let operands: Vec<kernels::Operand> = base
            .operands
            .iter()
            .map(|o| kernels::Operand {
                source: match o.source {
                    kernels::Source::OutWidth(i) => kernels::Source::OutRows(i),
                    other => other,
                },
                ..*o
            })
            .collect();
        let sig = kernels::KernelSig {
            operands: Box::leak(operands.into_boxed_slice()),
            ..*base
        };

        let lowered = lowered(vec![
            Arg::Arena {
                at: 0,
                width: 4096,
                bytes: 2,
            },
            Arg::Weight("bias".into()),
        ]);
        let d = declared(&[0], &[None, None]);
        assert_eq!(
            scalars(&sig, &lowered, &launch(8, 2), &d, &Store::default()),
            Err(Misplaced::Unresolved {
                at: 2,
                name: "width",
                source: "OutRows(0)".into(),
            }),
            "an underivable source names itself instead of contributing a zero"
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
