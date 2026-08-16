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
use model_ir::trace::ValueId;

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
    /// Bytes between one query row's custom-mask entry and the next.
    ///
    /// The FIRE's, not the pool's, which is why the resolver answers it from
    /// what it staged rather than from [`crate::resources::Shape`]: a mask
    /// rectangle is as wide
    /// as the widest row of the fire that supplied it, and the next fire's is
    /// a different number.
    AttentionMaskStride,
    /// How many rows the fire has.
    ///
    /// The FIRE's, like the mask pitch beside it and for the same reason: a
    /// `Shape` outlives every fire it serves, so the pool answers this from
    /// what it last staged.
    ///
    /// `LaunchRule::SdpaTiled` is why it exists. That grid rounds the rows UP
    /// to whole tiles, so the threads of a partial last tile are past the end
    /// and this scalar is what tells them -- see `geometry::grid`.
    Rows,
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
///
/// # It measures the rectangle the launch WRITES, and one kind of statement
/// reads a different one
///
/// `launch.rows` is the output's row space. For every statement in this tree
/// that is also the input's, so nothing has ever forced the two apart — and
/// where they part, this is wrong and quietly so. See
/// `a_gathers_input_is_measured_by_its_output_and_that_is_the_open_defect`,
/// which pins the arithmetic and names what it costs.
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
            if let Some(extent) = extent(arg, launch)
                && extent > held.size()
            {
                return Err(Unbindable::PastSeam {
                    value: *value,
                    extent,
                    seam: held.size(),
                });
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
            | kernels::Source::AttentionMaskStride
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
    // `None` is a refusal below and never a zero. Which of these a row may name
    // used to be settled by a walk over the table; with the table empty it is
    // settled by the COMPILER, because `reorder`'s `Source` match is exhaustive
    // and a variant added to `kernels::Source` fails to build until this
    // module says what it resolves to.
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
            kernels::Source::AttentionMaskStride => Some(FireNumber::AttentionMaskStride),
            kernels::Source::Rows => Some(FireNumber::Rows),
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
mod gathers {
    use super::extent;
    use model_compiler::lower::{Arg, Launch};

    /// A launch over `rows` rows, with nothing else stated.
    fn over(rows: u32) -> Launch {
        Launch {
            kernel: 0,
            rows: 0..rows,
            layers: 0..1,
            op: 0,
            args: 0..0,
            params: 0..0,
            peel: None,
            cond: Launch::NO_COND,
        }
    }

    /// [`extent`] measures the OUTPUT's rectangle, and a gather reads another.
    ///
    /// This is the open half of the lm-head fix, pinned so the diagnosis in
    /// `turns.rs` cannot rot into prose nobody checks — and so that whoever
    /// closes it has an oracle that fails when they do.
    ///
    /// `row_gather` compacts the rows a fire samples: its OUTPUT is one row
    /// per sampled row and its INPUT is the whole stream. Both are the same
    /// `Arg::Arena`-shaped operand, and this measures both by `launch.rows`.
    /// So the moment the launch is narrowed to what it writes — which is the
    /// point of narrowing it — the input binds one row and the shader reads
    /// row 31 of the stream. WGSL clamps that to ZERO rather than faulting, so
    /// the gathered row is zeros, the lm head projects zeros, every logit is
    /// equal, and argmax returns the last index.
    ///
    /// Measured on a 32-row prefill of Qwen3-0.6B sampling row 31: the input
    /// wanted 32 * 1024 * 2 bytes and was bound 1 * 1024 * 2.
    ///
    /// `KernelSig::whole` is NOT the signal, which is the obvious first guess:
    /// it means the kernel refuses a row SPLIT inside a peel's regions, which
    /// is a different question. Nothing in `Arg::Arena` carries a row count of
    /// its own, so the driver cannot state this locally — the lowering has to.
    #[test]
    fn a_gathers_input_is_measured_by_its_output_and_that_is_the_open_defect() {
        let row = Arg::Arena {
            at: 65536,
            width: 1024,
            bytes: 2,
        };

        // What the gather's OUTPUT wants when the launch is narrowed: one row.
        assert_eq!(extent(&row, &over(1)), Some(1024 * 2));
        // And what its INPUT needs over the same launch: the whole stream.
        // The two are the same call, so today they cannot differ.
        assert_eq!(extent(&row, &over(32)), Some(32 * 1024 * 2));
        assert_ne!(
            extent(&row, &over(1)),
            extent(&row, &over(32)),
            "if these ever agree this test is measuring nothing"
        );

        // The consequence, as arithmetic: bound one row, the shader's read of
        // row 31 is 63488 bytes past the end of its binding.
        let bound = extent(&row, &over(1)).expect("an arena operand measures");
        let wants = u64::from(31u32) * 1024 * 2;
        assert!(
            wants >= bound,
            "row 31 is inside a one-row binding, so the defect this pins is \
             gone and the test should be rewritten as the fix's oracle"
        );
    }
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

    /// The rows the tests below use, stated here rather than looked up.
    ///
    /// `kernels_wgpu::KERNELS` is EMPTY: all hundred of this backend's kernels
    /// and all 481 of its entrypoints are reached through a ROUTINE and an
    /// ARM, and nothing in that crate describes a launch positionally any
    /// more. A test that asks the table for a row is a test that goes quiet
    /// the moment the refactor it guards succeeds — and these are not about
    /// the table. They are about [`scalars`] and [`reorder`], which take a
    /// `KernelSig` and are still live non-test code: `dispatch::plan_by_row`
    /// calls both, and it is what serves any symbol `routine::armed` declines.
    ///
    /// So each row is written out at the shape it really had, taken from the
    /// family's own source before it crossed, with the `kernel!` and
    /// `operands!` spellings the table used. `driver-vulkan/src/binding.rs`
    /// states the same two KV writers for the same reason and by the same
    /// means. Scraping a live row only ever stood in for stating one.
    ///
    /// `attn/kv_write.wgsl`'s contiguous writer.
    fn kv_append_sig() -> kernels::KernelSig {
        kernels::kernel!(kv_append "kv_append",
            file = Some("attn/kv_write.wgsl"), launch = kernels::LaunchRule::PerHead,
            operands = kernels::operands![
                k_new: Buf <- kernels::Source::In(0),
                v_new: Buf <- kernels::Source::In(1),
                k_cache: BufMut <- kernels::Source::KvKeys,
                v_cache: BufMut <- kernels::Source::KvValues,
                pos: I32s <- kernels::Source::Positions,
                head_dim: I32 <- kernels::Source::Param(0),
                // The POOL's, not the statement's, and the two operands
                // `the_contiguous_kv_writer_is_refused_and_the_paged_one_is_not`
                // is about.
                k_head_stride: Usize <- kernels::Source::KvHeadStride,
                k_seq_stride: Usize <- kernels::Source::KvSeqStride,
            ],
            head_param = Some(0),
            axes = &[kernels_wgpu::axes::BF16])
    }

    /// `attn/kv_write.wgsl`'s paged writer.
    ///
    /// Sparse indices, and the gaps are stated: buffers 4, 6-9, 11 and 15
    /// belong to a shared ring ABI this kernel does not read. A row is
    /// POSITIONAL, so it lists them as `Unbound` rather than closing the gap
    /// and shifting everything after.
    fn kv_append_paged_sig() -> kernels::KernelSig {
        kernels::kernel!(kv_append_paged "kv_append_paged",
            file = Some("attn/kv_write.wgsl"), launch = kernels::LaunchRule::PerHead,
            operands = kernels::operands![
                k_new: Buf <- kernels::Source::In(0),
                v_new: Buf <- kernels::Source::In(1),
                k_pages: BufMut <- kernels::Source::KvKeys,
                v_pages: BufMut <- kernels::Source::KvValues,
                ring_4: Buf,
                head_dim: I32 <- kernels::Source::Param(0),
                ring_6: Buf,
                ring_7: Buf,
                ring_8: Buf,
                ring_9: Buf,
                page_size: I32 <- kernels::Source::KvPageSize,
                ring_11: Buf,
                n_kv_heads: I32 <- kernels::Source::Param(1),
                w_page: U32s <- kernels::Source::KvWritePage,
                w_off: U32s <- kernels::Source::KvWriteOffset,
                ring_15: Buf,
            ],
            head_param = Some(0),
            heads_param = Some(1),
            axes = &[kernels_wgpu::axes::BF16])
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
    ///
    /// Both rows are STATED — [`kv_append_sig`] and [`kv_append_paged_sig`] —
    /// because `attn` has crossed and the table cannot be asked for either any
    /// more. The refusal itself has not moved: `scalars` still returns
    /// `Misplaced::Contiguous` for the two stride sources, and
    /// `lowering::arm::contiguous_pool` is the routine plane's narrower
    /// restatement of the same rule, refusing `arm::kv_append`,
    /// `arm::sdpa_vector_decode` and its windowed sibling with
    /// `Refusal::Absent` whenever the fire's pool answers a page size.
    #[test]
    fn the_contiguous_kv_writer_is_refused_and_the_paged_one_is_not() {
        let store = Store::default();
        let placed = |sig: &kernels::KernelSig| {
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
            placed(&kv_append_paged_sig()).is_ok(),
            "the paged writer still places its scalars"
        );
        let why = placed(&kv_append_sig()).expect_err("the contiguous writer is refused");
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

    /// `moe/route.wgsl`'s router, as `kernels-wgpu`'s `moe` family stated it.
    ///
    /// Five buffers, of which the FOURTH is the params struct — which is what
    /// makes `Storage(3)` the ROW's answer and not a count the test did.
    fn router_topk_sig() -> kernels::KernelSig {
        kernels::kernel!(router_topk "router_topk", file = Some("moe/route.wgsl"),
            launch = kernels::LaunchRule::RouterLane,
            operands = kernels::operands![
                logits: Buf <- kernels::Source::In(0),
                expert_ids: BufMut <- kernels::Source::Out(0),
                expert_weights: BufMut <- kernels::Source::Out(1),
                params: Buf <- kernels::Source::Param(0),
                // The unscaled variant reads it and does nothing with it; the
                // slot is positional so it is listed, and `router_topk_scaled`
                // is the symbol that means it.
                per_expert_scale: Buf,
            ],
            axes = &[kernels_wgpu::axes::BF16])
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
        // `rms_single_row` stood here, then `row_gather`, then
        // `combine_sorted`, then `router_topk` — each until its family
        // retired, and `moe` was the last of the four. `router_topk`'s arm HAS
        // now landed, so the comment that stood here is spent: there is no
        // stated row left in the table to pick, and its sentence about the
        // claim moving to the routine plane's `bind` is only half true.
        //
        // `routine::bind` does stage a `@group(0)` storage block from the
        // body's ask — `Placed::Params` takes a position and no entry, and the
        // statement's own run goes in first — but it learns the POSITION from
        // where the body put the handle, not from an operand list. The claim
        // that `kernels_wgpu::bindings` answers this and the reflection is
        // only a check belongs to `scalars`, which `dispatch::plan_by_row`
        // still calls for every symbol `routine::armed` declines. So the row
        // is STATED at its real shape rather than the claim being let go.
        let sig = router_topk_sig();
        let low = with_params(vec![1, 2, 3, 4, 5]);
        let store = Store::default();
        let got = scalars(
            &sig,
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
    ///
    /// STATED at that row's real shape — [`kv_append_paged_sig`] — since
    /// `attn` has crossed. The placement is still `scalars`'s to make, and the
    /// two facts asserted are the ones a uniform block exists for: the scalars
    /// land at `ParamSlot::Uniform` rather than at a `@group(0)` entry, and
    /// the block is sized to a multiple of 16, which is what `wgpu` refuses a
    /// uniform binding for not being.
    #[test]
    fn a_rows_scalar_operands_are_placed_in_the_uniform_block() {
        let sig = kv_append_paged_sig();
        let fields = kernels_wgpu::uniform_layout(&sig);
        assert!(
            !fields.is_empty(),
            "this row was chosen for having scalars; it now has none"
        );
        let offsets: Vec<u32> = fields.iter().map(|f| f.offset).collect();
        let low = with_params((0..fields.len() as u32).map(|n| n + 1).collect());
        let store = Store::default();
        let got = scalars(
            &sig,
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

    // RETIRED: THE TABLE IS EMPTY, so the walk has no row to read.
    //
    // It asserted the assumption `scalars` rests on: that NO stated row names
    // both a `Buf`-kinded `Param`/`ParamF32` operand — a params STRUCT, which
    // takes a `@group(0)` entry and swallows the statement's whole scalar tail
    // — and scalar operands of its own, which ride the `@group(1)` uniform
    // block. `Params` carries ONE buffer, so a row with both would have had
    // its uniform fields silently dropped on the way to the device; the honest
    // fix would have been a second field, not a guess about which of the two
    // the shader really reads. It was pinned as a walk rather than argued,
    // because the shape is not hypothetical: `norm/rms.wgsl` declares a
    // `RmsParams` STORAGE struct and a `Strided` UNIFORM block in the same
    // file, and it was only that the variant stating operands
    // (`rms_single_row`) was not the variant with both.
    //
    // It BECAME BLIND, not true. Nothing established that no row wants both;
    // there are no rows. The walk's own floor is what made that audible rather
    // than letting it pass over an empty iterator — `assert!(stated >= 10)`,
    // written as "a FLOOR, and it falls as Stage 3 empties the table; it
    // exists so that a walk reading nothing cannot pass" — so this retires by
    // failing, which is the outcome the floor was for.
    //
    // The routine plane does not inherit the assumption, which is why it is
    // not restated as a synthetic row. `lowering::routine::bind` takes the
    // storage-block branch when the body asks for one and appends the body's
    // own packed scalars to the statement's run INSIDE that block, so a body
    // that has both does not lose either — the drop this walk was watching for
    // is unrepresentable there rather than unobserved. What is left refusable
    // is refused by name: `Unplanned::Blocks` for a body binding two parameter
    // blocks in one dispatch (`routine::tests::
    // two_parameter_blocks_in_one_dispatch_are_refused_by_name`) and
    // `Unplanned::Scalars` for a run wider than the module's uniform block
    // (`routine::tests::scalars_wider_than_the_modules_block_are_refused_by_
    // name`).

    /// The `add_bias` row, written out rather than scraped.
    ///
    /// `norm/add_bias.wgsl` is Qwen-2's attention biases, ported operand for
    /// operand from `kernels-metal`'s row, which is `kernels-cuda`'s and
    /// `kernels-vulkan`'s: IN PLACE over the value it biases, the bias off the
    /// statement's named weight, and the row width DERIVED rather than stated
    /// — an `AddBias` carries no scalars, because a bias vector's length is
    /// the projection's width and the trace knew it when it sized the output.
    ///
    /// `norm` has crossed and the row is gone, but what the two tests below
    /// are about is `binding`'s arm for `Source::OutWidth`, which is still
    /// reachable: a row anywhere in the fleet may name it, and the catch-all
    /// this file used to have answered zero. `driver-vulkan/src/binding.rs`
    /// keeps the same `add_bias_sig` for the same pair of tests.
    fn add_bias_sig() -> kernels::KernelSig {
        kernels::kernel!(add_bias "add_bias", file = Some("norm/add_bias.wgsl"),
            launch = kernels::LaunchRule::RouteRows,
            in_place = &[(0, 0)],
            operands = kernels::operands![
                out: BufMut <- kernels::Source::Out(0),
                bias: Buf <- kernels::Source::Weight(0),
                width: I32 <- kernels::Source::OutWidth(0),
            ],
            axes = &[kernels_wgpu::axes::BF16])
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
        // `add_bias`'s row stood here, and it was the last in the table to
        // name a derived width — `norm` retired and `Source::OutWidth` went
        // with it. The DERIVATION is still `scalars`'s to make, and a table
        // with no row exercising it is exactly how a source falls back to
        // zero unnoticed, so the row is STATED rather than the test deleted.
        // See [`add_bias_sig`].
        let sig = add_bias_sig();

        // The trace hands over the output and then the weight.
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
        let got = scalars(&sig, &lowered, &launch, &d, &Store::default()).expect("resolves");
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

    // RETIRED: THE TABLE IS EMPTY, so the sweep has no stated row to ask
    // about.
    //
    // It walked every row of `kernels_wgpu::KERNELS` that stated operands,
    // built a plan shaped from the ROW — one arena operand per input and
    // output, one weight per weight, because the derived sources are answered
    // from the launch's args and a plan that did not match would have made the
    // sweep pass for the wrong reason — and asked `scalars` for each. The
    // claim was that NO source a row names reaches `scalars`'s fallthrough:
    // every one either resolves or comes back as `Misplaced::Unresolved`
    // naming itself. It was written after the fact it would have caught.
    // `kernels-metal` grew a hundredth row for the Qwen-2 attention biases,
    // `add_bias`, whose `width` operand is `Source::OutWidth`; this module
    // answered that source with a ZERO, which is a row pitch of nothing — the
    // shader biases element 0 of every row, or launches a grid of nothing, and
    // returns success over the unbiased projection either way.
    //
    // It BECAME BLIND. Zero rows named zero sources, so nothing was
    // established about any of them; the walk stopped looking rather than
    // started agreeing. Its `assert_eq!(checked, 12)` is what said so out loud
    // instead of letting an empty iterator pass — the same floor
    // `no_stated_row_wants_a_params_struct_and_a_uniform_block_at_once` had,
    // and it worked the same way.
    //
    // Its own doc named what it could NOT catch, and that half survives
    // unchanged: a variant added to `kernels::Source` that no row uses is
    // invisible to any walk over rows, and the guard for it is the COMPILER —
    // `reorder` enumerates every variant of `Source` individually rather than
    // catching all, so a new one stops this build. That is not free elsewhere;
    // `driver-vulkan/src/binding.rs` catches all and answers `None`, so a new
    // source lands there silently.
    //
    // The half that was about ROWS has no counterpart on the routine plane
    // and does not need one, because the plane has no source vocabulary to
    // fall through. An arm computes each number itself and hands it over as an
    // `ArgValue`: `lowering::arm::Handles` mints a handle per ask and
    // `arm::Asked::{Operand, Params, Unbound, Kv, Table}` is the closed set of
    // what a body can want, so there is no lookup that can return a plausible
    // zero. A value an arm cannot produce is a `Refusal` (`arm::tests::
    // a_statement_the_arm_cannot_fill_is_refused`), which `lowering::routine::
    // plan` turns into a named `Unplanned::{Operand, NoCache, Absent}`. The
    // census that succeeded this one is `arm::tests::
    // every_entrypoint_is_claimed_by_the_stem_that_owns_it`, which walks all
    // 481 entrypoints and refuses one no stem claims — an orphan there cannot
    // be planned by ANY path, which is a stronger floor than this had.
    //
    // The control below outlives it. `a_source_this_driver_cannot_work_out_is
    // _refused_rather_than_zeroed` asserts the refusal itself against a
    // synthesized row, so `scalars`'s `Misplaced::Unresolved` is still pinned
    // by name; what is gone is the claim that no REAL row needs it.

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
        // Stated for the reason `a_derived_width_is_the_launchs_own_width`
        // above says: `add_bias` was the last row naming `OutWidth` and its
        // family has retired. `OutRows` is a source no row names either, which
        // is what makes it the right one to ask about — the question is what
        // `scalars` does with a source it cannot work out.
        //
        // Built by SWAPPING the one source on `add_bias`'s real row, so the
        // only thing that differs between this and
        // `a_derived_width_is_the_launchs_own_width`'s subject is the source
        // under test. `driver-vulkan/src/binding.rs` states its control the
        // same way and for the same reason.
        let base = add_bias_sig();
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
            ..base
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

    /// `rope/neox.wgsl`'s decode rotation, as `kernels-wgpu`'s `rope` family
    /// stated it.
    ///
    /// An IN-PLACE result, then a fire table, then three scalars. `reorder`
    /// never reaches the scalars — it stops at the first slot it cannot bind —
    /// but they are stated because a row is POSITIONAL and a truncated one is
    /// a different row.
    fn neox_decode_sig() -> kernels::KernelSig {
        kernels::kernel!(neox_decode "neox_decode", file = Some("rope/neox.wgsl"),
            launch = kernels::LaunchRule::Rope,
            operands = kernels::operands![
                x: BufMut <- kernels::Source::Out(0),
                position: I32s <- kernels::Source::Positions,
                scale: F32 <- kernels::Source::ParamF32(0),
                base: F32 <- kernels::Source::ParamF32(1),
                head_dim: I32 <- kernels::Source::Param(2),
            ],
            grid_param = Some(3),
            head_param = Some(2),
            axes = &[kernels_wgpu::axes::BF16])
    }

    /// `attn/sdpa_paged.wgsl`'s vector decode, as `kernels-wgpu`'s `attn`
    /// family stated it.
    ///
    /// Eleven storage buffers, which is why this was the row
    /// `over_downlevel_storage_limit` used to name. What matters here is only
    /// that its SECOND operand is the cache's KEY plane: a resolver holding no
    /// cache is refused one slot in, before the values are ever asked for, and
    /// that ordering is what makes `values: false` in the refusal mean
    /// something.
    fn sdpa_paged_decode_sig() -> kernels::KernelSig {
        kernels::kernel!(sdpa_paged_decode "sdpa_paged_decode",
        file = Some("attn/sdpa_paged.wgsl"),
        launch = kernels::LaunchRule::SdpaVector,
        operands = kernels::operands![
            queries: Buf <- kernels::Source::In(0),
            k_pages: Buf <- kernels::Source::KvKeys,
            v_pages: Buf <- kernels::Source::KvValues,
            out: BufMut <- kernels::Source::Out(0),
            gqa_factor: I32 <- kernels::Source::Param(0),
            position_ids: I32s <- kernels::Source::Positions,
            req_of_token: I32s <- kernels::Source::RequestOfToken,
            kv_page_indices: U32s <- kernels::Source::KvPageIndices,
            kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
            page_size: I32 <- kernels::Source::KvPageSize,
            n_kv_heads: I32 <- kernels::Source::Param(1),
            scale: F32 <- kernels::Source::ParamF32(2),
            attention_mask: U8s <- kernels::Source::AttentionMask,
            attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
            attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
            window: I32 <- kernels::Source::Param(4),
            sinks: Buf,
        ],
        lacks = &[kernels::Cap::Scores, kernels::Cap::PageMaskSink],
        axes = &[kernels::Axis {
            what: "head dim and page shape",
            points: &["_bfloat16_d_64", "_bfloat16_d_128", "_bfloat16_d_256",
                      "_bfloat16_d_512", "_bfloat16_d_64_p32",
                      "_bfloat16_d_128_p32", "_bfloat16_d_64_p32_sg8"],
        }])
    }

    /// Three refusals `reorder` builds that no test named.
    ///
    /// From the census in `tests/citations.rs`: sixty of ninety-seven refusal
    /// variants are asserted by name and thirty-seven are not, of which the
    /// `Unbindable` three are in the group it calls "reachable and untested".
    /// They are also the cheapest to reach — `reorder` is in the portable half,
    /// so none of this needs an adapter.
    ///
    /// Each asks a DIFFERENT question, which is why one test per refusal
    /// rather than one that takes what it gets:
    ///
    /// * `NoOperand` is the statement being shorter than the row;
    /// * `NoKvCache` is a resolver with no cache, and it must name the LAYER
    ///   and which half, because a fire binds one layer's keys and values as
    ///   two separate slots;
    /// * `NoDriverResource` is a fire table the driver never staged, and it
    ///   must name WHICH — `Positions` and `TokenIds` are staged by the same
    ///   call and a refusal that said only "a table" would not say which of
    ///   the six went missing.
    ///
    /// The test `Store` leaves `kv`, `number` and `table` defaulted, which is
    /// exactly the shape of a resolver serving a text that needs none — so the
    /// last two need no fixture at all beyond choosing a row that reads one.
    #[test]
    fn the_three_ways_a_row_outruns_what_the_driver_can_hand_it() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let store = Store::default();
        let arg = || Arg::Arena {
            at: 0,
            width: 8,
            bytes: 2,
        };

        // `neox_decode`'s row stood here — `x: Out(0)` then
        // `position: Positions` — until `rope` retired, and the first two
        // refusals were then SYNTHESIZED from that pair by spreading `..*base`
        // over whatever row still stated operands. THE TABLE IS EMPTY now, so
        // there is no base to spread either; both rows are STATED whole
        // instead. See [`neox_decode_sig`] and [`sdpa_paged_decode_sig`]. The
        // claim is about `reorder`'s refusals and not about which family shows
        // them, but the shapes are the real ones, so the ORDER the refusals
        // arrive in is the real one too.
        let rope = neox_decode_sig();
        let (slot, why) = reorder(
            &rope,
            &lowered(Vec::new()),
            &launch(1, 0),
            arena,
            &store,
            256,
        )
        .expect_err("a row cannot bind an operand the statement never stated");
        assert_eq!(slot, 0, "the refusal names the slot it stopped at");
        assert_eq!(why, Unbindable::NoOperand);

        // The same row's SECOND operand is `Source::Positions`, a fire table.
        // One arg satisfies `Out(0)`; nothing satisfies the table, because
        // this resolver stages none.
        let (slot, why) = reorder(
            &rope,
            &lowered(vec![arg()]),
            &launch(1, 1),
            arena,
            &store,
            256,
        )
        .expect_err("a table the driver never staged is not bindable");
        assert_eq!(slot, 1);
        assert_eq!(why, Unbindable::NoDriverResource(FireTable::Positions));
        assert!(
            why.to_string().contains("Positions"),
            "the refusal must name which table: {why}"
        );

        // `sdpa_paged_decode` reads the cache. Two args satisfy `In(0)` and
        // `Out(0)`; the keys have nowhere to come from.
        let attn = sdpa_paged_decode_sig();
        let (_, why) = reorder(
            &attn,
            &lowered(vec![arg(), arg()]),
            &launch(1, 2),
            arena,
            &store,
            256,
        )
        .expect_err("a paged attention cannot bind a cache the resolver has not got");
        assert_eq!(
            why,
            Unbindable::NoKvCache {
                layer: 0,
                values: false
            },
            "the KEYS are asked for before the values, and the refusal says \
             which half as well as which layer"
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
