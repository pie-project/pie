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
///     // `kv`, `slab`, `number` and `table` are defaulted; a resolver serving a text
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
    /// The refusal that produces is a named one, which
    /// names what was missing.
    fn kv(&self, _layer: u16, _values: bool) -> Option<&Self::Buffer> {
        None
    }

    /// A per-layer RECURRENT slab -- a gated DeltaNet's convolution or
    /// recurrent carry, by the name the kernel knows it as.
    ///
    /// STATE, like [`Resolve::kv`], and for the same reason: no traced value
    /// stands for it, so no plan mentions it and no arena holds it.
    ///
    /// # Why the default is `None` and not a placeholder
    ///
    /// `driver-metal` states this in as many words and the rule is worth
    /// copying exactly: a missing SCALE is a legitimate absence, so binding
    /// nothing is the honest answer there. A recurrent carry is not. **A scan
    /// handed a null carry reads zero, writes nothing back, and returns a
    /// fluent result that is wrong in a way no output check catches.**
    ///
    /// Neither backend allocates one today, so every `ssm` arm declines here,
    /// and that is what keeps the family's dispatch honestly DARK instead of
    /// quietly broken. `tests/hybrid_probe.rs` is where the decline is read.
    fn slab(&self, _layer: u16, _which: &'static str) -> Option<&Self::Buffer> {
        None
    }

    /// One of the fire's own numbers.
    ///
    /// A pool's shape, not a statement's scalar. A text that stated its page
    /// size would be right for one deployment and silently wrong for the next,
    /// so the kernel names the number and the driver answers it.
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
    // RETIRED: `FireNumber::Rows` -- nothing asks the fire for it.
    //
    // It was the row count a fire last staged, answered by the pool because a
    // `Shape` outlives every fire it serves. `LaunchRule::SdpaTiled` is why it
    // existed: that grid rounds the rows UP to whole tiles, so the threads of a
    // partial last tile run past the end and this scalar was what told them.
    //
    // A routine does not need the fire to tell it: the tiled attention bodies
    // take their row count as an ARGUMENT, and their arm reads it off the
    // launch rectangle the same way the grid does, so the padded grid and the
    // bound are derived from one number instead of two that could disagree.
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
    /// Which recurrent SLOT each request's state lives in.
    ///
    /// The gated DeltaNet's `*_slotted` kernels read it to find their carry,
    /// the way a paged attention reads a page table. `driver-metal` states it
    /// as `FireTable::RecurrentSlots` and its `gdn_prep_slotted` inserts the
    /// handle at position 13.
    ///
    /// **This driver holds no such table**, so a body asking for one is
    /// refused by name -- which is the same posture as
    /// [`Resolve::slab`] and for the same reason: a slot index the driver
    /// invents points a scan at another request's carry.
    RecurrentSlots,
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
    /// Source `None`. `kv_append_paged` keeps several, so that the rest of
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

// RETIRED: THE ROW PATH, which had no production caller left once
// `dispatch::plan_one` stopped forking.
//
// `plan_one` plans through the arm or answers `Undispatchable::Unknown`, and
// `plan_by_row` -- the only thing in this crate that ever handed a
// `kernels::KernelSig` to this module -- went with the fork. What stood here
// was `Runs`, `runs`, `reorder`, `scalars`, `is_buffer_kind` and `arg_width`,
// held up by nothing but their own unit tests. That is the dual maintenance
// the crossing exists to end: row-shaped code cannot go WRONG once the table
// is empty, only SILENT, and silent reads exactly like passing.
//
// WHAT THEY DID, since a reader who never saw them cannot audit what is gone.
//
// `runs` split a launch's `Arg` slice into three index runs -- inputs,
// outputs, weights. `Launch::args` states the weights as the ones that ARE
// `Arg::Weight`, and splits inputs from outputs by COUNT rather than by kind;
// only the row knew that count, as one past the highest `Source::Out(i)` it
// named, clamped to what the trace actually handed over. `Runs` was a type
// rather than three lines inside `reorder` because `scalars` needed the
// identical split to answer `Source::OutWidth`, and two copies of the
// arithmetic would have been two chances to disagree about which arg is an
// output -- a kernel handed the wrong pointer or the wrong row pitch, both of
// which compute rather than fail.
//
// `reorder` bound a rectangle in the ROW's operand order rather than the
// trace's. A plan states its operands as inputs, then outputs, then weights;
// a shader binds them in the order its row states, and those are not the same
// order -- `rms_single_row` was `x, w, out, params` against a trace's `In(0),
// Out(0), Weight(0)`, so positionally the norm read its own output as the
// weight and wrote the weight buffer. So `reorder` walked the row and filled
// slot k from whatever that row's k-th `Source` named: the trace's n-th
// input, the layer's KV keys or values, one of the fire tables,
// `Slot::Nothing` for an `Unbound` gap the module declares and nothing
// fills, and `Slot::Params` for a scalar, which takes a row slot whether or
// not it takes a bind-group entry. A row that stated NO operands fell back to
// the plan's own order, which was the majority path while it existed.
//
// `scalars` resolved each `kernels::Source` into a NUMBER and decided where
// the number went. The statement's own run answered `Const { v: i }` and
// `ParamF32(i)`; the resolver answered the pool's page size, the mask stride
// and the fire's rows; the LAUNCH answered the derived family -- `OutWidth`,
// `InWidth`, `OutElements`, `InElements` -- from its row span and its
// operands' stated widths, which is what `arg_width` read. The run then went
// either into the `@group(1)` uniform block at the offsets the shader
// declares, or, where the row gave a `Param` operand a buffer kind and so
// said "the rest of this run is a STRUCT", into the `@group(0)` entry
// `kernels_wgpu::bindings` named for it. `is_buffer_kind` was how an operand
// that owed the run nothing was told from one that owed it a word, read off
// the operand's KIND rather than off a list of source names.
//
// WHERE THE SAME WORK HAPPENS NOW. `lowering::hold`'s `Handles` mints a handle
// per ask -- `input`, `output`, `weight`, `table`, `kv`, `unbound`,
// `params_block` -- and records each as an `Asked`, whose five variants
// (`Operand`, `Params`, `Table`, `Unbound`, `Kv`) are the closed set that
// replaced the row's open source vocabulary; the numbers an arm cannot read
// off the statement with `stated`, `param` and `param_f32` it computes itself
// and passes as `ArgValue`s. `lowering::routine::state` runs the body and
// splits what it dispatched by VARIANT, buffers from scalars, so neither can
// renumber the other. `lowering::routine::bind` is then what `reorder` and
// `scalars` were together: the buffer list comes from the body's handles
// against what the arm resolved, the scalar run is packed from its
// `ArgValue`s at WGSL's alignment rather than concatenated, and the parameter
// block lands at `ParamSlot::Storage` where the body asked for one and
// `ParamSlot::Uniform` where the module declares one. Per-symbol cover is
// `arm::tests::handles_are_minted_in_the_order_the_body_asks` and
// `arm::tests::a_statement_the_arm_cannot_fill_is_refused`; the placement's
// is `routine::tests::a_usize_scalar_is_eight_aligned_in_the_block`,
// `two_parameter_blocks_in_one_dispatch_are_refused_by_name` and
// `scalars_wider_than_the_modules_block_are_refused_by_name`; the corpus-wide
// cover is `tests/arena.rs`'s
// `every_launchs_scalars_land_where_its_module_reads_them`, re-anchored off
// `scalars` onto `routine::state`, and
// `every_rectangle_of_every_real_plan_becomes_a_dispatch_or_a_named_refusal`.
// The contiguous-cache refusal `scalars` made by name is now
// `arm::contiguous_pool`, which narrows it rather than inheriting it: a fire
// whose pool states a page size is refused, a fire whose pool has none is
// served.
//
// WHAT IS LOST, AND NOT MERELY RELOCATED.
//
// * THE COMPILER NO LONGER GUARDS `kernels::Source`. `reorder`'s match named
//   all SIXTY-FOUR variants individually, with a comment saying why it must
//   never catch all, so a variant added to the shared vocabulary failed THIS
//   build until this module said whether it was a buffer or a number. Nothing
//   replaces that. `Asked` being closed is a weaker and different claim: it
//   says an arm cannot ask for something unresolvable, not that a source
//   added to `kernels` is noticed here. `driver-vulkan` never had the
//   property -- it caught all and answered `None` -- and now neither does
//   this backend.
// * FIVE REFUSALS BECOME UNCONSTRUCTIBLE AND UNNAMED. `Unbindable::
//   {NoOperand, NoDriverResource, NoKvCache}` and `Misplaced::{Unresolved,
//   Contiguous}` were built here and nowhere else. The variants stay --
//   `dispatch.rs` and `hold.rs` cite the last two by name for the refusals
//   that took their place -- but nothing produces them and no test names
//   them, so `tests/citations.rs`'s census is five short: its `UNNAMED` list
//   and the totals its own doc quotes have to move, and that edit is a
//   coverage LOSS being recorded, not progress.
// * THE PARAMS-STRUCT ASSUMPTION WENT UNVERIFIED RATHER THAN BECOMING TRUE.
//   `Params` carries one buffer, so a row naming both a `Buf`-kinded `Param`
//   operand -- a struct that swallows the whole scalar tail -- and scalar
//   operands of its own would have had its uniform fields silently dropped on
//   the way to the device. A walk over the table pinned that no row did both;
//   it retired blind when the table emptied. `routine::bind` does not inherit
//   the hazard, because it appends the body's own scalars INSIDE the storage
//   block and a body with both loses neither, but the claim about rows was
//   never settled and there is no table left to settle it against.
// * THE DERIVED FAMILY LOSES ITS WORKED EXAMPLE. `Misplaced::Unresolved`
//   exists because `kernels-metal` grew `add_bias` for the Qwen-2 attention
//   biases -- served as fluent wrong text until then, because no kernel added
//   them -- and its `width` operand was `Source::OutWidth`, which a
//   fallthrough here answered with a ZERO: a row pitch of nothing, which
//   biases element 0 of every row or launches a grid of nothing and reports
//   success over the unbiased projection. An arm computes its own widths from
//   the statement and `Facts`, so the shape has no door on this path; what is
//   gone is the example that showed what a plausible zero costs, and the two
//   tests that held it.
//
// `driver-metal` and `driver-vulkan` deleted their equivalents first. Vulkan
// took `binding::{reorder, scalars, descriptors, runs, is_buffer_kind, Runs}`
// and the nine unit tests that only exercised them, leaving `extent`,
// `resolve`, `bind`, `params` and `params_from` -- which, `descriptors` and
// `Slot` aside, is exactly what remains here.

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
}

// RETIRED: `Misplaced::{Unresolved, Contiguous}` -- nothing builds them.
//
// Both were raised by `scalars`, which read a row's `kernels::Source` column
// and is deleted with the rest of the row path.
//
// `Unresolved` said: a row named a `Source` this driver cannot work out, and a
// NAMED refusal is better than the zero it would otherwise place -- because
// zero is a PLAUSIBLE number. A width of zero is a grid of nothing, which
// dispatches nothing, reads back as the zeros the buffer was born with, and
// completes successfully. It was not hypothetical: `kernels-metal` grew a
// hundredth row, `add_bias` for the Qwen-2 attention biases that were being
// served as fluent wrong text because no kernel added them, and its `width`
// operand is `Source::OutWidth`, which this module resolved as zero.
//
// `Contiguous` said something worse. `Source::KvHeadStride` and
// `Source::KvSeqStride` appear on exactly the rows that walk the cache with
// two strides and no page table -- `kv_append`, `sdpa_vector_decode`,
// `sdpa_vector_decode_swa`. `resources::Shape` allocates `[page, token, head,
// dim]` for every fire this driver runs, so `h * head_stride + pos *
// seq_stride + d` is right only while a fire's pages happen to be physically
// consecutive from zero -- true of one freshly-allocated sequence and false of
// the second. It reads real memory and attends to the WRONG TOKENS: nothing
// faults, nothing is out of bounds, and the text stays fluent. The refusal was
// blanket rather than conditional on the translation being the identity,
// because a driver that served these SOMETIMES would be correct on the first
// request of a fresh cache and wrong afterwards.
//
// THAT SECOND CLAIM IS NOT LOST. `lowering::hold::contiguous_pool` makes it on
// the routine plane, for the same three kernels, by asking the FIRE rather
// than the row: a non-zero `FireNumber::KvPageSize` means the pool is paged,
// and a stride over it is `Refusal::Absent` with the reason in the message.
// `crates/model` reached the same conclusion from the third side and stopped
// emitting these kernels for a paged pool.
//
// THE FIRST CLAIM IS LOST, and there is no honest place to put it: a routine
// takes its scalars as typed arguments, so "a source this driver cannot work
// out" is not a state that can be reached -- the compiler refuses a body whose
// argument no arm supplies. That is the refactor working rather than a gap,
// but it means the `add_bias` class of defect is caught by the type checker
// now and not by a named refusal, and a reader looking for that refusal will
// not find one.

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
        }
    }
}

impl core::error::Error for Misplaced {}

/// Place one launch's scalars the way its module wants them.
///
/// The MODULE's reading of where the plan's own scalar run goes, and the only
/// reading left: the row that used to place its own is retired above
/// [`ParamSlot`], and a routine's scalars are packed and placed by
/// [`crate::lowering::routine::bind`].
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
    /// one module is ordinary -- but only where the tail is `None`. A buffer
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
/// A slot list answers "what does slot *k* of the launch hold"; this answers
/// "what does binding *k* of the module's storage group hold", and they differ
/// in two measured ways.
///
/// A scalar occupies a row slot whether or not it occupies a bind-group entry.
/// A row whose scalars ride the uniform block takes NO `@group(0)` entry for
/// them, so everything after them moves down; a row whose `Param` operand is a
/// `Buf` does take one, and it is a binding the plan never mentions.
///
/// And a row may be longer than a module: several rows end in `None`, one
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
    // that pins the two readings together: the row's unsourced slots and the
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
        // And an unsourced tail past the same module is dropped, not refused.
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
            // A fixture states no attention schedule to raise.
            preps: Vec::new(),
            readout: None,
        }
    }
}

/// A [`Resolve`] that answers a size for anything and holds no memory.
///
/// The device-free half of this module already has [`Placeholder`] for the
/// BUFFER; this is the other half of the same idea for the RESOLVER, so a
/// caller can plan a rectangle without an adapter, a checkpoint or a KV pool.
///
/// It exists for one production question. `engine`'s `widest_fire` measures
/// `max_forward_tokens` by lowering a text and asking whether every launch
/// fits the device's workgroup limit -- a search over a boundary, run before
/// any weights are loaded. That question needs the GRID, which is a fact about
/// the plan, and nothing about the bytes.
///
/// It answers `Some` to everything on purpose: a resolver that refused would
/// make a launch look unplannable for the wrong reason, and the caller would
/// read that as "does not fit".
#[derive(Debug, Clone, Copy)]
pub struct Unbacked(
    /// The size every answer reports.
    pub Placeholder,
);

impl Resolve for Unbacked {
    type Buffer = Placeholder;

    fn weight(&self, _name: &str) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn named(&self, _value: ValueId) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn kv(&self, _layer: u16, _values: bool) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn table(&self, _which: FireTable) -> Option<&Self::Buffer> {
        Some(&self.0)
    }

    fn number(&self, _which: FireNumber) -> Option<u32> {
        // ONE, not zero. Zero is a plausible number that a body may divide by
        // or size a grid with, and this resolver exists to answer "could this
        // be planned", not "what would it compute".
        Some(1)
    }
}
