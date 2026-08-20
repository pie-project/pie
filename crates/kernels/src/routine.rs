//! A routine is an ordinary `fn`, and its table row is derived from its
//! signature.
//!
//! Everything a backend needs to DECLARE a routine set, declaring none itself:
//! backends' kernel sets genuinely differ, so a shared vocabulary of SYMBOLS
//! would be a fiction. What is shared is the shape -- a `fn`, a derived
//! argument table, and one erased entry point -- and
//! [`macro@crate::routine`] produces the row from the signature, so the row
//! cannot drift from the code.
//!
//! [`Backend`] exists because [`Arg::unpack`] takes the backend's argument
//! value and [`KernelFn::invoke`] its context, and this crate can name
//! neither: it is the floor every plane stands on and depends on nothing.

use crate::Derived;
use crate::Ty;

/// WHAT AN OPERAND THAT IS NOT THERE LOOKS LIKE, ASKED OF THE VALUE.
///
/// `Option<M>` is how a signature says an operand may be absent; this is the
/// half only a backend's own value can answer, because nothing in this crate
/// can look inside one. CUDA's absence is a null pointer; a handle plane's is
/// whatever its binder mints for an operand the statement left empty.
///
/// Both methods default to *"this plane cannot express absence"*, which makes
/// `Option<M>` there always `Some` — the honest reading, since a binder that
/// cannot mint an absent value never produces one.
///
/// On the VALUE and not on [`Backend`] because [`Bind`] names no backend: a
/// body re-emitting a `None` has only the value type to ask.
pub trait Absent: Sized {
    /// Whether this is the absent value.
    fn is_absent(&self) -> bool {
        false
    }

    /// The absent value, where this plane has one.
    fn absent() -> Option<Self> {
        None
    }
}

/// One backend's two concrete types, so the machinery can be written once.
///
/// The implementor is a marker: it is never constructed and carries no state.
pub trait Backend: Copy + 'static {
    /// A value bound to one argument — the backend's `ArgValue`.
    type Value: Copy + Absent;
    /// What a routine body launches through.
    ///
    /// `?Sized`, because it is only ever named behind a reference and a
    /// backend may not be able to own its device. CUDA's is a struct holding
    /// the JIT cache and the cuBLAS handles; wgpu's is `dyn Encode`, because
    /// `kernels-wgpu` depends on `kernels` and nothing else — it embeds WGSL
    /// and cannot name an adapter, so the thing a body dispatches through has
    /// to be supplied by the driver. A `Sized` bound here would have forced
    /// that crate to take a `wgpu` dependency it exists not to have.
    type Ctx<'a>: ?Sized;

    /// The shape this value carries, if it carries one.
    ///
    /// On the backend and not on [`Arg`] because nothing in this crate can
    /// look inside `Self::Value` — it is an associated type and `kernels`
    /// names no backend.
    ///
    /// # BINDING ADDRESSES ALONE IS THE DEFAULT
    ///
    /// A region is `{address, rows, width}`, and a plane whose bound value is
    /// an address alone has no shape to give: refusing is the whole of its
    /// correct answer. The handle planes carry the launch's two widths as
    /// `Facts` fields, a per-LAUNCH statement rather than a per-operand one,
    /// which is a better place for them. Stating that per plane only produced
    /// three prose variants of one fact — and a required method every plane
    /// had to answer made *"which backend is this"* readable off the answer.
    ///
    /// The refusal stays unreachable until a table spells a fat `In<N, _>`,
    /// and the first that does finds out at its first fire. Only a plane that
    /// mints region-shaped values overrides.
    ///
    /// # Errors
    ///
    /// [`Refusal::Kind`] when the value is not region-shaped —
    /// [`Refusal::Absent`] when this backend has no region shape at all.
    fn region(value: &Self::Value) -> Result<Extent, Refusal> {
        let _ = value;
        Err(Refusal::Absent { what: "a region's shape: this binder binds addresses only" })
    }

}

/// How many rows a region has and how wide each one is, in elements.
///
/// Two numbers rather than one, and NOT an element count. `bind/facts.rs:110`
/// is why: a width of zero *"is also what a launch that states a
/// three-dimensional operand carries"*, so a product would collapse a stated
/// unknown into a stated zero and every body reading it would divide by it.
/// Kept apart, a body that needs the product writes the multiplication and a
/// body that needs the pitch reads the pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// Rows in this launch's rectangle — `Facts::rows.count`.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// A row pitch, which is not a width.
///
/// `#[repr(transparent)]`, so it costs nothing at run time and exists only to
/// be a different TYPE from the width beside it — three `i32`s in a row, two
/// of them leading dimensions, is where a swap goes unnoticed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Stride(pub i32);

// A STRIDE BINDS EXACTLY AS THE `i32` IT WRAPS. `#[repr(transparent)]` makes
// that true of the memory; this makes it true of the table. The wrapper adds
// no `Source` and changes no ABI -- it is a TYPE distinction and nothing
// else, which is the whole of what it is for.
impl<B: Backend> Arg<B> for Stride
where
    i32: Arg<B>,
{
    const TY: Ty = <i32 as Arg<B>>::TY;
    const SPELLING: &'static str = <i32 as Arg<B>>::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <i32 as Arg<B>>::unpack(value, at).map(Stride)
    }
}

impl core::ops::Deref for Stride {
    type Target = i32;

    fn deref(&self) -> &i32 {
        &self.0
    }
}

/// The allocation's shape: what is true of the address, not of one launch.
///
/// Rule F1: a pointer carries its layout, a launch applies a view. A layout is
/// 1:1 with the address, never changes while the address is valid, and is
/// never absent — there is no buffer with no extent, only a transport that
/// dropped it.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Rows, then elements per row.
    dims: [i32; 2],
    /// The pitch of each, in elements.
    strides: [i32; 2],
}

impl Layout {
    /// The layout a flattened extent can describe, and the only one it can.
    ///
    /// The leading stride IS the width. That is the packed claim, stated
    /// once, where a reader looking for it will find it.
    #[must_use]
    pub const fn packed(rows: i32, width: i32) -> Self {
        Self { dims: [rows, width], strides: [width, 1] }
    }

    /// How many rows the allocation holds.
    #[must_use]
    pub const fn rows(&self) -> i32 {
        self.dims[0]
    }

    /// Elements per row.
    #[must_use]
    pub const fn row_width(&self) -> i32 {
        self.dims[1]
    }

    /// The distance from one row's start to the next, in elements.
    ///
    /// Equal to [`Self::row_width`] under [`Self::packed`] and typed
    /// differently anyway — see [`Stride`]. A body that wants the pitch says
    /// so, and cannot get it by reading the width with a different variable
    /// name.
    #[must_use]
    pub const fn row_pitch(&self) -> Stride {
        Stride(self.strides[0])
    }
}

// `packed` is the only constructor, so asserting `layout.row_width()` against
// the width there proves it for every caller — which is what a `cargo
// check`-only regime can prove. The second assertion fails first when the
// lowering starts carrying real strides, and failing is what it is for.
const _: () = {
    let l = Layout::packed(7, 4096);
    assert!(l.rows() == 7);
    assert!(l.row_width() == 4096);
    assert!(l.row_pitch().0 == l.row_width());
    // A ZERO WIDTH IS A STATED UNKNOWN AND SURVIVES AS ONE. `bind/facts.rs`
    // is why a width of zero must not collapse into an element count: it is
    // also what a launch stating a three-dimensional operand carries.
    let none = Layout::packed(7, 0);
    assert!(none.row_width() == 0);
    assert!(none.row_pitch().0 == 0);
};

// `Provenance` AND `Side` STOOD HERE AND ARE BOTH DELETED.
//
// `Provenance` answered *"who supplies this argument"* — the statement, the
// environment, or either. With `Env` out of the parameter list EVERY parameter
// is the statement's, so the column had one value at every row and
// `arity_problem` had nothing left to filter.
//
// `Side` answered *"which side of the statement does it sit on"* — placed,
// declared, or ask the type. The MARK says: `In` and `Const` place, `Out` and
// `InOut` declare, and there is no third case now that a parameter cannot be
// unmarked. It was a column on every routine restating what the mark beside it
// already carried, and `Source` — which every reader already walks — carries
// the same fact as `Slot(Kind::In, _)`, `Slot(Kind::Out, _)`,
// `Slot(Kind::Weight, _)`, `Slot(Kind::Param, _)` and `Alias(_, _)`.

/// Why a routine did not launch.
///
/// A refusal is a VALUE, not a panic and not a log line: a caller that asked
/// for an empty rectangle wants `Ok`-shaped silence, and a caller that asked
/// for an unsupported width wants to fall back. Only the caller knows which,
/// so the distinction survives to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// An extent is zero or negative: there is nothing to launch.
    Empty {
        /// Which extent, in the routine's own word for it.
        what: &'static str,
    },
    /// An extent is real but below the routine's smallest unit of work.
    Narrow {
        /// Which extent.
        what: &'static str,
        /// What it was.
        at: i64,
    },
    /// An extent is above a ceiling the compiled kernel cannot exceed.
    Wide {
        /// Which extent.
        what: &'static str,
        /// What it was.
        at: i64,
        /// The largest this kernel was compiled for.
        max: i64,
    },
    /// A pointer the routine cannot dereference.
    Null {
        /// Which argument, by the name the `fn` gives its parameter.
        what: &'static str,
    },
    /// A pointer whose address does not meet the kernel's alignment.
    Misaligned {
        /// Which argument.
        what: &'static str,
    },
    /// A grid the device will not accept.
    Grid {
        /// Which axis, or what about it.
        what: &'static str,
        /// What it was.
        at: i64,
    },
    /// An argument the fire did not carry.
    Absent {
        /// Which argument.
        what: &'static str,
    },
    /// A fact no statement and no context carries.
    Unstated {
        /// The fact, named.
        what: &'static str,
    },
    /// Nothing declares this routine.
    Undeclared,
    /// The argument list is the wrong length for the routine.
    Arity {
        /// Arguments the signature takes.
        want: usize,
        /// Values the caller supplied.
        got: usize,
    },
    /// A value of the wrong kind for the argument it was bound to.
    Kind {
        /// Which position.
        at: usize,
        /// What the signature takes there.
        want: Ty,
    },
    /// The device refused the launch, or there was no device.
    Device {
        /// What the driver said, as this crate cannot own a CUDA error type.
        why: &'static str,
    },
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "nothing to launch: {what} is zero"),
            Self::Narrow { what, at } => {
                write!(f, "{what} is {at}, below the smallest unit of work")
            }
            Self::Wide { what, at, max } => {
                write!(
                    f,
                    "{what} is {at}, above the {max} this kernel was compiled for"
                )
            }
            Self::Null { what } => write!(f, "{what} is null"),
            Self::Misaligned { what } => write!(f, "{what} is not aligned as the kernel reads it"),
            Self::Grid { what, at } => {
                write!(f, "the grid's {what} is {at}, which will not launch")
            }
            Self::Absent { what } => write!(f, "the fire does not carry {what}"),
            Self::Unstated { what } => write!(f, "nothing states {what}"),
            Self::Undeclared => write!(f, "nothing declares it"),
            Self::Arity { want, got } => write!(f, "it takes {want} arguments and {got} arrived"),
            Self::Kind { at, want } => write!(f, "argument {at} is {want:?} and arrived otherwise"),
            Self::Device { why } => write!(f, "the device refused: {why}"),
        }
    }
}

impl core::error::Error for Refusal {}

/// One argument type, and what it contributes to the derived table.
///
/// Implemented by a backend for its own argument types. The two consts are
/// what the row is built from; [`Arg::unpack`] is what the erased call path
/// goes through.
pub trait Arg<B: Backend>: Sized {
    /// What this argument is, in the table's vocabulary.
    const TY: Ty;

    /// The question this type claims, or `None` when no type claims one.
    ///
    /// A mark that claims a SLOT answers `None` here and states its
    /// [`Arg::CLAIM`] instead: the slot's index is not a property of the type,
    /// it is a property of where the type sits, and only the signature knows
    /// that. [`KernelFn::SOURCES`] is where the two meet.
    const SOURCE: Option<crate::Source> = None;
    /// What position this mark claims, before its index is known.
    ///
    /// Defaults to [`Claim::Fixed`] — "this argument's source, if any, is
    /// already settled" — and is overridden by the four marks. Since every
    /// parameter of a columned routine IS a mark, `Fixed` is only ever the
    /// carriers' own answer, read through the mark that wraps them.
    const CLAIM: Claim = Claim::Fixed;
    /// How the backend's shader language spells this type, whole — the
    /// `const` and the star included.
    ///
    /// Empty for a type whose spelling the backend has not written down. Not
    /// read by anything today; carried because it derives at zero cost and is
    /// what a generated cross-check against the real kernel declaration needs.
    const SPELLING: &'static str = "";

    /// Recover this argument from the value bound at position `at`.
    ///
    /// # Errors
    ///
    /// [`Refusal::Kind`] if the value is not of this argument's kind.
    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal>;
}

// `Env<T, K>` AND `Keyed` STOOD HERE AND ARE DELETED.
//
// `Env` was the mark that named a SUPPLIER where the other four named a
// quality, and it was the one mark that was not positional. Most of what it
// wrapped was never the environment's: 71 keys over 936 uses are checkpoint
// configuration fixed the moment the model is read, and they were `Env` only
// because the statement had no channel to carry them. [`Const`] is that
// channel.
//
// What genuinely varies per fire — the batch, the plan and the allocator's
// addresses — did not need a PARAMETER, only an answer. A body asks for those
// through [`Asks::ask`], which routes to the same `Holds::fact` the column
// used to route to: the ANSWERING side is unchanged, only the asking side.
//
// With `Env` gone every parameter is positional and every mark is the same
// kind of word, which is the whole of what this refactor is for.

/// THE RUNTIME'S SIDE OF AN ASK: one value, resolved from the column's own
/// vocabulary.
///
/// The answering side of this refactor does NOT change. A driver already
/// resolves a `(Ty, Source)` pair for every argument it binds — that is
/// [`crate::bind::one`], and [`crate::bind::Holds::fact`] is the one method it
/// implements to answer a [`crate::Source::Named`]. What changes is that a
/// BODY can now ask the same question, instead of the question having to be a
/// parameter for the column to carry it.
///
/// Object-safe on purpose: three of the four planes reach their driver through
/// a `dyn` trait, so the resolver crosses as a trait object too.
pub trait Answers<B: Backend> {
    /// The value this `(Ty, Source)` pair binds to, for this fire.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when this backend answers no such fact, and
    /// whatever the fact's own absence means otherwise.
    fn resolve(&self, ty: Ty, source: crate::Source) -> Result<B::Value, Refusal>;
}

/// WHAT ONLY THE ENGINE'S RUNTIME CAN ANSWER, asked by the body that needs it.
///
/// # The rule
///
/// **`ask` is for what only the engine's runtime can answer.** A fact the
/// checkpoint fixes at load time is a constant, and a constant belongs in the
/// statement — as a [`Const`] parameter, positional like every other.
/// `keys::HeadDim` is not asked, because a head dimension is not something
/// this batch made; `keys::Rows` is, because two batches differ.
///
/// The test for anything added later: *"two fires of the same model, on the
/// same deployment, can see different answers here"*. `Rows` passes.
/// `KvKeys` passes — the allocator moved. `HeadDim` does not, and no amount of
/// the driver knowing it makes it pass.
///
/// # Why the carrier is written at the call
///
/// Because a fact's `Value` is one concrete type and a plane's carrier is not.
/// `keys::Positions` declares `*const i32`; that IS the carrier on CUDA and is
/// meaningless on a plane that binds a handle. So the call names both, in the
/// order `Env<T, K>` named them — the carrier first, because it is the half
/// that is always there:
///
/// ```ignore
/// let positions = ctx.ask::<Tensor<i32>, keys::Positions>()?;   // a plane's handle
/// let rows      = ctx.ask::<i32, keys::Rows>()?;                // this batch's count
/// ```
///
/// # What it costs
///
/// `ask` is a CALL, not a declaration, so the derived column no longer
/// enumerates it and a driver test can no longer walk that column to ask
/// *"does this backend answer every fact its own kernels name"*. `#[routine]`
/// collects `ask::<_, keys::X>` out of the body instead — same fidelity as the
/// parameter run, and it cannot drift from the calls — but it misses a fact
/// asked inside a helper. That is a real step down from a type-system
/// guarantee to a syntactic one, and it is accepted deliberately.
pub trait Asks<B: Backend>: Answers<B> {
    /// The environment's answer to `K`, in the carrier `C` this plane binds.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when this backend answers no such fact;
    /// [`Refusal::Kind`] when the answer is not of `C`'s kind.
    fn ask<C: Arg<B>, K: crate::keys::Fact>(&self) -> Result<C, Refusal> {
        // THE KEY'S OWN SOURCE, WHICH IS NOT ALWAYS A `Named`.
        // `keys::WindowOrNone` is `Or(Named("window_left"), Lit(-1))`, and
        // `keys::GqaFactor` was arithmetic over two facts. Resolving the
        // SOURCE rather than the string is what keeps a chain a chain.
        let v = self.resolve(C::TY, <K as crate::keys::Fact>::SOURCE)?;
        C::unpack(&v, 0)
    }

    /// The statement's whole scalar run, as the one buffer a kernel that reads
    /// a struct takes.
    ///
    /// What `Env<Buf, keys::Params>` spelled at 97 signatures. It is not a
    /// fact and never was: the block is the statement's own params run, staged
    /// by the driver because six of this tree's shader modules read their
    /// parameters out of a storage block rather than a push range.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] on a plane that stages no such block — which is
    /// CUDA, where scalars are passed one at a time.
    fn params(&self) -> Result<B::Value, Refusal> {
        self.resolve(Ty::Buf, crate::Source::Slot(crate::Kind::Params, 0))
    }

    /// The statement's `n`-th scalar, read as an `i32`.
    ///
    /// # Why a body ever reaches past its own marks
    ///
    /// Because a routine that forwards [`Self::params`] as a BLOCK has no
    /// slots to spare. Its params run is the shader's own layout — read by
    /// field, not by position — so a `Const<i32>` derived onto slot 0 reads
    /// that block's first word, and there is no mark that can name the
    /// eleventh. `Param<11, i32>` said exactly this before the marks, and
    /// `gdn_core_recurrent_prefill`'s tiling is where it still has to be said:
    /// the body picks its compiled point from two words the SHADER also reads.
    ///
    /// **Not a general escape hatch.** A number a routine can take as a
    /// `Const` should be one — the mark is what makes the arity checkable, and
    /// `check_plan` counts marks. This is for the case where the same run
    /// serves two readers and the other one is a block.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when the statement carries no such scalar.
    fn param(&self, n: u8) -> Result<i32, Refusal>
    where
        i32: Arg<B>,
    {
        <i32 as Arg<B>>::unpack(
            &self.resolve(<i32 as Arg<B>>::TY, crate::Source::Slot(crate::Kind::Param, n))?,
            usize::from(n),
        )
    }

    /// The operand this routine deliberately leaves ABSENT.
    ///
    /// An argument list is positional, so an absence still occupies a cell —
    /// which is what `Env<Buf, keys::Absent>` was for at eighteen signatures,
    /// and `Env<*const T, keys::Unstated>` at fifty-three. Neither was a
    /// question and neither had an answerer; both were a launcher supplying
    /// its own null. A body says so where it says everything else about its
    /// argument list.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] on a plane whose binder mints no null.
    fn absent(&self) -> Result<B::Value, Refusal> {
        self.resolve(Ty::Buf, crate::Source::Lit(crate::Lit::Null))
    }
}

// EVERY ANSWERER ASKS. `Asks` carries no state of its own -- it is the
// vocabulary a body writes, over the one method a plane implements -- so
// there is nothing for a plane to get wrong by implementing it itself.
impl<B: Backend, T: Answers<B> + ?Sized> Asks<B> for T {}

/// ONE DISPATCH, as a routine body states it — on every plane.
///
/// # What it replaced
///
/// Two entry points that carried the same four facts. CUDA's was four
/// POSITIONAL arguments:
///
/// ```ignore
/// ctx.launch("norm/rmsnorm.cuh", "::pie::norm::rmsnorm<bf16, 256>", per_row(rows), &args)
/// ```
///
/// and the shader planes' was already a struct, so the difference was never
/// in what a body knew — only in how it said it. The positional form is also
/// the one that could go wrong quietly: `file` and `entrypoint` are both
/// `&str`, so a swap type-checks.
///
/// # The two counts
///
/// [`Self::lanes`] is TOTAL WORK and [`Self::group`] its divisor, on every
/// plane. CUDA bodies used to state the grid — how many BLOCKS — and the
/// block size beside it; the same pair, counted from the other end. Stating
/// the total and the divisor is what makes the field mean one thing
/// everywhere: the driver divides, and `lanes / group` returns the grid the
/// body would have written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    /// The file the point lives in — a `.cuh`, a `.metal`, a `.wgsl`, a `.spv`.
    ///
    /// Stated on every plane. What it names differs with what the plane loads:
    /// CUDA, Metal and wgpu name a SOURCE that holds many points and gets
    /// compiled, while vulkan names the compiled ARTIFACT, because its build
    /// emits one module per entrypoint per tier. A vulkan body composes it
    /// from the tier its `Ctx` reports, which is how tier selection reads off
    /// the body rather than happening behind it.
    pub file: &'static str,
    /// Which point in it: an entrypoint name, or on CUDA the fully qualified
    /// template-id NVRTC is asked to lower.
    pub entrypoint: &'static str,
    /// The TRANSLATION UNIT's own name, where it is not the file's.
    ///
    /// Empty means "the file names it", which is every point but two lattices:
    /// FA2 and XQA compile one file several ways, so a point there is reached
    /// by a name the file does not carry. Said HERE rather than by a second
    /// dispatch method taking a resolved root -- one such method existed, was
    /// `unsafe`, and was the only reason three routine bodies on one plane
    /// carried an `unsafe` block the other three planes had no counterpart
    /// for.
    pub unit: &'static str,
    /// Total work in each dimension, in ELEMENTS of work.
    ///
    /// A body must not state a zero. `vkCmdDispatch(0, 1, 1)` is legal Vulkan
    /// that runs nothing and reports success — the failure this surface exists
    /// to make impossible, met twice on that backend. A body with nothing to
    /// do returns [`Refusal::Empty`].
    pub lanes: [u32; 3],
    /// The divisor: threads per block, per threadgroup, per workgroup.
    ///
    /// `[0, 0, 0]` where the SHADER TEXT declares it and the driver recovers
    /// it — `[numthreads]` lands in the SPIR-V as `OpExecutionMode LocalSize`,
    /// and a body that restated it would carry a second copy of a number it
    /// cannot see.
    pub group: [u32; 3],
    /// Dynamic shared memory, in bytes. Zero on a plane that has none.
    pub smem: u32,
    /// The launch needs every block resident at once. CUDA's only.
    pub cooperative: bool,
}

impl Fire {
    /// The point `entrypoint` in `file`, with nothing else stated.
    #[must_use]
    pub const fn at(file: &'static str, entrypoint: &'static str) -> Self {
        Self {
            file,
            entrypoint,
            unit: "",
            lanes: [0, 0, 0],
            group: [0, 0, 0],
            smem: 0,
            cooperative: false,
        }
    }

    /// This fire, in the translation unit called `unit` rather than the one
    /// the file names.
    #[must_use]
    pub const fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    /// This fire, over `lanes` elements of work.
    #[must_use]
    pub const fn lanes(mut self, lanes: [u32; 3]) -> Self {
        self.lanes = lanes;
        self
    }

    /// This fire, divided `group` ways.
    #[must_use]
    pub const fn group(mut self, group: [u32; 3]) -> Self {
        self.group = group;
        self
    }

    /// This fire, with `smem` bytes of dynamic shared memory.
    #[must_use]
    pub const fn smem(mut self, smem: u32) -> Self {
        self.smem = smem;
        self
    }

    /// This fire, needing every block resident at once.
    #[must_use]
    pub const fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    /// ONE THREAD PER ELEMENT, in groups of `group`.
    #[must_use]
    pub const fn flat(self, n: u32, group: u32) -> Self {
        self.lanes([n, 1, 1]).group([group, 1, 1])
    }

    /// ONE GROUP PER ROW, `group` threads wide.
    ///
    /// The lane count is the PRODUCT, because `lanes` is total work: a row per
    /// group and `group` threads in it is `rows * group` elements.
    #[must_use]
    pub const fn per_row(self, rows: u32, group: u32) -> Self {
        self.lanes([rows.saturating_mul(group), 1, 1]).group([group, 1, 1])
    }

    /// A grid stated as GROUPS on all three axes, with the divisor beside it.
    ///
    /// What a CUDA body used to write as `Launch::grid(grid, block)`. The
    /// lanes are the componentwise product; [`Self::grid`] divides back.
    #[must_use]
    pub const fn groups(self, grid: [u32; 3], group: [u32; 3]) -> Self {
        self.lanes([
            grid[0].saturating_mul(group[0]),
            grid[1].saturating_mul(group[1]),
            grid[2].saturating_mul(group[2]),
        ])
        .group(group)
    }

    /// This fire, with a geometry a helper computed.
    ///
    /// The one place a plane's own grid type meets the shared one: CUDA's
    /// families keep helpers like `elementwise(width, rows)` that return a
    /// `Launch`, and `Launch::into_fire` feeds it here rather than making
    /// every helper return a `Fire` it has no file or entrypoint for.
    #[must_use]
    pub const fn geometry(mut self, lanes: [u32; 3], group: [u32; 3], smem: u32, cooperative: bool) -> Self {
        self.lanes = lanes;
        self.group = group;
        self.smem = smem;
        self.cooperative = cooperative;
        self
    }

    /// This fire, with a geometry the plane's own helper computed.
    ///
    /// The one place a backend's grid type meets the shared one. CUDA's
    /// families keep helpers like `elementwise(width, rows)` that answer with
    /// a `Launch`; this takes it rather than making every helper name a file
    /// and an entrypoint it knows nothing about.
    #[must_use]
    pub fn apply<G: Geometry>(self, g: G) -> Self {
        g.apply_to(self)
    }

    /// How many groups the driver will launch: `lanes / group`, rounded up.
    ///
    /// The grid a CUDA body used to state. Componentwise, and a zero divisor
    /// answers the lane count unchanged — which is the shader planes' case,
    /// where the text declares the divisor and this is not consulted.
    #[must_use]
    pub const fn grid(&self) -> [u32; 3] {
        let mut out = [0u32; 3];
        let mut i = 0;
        while i < 3 {
            out[i] = if self.group[i] == 0 {
                self.lanes[i]
            } else {
                self.lanes[i].div_ceil(self.group[i])
            };
            i += 1;
        }
        out
    }
}





// `Clone`/`Copy` BY HAND, BECAUSE `derive` PUTS THE BOUND ON THE PARAMETER.
// A derived `Copy` on `In<E>` asks for `E: Copy` -- the ELEMENT -- when what
// has to be `Copy` is the carrier the element names. `bf16` is a pointee that
// nothing copies; `*const bf16` is.
impl<E: Elem> Clone for In<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for In<E> {}

impl<E: Elem> Clone for Out<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for Out<E> {}

impl<E: Elem> Clone for InOut<E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E: Elem> Copy for InOut<E> {}


/// A carrier, as the value a launch is given.
///
/// THE INVERSE OF [`Arg::unpack`], and one name on four planes. It was two:
/// CUDA's `Abi::arg` and the shader planes' `Bind::v`, doing the same job
/// under different words, so a body could be told apart by which one it
/// called. The marks delegate through it, which is why a body writes
/// `x.arg()` and not `x.ptr.arg()` -- a mark says which SLOT a carrier came
/// from, which is the table's business and never the encoder's.
pub trait Bind<V>: Copy {
    /// This carrier, as a bound value.
    fn arg(self) -> V;
}

// THE MARKS BIND AS THEIR CARRIERS DO, on every plane.
impl<V, E: Elem> Bind<V> for In<E>
where
    E::Read: Bind<V>,
{
    fn arg(self) -> V {
        self.ptr.arg()
    }
}

/// A carrier bound the way a WRITE binds it.
///
/// [`Elem::Write`] is `*mut T` on CUDA and `Tensor<E>` on the shader planes,
/// and on the shader planes it is *the same type* as [`Elem::Read`] -- one
/// handle, one `Bind` impl. So `Out` and `InOut` binding through `Bind` alone
/// erased the direction on exactly the planes whose value type carries it:
/// every operand came out `ArgValue::Buffer` and none came out `BufferMut`.
///
/// That is not cosmetic. `driver-metal`'s `touches` reads `ArgValue::BufferMut`
/// to learn WHICH BUFFERS A DISPATCH WRITES, so a plane that never produces one
/// tells the driver every dispatch writes nothing -- and the hazard tracking
/// that answer feeds is the whole of what orders one encoder's work.
///
/// The mark picks the trait; the trait picks the constructor. On CUDA both
/// arrive at the same place, because there `*const T` and `*mut T` already
/// differ.
pub trait BindMut<V>: Copy {
    /// This carrier, as a bound value a launch may WRITE.
    fn arg_mut(self) -> V;
}

// A POINTER ALREADY SAYS IT. `*mut T` is not `*const T`, so CUDA's direction
// survives `Bind` and this is the identity.
impl<V, T> BindMut<V> for *mut T
where
    *mut T: Bind<V>,
{
    fn arg_mut(self) -> V {
        self.arg()
    }
}

impl<V, E: Elem> Bind<V> for Out<E>
where
    E::Write: BindMut<V>,
{
    fn arg(self) -> V {
        self.ptr.arg_mut()
    }
}

impl<V, E: Elem> Bind<V> for InOut<E>
where
    E::Write: BindMut<V>,
{
    fn arg(self) -> V {
        self.ptr.arg_mut()
    }
}

// A `Const` BINDS AS WHAT IT HOLDS: a weight plane's handle, or the scalar
// the statement placed in its params run.
impl<V, C: ConstRun> Bind<V> for Const<C>
where
    C::Held: Bind<V>,
{
    fn arg(self) -> V {
        self.v.arg()
    }
}

/// A backend's own launch geometry, as a [`Fire`] can take it.
///
/// One method, so a plane that computes grids in its own vocabulary can hand
/// one over without every helper learning what a file is.
pub trait Geometry {
    /// This geometry, applied to `fire`.
    #[must_use]
    fn apply_to(self, fire: Fire) -> Fire;
}

// THE SHARED FORM APPLIES AS ITSELF, so a plane whose helpers already answer
// in lanes and groups needs no type of its own.
impl Geometry for [u32; 3] {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.lanes(self)
    }
}

/// Lanes AND the divisor, for a plane whose shader text does not declare one.
///
/// # Why this exists rather than a second builder call
///
/// [`Fire::group`] is a method and a body could chain it, and for a while one
/// plane did: metal wrote `.lanes(x).group(y)` at ninety-nine sites while the
/// other three wrote `.lanes(x)` and CUDA wrote `.apply(..)`. Three spellings
/// of "how big is this launch", and which one a file used named the backend.
///
/// The DIFFERENCE IS REAL and belongs somewhere: `[numthreads]` lands in the
/// SPIR-V and the WGSL, so vulkan and wgpu have nothing to say about the
/// divisor, and metal must state it. So it is said in the HELPER'S RETURN
/// TYPE, where a plane fact belongs, and every body on every plane reads
/// `.apply(g)` and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Total work in each dimension.
    pub lanes: [u32; 3],
    /// The divisor.
    pub group: [u32; 3],
}

impl Grid {
    /// `lanes` elements of work, divided `group` ways.
    #[must_use]
    pub const fn of(lanes: [u32; 3], group: [u32; 3]) -> Self {
        Self { lanes, group }
    }
}

impl Geometry for Grid {
    fn apply_to(self, fire: Fire) -> Fire {
        fire.lanes(self.lanes).group(self.group)
    }
}

/// What position a mark claims, before its index is known.
///
/// # Why the index is not on the type
///
/// It used to be: `In<0, T>`, `Out<1, T>`. Every one of those numbers was
/// written by hand, 1,271 of them, and a wrong one COMPILES -- it binds the
/// operand at another index, and `quant.rs` carries the warning in prose:
/// *"`In(0)` here would compile and bind the index run where the activation
/// belongs."*
///
/// The numbers were measured before they were deleted. Across 185 CUDA
/// routines the stated index equalled the position in the signature at every
/// site but thirteen, and every one of the thirteen was the SAME hole: a
/// trace operand that no parameter named, because an output was standing in
/// for it. [`InOut`] names it, and with that the two agree everywhere.
///
/// So the index is not stated. It is the mark's position among the marks,
/// which is a fact the signature already carries and cannot get wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    /// No position: the source is settled by the type, or there is none.
    Fixed,
    /// The next operand the statement places.
    In,
    /// The next result it declares.
    Out,
    /// The next operand AND the next result, at one address.
    InOut,
    /// The next weight — what a [`Const`] over a tensor carrier claims.
    Weight,
    /// The next scalar in the statement's params run.
    ///
    /// [`Const`]'s other run. The params run lost its mark when `Param<N, T>`
    /// was deleted and the eighteen scalars that used it became
    /// `fact!(stated ..)` keys resolving through [`crate::Source::Named`] —
    /// which no driver answers, so every routine taking one was unreachable.
    /// This is the mark those scalars had been missing.
    Param,
    /// The same slot, read as a FLOAT.
    ///
    /// The params run is an untyped `u32` run and a scale is a float, so the
    /// READING is a different channel rather than a different type at one.
    ParamF32,
}

/// One operand the statement places: its carrier AND the shape it was given.
///
/// `T` IS THE CARRIER AND NOT THE POINTEE. `In<*const bf16>` on CUDA and
/// `In<Buf>` on a shader plane are the same type: the two differ only in what
/// a launch can SEE of the operand, and that is what `rows` and `width`
/// already say. A plane whose binder carries no rectangle answers zero for
/// both, which is the value [`Region`] already refuses on.
///
/// It carries `rows` and `width` because the statement gave them; tearing
/// them off is what put `#[source(OutWidth(0))] width: i32` in forty-seven
/// signatures. It does not `Deref`: with three fields, picking the address
/// would make `y.rows` and `*y` two kinds of access to one value.
#[derive(Debug)]
pub struct In<E: Elem> {
    /// The carrier the element names for a READ: a `*const E` on CUDA, a
    /// binding index on a shader plane. See [`Elem::Read`].
    pub ptr: E::Read,
    /// Rows in this launch's rectangle. Zero where the plane states none.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// One result the statement declares. [`In`]'s counterpart, and fat for the
/// same reason.
#[derive(Debug)]
pub struct Out<E: Elem> {
    /// The carrier the element names for a WRITE. See [`Elem::Write`].
    pub ptr: E::Write,
    /// Rows in this launch's rectangle. Zero where the plane states none.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// ONE ADDRESS WEARING BOTH SLOTS: the statement places it and declares it.
///
/// # What this replaced
///
/// `in_place = &[(0, 0)]`, stated on the routine's ROW, forty-five times.
/// The row sat forty lines from the parameters its numbers indexed, so it had
/// to re-identify them in prose -- `norm.rs` explains that `(1, 0)` means
/// *"output 1 is one past the only declared result"*, an index for a result
/// that does not exist, written to say that an input is also written.
///
/// The measurement said the signature already knew. Of fifty-five stated
/// pairs, fifty-two described ONE parameter standing in two slots, and the
/// proof was the hole it left: `residual_add` takes `In<1>` with no `In<0>`,
/// because output 0 IS input 0. A mark that says so closes the hole and the
/// pair derives -- see [`crate::Source::Alias`].
///
/// Not `in_place`, which was the ALLOCATOR's record of the same fact, read
/// off the row. This is the fact itself, at the parameter that is it.
#[derive(Debug)]
pub struct InOut<E: Elem> {
    /// The carrier for a WRITE: an aliased operand is driven both ways, and
    /// the writing form is the one that can do both.
    pub ptr: E::Write,
    /// Rows in this launch's rectangle. Zero where the plane states none.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// The routine's next weight, named OR positional.
///
/// One mark over two reads, and the chain is what makes that honest: an
/// `OpKind::Launch` puts a weight in the operand list where it is positional,
/// while a semantic op like `OpKind::Rmsnorm` carries only a NAME on
/// `LaunchSpec::weight`. Two marks meant a routine had to know which shape of
/// op would reach it -- reading only the name is what made gemma-4 refuse at
/// its PLE prologue.
///
/// [`crate::Source::Or`] already resolves *"the first if the statement
/// carries one, the second otherwise"*, and the binder already had an arm for
/// each half. So the two reads are one source and the caller states neither.
/// THE STATEMENT PLACED IT AND THE LAUNCH ONLY READS IT.
///
/// The fourth mark, and a QUALITY like the other three. `Weight` was the one
/// domain noun in a set of qualities, and beside it `Env` named a supplier
/// rather than a direction — so the set said three different kinds of thing.
/// `Const` says one: *"the statement carries this and the fire cannot change
/// it"*. Which RUN it lands in is the carrier's business, not the mark's:
///
/// | carrier | run | example |
/// | --- | --- | --- |
/// | `Tensor<E>` | the weights | `Const<Tensor<bf16>>` — a weight plane |
/// | `i32`, `u32`, `bool` | the params | `Const<i32>` — a scalar at `params[n]` |
/// | `f32` | the params, read as float | `Const<f32>` — the same slot's bits |
///
/// # Why the scalar half had to exist
///
/// Because most of what this tree called *"the environment"* is checkpoint
/// configuration that never changes after load, and it was `Env` only because
/// the statement had no channel to carry it. A head dimension is not something
/// this batch made. Asking for one costs three things: a `Holds::fact` arm in
/// all four drivers, a guard the type could have made — a value that arrives
/// by ask can be absent, so the body checks — and the provenance question,
/// re-opened. `Const` is the channel, and a fact the statement carries cannot
/// be absent, because `arity_problem` refused the statement first.
///
/// # The named weight chain is inherited, not replaced
///
/// An `OpKind::Launch` puts a weight in the operand list where it is
/// positional, while a semantic op like `OpKind::Rmsnorm` carries only a NAME.
/// [`resolve`] gives a tensor `Const` the same [`crate::Source::Or`] chain
/// `Weight` had — the named bank first and the positional one after.
#[derive(Debug)]
pub struct Const<C: ConstRun> {
    /// What arrived: a read carrier for a tensor, the number for a scalar.
    ///
    /// A body reads a scalar through [`Deref`](core::ops::Deref) — `*head_dim`
    /// — and a tensor through [`Bind::arg`], exactly as the mark it replaced.
    pub v: C::Held,
}

/// Which run a [`Const`] lands in, and what it holds when it gets there.
///
/// The one trait a mark's carrier is asked about, because `Const` is the one
/// mark whose run depends on its carrier. A `Tensor<E>` is a weight, a scalar
/// is a param, and an `f32` is a param read through the float channel.
///
/// `Tensor<E>` implements this in the PLANE that declares it — CUDA's holds a
/// pointer where a shader's holds a binding index, which is the same split
/// [`Elem`] already carries and for the same reason.
pub trait ConstRun {
    /// Which run the statement placed this in.
    const RUN: Claim;
    /// How the argument binds.
    const TY: Ty;
    /// What a `Const` of this carrier holds.
    type Held: Copy;
}

// THE SCALAR RUN. `i32`, `u32` and `bool` share the params channel and `f32`
// takes the float reading of the same slot -- the run is a `Vec<u32>` and the
// BITS are the value, which `Handles::param_f32` already reads back.
impl ConstRun for i32 {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::I32;
    type Held = i32;
}

impl ConstRun for u32 {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::U32;
    type Held = u32;
}

impl ConstRun for f32 {
    const RUN: Claim = Claim::ParamF32;
    const TY: Ty = Ty::F32;
    type Held = f32;
}

impl ConstRun for bool {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::Bool;
    type Held = bool;
}

impl ConstRun for i64 {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::I64;
    type Held = i64;
}

impl ConstRun for usize {
    const RUN: Claim = Claim::Param;
    const TY: Ty = Ty::Usize;
    type Held = u64;
}

impl<C: ConstRun> Clone for Const<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: ConstRun> Copy for Const<C> {}

impl<C: ConstRun> Const<C> {
    /// Carry `v` as the statement's constant.
    pub const fn new(v: C::Held) -> Self {
        Self { v }
    }

    /// What the statement carried.
    pub const fn get(self) -> C::Held {
        self.v
    }
}

// A SCALAR `Const` DEREFS AND A TENSOR ONE DOES NOT NEED TO. `*head_dim` is
// how every body already read the number, and the carrier for a tensor is a
// handle a body passes on rather than reads through.
impl<C: ConstRun> core::ops::Deref for Const<C> {
    type Target = C::Held;

    fn deref(&self) -> &C::Held {
        &self.v
    }
}




/// A pointee: something a device address can point at, with both of its
/// pointer ABIs.
///
/// Not `jit::abi::Inst`, which is a C++ instantiation marker — how the device
/// text spells a type — and has implementors that are pure markers with no
/// pointer ABI at all. This is the host-side pair.

// `Held<Q, T>` STOOD HERE AND WAS THE LAST OF THE FIFTEEN. It set
// `SOURCE = Q::SOURCE` while forwarding `PROV = T::PROV`, and its doc said
// why: *"The op DOES carry an operand for `conv_state` and the arity checker
// counts it; the driver just never reads it."* THE PREMISE WAS NOT TRUE, and
// the same doc had already named the way to find out -- *"an operand the
// driver never reads is an operand doing nothing, and the fix is upstream in
// whatever emits the op."*
//
// Upstream says: `TraceBuilder::gdn_prep` pushes `vec![qkv, a, b]` and no
// state slab, and `model-ir::kernels::arity_problem` only runs on
// `OpKind::Launch` -- so on the semantic ops that reach `gdn_core` the
// provenance was never read at all. On the ops that ARE launched it was read
// and was wrong: `kv_append`'s cache pair and positions arrive on
// `OpKind::Launch::state`, which is a FIELD beside `inputs` and not one of
// them, so a `Trace` provenance counted three reads against the two operands
// the statement places. `Env` is what those parameters always were.
//
// So the row a wrapper existed to produce was a row no reader wanted. It is
// `Env<T, Q>` now, which says the same thing about the source and the honest
// thing about who supplies it.

pub trait Elem: 'static {
    /// What a launch that READS this element is handed.
    ///
    /// # Why the carrier is here and not in the signature
    ///
    /// Because the direction is the MARK's, and it was being said twice.
    /// CUDA wrote `In<*const bf16>` and `Out<*mut bf16>`; the shader planes
    /// wrote `In<Buf>` and `Out<BufMut>` -- four spellings for two facts, and
    /// a `BufMut` differs from a `Buf` in nothing a shader body can observe
    /// (both are `ArgValue::Buffer(handle)`, and every body only calls `.v()`).
    ///
    /// It could not simply be deleted, because on CUDA the mutability is
    /// Rust's own type system doing real work: a body writes `y.ptr.cast_
    /// const()` and could not if `ptr` were already `*const`. So the element
    /// names both carriers and the mark picks the one its direction needs.
    ///
    /// # What the mark did NOT unify, and cannot
    ///
    /// The ELEMENT. This doc used to claim that *"`In<bf16>` gets a `*const
    /// bf16` on CUDA and a `Buf` on a shader plane, from the same signature"*,
    /// and the type system does not permit it: `Read` is ONE associated type
    /// with no backend parameter, so `kernels_cuda::jit::abi::bf16` resolves
    /// it to a pointer on every plane and [`crate::shader::bf16`] resolves it
    /// to a binding index on every plane. No signature is shared, and none
    /// ever was -- no shader routine has ever written `In<bf16>` meaning
    /// CUDA's.
    ///
    /// What the marks unified is the DIRECTION, and that is the whole of it.
    /// The two `bf16`s are deliberately spelled alike -- one element, one
    /// word, whichever file you are in -- and that is a convention this trait
    /// enforces nothing about. Making it a guarantee would take an
    /// `Elem<B: Backend>`, which puts a backend parameter on every mark.
    type Read: Copy;
    /// What a launch that WRITES it is handed. [`Self::Read`]'s counterpart.
    type Write: Copy;

    /// This carrier, advanced by `elems` elements.
    ///
    /// # Why the element has to say
    ///
    /// Because a window is `start * pitch` elements along, and how you go
    /// along depends on what the carrier IS. CUDA offsets a pointer; a shader
    /// plane binds a whole buffer and its handle does not move, so the window
    /// has to reach the shader as a scalar instead -- which is what those
    /// planes already do, and why their handles answer with themselves.
    ///
    /// # Safety
    ///
    /// On a pointer carrier the result must stay inside the allocation. Every
    /// caller here proves that first: [`In::window`] bounds `start` against
    /// the operand's own rows before it asks.
    unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read;

    /// [`Self::advance_read`], for the writing carrier.
    ///
    /// # Safety
    ///
    /// [`Self::advance_read`]'s.
    unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write;

    /// How the device text spells a `const` pointer to this.
    const CPP_CONST: &'static str;
    /// How it spells a mutable one.
    const CPP_MUT: &'static str;
    /// The [`Ty`] a read binds as.
    const TY_CONST: Ty;
    /// The [`Ty`] a write binds as.
    const TY_MUT: Ty;
}

// THE POINTEES `ptr_abi!` DECLARES, IMPLEMENTED HERE BECAUSE THE ORPHAN RULE
// PUTS THEM HERE. `kernels-cuda` cannot write `impl kernels::Elem for f32` --
// neither the trait nor the type is local to it -- so the primitive pointees
// are this crate's, and the macro that owns the matching `Abi` impls asserts
// agreement with these at compile time instead of restating them.
//
// The local pointees (`bf16`, `f16`, `fp8_e4m3`, `u16_`) are `kernels-cuda`'s
// own types and it implements this trait for them there, where the orphan
// rule allows it and where their C++ spellings already live.
macro_rules! prim_elem {
    ($t:ty, $cc:literal, $cm:literal, $tc:ident, $tm:ident) => {
        impl Elem for $t {
            // A POINTEE'S CARRIERS ARE ITS TWO POINTERS. This is the CUDA
            // shape, and the shader planes give their handles `Read = Write =
            // Self` instead -- a binding index has no second form.
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                // SAFETY: the trait's obligation, forwarded to the caller.
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                // SAFETY: as above.
                unsafe { write.add(elems) }
            }
            const CPP_CONST: &'static str = $cc;
            const CPP_MUT: &'static str = $cm;
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }
    };
}

prim_elem!(i32, "const ::std::int32_t*", "::std::int32_t*", I32s, I32sMut);
// `BufMut` AND NOT `I64sMut`, WHICH IS THE `ptr_abi!` LINE'S OWN ASYMMETRY:
// nothing in the tree writes an `int64_t*` parameter, so the mutable
// direction falls back to the untyped buffer. The cross-check in
// `kernels-cuda/src/jit/abi.rs` is what keeps this odd pair honest rather
// than a reader's memory.
prim_elem!(i64, "const ::std::int64_t*", "::std::int64_t*", I64s, BufMut);
prim_elem!(i8, "const ::std::int8_t*", "::std::int8_t*", I8s, I8sMut);
prim_elem!(u32, "const ::std::uint32_t*", "::std::uint32_t*", U32s, U32sMut);
prim_elem!(u8, "const ::std::uint8_t*", "::std::uint8_t*", U8s, U8sMut);
prim_elem!(u16, "const ::std::uint16_t*", "::std::uint16_t*", U16s, U16sMut);
prim_elem!(f32, "const float*", "float*", F32s, F32sMut);
// THE OPAQUE POINTEE, AND IT IS A REAL ONE. §5 of the spec says the case that
// would earn `In<N, Table<Ptr>>` is a non-tensor buffer in operand position,
// *"of which there are currently zero"* -- but `void*` in ELEMENT position is
// not that case: it is a buffer whose element type the STATEMENT did not
// state, which is a transport gap and not a different kind of operand.
prim_elem!(core::ffi::c_void, "const void*", "void*", Buf, BufMut);

// THE POINTER-POINTEES: `void**` AND ITS TYPED SIBLINGS.
//
// §5 of the spec says `In<N, Table<Ptr>>` opens when a non-tensor buffer
// appears in operand position, *"of which there are currently zero"*. There
// are eight: `moe::build_moe_ptrs_aligned`'s six operands and two `gemm`
// activation tables are `const T**`/`T**`, whose POINTEE is itself a pointer.
//
// They need no new wrapper. A pointee is a thing an address points at, and a
// pointer is one of those; what the case actually needed was `Elem` impls,
// which is what these are. `ptr_abi!` already declares the `Abi` pairs
// (`abi.rs:469` onward, with the comment that `CPP` must spell
// `const bf16**` and not the untyped `const void**`), and `elem_agrees!`
// cannot cross-check these because the `Abi` for `*const *const T` is keyed
// on a type this crate cannot name.
macro_rules! ptr_elem {
    ($t:ty, $cc:literal, $cm:literal, $tc:ident, $tm:ident) => {
        impl Elem for $t {
            type Read = *const $t;
            type Write = *mut $t;

            unsafe fn advance_read(read: Self::Read, elems: usize) -> Self::Read {
                // SAFETY: the trait's obligation, forwarded to the caller.
                unsafe { read.add(elems) }
            }

            unsafe fn advance_write(write: Self::Write, elems: usize) -> Self::Write {
                // SAFETY: as above.
                unsafe { write.add(elems) }
            }
            const CPP_CONST: &'static str = $cc;
            const CPP_MUT: &'static str = $cm;
            const TY_CONST: Ty = Ty::$tc;
            const TY_MUT: Ty = Ty::$tm;
        }
    };
}

ptr_elem!(*const core::ffi::c_void, "const void* const*", "const void**", BufArray, BufArrayOut);
ptr_elem!(*mut core::ffi::c_void, "void* const*", "void**", BufArrayMut, BufArrayOutMut);
ptr_elem!(*const u8, "const ::std::uint8_t* const*", "const ::std::uint8_t**", BufArrayOut, BufArrayOut);
ptr_elem!(*const i32, "const ::std::int32_t* const*", "const ::std::int32_t**", BufArrayOut, BufArrayOut);

/// What this launch touches of an allocation.
///
/// A [`Layout`] belongs to the address and is never absent; a region is one
/// view of it, produced by applying a `(start, count)` and checked where it is
/// applied. Two different facts.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region<P> {
    /// The device address this view starts at.
    pub ptr: P,
    /// Rows in it.
    pub rows: i32,
    /// Elements per row — always stated, because a region that could not
    /// state one was refused rather than built.
    pub width: i32,
    /// The distance between rows. See [`Stride`].
    pub stride: Stride,
}

impl<P> Region<P> {
    /// Rows times width, saturating.
    ///
    /// The product a body writes when it wants an element count, kept off
    /// [`Extent`] for the reason that type gives: a width of zero is a stated
    /// unknown and a product would collapse it into a stated zero. A region
    /// cannot have a zero width -- that is what building one checks -- so the
    /// product is safe HERE and was not safe there.
    #[must_use]
    pub const fn elements(&self) -> i32 {
        self.rows.saturating_mul(self.width)
    }
}

// ── THE VIEW API ──────────────────────────────────────────────────────────
//
// `all()` ABSORBS THE GUARD: around 320 `.width` reads in `kernels-cuda` were
// preceded by a hand-written width check, every one the same -- a launch over
// a pitch of nothing is not a launch. Made where the view is built, it stops
// being 320 opportunities to forget.
//
// It TAKES the refusal's word rather than inventing one: the sites it replaces
// name the FACT that was missing, and a generic message would say only which
// operand failed.

impl<E: Elem> In<E> {
    /// This operand's view over a row count the CALLER supplies: the half of
    /// a region a signature can state, with the other half supplied where it
    /// is known. The other half is `self.width`, which the plane fills where it
    /// states a rectangle and leaves zero where it does not.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::Read>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows, width: self.width, stride: Stride(self.width) })
    }
    /// A WINDOW into the operand: `count` rows starting at row `start`.
    ///
    /// The offset is `start * stride * size_of::<E>()`, which needs the
    /// element type in the wrapper; before that a windowed view had to be
    /// handed the element size, and `gemm/lora.rs` wrote the arithmetic by
    /// hand with `* 2` for `bf16` spelled as a literal.
    ///
    /// Named `window` and not `rows` because [`In`] has a `rows` FIELD, and a
    /// reader looking at `x.rows` could not tell which they were getting.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what` when the statement gave no row width,
    /// and [`Refusal::Wide`] when the window runs past the operand's rows.
    pub fn window(
        &self,
        start: u32,
        count: i32,
        what: &'static str,
    ) -> Result<Region<E::Read>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        let end = i64::from(start).saturating_add(i64::from(count.max(0)));
        if end > i64::from(self.rows) {
            return Err(Refusal::Wide { what, at: end, max: i64::from(self.rows) });
        }
        // SAFETY: the bound above proves `start` is within the operand's own
        // rows, and `width` is its pitch, so the product is an offset the
        // allocation covers.
        // SAFETY: the bound above proves the offset lies inside the operand.
        let ptr = unsafe { E::advance_read(self.ptr, start as usize * self.width as usize) };
        Ok(Region { ptr, rows: count, width: self.width, stride: Stride(self.width) })
    }

    /// This launch's whole view of the operand.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what` when the statement gave no row
    /// width. That is `stated_width`'s refusal, made where the view is built
    /// instead of at each reader.
    pub fn all(&self, what: &'static str) -> Result<Region<E::Read>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows: self.rows, width: self.width, stride: Stride(self.width) })
    }
}

impl<E: Elem> Out<E> {
    /// This operand's view over a row count the CALLER supplies. See
    /// [`In::over`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows, width: self.width, stride: Stride(self.width) })
    }

    pub fn all(&self, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows: self.rows, width: self.width, stride: Stride(self.width) })
    }
}

// THE MIDDLE PAIR HAVE A WIDTH AND NO ROW COUNT, so a view built from one
// carries the rows the CALLER knows.
// ── THE LAYOUT IS REACHABLE FROM THE POINTER ─────────────────────────────
//
// An accessor and not a FIELD: `In { ptr, rows, width }` is written by hand at
// 118 sites, and a fourth field would rewrite all of them to carry a value
// derived from two they already hold.
//
// What it buys is the seam. When the transport widens to carry real dims and
// strides, `layout()` stops calling `packed` and starts reading what arrived,
// inside these four impls, with no signature moving.

impl<E: Elem> In<E> {
    /// The allocation's shape, as much of it as the transport delivered.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}

impl<E: Elem> Out<E> {
    /// See [`In::layout`].
    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}

// THE MIDDLE PAIR CLAIM A WIDTH AND NO ROW COUNT, so their layout says zero
// rows rather than inventing one. A row count is the FIRE's and a width is the
// STATEMENT's -- E5's asymmetry -- and these two wrappers exist precisely
// because a signature could not accept the half that is true.
// The forwarding, written out once per wrapper rather than by a macro.
//
// A `macro_rules!` over the six was the first draft and was deleted: three of
// them are generic over a type, one over a const AND a type, one over a
// marker, and one over nothing at all, so the macro needed a different arm
// per wrapper and saved no lines while hiding which `Provenance` each one
// takes. The impls are mechanical; that is not the same as being repetitive.

// THE FIVE MARKS' IMPLS.
//
// Each states its `Claim` and nothing about an index: the carrier states the
// `Ty`, the spelling and how the value is recovered, and the mark states which
// KIND of slot it wants. `resolve` hands out the numbers, because only the
// signature knows the order and a type cannot see it.
//
// THE COMPOSITION RULE, stated once because it is the whole of what replaced
// fifteen wrappers: A MARK'S SOURCE IS ITS OWN SLOT, OR THE QUESTION ITS KEY
// NAMES. A mark that claims a slot -- `In`, `Out`, `InOut`, `Weight` -- has
// its position for a source and needs no key; `Env<T, K>` has no position and
// takes its source, and its PROVENANCE, from `K`.
//
// A CHAIN IS THE KEY'S, NOT A WRAPPER'S. `keys::WindowOrNone` is
// `Or(Named("window_left"), Lit(-1))` in one place, where `Param<4, Env<i32,
// keys::NoSlidingWindow>>` spelled the same chain at twenty-one signatures --
// twenty-one chances to spell a different fallback for one question. Nesting
// carriers said WHERE to look before WHO answers; a key says who answers, and
// where is that answerer's business.

impl<B: Backend, E: Elem> Arg<B> for In<E>
where
    E::Read: Arg<B>,
{
    // THE ELEMENT STATES THE TYPE, THE MARK THE DIRECTION -- which is what
    // took `*const`/`*mut` and `Buf`/`BufMut` out of every signature.
    const TY: Ty = E::TY_CONST;
    // THE ELEMENT'S C++ WORD, OR THE CARRIER'S OWN. CUDA's elements name a
    // pointer spelling and a shader plane's name none -- a shading language's
    // spelling belongs to the `Lang`, which only the carrier's own `Arg` impl
    // can see. Taking the first non-empty gives every plane the spelling it
    // actually has, instead of the empty string three of them used to derive.
    const SPELLING: &'static str = spelling(E::CPP_CONST, <E::Read as Arg<B>>::SPELLING);
    // NOT the slot -- `resolve` supplies that. What survives here is the
    // CARRIER's source, which the chain needs as its second half.
    const SOURCE: Option<crate::Source> = <E::Read as Arg<B>>::SOURCE;
    const CLAIM: Claim = Claim::In;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <E::Read as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = extent_of::<B>(value)?;
        Ok(In { ptr, rows, width })
    }
}

impl<B: Backend, E: Elem> Arg<B> for Out<E>
where
    E::Write: Arg<B>,
{
    const TY: Ty = E::TY_MUT;
    const SPELLING: &'static str = spelling(E::CPP_MUT, <E::Write as Arg<B>>::SPELLING);
    const SOURCE: Option<crate::Source> = <E::Write as Arg<B>>::SOURCE;
    const CLAIM: Claim = Claim::Out;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <E::Write as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = extent_of::<B>(value)?;
        Ok(Out { ptr, rows, width })
    }
}

impl<B: Backend, E: Elem> Arg<B> for InOut<E>
where
    E::Write: Arg<B>,
{
    const TY: Ty = E::TY_MUT;
    // `Declared`, WITH `Out` AND NOT WITH `In`, and the arity rule is why.
    // `arity_problem` reads this to decide whether the parameter counts
    // against the operands the statement places or the results it declares.
    // An aliased buffer is ONE address the statement placed once; counting it
    // on both sides would make every routine that aliases read one operand
    // more than the statement carries. The input slot it also wears is an
    // ALIASING fact, which `Source::Alias` carries and the allocator reads.
    const SPELLING: &'static str = spelling(E::CPP_MUT, <E::Write as Arg<B>>::SPELLING);
    const SOURCE: Option<crate::Source> = <E::Write as Arg<B>>::SOURCE;
    const CLAIM: Claim = Claim::InOut;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <E::Write as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = extent_of::<B>(value)?;
        Ok(InOut { ptr, rows, width })
    }
}

// THE FOURTH MARK'S IMPL, AND THE ONLY ONE WHOSE CLAIM IS NOT A CONSTANT OF
// THE MARK. `ConstRun` is asked, and a `Tensor<E>` answers `Claim::Weight`
// where a scalar answers `Claim::Param` -- which is what makes one mark serve
// the weight run and the params run without a second word for it.
impl<B: Backend, C: ConstRun> Arg<B> for Const<C>
where
    C::Held: Arg<B>,
{
    const TY: Ty = C::TY;
    const SPELLING: &'static str = <C::Held as Arg<B>>::SPELLING;
    const SOURCE: Option<crate::Source> = <C::Held as Arg<B>>::SOURCE;
    const CLAIM: Claim = C::RUN;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <C::Held as Arg<B>>::unpack(value, at).map(|v| Const { v })
    }
}

/// AN OPERAND THAT MAY NOT BE THERE, SPELLED IN THE LANGUAGE'S OWN OPTIONAL.
///
/// `Option<Const<Tensor<f32>>>` is a bias a checkpoint may or may not carry;
/// `Option<In<Tensor<bf16>>>` is an input a statement may or may not place.
/// The mark keeps saying which run and which direction, and the `Option` says
/// the one thing left — whether it arrived.
///
/// # WHY THIS REPLACED A TYPE PER SPELLING
///
/// CUDA carried `MaybeConst<T>`, which was `Option<NonNull<T>>` inside a
/// newtype, and it existed for a reason that has since gone: back when a
/// nullable `const` pointer had to be told apart from a nullable mutable one
/// by its TYPE, because there was no mark to say the direction. `Const` says
/// it now, so `Const<Tensor<MaybeConst<f32>>>` stated const twice and absence
/// in a spelling nothing else in the tree used — 22 sites of `MaybeConst`
/// beside 9 of a bare `Option<NonNull<T>>` for the same idea.
///
/// It also could not reach the other three marks. A nullable INPUT had no
/// spelling at all, because `MaybeConst` was a const pointer by construction.
/// Wrapping the mark instead of the pointee is what makes one word serve all
/// four.
///
/// # The two halves, and which side answers each
///
/// The signature says *may be absent*; only the backend knows what an absent
/// VALUE looks like, so [`Backend::is_absent`] and [`Backend::absent`] answer
/// A plane that has no such value never produces one, its `is_absent` is
/// `false`, and every `Option` there unpacks as `Some` — which is what a plane
/// whose binder cannot mint an absence should say.
impl<B: Backend, M: Arg<B>> Arg<B> for Option<M> {
    const TY: Ty = M::TY;
    const SPELLING: &'static str = M::SPELLING;
    const SOURCE: Option<crate::Source> = M::SOURCE;
    const CLAIM: Claim = M::CLAIM;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        if value.is_absent() {
            return Ok(None);
        }
        M::unpack(value, at).map(Some)
    }
}

/// The same, on the way out: an operand a body did not get is re-emitted as
/// the backend's own absent value.
///
/// A plane with no such value cannot be holding a `None` — nothing could have
/// produced it — so the `unwrap_or_else` is unreachable rather than lenient,
/// and it panics with the sentence that says why.
impl<V: Absent, M: Bind<V>> Bind<V> for Option<M> {
    fn arg(self) -> V {
        match self {
            Some(m) => m.arg(),
            None => V::absent().expect(
                "a body holds `None` for an operand on a plane whose binder cannot mint one",
            ),
        }
    }
}


/// The element's own C++ word, or the carrier's, whichever was written down.
///
/// [`Arg::SPELLING`] reserves the empty string for *"this backend has not
/// written one down"*, so taking the first non-empty is exactly "ask the one
/// that knows".
const fn spelling(elem: &'static str, carrier: &'static str) -> &'static str {
    if elem.is_empty() { carrier } else { elem }
}

// `or_null` STOOD HERE -- *"the carrier's source, with the absent case given a
// shape"*, wrapping an `Option<Source>` into a `Source::Or` chain's second
// half because that variant holds a reference and needs a VALUE first. Its own
// doc said the `None` arm was never read; the whole function is never called
// now, and the two `_or_null` FACTS in `keys.rs` are unrelated -- they are
// named keys a driver answers, not this.

// THE TWO NAMED WEIGHT CHAINS, AS `static`s BECAUSE `Source::Or` HOLDS
// REFERENCES. `Facts` carries `w_named` and `w_named2` and nothing else, so a
// third weight has no NAME to be reached by -- only the positional bank. That
// used to be a `compile_error!` in the macro; here it is the shorter arm of
// the match below.
static NAMED_W0: crate::Source =
    crate::Source::Named(<crate::keys::NamedWeight as crate::keys::Fact>::KEY);
static NAMED_W1: crate::Source =
    crate::Source::Named(<crate::keys::NamedWeight2 as crate::keys::Fact>::KEY);
static BANK_0: crate::Source = crate::Source::Slot(crate::Kind::Weight, 0);
static BANK_1: crate::Source = crate::Source::Slot(crate::Kind::Weight, 1);

/// The launch's rectangle at `at`, or zeros on a plane that states none.
///
/// A PLANE WITH NO REGION SHAPE IS NOT A REFUSAL. `In` used to propagate
/// `B::region`'s error, which is why the handle planes needed a second pair of
/// marks: their binders bind addresses only and every operand would have
/// refused. Zero is already this crate's word for *"the statement gave no
/// width"* -- [`Region`] refuses on it and `Layout::packed(0, 0)` is a legal
/// empty -- so the absent case lands on the value every reader already checks.
fn extent_of<B: Backend>(value: &B::Value) -> Result<Extent, Refusal> {
    match B::region(value) {
        Ok(e) => Ok(e),
        Err(Refusal::Absent { .. } | Refusal::Unstated { .. }) => Ok(Extent { rows: 0, width: 0 }),
        Err(e) => Err(e),
    }
}

/// Hand out the slot numbers, in signature order.
///
/// THE ONE PLACE AN INDEX IS DECIDED. Every mark says which KIND of slot it
/// wants and nothing more; the order they appear in is the order the statement
/// placed them, so the running counters ARE the indices. That is what the
/// measurement said before the const generics came off: at 172 of 185 CUDA
/// routines the hand-written number already equalled this count, and all
/// thirteen exceptions were operands an alias had hidden.
///
/// `carriers` rides along because a mark's source may be a CHAIN -- the slot
/// if the statement fills it, the carrier's own question if it does not.
#[must_use]
pub const fn resolve<const N: usize>(
    claims: [Claim; N],
    carriers: [Option<crate::Source>; N],
) -> [Option<crate::Source>; N] {
    let mut out = [None; N];
    let (mut ins, mut outs, mut weights) = (0u8, 0u8, 0u8);
    // ONE COUNTER FOR THE PARAMS RUN, NOT TWO. `Claim::Param` and
    // `Claim::ParamF32` are two READINGS of one channel -- the run is a
    // `Vec<u32>` and the bits are the value -- so a float and an integer
    // constant that follow one another occupy slots `n` and `n + 1`, which is
    // what `model-dsl` writes and what `Handles::param`/`param_f32` read back.
    let mut params = 0u8;
    let mut i = 0;
    while i < N {
        out[i] = match claims[i] {
            Claim::Fixed => carriers[i],
            Claim::In => {
                let at = ins;
                ins += 1;
                Some(crate::Source::Slot(crate::Kind::In, at))
            }
            Claim::Out => {
                let at = outs;
                outs += 1;
                Some(crate::Source::Slot(crate::Kind::Out, at))
            }
            // BOTH COUNTERS MOVE, which is the whole of what the mark says:
            // the statement placed an operand here AND declared a result here,
            // at one address. `Source::Alias` carries the pair so the
            // allocator can give them one offset.
            Claim::InOut => {
                let (i_at, o_at) = (ins, outs);
                ins += 1;
                outs += 1;
                Some(crate::Source::Alias(i_at, o_at))
            }
            // THE PARAMS RUN, WHICH `Param<N, T>` USED TO NUMBER BY HAND.
            // The slot is the mark's position among the params marks, exactly
            // as an operand's is among the operand marks -- which is what
            // closes the hole `vec![0, rows, cols]` was holding open in
            // `model-dsl`, a placeholder written so `rows` would land at 1.
            Claim::Param => {
                let at = params;
                params += 1;
                Some(crate::Source::Slot(crate::Kind::Param, at))
            }
            Claim::ParamF32 => {
                let at = params;
                params += 1;
                Some(crate::Source::Slot(crate::Kind::ParamF32, at))
            }
            Claim::Weight => {
                let at = weights;
                weights += 1;
                // The named bank first and the positional one after, in the
                // order the binder already tried them.
                match at {
                    0 => Some(crate::Source::Or(&NAMED_W0, &BANK_0)),
                    1 => Some(crate::Source::Or(&NAMED_W1, &BANK_1)),
                    n => Some(crate::Source::Slot(crate::Kind::Weight, n)),
                }
            }
        };
        i += 1;
    }
    out
}

/// A `fn` that can serve as a routine body.
///
/// `M` is the parameter tuple and exists only to disambiguate: without it the
/// blanket impls for two different arities would overlap, since a single `F`
/// could in principle implement `Fn` at both. It is inferred at every use and
/// never written.
pub trait KernelFn<B: Backend, M>: Copy {
    /// This signature's arguments, in the order it takes them.
    ///
    /// No `Source` column: this const is a blanket impl built from
    /// `$arg::TY, $arg::PROV` — entirely from parameter TYPES — and a type
    /// cannot supply a source. `#[source(..)]` is consumed at expansion, and
    /// positional inference has nowhere to live, since `*const T` is the same
    /// type at every position. Sources travel with
    /// [`Derivation`](crate::Derivation) instead.
    const ARGS: &'static [Ty];
    /// WHICH of the environment's questions each argument is, where a type
    /// says. See [`Ask`].
    const SOURCES: &'static [Option<crate::Source>];
    /// The same arguments as the backend's shader language spells them.
    const SPELLING: &'static [&'static str];

    /// Unpack `args` against the signature and run the body.
    ///
    /// # Errors
    ///
    /// Whatever the body refuses, or [`Refusal::Arity`] / [`Refusal::Kind`]
    /// if the list does not fit the signature.
    fn invoke<'x>(self, ctx: &'x B::Ctx<'x>, args: &[B::Value]) -> Result<(), Refusal>;
}

/// Stamp [`KernelFn`] for one arity.
macro_rules! impl_kernel_fn {
    ($(($arg:ident, $at:tt)),* $(,)?) => {
        impl<B: Backend, F, $($arg: Arg<B>),*> KernelFn<B, ($($arg,)*)> for F
        where
            F: for<'x> Fn(&'x B::Ctx<'x>, $($arg),*) -> Result<(), Refusal> + Copy,
        {
            const ARGS: &'static [Ty] = &[$($arg::TY),*];
            // THE ONE PLACE THE INDICES ARE DECIDED. A mark carries a `Claim`
            // and no number, so the column cannot be a per-type map any more:
            // `resolve` walks the claims IN SIGNATURE ORDER and hands out the
            // slots. See [`resolve`] for why the order is the answer.
            const SOURCES: &'static [Option<crate::Source>] =
                &const { resolve([$($arg::CLAIM),*], [$($arg::SOURCE),*]) };
            const SPELLING: &'static [&'static str] = &[$($arg::SPELLING),*];

            fn invoke<'x>(self, ctx: &'x B::Ctx<'x>, args: &[B::Value]) -> Result<(), Refusal> {
                // Fully qualified because one `F` may be a routine for more
                // than one backend, which leaves `Self::ARGS` ambiguous.
                let want = <Self as KernelFn<B, ($($arg,)*)>>::ARGS.len();
                if args.len() != want {
                    return Err(Refusal::Arity { want, got: args.len() });
                }
                self(ctx, $($arg::unpack(&args[$at], $at)?),*)
            }
        }
    };
}

// Arity 0 through 24. The ceiling is measured, not chosen: the widest live
// signature takes 24 arguments (CUDA's fused QKV decode dispatch, and MLA's
// bf16 prepare). A signature past the ceiling fails to compile at its
// `routine!` line, which is where it should.
impl_kernel_fn!();
impl_kernel_fn!((A0, 0));
impl_kernel_fn!((A0, 0), (A1, 1));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2), (A3, 3));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2), (A3, 3), (A4, 4));
impl_kernel_fn!((A0, 0), (A1, 1), (A2, 2), (A3, 3), (A4, 4), (A5, 5));
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23)
);

// ── The ceiling is 36, which is where `impl_kernel_fn!` stops ──
//
// Not a constraint: unfolding fa2's plan aggregate into flat named facts
// needed 28–35 arguments, so the macro was stamped further.

impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32),
    (A33, 33)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32),
    (A33, 33),
    (A34, 34)
);
impl_kernel_fn!(
    (A0, 0),
    (A1, 1),
    (A2, 2),
    (A3, 3),
    (A4, 4),
    (A5, 5),
    (A6, 6),
    (A7, 7),
    (A8, 8),
    (A9, 9),
    (A10, 10),
    (A11, 11),
    (A12, 12),
    (A13, 13),
    (A14, 14),
    (A15, 15),
    (A16, 16),
    (A17, 17),
    (A18, 18),
    (A19, 19),
    (A20, 20),
    (A21, 21),
    (A22, 22),
    (A23, 23),
    (A24, 24),
    (A25, 25),
    (A26, 26),
    (A27, 27),
    (A28, 28),
    (A29, 29),
    (A30, 30),
    (A31, 31),
    (A32, 32),
    (A33, 33),
    (A34, 34),
    (A35, 35)
);

/// A routine body with its types erased — what [`call`](Routine::body) goes
/// through.
pub type Body<B> =
    for<'x> fn(&'x <B as Backend>::Ctx<'x>, &[<B as Backend>::Value]) -> Result<(), Refusal>;

/// One routine's table row, and the body behind it.
///
/// The first three fields are derived from the signature. The last three are
/// STATED: they are facts about how a trace may use the routine, which no
/// signature carries.
pub struct Routine<B: Backend> {
    /// The routine's name, which is the `fn`'s name — or, for a generic body,
    /// that name with its instantiation joined on.
    pub name: &'static str,
    /// What a trace prefixes this routine's symbol with.
    ///
    /// DERIVED FROM `module_path!()` AT THE `fn`, which is what retired
    /// `Family`. A family was a container holding one namespace and a group of
    /// routines, and the namespace was already computed from the module the
    /// group was declared in -- so attaching it per-routine left the container
    /// holding nothing but the list.
    ///
    /// It also settles a question the grouping had to explain: `attn::xqa` and
    /// `attn::fa2` were two families sharing one namespace, described in a
    /// comment on `FAMILIES`. `namespace` takes the first segment after the
    /// crate root, so both simply answer `"attn"`.
    pub namespace: &'static str,
    /// Its arguments, derived from the signature.
    pub args: &'static [Ty],
    /// WHICH of the environment's questions each argument is, where a type
    /// says. See [`Ask`] and [`KernelFn::SOURCES`].
    pub sources: &'static [Option<crate::Source>],
    /// The same arguments as the backend's shader language spells them,
    /// derived from the signature. See [`Arg::SPELLING`].
    pub spelling: &'static [&'static str],
    /// The erased body.
    pub body: Body<B>,
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// This statement participates in the depth-prefix plan.
    pub depth_prefix_plan: bool,
    /// What `#[routine]` read off the launcher's signature, if the row says.
    ///
    /// It IS derived — that is [`Derived`]'s whole subject — and it no longer
    /// has to be named. `routine!` reads it through `<$f as
    /// kernels::Derivation>::DERIVED`, the impl `#[routine]` emits against a
    /// marker wearing the launcher's own name, so the column arrives with the
    /// body instead of through an uppercased const the caller had to spell.
    ///
    /// An empty column still reads as `&[]`, exactly like a `None`
    /// [`Derived::source`] does at the operand level: *"the row has not
    /// said"*, not *"there is nothing to say"*. Exactly one row means it —
    /// `attn::qkv_decode_fused_dispatch`, which says so with `uncolumned`.
    pub derived: &'static [Derived],
    /// No trace may state this symbol: it is a body another routine calls.
    ///
    /// NOT `untraced`, which is a different fact wearing a similar word.
    /// `#[routine(untraced)]` says *"no source column, and a string dispatch
    /// must refuse"* — the eight FA2 launchers say it today and a trace names
    /// every one of them. This says *"the trace vocabulary does not contain
    /// this name"*, which no column can show either way.
    ///
    /// What it is for: `gemm::act_x_wt_bf16` takes `beta` as a parameter and
    /// `gemm::act_x_w` is the forwarder that pins it at `0.0`. Both carry full
    /// columns, so a resolver deriving membership from the column alone admits
    /// the body — same operands, same buffers, a `beta` off the statement
    /// instead of the symbol, and nothing downstream to notice. Thirty-one
    /// routines are in that position.
    pub internal: bool,
    /// The facts this routine's BODY asks the environment for.
    ///
    /// Not parameters: `ctx.ask::<f32, keys::RmsEps>()` is a call, so it has no
    /// entry in [`Self::sources`] to be `None` and a reader walking the column
    /// cannot see it. `#[routine]` scans the turbofishes and lists them here,
    /// which is what lets a driver answer *"can I supply everything this
    /// routine will ask for"* before a fire rather than during one.
    ///
    /// It MISSES a fact asked inside a helper — a syntactic guarantee where the
    /// parameter run has a type-system one, accepted deliberately.
    pub asked: &'static [&'static str],
    /// This routine refuses a statement the driver JOINED into another.
    ///
    /// See [`Self::no_join`](Routine::no_join) for why a precondition is a row
    /// fact rather than a [`crate::Source`].
    pub no_join: bool,
    /// The DRIVER fires this by a typed call, not the operand column.
    ///
    /// A text may name it -- that is what separates this from
    /// [`Self::internal`] -- and what runs it is the driver's own dispatch,
    /// because the body needs something a query-only binder must not hand out:
    /// a cuBLAS handle, a fire-scoped state pointer, a resolver-owned aux slot.
    ///
    /// **Not derivable from the column's shape, which is why it is stated.**
    /// The eight rows that said this in a table ranged from no column at all
    /// (`comm::all_reduce_bf16`) through a deliberately empty source run
    /// (`gemm::lora_qkv_correction`) to a column that resolves at every
    /// position (`moe::moe_grouped_gemm_bf16`) -- and that last one is the
    /// case that matters: a driver op and a routine share the name, so a
    /// resolver reading only the column would bind the operands correctly and
    /// run a different implementation.
    pub driver: bool,
}

/// The first path segment after the crate root, out of a `module_path!()`.
///
/// `kernels_cuda::attn::fa2` -> `attn`. Written as a byte scan over
/// `as_bytes()` because this runs in a `const` initialiser, where
/// `str::split` and `Iterator::nth` are not available.
///
/// It lived in `kernels-cuda` as `segment_after_crate`, private to the
/// `Family` it fed. The routines carry their own namespace now, so it belongs
/// where the routines are defined and the two `unsafe` blocks are the same
/// two: a range on `::` boundaries, both ends ASCII, so no multi-byte
/// sequence can be cut.
///
/// # Panics
///
/// If `module_path` names the crate root itself, which has no family segment
/// to take -- every symbol it offered would be a bare routine name that no
/// trace can state. A `const` call makes that a build failure.
#[must_use]
pub const fn namespace(module_path: &'static str) -> &'static str {
    let bytes = module_path.as_bytes();
    let mut start = 0;
    while start + 1 < bytes.len() {
        if bytes[start] == b':' && bytes[start + 1] == b':' {
            start += 2;
            break;
        }
        start += 1;
    }
    assert!(
        start > 0 && start < bytes.len(),
        "a routine at the crate root has no namespace"
    );
    let mut end = start;
    while end + 1 < bytes.len() {
        if bytes[end] == b':' && bytes[end + 1] == b':' {
            break;
        }
        end += 1;
    }
    if end + 1 == bytes.len() {
        end = bytes.len();
    }
    // SAFETY: `start..end` lies on `::` boundaries or on the string's ends,
    // and `:` is ASCII, so the range begins and ends on a char boundary.
    unsafe {
        core::str::from_utf8_unchecked(core::slice::from_raw_parts(
            bytes.as_ptr().add(start),
            end - start,
        ))
    }
}

/// The `(output, input)` pairs one source column aliases.
///
/// DERIVED, where `in_place = &[(0, 0)]` used to be stated on the row. Free
/// rather than a method so that [`Routine`] and [`Declared`] share one
/// definition; both hold the same column.
///
/// Output first, as every reader expects.
#[must_use]
pub fn aliased(sources: &[Option<crate::Source>]) -> Vec<(u32, u32)> {
    sources
        .iter()
        .filter_map(|s| match s {
            Some(crate::Source::Alias(i, o)) => Some((u32::from(*o), u32::from(*i))),
            _ => None,
        })
        .collect()
}

impl<B: Backend> Routine<B> {
    /// This routine's trace symbol: `namespace::name`.
    ///
    /// `Family::symbol` was this, from the outside. It is the routine's own
    /// answer now, because the routine carries both halves.
    #[must_use]
    pub fn symbol(&self) -> String {
        format!("{}::{}", self.namespace, self.name)
    }

    /// Whether `symbol` names this routine.
    ///
    /// Compares without building the `String` that [`Self::symbol`] would,
    /// because this is asked once per launched op of every model that loads.
    #[must_use]
    pub fn answers(&self, symbol: &str) -> bool {
        symbol
            .strip_prefix(self.namespace)
            .and_then(|t| t.strip_prefix("::"))
            .is_some_and(|t| t == self.name)
    }

    /// Which of this routine's results share an address with an operand.
    ///
    /// See [`aliased`], and [`InOut`] for what replaced the stated column.
    #[must_use]
    pub fn in_place(&self) -> Vec<(u32, u32)> {
        aliased(self.sources)
    }

    /// This routine, marked as consuming its whole operand.
    #[must_use]
    pub const fn whole(mut self) -> Self {
        self.whole = true;
        self
    }

    /// This routine, marked as participating in the depth-prefix plan.
    #[must_use]
    pub const fn depth_prefix_plan(mut self) -> Self {
        self.depth_prefix_plan = true;
        self
    }

    /// This routine, marked as refusing a statement the driver JOINED.
    ///
    /// A precondition and not a source, which is why it is a row fact: every
    /// [`crate::Source`] FILLS a slot, and this requires two to be empty. The
    /// FA2 dispatches are the audience -- an aux value or a per-head reading
    /// changes the arithmetic rather than the operands, so a join binds right
    /// and computes wrong, and `arms/fa2.rs`'s `no_join_extras` was that check
    /// written eight times inside the arms it kept alive.
    #[must_use]
    pub const fn no_join(mut self) -> Self {
        self.no_join = true;
        self
    }

    /// This routine, marked as outside the trace vocabulary.
    ///
    /// A body other routines call, not a symbol a text may state. See
    /// [`Self::internal`](Routine::internal) for the pair that made it
    /// necessary.
    #[must_use]
    pub const fn internal(mut self) -> Self {
        self.internal = true;
        self
    }

    /// This routine, marked as fired by the driver rather than by its column.
    ///
    /// See [`Self::driver`](Routine::driver).
    #[must_use]
    pub const fn driver(mut self) -> Self {
        self.driver = true;
        self
    }

    /// This routine, with `#[routine]`'s operand column attached.
    ///
    /// Kept as a builder because `kernels::routine!` is backend-agnostic and
    /// the wrappers decide whether their launchers carry a column at all.
    /// CUDA's injects it; the shader backends have no `#[routine]` to read.
    #[must_use]
    pub const fn derived(mut self, operands: &'static [Derived]) -> Self {
        self.derived = operands;
        self
    }

    /// This routine, with its source column STATED rather than derived.
    ///
    /// # For rows that have no signature to derive one from
    ///
    /// [`untraced`] rows are the whole of the audience: their bodies are
    /// typed calls the driver makes, not `fn`s whose parameters are marks, so
    /// `sources` is empty and everything read out of it -- [`aliased`] above
    /// all -- comes back empty too. One row needs it not to:
    /// `comm::all_reduce_residual_rmsnorm_bf16` updates the residual IN PLACE
    /// and declares a result, which the row used to say beside it as
    /// `in_place = &[(0, 1)]`, and an allocator that does not hear it hands
    /// the launch a fresh buffer for a result the kernel never writes.
    ///
    /// Not a way back to stating what a signature says. A `#[routine]` row
    /// derives its column and this would be shadowed by it; the only caller
    /// is a row that has no column at all.
    #[must_use]
    pub const fn stating(mut self, sources: &'static [Option<crate::Source>]) -> Self {
        self.sources = sources;
        self
    }

    /// This routine, with the facts its body asks for listed.
    ///
    /// `#[routine]` is the only caller: the list is scanned off the body's
    /// turbofishes and arrives on the same marker the derived column does.
    #[must_use]
    pub const fn asking(mut self, asked: &'static [&'static str]) -> Self {
        self.asked = asked;
        self
    }
}

/// One routine's row with its backend forgotten.
///
/// The machinery is generic over [`Backend`], so three backends' `ROUTINES`
/// are three unrelated types and cannot be put in one list. This is the view
/// that can: the derived argument list and the three stated facts, which are
/// exactly the columns `.wiki/kernel-x/refactor-bigplan.md` §3's cross-backend
/// agreement gate compares. The body and everything device-shaped is left
/// behind on purpose — grids, tiers and entrypoint spellings are properly
/// per-backend.
// `PartialEq` AND NOT `Eq`: `sources` carries a `Source`, whose comparisons
// are all `==` and which nothing puts in a set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Declared {
    /// The routine's name.
    pub name: &'static str,
    /// What a trace prefixes it with. See [`Routine::namespace`].
    pub namespace: &'static str,
    /// Its arguments, derived from the signature.
    pub args: &'static [Ty],
    /// WHICH of the environment's questions each argument is, where a type
    /// says. See [`Ask`] and [`KernelFn::SOURCES`].
    pub sources: &'static [Option<crate::Source>],
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// This statement participates in the depth-prefix plan.
    pub depth_prefix_plan: bool,
    /// The parameter names and their NULLABILITY, in [`Self::args`] order.
    ///
    /// Carried here because a nullable operand is an OPTIONAL one: the
    /// statement need not place it, and a reader comparing the signature's
    /// operand count against the statement's has to know which of the two
    /// counts is allowed to be short. `Provenance::Either` said this before
    /// the marks, on a wrapper; the carrier says it now, and this is how it
    /// reaches a reader outside the crate.
    pub derived: &'static [crate::Derived],
}

impl Declared {
    /// Which of this row's results share an address with an operand.
    ///
    /// See [`aliased`].
    #[must_use]
    pub fn in_place(&self) -> Vec<(u32, u32)> {
        aliased(self.sources)
    }
}

impl<B: Backend> Routine<B> {
    /// This row, with the backend forgotten, for a cross-backend comparison.
    #[must_use]
    pub const fn declared(&self) -> Declared {
        Declared {
            name: self.name,
            namespace: self.namespace,
            args: self.args,
            sources: self.sources,
            whole: self.whole,
            depth_prefix_plan: self.depth_prefix_plan,
            derived: self.derived,
        }
    }
}

impl<B: Backend> core::fmt::Debug for Routine<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Routine")
            .field("name", &self.name)
            .field("args", &self.args)
            .field("whole", &self.whole)
            .field("depth_prefix_plan", &self.depth_prefix_plan)
            .finish_non_exhaustive()
    }
}

/// The argument table of a routine `fn`, read off its signature.
///
/// The value is discarded: a `fn` item is a zero-sized type, and everything
/// wanted is in its type.
#[must_use]
pub const fn describe<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [Ty] {
    F::ARGS
}

/// The same signature's QUESTIONS, in the same order. See [`Ask`].
#[must_use]
pub const fn sources<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [Option<crate::Source>]
{
    F::SOURCES
}

/// The same signature, as the backend's shader language spells it.
#[must_use]
pub const fn spell<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [&'static str] {
    F::SPELLING
}

/// A symbol the DRIVER fires by path, declared without a statement's argument
/// list.
///
/// A `routine!` says *a trace statement binds this*, which is what forces
/// every parameter to be [`Arg`]. Some kernels take a communicator, a layer's
/// page geometry, a weight REPRESENTATION -- properties of the fire that no
/// statement mentions. They are still symbols a lowered model text may name,
/// so `check_plan` must be able to look them up at load.
///
/// It produces a [`Routine`] with `args` and `spelling` empty and a `body`
/// that REFUSES: reaching it means something dispatched by string a symbol
/// its driver calls by path. The `fn` is NAMED, so deleting it fails this
/// expansion rather than leaving a symbol nothing runs.
#[macro_export]
macro_rules! untraced {
    // THE FORM `#[routine(untraced)]` EMITS.
    ($backend:ty, $name:literal, $body:expr, namespace = $ns:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        // The point of naming `$body` rather than only stringifying it: a
        // declaration whose `fn` has been deleted is the defect this table was
        // rebuilt to make impossible.
        #[allow(dead_code)]
        fn names_a_real_fn() {
            let _ = $body;
        }
        fn by_path<'x>(
            _ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            _args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            ::core::result::Result::Err($crate::routine::Refusal::Absent {
                what: "a statement-bound body: this symbol is declared so a model \
                       text may name it, and fired by the driver through a typed \
                       call rather than by string",
            })
        }
        $crate::routine::Routine::<$backend> {
            name: $name,
            namespace: $ns,
            args: &[],
            sources: &[],
            spelling: &[],
            body: by_path,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            asked: &[],
            no_join: false,
            driver: false,
        }
        $(.$fact($($value)?))*
    }};
    ($backend:ty, $body:ident $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        // The point of naming `$body` rather than only stringifying it. A
        // declaration whose `fn` has been deleted is the defect this whole
        // table was rebuilt to make impossible, and a `stringify!` alone
        // would reintroduce it one macro later.
        #[allow(dead_code)]
        fn names_a_real_fn() {
            let _ = $body;
        }
        fn by_path<'x>(
            _ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            _args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            ::core::result::Result::Err($crate::routine::Refusal::Absent {
                what: "a statement-bound body: this symbol is declared so a model \
                       text may name it, and fired by the driver through a typed \
                       call rather than by string",
            })
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($body),
            args: &[],
            sources: &[],
            spelling: &[],
            body: by_path,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            asked: &[],
            no_join: false,
            driver: false,
        }
        $(.$fact($($value)?))*
    }};
}

/// One routine's row, from its `fn` and nothing else.
///
/// # Not a public surface
///
/// `#[routine]` is the only caller, and the only one there should be: it is
/// where a routine's name, namespace, facts and registration are decided
/// together. Four backends used to wrap this with their own [`Backend`] filled
/// in, so a membership list could name only the `fn`; there is no membership
/// list, so there is nothing to wrap.
///
/// Trailing facts are the `const` builders of [`Routine`], named:
/// `routine!(B, rope_bf16, whole)`.
///
/// `in_place` IS NOT AMONG THEM ANY MORE. It was, forty-five times, and it is
/// [`InOut`] now: the pairs derive from the parameter that wears both slots,
/// through [`crate::Source::Alias`], so the numbers cannot disagree with the
/// signature they index.
///
/// A generic body answers several symbols, as
/// `routine!(B, rope_bf16 = rope::<bf16, 256>)`; the name is written out
/// because `stringify!` would answer `rope::<bf16, 256>`, which no trace can
/// state. It is the one place a routine's name is typed by hand, and it is
/// where the instantiation is chosen, so the two cannot drift.
#[macro_export]
macro_rules! routine {
    // THE FORM `#[routine]` EMITS: a composed name, an instantiated body and
    // the namespace the attribute read off `module_path!()`.
    ($backend:ty, $name:literal, $body:expr, namespace = $ns:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: $name,
            namespace: $ns,
            args: $crate::routine::describe::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            asked: &[],
            no_join: false,
            driver: false,
        }
        $(.$fact($($value)?))*
    }};
    ($backend:ty, $name:ident = $body:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($name),
            // EMPTY, AND SAID SO. A row written by hand states no module, and
            // `namespace` is what `#[routine]` reads off `module_path!()` --
            // there is nothing here to read it off.
            namespace: "",
            args: $crate::routine::describe::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            asked: &[],
            no_join: false,
            driver: false,
        }
        $(.$fact($($value)?))*
    }};
    ($backend:ty, $body:ident $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        // A `fn` item is zero-sized, so this names `$body` without capturing
        // it -- which is what lets the shim be a plain `fn` pointer.
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($body),
            // EMPTY, AND SAID SO. See the arm above: a row written by hand
            // states no module, and there is nothing here to read one off.
            namespace: "",
            args: $crate::routine::describe::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            derived: &[],
            internal: false,
            asked: &[],
            no_join: false,
            driver: false,
        }
        $(.$fact($($value)?))*
    }};
}

// ── THE WINDOW'S BOUNDS CHECK, PINNED ─────────────────────────────────────
//
// `In::window` is Stage 6's payoff and Stage 7's substrate: an offset is
// `start * stride * size_of::<E>()`, and the element size arrived when the
// element became the wrapper's parameter. What the view adds over the hand
// arithmetic it replaces is the bound -- `quant::mxfp4_scales_to_marlin_e8m0`
// bounds-checks three ways on the DEVICE (`kernels/quant/mxfp4_marlin.cuh:174`)
// because nothing on the host could do it once.
//
// A view that refuses out of range is only useful if it actually refuses, and
// `Refusal` is not `PartialEq` in a const context, so the shape is matched
// rather than compared.
const _: () = {
    // The arithmetic the bound is made of, checked where it cannot drift from
    // the code above: a window ending past the operand's rows is `Wide`.
    let rows: i64 = 7;
    let start: i64 = 5;
    let count: i64 = 3;
    assert!(start + count > rows);
    // And one that fits is not.
    assert!(start + (count - 1) <= rows);
};

// ── Constructors, for the fixtures that build a row by hand ──


impl<E: Elem> In<E> {
    /// Wear the next input slot for this carrier, with no rectangle stated.
    pub const fn new(ptr: E::Read) -> Self {
        Self { ptr, rows: 0, width: 0 }
    }
}

impl<E: Elem> Out<E> {
    /// Wear the next result slot, with no rectangle stated.
    pub const fn new(ptr: E::Write) -> Self {
        Self { ptr, rows: 0, width: 0 }
    }
}

impl<E: Elem> InOut<E> {
    /// Wear both slots at one address, with no rectangle stated.
    pub const fn new(ptr: E::Write) -> Self {
        Self { ptr, rows: 0, width: 0 }
    }

    /// A WINDOW into the operand: `count` rows starting at row `start`. See
    /// [`In::window`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what` when the statement gave no row width,
    /// and [`Refusal::Wide`] when the window runs past the operand's rows.
    pub fn window(
        &self,
        start: u32,
        count: i32,
        what: &'static str,
    ) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        let end = i64::from(start).saturating_add(i64::from(count.max(0)));
        if end > i64::from(self.rows) {
            return Err(Refusal::Wide { what, at: end, max: i64::from(self.rows) });
        }
        // SAFETY: the bound above proves the offset lies inside the operand.
        let ptr = unsafe { E::advance_write(self.ptr, start as usize * self.width as usize) };
        Ok(Region { ptr, rows: count, width: self.width, stride: Stride(self.width) })
    }

    /// This operand's view over a row count the CALLER supplies. See
    /// [`In::over`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows, width: self.width, stride: Stride(self.width) })
    }

    /// This launch's whole view of the operand. See [`In::all`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn all(&self, what: &'static str) -> Result<Region<E::Write>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows: self.rows, width: self.width, stride: Stride(self.width) })
    }

    /// See [`In::layout`].
    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}

