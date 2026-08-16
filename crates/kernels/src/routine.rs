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

/// One backend's two concrete types, so the machinery can be written once.
///
/// The implementor is a marker: it is never constructed and carries no state.
pub trait Backend: Copy + 'static {
    /// A value bound to one argument — the backend's `ArgValue`.
    type Value: Copy;
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

    /// The shape the value at `at` carries, if it carries one.
    ///
    /// On the backend and not on [`Arg`] because nothing in this crate can
    /// look inside `Self::Value` — it is an associated type and `kernels`
    /// names no backend.
    ///
    /// # Errors
    ///
    /// [`Refusal::Kind`] when the value is not region-shaped —
    /// [`Refusal::Unstated`] when this backend has no region shape at all.
    fn region(value: &Self::Value, at: usize) -> Result<Extent, Refusal>;
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
    const PROV: Provenance = <i32 as Arg<B>>::PROV;
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

/// Who supplies an argument.
///
/// The distinction is not "who owns the memory" but "who can be asked for it
/// at trace time": a [`Provenance::Trace`] argument is stated by the program
/// being run, and an [`Provenance::Env`] one is a fact about the execution
/// environment — a position vector, a plan, a workspace — which the program
/// never names and the runtime always has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// The statement supplies it.
    Trace,
    /// The execution environment supplies it.
    Env,
    /// The statement supplies it IF it states one, and the environment has a
    /// standing answer if it does not.
    ///
    /// `Source::Or`'s spelling in a signature, and a third case rather than
    /// a shade of the other two: FA2's `o` is the statement's result on a text
    /// that names one and the guard-owned arena on a text that does not, and
    /// the SAME symbol serves both. A rule reading this column counts an
    /// `Either` argument as permitted, not required.
    ///
    Either,
}

/// Which SIDE of a statement an argument sits on — the operand question,
/// asked separately from the direction question.
///
/// The wrapper says which OPERAND, the pointer says which DIRECTION the kernel
/// drives it, and the two are independent facts. `In<0, *mut T>` occupies
/// input slot 0 and is written through; a rule partitioning by pointer
/// mutability reads it as a result the statement forgot to declare, which it
/// is not. Three routines spell that shape. The arity rule reads this SLOT for
/// anything a position wrapper claims and falls back to [`crate::Ty::binds`]
/// otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// No position wrapper claimed it — ask the type which way it binds.
    OfType,
    /// An operand the statement PLACES: an input or a weight.
    Placed,
    /// A result the statement DECLARES.
    Declared,
}

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
    /// Who supplies it.
    ///
    /// Defaults to [`Provenance::Trace`]; only [`Env`] and [`Aux`] override
    /// it. The position wrappers — [`In`], [`Out`], [`InOut`], [`Weight`],
    /// [`Bank`], [`Unbound`] — forward `T::PROV`, which is what makes them
    /// arity-inert: they answer WHERE an argument sits and nothing about who
    /// supplies it.
    const PROV: Provenance = Provenance::Trace;

    /// The question this type claims, or `None` when no type claims one.
    const SOURCE: Option<crate::Source> = None;
    /// Which side of the statement it sits on, when a position wrapper says.
    ///
    /// Defaults to [`Side::OfType`] — "nothing claimed a slot, read the
    /// type" — and is overridden by the six POSITION wrappers, which are the
    /// only things that know. See [`Side`] for what the answer is for.
    const SIDE: Side = Side::OfType;
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

/// The environment supplies this argument, not the statement.
///
/// A wrapper rather than a table column, so that the fact is stated exactly
/// where the argument is — in the signature — and derives from there.
#[derive(Clone, Copy, Debug)]
pub struct Env<T>(pub T);

impl<T> Env<T> {
    /// The wrapped argument.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> core::ops::Deref for Env<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<B: Backend, T: Arg<B>> Arg<B> for Env<T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    const PROV: Provenance = Provenance::Env;
    const SPELLING: &'static str = T::SPELLING;
    // Absence does not change WHICH question is asked, so the wrapper
    // forwards rather than shadowing what it wraps.
    const SOURCE: Option<crate::Source> = T::SOURCE;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Env)
    }
}

/// The statement's `N`th operand: its address AND the shape it was given.
///
/// A bare `*const T` derives `Source::In` of the next unused index, which is
/// right until a signature puts a pointer where the counting does not expect
/// one. AN INDEX IS A FACT, AND COUNTING IS HOW A DESIGN GUESSES ONE:
/// `residual_add` and `geglu_tanh` have identical parameter shapes and
/// identical `in_place` rows and need different indices, so no correction
/// derives both.
///
/// It carries `rows` and `width` because the statement gave them; tearing
/// them off is what put `#[source(OutWidth(0))] width: i32` in forty-seven
/// signatures. It does not `Deref`: with three fields, picking the address
/// would make `y.rows` and `*y` two kinds of access to one value.
///
#[derive(Clone, Copy, Debug)]
pub struct In<const N: usize, E: Elem> {
    /// The device address.
    pub ptr: *const E,
    /// Rows in this launch's rectangle.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// The statement's `N`th result. [`In`]'s counterpart, and fat for the same
/// reason.
#[derive(Clone, Copy, Debug)]
pub struct Out<const N: usize, E: Elem> {
    /// The device address.
    pub ptr: *mut E,
    /// Rows in this launch's rectangle.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// The routine's weight, named rather than positional.
///
/// Derives `WeightNamed`, not `Weight(n)`, and the difference is real: an
/// `OpKind::Launch` puts a weight in the operand list where it is positional,
/// while a semantic op like `OpKind::Rmsnorm` carries only a NAME on
/// `LaunchSpec::weight`. Reading only the first is what made gemma-4 refuse
/// at its PLE prologue for a statement that named one.
///
/// See [`Bank`] for the positional spelling.

#[derive(Clone, Copy, Debug)]
pub struct Weight<const N: usize, T: ?Sized> {
    /// The device address. A weight is READ, so the direction is settled.
    ///
    /// `T` is a POINTEE on CUDA — `Weight<0, bf16>` holds a `*const bf16`,
    /// spelled by `Elem` — and the bound argument type on the shader planes,
    /// where `Weight<0, Buf>` holds a handle with no const/mut pair.
    pub ptr: T,
}

/// A parameter with no source, said out loud.
///
/// For POINTERS only. A scalar with no source already derives none, so the
/// wrapper would be noise; a bare `*const T` is different — `derive_all`
/// counts it into the next input slot, so a driver-owned pointer that says
/// nothing is read as an operand the statement placed. This says nothing
/// loudly enough to stop that.

#[derive(Clone, Copy, Debug)]
pub struct Unbound<T> {
    /// The device address.
    pub ptr: T,
}

/// The positional weight bank: `b.args[spec.n_in + spec.n_out + N]`.
///
/// [`Weight`] and this are one word in English and two reads at runtime. A
/// statement saying `weights: [w]` puts `w` in `Facts::weight_named[0]`, which
/// [`Weight`] reads; a statement placing its weights positionally puts them
/// after every input and output, which this reads.

#[derive(Clone, Copy, Debug)]
pub struct Bank<const N: usize, E: Elem> {
    /// The device address. A bank is READ, so the direction is settled.
    pub ptr: *const E,
}

// `Rows`, `Width<S>` AND `mod slot` STOOD HERE, and a region deleted all
// three. `Rows` was `Facts::rows.count` as a parameter and `Width<slot::
// Out<0>>` was a parameter whose entire content was *"the shape of the
// parameter two lines up"*; `y.rows` and `y.width` are the same two numbers
// reached through the region that proves the launch HAS that result, which is
// rule E1: *"a body that wants a width asks the region it already holds."*
//
// ONE GUARD DIED WITH THEM AND HAD TO BE PUT BACK BY HAND. `Width<slot::
// Out<0>>` derived `OutWidth(0)`, which the binder answered with `?`; a
// region's width is `unwrap_or(0)`. `kernels-cuda/src/rope.rs`'s private
// `q_heads` is that refusal restored beside the division it protects, which
// is where it should have been all along.

/// The driver's `N`th aux slab for this layer.
///
/// # Why the index cannot be derived and must be written
///
/// The INDEX is not in the name and cannot be: `fact_of` keys on a spelling,
/// `Aux` carries a `u8`, and `dt` is the zeroth slot only by the join's
/// convention. The const comes FIRST because the index is the fact being
/// stated and the pointee is the plumbing.
///
/// [`Provenance::Env`] is not optional here: an aux slab is handed over by
/// the driver, no statement places it, and dropping it would change the arity
/// §6.2 checks.
#[derive(Clone, Copy, Debug)]
pub struct Aux<const N: usize, T>(pub T);

/// The `N`th scalar the statement carries — [`Kind::Param`]'s spelling.
///
/// Not `In<N, i32>`: [`In`]'s `N` indexes the operand run and this one indexes
/// `spec.params[]`. ONE INDEX OVER TWO ARRAYS shipped as a bug once already,
/// `Weight<1, _>` on `gemv_bf16`'s bias resolving against the wrong table.
///
/// A scalar needs the wrapper even though [`Unbound`] argued otherwise: an
/// unmarked scalar meant both "the statement states this" and "this is the
/// launcher's own constant". After this, unmarked means only the second.
///
/// [`Kind::Param`]: crate::Kind::Param
#[derive(Clone, Copy, Debug)]
pub struct Param<const N: usize, T>(pub T);

/// The `N`th param slot read as a FLOAT — [`Kind::ParamF32`]'s spelling.
///
/// A separate type and not `Param<N, f32>`, mirroring the `Kind` split: the
/// params channel is a byte run with no element type, so *"the Nth param,
/// read as f32"* is a different CHANNEL from *"the Nth param"* rather than a
/// different reading of the same one. `Kind` makes that distinction and a
/// type parameter would erase it back.
///
/// [`Kind::ParamF32`]: crate::Kind::ParamF32
#[derive(Clone, Copy, Debug)]
pub struct ParamF32<const N: usize>(pub f32);

/// AN OPERAND THE LAUNCH READS AND WRITES, said out loud.
///
/// The three sites where direction is not the wrapper's fact are all one
/// shape: a statement declares a buffer as an INPUT, the kernel writes through
/// it, and no result is declared. `In<N, *mut T>` reads as a typo rather than
/// as a claim.
///
/// It derives `Source::Slot(Kind::In, N)` like [`In`] does; what changes is
/// that the mutation is stated where a reader looks for it.
///
/// Not `in_place`, which is the ALLOCATOR's record that two operands share an
/// address. This is one operand both read and written.
#[derive(Clone, Copy, Debug)]
pub struct InOut<const N: usize, E: Elem> {
    /// The device address.
    pub ptr: *mut E,
    /// Rows in this launch's rectangle.
    pub rows: i32,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

impl<const N: usize, E: Elem> InOut<N, E> {
    /// A WINDOW into the operand: `count` rows starting at row `start`.
    ///
    /// An offset is `start * stride * size_of::<E>()`, which needs the element
    /// type in the wrapper; before that a windowed view had to be handed the
    /// element size, and `gemm/lora.rs` wrote the arithmetic by hand with `* 2`
    /// for `bf16` spelled as a literal.
    ///
    /// Named `window` and not `rows` because [`Out`] has a `rows` FIELD, and a
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
    ) -> Result<Region<*mut E>, Refusal> {
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
        let ptr = unsafe { self.ptr.add(start as usize * self.width as usize) };
        Ok(Region { ptr, rows: count, width: self.width, stride: Stride(self.width) })
    }

    /// See [`In::all`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    /// This launch's whole view of the operand. See [`In::all`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn all(&self, what: &'static str) -> Result<Region<*mut E>, Refusal> {
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

impl<const N: usize, B: Backend, E: Elem> Arg<B> for InOut<N, E>
where
    *mut E: Arg<B>,
{
    const TY: Ty = E::TY_MUT;
    const PROV: Provenance = <*mut E as Arg<B>>::PROV;
    // `Placed`, with `In`: the statement PLACES this operand. That it is also
    // written is a fact about the KERNEL, and `Side` answers "which end of the
    // statement", which has one answer here.
    const SIDE: Side = Side::Placed;
    const SPELLING: &'static str = E::CPP_MUT;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <*mut E as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = B::region(value, at)?;
        Ok(InOut { ptr, rows, width })
    }
}

/// A pointee: something a device address can point at, with both of its
/// pointer ABIs.
///
/// Not `jit::abi::Inst`, which is a C++ instantiation marker — how the device
/// text spells a type — and has implementors that are pure markers with no
/// pointer ABI at all. This is the host-side pair.

// ── The shader planes' slot vocabulary ──
//
// Disjoint from `In`/`Out`/`Weight` above, which take a POINTEE and derive the
// C++ spelling from `Elem`. These take the bound argument type and claim only
// an index, which is what metal, vulkan and wgpu bind.

/// A source spelled as a type, so that [`Reckoned`] can state one.
pub trait Says {
    /// The source this marker spells.
    const SOURCE: crate::Source;
}

/// The fact `Q`, by name: `Say<keys::Width>`.
#[derive(Clone, Copy, Debug)]
pub struct Say<Q>(core::marker::PhantomData<Q>);

impl<Q: crate::keys::Fact> Says for Say<Q> {
    const SOURCE: crate::Source = crate::Source::Named(Q::KEY);
}

/// The statement's `N`th scalar: `Nth<1>`.
#[derive(Clone, Copy, Debug)]
pub struct Nth<const N: usize>;

impl<const N: usize> Says for Nth<N> {
    const SOURCE: crate::Source = crate::Source::Slot(crate::Kind::Param, N as u8);
}

/// `A` if the statement carries it, else `B`: `Else<Nth<1>, Say<keys::Width>>`.
#[derive(Clone, Copy, Debug)]
pub struct Else<A, B>(core::marker::PhantomData<(A, B)>);

impl<A: Says, B: Says> Says for Else<A, B> {
    const SOURCE: crate::Source = crate::Source::Or(&A::SOURCE, &B::SOURCE);
}

/// `A` times `B`: `Times<Say<keys::Width>, Say<keys::Rows>>`.
#[derive(Clone, Copy, Debug)]
pub struct Times<A, B>(core::marker::PhantomData<(A, B)>);

impl<A: Says, B: Says> Says for Times<A, B> {
    const SOURCE: crate::Source = crate::Source::Times(&A::SOURCE, &B::SOURCE);
}

/// `A` over `B`, refused when `B` is zero: `Over<Say<keys::Width>, Nth<1>>`.
#[derive(Clone, Copy, Debug)]
pub struct Over<A, B>(core::marker::PhantomData<(A, B)>);

impl<A: Says, B: Says> Says for Over<A, B> {
    const SOURCE: crate::Source = crate::Source::Over(&A::SOURCE, &B::SOURCE);
}

/// Input slot `N`, with no claim about its shape.
///
/// [`In`] says "input `N`, a `rows × width` rectangle"; this says only the
/// first half. Both set `Derived::stated`, so both stop the macro counting.
///
/// A separate type and not an `Option` on [`In`]: roughly half the parameters
/// the macro reaches by position have no honest rectangle to offer. DO NOT
/// MANUFACTURE ONE -- invented rows and width are worse than a counted index,
/// because the launcher will believe them. At the TYPE, reading `x.width`
/// fails to compile; at the VALUE it would compile and unwrap.
///
/// Like [`In`], a stated index SETS the counter to `N + 1`.
#[derive(Clone, Copy, Debug)]
pub struct InSlot<const N: usize, T> {
    /// The device address.
    pub ptr: T,
}

/// Output slot `N`, with no claim about its shape.
///
/// [`InSlot`]'s doc carries the argument; this is its other half. The one
/// thing worth adding is that an output with no stated extent is the more
/// common case rather than the rarer one, because a result's rows and width
/// are what the STATEMENT placed and a launcher that writes through a
/// pointer it was handed often has no business restating them.
///
/// `norm::rmsnorm_bf16_with_fp16`'s `y_fp16` is the shape to keep in mind:
/// it is `Option<NonNull<_>>`, and wrapping it here preserves that, because
/// `classify` reads `nullable` from the WRAPPED type. A slot mark states
/// which result it is and says nothing about whether the statement placed
/// one — those are different questions and `Or<T>` answers the second.
#[derive(Clone, Copy, Debug)]
pub struct OutSlot<const N: usize, T> {
    /// The device address.
    pub ptr: T,
}

/// Input `N`, with its row WIDTH and no claim about how many rows.
///
/// The third shape between [`In`]'s whole rectangle and [`InSlot`]'s bare
/// address: a signature must be able to accept the half of a region that is
/// true. A REGION THAT CANNOT BE HALF-BUILT IS A FACT ABOUT THE STRUCT, NOT
/// ABOUT THE WORLD.
///
/// It is NOT `Slot(Kind::InWidth, n)` under another name -- that would be a
/// SELECTOR, and the set would grow per (kind, index). This derives
/// `Source::Slot(Kind::In, N)` like [`In`] does; the selection lives in the
/// type, not in the row.
///
/// The rows are ABSENT rather than zero because a zero would be a claim: the
/// binder mints regions with `unwrap_or(0)`, so a fictional `rows` reads as a
/// number rather than a refusal. The field is not here, so the mistake is not
/// available.
#[derive(Clone, Copy, Debug)]
pub struct InRow<const N: usize, T> {
    /// The device address.
    pub ptr: T,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}

/// Result `N`, with its row width and no claim about how many rows.
///
/// [`InRow`]'s counterpart and its doc carries the argument. The one thing
/// worth adding is that the output side is where the hole was found: the
/// fourteen width and shape marks that Stage 3 could not dispose of are
/// mostly `OutWidth(n)`, and `gemv_bf16` carries one of each.
#[derive(Clone, Copy, Debug)]
pub struct OutRow<const N: usize, T> {
    /// The device address.
    pub ptr: T,
    /// Elements per row. Zero where the statement gave none.
    pub width: i32,
}



/// An optional pointer this routine leaves ABSENT: `Null<Buf>`.
///
/// Fourteen arguments in the metal plane are bound from `state(None)` --
/// gpt-oss's per-head sink logits on a family that has none, a routed
/// matmul's bias on the form without one, six ring-buffer slots a paged
/// append does not use. The arm reaches for nothing on purpose, and until
/// now the row said nothing at all, which reads the same as an argument
/// nobody has got round to.
///
/// This is `Source::Lit(Lit::Null)`, and it is the one source that needs no
/// resolver: the answer is the absence.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Null<T> {
    /// The value, which is whatever the backend spells an absent pointer.
    pub v: T,
}

impl<T> Null<T> {
    /// Carry `v` as the pointer this routine leaves absent.
    pub const fn new(v: T) -> Self {
        Self { v }
    }
}

impl<T> core::ops::Deref for Null<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.v
    }
}

impl<B: Backend, T: Arg<B>> Arg<B> for Null<T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    const PROV: Provenance = T::PROV;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(crate::Source::Lit(crate::Lit::Null));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Self::new)
    }
}



/// The statement's `N`th scalar IF IT CARRIES ONE, else the fact `Q`:
/// `ParamOr<3, keys::RotaryWidth, i32>`.
///
/// [`Param`] says the statement MUST carry it; this says it MAY, and names
/// what stands in when it does not.
///
/// A chain and not a default: gemma-4 rotates a quarter of each full-attention
/// head and all of each sliding one, so a fire-wide `rotary_width` describes
/// neither layer -- while every single-shape deployment states nothing and
/// means the fire's number. The chain is the only thing true about both.
///
/// Zero is absent. See [`Source::Or`](crate::Source::Or).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ParamOr<const N: usize, Q, T> {
    /// The value.
    pub v: T,
    ask: core::marker::PhantomData<Q>,
}

impl<const N: usize, Q, T> ParamOr<N, Q, T> {
    /// Carry `v` as the statement's `N`th scalar or `Q`'s answer.
    pub const fn new(v: T) -> Self {
        Self {
            v,
            ask: core::marker::PhantomData,
        }
    }
}

impl<const N: usize, Q, T> core::ops::Deref for ParamOr<N, Q, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.v
    }
}

impl<const N: usize, B: Backend, Q: crate::keys::Fact, T: Arg<B>> Arg<B> for ParamOr<N, Q, T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    // The statement carries it when it has an opinion, so the trace is
    // where it comes from when it comes from anywhere placeable.
    const PROV: Provenance = Provenance::Trace;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(crate::Source::Or(
        &crate::Source::Slot(crate::Kind::Param, N as u8),
        &crate::Source::Named(Q::KEY),
    ));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Self::new)
    }
}

/// The statement's `N`th scalar IF IT CARRIES ONE, else the literal `L`:
/// `ParamOrLit<4, -1, i32>`.
///
/// The other half of [`ParamOr`], and the measurement is what separated
/// them: seven of the paged-attention sites fell back to a number no move of
/// the fire's geometry could shift, which is what a literal looks like from
/// outside. `-1` is "no sliding window" and `0` is "no mask stride"; both are
/// sentinels the shader reads, not shapes anything derives.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ParamOrLit<const N: usize, const L: i32, T> {
    /// The value.
    pub v: T,
}

impl<const N: usize, const L: i32, T> ParamOrLit<N, L, T> {
    /// Carry `v` as the statement's `N`th scalar or the literal `L`.
    pub const fn new(v: T) -> Self {
        Self { v }
    }
}

impl<const N: usize, const L: i32, T> core::ops::Deref for ParamOrLit<N, L, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.v
    }
}

impl<const N: usize, const L: i32, B: Backend, T: Arg<B>> Arg<B> for ParamOrLit<N, L, T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    const PROV: Provenance = Provenance::Trace;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(crate::Source::Or(
        &crate::Source::Slot(crate::Kind::Param, N as u8),
        &crate::Source::Lit(crate::Lit::I32(L)),
    ));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Self::new)
    }
}

/// An argument the driver COMPUTES from what it already knows:
/// `Reckoned<Times<Say<keys::Width>, Say<keys::Rows>>, i32>`.
///
/// The provenance forwards, as it does for [`Held`] and [`Null`]: whether a
/// row places this argument is a property of the argument, and the arithmetic
/// that reaches it changes nothing about that.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Reckoned<E, T> {
    /// The value.
    pub v: T,
    how: core::marker::PhantomData<E>,
}

impl<E, T> Reckoned<E, T> {
    /// Carry `v` as the number `E` computes.
    pub const fn new(v: T) -> Self {
        Self {
            v,
            how: core::marker::PhantomData,
        }
    }
}

impl<E, T> core::ops::Deref for Reckoned<E, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.v
    }
}

impl<E: Says, B: Backend, T: Arg<B>> Arg<B> for Reckoned<E, T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    const PROV: Provenance = T::PROV;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(E::SOURCE);

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Self::new)
    }
}







/// The driver's own answer to `Q`, at an argument the STATEMENT still places:
/// `Held<keys::ConvState, F32s>`.
///
/// Not [`Ask`], which sets `Provenance::Env` -- a claim that the trace names
/// no operand here. The op DOES carry an operand for `conv_state` and the
/// arity checker counts it; the driver just never reads it, resolving the slab
/// from its own pool. So the source is named while the provenance stays the
/// carrier's, and moving it to `Env` would fail arity on every `gdn` op.
///
/// That the two disagree is a finding: an operand the driver never reads is an
/// operand doing nothing, and the fix is upstream in whatever emits the op.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Held<Q, T> {
    /// The value.
    pub v: T,
    of: core::marker::PhantomData<Q>,
}

impl<Q, T> Held<Q, T> {
    /// Carry `v` as the driver's answer to `Q`.
    pub const fn new(v: T) -> Self {
        Self {
            v,
            of: core::marker::PhantomData,
        }
    }
}

impl<Q, T> core::ops::Deref for Held<Q, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.v
    }
}

impl<B: Backend, Q: crate::keys::Fact, T: Arg<B>> Arg<B> for Held<Q, T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    // FORWARDED, and that is the whole point -- see the type's doc.
    const PROV: Provenance = T::PROV;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(Q::SOURCE);

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Self::new)
    }
}

/// [`Env`] with the question filled in: `Ask<keys::TokenIds, I32s>`.
///
/// Not `Env<keys::TokenIds>`, which is the CUDA spelling and does not compile
/// on a shader plane: a fact's `Value` is the pointer CUDA passes, and a
/// shader carrier is a BINDING INDEX, `Buf(9)`, that the encoder's ordering
/// makes mean an address. There is no honest `unpack` between them.
///
/// So the fact and the carrier are stated separately, and the fact is the same
/// `keys::` type CUDA names -- which is the point: `"token_ids"` appears once
/// in the tree.
///
/// It sets [`Provenance::Env`] itself rather than sitting inside an `Env<_>`;
/// `Env<Ask<_, _>>` would be the provenance twice, once vaguely.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Ask<Q, T> {
    /// The carrier, in the plane's own vocabulary.
    pub ptr: T,
    /// The question. A [`crate::keys::Fact`], carried in the type only.
    pub of: core::marker::PhantomData<Q>,
}

impl<Q, T> core::ops::Deref for Ask<Q, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

impl<B: Backend, Q: crate::keys::Fact, T: Arg<B>> Arg<B> for Ask<Q, T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    const PROV: Provenance = Provenance::Env;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(Q::SOURCE);

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(|ptr| Ask { ptr, of: core::marker::PhantomData })
    }
}

/// The staged parameter block: `Block<Buf>`.
///
/// Every scalar the statement carries, laid out as one struct and bound as
/// one buffer. `Env<Buf>` said only that the binder supplies it, which is the
/// same thing it said about the KV pool and the rope frequencies; this says
/// which. See [`Kind::Params`].
///
/// [`Kind::Params`]: crate::Kind::Params
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Block<T> {
    /// The buffer.
    pub v: T,
}

impl<T> Block<T> {
    /// Wrap the block the binder staged.
    pub const fn new(v: T) -> Self {
        Self { v }
    }
}

impl<T> core::ops::Deref for Block<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.v
    }
}

impl<B: Backend, T: Arg<B>> Arg<B> for Block<T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    // The `Env<_>` this replaces was already saying it, exactly as `Aux`
    // does; leaving it at the default would be a silent arity change.
    const PROV: Provenance = Provenance::Env;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(crate::Source::Slot(crate::Kind::Params, 0));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(|v| Self { v })
    }
}

pub trait Elem: 'static {
    /// How the device text spells a `const` pointer to this.
    const CPP_CONST: &'static str;
    /// How it spells a mutable one.
    const CPP_MUT: &'static str;
    /// The [`Ty`] a `*const` to this binds as.
    const TY_CONST: Ty;
    /// The [`Ty`] a `*mut` to this binds as.
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

impl<const N: usize, E: Elem> In<N, E> {
    /// This operand's view over a row count the CALLER supplies: the half of
    /// a region a signature can state, with the other half supplied where it
    /// is known. [`InRow`] states the same half in the signature itself.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<*const E>, Refusal> {
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
    ) -> Result<Region<*const E>, Refusal> {
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
        let ptr = unsafe { self.ptr.add(start as usize * self.width as usize) };
        Ok(Region { ptr, rows: count, width: self.width, stride: Stride(self.width) })
    }

    /// This launch's whole view of the operand.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what` when the statement gave no row
    /// width. That is `stated_width`'s refusal, made where the view is built
    /// instead of at each reader.
    pub fn all(&self, what: &'static str) -> Result<Region<*const E>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows: self.rows, width: self.width, stride: Stride(self.width) })
    }
}

impl<const N: usize, E: Elem> Out<N, E> {
    /// This operand's view over a row count the CALLER supplies. See
    /// [`In::over`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] naming `what`.
    pub fn over(&self, rows: i32, what: &'static str) -> Result<Region<*mut E>, Refusal> {
        if self.width <= 0 {
            return Err(Refusal::Absent { what });
        }
        Ok(Region { ptr: self.ptr, rows, width: self.width, stride: Stride(self.width) })
    }

    pub fn all(&self, what: &'static str) -> Result<Region<*mut E>, Refusal> {
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

impl<const N: usize, E: Elem> In<N, E> {
    /// The allocation's shape, as much of it as the transport delivered.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        Layout::packed(self.rows, self.width)
    }
}

impl<const N: usize, E: Elem> Out<N, E> {
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

impl<const N: usize, B: Backend, E: Elem> Arg<B> for In<N, E>
where
    *const E: Arg<B>,
{
    // F3: THE ELEMENT IS THE PARAMETER, THE DIRECTION IS THE WRAPPER'S. The
    // `Ty` used to come from a pointer type the signature restated at 494 of
    // 497 sites; it comes from the element and the side now, which is the
    // pair `ptr_abi!` already wrote on one line.
    const TY: Ty = E::TY_CONST;
    const PROV: Provenance = <*const E as Arg<B>>::PROV;
    const SIDE: Side = Side::Placed;
    const SPELLING: &'static str = E::CPP_CONST;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <*const E as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = B::region(value, at)?;
        Ok(In { ptr, rows, width })
    }
}

impl<const N: usize, B: Backend, E: Elem> Arg<B> for Out<N, E>
where
    *mut E: Arg<B>,
{
    /// See [`In`]'s impl: the element states the type, the wrapper the side.
    const TY: Ty = E::TY_MUT;
    const PROV: Provenance = <*mut E as Arg<B>>::PROV;
    const SIDE: Side = Side::Declared;
    const SPELLING: &'static str = E::CPP_MUT;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = <*mut E as Arg<B>>::unpack(value, at)?;
        let Extent { rows, width } = B::region(value, at)?;
        Ok(Out { ptr, rows, width })
    }
}

impl<const N: usize, B: Backend, E: Elem> Arg<B> for Weight<N, *const E>
where
    *const E: Arg<B>,
{
    const TY: Ty = E::TY_CONST;
    const PROV: Provenance = <*const E as Arg<B>>::PROV;
    const SIDE: Side = Side::Placed;
    const SPELLING: &'static str = E::CPP_CONST;
    const SOURCE: Option<crate::Source> = Some(crate::Source::Slot(crate::Kind::Weight, N as u8));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <*const E as Arg<B>>::unpack(value, at).map(|ptr| Weight { ptr })
    }
}

// BOTH TRANSPARENT, INCLUDING `PROV`, and for `Unbound` that is a decision
// rather than a default taken. `Aux` above overrides `PROV` to `Env` because
// the `Env<_>` it replaced was already saying so; `Unbound` replaces a BARE
// pointer, which counts as a read in `arity_problem`, and an `Env` here
// would silence that count by claiming the runtime always supplies one.
// That is a stronger claim than the evidence at these parameters supports,
// and the point of the wrapper is to move a mark into the compiler's reach
// WITHOUT moving anything else.
impl<const N: usize, B: Backend, E: Elem> Arg<B> for Bank<N, E>
where
    *const E: Arg<B>,
{
    const TY: Ty = E::TY_CONST;
    const PROV: Provenance = <*const E as Arg<B>>::PROV;
    const SIDE: Side = Side::Placed;
    const SPELLING: &'static str = E::CPP_CONST;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <*const E as Arg<B>>::unpack(value, at).map(|ptr| Bank { ptr })
    }
}

// THE SLOT PAIR is transparent for the same reason the others are, and
// differs from `In`/`Out` in one respect only: those call
// `B::region(value, at)?` to fill `rows` and `width`, and these do not, so a
// launch whose statement carries no extent at this position still unpacks. A
// shape that does not exist should not be reachable, and should also not be
// REQUIRED.
//
// THE MIDDLE PAIR calls `B::region` and then THROWS THE ROWS AWAY. That is
// not waste: `region` is one query answering both extents together, so there
// is nothing cheaper to ask for, and what changes is what the launcher can
// SEE. They refuse exactly where `In`/`Out` refuse, unlike the slot pair --
// which is right, because a wrapper asking for a width against a statement
// that states no rectangle has none to give.
impl<B: Backend, T: Arg<B>> Arg<B> for Unbound<T> {
    const TY: Ty = T::TY;
    const SIDE: Side = T::SIDE;
    // `Either`, AND IT USED TO BE `T::PROV` WITH `Or<_>` WRITTEN INSIDE TO GET
    // HERE. Three sites spelled `Unbound<Or<*const T>>`, which is this type
    // counting a parameter as a placed read and the inner wrapper taking the
    // count back off -- two halves of one claim, on one parameter, disagreeing.
    //
    // The claim `Unbound` makes is that NOTHING supplies this argument. A
    // thing nothing supplies is not a thing the STATEMENT places, so counting
    // it in `arity_problem`'s `reads` was the error and `Or` was the patch.
    //
    // Not `Env`, and the old doc's reason stands: `Env` claims the runtime
    // always supplies one, which is a stronger statement than these
    // parameters support. `Either` says only that the statement may not have
    // placed it, which is exactly what `Unbound` already means.
    const PROV: Provenance = Provenance::Either;
    const SPELLING: &'static str = T::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(|ptr| Unbound { ptr })
    }
}

impl<const N: usize, T> core::ops::Deref for Aux<N, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<B: Backend, const N: usize, T: Arg<B>> Arg<B> for Aux<N, T> {
    const TY: Ty = T::TY;
    // The one wrapper below that is not transparent to provenance. See the
    // type's doc: the `Env<_>` this replaces was already saying it, so
    // leaving it at the default would be a silent arity change rather than a
    // preserved one.
    const PROV: Provenance = Provenance::Env;
    const SPELLING: &'static str = T::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Aux)
    }
}

impl<const N: usize, T> core::ops::Deref for Param<N, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<const N: usize> core::ops::Deref for ParamF32<N> {
    type Target = f32;

    fn deref(&self) -> &f32 {
        &self.0
    }
}

// TRANSPARENT TO PROVENANCE, AND THAT IS THE POINT. A param is the
// STATEMENT's — `Provenance::Trace`, which is `Arg`'s default and what a bare
// scalar already derives — so wrapping one changes the `Source` and nothing
// else. `Aux` overrides `PROV` because it replaced an `Env<_>` that was
// already claiming the driver supplies it; there is no such claim here.
//
// `Side::OfType` and not `Placed`: a param sits on neither side of the
// statement, because it is not an operand. `Side` answers "which end of the
// rectangle" for something that has no rectangle, so the honest answer is the
// one that says nothing was claimed.
impl<const N: usize, B: Backend, T: Arg<B>> Arg<B> for Param<N, T> {
    const TY: Ty = T::TY;
    const PROV: Provenance = T::PROV;
    const SPELLING: &'static str = T::SPELLING;
    const SOURCE: Option<crate::Source> = Some(crate::Source::Slot(crate::Kind::Param, N as u8));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Param)
    }
}

impl<const N: usize, B: Backend> Arg<B> for ParamF32<N>
where
    f32: Arg<B>,
{
    const TY: Ty = <f32 as Arg<B>>::TY;
    const PROV: Provenance = <f32 as Arg<B>>::PROV;
    const SPELLING: &'static str = <f32 as Arg<B>>::SPELLING;
    // THE SAME LINE `Param<N, T>` CARRIES, and it went missing here when the
    // slot vocabulary was restored beside CUDA's. Without it this inherits
    // `Arg`'s default `None`, and `bind` refuses every argument with no
    // source: `Unstated { what: "an argument whose signature does not say
    // where it comes from" }`. `neox_mb` takes two `ParamF32`, so the refusal
    // landed on rope and took 19 of `driver-wgpu`'s 23 serving tests with it
    // — and would have taken `kernels-metal`'s too, whose `neox_mb` has the
    // identical signature.
    const SOURCE: Option<crate::Source> = Some(crate::Source::Slot(crate::Kind::Param, N as u8));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        <f32 as Arg<B>>::unpack(value, at).map(ParamF32)
    }
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
    const ARGS: &'static [(Ty, Provenance)];
    /// Which side of the statement each argument sits on, in the same order.
    ///
    /// A SECOND array rather than a third column on [`Self::ARGS`], because
    /// the pair is read by `driver-metal`'s encoder as well and widening a
    /// tuple two crates destructure would touch every one of them to say one
    /// new thing about three routines.
    const SIDES: &'static [Side];
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
            const ARGS: &'static [(Ty, Provenance)] = &[$(($arg::TY, $arg::PROV)),*];
            const SIDES: &'static [Side] = &[$($arg::SIDE),*];
            const SOURCES: &'static [Option<crate::Source>] = &[$($arg::SOURCE),*];
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
    /// The routine's name, which is the `fn`'s name.
    pub name: &'static str,
    /// Its arguments, derived from the signature.
    pub args: &'static [(Ty, Provenance)],
    /// Which side of the statement each argument sits on. See [`Side`].
    pub sides: &'static [Side],
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
    /// `(output, input)` pairs that must be given the same address.
    ///
    /// OUTPUT FIRST, which three readers fix. Almost every pair is `(0, 0)`
    /// and symmetric; `apply_per_expert_scale`'s `(0, 1)` and
    /// `token_batched_weighted_sum_add`'s `(0, 2)` are the two that are not.
    ///
    /// In TRACE-OPERAND indices, not argument positions: a routine takes its
    /// arguments in whatever order the kernel wants, and the aliasing is a
    /// fact about the statement. So this is stated, not derived.
    pub in_place: &'static [(u32, u32)],
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
}

impl<B: Backend> Routine<B> {
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

    /// This routine, with its aliasing pairs stated.
    #[must_use]
    pub const fn in_place(mut self, pairs: &'static [(u32, u32)]) -> Self {
        self.in_place = pairs;
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
    /// The routine's name, which is the `fn`'s name.
    pub name: &'static str,
    /// Its arguments, derived from the signature.
    pub args: &'static [(Ty, Provenance)],
    /// Which side of the statement each argument sits on. See [`Side`].
    pub sides: &'static [Side],
    /// WHICH of the environment's questions each argument is, where a type
    /// says. See [`Ask`] and [`KernelFn::SOURCES`].
    pub sources: &'static [Option<crate::Source>],
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// This statement participates in the depth-prefix plan.
    pub depth_prefix_plan: bool,
    /// `(output, input)` pairs that must be given the same address.
    ///
    /// OUTPUT FIRST; see [`Routine::in_place`] for the three readers that fix
    /// the order.
    pub in_place: &'static [(u32, u32)],
}

impl<B: Backend> Routine<B> {
    /// This row, with the backend forgotten, for a cross-backend comparison.
    #[must_use]
    pub const fn declared(&self) -> Declared {
        Declared {
            name: self.name,
            args: self.args,
            sides: self.sides,
            sources: self.sources,
            whole: self.whole,
            depth_prefix_plan: self.depth_prefix_plan,
            in_place: self.in_place,
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
            .field("in_place", &self.in_place)
            .finish_non_exhaustive()
    }
}

/// The argument table of a routine `fn`, read off its signature.
///
/// The value is discarded: a `fn` item is a zero-sized type, and everything
/// wanted is in its type.
#[must_use]
pub const fn describe<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [(Ty, Provenance)] {
    F::ARGS
}

/// The same signature's SLOTS, in the same order. See [`Side`].
#[must_use]
pub const fn sides<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [Side] {
    F::SIDES
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
macro_rules! driver_bound {
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
            sides: &[],
            sources: &[],
            spelling: &[],
            body: by_path,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
            derived: &[],
        }
        $(.$fact($($value)?))*
    }};
}

/// One routine's row, from its `fn` and nothing else.
///
/// Backends wrap this with their own [`Backend`] type filled in, so that a
/// routine declaration names only the `fn`:
///
/// ```ignore
/// macro_rules! routine {
///     ($f:ident $(, $($rest:tt)*)?) => {
///         ::kernels::routine!($crate::Cuda, $f $(, $($rest)*)?)
///     };
/// }
/// ```
///
/// Trailing facts are the `const` builders of [`Routine`], named:
/// `routine!(B, rope_bf16, whole, in_place = &[(0, 0)])`.
///
/// A generic body answers several symbols, as
/// `routine!(B, rope_bf16 = rope::<bf16, 256>)`; the name is written out
/// because `stringify!` would answer `rope::<bf16, 256>`, which no trace can
/// state. It is the one place a routine's name is typed by hand, and it is
/// where the instantiation is chosen, so the two cannot drift.
#[macro_export]
macro_rules! routine {
    ($backend:ty, $name:ident = $body:expr $(, $fact:ident $(= $value:expr)?)* $(,)?) => {{
        fn shim<'x>(
            ctx: &'x <$backend as $crate::routine::Backend>::Ctx<'x>,
            args: &[<$backend as $crate::routine::Backend>::Value],
        ) -> ::core::result::Result<(), $crate::routine::Refusal> {
            <_ as $crate::routine::KernelFn<$backend, _>>::invoke($body, ctx, args)
        }
        $crate::routine::Routine::<$backend> {
            name: ::core::stringify!($name),
            args: $crate::routine::describe::<$backend, _, _>($body),
            sides: $crate::routine::sides::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
            derived: &[],
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
            args: $crate::routine::describe::<$backend, _, _>($body),
            sides: $crate::routine::sides::<$backend, _, _>($body),
            sources: $crate::routine::sources::<$backend, _, _>($body),
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
            derived: &[],
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

// ── The slot vocabulary's impls ──

impl<const N: usize, B: Backend, T: Arg<B>> Arg<B> for InRow<N, T> {
    const TY: Ty = T::TY;
    const PROV: Provenance = T::PROV;
    const SPELLING: &'static str = T::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = T::unpack(value, at)?;
        let Extent { width, .. } = B::region(value, at)?;
        Ok(InRow { ptr, width })
    }
}

impl<const N: usize, B: Backend, T: Arg<B>> Arg<B> for OutRow<N, T> {
    const TY: Ty = T::TY;
    const PROV: Provenance = T::PROV;
    const SPELLING: &'static str = T::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        let ptr = T::unpack(value, at)?;
        let Extent { width, .. } = B::region(value, at)?;
        Ok(OutRow { ptr, width })
    }
}

impl<const N: usize, B: Backend, T: Arg<B>> Arg<B> for InSlot<N, T> {
    const TY: Ty = T::TY;
    const PROV: Provenance = T::PROV;
    const SIDE: Side = Side::Placed;
    const SPELLING: &'static str = T::SPELLING;
    // The slot IS the source, and STAGE 2 already proved it against every
    // arm. Forwarding `T::SOURCE` left that proof out of the column the
    // binder reads.
    const SOURCE: Option<crate::Source> =
        Some(crate::Source::Slot(crate::Kind::In, N as u8));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(|ptr| InSlot { ptr })
    }
}

impl<const N: usize, B: Backend, T: Arg<B>> Arg<B> for OutSlot<N, T> {
    const TY: Ty = T::TY;
    const PROV: Provenance = T::PROV;
    const SIDE: Side = Side::Declared;
    const SPELLING: &'static str = T::SPELLING;
    // The slot IS the source, and STAGE 2 already proved it against every
    // arm. Forwarding `T::SOURCE` left that proof out of the column the
    // binder reads.
    const SOURCE: Option<crate::Source> =
        Some(crate::Source::Slot(crate::Kind::Out, N as u8));

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(|ptr| OutSlot { ptr })
    }
}

impl<const N: usize, T> core::ops::Deref for InSlot<N, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

impl<const N: usize, T> core::ops::Deref for OutSlot<N, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

impl<const N: usize, T> core::ops::Deref for Weight<N, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

// ── Constructors, for the fixtures that build a row by hand ──

impl<const N: usize, T> Param<N, T> {
    /// Wrap a value the statement carried in param slot `N`.
    #[must_use]
    pub const fn new(v: T) -> Self {
        Self(v)
    }
}

impl<const N: usize> ParamF32<N> {
    /// Wrap a float the statement carried.
    #[must_use]
    pub const fn new(v: f32) -> Self {
        Self(v)
    }
}

impl<const N: usize, T> InSlot<N, T> {
    /// Wear slot `N` for this pointer.
    pub const fn new(ptr: T) -> Self {
        Self { ptr }
    }
}

impl<const N: usize, T> OutSlot<N, T> {
    /// Wear result slot `N` for this pointer.
    pub const fn new(ptr: T) -> Self {
        Self { ptr }
    }
}

impl<const N: usize, T> Weight<N, T> {
    /// Wear named-weight slot `N` for this pointer.
    pub const fn new(ptr: T) -> Self {
        Self { ptr }
    }
}

impl<Q, T> Ask<Q, T> {
    /// Carry `ptr` as the answer to question `Q`.
    pub const fn new(ptr: T) -> Self {
        Self { ptr, of: core::marker::PhantomData }
    }
}
