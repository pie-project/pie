//! A routine is an ordinary `fn`, and its table row is derived from its
//! signature.
//!
//! The machinery here is everything a backend needs to DECLARE a routine set.
//! It declares no routines itself and names no kernel: backends' kernel sets
//! genuinely differ — CUDA's `mla_fa2` has no Metal twin and never will — so a
//! shared vocabulary of symbols would be a fiction. What is shared is the
//! shape: a `fn`, a derived argument table, and one erased entry point.
//!
//! The pattern is axum's and bevy's. A routine is written as a plain function
//! taking the backend's context and its arguments as ordinary Rust types:
//!
//! ```ignore
//! pub fn rope_apply(ctx: &Ctx, q: Bf16sMut, positions: Env<I32s>, rows: i32)
//!     -> Result<(), Refusal>
//! { ... }
//! ```
//!
//! [`macro@crate::routine`] then produces its table row *from the signature* — the row
//! cannot drift from the code, because there is only one statement of it.
//!
//! ## Why [`Backend`]
//!
//! [`Arg::unpack`] takes the backend's argument value and [`KernelFn::invoke`]
//! its context, and this crate can name neither: it is the floor both the CUDA
//! and the Metal tables stand on and it depends on nothing. One marker type
//! per backend carries the two as associated types. This is not the
//! generalisation over launch mechanics that a second live backend would
//! force — `Ctx` is opaque here — it is only what lets the trait be written
//! down at all.

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
}

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
    /// Who supplies it. Overridden only by [`Env`].
    const PROV: Provenance = Provenance::Trace;
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
    const PROV: Provenance = Provenance::Env;
    const SPELLING: &'static str = T::SPELLING;

    fn unpack(value: &B::Value, at: usize) -> Result<Self, Refusal> {
        T::unpack(value, at).map(Env)
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
    const ARGS: &'static [(Ty, Provenance)];
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
    /// The same arguments as the backend's shader language spells them,
    /// derived from the signature. See [`Arg::SPELLING`].
    pub spelling: &'static [&'static str],
    /// The erased body.
    pub body: Body<B>,
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// This statement participates in the depth-prefix plan.
    pub depth_prefix_plan: bool,
    /// `(input, output)` pairs that must be given the same address.
    ///
    /// In TRACE-OPERAND indices, which are not argument positions: a routine
    /// takes its arguments in whatever order the kernel wants them, and the
    /// aliasing is a fact about the statement. So this is stated, not derived.
    pub in_place: &'static [(u32, u32)],
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Declared {
    /// The routine's name, which is the `fn`'s name.
    pub name: &'static str,
    /// Its arguments, derived from the signature.
    pub args: &'static [(Ty, Provenance)],
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// This statement participates in the depth-prefix plan.
    pub depth_prefix_plan: bool,
    /// `(input, output)` pairs that must be given the same address.
    pub in_place: &'static [(u32, u32)],
}

impl<B: Backend> Routine<B> {
    /// This row, with the backend forgotten, for a cross-backend comparison.
    #[must_use]
    pub const fn declared(&self) -> Declared {
        Declared {
            name: self.name,
            args: self.args,
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

/// The same signature, as the backend's shader language spells it.
#[must_use]
pub const fn spell<B: Backend, M, F: KernelFn<B, M>>(_body: F) -> &'static [&'static str] {
    F::SPELLING
}

/// A symbol the DRIVER fires by path, declared without a statement's
/// argument list.
///
/// # The distinction, and why it is not a weaker [`routine!`](crate::routine!)
///
/// A `routine!` says: *a trace statement binds this*. That is what forces
/// every parameter to be [`Arg`] — [`KernelFn::invoke`] recovers them from a
/// `&[Value]` the statement produced, so a parameter no statement can supply
/// is a parameter the extractor cannot describe.
///
/// (It said `call` until `4d2753b4d`'s CI run denied the ambiguity beside it.
/// There is no `call` in this crate and there never was on this branch; the
/// link resolved to the MODULE, so the name had been wrong in silence.)
///
/// Some kernels are not like that, and no amount of porting makes them so.
/// A paged-KV write takes the layer's page geometry; an all-reduce takes a
/// communicator; a quantised GEMM takes a weight REPRESENTATION. Each is a
/// property of the fire or of the deployment, assembled by the driver, and a
/// statement mentions none of them. **They are still symbols a lowered model
/// text may name**, so the compiler must be able to look them up, and
/// `check_plan` refuses a model at load whose launched symbol is undeclared.
///
/// Both halves of that were true before this macro existed, and the cost was
/// a hand-written table — `not_yet_crossed.rs` in the CUDA backend, 21 rows,
/// each transcribing columns that a `fn` was sitting right beside. Every row
/// in it was a symbol whose BODY existed and whose ARGUMENTS were not a
/// statement's. This macro is that observation:
///
/// > A symbol is declared by the `fn` that runs it. Whether a STATEMENT can
/// > call it is a different question, and answering "no" is not a reason to
/// > write the declaration out by hand.
///
/// # What it produces
///
/// A [`Routine`] like `routine!`'s, with three differences, each of which is
/// a fact rather than a compromise:
///
/// * `args` is empty, because no statement supplies them. `KernelSig::args`'
///   own doc calls an empty list UNSTATED, which is exactly right here.
/// * `spelling` likewise.
/// * `body` REFUSES. Reaching it means something dispatched this symbol by
///   string, and the driver that owns it calls it by path — typed, and
///   checked by the compiler rather than by [`Refusal::Kind`] at first fire.
///   The refusal names that, so a wrong call site is a sentence and not a
///   mystery.
///
/// The `fn` is NAMED and not merely stringified, so a declaration cannot
/// outlive its body: deleting the `fn` fails this macro's expansion rather
/// than leaving a symbol nothing runs.
///
/// Trailing facts are [`Routine`]'s `const` builders, as `routine!`'s are.
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
            spelling: &[],
            body: by_path,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
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
/// # A generic body, and why the name is stated then
///
/// A routine may be generic — over its element type, over a block width — and
/// then one `fn` answers several trace symbols:
///
/// ```ignore
/// routine!(B, rope_bf16 = rope::<bf16, 256>)
/// routine!(B, rope_f16  = rope::<f16, 256>)
/// ```
///
/// The name is written out in that form because `stringify!` of the body would
/// answer `rope::<bf16, 256>`, which is not a symbol any trace can state. This
/// is the ONE place a routine's name is typed by hand, and it is the place the
/// instantiation is chosen — so the two cannot drift apart without the line
/// itself being wrong. The plain form derives the name and is what a routine
/// with nothing to vary still uses.
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
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
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
            spelling: $crate::routine::spell::<$backend, _, _>($body),
            body: shim,
            whole: false,
            depth_prefix_plan: false,
            in_place: &[],
        }
        $(.$fact($($value)?))*
    }};
}
