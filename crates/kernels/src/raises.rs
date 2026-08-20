//! Raises, as types — and [`Struct`], the carrier that binds one.
//!
//! A fire RAISES things before it launches anything: an attention schedule, a
//! routing table, a carve of the workspaces. Each is one object with one
//! lifetime, made once and read by every statement that names it. The word is
//! this tree's own — `fn raise_attn_plans`, *"one prefill plan is raised"*,
//! *"the workspace it was raised against"* — and it is already in the refusal
//! for the object that has, until now, had no name:
//!
//! ```ignore
//! Refusal::Unstated { what: "the plan this fire did not raise" }
//! ```
//!
//! # Why this is not [`crate::keys`]
//!
//! A fact is a QUESTION a body asks and gets one scalar back from. `ask` is a
//! CALL, and [`crate::routine::Asks`]'s own doc records what that costs: the
//! derived column no longer enumerates it, so a driver test cannot walk the
//! column and ask whether a backend answers everything its kernels name.
//!
//! A raise is an OPERAND. `In<Struct<Fa2Prefill>>` is a mark like any other —
//! positional, counted by `arity_problem` and `check_plan`, enumerated in the
//! column. That is the whole of what this module buys, and it is why the two
//! files are parallel rather than one:
//!
//! | a fact — one scalar | a raise — one object |
//! | --- | --- |
//! | `keys.rs` | `raises.rs` |
//! | [`crate::keys::Fact`] | [`Raise`] |
//! | `fact!` | `raise!` |
//! | `ctx.ask::<i32, keys::Rows>()` — a call | `In<Struct<Fa2Prefill>>` — a mark |
//!
//! # Why the declarations are not here
//!
//! Because a raise's `Value` is a PLANE's own type and this crate has no
//! dependencies. `keys.rs` can hold its facts because a fact's value is `f32`
//! or `*const i32` — spellings this crate can write. `PrefillPlanCache` is
//! `kernels-cuda`'s, so `raise!(Fa2Prefill = …)` is written there, the way
//! [`crate::routine::Elem`] lives here and each plane's `Tensor` lives with
//! the plane.
//!
//! # It never reaches a kernel
//!
//! A raised object is a HOST aggregate a body reads to fill the block a kernel
//! does take. `.wiki/kilimanjaro.md`'s **D1** — *"a routine takes fields, never
//! a struct"* — is about the other case, and its argument is a layout one: an
//! aggregate would trade a typed reference for `ArgValue::Bytes`, *"whose
//! layout agreement is not checked and cannot be here"*. [`Struct`] does not
//! make that trade. Its carriers are `*const T::Value` and `*mut T::Value`, so
//! it binds as a POINTER and keeps the typed reference D1 is protecting —
//! there is one declaration, in Rust, and no second one to agree with.

use crate::Ty;
use crate::routine::Elem;

/// One object a fire raises, named by a type.
///
/// A unit struct and not a newtype, which is where this parts from
/// [`crate::keys::Fact`]. A fact is a newtype because its value RIDES in the
/// field — a hand arm builds one and hands the scalar over. A raise's value is
/// the plane's own aggregate, reached by pointer and never moved, so there is
/// nothing for a field to hold.
pub trait Raise: 'static {
    /// The word, written exactly once in the tree — at the `raise!`.
    const KEY: &'static str;

    /// What the raise yields, as the plane's own type.
    ///
    /// Not `Copy` and not `Clone`, deliberately: `PrefillPlanCache` is neither
    /// (`attn/fa2/plan.rs:409` says so and gives the reason), and a raised
    /// object is read through a borrow rather than moved. A bound that forced
    /// either would exclude the first two types this trait exists for.
    type Value: 'static;
}

/// THE CARRIER: one raised object, by reference.
///
/// Sits beside `Tensor<E>` and `Table<Ptr>` and answers what a carrier
/// answers — *what does the body get to hold?* `Tensor<bf16>` yields a
/// [`crate::routine::Region`]; `Struct<Fa2Prefill>` yields a
/// `*const PrefillPlanCache`.
///
/// # Never constructed
///
/// Like every other [`Elem`], this is a type the marks are written over. The
/// value that crosses is `Self::Read`, and `In<Struct<T>>` holds one of those.
pub struct Struct<T: Raise>(core::marker::PhantomData<T>);

// `Debug` BY HAND, because deriving it would demand `T: Debug` for a field
// that holds nothing. The name is what a reader wants out of a refusal anyway.
impl<T: Raise> core::fmt::Debug for Struct<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Struct({})", T::KEY)
    }
}

impl<T: Raise> Elem for Struct<T> {
    type Read = *const T::Value;
    type Write = *mut T::Value;

    /// A RAISE IS ONE OBJECT, so there is nowhere to advance to.
    ///
    /// `In::window` and `In::over` are written over every `Elem` and mean
    /// nothing here: a raised plan has no rows to take a window of. Returning
    /// the pointer unchanged is the honest answer for the degenerate case —
    /// the alternative is `add(elems)` on a type where `elems` can only ever
    /// be zero, which reads as an offset somebody might one day supply.
    ///
    /// # Safety
    ///
    /// Trivially upheld: the result is the argument.
    unsafe fn advance_read(read: Self::Read, _elems: usize) -> Self::Read {
        read
    }

    /// [`Self::advance_read`]'s counterpart, and the same degenerate case.
    ///
    /// # Safety
    ///
    /// [`Self::advance_read`]'s.
    unsafe fn advance_write(write: Self::Write, _elems: usize) -> Self::Write {
        write
    }

    // NO C++ SPELLING, AND THE EMPTINESS IS THE CLAIM. `Elem`'s two `CPP_`
    // consts exist so a Rust mirror and a `__global__`'s parameter can be
    // checked against each other. A raised object never reaches a
    // `__global__`, so there is no declaration to check and a string here
    // would be a spelling nothing spells -- which is the sentence
    // `kernels-cuda/src/attn/fa2/mod.rs:1265` already writes about the two
    // plan aggregates this carrier generalises.
    const CPP_CONST: &'static str = "";
    const CPP_MUT: &'static str = "";

    const TY_CONST: Ty = Ty::Raised;
    const TY_MUT: Ty = Ty::Raised;
}

/// Declare a raise.
///
/// ```ignore
/// raise!(Fa2Prefill = "fa2.prefill" => crate::attn::fa2::plan::PrefillPlanCache);
/// ```
///
/// The key is the word the refusal uses and the trace names the value by. It
/// is written HERE and at no use site, which is [`crate::keys`]'s rule and the
/// reason that file's preamble can say the word appears once in the tree.
#[macro_export]
macro_rules! raise {
    ($(#[$m:meta])* $name:ident = $key:literal => $value:ty) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl $crate::raises::Raise for $name {
            const KEY: &'static str = $key;
            type Value = $value;
        }
    };
}
