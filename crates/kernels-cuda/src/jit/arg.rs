//! Every crossing type is a routine argument, through the [`Abi`] impl it
//! already has.
//!
//! [`Abi`](crate::jit::Abi) was written to be the one place a crossing type's
//! C++ spelling, its marshalling tag and its nullability are stated. That is
//! three of [`kernels::Arg`]'s columns, and `unpack` -- the fourth -- is on
//! the same impl, so the two directions cannot disagree about which
//! [`ArgValue`](crate::jit::ArgValue) a type is.
//!
//! ## Why this is a macro and not one blanket impl
//!
//! `impl<T: Abi> Arg<Cuda> for T` is what this wants to be, and the orphan
//! rule refuses it: in `impl<..> ForeignTrait<T1..Tn> for T0`, an uncovered
//! type parameter may not appear before the first local type, and here `T` is
//! `T0` while the only local type is `Cuda` at `T1`.
//!
//! So the impls are stamped per type instead -- by the same macro that stamps
//! the `Abi` impl, at the same site, from the same facts. Nothing is stated
//! twice: every column forwards.

/// Stamp [`kernels::Arg`] for a type that already implements
/// [`Abi`](crate::jit::Abi).
///
/// Invoked beside each `impl Abi`, so a crossing type is still one impl in
/// one place.
#[macro_export]
macro_rules! arg_via_abi {
    ($($rust:ty),* $(,)?) => {
        $(
            impl ::kernels::routine::Arg<$crate::jit::Cuda> for $rust {
                const TY: ::kernels::Ty = <$rust as $crate::jit::Abi>::TY;
                const SPELLING: &'static str = <$rust as $crate::jit::Abi>::CPP;

                fn unpack(
                    value: &$crate::jit::ArgValue,
                    at: usize,
                ) -> ::core::result::Result<Self, ::kernels::routine::Refusal> {
                    <$rust as $crate::jit::Abi>::unpack(value, at)
                }
            }
        )*
    };
}
