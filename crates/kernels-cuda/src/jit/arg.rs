//! Every crossing type is a routine argument, through the [`Abi`] impl it
//! already has.
//!
//! [`Abi`](crate::jit::Abi) was written to be the one place a crossing type's
//! C++ spelling, its marshalling tag and its nullability are stated. That is
//! three of [`kernels::Arg`]'s columns, and `unpack` -- the fourth -- is on the
//! same impl, so the two directions cannot disagree.
//!
//! `impl<T: Abi> Arg<Cuda> for T` is what this wants to be, and the orphan
//! rule refuses it: an uncovered type parameter may not appear before the first
//! local type, and here `T` is `T0` while the only local type is `Cuda` at
//! `T1`. So the impls are stamped per type instead, by the same macro that
//! stamps the `Abi` impl, from the same facts: every column forwards.

/// Stamp [`kernels::Arg`] for a type that already implements
/// [`Abi`](crate::jit::Abi), beside each `impl Abi`, so a crossing type is
/// still one impl in one place.
#[macro_export]
macro_rules! arg_via_abi {
    ($($rust:ty),* $(,)?) => {
        $(
            impl ::kernels::routine::Arg<$crate::jit::Cuda> for $rust {
                const TY: ::kernels::Ty = <$rust as $crate::jit::Abi>::TY;
                // NULLABLE IS `Provenance::Either`: `Abi::NULLABLE` marks the
                // two spellings that accept a null -- `Option<NonNull<T>>` and
                // `MaybeConst<T>` -- which is §6.2's optional operand, so every
                // nullable parameter is excused from the arity count and
                // `routine::Or<T>` is left for the raw pointer positions that
                // take a null by convention rather than by type.
                const PROV: ::kernels::routine::Provenance =
                    if <$rust as $crate::jit::Abi>::NULLABLE {
                        ::kernels::routine::Provenance::Either
                    } else {
                        ::kernels::routine::Provenance::Trace
                    };
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
