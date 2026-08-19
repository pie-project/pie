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

/// Stamp [`kernels::Arg`] and [`kernels::routine::Bind`] for a type that
/// already implements [`Abi`](crate::jit::Abi), beside each `impl Abi`, so a
/// crossing type is still one impl in one place.
///
/// BOTH DIRECTIONS, FOR THE SAME ORPHAN REASON. `Bind` is `kernels::routine`'s
/// and `ArgValue` is this crate's, so `impl<T: Abi + Copy> Bind<ArgValue> for T`
/// puts an uncovered `T` at `T0` ahead of the first local type exactly as the
/// `Arg` blanket did. Stamping it here costs nothing and keeps the two
/// crossings on one line: a type that can be bound is a type that can be
/// unpacked, and neither can be added without the other.
#[macro_export]
macro_rules! arg_via_abi {
    ($($rust:ty),* $(,)?) => {
        $(
            impl ::kernels::routine::Arg<$crate::jit::Cuda> for $rust {
                const TY: ::kernels::Ty = <$rust as $crate::jit::Abi>::TY;
                // `PROV` STOOD HERE AND `Provenance` IS DELETED. It read
                // `Abi::NULLABLE` to excuse the two spellings that accept a
                // null -- `Option<NonNull<T>>` and `MaybeConst<T>` -- from the
                // arity count. Nullability is still derived, by `#[routine]`
                // off the SYNTAX, into `Derived::nullable`; what is gone is the
                // second, weaker claim that a nullable parameter is supplied by
                // someone other than the statement. Every parameter is the
                // statement's now.
                const SPELLING: &'static str = <$rust as $crate::jit::Abi>::CPP;

                fn unpack(
                    value: &$crate::jit::ArgValue,
                    at: usize,
                ) -> ::core::result::Result<Self, ::kernels::routine::Refusal> {
                    <$rust as $crate::jit::Abi>::unpack(value, at)
                }
            }

            impl ::kernels::routine::Bind<$crate::jit::ArgValue> for $rust {
                fn arg(self) -> $crate::jit::ArgValue {
                    <$rust as $crate::jit::Abi>::arg(&self)
                }
            }
        )*
    };
}
