#[macro_export]
macro_rules! arg_via_abi {
    // A POINTER TAKES THE UNPACKING AND NOT THE BINDING. `Bind` for a raw
    // pointer is `kernels`' own, one impl over every pointee, because a
    // per-pointee one cannot be written for an element a `points` family
    // method only knows as `T: Scalar`.
    (addressed $($rust:ty),* $(,)?) => {
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

            impl ::kernels::routine::Bind<$crate::jit::ArgValue> for $rust {
                fn arg(self) -> $crate::jit::ArgValue {
                    <$rust as $crate::jit::Abi>::arg(&self)
                }
            }
        )*
    };
}
