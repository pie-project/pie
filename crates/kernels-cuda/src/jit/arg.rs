#[macro_export]
macro_rules! arg_via_abi {

    (addressed $($rust:ty),* $(,)?) => {
        $(
            impl ::kernels::plane::Arg<$crate::jit::Cuda> for $rust {
                const TY: ::kernels::Ty = <$rust as $crate::jit::Abi>::TY;

                fn unpack(
                    value: &$crate::jit::ArgValue,
                    at: usize,
                ) -> ::core::result::Result<Self, ::kernels::plane::Refusal> {
                    <$rust as $crate::jit::Abi>::unpack(value, at)
                }
            }
        )*
    };

    ($($rust:ty),* $(,)?) => {
        $(
            impl ::kernels::plane::Arg<$crate::jit::Cuda> for $rust {
                const TY: ::kernels::Ty = <$rust as $crate::jit::Abi>::TY;

                fn unpack(
                    value: &$crate::jit::ArgValue,
                    at: usize,
                ) -> ::core::result::Result<Self, ::kernels::plane::Refusal> {
                    <$rust as $crate::jit::Abi>::unpack(value, at)
                }
            }

            impl ::kernels::plane::Bind<$crate::jit::ArgValue> for $rust {
                fn arg(self) -> $crate::jit::ArgValue {
                    <$rust as $crate::jit::Abi>::arg(&self)
                }
            }
        )*
    };
}
