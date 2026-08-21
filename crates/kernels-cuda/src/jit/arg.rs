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

            impl ::kernels::routine::Bind<$crate::jit::ArgValue> for $rust {
                fn arg(self) -> $crate::jit::ArgValue {
                    <$rust as $crate::jit::Abi>::arg(&self)
                }
            }
        )*
    };
}
