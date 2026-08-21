
macro_rules! raise_abi {
    ($($value:ty),* $(,)?) => {
        $(
            impl $crate::jit::Abi for *const $value {
                const CPP: &'static str = "";
                const TY: kernels::Ty = kernels::Ty::Raised;

                fn arg(&self) -> $crate::jit::ArgValue {
                    $crate::jit::ArgValue::Ptr((*self).cast::<core::ffi::c_void>().cast_mut())
                }

                fn unpack(
                    value: &$crate::jit::ArgValue,
                    at: usize,
                ) -> Result<Self, kernels::Refusal> {
                    match value {
                        $crate::jit::ArgValue::Ptr(p) => Ok(p.cast::<$value>().cast_const()),
                        _ => Err(kernels::Refusal::Kind { at, want: kernels::Ty::Raised }),
                    }
                }
            }

            $crate::arg_via_abi!(*const $value);
        )*
    };
}

raise_abi!(
    crate::attn::fa2::plan::PrefillPlanCache,
    crate::attn::fa2::plan::DecodePlanCache,
    crate::views::PagedKvView,
    crate::views::RecurrentView,
    crate::views::MaskView,
    crate::views::ExpertWeightsView,
    crate::views::MoeBanksView,
    crate::views::GemmGroupsView,
);

kernels::raise!(
    Fa2Prefill = "fa2.prefill" => crate::attn::fa2::plan::PrefillPlanCache
);

kernels::raise!(
    Fa2Decode = "fa2.decode" => crate::attn::fa2::plan::DecodePlanCache
);
