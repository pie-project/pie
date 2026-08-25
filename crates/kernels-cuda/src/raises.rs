macro_rules! raise_abi {
    ($($value:ty),* $(,)?) => {
        $(
            impl $crate::jit::Abi for *const $value {
                const CPP: &'static str = "";

                fn arg(&self) -> $crate::jit::ArgValue {
                    $crate::jit::ArgValue::Ptr((*self).cast::<core::ffi::c_void>().cast_mut())
                }
            }
        )*
    };
}

raise_abi!(
    crate::attn::fa2::plan::PrefillPlanCache,
    crate::attn::fa2::plan::DecodePlanCache,
    crate::attn::MlaPlan,
    crate::views::PagedKvView,
    crate::views::RecurrentView,
    crate::views::MaskView,
    crate::views::ExpertWeightsView,
    crate::views::MoeBanksView,
    crate::views::GemmGroupsView,
    crate::views::ScoreView,
);

kernels::raise!(
    Fa2Prefill = "fa2.prefill" => crate::attn::fa2::plan::PrefillPlanCache
);

kernels::raise!(
    Fa2Decode = "fa2.decode" => crate::attn::fa2::plan::DecodePlanCache
);

kernels::raise!(

    MlaPlanned = "mla.plan" => crate::attn::MlaPlan
);
