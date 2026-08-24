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

            $crate::arg_via_abi!(addressed *const $value);
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
    /// The latent attention's SCHEDULE, measured on the host against this
    /// fire's page table and uploaded into the workspace it names.
    ///
    /// The `Fa2Decode` seam exactly: `attn::plan::mla::plan` walks HOST
    /// slices -- `qo_indptr`, `kv_indptr`, `kv_len_arr` -- to carve a
    /// work-stealing schedule into an int arena, and the launch reads that
    /// arena and nothing else. A statement carries a query, a page row and
    /// three numbers; a body that built one of these would have to read the
    /// device CSR back to the host mid-fire, which is a sync a capture
    /// cannot record. So the plane stages it and the four `mla.attention_*`
    /// points stay claim-only, resolving through these routines' `canon`.
    MlaPlanned = "mla.plan" => crate::attn::MlaPlan
);
