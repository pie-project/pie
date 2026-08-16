//! What a trace that states one of `gemm`'s symbols binds to.
//!
//! A cuBLAS body does not keep a row out of the derived column:
//! `table::dispatch` ends in `call_with_cublas` and hands the handle to every
//! column.

use super::Bound;

/// Every symbol this family accounts for.
pub static ARMS: &[Bound] = &[
    // Hand-dispatched: the body needs the fire's `LoraFireState` and a
    // resolver-owned aux slot, neither of which a query-only `Cx` may hand out.
    Bound::driver("gemm::lora_qkv_correction"),
    // Binds, then refuses every fire: `ROUTINES` has no `gemm::act_x_w` row,
    // the routine being `gemm::act_x_wt_bf16`, renamed at the ABI. Behind that,
    // `beta` is 0.0 here and 1.0 on the twin, chosen by the stated symbol.
    Bound::derived("gemm::act_x_w"),
    // The above, plus a residual no parameter exists to carry.
    Bound::derived("gemm::act_x_w_acc"),
    Bound::derived("gemm::act_x_wt_bf16_out_fp32"),
];
