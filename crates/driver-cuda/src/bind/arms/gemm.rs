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
    // A DECLARED FORWARDER, not a name the ABI dropped. `gemm::act_x_w` has a
    // `ROUTINES` row of its own whose body is `act_x_wt_bf16` with `beta`
    // pinned by `#[lit(0.0)]`, and `act_x_w_acc` is the same call at 1.0 —
    // which is what makes both crossable: the number that separates the twins
    // is stated by the SYMBOL, so each row's column carries its own literal
    // and neither needs a parameter a trace would have to supply.
    Bound::derived("gemm::act_x_w"),
    // The accumulating twin also takes the residual, as `In<1, _>` aliasing
    // `y` — declared, not missing: its `ROUTINES` row states
    // `in_place = &[(1, 0)]`, and the body reads the operand only to keep the
    // allocator from handing this launch a buffer nothing wrote.
    Bound::derived("gemm::act_x_w_acc"),
    Bound::derived("gemm::act_x_wt_bf16_out_fp32"),
];
