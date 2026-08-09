//! What happens when a trace states one of `xqa`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda-new`. They read the driver's
//! own vocabulary through `Cx`, so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use super::Bound;

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[Bound {
    symbol: "attn::attention_xqa_decode_bf16_prepared",
    arm: None,
    unbound: Some(
        "the row's own `needs = Prepare::FireWide`, which nothing discharges. \
        The host program is no longer what is missing -- it is \
        `x::xqa::xqa_decode_bf16`, and `driver-cuda/src/fire/xqa.rs::decode` \
        is the workspace carve above it -- and the operands are statable: \
        `attn_workspace()` and `sm_scale()` are `Cx` queries, `kv_layer()` \
        carries the pages and `plan()` the CSR. What is not is the ORDER. \
        This kernel reads a dense page table that `attn::build_xqa_metadata` \
        must have written earlier in the same fire, on the same stream; \
        `Prepare::FireWide` is the obligation that says so and is read by no \
        code in this repository -- not `model-compiler`, not \
        `bind::dispatch`, not `fire::launch`. An arm written today would \
        launch against whatever the workspace held from the last fire, which \
        is a plausible answer rather than a missing one",
    ),
}];
