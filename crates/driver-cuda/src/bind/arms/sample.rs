//! What happens when a trace states one of `sample`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.



use super::Bound;

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[Bound {
    symbol: "sample::lm_head_gemv_argmax_int8",
    arm: None,
    unbound: Some(
        "sample::lm_head_gemv_argmax_int8: all eight operands were unsourced. \
         The two that still are: the int8 head and its per-row dequant scale \
         are named weights, and no model text states this symbol, so there is \
         no statement to read the two names off. The host program is public",
    ),
}];
