//! What a trace that states one of `sample`'s symbols binds to.

use super::Bound;

/// Every symbol this family binds -- the three plain argmaxes are unstated.
pub static ARMS: &[Bound] = &[Bound::derived("sample::lm_head_gemv_argmax_int8")];
