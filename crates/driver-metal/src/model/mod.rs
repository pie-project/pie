//! What is left of the driver's model knowledge, and where it is going.
//!
//! This module used to hold the whole executor. Binding, dispatch, grids,
//! frames and resolution moved to [`crate::lowering`], the KV shape to
//! [`crate::layout::kv`], and everything that allocates to `crate::gpu`.
//!
//! The two files here are the ones that cannot move *inside* this crate,
//! because their destination is another one.
//! `.wiki/driver/real-metal-north-star.md` §4 states the rule they violate:
//!
//! > A family name inside the driver is a fact that failed to reach the
//! > crate that owns it.
//!
//! * [`text`] — which text the loaded checkpoint is. A LOOKUP, not a choice:
//!   remove it and the same kernels fire. It still spells family names, so
//!   it goes to `crates/model` with `facts.rs` and `batch/geometry*.rs`.
//! * [`rope`] — the rope tables a text asks for.

pub mod rope;
pub mod text;
