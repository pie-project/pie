//! What is left of the driver's model knowledge, and where it went.
//!
//! This module used to hold the whole executor. Binding, dispatch, grids,
//! frames and resolution moved to [`crate::lowering`], the KV shape to
//! [`crate::layout::kv`], and everything that allocates to `crate::gpu`.
//!
//! The files here are the ones that cannot move *inside* this crate, because
//! their destination is another one.
//! `.wiki/driver/real-metal-north-star.md` §4 states the rule they violate:
//!
//! > A family name inside the driver is a fact that failed to reach the
//! > crate that owns it.
//!
//! * [`binding`] — what a Metal load OBSERVED (an affine point, an expert
//!   bank's format, three build capabilities) and the one door to the row's
//!   Metal text. It spells no family name at all, which is why it stays.
//! * [`rope`] — the rope tables a text asks for.
//!
//! # `text` has left
//!
//! `model/text.rs` was the third file here, and it held the rule's worst
//! violation: an eleven-entry table of architecture STRINGS, a `canonical()`
//! that folded spellings onto them, and a `facts_from_with()` that rebuilt
//! the model's own twenty-nine facts out of nine `has_tensor` probes. It
//! described a model, in a driver, from a string a row had already handed
//! out — the third dispatch key for one identity.
//!
//! It is gone rather than moved. Nothing of it needed a new home, because
//! `crates/model` already stated every fact it reconstructed:
//! `Variant::trace` now takes a `catalog::Deployed` whose `backend` names
//! which driver is asking, so the Metal text is the ROW's answer for Metal,
//! and this crate's whole remaining contribution is the six observations in
//! [`binding`] that no row can make. See that module's header for the two
//! bugs the second key cost — a gemma-4 claimed at the door and refused after
//! 17 GB of staging, and a norm variant decided by which norms a checkpoint
//! shipped.

pub mod binding;
pub mod rope;
