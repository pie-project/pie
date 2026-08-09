//! The family ladder, and nothing else any more.
//!
//! What was here that is family-neutral has left: the launch ABI is
//! [`crate::lowering::abi`] and the geometry-derived kernel params are
//! [`crate::lowering::consts`].
//!
//! What remains is the thing `.wiki/driver/real-metal-north-star.md` §3
//! measures and asks to be deleted rather than moved: `geometry` is a
//! model definition inside the driver, and `geometry_facts` is the
//! per-family projection ladder that fills it — *"projecting rather than
//! branching… the per-family ladder this crate is retiring."* Both go to
//! `crates/model` with `facts.rs`.
//!
//! `logits` (bf16 widening) and `timing` (dispatch attribution) were
//! parked here too, and are gone: **neither had a single caller.** They
//! were not part of #7's 2,608 lines — that figure is exactly
//! `facts.rs` + `geometry.rs` + `geometry_facts.rs` — so they were 465
//! lines waiting to be moved somewhere by a refactor that had no reason
//! to move them. Deleting code with no caller is not a step toward the
//! north star; it is removing something that would otherwise have to be
//! carried through every step.

pub(crate) mod geometry;
pub(crate) mod geometry_facts;

pub use geometry::{AffineFormat, DecodeGeometry};
pub use geometry_facts::{
    GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K, geometry_from_facts,
};
