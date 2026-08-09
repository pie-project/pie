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
//! `crates/model` with `facts.rs`, and `logits` and `timing` are the two
//! readback helpers that were parked beside them.

pub(crate) mod geometry;
pub(crate) mod geometry_facts;
pub(crate) mod logits;
pub(crate) mod timing;

pub use geometry::{AffineFormat, DecodeGeometry};
pub use geometry_facts::{
    GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K, geometry_from_facts,
};
pub use logits::{LengthMismatch, bf16_to_f32, widen, widen_into};
pub use timing::{
    Ablation, BoundaryMismatch, DispatchAttribution, DispatchInfo, StepAttribution, attribute_step,
};
