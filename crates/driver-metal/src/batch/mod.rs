//! The one shape a checkpoint decides, and how a catalog row states it.
//!
//! What was here that is family-neutral has left: the launch ABI is
//! [`crate::lowering::abi`] and the geometry-derived kernel params are
//! [`crate::lowering::consts`].
//!
//! What was here that was a MODEL DEFINITION has left too, and that is
//! the change worth reading. `.wiki/driver/real-metal-north-star.md` §3
//! measured 2,608 lines across `facts.rs`, `geometry.rs` and
//! `geometry_facts.rs` and asked for them to be *deleted rather than
//! moved* — *"projecting rather than branching… the per-family ladder
//! this crate is retiring."* Two of the three are gone:
//!
//!   * `facts.rs` held a private `ModelFacts` this driver parsed out of a
//!     `pie.model/1` JSON descriptor, plus an `arch_stem` that lowercased
//!     `Qwen3MoeForCausalLM` into a dispatch key. The descriptor does not
//!     exist any more and neither does the dispatch: a checkpoint is
//!     matched to a `model::catalog` row BY ITS TENSORS.
//!   * `geometry_facts.rs` was the 888-line projection ladder that merged
//!     four family-prefixed blocks (`ll_*`, `go_*`, `g4_*`, `q35_*`) back
//!     into one shape by asking which block had been filled.
//!
//! [`geometry`] is what remains, and it is no longer a model definition:
//! [`DecodeGeometry`] holds the METAL-side numbers — the affine point
//! that names a kernel symbol, the simdgroup-bounded GDN strides, the
//! pool capacities an operator sets — and [`geometry_from_deployment`]
//! fills it from a `model::deployment::Deployment` the row projected. The
//! refusals moved with it unchanged, because every one of them was always
//! a Metal limit rather than a statement about a config.

pub(crate) mod geometry;

pub use geometry::{
    AffineFormat, DecodeGeometry, GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K,
    geometry_from_deployment,
};
