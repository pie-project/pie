//! The batch subsystem: scheduling, composition, and the forward's shape.
//!
//! `csrc/src/batch/` is ~11.6k lines and is mostly not about the GPU: it
//! derives batch shapes from the CSR view the engine marshals, composes
//! channel tickets, colors scratch, and only at the end encodes a forward.
//! The port follows the crate's rule — portable half first, into modules a
//! Linux `cargo test` reaches — and `PARITY-BATCH.md` is its ledger.
//!
//! [`schedule`] is the batch shape: request spans, the token→request
//! expansion, and the paged-geometry gate that runs before any pool cell can
//! be addressed. [`mask`] answers whether a wire attention mask says
//! anything the kernel's own causal predicate does not already enforce.

mod abi;
mod consts;
mod geometry;
mod geometry_facts;
mod logits;
mod timing;

pub use abi::{
    ArgmaxParams, ForwardGraphKey, IO_SLOT_COUNT, IoSlot, Kernel, PAGE_BUCKET_GRAN, Region,
    SCRATCH_POOL,
};
pub use consts::{
    ExpertCombineParams, GatedRmsParams, GdnCoreParams, KN, MoeRouteParams, RmsParams,
    RouterParams, gdn_core_params, is_qmv, is_routed, qmv_kn,
};
pub use geometry::{AffineFormat, DecodeGeometry};
pub use geometry_facts::{
    GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K, geometry_from_facts,
};
pub use logits::{LengthMismatch, bf16_to_f32, widen, widen_into};
pub use timing::{
    Ablation, BoundaryMismatch, DispatchAttribution, DispatchInfo, StepAttribution, attribute_step,
};
