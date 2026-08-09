//! The checkpoint loader's portable half: heap planning.
//!
//! Everything here is offset arithmetic over
//! [`DecodeGeometry`](crate::batch::DecodeGeometry), compiled and tested on
//! any host. The Metal side — allocating the heap, binding the argument
//! tables, staging tensors — layers on top and stays under
//! `gpu/weights/`. The ledger is `.wiki/driver/progress-metal.md`.

mod plan;
mod slab;

pub use plan::{LoadPlanError, compile_load_plan_for, metal_storage_target};
pub use slab::{ExpertSlab, SlabError, SlabTensor};
