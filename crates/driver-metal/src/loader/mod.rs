//! The checkpoint loader's portable half: heap planning — offset arithmetic
//! over [`DecodeGeometry`](crate::batch::DecodeGeometry); the Metal side (heap
//! allocation, argument tables, tensor staging) layers on top under `gpu/weights/`.

mod plan;
mod slab;

pub use plan::{LoadPlanError, compile_load_plan_for, metal_storage_target};
pub use slab::{ExpertSlab, SlabError, SlabTensor};
