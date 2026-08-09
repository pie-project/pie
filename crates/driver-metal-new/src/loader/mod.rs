//! The checkpoint loader's portable half: heap planning.
//!
//! Everything here is offset arithmetic over
//! [`DecodeGeometry`](crate::batch::DecodeGeometry), compiled and tested on
//! any host. The Metal side — allocating the heap, binding the argument
//! tables, staging tensors — layers on top and stays under
//! `src/metal/`. The ledger is `PARITY-LOADER.md`.

mod plan;
mod slab;

pub use plan::{
    LoadPlanError, METAL_MAX_TILE_BYTES, METAL_PREFERRED_ALIGNMENT, METAL_TILE_MAP_MASK, TestFacts,
    compile_load_plan, descriptor_for_testing, metal_storage_target, plan_ties_embeddings,
};
pub use slab::{ExpertSlab, SlabError, SlabTensor};
