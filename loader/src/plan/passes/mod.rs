//! The passes that run over a finished plan, and the order they run in.
//!
//! The order is the whole file: every pass here reads a plan the one before it
//! produced, and three of them only work because an earlier one has already
//! run — nothing can be coalesced into arena-relative writes before
//! `assign-persistent-offsets` has decided where the arena puts things, and
//! nothing can be counted before the coalescing is done. Under v1 this order
//! was seven consecutive statements in the middle of a 954-line file.
//!
//! The four modules split by what a pass *does*, not by what it touches:
//! [`arena`] assigns the persistent offsets everything downstream reads,
//! [`rewrite`] rewrites the schedule, [`memory`] recounts it, and [`validate`]
//! only refuses. Nothing here is re-exported as a prelude — a pass names what
//! it uses, so moving one is a matter of moving its imports with it.

use crate::error::Result;
use crate::plan::LoadPlan;
use crate::plan::pass::{FnPass, Pass};

mod arena;
mod memory;
mod rewrite;
pub mod tile;
mod validate;

#[cfg(test)]
mod tests;

/// The pipeline.
///
/// Adding a pass is adding a line here, which is the point: under v1 the same
/// change meant editing the middle of `StorageCompiler::lower`, where the pass
/// list was indistinguishable from the code that built the plan in the first
/// place.
pub fn all() -> Vec<Box<dyn Pass>> {
    vec![
        boxed(
            "assign-persistent-offsets",
            arena::assign_persistent_offsets,
        ),
        boxed(
            "coalesce-persistent-arena-writes",
            rewrite::coalesce_persistent_arena_writes,
        ),
        boxed("hoist-bulk-arena-writes", rewrite::hoist_bulk_extent_writes),
        boxed(
            "slab-scatter-arena-writes",
            rewrite::build_slab_scatter_writes,
        ),
        boxed(
            "merge-adjacent-extent-writes",
            rewrite::merge_adjacent_extent_writes,
        ),
        boxed("recompute-memory-plan", memory::recompute_memory_plan),
        boxed("validate-fill-order", validate::validate_fill_order),
        boxed("validate-target-support", validate::validate_target_support),
        boxed(
            "validate-persistent-layout",
            validate::validate_persistent_layout,
        ),
    ]
}

fn boxed(name: &'static str, run: fn(&mut LoadPlan) -> Result<usize>) -> Box<dyn Pass> {
    Box::new(FnPass { name, run })
}
