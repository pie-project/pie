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
//! `arena` assigns the persistent offsets everything downstream reads,
//! `rewrite` rewrites the schedule, `memory` recounts it, and `validate`
//! only refuses. Nothing here is re-exported as a prelude — a pass names what
//! it uses, so moving one is a matter of moving its imports with it.

use crate::plan::pass::Pass;

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
pub fn all() -> &'static [Pass] {
    &[
        Pass {
            name: "assign-persistent-offsets",
            run: arena::assign_persistent_offsets,
        },
        Pass {
            name: "coalesce-persistent-arena-writes",
            run: rewrite::coalesce_persistent_arena_writes,
        },
        Pass {
            name: "hoist-bulk-arena-writes",
            run: rewrite::hoist_bulk_extent_writes,
        },
        Pass {
            name: "group-shared-source-reads",
            run: rewrite::group_shared_source_reads,
        },
        Pass {
            name: "merge-adjacent-extent-writes",
            run: rewrite::merge_adjacent_extent_writes,
        },
        Pass {
            name: "recompute-memory-plan",
            run: memory::recompute_memory_plan,
        },
        Pass {
            name: "validate-fill-order",
            run: validate::validate_fill_order,
        },
        Pass {
            name: "validate-target-support",
            run: validate::validate_target_support,
        },
        Pass {
            name: "validate-scale-factors",
            run: validate::validate_scale_factors,
        },
        Pass {
            name: "validate-persistent-layout",
            run: validate::validate_persistent_layout,
        },
    ]
}
