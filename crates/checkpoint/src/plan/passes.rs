//! The passes that run over a finished plan, and the order they run in: each
//! pass reads a plan the one before it produced, and some only work because
//! an earlier one already ran (e.g. coalescing needs `assign-persistent-offsets`
//! to have placed things first). `arena` assigns persistent offsets,
//! `rewrite` rewrites the schedule, `memory` recounts it, `validate` only
//! refuses; the split is also a [`Stage`] order, enforced by `run_passes`
//! since validators must run last.

use crate::plan::pass::{Pass, Stage};

mod arena;
mod memory;
mod rewrite;
mod stage;
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
///
/// THERE IS ONE LIST AND TWO READERS. [`pass::run_all`] runs it whole;
/// [`pass::run_arenaless`] runs it minus the entries whose
/// [`for_arena`](Pass::for_arena) says they exist to serve an arena, which is
/// the pipeline a plan compiled for a streaming execution wants. The second
/// reader is a filter and not a second list, so a pass added below is in both.
///
/// [`pass::run_all`]: crate::plan::pass::run_all
/// [`pass::run_arenaless`]: crate::plan::pass::run_arenaless
pub fn all() -> &'static [Pass] {
    &[
        Pass {
            name: "assign-persistent-offsets",
            stage: Stage::Rewrite,
            for_arena: false,
            run: arena::assign_persistent_offsets,
        },
        // After the resident layout, because a staging buffer goes BEHIND the
        // resident tensors and needs to know where they end; before the
        // coalescing, because the writes it emits are ordinary buffer-relative
        // ones and it is the coalescer's job to decide what becomes a bulk
        // arena write. (It decides "not these": a staging write must stay
        // beside the transform that reads it, and `hoist-bulk-arena-writes`
        // moves every bulk write to the front.)
        Pass {
            name: "stage-device-transforms",
            stage: Stage::Rewrite,
            for_arena: false,
            run: stage::stage_device_transforms,
        },
        Pass {
            name: "coalesce-persistent-arena-writes",
            stage: Stage::Rewrite,
            for_arena: true,
            run: rewrite::coalesce_persistent_arena_writes,
        },
        Pass {
            name: "hoist-bulk-arena-writes",
            stage: Stage::Rewrite,
            for_arena: true,
            run: rewrite::hoist_bulk_extent_writes,
        },
        Pass {
            name: "recompute-memory-plan",
            stage: Stage::Rewrite,
            for_arena: false,
            run: memory::recompute_memory_plan,
        },
        // LAST of the rewrites, and that is what makes the checks below able
        // to speak about a kernel at all: this is what names them.
        Pass {
            name: "lower-backend-tiling",
            stage: Stage::Rewrite,
            for_arena: false,
            run: tile::lower_backend_tiling,
        },
        Pass {
            name: "validate-fill-order",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_fill_order,
        },
        Pass {
            name: "validate-target-support",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_target_support,
        },
        Pass {
            name: "validate-bound-encodings",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_bound_encodings,
        },
        Pass {
            name: "validate-scale-factors",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_scale_factors,
        },
        Pass {
            name: "validate-persistent-layout",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_persistent_layout,
        },
        Pass {
            name: "validate-kernel-operands",
            stage: Stage::Check,
            for_arena: false,
            run: validate::validate_kernel_operands,
        },
    ]
}
