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
//!
//! That last split is also a [`Stage`]: the validators come last because what
//! they prove has to hold of the plan the compiler hands back, and a rewrite
//! scheduled after one would quietly void it. `run_passes` enforces the
//! ordering rather than leaving it to whoever appends the next line.

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
pub fn all() -> &'static [Pass] {
    &[
        Pass {
            name: "assign-persistent-offsets",
            stage: Stage::Rewrite,
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
            run: stage::stage_device_transforms,
        },
        Pass {
            name: "coalesce-persistent-arena-writes",
            stage: Stage::Rewrite,
            run: rewrite::coalesce_persistent_arena_writes,
        },
        Pass {
            name: "hoist-bulk-arena-writes",
            stage: Stage::Rewrite,
            run: rewrite::hoist_bulk_extent_writes,
        },
        Pass {
            name: "recompute-memory-plan",
            stage: Stage::Rewrite,
            run: memory::recompute_memory_plan,
        },
        // LAST of the rewrites, and that is what makes the checks below able
        // to speak about a kernel at all: this is what names them.
        Pass {
            name: "lower-backend-tiling",
            stage: Stage::Rewrite,
            run: tile::lower_backend_tiling,
        },
        Pass {
            name: "validate-fill-order",
            stage: Stage::Check,
            run: validate::validate_fill_order,
        },
        Pass {
            name: "validate-target-support",
            stage: Stage::Check,
            run: validate::validate_target_support,
        },
        Pass {
            name: "validate-scale-factors",
            stage: Stage::Check,
            run: validate::validate_scale_factors,
        },
        Pass {
            name: "validate-persistent-layout",
            stage: Stage::Check,
            run: validate::validate_persistent_layout,
        },
        Pass {
            name: "validate-kernel-operands",
            stage: Stage::Check,
            run: validate::validate_kernel_operands,
        },
    ]
}
