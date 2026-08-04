//! [`FrameSubmission`] — the sealed-frame request the scheduler hands to a
//! driver backend's `launch` verb (ABI v14). A frame carries its lane roster
//! and frame-invariant tables once, plus one [`StepSubmission`] per forward
//! step; the driver executes the steps as one closed system with a single
//! completion.

use pie_driver_abi::PieTerminalCell;

use super::command::LaunchPlan;

/// One forward step: the batch geometry (wire form) plus per-step metadata.
/// Batch members reference the frame roster through `roster_rows` and are
/// partitioned into ordered geometry-homogeneous sub-batches.
#[derive(Debug, Clone, PartialEq)]
pub struct StepSubmission {
    pub plan: LaunchPlan,
    /// Indices into [`FrameSubmission::instance_ids`], one per batch member,
    /// in sub-batch order.
    pub roster_rows: Vec<u32>,
    /// CSR over `roster_rows`; sub-batch `b` spans members
    /// `[sub_batch_indptr[b], sub_batch_indptr[b+1])`.
    pub sub_batch_indptr: Vec<u32>,
    /// `PIE_GEOMETRY_CLASS_*` per sub-batch.
    pub sub_batch_class: Vec<u32>,
    pub terminal_cells: Vec<*mut PieTerminalCell>,
    /// Program → wire-request attribution CSR (`roster_rows.len() + 1`
    /// entries): member `p` owns wire request rows
    /// `[row_indptr[p], row_indptr[p+1])`. Batched fires contribute one row
    /// each (a device-geometry fire's row is an empty placeholder the driver
    /// replaces with channel-resolved geometry).
    pub program_row_indptr: Vec<u32>,
    /// The fire planner's hook-free prefix in WIRE request rows
    /// (`fire_plan`'s qkv_postprocess site, converted through
    /// `program_row_indptr` — the planner's first consumed lowering).
    /// `PIE_HOOK_FREE_PREFIX_UNPLANNED` when the step carries no
    /// attribution to convert through; the driver then derives it alone.
    pub planned_hook_free_prefix_rows: u32,
    /// NS-2: leading wire rows with no user mask (hook-free steps only;
    /// `PIE_UNMASKED_PREFIX_UNPLANNED` otherwise).
    pub planned_unmasked_prefix_rows: u32,
    /// STRUCTURAL S-2: leading members at FULL depth (the depth
    /// seriation's request split; the truncated suffix's uniform k rides
    /// `planned_max_layers`). `PIE_FULL_DEPTH_UNPLANNED` = a uniform
    /// fire (solo truncated or all-full — today's shapes).
    pub planned_full_depth_rows: u32,
    /// V2 rung ③a: the region table (north-star-dsl.md "RUNG ③ SPEC") —
    /// maximal runs of members sharing an axis signature and depth
    /// operand, boundaries in WIRE rows through the attribution CSR.
    /// Empty = no table (the words' UNPLANNED discipline); the driver
    /// derives the scalar words from a present table and refuses drift.
    pub region_row_indptr: Vec<u32>,
    /// Axis bitset per region (`PIE_REGION_SIG_*`).
    pub region_sig: Vec<u32>,
    /// Depth operand per region (`PIE_MAX_LAYERS_FULL` = full).
    pub region_k: Vec<u32>,
    pub logical_fire_ids: Vec<u64>,
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
    pub channel_ticket_indptr: Vec<u32>,
}

/// The sealed frame handed to `DriverBackend::launch`.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSubmission {
    /// Lane roster: every bound instance participating in any step, in
    /// scheduler order. No duplicates.
    pub instance_ids: Vec<u64>,
    /// Frame-union WorkingSet page translation (committed mapping overlaid
    /// with ALL steps' prepared write targets) + its CSR partition, one
    /// segment per roster entry.
    pub kv_translation: Vec<u32>,
    pub kv_translation_indptr: Vec<u32>,
    /// Exclusive physical KV page high-water after the LAST step — the
    /// frame-union admission demand.
    pub required_kv_pages: u32,
    /// The frame's steps in execution order. Never empty.
    pub steps: Vec<StepSubmission>,
}
