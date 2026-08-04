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
