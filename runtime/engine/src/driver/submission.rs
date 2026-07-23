//! [`LaunchSubmission`] — the batched-fire wire request the scheduler hands
//! to a driver backend's `launch` verb.

use pie_driver_abi::PieTerminalCell;

use super::command::LaunchPlan;

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchSubmission {
    pub plan: LaunchPlan,
    pub instance_ids: Vec<u64>,
    pub terminal_cells: Vec<*mut PieTerminalCell>,
    /// Flattened per-instance WorkingSet page translations (see
    /// [`LaunchPlan::kv_translation`]) + their CSR partition.
    pub kv_translation: Vec<u32>,
    pub kv_translation_indptr: Vec<u32>,
    /// Program → wire-request attribution CSR (`instance_ids.len() + 1`
    /// entries): program `p` owns wire request rows
    /// `[row_indptr[p], row_indptr[p+1])`. Batched fires contribute one row
    /// each (a device-geometry fire's row is an empty placeholder the driver
    /// replaces with channel-resolved geometry); a prebuilt solo plan owns
    /// every row it shipped.
    pub program_row_indptr: Vec<u32>,
    pub logical_fire_ids: Vec<u64>,
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
    pub channel_ticket_indptr: Vec<u32>,
    /// Frame-group settlement marker (`PieLaunchDesc.settle_defer`): `true`
    /// on every non-tail wave of a sealed frame; the tail (and every k=1 /
    /// rider / makeup / shutdown-drain launch) posts `false` and drains the
    /// group's deferred settlement.
    pub settle_defer: bool,
}
