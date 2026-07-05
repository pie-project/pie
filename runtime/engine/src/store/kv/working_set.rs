//! Thin WIT/resource handle for `kv-working-set` (kv_refact.md,
//! `store/kv/working_set.rs`). All substantive operations delegate to the
//! owning `KvStore`, resolved through `store::registry` by `(model, driver)`.

use super::page_table::WorkingSetId;
use crate::driver::DriverId;

/// Host resource state behind the `pie:inferlet/working-set.kv-working-set`
/// WIT resource.
#[derive(Debug, Clone, Copy)]
pub struct KvWorkingSet {
    pub model: usize,
    pub driver: DriverId,
    pub id: WorkingSetId,
    /// Tokens per KV page (cached from the store registry at construction).
    pub page_size: u32,
}
