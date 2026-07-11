//! Thin WIT/resource handle for `rs-working-set` (kv_refact.md,
//! `store/rs/working_set.rs`). All substantive operations delegate to the
//! owning `RsStore`, resolved through `store::registry` by `(model, driver)`.

use super::{RsGeometry, RsWorkingSetId};
use crate::driver::DriverId;

/// Host resource state behind the `pie:inferlet/working-set.rs-working-set`
/// WIT resource.
#[derive(Debug, Clone, Copy)]
pub struct RsWorkingSet {
    pub model: usize,
    pub driver: DriverId,
    pub id: RsWorkingSetId,
    /// Model RS geometry (cached from model caps at construction).
    pub geom: RsGeometry,
}
