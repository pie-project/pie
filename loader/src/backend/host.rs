//! Reference lowering.
//!
//! The host executor (`crate::host`) is a correctness oracle, not a
//! production path (§10.3): it exists so a plan can be run without a GPU and
//! compared against `crate::reference`. Its virtue is that it does the most
//! obvious thing, so it deliberately declines every optimization — no fusion,
//! and tiling driven only by the target's declared budget rather than by any
//! knowledge of scratch layout.
//!
//! That makes `host` the differential baseline the paper's experiments compare
//! `cuda` against: any behavioural difference between the two is a backend rule,
//! by construction, because nothing else differs.

use super::{Backend, TileLowering, TileMapFacts};
use crate::load_plan::{StorageTarget, TILE_MAP_CAST, TILE_MAP_REBLOCK, TransformFusion};

/// The transforms `host_executor` implements. Not a device capability, so it is
/// not part of the C surface.
pub const TILE_MAP_MASK: u32 = TILE_MAP_CAST | TILE_MAP_REBLOCK;

pub struct Host;

impl Backend for Host {
    fn name(&self) -> &'static str {
        "host"
    }

    fn tile_map_mask(&self) -> u32 {
        TILE_MAP_MASK
    }

    /// `0` rows per tile keeps `host_executor`'s existing behaviour, which
    /// derives its own tiling from `max_tile_bytes` at run time. The reference
    /// executor is allowed to do that precisely because it is not the thing
    /// whose execution the plan is supposed to determine.
    fn lower_tile_map(&self, _facts: &TileMapFacts, _target: &StorageTarget) -> TileLowering {
        TileLowering {
            rows_per_tile: 0,
            fusion: TransformFusion::None,
        }
    }
}
