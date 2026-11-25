//! The pressure bucket the default cluster policy places by: room only
//! idle work or idle cache holds, on the device or below it, never
//! saturates it.

use super::*;

use std::collections::HashSet;
use std::sync::Arc;

use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::RsUnits;

/// Each tier's `(free, total)` and what of it only idle cache holds.
#[derive(Default)]
struct TierPool {
    device: (u32, u32),
    device_cache: u32,
    host: (u32, u32),
    disk: (u32, u32),
    host_cache: u32,
    disk_cache: u32,
    /// The capacity gate's end and census (none: no gate).
    context: Option<u32>,
    census: Option<super::pool::CapacityCensus>,
}

impl PoolPort for TierPool {
    fn prepare_demote_now(&self) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn drop_disallowed(&self) -> usize {
        0
    }
    fn chunk_stats(&self) -> ChunkStats {
        ChunkStats {
            pages: self.device,
            rs: Default::default(),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        self.host
    }
    fn disk_stats(&self) -> (u32, u32) {
        self.disk
    }
    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        Default::default()
    }
    fn reclaim_idle(&self, _: super::pool::IdleShort) -> super::pool::IdleReclaim {
        super::pool::IdleReclaim::default()
    }
    fn reclaimable_idle(&self) -> u32 {
        self.device_cache
    }
    fn idle_below_device(&self) -> (u32, u32) {
        (self.host_cache, self.disk_cache)
    }
    fn suspend_capable(&self) -> bool {
        self.host.1 > 0
    }
    fn locus(&self) -> (usize, usize) {
        (0, 0)
    }
    fn reserve_device(&self, _count: u32) -> Option<Vec<PhysicalKvPageId>> {
        None
    }
    fn reserve_device_up_to(&self, _count: u32) -> Vec<PhysicalKvPageId> {
        Vec::new()
    }
    fn release_device(&self, _pages: Vec<PhysicalKvPageId>) {}
    fn reserve_rs(&self, _need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        None
    }
    fn release_rs(&self, _grant: crate::store::rs::RsGrant) {}
    fn reclaim_sharing(&self, _working_sets: &[crate::store::rs::RsWorkingSetId]) -> usize {
        0
    }
    fn prepare_sweep(
        &self,
        _keep: &HashSet<crate::store::kv::page_table::WorkingSetId>,
        _room: usize,
        _order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        _wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn prepare_demote(&self, _room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn context_pages(&self) -> Option<u32> {
        self.context
    }
    fn capacity_census(&self) -> Option<super::pool::CapacityCensus> {
        self.census
    }
}

fn bucket(pool: TierPool) -> u8 {
    ResidencyPlanner::new(Arc::new(pool) as Arc<dyn PoolPort>).kv_pressure_bucket()
}

/// A bounded host tier's shape: 1350 host and 5400 disk pages.
const HOST: u32 = 1350;
const DISK: u32 = 5400;

#[test]
fn host_and_disk_full_of_live_pages_still_report_pressure() {
    let b = bucket(TierPool {
        device: (512, 1024),
        host: (0, HOST),
        disk: (0, DISK),
        ..Default::default()
    });
    assert_eq!(b, 255);
    // Cache on the host is room; the disk's live pages still count.
    let b = bucket(TierPool {
        device: (512, 1024),
        host: (0, HOST),
        disk: (0, DISK),
        host_cache: HOST,
        ..Default::default()
    });
    assert_eq!(
        b,
        (f64::from(DISK) / f64::from(HOST + DISK) * 255.0).round() as u8
    );
}

#[test]
fn idle_cache_on_the_device_is_headroom() {
    let full = TierPool {
        device: (0, 1024),
        ..Default::default()
    };
    assert_eq!(bucket(full), 255);
    let b = bucket(TierPool {
        device: (0, 1024),
        device_cache: 1024,
        ..Default::default()
    });
    assert_eq!(b, 0, "a device full of idle cache holds no launch");
}

