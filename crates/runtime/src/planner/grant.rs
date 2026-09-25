use std::sync::Arc;

use super::PoolPort;
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::RsSlotId;

/// A kv page pool, one per page-id space: every full-context kv space
/// shares the paged ids, and the windowed spaces have their own, held only
/// while some sequence's window reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPool {
    Paged,
    Windowed,
}

impl KvPool {
    pub const ALL: [KvPool; 2] = [KvPool::Paged, KvPool::Windowed];
}

/// Pages per pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvPages([u32; 2]);

impl KvPages {
    #[must_use]
    pub fn paged(pages: u32) -> Self {
        KvPages([pages, 0])
    }

    #[must_use]
    pub fn new(paged: u32, windowed: u32) -> Self {
        KvPages([paged, windowed])
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == [0, 0]
    }

    #[must_use]
    pub fn saturating_sub(self, have: KvPages) -> KvPages {
        KvPages(std::array::from_fn(|at| {
            self.0[at].saturating_sub(have.0[at])
        }))
    }

    #[must_use]
    pub fn covers(self, need: KvPages) -> bool {
        need.saturating_sub(self).is_zero()
    }
}

impl std::ops::Index<KvPool> for KvPages {
    type Output = u32;
    fn index(&self, pool: KvPool) -> &u32 {
        &self.0[pool as usize]
    }
}

impl std::ops::IndexMut<KvPool> for KvPages {
    fn index_mut(&mut self, pool: KvPool) -> &mut u32 {
        &mut self.0[pool as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demand {
    pub kv_pages: KvPages,
    pub rs_slots: u32,
}

impl Demand {
    pub fn is_zero(&self) -> bool {
        self.kv_pages.is_zero() && self.rs_slots == 0
    }
}

/// Pages reserved in each pool, in that pool's own ids.
pub struct DevicePageReservation {
    pages: [Vec<PhysicalKvPageId>; 2],
    port: Option<Arc<dyn PoolPort>>,
}

impl std::fmt::Debug for DevicePageReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePageReservation")
            .field("pages", &self.pages)
            .finish_non_exhaustive()
    }
}

impl Default for DevicePageReservation {
    fn default() -> Self {
        Self::empty()
    }
}

impl DevicePageReservation {
    pub(super) fn new(pool: KvPool, pages: Vec<PhysicalKvPageId>, port: Arc<dyn PoolPort>) -> Self {
        let mut reservation = Self {
            pages: [Vec::new(), Vec::new()],
            port: Some(port),
        };
        reservation.pages[pool as usize] = pages;
        reservation
    }

    pub(super) fn empty() -> Self {
        Self {
            pages: [Vec::new(), Vec::new()],
            port: None,
        }
    }

    pub(super) fn len(&self, pool: KvPool) -> usize {
        self.pages[pool as usize].len()
    }

    pub(super) fn lens(&self) -> KvPages {
        let len = |pool: KvPool| u32::try_from(self.len(pool)).unwrap_or(u32::MAX);
        KvPages::new(len(KvPool::Paged), len(KvPool::Windowed))
    }

    pub(super) fn absorb(&mut self, mut other: DevicePageReservation) {
        if self.port.is_none() {
            self.port = other.port.clone();
        }
        for pool in KvPool::ALL {
            self.pages[pool as usize].append(&mut other.pages[pool as usize]);
        }
    }

    pub(super) fn donate(&mut self, count: KvPages) -> DevicePageReservation {
        let mut out = DevicePageReservation {
            pages: [Vec::new(), Vec::new()],
            port: self.port.clone(),
        };
        for pool in KvPool::ALL {
            let pages = &mut self.pages[pool as usize];
            let n = (count[pool] as usize).min(pages.len());
            out.pages[pool as usize] = pages.drain(..n).collect();
        }
        out
    }

    pub(super) fn lend(&mut self, pool: KvPool) -> &mut Vec<PhysicalKvPageId> {
        &mut self.pages[pool as usize]
    }
}

impl Drop for DevicePageReservation {
    fn drop(&mut self) {
        let Some(port) = self.port.take() else {
            return;
        };
        for pool in KvPool::ALL {
            let pages = std::mem::take(&mut self.pages[pool as usize]);
            if !pages.is_empty() {
                port.release_device(pool, pages);
            }
        }
    }
}

pub struct RsSlotReservation {
    slots: Vec<RsSlotId>,
    port: Option<Arc<dyn PoolPort>>,
}

impl std::fmt::Debug for RsSlotReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RsSlotReservation")
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

impl RsSlotReservation {
    pub(super) fn new(slots: Vec<RsSlotId>, port: Arc<dyn PoolPort>) -> Self {
        Self {
            slots,
            port: Some(port),
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            slots: Vec::new(),
            port: None,
        }
    }
}

impl Drop for RsSlotReservation {
    fn drop(&mut self) {
        if self.slots.is_empty() {
            return;
        }
        if let Some(port) = self.port.take() {
            port.release_rs(std::mem::take(&mut self.slots));
        }
    }
}

#[derive(Debug)]
pub struct AllocationGrant {
    demand: Demand,
    kv: DevicePageReservation,
    rs: RsSlotReservation,
}

impl AllocationGrant {
    pub(super) fn new(demand: Demand, kv: DevicePageReservation, rs: RsSlotReservation) -> Self {
        Self { demand, kv, rs }
    }

    pub fn empty() -> Self {
        Self {
            demand: Demand::default(),
            kv: DevicePageReservation::empty(),
            rs: RsSlotReservation::empty(),
        }
    }

    pub fn demand(&self) -> Demand {
        self.demand
    }

    pub fn remaining_kv(&self, pool: KvPool) -> usize {
        self.kv.len(pool)
    }

    pub fn remaining_rs(&self) -> usize {
        self.rs.slots.len()
    }

    pub fn lend_kv(&mut self, pool: KvPool) -> &mut Vec<PhysicalKvPageId> {
        self.kv.lend(pool)
    }

    pub fn lend_rs(&mut self) -> &mut Vec<RsSlotId> {
        &mut self.rs.slots
    }
}
