use std::sync::Arc;

use super::PoolPort;
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::RsSlotId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demand {
    pub kv_pages: u32,
    pub rs_slots: u32,
}

impl Demand {
    pub fn is_zero(&self) -> bool {
        self.kv_pages == 0 && self.rs_slots == 0
    }
}

pub struct DevicePageReservation {
    pages: Vec<PhysicalKvPageId>,
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
    pub(super) fn new(pages: Vec<PhysicalKvPageId>, port: Arc<dyn PoolPort>) -> Self {
        Self {
            pages,
            port: Some(port),
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            pages: Vec::new(),
            port: None,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.pages.len()
    }

    pub(super) fn absorb(&mut self, mut other: DevicePageReservation) {
        if self.port.is_none() {
            self.port = other.port.clone();
        }
        self.pages.append(&mut other.pages);
    }

    pub(super) fn donate(&mut self, count: u32) -> DevicePageReservation {
        let n = (count as usize).min(self.pages.len());
        DevicePageReservation {
            pages: self.pages.drain(..n).collect(),
            port: self.port.clone(),
        }
    }

    pub(super) fn lend(&mut self) -> &mut Vec<PhysicalKvPageId> {
        &mut self.pages
    }
}

impl Drop for DevicePageReservation {
    fn drop(&mut self) {
        let Some(port) = self.port.take() else {
            return;
        };
        let pages = std::mem::take(&mut self.pages);
        if !pages.is_empty() {
            port.release_device(pages);
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

    pub(super) fn lend(&mut self) -> &mut Vec<RsSlotId> {
        &mut self.slots
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

    pub fn remaining_kv(&self) -> usize {
        self.kv.len()
    }

    pub fn remaining_rs(&self) -> usize {
        self.rs.slots.len()
    }

    pub fn lend_kv(&mut self) -> &mut Vec<PhysicalKvPageId> {
        self.kv.lend()
    }

    pub fn lend_rs(&mut self) -> &mut Vec<RsSlotId> {
        &mut self.rs.slots
    }
}
