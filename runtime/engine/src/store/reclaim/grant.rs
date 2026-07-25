//! The owned grant: what an acquisition returns and the only shape in which
//! reserved capacity travels. A grant owns concrete device page ids and RS
//! slot ids; store preparation functions drain exactly the prefix they
//! consume through [`AllocationGrant::lend_kv`]/[`AllocationGrant::lend_rs`]
//! while the Drop guard stays armed — consumed capacity leaves through a
//! store transaction, surplus returns through Drop, and nothing is ever
//! hand-released.

use std::sync::Arc;

use super::PoolPort;
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::RsSlotId;

/// A fire's sized ask: device pages and RS folded slots together, so a fire
/// never half-succeeds (D5: RS acquisition is unified into the grant).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demand {
    pub kv_pages: u32,
    pub rs_slots: u32,
}

impl Demand {
    pub fn kv(kv_pages: u32) -> Self {
        Self {
            kv_pages,
            rs_slots: 0,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.kv_pages == 0 && self.rs_slots == 0
    }

    /// Component-wise max — duplicate waits from one pipeline aggregate into
    /// one queue slot holding the larger ask.
    pub(super) fn max(self, other: Self) -> Self {
        Self {
            kv_pages: self.kv_pages.max(other.kv_pages),
            rs_slots: self.rs_slots.max(other.rs_slots),
        }
    }

    pub(super) fn covers(self, other: Self) -> bool {
        self.kv_pages >= other.kv_pages && self.rs_slots >= other.rs_slots
    }
}

/// Reserved device page ids. Dropping returns whatever remains to the pool
/// through the port; draining (prefix lending) is the only other way out.
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

    /// Fold another reservation from the same port into this one
    /// (head-first-claim accumulation).
    pub(super) fn absorb(&mut self, mut other: DevicePageReservation) {
        if self.port.is_none() {
            self.port = other.port.clone();
        }
        self.pages.append(&mut other.pages);
        other.port = None;
    }

    /// Split up to `count` pages off as their own reservation (the head
    /// reclaiming a younger entry's stranded partial accumulation). Pure
    /// bookkeeping — no port call.
    pub(super) fn donate(&mut self, count: usize) -> DevicePageReservation {
        let n = count.min(self.pages.len());
        DevicePageReservation {
            pages: self.pages.drain(..n).collect(),
            port: self.port.clone(),
        }
    }
}

impl Drop for DevicePageReservation {
    fn drop(&mut self) {
        if self.pages.is_empty() {
            return;
        }
        if let Some(port) = self.port.take() {
            port.release_device(std::mem::take(&mut self.pages));
        }
    }
}

/// Reserved RS folded-slot ids; same ownership protocol as
/// [`DevicePageReservation`].
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

/// One acquisition's result: everything the fire needs, owned together.
/// Acquired once, lent by exact prefix during the build, disposed once —
/// disposal IS the rollback.
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

    /// A grant carrying nothing, for zero-demand fires: the build path is
    /// uniform (everything goes through prefix lending) and disposal is free.
    pub fn empty() -> Self {
        Self {
            demand: Demand::default(),
            kv: DevicePageReservation::empty(),
            rs: RsSlotReservation::empty(),
        }
    }

    /// The ask this grant satisfied (diagnostics; the live remainder is
    /// [`Self::remaining_kv`]/[`Self::remaining_rs`]).
    pub fn demand(&self) -> Demand {
        self.demand
    }

    /// Reserved device pages not yet consumed by a store preparation.
    pub fn remaining_kv(&self) -> usize {
        self.kv.pages.len()
    }

    /// Reserved RS slots not yet consumed by a store preparation.
    pub fn remaining_rs(&self) -> usize {
        self.rs.slots.len()
    }

    /// Prefix-lend the device pages: store preparation drains exactly what
    /// it consumes while the Drop guard stays armed.
    pub fn lend_kv(&mut self) -> &mut Vec<PhysicalKvPageId> {
        &mut self.kv.pages
    }

    /// Prefix-lend the RS slots — same protocol as [`Self::lend_kv`].
    pub fn lend_rs(&mut self) -> &mut Vec<RsSlotId> {
        &mut self.rs.slots
    }
}
