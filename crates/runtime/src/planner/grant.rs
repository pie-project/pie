use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use tokio::sync::Notify;

use super::PoolPort;
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::rs::{RsGrant, RsUnits};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demand {
    pub kv_pages: u32,
    pub rs: RsUnits,
}

impl Demand {
    pub fn is_zero(&self) -> bool {
        self.kv_pages == 0 && self.rs.is_zero()
    }
}

/// Room given back by a grant dropped with pages or slots unspent, told
/// the planner without its lock: the drop may run under it (a waiter
/// cancelled, its served grant dropped with its entry), so it only bumps
/// [`Self::epoch`] and signals, never plans. The planner reads the epoch
/// before a waiter parks and after every wake, and around each plan pass,
/// and plans again where it moved.
#[derive(Default)]
pub(super) struct RoomReturns {
    epoch: AtomicU64,
    /// The planner's queue length, kept by `with_inner`: a drop with no
    /// one waiting wakes nothing.
    pub(super) waiters: AtomicUsize,
    /// The drain task's wake, once armed: a drop plans through it.
    pub(super) drain: OnceLock<Arc<Notify>>,
    /// Where no drain task is armed: the parked waiters, each to plan.
    room: Notify,
}

impl RoomReturns {
    /// Grants dropped with room unspent so far.
    pub(super) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Wakes on room given back while no drain task is armed (with one,
    /// the drain plans and serves the waiter through its own notify).
    pub(super) fn returned(&self) -> tokio::sync::futures::Notified<'_> {
        self.room.notified()
    }

    /// Room came back: the epoch moves, then the planner is woken, both
    /// lock-free (`Notify` takes only its own lock).
    fn given(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        if self.waiters.load(Ordering::Acquire) == 0 {
            return;
        }
        match self.drain.get() {
            Some(drain) => drain.notify_one(),
            None => self.room.notify_waiters(),
        }
    }
}

pub struct DevicePageReservation {
    pages: Vec<PhysicalKvPageId>,
    port: Option<Arc<dyn PoolPort>>,
    /// Told when a grant holding these drops them unspent.
    returns: Option<Arc<RoomReturns>>,
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
    pub(super) fn new(
        pages: Vec<PhysicalKvPageId>,
        port: Arc<dyn PoolPort>,
        returns: Arc<RoomReturns>,
    ) -> Self {
        Self {
            pages,
            port: Some(port),
            returns: Some(returns),
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            pages: Vec::new(),
            port: None,
            returns: None,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.pages.len()
    }

    pub(super) fn absorb(&mut self, mut other: DevicePageReservation) {
        if self.port.is_none() {
            self.port = other.port.clone();
        }
        if self.returns.is_none() {
            self.returns = other.returns.clone();
        }
        self.pages.append(&mut other.pages);
    }

    pub(super) fn donate(&mut self, count: u32) -> DevicePageReservation {
        let n = (count as usize).min(self.pages.len());
        DevicePageReservation {
            pages: self.pages.drain(..n).collect(),
            port: self.port.clone(),
            returns: self.returns.clone(),
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
    grant: RsGrant,
    port: Option<Arc<dyn PoolPort>>,
    returns: Option<Arc<RoomReturns>>,
}

impl std::fmt::Debug for RsSlotReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RsSlotReservation")
            .field("grant", &self.grant)
            .finish_non_exhaustive()
    }
}

impl RsSlotReservation {
    pub(super) fn new(grant: RsGrant, port: Arc<dyn PoolPort>, returns: Arc<RoomReturns>) -> Self {
        Self {
            grant,
            port: Some(port),
            returns: Some(returns),
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            grant: RsGrant::default(),
            port: None,
            returns: None,
        }
    }

    pub(super) fn lend(&mut self) -> &mut RsGrant {
        &mut self.grant
    }
}

impl Drop for RsSlotReservation {
    fn drop(&mut self) {
        if self.grant.is_empty() {
            return;
        }
        if let Some(port) = self.port.take() {
            port.release_rs(std::mem::take(&mut self.grant));
        }
    }
}

#[derive(Debug)]
pub struct AllocationGrant {
    kv: DevicePageReservation,
    rs: RsSlotReservation,
}

impl AllocationGrant {
    pub(super) fn new(kv: DevicePageReservation, rs: RsSlotReservation) -> Self {
        Self { kv, rs }
    }

    pub fn empty() -> Self {
        Self {
            kv: DevicePageReservation::empty(),
            rs: RsSlotReservation::empty(),
        }
    }

    pub fn remaining_kv(&self) -> usize {
        self.kv.len()
    }

    pub fn remaining_rs(&self) -> RsUnits {
        self.rs.grant.units()
    }

    pub fn lend_kv(&mut self) -> &mut Vec<PhysicalKvPageId> {
        self.kv.lend()
    }

    pub fn lend_rs(&mut self) -> &mut RsGrant {
        &mut self.rs.grant
    }
}

/// What the holder did not spend goes back to the pool first, then the
/// planner is told ([`RoomReturns`]): a head that room serves is planned
/// at once, not at the next unrelated completion.
impl Drop for AllocationGrant {
    fn drop(&mut self) {
        let unspent = self.kv.len() > 0 || !self.rs.grant.is_empty();
        let returns = self.kv.returns.take().or_else(|| self.rs.returns.take());
        drop(std::mem::take(&mut self.kv));
        drop(std::mem::replace(&mut self.rs, RsSlotReservation::empty()));
        if unspent && let Some(returns) = returns {
            returns.given();
        }
    }
}
