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

/// A pass woken by an arrival or a call of its own.
pub(super) const WAKE_CALL: u8 = 0;
/// By room given back meeting the head's want (the pool's room signal).
pub(super) const WAKE_SIGNAL: u8 = 1;
/// By a poke: a fire landing, a move ending, pages freed.
pub(super) const WAKE_POKE: u8 = 2;
/// By the beat.
pub(super) const WAKE_BEAT: u8 = 3;

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
    /// Device slots come free on any path (the pool's release or a fire's
    /// epoch retiring, `SlotPool::set_room_signal`), counted.
    freed: AtomicU64,
    /// The count of [`Self::freed`] the last plan pass began on: moved
    /// since, the drain task's tick plans again (the backstop).
    freed_at_plan: AtomicU64,
    /// A plan pass runs: slots it frees itself (a stale reservation, pages
    /// gathered for a head that left) wake nothing, or a pass would wake
    /// its own next pass.
    pub(super) planning: std::sync::atomic::AtomicBool,
    /// What woke the next pass ([`WAKE_SIGNAL`] and the rest): what served
    /// a long wait is said by it.
    pub(super) wake_cause: std::sync::atomic::AtomicU8,
    /// What the queue's head lacks, as the last plan pass left it
    /// ([`super::ResidencyPlanner::publish_wants`]): the device's free
    /// chunks its runs take, the kv list's free slots its pages take, and
    /// the swap tiers' free slots its victims' moves wait on; `u64::MAX`
    /// where it lacks none of that kind. Room coming free wakes the planner
    /// only once a count crosses one: trickling frees that meet no need
    /// plan nothing (woken on each, the planner admitted what fit at that
    /// instant and split fires into smaller batches).
    pub(super) want_chunks: AtomicU64,
    pub(super) want_list: AtomicU64,
    pub(super) want_swap: AtomicU64,
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
    /// Device slots came free: counted, and, outside a plan pass, the
    /// planner woken while anyone waits. Lock-free: the pool's release
    /// calls it under its own lock and its caller's.
    pub(super) fn slots_freed(&self, count: crate::store::slots::RoomCount) {
        self.freed.fetch_add(1, Ordering::AcqRel);
        if self.planning.load(Ordering::Acquire) || self.waiters.load(Ordering::Acquire) == 0 {
            return;
        }
        if !self.met(count) {
            return;
        }
        self.wake_cause.store(WAKE_SIGNAL, Ordering::Release);
        match self.drain.get() {
            Some(drain) => drain.notify_one(),
            None => self.room.notify_waiters(),
        }
    }

    /// Whether `count` meets what the head lacks of its tier.
    pub(super) fn met(&self, count: crate::store::slots::RoomCount) -> bool {
        use crate::store::slots::RoomTier;
        match count.tier {
            RoomTier::Device => {
                count.chunks >= self.want_chunks.load(Ordering::Acquire)
                    || count.slots >= self.want_list.load(Ordering::Acquire)
            }
            RoomTier::Swap => count.slots >= self.want_swap.load(Ordering::Acquire),
        }
    }

    /// The head lacks nothing room coming free can meet.
    pub(super) fn want_nothing(&self) {
        self.want_chunks.store(u64::MAX, Ordering::Release);
        self.want_list.store(u64::MAX, Ordering::Release);
        self.want_swap.store(u64::MAX, Ordering::Release);
    }

    /// A plan pass begins: the frees it reads as given.
    pub(super) fn plan_begins(&self) {
        self.freed_at_plan
            .store(self.freed.load(Ordering::Acquire), Ordering::Release);
    }

    /// Slots came free since the last plan pass began.
    pub(super) fn freed_since_plan(&self) -> bool {
        self.freed.load(Ordering::Acquire) != self.freed_at_plan.load(Ordering::Acquire)
    }

    pub(super) fn given(&self) {
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
