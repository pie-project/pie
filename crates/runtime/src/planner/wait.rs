use std::sync::Arc;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::ProcessId;
use super::error::PlannerError;
use super::grant::{AllocationGrant, Demand};
use super::rank::Rank;

/// Kept current by [`Inner::reschedule`]; the id alone names the entry.
pub(super) type EntryKey = (Rank, u64);

pub(super) enum WaitKind {
    Allocation {
        demand: Demand,
        notify: Arc<Notify>,
        outcome: Option<Result<AllocationGrant, PlannerError>>,
        yielded: bool,
        /// An activation rise's room ([`super::ResidencyPlanner::acquire_rise`]):
        /// its pages are given back at once, so it holds nothing past its
        /// grant, and the requester's own pages it moves are no ask of its.
        rise: bool,
    },
    Restore {
        demand: Demand,
    },
    /// A bind short of device room for its guest state, `pieces` runs of
    /// `run` adjacent chunks (`store::roles::RolesError::BindShort`):
    /// funded as a restore short of room for its roles is, idle cache
    /// first, then a resident parked by the planner's victim choice;
    /// `served` once the pool holds the runs free, the bind to take them.
    /// `pages`: the kv pages that, moved out whole, free its chunks.
    /// `withdrawn` once an eviction of its own process began
    /// ([`super::inner::Inner::begin_eviction`]): the room it waits for is
    /// asked again after its restore, and nothing is funded for it
    /// meanwhile.
    BindRoom {
        run: u32,
        pieces: u32,
        /// Single chunks beside its runs: its tables, rings and adapter.
        singles: u32,
        pages: u32,
        notify: Arc<Notify>,
        served: bool,
        withdrawn: bool,
        /// A rung raise short of room, its rung's tokens
        /// ([`super::ResidencyPlanner::fund_rise`]): served once the raise
        /// fits ([`super::PoolPort::rise_room`]), `pieces` (of a run of
        /// one) the free chunks it takes, kept current by each plan, so
        /// its victims are picked by the chunk map as a bind's are. Once
        /// served it holds the queue until the raise is taken (its waiter
        /// leaves): no head behind it is served into the room it was
        /// funded.
        rise: Option<u32>,
        /// It waits for what idle cache gives alone
        /// ([`super::ResidencyPlanner::wait_idle_room`]).
        idle: Option<IdleWait>,
    },
}

/// Where a bind's wait for idle cache's room stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IdleWait {
    Waits,
    /// Idle cache gives no more.
    Out,
}

pub(super) struct Waiter {
    pub(super) pid: ProcessId,
    pub(super) kind: WaitKind,
    /// When it was queued: a wait past [`LONG_WAIT`] is said with what
    /// woke the pass that served it.
    pub(super) since: crate::rt::Instant,
    /// What it lacked the last time a pass ended with it the queue's head
    /// ([`super::ResidencyPlanner::publish_wants`]): said with a long
    /// wait, so a wait served by a poke names the lack the room signal
    /// did not measure. `"behind the head"` where it never headed.
    pub(super) lacked: &'static str,
    /// The wants published for it then (the device's free chunks, the kv
    /// list's free slots, the swap tiers' free slots; `u64::MAX` for none)
    /// and the free chunks and pages standing then: set beside the counts
    /// at its serve, a long wait says whether its room came as a count the
    /// signal measures.
    pub(super) wanted: ([u64; 3], [u32; 2]),
}

/// A wait served past [`LONG_WAIT`]: its kind, what it lacked and the wants
/// published for it as it last headed the queue ([`Waiter::lacked`],
/// [`Waiter::wanted`]), and how long.
pub(super) struct LongWait {
    pub(super) kind: &'static str,
    pub(super) lacked: &'static str,
    pub(super) wanted: ([u64; 3], [u32; 2]),
    pub(super) waited: std::time::Duration,
}

/// A wait long enough that what finally served it is said: a field's gaps
/// between dispatches past it were what one serve waited on.
pub(super) const LONG_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

impl Waiter {
    pub(super) fn is_unmet(&self) -> bool {
        match &self.kind {
            WaitKind::Allocation {
                outcome, yielded, ..
            } => outcome.is_none() && !yielded,
            WaitKind::Restore { .. } => true,
            WaitKind::BindRoom {
                served, withdrawn, ..
            } => !served && !withdrawn,
        }
    }

    /// The pages its own room stands for, beside its kv (a bind's runs of
    /// chunks, [`WaitKind::BindRoom`]).
    /// A rung raise served and not yet taken: it holds the queue
    /// ([`WaitKind::BindRoom::rise`]).
    /// Served and not yet taken: an allocation's grant waiting for its
    /// waiter, a bind's room served and not taken.
    pub(super) fn untaken(&self) -> bool {
        match &self.kind {
            WaitKind::Allocation {
                outcome, yielded, ..
            } => outcome.is_some() && !yielded,
            WaitKind::Restore { .. } => false,
            WaitKind::BindRoom {
                served, withdrawn, ..
            } => *served && !withdrawn,
        }
    }

    pub(super) fn rise_untaken(&self) -> bool {
        matches!(
            self.kind,
            WaitKind::BindRoom {
                rise: Some(_),
                served: true,
                withdrawn: false,
                ..
            }
        )
    }

    pub(super) fn room_pages(&self) -> u32 {
        match &self.kind {
            WaitKind::BindRoom { pages, .. } => *pages,
            _ => 0,
        }
    }

    pub(super) fn kv_need(&self) -> u32 {
        match &self.kind {
            WaitKind::Allocation { demand, .. } | WaitKind::Restore { demand } => demand.kv_pages,
            WaitKind::BindRoom { .. } => 0,
        }
    }

    pub(super) fn rs_need(&self) -> RsUnits {
        match &self.kind {
            WaitKind::Allocation { demand, .. } | WaitKind::Restore { demand } => demand.rs,
            WaitKind::BindRoom { .. } => RsUnits::default(),
        }
    }

    pub(super) fn kind_name(&self) -> &'static str {
        match self.kind {
            WaitKind::Allocation { .. } => "allocation",
            WaitKind::Restore { .. } => "restore",
            WaitKind::BindRoom { rise: Some(_), .. } => "rise room",
            WaitKind::BindRoom { .. } => "bind room",
        }
    }
}
