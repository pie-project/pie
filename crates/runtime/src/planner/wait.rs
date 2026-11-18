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
    BindRoom {
        run: u32,
        pieces: u32,
        pages: u32,
        notify: Arc<Notify>,
        served: bool,
    },
}

pub(super) struct Waiter {
    pub(super) pid: ProcessId,
    pub(super) kind: WaitKind,
}

impl Waiter {
    pub(super) fn is_unmet(&self) -> bool {
        match &self.kind {
            WaitKind::Allocation {
                outcome, yielded, ..
            } => outcome.is_none() && !yielded,
            WaitKind::Restore { .. } => true,
            WaitKind::BindRoom { served, .. } => !served,
        }
    }

    /// The pages its own room stands for, beside its kv (a bind's runs of
    /// chunks, [`WaitKind::BindRoom`]).
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
            WaitKind::BindRoom { .. } => "bind room",
        }
    }
}
