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
    },
    Restore {
        demand: Demand,
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
        }
    }

    pub(super) fn kv_need(&self) -> u32 {
        match &self.kind {
            WaitKind::Allocation { demand, .. } | WaitKind::Restore { demand } => demand.kv_pages,
        }
    }

    pub(super) fn rs_need(&self) -> RsUnits {
        match &self.kind {
            WaitKind::Allocation { demand, .. } | WaitKind::Restore { demand } => demand.rs,
        }
    }

    pub(super) fn kind_name(&self) -> &'static str {
        match self.kind {
            WaitKind::Allocation { .. } => "allocation",
            WaitKind::Restore { .. } => "restore",
        }
    }
}
