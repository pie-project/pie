//! Whether a launch may go to the workers at all. Whether there is room
//! for it is the cluster policy's (`placement/workers`, an empty row holds
//! it); the gateway refuses only what no worker could take.

use controller_api::{Health, RoutingTable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Reject(RejectReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    NoHealthyWorker,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::NoHealthyWorker => f.write_str("no healthy worker"),
        }
    }
}

pub fn admit(table: &RoutingTable) -> AdmissionDecision {
    if table.workers.iter().any(|w| w.health == Health::Healthy) {
        AdmissionDecision::Admit
    } else {
        AdmissionDecision::Reject(RejectReason::NoHealthyWorker)
    }
}
