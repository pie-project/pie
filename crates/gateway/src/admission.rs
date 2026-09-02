//! Resource-aware admission — the coarse, cluster-level gate, first step in
//! the per-turn admission -> route -> dispatch sequence. Gates on cluster
//! resources (KV-cache pressure, in-flight sequences) rather than request
//! volume, which the edge already gates.
//!
//! Deliberately coarse: "should the cluster admit this turn at all?". The
//! authoritative per-worker decision is the worker's own final admission
//! during dispatch, so a slightly stale [`RoutingTable`] here is safe.

use controller_api::{Health, RoutingTable};

/// Thresholds for the coarse cluster gate. Per-worker headroom is judged against
/// these; the cluster is admitted as long as *some* healthy worker has headroom.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionConfig {
    /// KV-pressure bucket (0 = empty headroom … 255 = saturated) at or above
    /// which a worker is considered to have no KV headroom.
    pub kv_saturate_bucket: u8,
    /// In-flight sequence count at or above which a worker is considered full.
    pub max_inflight_per_worker: u32,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        // Conservative: treat a worker as full only when genuinely near-saturated.
        Self {
            kv_saturate_bucket: 240,
            max_inflight_per_worker: 256,
        }
    }
}

/// The coarse gate's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// The cluster has headroom — proceed to routing.
    Admit,
    /// The cluster cannot take the turn right now.
    Reject(RejectReason),
}

/// Why admission declined a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// No healthy worker in the cluster has KV / in-flight headroom right now.
    ClusterSaturated,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::ClusterSaturated => {
                f.write_str("cluster saturated: no healthy worker has KV/seq headroom")
            }
        }
    }
}

/// Coarse cluster admission: admit iff *some* healthy worker still has both
/// KV and in-flight headroom. Admission only asks whether the cluster has
/// capacity; [`route`](crate::route) restricts the actual pick to `healthy
/// ∩ connected`.
pub fn admit(table: &RoutingTable, cfg: &AdmissionConfig) -> AdmissionDecision {
    let has_headroom = table.workers.iter().any(|w| {
        w.health == Health::Healthy
            && w.coarse_load.kv_pressure_bucket < cfg.kv_saturate_bucket
            && w.coarse_load.inflight < cfg.max_inflight_per_worker
    });
    if has_headroom {
        AdmissionDecision::Admit
    } else {
        AdmissionDecision::Reject(RejectReason::ClusterSaturated)
    }
}

