//! Resource-aware admission — the coarse, cluster-level gate (design §7, first
//! step in the per-turn `admission → route → dispatch` sequence).
//!
//! This is the gateway's "real rate limit": it gates on cluster **resources**
//! (KV-cache pressure, in-flight sequences) rather than request volume — the
//! thing the edge structurally cannot do (a request can be 100 or 100k tokens,
//! so req/s does not describe load; the edge gates *volume*, the gateway gates
//! *resources*, and they do not overlap).
//!
//! It is deliberately **coarse**: "should the cluster admit this turn at all?".
//! The authoritative, per-worker decision is the worker's own final admission
//! ([`Accepted::Reject`](pie_schema::gateway::Accepted) during dispatch), so a
//! slightly stale [`RoutingTable`] here is safe. v1 reads only the controller's
//! pushed coarse load (`RoutingTable.coarse_load`, already flowing via
//! `watch_gateway`); real per-tenant token-budget accounting is a later
//! graduation that grows its own inputs (it does not belong on the spine).
//!
//! Reached through [`RoutingHandle::admit`](crate::route::RoutingHandle::admit)
//! so the cluster-table read is single-sourced off the one routing watch; the
//! logic itself lives here as its own §7 step.

use pie_schema::control::{Health, RoutingTable};

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
        // Conservative spine defaults: treat a worker as full only when it is
        // genuinely near-saturated, so admission rejects only when the *whole*
        // healthy fleet is out of headroom (the worker's own per-turn admission
        // is the precise backstop for everything short of that).
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

/// Coarse cluster admission: admit iff *some* healthy worker still has both KV
/// and in-flight headroom. Reads only the controller-pushed `RoutingTable`
/// coarse load (no per-request cost in v1). A worker that is healthy in the
/// table but not yet dialed-in is still counted here — admission only asks
/// whether the cluster has capacity; [`route`](crate::route) is what restricts
/// the actual pick to `healthy ∩ connected`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use pie_schema::control::{Role, RoutableWorker, WorkerId, WorkerStatus};

    fn worker(id: u64, health: Health, kv: u8, inflight: u32) -> RoutableWorker {
        RoutableWorker {
            id: WorkerId(id),
            addr: format!("10.0.0.{id}:7000"),
            role: Role::Decode,
            model: "m".to_string(),
            health,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: kv,
                inflight,
            },
        }
    }

    fn table(workers: Vec<RoutableWorker>) -> RoutingTable {
        RoutingTable { epoch: 1, workers }
    }

    #[test]
    fn admits_when_some_healthy_worker_has_headroom() {
        let cfg = AdmissionConfig::default();
        let t = table(vec![
            worker(1, Health::Healthy, 250, 300), // saturated
            worker(2, Health::Healthy, 100, 5),   // headroom ✓
        ]);
        assert_eq!(admit(&t, &cfg), AdmissionDecision::Admit);
    }

    #[test]
    fn rejects_when_all_healthy_workers_are_saturated() {
        let cfg = AdmissionConfig::default();
        let t = table(vec![
            worker(1, Health::Healthy, 250, 1),   // kv over bucket
            worker(2, Health::Healthy, 10, 300),  // inflight over cap
            worker(3, Health::Unreachable, 0, 0), // headroom but not healthy
        ]);
        assert_eq!(
            admit(&t, &cfg),
            AdmissionDecision::Reject(RejectReason::ClusterSaturated)
        );
    }

    #[test]
    fn rejects_on_empty_table() {
        assert_eq!(
            admit(&table(vec![]), &AdmissionConfig::default()),
            AdmissionDecision::Reject(RejectReason::ClusterSaturated)
        );
    }

    #[test]
    fn unhealthy_headroom_does_not_admit() {
        let cfg = AdmissionConfig::default();
        let t = table(vec![worker(1, Health::Degraded, 0, 0)]);
        assert_eq!(
            admit(&t, &cfg),
            AdmissionDecision::Reject(RejectReason::ClusterSaturated)
        );
    }
}
