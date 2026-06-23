//! Control-plane wire contract — the disaggregated-serving controller.
//!
//! The controller keeps a registry of workers + gateways, pushes each worker
//! its `Neighbors` (TP siblings + prefill↔decode partners) and each gateway the
//! `RoutingTable` (worker roster + coarse load) via long-poll watches, and
//! tracks liveness from heartbeats. These are the shared data types those calls
//! carry; the `#[tarpc::service] trait Control` that carries them lives in the
//! thin `pie-control` crate so this floor stays tarpc-free.
//!
//! Like [`cluster`](crate::cluster)/[`kv`](crate::kv)/[`message`](crate::message),
//! this is plain serde control vocabulary — deliberately NOT `#[schema]`/rkyv
//! (it never rides the zero-copy tensor ring), so it is NOT part of
//! `SCHEMA_HASH`.
//!
//! These types are reached as `pie_schema::control::*` (NOT flat-re-exported at
//! the crate root) because the names `WorkerId`/`WorkerInfo`/`Role` deliberately
//! shadow the legacy [`cluster`](crate::cluster) ones this redesign replaces;
//! the module path keeps the two contracts unambiguous until the legacy control
//! bits are removed.

use serde::{Deserialize, Serialize};

use crate::capabilities::DriverCapabilities;

// ───────────────────────────── opaque ids ─────────────────────────────

/// Controller-minted, cluster-unique worker handle. Newtype so it can never be
/// confused with a gateway id or any other counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub u64);

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "worker#{}", self.0)
    }
}

/// Controller-minted, cluster-unique gateway handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GatewayId(pub u64);

impl std::fmt::Display for GatewayId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gateway#{}", self.0)
    }
}

/// Either kind of cluster member, carried by the unified [`heartbeat`] call so
/// the controller can route liveness to the right registry. Workers and gateways
/// are minted into separate id spaces, so a single flat counter can't identify a
/// node — hence an enum, not a bare newtype.
///
/// [`heartbeat`]: ../../pie_control/trait.Control.html
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeId {
    Worker(WorkerId),
    Gateway(GatewayId),
}

impl From<WorkerId> for NodeId {
    fn from(id: WorkerId) -> Self {
        NodeId::Worker(id)
    }
}

impl From<GatewayId> for NodeId {
    fn from(id: GatewayId) -> Self {
        NodeId::Gateway(id)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeId::Worker(id) => write!(f, "{id}"),
            NodeId::Gateway(id) => write!(f, "{id}"),
        }
    }
}

// ──────────────────────────── role / health ───────────────────────────

/// What stage of inference a worker serves. Declared once at registration and
/// immutable thereafter (a worker re-registers to change role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Consumes prompt tokens and produces the initial KV state.
    Prefill,
    /// Consumes KV state and produces output tokens step by step.
    Decode,
    /// Encodes non-text modalities (image / audio) into embeddings.
    Encode,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Role::Prefill => "prefill",
            Role::Decode => "decode",
            Role::Encode => "encode",
        })
    }
}

/// Liveness verdict the controller derives from heartbeat receipt time
/// (controller-side clock — no worker-clock skew). Surfaced to gateways so
/// routing can avoid degraded/unreachable workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Health {
    /// Heartbeats are arriving on time.
    Healthy,
    /// Heartbeats are late but the node has not yet timed out.
    Degraded,
    /// No heartbeat within the timeout window.
    Unreachable,
}

// ─────────────────────────── registration info ────────────────────────

/// Static identity a worker declares when it joins. Dynamic load is pushed
/// separately as [`WorkerStatus`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Inference stage this worker serves (immutable for the worker's lifetime).
    pub role: Role,
    /// Model the worker serves (e.g. `"llama3-8b"`).
    pub model: String,
    /// Where peers reach this worker's control/data endpoint
    /// (e.g. `"10.0.0.4:7000"`).
    pub addr: String,
    /// What the worker's driver can do (page geometry, forward limits, arch,
    /// …) — the existing driver-handshake capability descriptor.
    pub capability: DriverCapabilities,
}

/// Static identity a gateway declares when it joins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayInfo {
    /// Where the gateway is reachable (e.g. `"10.0.0.9:8080"`).
    pub addr: String,
}

// ──────────────────────────── reported load ───────────────────────────

/// Coarse, frequently-pushed load a worker reports. Intentionally low-cardinality
/// so the controller can coalesce/route on it without churn (the KV pressure is a
/// quantized bucket, not a raw page count).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerStatus {
    /// Quantized KV-cache pressure bucket (0 = empty headroom … 255 = saturated).
    pub kv_pressure_bucket: u8,
    /// In-flight requests on this worker.
    pub inflight: u32,
}

/// Heartbeat reply. `ReRegister` tells a node the controller has no record of it
/// (e.g. the controller restarted, soft-state lost) so it must re-register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ack {
    /// Liveness recorded; carry on.
    Ok,
    /// Unknown to the controller — re-register from scratch.
    ReRegister,
}

// ───────────────────────── pushed watch views ─────────────────────────

/// One peer in a worker's neighbor set. The worker groups these itself by
/// `role`: same-role+model peers are TP siblings; opposite-role peers are
/// prefill↔decode partners.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeighborPeer {
    pub id: WorkerId,
    pub addr: String,
    pub role: Role,
}

/// A worker's scoped view, pushed by `watch_worker`: who it should coordinate
/// with (TP group + prefill↔decode partners). `epoch` is the membership cursor
/// the worker re-polls with (`since`); the controller replies only once `epoch`
/// advances past the worker's last-seen value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighbors {
    pub epoch: u64,
    pub peers: Vec<NeighborPeer>,
}

/// One worker as seen by a gateway for routing decisions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableWorker {
    pub id: WorkerId,
    pub addr: String,
    pub role: Role,
    pub model: String,
    /// Liveness verdict (controller-derived).
    pub health: Health,
    /// Latest coarse load the worker reported.
    pub coarse_load: WorkerStatus,
}

/// The gateway's global view, pushed by `watch_gateway`: the full worker roster
/// + coarse load. `epoch` is the membership cursor the gateway re-polls with.
/// (Every gateway gets the same global view, so `watch_gateway` takes no id.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingTable {
    pub epoch: u64,
    pub workers: Vec<RoutableWorker>,
}
