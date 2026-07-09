//! pie-controller-rpc — the controller's control-plane RPC contract.
//!
//! Unifies two things that used to live apart:
//!   * the control-plane **data vocabulary** (registry / liveness / routing
//!     types, formerly the schema crate's `control` module), and
//!   * the `Control` **tarpc service** that carries them (formerly a separate
//!     thin RPC crate).
//!
//! Workers and gateways dial the controller through the macro-generated
//! [`ControlClient`]; the controller implements [`Control`]. The controller
//! keeps a registry of workers + gateways, pushes each worker its [`Neighbors`]
//! (TP siblings + prefill↔decode partners) and each gateway the [`RoutingTable`]
//! (worker roster + coarse load) via long-poll watches, and tracks liveness from
//! heartbeats.
//!
//! These are plain serde — deliberately NOT rkyv (`#[schema]`): the control
//! plane is cross-node, low-rate Rust↔Rust and never rides the zero-copy tensor
//! ring, so it is NOT part of `SCHEMA_HASH`. Cluster-unique id atoms
//! ([`WorkerId`]/[`GatewayId`]/[`NodeId`]) come from the leaf `pie-ids` crate and
//! are re-exported here for ergonomics.

use serde::{Deserialize, Serialize};

// Cross-node id atoms live in the leaf `pie-ids` crate; re-export them so
// consumers can still reach them as `pie_controller_rpc::{WorkerId, …}`.
pub use pie_ids::{GatewayId, NodeId, WorkerId};

// `DriverCapabilities` is owned by `pie-driver-abi` (`capabilities.rs`);
// `WorkerInfo.capability` carries it.
use pie_driver_abi::capabilities::DriverCapabilities;

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
    /// The gateway's **worker-facing dial-in** endpoint (e.g. `"10.0.0.9:8001"`)
    /// — the address workers dial INTO (M3 fan-in), NOT the client edge. The
    /// controller republishes this in each worker's [`Neighbors`] gateway roster
    /// so workers know where to dial.
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

/// One gateway a worker should dial INTO (M3 dial-in fan-in). The roster is
/// **global** — every worker dials the same live gateway set — so each worker's
/// [`Neighbors`] carries the identical list. Keyed by `addr` on the worker side
/// (the dial target); `id` is for logging/observability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayEndpoint {
    pub id: GatewayId,
    pub addr: String,
}

/// A worker's scoped view, pushed by `watch_worker`: who it should coordinate
/// with (TP group + prefill↔decode partners) plus the global gateway roster it
/// should dial INTO. `epoch` is the membership cursor the worker re-polls with
/// (`since`); the controller replies only once `epoch` advances past the
/// worker's last-seen value. Gateway join/leave bumps this same epoch, so a
/// worker learns the live gateway set over the connection it already long-polls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighbors {
    pub epoch: u64,
    pub peers: Vec<NeighborPeer>,
    /// The live gateway roster (global; same for every worker). The worker
    /// reconciles its dial-in links against this on each update.
    pub gateways: Vec<GatewayEndpoint>,
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
/// and its coarse load. `epoch` is the membership cursor the gateway re-polls
/// with. (Every gateway gets the same global view, so `watch_gateway` takes no
/// id.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingTable {
    pub epoch: u64,
    pub workers: Vec<RoutableWorker>,
}

// ──────────────────────────── Control service ─────────────────────────

/// The controller's RPC surface. Registry of workers + gateways; pushes neighbor
/// views / routing tables via long-poll watches; tracks liveness via heartbeats.
///
/// Workers and gateways dial the controller through the macro-generated
/// `ControlClient`; the controller implements this trait.
#[tarpc::service]
pub trait Control {
    /// Register a worker; returns its controller-minted [`WorkerId`]. The worker
    /// then calls `watch_worker(id, since = 0)` to get its initial neighbor view
    /// (returns immediately, since `0 < current_epoch`).
    async fn register_worker(info: WorkerInfo) -> WorkerId;

    /// Register a gateway; returns its controller-minted [`GatewayId`].
    async fn register_gateway(info: GatewayInfo) -> GatewayId;

    /// Liveness ping from either node kind (unified). [`Ack::ReRegister`] means
    /// the controller has no record of this id (it restarted / the node timed
    /// out) — the node must re-register. This is the sole eviction signal.
    async fn heartbeat(id: NodeId) -> Ack;

    /// Push a worker's coarse load (write-only, returns nothing). Separate from
    /// `heartbeat` so frequent load updates can be coalesced without disturbing
    /// membership.
    async fn report_worker(id: WorkerId, status: WorkerStatus);

    /// Long-poll a worker's neighbor view. Blocks until the worker epoch advances
    /// past `since`, then returns the scoped [`Neighbors`] (which carries the new
    /// epoch to re-poll with).
    async fn watch_worker(id: WorkerId, since: u64) -> Neighbors;

    /// Long-poll the global routing table. Blocks until the gateway epoch
    /// advances past `since`, then returns the [`RoutingTable`] (which carries
    /// the new epoch). Unscoped — every gateway gets the same global view.
    async fn watch_gateway(since: u64) -> RoutingTable;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_table_serde_round_trip() {
        let table = RoutingTable {
            epoch: 7,
            workers: vec![RoutableWorker {
                id: WorkerId(3),
                addr: "10.0.0.4:7000".into(),
                role: Role::Decode,
                model: "llama3-8b".into(),
                health: Health::Healthy,
                coarse_load: WorkerStatus {
                    kv_pressure_bucket: 42,
                    inflight: 5,
                },
            }],
        };

        let json = serde_json::to_string(&table).expect("serialize");
        let back: RoutingTable = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(table, back);
    }

    #[test]
    fn node_id_routes_either_kind() {
        assert_eq!(NodeId::from(WorkerId(1)), NodeId::Worker(WorkerId(1)));
        assert_eq!(NodeId::from(GatewayId(2)), NodeId::Gateway(GatewayId(2)));
    }
}
