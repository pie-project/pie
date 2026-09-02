//! The controller's control-plane RPC contract. Workers and gateways dial the
//! controller through the macro-generated [`ControlClient`]; the controller
//! implements [`Control`], pushing each worker its [`Neighbors`] and each
//! gateway the [`RoutingTable`] via long-poll watches, and tracking liveness
//! from heartbeats.

use serde::{Deserialize, Serialize};

pub use ids::{GatewayId, NodeId, WorkerId};

// ──────────────────────────── role / health ───────────────────────────

/// What stage of inference a worker serves. Declared once at registration and
/// immutable thereafter (a worker re-registers to change role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Consumes prompt tokens and produces the initial KV state.
    #[serde(alias = "Prefill")]
    Prefill,
    /// Consumes KV state and produces output tokens step by step.
    #[serde(alias = "Decode")]
    Decode,
    /// Encodes non-text modalities (image / audio) into embeddings.
    #[serde(alias = "Encode")]
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

impl std::str::FromStr for Role {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "encode" => Ok(Self::Encode),
            other => Err(format!(
                "invalid worker role {other:?}; expected decode, prefill, or encode"
            )),
        }
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
}

/// Static identity a gateway declares when it joins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayInfo {
    /// The gateway's worker-facing dial-in endpoint (not the client edge).
    /// Republished in each worker's [`Neighbors`] gateway roster.
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

/// One gateway a worker should dial into. The roster is global (every worker
/// dials the same live set); keyed by `addr`, `id` is for observability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayEndpoint {
    pub id: GatewayId,
    pub addr: String,
}

/// A worker's scoped view, pushed by `watch_worker`: its neighbor peers plus
/// the global gateway roster. `epoch` is the membership cursor re-polled via
/// `since`; gateway join/leave bumps the same epoch.
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

/// The gateway's global view, pushed by `watch_gateway`: the full worker
/// roster and its coarse load. Every gateway gets the same view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingTable {
    pub epoch: u64,
    pub workers: Vec<RoutableWorker>,
}

// ──────────────────────────── Control service ─────────────────────────

/// The controller's RPC surface. Registry of workers + gateways; pushes
/// neighbor views / routing tables via long-poll watches; tracks liveness.
#[tarpc::service]
pub trait Control {
    /// Register a worker; returns its controller-minted [`WorkerId`].
    async fn register_worker(info: WorkerInfo) -> WorkerId;

    /// Register a gateway; returns its controller-minted [`GatewayId`].
    async fn register_gateway(info: GatewayInfo) -> GatewayId;

    /// Liveness ping from either node kind. [`Ack::ReRegister`] means the
    /// controller has no record of this id and it must re-register.
    async fn heartbeat(id: NodeId) -> Ack;

    /// Push a worker's coarse load; separate from `heartbeat` so load updates
    /// don't disturb membership.
    async fn report_worker(id: WorkerId, status: WorkerStatus);

    /// Long-poll a worker's neighbor view; blocks until epoch advances past `since`.
    async fn watch_worker(id: WorkerId, since: u64) -> Neighbors;

    /// Long-poll the global routing table; blocks until epoch advances past `since`.
    async fn watch_gateway(since: u64) -> RoutingTable;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_parses_cli_spelling() {
        assert_eq!("decode".parse::<Role>().unwrap(), Role::Decode);
        assert_eq!("PREFILL".parse::<Role>().unwrap(), Role::Prefill);
        assert_eq!("encode".parse::<Role>().unwrap(), Role::Encode);
        assert!("worker".parse::<Role>().is_err());
        assert_eq!(
            serde_json::from_str::<Role>("\"Decode\"").unwrap(),
            Role::Decode
        );
        assert_eq!(
            serde_json::from_str::<Role>("\"prefill\"").unwrap(),
            Role::Prefill
        );
    }

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
