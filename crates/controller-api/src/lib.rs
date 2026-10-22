use serde::{Deserialize, Serialize};

pub use ids::{GatewayId, NodeId, WorkerId};

/// The environment variable a worker the controller launched finds its
/// launch token in ([`WorkerInfo::launch`]).
pub const LAUNCH_ENV: &str = "PIE_LAUNCH";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[serde(alias = "Prefill")]
    Prefill,
    #[serde(alias = "Decode")]
    Decode,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Health {
    Healthy,
    Degraded,
    Unreachable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub role: Role,
    pub model: String,
    /// The engine kind it serves on, as its config spells it (`[engine]
    /// type`): with the model, the shape a scale table counts it under.
    #[serde(default)]
    pub engine: String,
    pub addr: String,
    /// The token the controller's actuator launched it with
    /// (`PIE_LAUNCH`), so the controller knows which of its launches it is.
    #[serde(default)]
    pub launch: Option<String>,
    /// Its process id, so a controller that did not spawn it, restarted
    /// after the one that did, can still signal a launch it adopts.
    #[serde(default)]
    pub pid: Option<u32>,
    /// The id this worker had, registering again (a controller restarted,
    /// or one that timed it out): granted back if no one holds it, so the
    /// worker keeps its links and its turns.
    #[serde(default)]
    pub prior: Option<WorkerId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayInfo {
    pub addr: String,
    /// The id this gateway had, registering again; as for a worker.
    #[serde(default)]
    pub prior: Option<GatewayId>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub kv_pressure_bucket: u8,
    pub inflight: u32,
    #[serde(default)]
    pub running: u32,
    #[serde(default)]
    pub waiting: u32,
    /// Cached KV counts as free; 0 when unknown.
    #[serde(default)]
    pub capacity_tokens: u64,
    #[serde(default)]
    pub used_tokens: u64,
    /// Taking no new turns: it is about to go, once what it runs has moved
    /// or ended.
    #[serde(default)]
    pub draining: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ack {
    Ok,
    ReRegister,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayEndpoint {
    pub id: GatewayId,
    pub addr: String,
}

/// What a worker hears from the controller: the gateways to link to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighbors {
    pub epoch: u64,
    pub gateways: Vec<GatewayEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableWorker {
    pub id: WorkerId,
    pub addr: String,
    pub role: Role,
    pub model: String,
    #[serde(default)]
    pub engine: String,
    pub health: Health,
    pub coarse_load: WorkerStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingTable {
    pub epoch: u64,
    pub workers: Vec<RoutableWorker>,
    /// The controller launches and stops workers to meet a scale table: it
    /// has slots to launch into (`cluster.scale`).
    #[serde(default)]
    pub scales: bool,
    /// The gateway that authors the cluster-wide tables (scale,
    /// replication, retention): the live one with the lowest id. Every
    /// gateway runs its cluster policies; only this one enacts those
    /// tables, and the controller takes a scale table only from it.
    #[serde(default)]
    pub author: Option<GatewayId>,
}

/// A shape of worker, and how many of it the cluster should run.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScaleEntry {
    /// The model as workers serve it (`[model] name`); None: workers that
    /// load no model.
    pub model: Option<String>,
    pub engine: String,
    pub workers: u32,
}

/// Where a shape a scale table names stands, for whoever converges on it:
/// the controller's own actuator, or an operator in its place.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaleStatus {
    pub model: Option<String>,
    pub engine: String,
    pub desired: u32,
    /// Registered and taking turns.
    pub serving: u32,
    /// Launched by the actuator and not registered yet.
    pub launching: u32,
    /// Stopped by the actuator and still draining.
    pub stopping: u32,
    /// Slots of the shape the actuator has free to launch into.
    pub free: u32,
}

#[tarpc::service]
pub trait Control {
    async fn register_worker(info: WorkerInfo) -> WorkerId;

    async fn register_gateway(info: GatewayInfo) -> GatewayId;

    async fn heartbeat(id: NodeId) -> Ack;

    async fn report_worker(id: WorkerId, status: WorkerStatus);

    async fn watch_worker(id: WorkerId, since: u64) -> Neighbors;

    async fn watch_gateway(since: u64) -> RoutingTable;

    /// The scale table a cluster policy installed: level-triggered, the
    /// whole table each time; a shape it does not name keeps what it has.
    /// Taken only from `gateway` while it is the author
    /// ([`RoutingTable::author`]); from any other, ignored.
    async fn desire_scale(gateway: GatewayId, entries: Vec<ScaleEntry>);

    /// Each shape the scale table names, desired against current.
    async fn scale_status() -> Vec<ScaleStatus>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lib_every_case() {
        role_parses_cli_spelling();
        routing_table_serde_round_trip();
        node_id_routes_either_kind();
    }

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

    fn routing_table_serde_round_trip() {
        let table = RoutingTable {
            epoch: 7,
            scales: false,
            author: Some(GatewayId(2)),
            workers: vec![RoutableWorker {
                id: WorkerId(3),
                addr: "10.0.0.4:7000".into(),
                role: Role::Decode,
                model: "llama3-8b".into(),
                engine: "cuda_native".into(),
                health: Health::Healthy,
                coarse_load: WorkerStatus {
                    kv_pressure_bucket: 42,
                    inflight: 5,
                    ..WorkerStatus::default()
                },
            }],
        };

        let json = serde_json::to_string(&table).expect("serialize");
        let back: RoutingTable = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(table, back);
    }

    fn node_id_routes_either_kind() {
        assert_eq!(NodeId::from(WorkerId(1)), NodeId::Worker(WorkerId(1)));
        assert_eq!(NodeId::from(GatewayId(2)), NodeId::Gateway(GatewayId(2)));
    }
}
