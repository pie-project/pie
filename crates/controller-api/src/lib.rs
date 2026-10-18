use serde::{Deserialize, Serialize};

pub use ids::{GatewayId, NodeId, WorkerId};

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
    pub addr: String,
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
pub struct NeighborPeer {
    pub id: WorkerId,
    pub addr: String,
    pub role: Role,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayEndpoint {
    pub id: GatewayId,
    pub addr: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighbors {
    pub epoch: u64,
    pub peers: Vec<NeighborPeer>,
    pub gateways: Vec<GatewayEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutableWorker {
    pub id: WorkerId,
    pub addr: String,
    pub role: Role,
    pub model: String,
    pub health: Health,
    pub coarse_load: WorkerStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingTable {
    pub epoch: u64,
    pub workers: Vec<RoutableWorker>,
}

#[tarpc::service]
pub trait Control {
    async fn register_worker(info: WorkerInfo) -> WorkerId;

    async fn register_gateway(info: GatewayInfo) -> GatewayId;

    async fn heartbeat(id: NodeId) -> Ack;

    async fn report_worker(id: WorkerId, status: WorkerStatus);

    async fn watch_worker(id: WorkerId, since: u64) -> Neighbors;

    async fn watch_gateway(since: u64) -> RoutingTable;
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
            workers: vec![RoutableWorker {
                id: WorkerId(3),
                addr: "10.0.0.4:7000".into(),
                role: Role::Decode,
                model: "llama3-8b".into(),
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
