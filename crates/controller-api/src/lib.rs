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

/// What a worker holds, as it tells the controller's directory: the sealed
/// pages it gained and lost by key since its last report (all of them when
/// `full`), the runs it keeps whole from token 0 (gained in full, lost by
/// their last key), and where it serves its pages' bytes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateReport {
    pub peer: Option<String>,
    /// Where this worker's peer link listens: other workers send it passes
    /// there.
    #[serde(default)]
    pub link: Option<String>,
    pub added: Vec<[u8; 32]>,
    pub removed: Vec<[u8; 32]>,
    pub full: bool,
    pub runs_added: Vec<Vec<[u8; 32]>>,
    pub runs_removed: Vec<[u8; 32]>,
    /// A pool: it loads no model and keeps what it is sent.
    pub pool: bool,
    /// What the worker policy published for the cluster plane, as
    /// `(subject, name, JSON value)`; all of it, whenever any changes.
    pub published: Option<Vec<(String, String, String)>>,
    /// Pages the persistent disk tier gained and lost a copy of since the
    /// last report (all of it when `full`).
    #[serde(default)]
    pub disk_added: Vec<[u8; 32]>,
    #[serde(default)]
    pub disk_removed: Vec<[u8; 32]>,
    /// The runs whose owner asked a prefix of them kept on disk: each run's
    /// last key and the tokens asked; all of it, whenever any changes.
    #[serde(default)]
    pub persist_hints: Option<Vec<([u8; 32], u64)>>,
    /// The tokens one kv page holds, and whether the disk tier is kept
    /// across runs; sent with a full report.
    #[serde(default)]
    pub disk_tier: Option<(u32, bool)>,
}

/// A sealed page's key.
pub type Key = [u8; 32];

/// A run some worker keeps whole, by its last key: the workers (pools among
/// them) that hold every page of it, those whose persistent disk tier does,
/// and the most leading tokens of it some owner asked kept on disk.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupView {
    pub keys: Vec<Key>,
    pub sites: Vec<WorkerId>,
    pub disk_sites: Vec<WorkerId>,
    pub persist_tokens: Option<u64>,
}

/// The controller's directory as the gateways read it: what each worker
/// holds, as of `epoch`. Soft state: rebuilt from the workers' reports.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DirectoryView {
    pub epoch: u64,
    pub groups: Vec<(Key, GroupView)>,
    pub pools: Vec<WorkerId>,
    /// Where each worker serves its pages' bytes.
    pub peers: Vec<(WorkerId, String)>,
    /// What each worker's policy published about the worker itself, as
    /// `(name, JSON value)`.
    pub published: Vec<(WorkerId, Vec<(String, String)>)>,
    /// Each worker's kv page in tokens, and whether its disk tier is kept.
    pub disk_tiers: Vec<(WorkerId, (u32, bool))>,
    /// The pages each worker's persistent disk tier keeps a copy of.
    pub disk: Vec<(WorkerId, Vec<Key>)>,
}

/// How a process ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Completed,
    Failed,
    Cancelled,
}

/// Props as JSON text by name: this crate takes no `client-api`.
pub type JsonProps = Vec<(String, String)>;

/// What a gateway tells the cluster policy about its processes, in order. A
/// process is `req`, the gateway's own id for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessEvent {
    /// A launch at the gate: `seq` counts this gateway's arrivals. The
    /// prompt travels as the blocks its text hashes to, not as text.
    Arrived {
        req: u64,
        seq: u64,
        tenant: String,
        session: Option<String>,
        props: JsonProps,
        prompt: Option<Vec<u64>>,
    },
    Dispatched {
        req: u64,
        worker: WorkerId,
    },
    Counted {
        req: u64,
        prompt: u64,
        generated: u64,
    },
    /// A running process moved to `worker`, or (none) could not.
    Moved {
        req: u64,
        worker: Option<WorkerId>,
        why: String,
    },
    Exited {
        req: u64,
        outcome: Outcome,
    },
    /// `<tenant>/<session>`'s props changed: `null` removes one.
    SessionProps {
        id: String,
        changes: JsonProps,
    },
    SessionClosed {
        id: String,
    },
    /// The workers linked to this gateway now.
    Workers(Vec<WorkerId>),
}

/// Where the cluster policy puts a process.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementView {
    pub worker: Option<WorkerId>,
    /// Passed on with the process: the worker plane reads `cluster/<name>`.
    pub props: JsonProps,
    /// Workers its passes may run on, best first.
    pub executors: Vec<WorkerId>,
}

/// A copy of a group a worker is to take, carried by one gateway: workers
/// dial gateways, not the controller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicateOrder {
    pub group: String,
    pub worker: WorkerId,
    pub keys: Vec<Key>,
    /// Where a worker that holds it serves its pages; none for a copy the
    /// worker makes from its own (`disk`).
    pub from: String,
    pub retention: i64,
    pub disk: bool,
    /// The table that wants it, for the log.
    pub table: String,
}

/// The order launches leave a gateway's gate in: each process's key (one
/// not named is 0), and the key below which a process is held.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleView {
    pub keys: Vec<(u64, i64)>,
    pub floor: Option<i64>,
}

/// The cluster policy's decisions about one gateway's processes,
/// level-triggered, as of `epoch`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GatewayDecisions {
    pub epoch: u64,
    /// Cluster policies run: the gateway tells them of its processes.
    pub active: bool,
    /// How long a launch waits for the programs to hear it, in ms.
    pub gate_ms: u64,
    /// The highest arrival `seq` every program has heard, its decisions
    /// drained; `u64::MAX` with no program to hear it.
    pub heard: u64,
    /// The gate's order. None: arrival order.
    pub schedule: Option<ScheduleView>,
    pub placements: Vec<(u64, PlacementView)>,
    /// Processes to end, with the reason: refused at the gate, cancelled
    /// once admitted.
    pub terminated: Vec<(u64, String)>,
    /// For the gateway that carries worker calls: the copies to ask for.
    pub replicate: Vec<ReplicateOrder>,
}

#[tarpc::service]
pub trait Control {
    async fn register_worker(info: WorkerInfo) -> WorkerId;

    async fn register_gateway(info: GatewayInfo) -> GatewayId;

    async fn heartbeat(id: NodeId) -> Ack;

    async fn report_worker(id: WorkerId, status: WorkerStatus);

    async fn watch_worker(id: WorkerId, since: u64) -> Neighbors;

    async fn watch_gateway(since: u64) -> RoutingTable;

    /// Each shape the scale table names, desired against current.
    async fn scale_status() -> Vec<ScaleStatus>;

    /// What `worker` holds, gained and lost since its last report: the
    /// directory's one input. A pool's comes through the gateway it links to.
    async fn report_state(worker: WorkerId, report: StateReport);

    /// `worker` holds nothing any more: its link to the gateway that relayed
    /// its reports went.
    async fn forget_state(worker: WorkerId);

    /// The workers other than `asker`, reachable by their peers, that hold a
    /// run of `keys` from the first: id, address and run length, longest
    /// first.
    async fn locate(asker: WorkerId, keys: Vec<Key>) -> Vec<(WorkerId, String, u32)>;

    /// Where `worker`'s peer link listens.
    async fn link(worker: WorkerId) -> Option<String>;

    /// The directory once its epoch is past `since`; long-polled.
    async fn watch_directory(since: u64) -> DirectoryView;

    /// What happened to `gateway`'s processes, in order: the cluster
    /// policy's input. A gateway sends one batch at a time.
    async fn cluster_events(gateway: GatewayId, events: Vec<ProcessEvent>);

    /// The cluster policy's decisions about `gateway`'s processes once their
    /// epoch is past `since`; long-polled.
    async fn watch_decisions(gateway: GatewayId, since: u64) -> GatewayDecisions;
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
                engine: "cuda".into(),
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
