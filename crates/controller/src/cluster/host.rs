//! The cluster policy on its own thread, in the controller: every gateway's
//! processes in one record, built from the events the gateways send, and
//! the programs' decisions about each gateway's processes kept as that
//! gateway's view ([`GatewayDecisions`]).
//!
//! A process is `"<gateway>.<req>"` to the programs: each gateway counts its
//! own requests from 0.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use client_api::Props;
use client_api::props::{PROMPT_BLOCK_BYTES, PromptIndex, TOKENS_PER_BYTE};
use controller_api::{
    DirectoryView, GatewayDecisions, Health, Key, PlacementView, ProcessEvent, ReplicateOrder,
    Role, RoutableWorker, RoutingTable, ScaleEntry, ScheduleView,
};
use ids::{GatewayId, WorkerId};
use tokio::sync::watch;

use super::program::{self, Commit, Fault, LoadError, Plane, Program, Table};
use super::wit::{self, Edge, Outcome, Subject, SubjectKind, Value};

/// What this host carries whatever its controller can do. `cluster.scale`
/// is not among them: it is there only while the controller has an actuator
/// to meet a scale table, so a policy `uses` it and reads it at run time.
fn capability(name: &str) -> bool {
    // Runs replicate to pools and to serving workers' host tiers; a group
    // retained above key 0 keeps a copy in a pool; a running turn that has
    // sent no output moves by ending where it runs and running again where
    // it goes; the workers a placement names run its process's passes.
    matches!(
        name,
        "cluster.hold"
            | "state.replicate"
            | "state.retain"
            | "placement.migrate"
            | "placement.executors"
    )
}

/// How often the policy hears what the directory holds, and its replication
/// and retention tables are enacted again.
const STATE_TICK: Duration = Duration::from_secs(1);

fn hex(key: &[u8; 32]) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Default)]
struct Tables {
    schedule: Option<Table>,
    placement: HashMap<String, wit::Placement>,
    /// Each process's executors, best first.
    executors: HashMap<String, Vec<String>>,
}

/// What the programs decided, kept for the gateways.
struct Shared {
    tables: RwLock<Tables>,
    /// The scale table, for the actuator: none until a program granted
    /// `scale` writes one.
    scale: watch::Sender<Option<Vec<ScaleEntry>>>,
    /// Each gateway's highest arrival every program has heard.
    heard: Mutex<HashMap<GatewayId, u64>>,
    /// No program runs: every gate is open.
    dead: AtomicBool,
    gate_ms: AtomicU64,
    /// Processes the programs ended, and why, until they exit.
    terminated: Mutex<BTreeMap<String, String>>,
    /// The copies the state tables want and the directory does not show,
    /// for the gateway that carries worker calls.
    orders: Mutex<Vec<ReplicateOrder>>,
    courier: Mutex<Option<GatewayId>>,
    views: Mutex<HashMap<GatewayId, watch::Sender<GatewayDecisions>>>,
}

/// The cluster policies, running.
pub struct Host {
    events: Sender<Event>,
    shared: Arc<Shared>,
}

enum Event {
    Process(GatewayId, ProcessEvent),
    /// Swap every program for these (name, component), loaded whole first:
    /// the answer is the new set's gate, or why it did not load.
    Reload {
        programs: Vec<(String, Vec<u8>)>,
        answer: tokio::sync::oneshot::Sender<Result<u64, LoadError>>,
    },
    /// The directory as of now.
    State(Arc<DirectoryView>),
}

/// A process's name to the programs.
fn pid(gateway: GatewayId, req: u64) -> String {
    format!("{}.{req}", gateway.0)
}

/// The gateway a process is at, and its id there.
fn at_gateway(pid: &str) -> Option<(GatewayId, u64)> {
    let (gateway, req) = pid.split_once('.')?;
    Some((GatewayId(gateway.parse().ok()?), req.parse().ok()?))
}

impl Host {
    /// Starts the cluster policies `programs` names, (spec, component), on
    /// one thread: each call goes to every one, and each writes what its
    /// manifest grants it, which no other may.
    pub fn start(
        programs: Vec<(String, Vec<u8>)>,
        routing: watch::Receiver<RoutingTable>,
        mut directory: watch::Receiver<DirectoryView>,
    ) -> Result<Arc<Self>, LoadError> {
        // A program is named by its built-in name, or by its file's stem: the
        // name keys what it publishes, so it holds no `/`.
        let programs: Vec<(String, Vec<u8>)> = programs
            .into_iter()
            .map(|(spec, component)| (program_name(&spec), component))
            .collect();
        let shared = Arc::new(Shared {
            tables: RwLock::default(),
            scale: watch::Sender::new(None),
            heard: Mutex::default(),
            dead: AtomicBool::new(false),
            gate_ms: AtomicU64::new(0),
            terminated: Mutex::default(),
            orders: Mutex::default(),
            courier: Mutex::default(),
            views: Mutex::default(),
        });
        let (events, rx) = channel();
        let names: Vec<String> = programs.iter().map(|(name, _)| name.clone()).collect();
        let manifests = program::spawn(programs, capability, "pie-cluster-policy", {
            let shared = shared.clone();
            move |programs| run(programs, rx, shared, routing)
        })?;
        shared
            .gate_ms
            .store(gate_of(manifests.iter()), Ordering::Release);

        let ticks = events.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(STATE_TICK).await;
                let view = Arc::new(directory.borrow_and_update().clone());
                if ticks.send(Event::State(view)).is_err() {
                    return;
                }
            }
        });
        tracing::info!(policies = ?names, "cluster policies started");
        Ok(Arc::new(Self { events, shared }))
    }

    /// What happened to `gateway`'s processes, in order.
    pub fn events(&self, gateway: GatewayId, events: Vec<ProcessEvent>) {
        for event in events {
            let _ = self.events.send(Event::Process(gateway, event));
        }
    }

    /// The programs' decisions about `gateway`'s processes, as they change.
    pub fn decisions(&self, gateway: GatewayId) -> watch::Receiver<GatewayDecisions> {
        let mut views = self.shared.views.lock().unwrap();
        if let Some(view) = views.get(&gateway) {
            return view.subscribe();
        }
        let mut view = decisions_of(&self.shared, gateway);
        view.epoch = 1;
        let (tx, rx) = watch::channel(view);
        views.insert(gateway, tx);
        rx
    }

    /// Swaps every running program for `programs` (spec, component), as a
    /// SIGHUP asks. The new set loads whole before anything changes; on an
    /// error the old one runs on. The old programs stop as a dead one does:
    /// their tables and what they published are withdrawn, and the others
    /// hear it. The new ones start with no state of their own, a program
    /// in both sets too, and read the cluster as it stands.
    pub async fn reload(&self, programs: Vec<(String, Vec<u8>)>) -> Result<(), LoadError> {
        if programs.is_empty() {
            return Err(LoadError::Instantiate(
                "a reload names at least one cluster policy; a controller with none starts \
                 without them"
                    .into(),
            ));
        }
        let programs = programs
            .into_iter()
            .map(|(spec, component)| (program_name(&spec), component))
            .collect();
        let (answer, answered) = tokio::sync::oneshot::channel();
        self.events
            .send(Event::Reload { programs, answer })
            .map_err(|_| LoadError::Instantiate("the policy thread has ended".into()))?;
        let gate_ms = answered
            .await
            .map_err(|_| LoadError::Instantiate("the policy thread has ended".into()))??;
        self.shared.gate_ms.store(gate_ms, Ordering::Release);
        publish(&self.shared);
        Ok(())
    }

    /// The scale table the policies installed (`install-scale`), whole;
    /// none while no program granted `scale` has written one.
    #[must_use]
    pub fn scale_watch(&self) -> watch::Receiver<Option<Vec<ScaleEntry>>> {
        self.shared.scale.subscribe()
    }
}

/// `gateway`'s view as the programs leave it now, epoch 0.
fn decisions_of(shared: &Shared, gateway: GatewayId) -> GatewayDecisions {
    let mine = |id: &str| {
        at_gateway(id)
            .filter(|(g, _)| *g == gateway)
            .map(|(_, req)| req)
    };
    let tables = shared.tables.read().unwrap();
    let schedule = tables.schedule.as_ref().map(|table| {
        let mut keys: Vec<(u64, i64)> = table
            .keys
            .iter()
            .filter_map(|(id, key)| Some((mine(id)?, *key)))
            .collect();
        keys.sort_unstable();
        ScheduleView {
            keys,
            floor: table.floor,
        }
    });
    let ids: BTreeSet<&String> = tables
        .placement
        .keys()
        .chain(tables.executors.keys())
        .collect();
    let placements = ids
        .into_iter()
        .filter_map(|id| {
            let req = mine(id)?;
            let placement = tables.placement.get(id);
            Some((
                req,
                PlacementView {
                    worker: placement
                        .and_then(|p| p.worker.as_deref()?.parse().ok())
                        .map(WorkerId),
                    props: placement
                        .map(|p| {
                            p.props
                                .iter()
                                .filter_map(|(name, value)| {
                                    Some((name.clone(), json(value)?.to_string()))
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    executors: tables
                        .executors
                        .get(id)
                        .into_iter()
                        .flatten()
                        .filter_map(|w| w.parse().ok().map(WorkerId))
                        .collect(),
                },
            ))
        })
        .collect();
    drop(tables);
    let terminated = shared
        .terminated
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(id, why)| Some((mine(id)?, why.clone())))
        .collect();
    let heard = if shared.dead.load(Ordering::Acquire) {
        u64::MAX
    } else {
        shared
            .heard
            .lock()
            .unwrap()
            .get(&gateway)
            .copied()
            .unwrap_or(0)
    };
    let replicate = if *shared.courier.lock().unwrap() == Some(gateway) {
        shared.orders.lock().unwrap().clone()
    } else {
        Vec::new()
    };
    GatewayDecisions {
        epoch: 0,
        active: true,
        gate_ms: shared.gate_ms.load(Ordering::Acquire),
        heard,
        schedule,
        placements,
        terminated,
        replicate,
    }
}

/// Each gateway's view, sent where it changed.
fn publish(shared: &Shared) {
    let gateways: Vec<GatewayId> = shared.views.lock().unwrap().keys().copied().collect();
    for gateway in gateways {
        let view = decisions_of(shared, gateway);
        if let Some(tx) = shared.views.lock().unwrap().get(&gateway) {
            tx.send_if_modified(|current| {
                let epoch = current.epoch;
                current.epoch = 0;
                let changed = *current != view;
                *current = GatewayDecisions {
                    epoch: if changed { epoch + 1 } else { epoch },
                    ..view
                };
                changed
            });
        }
    }
}

fn run(
    mut programs: Vec<Program>,
    events: Receiver<Event>,
    shared: Arc<Shared>,
    routing: watch::Receiver<RoutingTable>,
) {
    let mut registry = Registry::default();
    // Each gateway's last arrival heard, owed to its gate until the rounds
    // its calls' changes carry to the others have run too.
    let mut owed: HashMap<GatewayId, u64> = HashMap::new();
    // Each gateway's last arrival seen at all: what a set coming back to
    // life has heard.
    let mut last_seq: HashMap<GatewayId, u64> = HashMap::new();
    loop {
        // What the last round's calls changed reaches the others now.
        let carried = std::mem::take(&mut registry.carried);
        let first = if !carried.is_empty() {
            match events.try_recv() {
                Ok(event) => Some(event),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => return,
            }
        } else {
            match programs.iter().filter_map(Program::alarm).min() {
                Some(at_ms) => {
                    let wait = at_ms.saturating_sub(now_ms());
                    match events.recv_timeout(Duration::from_millis(wait)) {
                        Ok(event) => Some(event),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => return,
                    }
                }
                None => match events.recv() {
                    Ok(event) => Some(event),
                    Err(_) => return,
                },
            }
        };
        let mut edges = Vec::new();
        let mut heard: HashMap<GatewayId, u64> = HashMap::new();
        let mut reloads = Vec::new();
        registry.forget_exited();
        for event in first.into_iter().chain(events.try_iter()) {
            match event {
                Event::Reload { programs, answer } => reloads.push((programs, answer)),
                event => {
                    if let Event::Process(gateway, ProcessEvent::Arrived { seq, .. }) = &event {
                        let at = heard.entry(*gateway).or_default();
                        *at = (*at).max(*seq);
                        let last = last_seq.entry(*gateway).or_default();
                        *last = (*last).max(*seq);
                    }
                    registry.apply(event, &mut edges, &shared);
                }
            }
        }
        for (next, answer) in reloads {
            match program::load_all(&next, capability) {
                Ok(next) => {
                    let (old, new): (Vec<&str>, Vec<&str>) = (
                        programs.iter().map(Program::name).collect(),
                        next.iter().map(Program::name).collect(),
                    );
                    tracing::info!(?old, ?new, "cluster policies reloaded");
                    // The old ones stop as a dead one does.
                    for program in &programs {
                        withdraw(&shared, &mut registry, program.name(), program.manifest());
                    }
                    let gate_ms = gate_of(next.iter().map(Program::manifest));
                    programs = next;
                    if shared.dead.swap(false, Ordering::AcqRel) {
                        // The launches let through while none ran stay heard.
                        *shared.heard.lock().unwrap() = last_seq.clone();
                    }
                    let _ = answer.send(Ok(gate_ms));
                }
                Err(error) => {
                    tracing::error!(%error, "cluster policies not reloaded; the running ones go on");
                    let _ = answer.send(Err(error));
                }
            }
        }
        // A worker the controller now grades degraded is draining.
        let degraded: BTreeSet<String> = routing
            .borrow()
            .workers
            .iter()
            .filter(|w| w.health == Health::Degraded)
            .map(|w| w.id.0.to_string())
            .collect();
        for worker in degraded.difference(&registry.draining) {
            edges.push(Edge::WorkerDraining(worker.clone()));
        }
        registry.draining = degraded;

        let now = now_ms();
        let mut withdrawn = Vec::new();
        for (at, program) in programs.iter_mut().enumerate() {
            let mut edges = edges.clone();
            edges.extend(
                carried
                    .iter()
                    .filter(|(author, _)| author != program.name())
                    .map(|(_, edge)| edge.clone()),
            );
            if program.alarm().is_some_and(|at_ms| at_ms <= now) {
                edges.push(Edge::Timer);
            }
            if edges.is_empty() {
                continue;
            }
            let plane = registry.snapshot(&routing.borrow(), &shared.tables.read().unwrap());
            match program.react(Box::new(plane), edges, now) {
                Ok(commit) => {
                    tracing::debug!(
                        policy = program.name(),
                        schedule = commit.schedule.as_ref().map(|t| t.keys.len()),
                        floor = commit.schedule.as_ref().and_then(|t| t.floor),
                        placement = commit.placement.as_ref().map(|t| t.len()),
                        executors = commit.executors.as_ref().map(|t| t.len()),
                        published = commit.publish.len(),
                        "cluster policy call"
                    );
                    let author = program.name().to_string();
                    enact(&shared, &mut registry, commit, &author);
                }
                Err(Fault::Dead(error)) => {
                    tracing::error!(policy = program.name(), %error, "cluster policy is dead; its tables are withdrawn");
                    withdraw(&shared, &mut registry, program.name(), program.manifest());
                    withdrawn.push(at);
                }
                Err(fault) => {
                    tracing::warn!(policy = program.name(), %fault, "cluster policy call discarded");
                }
            }
        }
        for at in withdrawn.into_iter().rev() {
            programs.remove(at);
        }
        // With none left, launches go by the gateway's default until a
        // reload starts a set again.
        if programs.is_empty() {
            shared.dead.store(true, Ordering::Release);
            publish(&shared);
            continue;
        }
        // A replication or retention table stands until replaced: what it
        // asks for and is not yet so is asked of the gateway that carries
        // worker calls, the routing table's author.
        *shared.courier.lock().unwrap() = routing.borrow().author;
        if registry.ticked || registry.replaced {
            registry.ticked = false;
            registry.replaced = false;
            *shared.orders.lock().unwrap() = registry.orders();
        }
        // A launch's gate opens once every program has heard it and what
        // their calls about it changed has reached the others: a program
        // reading another's placement then reads the one made for it.
        for (gateway, seq) in heard {
            let at = owed.entry(gateway).or_default();
            *at = (*at).max(seq);
        }
        if registry.carried.is_empty() {
            let mut heard = shared.heard.lock().unwrap();
            for (gateway, seq) in owed.drain() {
                let at = heard.entry(gateway).or_default();
                *at = (*at).max(seq);
            }
        }
        publish(&shared);
    }
}

/// A dead program's tables revert to empty, and what it published is
/// withdrawn; the others' stand.
fn withdraw(
    shared: &Shared,
    registry: &mut Registry,
    author: &str,
    manifest: &super::manifest::Manifest,
) {
    {
        let mut tables = shared.tables.write().unwrap();
        if manifest.grants("schedule") {
            tables.schedule = None;
        }
        let (placement, executors) = (manifest.grants("placement"), manifest.grants("executors"));
        if placement || executors {
            registry.replace_placement(&mut tables, author, |tables| {
                if placement {
                    tables.placement.clear();
                }
                if executors {
                    tables.executors.clear();
                }
            });
        }
    }
    if manifest.grants("scale") {
        // No shape named: each keeps what it has.
        shared.scale.send_replace(Some(Vec::new()));
    }
    if manifest.grants("replication") {
        registry.replace_replication(author, Vec::new());
    }
    if manifest.grants("retention") {
        registry.replace_retention(author, Table::default());
    }
    let prefix = format!("{author}/");
    let gone: Vec<(Subject, String, Option<Value>)> = registry
        .said
        .values()
        .flat_map(|(subject, said)| {
            said.keys()
                .filter_map(|key| key.strip_prefix(&prefix))
                .map(|name| (subject.clone(), name.to_string(), None))
                .collect::<Vec<_>>()
        })
        .collect();
    registry.say(author, gone);
}

/// How long a launch waits for the slowest of `manifests` to hear it.
fn gate_of<'a>(manifests: impl Iterator<Item = &'a super::manifest::Manifest>) -> u64 {
    manifests.map(|m| m.gate_ms).max().unwrap_or(0)
}

fn program_name(spec: &str) -> String {
    std::path::Path::new(spec)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map_or_else(|| spec.to_string(), str::to_string)
}

fn enact(shared: &Shared, registry: &mut Registry, commit: Commit, author: &str) {
    if let Some(scale) = commit.scale {
        let entries: Vec<ScaleEntry> = scale
            .into_iter()
            .map(|s| ScaleEntry {
                model: s.model,
                engine: s.engine,
                workers: s.workers,
            })
            .collect();
        shared.scale.send_if_modified(|table| {
            let changed = table.as_ref() != Some(&entries);
            *table = Some(entries);
            changed
        });
    }
    if let Some(replication) = commit.state.replication {
        registry.replace_replication(author, replication);
    }
    if let Some(retention) = commit.state.retention {
        registry.replace_retention(author, retention);
    }
    registry.say(author, commit.publish);
    {
        let mut tables = shared.tables.write().unwrap();
        if let Some(schedule) = commit.schedule {
            tables.schedule = Some(schedule);
        }
        // A running process given another worker moves there: its gateway
        // sees the placement name another worker than the one it runs on.
        if commit.placement.is_some() || commit.executors.is_some() {
            let (placement, executors) = (commit.placement, commit.executors);
            registry.replace_placement(&mut tables, author, |tables| {
                if let Some(executors) = executors {
                    tables.executors = executors;
                }
                if let Some(placement) = placement {
                    tables.placement = placement;
                }
            });
        }
    }
    // Not yet admitted, it is refused; admitted, its gateway cancels it.
    for (id, reason) in commit.terminate {
        if !registry.procs.contains_key(&id) {
            continue;
        }
        registry.terminated.insert(id.clone());
        shared
            .terminated
            .lock()
            .unwrap()
            .entry(id)
            .or_insert(reason);
    }
}

/// A process's committed placement as another program reads it: its
/// worker, its executors and its props.
type Placed = (Option<String>, Vec<String>, Vec<(String, ValueKey)>);

/// A value as it compares: its kind and its payload, a float by its bits,
/// so a value is changed exactly when what a reader would read changed.
type ValueKey = (u8, String);

fn value_key(value: &Value) -> ValueKey {
    match value {
        Value::Boolean(b) => (0, b.to_string()),
        Value::Int(i) => (1, i.to_string()),
        Value::Float(f) => (2, f.to_bits().to_string()),
        Value::Text(text) => (3, text.clone()),
        Value::Json(text) => (4, text.clone()),
    }
}

fn placed_all(tables: &Tables) -> HashMap<String, Placed> {
    let ids: BTreeSet<&String> = tables
        .placement
        .keys()
        .chain(tables.executors.keys())
        .collect();
    ids.into_iter()
        .map(|id| {
            let placement = tables.placement.get(id);
            let placed = (
                placement.and_then(|p| p.worker.clone()),
                tables.executors.get(id).cloned().unwrap_or_default(),
                placement
                    .map(|p| {
                        p.props
                            .iter()
                            .map(|(n, v)| (n.clone(), value_key(v)))
                            .collect()
                    })
                    .unwrap_or_default(),
            );
            (id.clone(), placed)
        })
        .collect()
}

/// The workers each group's replication table asks for.
fn replicated(table: &[wit::Replicate]) -> BTreeMap<String, Vec<String>> {
    let mut sites: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in table {
        sites
            .entry(entry.group.clone())
            .or_default()
            .extend(entry.at.iter().map(|site| site.worker.clone()));
    }
    for workers in sites.values_mut() {
        workers.sort_unstable();
        workers.dedup();
    }
    sites
}

/// Where a subject's publications are kept.
fn subject_key(subject: &Subject) -> String {
    match subject {
        Subject::Session(id) => format!("session:{id}"),
        Subject::Process(id) => format!("process:{id}"),
        Subject::WorkingSet(id) => format!("working-set:{id}"),
        Subject::Worker(id) => format!("worker:{id}"),
        Subject::Group(id) => format!("group:{id}"),
        Subject::Link(id) => format!("link:{id}"),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[derive(Debug, Clone)]
struct Proc {
    arrival_ms: u64,
    tenant: String,
    /// Scoped by tenant.
    session: Option<String>,
    /// Shared with every call's view: a `prompt` can be long.
    props: Arc<HashMap<String, Value>>,
    prompt: Option<Arc<[u64]>>,
    worker: Option<String>,
    counted: Option<(u64, u64)>,
}

/// A run some worker keeps whole, and the workers (pools among them) that
/// hold every page of it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Group {
    keys: Arc<Vec<Key>>,
    sites: Vec<WorkerId>,
    /// The workers whose persistent disk tier keeps every page of it.
    disk_sites: Vec<WorkerId>,
    /// The most leading tokens of it some worker's owner asked kept on disk.
    persist_tokens: Option<u64>,
}

#[derive(Default)]
struct Registry {
    /// Ordered, so `candidates` is stable.
    procs: BTreeMap<String, Proc>,
    /// Exited in the events a call is about: it can still read them.
    exited: BTreeMap<String, Proc>,
    workers: BTreeSet<String>,
    sessions: HashMap<String, Props>,
    terminated: HashSet<String>,
    /// As far as the router can tell: no worker reports its evictions.
    held: HashMap<String, Arc<PromptIndex>>,
    /// The directory's groups, by their last key's hex, and its pools.
    groups: HashMap<String, Group>,
    /// The directory as last heard: where workers serve their pages, and
    /// what their disk tiers keep.
    directory: Arc<DirectoryView>,
    /// The workers linked to each gateway: `workers` is all of them.
    linked: HashMap<GatewayId, BTreeSet<String>>,
    pools: Vec<String>,
    /// The replication table the policy installed.
    replication: Vec<wit::Replicate>,
    /// The retention table the policy installed: a key per group.
    retention: Table,
    /// The directory was heard since `replicate` last ran.
    ticked: bool,
    /// A replication or retention table was replaced since `replicate`
    /// last ran.
    replaced: bool,
    /// What each worker's policy published about it, by name.
    published: HashMap<String, HashMap<String, serde_json::Value>>,
    /// Workers draining, as of the routing table last heard.
    draining: BTreeSet<String>,
    /// What cluster programs published, by subject: each value under
    /// `<program>/<name>`.
    said: HashMap<String, (Subject, BTreeMap<String, Value>)>,
    /// Edges a call caused, for every program but its author's next call.
    carried: Vec<(String, Edge)>,
}

/// The prompt blocks the router counts a worker as holding: about 4M tokens
/// of text.
const HELD_BLOCKS: usize = 1 << 18;

/// A gateway's event, as the registry takes it.
enum Local {
    Arrived {
        id: String,
        tenant: String,
        session: Option<String>,
        props: Props,
        prompt: Option<Arc<[u64]>>,
    },
    Dispatched {
        id: String,
        worker: String,
    },
    Counted {
        id: String,
        prompt: u64,
        generated: u64,
    },
    Exited {
        id: String,
        outcome: Outcome,
    },
    Moved {
        id: String,
        worker: Option<String>,
        why: String,
    },
    SessionProps {
        id: String,
        changes: Props,
    },
    SessionClosed {
        id: String,
    },
}

/// Props sent as JSON text, back as values; one that does not parse is
/// dropped.
fn props_of(json: Vec<(String, String)>) -> Props {
    json.into_iter()
        .filter_map(|(name, text)| Some((name, serde_json::from_str(&text).ok()?)))
        .collect()
}

impl Registry {
    fn apply(&mut self, event: Event, edges: &mut Vec<Edge>, shared: &Shared) {
        let local = match event {
            Event::Process(gateway, event) => match event {
                ProcessEvent::Arrived {
                    req,
                    tenant,
                    session,
                    props,
                    prompt,
                    ..
                } => Local::Arrived {
                    id: pid(gateway, req),
                    tenant,
                    session,
                    props: props_of(props),
                    prompt: prompt.map(Into::into),
                },
                ProcessEvent::Dispatched { req, worker } => Local::Dispatched {
                    id: pid(gateway, req),
                    worker: worker.0.to_string(),
                },
                ProcessEvent::Counted {
                    req,
                    prompt,
                    generated,
                } => Local::Counted {
                    id: pid(gateway, req),
                    prompt,
                    generated,
                },
                ProcessEvent::Moved { req, worker, why } => Local::Moved {
                    id: pid(gateway, req),
                    worker: worker.map(|w| w.0.to_string()),
                    why,
                },
                ProcessEvent::Exited { req, outcome } => {
                    let id = pid(gateway, req);
                    shared.terminated.lock().unwrap().remove(&id);
                    Local::Exited {
                        id,
                        outcome: match outcome {
                            controller_api::Outcome::Completed => Outcome::Completed,
                            controller_api::Outcome::Failed => Outcome::Failed,
                            controller_api::Outcome::Cancelled => Outcome::Cancelled,
                        },
                    }
                }
                ProcessEvent::SessionProps { id, changes } => Local::SessionProps {
                    id,
                    changes: props_of(changes),
                },
                ProcessEvent::SessionClosed { id } => Local::SessionClosed { id },
                ProcessEvent::Workers(now) => {
                    self.linked
                        .insert(gateway, now.iter().map(|w| w.0.to_string()).collect());
                    let now: BTreeSet<String> = self.linked.values().flatten().cloned().collect();
                    for removed in self.workers.difference(&now) {
                        edges.push(Edge::WorkerRemoved(removed.clone()));
                        self.held.remove(removed);
                    }
                    self.workers = now;
                    return;
                }
            },
            // The policy thread swaps programs before a round applies its
            // events; one here has no set to swap.
            Event::Reload { answer, .. } => {
                let _ = answer.send(Err(LoadError::Instantiate(
                    "a reload reached the registry, not the policy thread".into(),
                )));
                return;
            }
            Event::State(view) => {
                self.state(view, edges);
                return;
            }
        };
        self.apply_local(local, edges);
    }

    /// The directory as of now: its groups' copies, appeared and gone, as
    /// edges.
    fn state(&mut self, view: Arc<DirectoryView>, edges: &mut Vec<Edge>) {
        let groups: HashMap<String, Group> = view
            .groups
            .iter()
            .map(|(last, group)| {
                (
                    hex(last),
                    Group {
                        keys: Arc::new(group.keys.clone()),
                        sites: group.sites.clone(),
                        disk_sites: group.disk_sites.clone(),
                        persist_tokens: group.persist_tokens,
                    },
                )
            })
            .collect();
        let pools: Vec<String> = view.pools.iter().map(|w| w.0.to_string()).collect();
        let published: HashMap<String, HashMap<String, serde_json::Value>> = view
            .published
            .iter()
            .map(|(worker, own)| {
                let own = own
                    .iter()
                    .filter_map(|(name, json)| {
                        Some((name.clone(), serde_json::from_str(json).ok()?))
                    })
                    .collect();
                (worker.0.to_string(), own)
            })
            .collect();
        self.directory = view;
        self.apply_state(groups, pools, published, edges);
    }

    fn apply_local(&mut self, event: Local, edges: &mut Vec<Edge>) {
        match event {
            Local::Moved { id, worker, why } => {
                if let Some(proc) = self.procs.get_mut(&id) {
                    match worker {
                        Some(worker) => {
                            proc.worker = Some(worker.clone());
                            edges.push(Edge::Migrated(wit::Dispatch {
                                process: id,
                                worker,
                            }));
                        }
                        None => {
                            tracing::info!(process = %id, %why, "cluster policy: a move no worker took");
                            edges.push(Edge::MigrationFailed(id));
                        }
                    }
                }
            }
            Local::Arrived {
                id,
                tenant,
                session,
                props,
                prompt,
            } => {
                self.procs.insert(
                    id.clone(),
                    Proc {
                        arrival_ms: now_ms(),
                        session: session.map(|session| format!("{tenant}/{session}")),
                        tenant,
                        props: Arc::new(props.into_iter().filter_map(prop_value).collect()),
                        prompt,
                        worker: None,
                        counted: None,
                    },
                );
                edges.push(Edge::Arrived(id));
            }
            Local::Dispatched { id, worker } => {
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.worker = Some(worker.clone());
                    // What it prefills there, the worker holds after it:
                    // the router counts it so, as a cache-aware router does.
                    if let Some(prompt) = &proc.prompt {
                        let held = Arc::make_mut(self.held.entry(worker.clone()).or_default());
                        held.put(&id, prompt, TOKENS_PER_BYTE * PROMPT_BLOCK_BYTES as f64);
                        held.trim(HELD_BLOCKS);
                    }
                    edges.push(Edge::Dispatched(wit::Dispatch {
                        process: id,
                        worker,
                    }));
                }
            }
            Local::Counted {
                id,
                prompt,
                generated,
            } => {
                tracing::debug!(process = %id, prompt, generated, "cluster policy: a process counted its tokens");
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.counted = Some((prompt, generated));
                }
            }
            Local::Exited { id, outcome } => {
                self.terminated.remove(&id);
                if let Some(proc) = self.procs.remove(&id) {
                    self.exited.insert(id.clone(), proc);
                    edges.push(Edge::Exited(wit::Exit {
                        process: id,
                        outcome,
                    }));
                }
            }
            Local::SessionProps { id, changes } => {
                let current = self.sessions.remove(&id).unwrap_or_default();
                let (merged, names) = match client_api::props::merge(&current, changes) {
                    Ok(merged) => merged,
                    Err(why) => {
                        tracing::warn!(session = %id, %why, "cluster policy: session props refused");
                        (current, Vec::new())
                    }
                };
                if !merged.is_empty() {
                    self.sessions.insert(id.clone(), merged);
                }
                if !names.is_empty() {
                    edges.push(Edge::PropChanged(wit::Change {
                        subject: Subject::Session(id),
                        names,
                    }));
                }
            }
            Local::SessionClosed { id } => {
                self.sessions.remove(&id);
                self.said
                    .remove(&subject_key(&Subject::Session(id.clone())));
                edges.push(Edge::Closed(id));
            }
        }
    }

    fn apply_state(
        &mut self,
        groups: HashMap<String, Group>,
        pools: Vec<String>,
        published: HashMap<String, HashMap<String, serde_json::Value>>,
        edges: &mut Vec<Edge>,
    ) {
        self.published = published;
        let tier = |pools: &[String], worker: &str| {
            if pools.iter().any(|p| p == worker) {
                wit::Tier::Pool
            } else {
                wit::Tier::Host
            }
        };
        for (id, group) in &groups {
            let before = self
                .groups
                .get(id)
                .map(|g| g.sites.as_slice())
                .unwrap_or(&[]);
            for site in group.sites.iter().filter(|s| !before.contains(s)) {
                edges.push(Edge::Copied(wit::Residency {
                    group: id.clone(),
                    site: wit::Site {
                        worker: site.0.to_string(),
                        tier: tier(&pools, &site.0.to_string()),
                    },
                }));
            }
        }
        for (id, group) in &self.groups {
            let after = groups.get(id).map(|g| g.sites.as_slice()).unwrap_or(&[]);
            for site in group.sites.iter().filter(|s| !after.contains(s)) {
                edges.push(Edge::Dropped(wit::Residency {
                    group: id.clone(),
                    site: wit::Site {
                        worker: site.0.to_string(),
                        tier: tier(&self.pools, &site.0.to_string()),
                    },
                }));
            }
        }
        // A persistent disk tier's copies, as sites of tier `disk`.
        let disk = |groups: &HashMap<String, Group>, id: &str| {
            groups
                .get(id)
                .map(|g| g.disk_sites.clone())
                .unwrap_or_default()
        };
        for id in groups
            .keys()
            .chain(self.groups.keys())
            .collect::<BTreeSet<_>>()
        {
            let (before, after) = (disk(&self.groups, id), disk(&groups, id));
            let site = |worker: &WorkerId| wit::Residency {
                group: id.clone(),
                site: wit::Site {
                    worker: worker.0.to_string(),
                    tier: wit::Tier::Disk,
                },
            };
            edges.extend(
                after
                    .iter()
                    .filter(|w| !before.contains(w))
                    .map(|w| Edge::Copied(site(w))),
            );
            edges.extend(
                before
                    .iter()
                    .filter(|w| !after.contains(w))
                    .map(|w| Edge::Dropped(site(w))),
            );
        }
        // A process that asked a cached run kept on disk changes the
        // run's `persist-tokens` and nothing else of it.
        for (id, group) in &groups {
            let before = self.groups.get(id).and_then(|g| g.persist_tokens);
            if group.persist_tokens != before {
                edges.push(Edge::PropChanged(wit::Change {
                    subject: Subject::Group(id.clone()),
                    names: vec!["persist-tokens".into()],
                }));
            }
        }
        self.groups = groups;
        self.pools = pools;
        self.ticked = true;
    }

    /// The processes the last round's calls read as exited are gone, and
    /// what programs published about them with them, without an edge.
    fn forget_exited(&mut self) {
        for id in std::mem::take(&mut self.exited).into_keys() {
            self.said.remove(&subject_key(&Subject::Process(id)));
        }
    }

    /// Changes the committed placement by `f`, and carries
    /// `placement-changed`, caused by `author`, for each process whose
    /// placement differs.
    fn replace_placement(
        &mut self,
        tables: &mut Tables,
        author: &str,
        f: impl FnOnce(&mut Tables),
    ) {
        let before = placed_all(tables);
        f(tables);
        let after = placed_all(tables);
        self.carry_placements(author, &before, &after);
    }

    /// Replaces the replication table, and carries `replication-changed`,
    /// caused by `author`, for each group whose sites differ.
    fn replace_replication(&mut self, author: &str, table: Vec<wit::Replicate>) {
        let before = replicated(&self.replication);
        self.replication = table;
        self.replaced = true;
        self.carry_replications(author, &before);
    }

    /// Replaces the retention table, and carries `retention-changed`,
    /// caused by `author`, for each group whose key differs (one no entry
    /// names is at 0).
    fn replace_retention(&mut self, author: &str, table: Table) {
        let before = std::mem::replace(&mut self.retention, table);
        self.replaced = true;
        let groups: BTreeSet<&String> = before
            .keys
            .keys()
            .chain(self.retention.keys.keys())
            .collect();
        let mut changed = false;
        for group in groups {
            if before.key(group) != self.retention.key(group) {
                changed = true;
                self.carried
                    .push((author.to_string(), Edge::RetentionChanged(group.clone())));
            }
        }
        if changed && self.pools.is_empty() && self.retention.keys.values().any(|&key| key > 0) {
            tracing::info!(
                policy = author,
                "retention table taken; no pool is known to keep a copy in, so none is kept"
            );
        }
    }

    /// The copies to ask for: (group, worker, the table that wants it).
    fn wanted(&self) -> Vec<(String, WorkerId, &'static str)> {
        let mut wanted = Vec::new();
        for entry in &self.replication {
            let Some(group) = self.groups.get(&entry.group) else {
                continue;
            };
            for site in entry.at.iter().filter(|site| site.tier != wit::Tier::Disk) {
                let Ok(target) = site.worker.parse::<u64>().map(WorkerId) else {
                    continue;
                };
                if !group.sites.contains(&target) {
                    wanted.push((entry.group.clone(), target, "replication"));
                }
            }
        }
        wanted.extend(
            self.retained()
                .into_iter()
                .map(|(group, pool)| (group, pool, "retention")),
        );
        wanted
    }

    /// The copies the replication table asks at tier `disk`: (group,
    /// worker, the leading tokens wanted). Whether one is there already is
    /// the directory's to say.
    fn wanted_on_disk(&self) -> Vec<(String, WorkerId, Option<u32>)> {
        self.replication
            .iter()
            .flat_map(|entry| {
                entry
                    .at
                    .iter()
                    .filter(|site| site.tier == wit::Tier::Disk)
                    .filter_map(|site| {
                        let target = site.worker.parse::<u64>().map(WorkerId).ok()?;
                        Some((entry.group.clone(), target, entry.upto))
                    })
            })
            .collect()
    }

    /// The copies the state tables want and the directory does not show,
    /// each to be asked of its worker: for each site the replication table
    /// names that does not hold its group, and for each group retained above
    /// key 0 that no pool holds, a pool (the one a hash of the group picks),
    /// highest key first, each from a site that holds it; and for each site
    /// at tier `disk` that holds its group, the first `upto` tokens of it,
    /// cut at that worker's page, unless its disk tier keeps them already.
    /// That a pool holds a group is what the pool reports, as a worker does:
    /// the group stays in the directory, with the pool its site, after every
    /// worker lets it go, and comes back when the pool restarts.
    fn orders(&self) -> Vec<ReplicateOrder> {
        let directory = &self.directory;
        let peer = |worker: WorkerId| {
            directory
                .peers
                .iter()
                .find(|(w, _)| *w == worker)
                .map(|(_, peer)| peer.clone())
        };
        let mut orders = Vec::new();
        for (group_id, target, table) in self.wanted() {
            let Some(group) = self.groups.get(&group_id) else {
                continue;
            };
            let Some(from) = group.sites.iter().find_map(|holder| peer(*holder)) else {
                continue;
            };
            orders.push(ReplicateOrder {
                retention: self.retention.key(&group_id),
                keys: group.keys.as_ref().clone(),
                group: group_id,
                worker: target,
                from,
                disk: false,
                table: table.to_string(),
            });
        }
        for (group_id, target, upto) in self.wanted_on_disk() {
            let Some(group) = self.groups.get(&group_id) else {
                continue;
            };
            if !group.sites.contains(&target) {
                continue;
            }
            let Some((_, (page_size, true))) =
                directory.disk_tiers.iter().find(|(w, _)| *w == target)
            else {
                continue;
            };
            let pages = upto.map_or(group.keys.len(), |tokens| {
                (tokens as usize / (*page_size).max(1) as usize).min(group.keys.len())
            });
            let keys: Vec<Key> = group.keys[..pages].to_vec();
            let on_disk = directory
                .disk
                .iter()
                .find(|(w, _)| *w == target)
                .is_some_and(|(_, disk)| {
                    let disk: HashSet<&Key> = disk.iter().collect();
                    keys.iter().all(|key| disk.contains(key))
                });
            // Groups whose prefixes cut to the same pages are one copy.
            let asked = orders.iter().any(|order: &ReplicateOrder| {
                order.disk && order.worker == target && order.keys == keys
            });
            if keys.is_empty() || on_disk || asked {
                continue;
            }
            orders.push(ReplicateOrder {
                retention: self.retention.key(&group_id),
                group: group_id,
                worker: target,
                keys,
                from: String::new(),
                disk: true,
                table: "replication".into(),
            });
        }
        orders
    }

    /// For each group the directory shows, retained above key 0 and held by
    /// no pool, the pool to take a copy: the one a hash of the group picks.
    /// Highest key first.
    fn retained(&self) -> Vec<(String, WorkerId)> {
        let pools: Vec<WorkerId> = self
            .pools
            .iter()
            .filter_map(|pool| pool.parse().ok().map(WorkerId))
            .collect();
        if pools.is_empty() {
            return Vec::new();
        }
        let mut retained: Vec<(&String, i64)> = self
            .retention
            .keys
            .iter()
            .filter(|(_, key)| **key > 0)
            .map(|(group, key)| (group, *key))
            .collect();
        retained.sort_by_key(|(group, key)| (Reverse(*key), *group));
        retained
            .into_iter()
            .filter_map(|(id, _)| {
                let group = self.groups.get(id)?;
                if group.sites.iter().any(|site| pools.contains(site)) {
                    return None;
                }
                let mut hasher = std::hash::DefaultHasher::new();
                std::hash::Hash::hash(id, &mut hasher);
                let at = std::hash::Hasher::finish(&hasher) % pools.len() as u64;
                Some((id.clone(), pools[at as usize]))
            })
            .collect()
    }

    /// `placement-changed` for each process whose committed placement
    /// differs, caused by `author`.
    fn carry_placements(
        &mut self,
        author: &str,
        before: &HashMap<String, Placed>,
        after: &HashMap<String, Placed>,
    ) {
        let ids: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
        for id in ids {
            if before.get(id) != after.get(id) {
                self.carried
                    .push((author.to_string(), Edge::PlacementChanged(id.clone())));
            }
        }
    }

    /// `replication-changed` for each group whose committed sites differ
    /// from `before`, caused by `author`.
    fn carry_replications(&mut self, author: &str, before: &BTreeMap<String, Vec<String>>) {
        let after = replicated(&self.replication);
        let groups: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
        for group in groups {
            if before.get(group) != after.get(group) {
                self.carried
                    .push((author.to_string(), Edge::ReplicationChanged(group.clone())));
            }
        }
    }

    /// `author`'s publications, in order; `published` for each subject one
    /// of them changed.
    fn say(&mut self, author: &str, said: Vec<(Subject, String, Option<Value>)>) {
        let mut changed: Vec<(String, Subject)> = Vec::new();
        for (subject, name, value) in said {
            let key = subject_key(&subject);
            let full = format!("{author}/{name}");
            let entry = self
                .said
                .entry(key.clone())
                .or_insert_with(|| (subject.clone(), BTreeMap::new()));
            if entry.1.get(&full).map(value_key) == value.as_ref().map(value_key) {
                continue;
            }
            match value {
                Some(value) => entry.1.insert(full, value),
                None => entry.1.remove(&full),
            };
            if entry.1.is_empty() {
                self.said.remove(&key);
            }
            if !changed.iter().any(|(k, _)| *k == key) {
                changed.push((key, subject));
            }
        }
        for (_, subject) in changed {
            self.carried
                .push((author.to_string(), Edge::Published(subject)));
        }
    }

    fn snapshot(&self, routing: &RoutingTable, tables: &Tables) -> Snapshot {
        Snapshot {
            placement: tables.placement.clone(),
            executors: tables.executors.clone(),
            replication: replicated(&self.replication),
            retention: self.retention.keys.clone(),
            said: self
                .said
                .iter()
                .map(|(key, (_, said))| (key.clone(), said.clone()))
                .collect(),
            groups: self.groups.clone(),
            pools: self.pools.clone(),
            published: self.published.clone(),
            procs: self.procs.clone(),
            exited: self.exited.clone(),
            sessions: self.sessions.clone(),
            held: self.held.clone(),
            workers: routing
                .workers
                .iter()
                .filter(|w| self.workers.contains(&w.id.0.to_string()))
                .map(|w| (w.id.0.to_string(), w.clone()))
                .collect(),
            scales: routing.scales,
            disk_tiers: self
                .directory
                .disk_tiers
                .iter()
                .map(|(worker, (page, persistent))| {
                    let pages = self
                        .directory
                        .disk
                        .iter()
                        .find(|(w, _)| w == worker)
                        .map_or(0, |(_, keys)| keys.len());
                    (worker.0.to_string(), (*page, *persistent, pages))
                })
                .collect(),
        }
    }
}

fn prop_value((name, value): (String, serde_json::Value)) -> Option<(String, Value)> {
    use serde_json::Value as Json;
    let value = match value {
        Json::Null => return None,
        Json::Bool(b) => Value::Boolean(b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64()?),
        },
        Json::String(s) => Value::Text(s),
        json @ (Json::Array(_) | Json::Object(_)) => Value::Json(json.to_string()),
    };
    Some((name, value))
}

fn json(value: &Value) -> Option<serde_json::Value> {
    use serde_json::Value as Json;
    Some(match value {
        Value::Boolean(b) => Json::Bool(*b),
        Value::Int(i) => Json::from(*i),
        Value::Float(f) => serde_json::Number::from_f64(*f).map(Json::Number)?,
        Value::Text(s) => Json::String(s.clone()),
        Value::Json(text) => serde_json::from_str(text).ok()?,
    })
}

struct Snapshot {
    /// What the programs' calls have committed, read-only.
    placement: HashMap<String, wit::Placement>,
    executors: HashMap<String, Vec<String>>,
    replication: BTreeMap<String, Vec<String>>,
    retention: HashMap<String, i64>,
    said: HashMap<String, BTreeMap<String, Value>>,
    groups: HashMap<String, Group>,
    pools: Vec<String>,
    published: HashMap<String, HashMap<String, serde_json::Value>>,
    procs: BTreeMap<String, Proc>,
    exited: BTreeMap<String, Proc>,
    sessions: HashMap<String, Props>,
    held: HashMap<String, Arc<PromptIndex>>,
    workers: BTreeMap<String, RoutableWorker>,
    /// The controller has an actuator to meet a scale table.
    scales: bool,
    /// Each worker's disk tier: its page in tokens, whether it is kept
    /// across runs, and the pages it keeps a copy of.
    disk_tiers: HashMap<String, (u32, bool, usize)>,
}

fn text(value: impl Into<String>) -> Option<Value> {
    Some(Value::Text(value.into()))
}

impl Plane for Snapshot {
    fn candidates(&self, kind: SubjectKind) -> Vec<String> {
        match kind {
            SubjectKind::Process => self.procs.keys().cloned().collect(),
            SubjectKind::Session => {
                let sessions: BTreeSet<&String> = self
                    .procs
                    .values()
                    .filter_map(|p| p.session.as_ref())
                    .chain(self.sessions.keys())
                    .collect();
                sessions.into_iter().cloned().collect()
            }
            SubjectKind::Worker => self
                .workers
                .keys()
                .cloned()
                .chain(self.pools.iter().cloned())
                .collect(),
            SubjectKind::Group => {
                let mut ids: Vec<String> = self.groups.keys().cloned().collect();
                ids.sort_unstable();
                ids
            }
            SubjectKind::WorkingSet | SubjectKind::Link => Vec::new(),
        }
    }

    fn attr(&self, of: &Subject, name: &str) -> Option<Value> {
        // What a cluster program published, under its name.
        if let Some(said) = name
            .strip_prefix("published/")
            .and_then(|key| self.said.get(&subject_key(of))?.get(key))
        {
            return Some(said.clone());
        }
        match of {
            Subject::Process(id) if name.starts_with("placement/") => match name {
                "placement/worker" => self.placement.get(id)?.worker.clone().and_then(text),
                "placement/executors" => Some(Value::Json(
                    serde_json::to_string(self.executors.get(id)?).unwrap_or_default(),
                )),
                _ => {
                    let prop = name.strip_prefix("placement/props/")?;
                    let placement = self.placement.get(id)?;
                    placement
                        .props
                        .iter()
                        .find(|(n, _)| n == prop)
                        .map(|(_, value)| value.clone())
                }
            },
            Subject::Group(id) if name == "replication/sites" => Some(Value::Json(
                serde_json::to_string(self.replication.get(id)?).unwrap_or_default(),
            )),
            Subject::Group(id) if name == "retention/key" => {
                self.retention.get(id).map(|key| Value::Int(*key))
            }
            Subject::Process(id) => {
                let (proc, live) = match (self.procs.get(id), self.exited.get(id)) {
                    (Some(proc), _) => (proc, true),
                    (None, Some(proc)) => (proc, false),
                    (None, None) => return None,
                };
                let int = |n: Option<u64>| n.map(|n| Value::Int(n as i64));
                match name {
                    "state" if !live => None,
                    "state" if proc.worker.is_some() => text("running"),
                    "state" => text("pending"),
                    // An exited process has told the router its counts; before
                    // that its prompt is estimated from its text.
                    "prompt-tokens" => int(proc.counted.map(|(prompt, _)| prompt).or_else(|| {
                        proc.prompt.as_ref().map(|blocks| {
                            (blocks.len() as f64 * PROMPT_BLOCK_BYTES as f64 * TOKENS_PER_BYTE)
                                as u64
                        })
                    })),
                    "max-tokens" => match proc.props.get(client_api::props::MAX_TOKENS) {
                        Some(Value::Int(n)) if *n >= 0 => Some(Value::Int(*n)),
                        _ => None,
                    },
                    "generated-tokens" => int(proc.counted.map(|(_, generated)| generated)),
                    "arrival-ms" => Some(Value::Int(proc.arrival_ms as i64)),
                    "tenant" => text(proc.tenant.clone()),
                    "session" => proc.session.clone().and_then(text),
                    "worker" => proc.worker.clone().and_then(text),
                    _ => {
                        // What the worker holds of its prompt, estimated
                        // from its text: there are no tokens here.
                        let worker = name.strip_prefix("cached-tokens/")?;
                        let prompt = proc.prompt.as_ref()?;
                        self.workers.get(worker)?;
                        let held = self
                            .held
                            .get(worker)
                            .map_or(0, |held| held.held_tokens(prompt));
                        Some(Value::Int(held as i64))
                    }
                }
            }
            Subject::Session(id) => {
                let mut procs = self
                    .procs
                    .iter()
                    .filter(|(_, p)| p.session.as_deref() == Some(id.as_str()))
                    .peekable();
                let tenant = match procs.peek() {
                    Some((_, first)) => first.tenant.clone(),
                    None if self.sessions.contains_key(id) => id
                        .split_once('/')
                        .map_or(id.as_str(), |(t, _)| t)
                        .to_string(),
                    None => return None,
                };
                match name {
                    "tenant" => text(tenant),
                    _ => None,
                }
            }
            Subject::Worker(id) if name.starts_with("published/") => self
                .published
                .get(id)
                .and_then(|map| map.get(&name["published/".len()..]))
                .and_then(|value| prop_value((String::new(), value.clone())))
                .map(|(_, value)| value),
            Subject::Worker(id) if self.pools.contains(id) => match name {
                "pool" => Some(Value::Boolean(true)),
                _ => None,
            },
            Subject::Group(id) => {
                let group = self.groups.get(id)?;
                match name {
                    "pages" => Some(Value::Int(group.keys.len() as i64)),
                    // The workers holding a copy, as a JSON array of ids.
                    "sites" => Some(Value::Json(
                        serde_json::to_string(
                            &group
                                .sites
                                .iter()
                                .map(|w| w.0.to_string())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap_or_default(),
                    )),
                    // The workers whose persistent disk tier keeps all of
                    // it, as a JSON array of ids.
                    "disk-sites" => Some(Value::Json(
                        serde_json::to_string(
                            &group
                                .disk_sites
                                .iter()
                                .map(|w| w.0.to_string())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap_or_default(),
                    )),
                    // The leading tokens of it a process asked kept on
                    // disk (`persist-prefix-tokens`, `persist-system`).
                    "persist-tokens" => group
                        .persist_tokens
                        .map(|tokens| Value::Int(i64::try_from(tokens).unwrap_or(i64::MAX))),
                    _ => None,
                }
            }
            Subject::Worker(id) => {
                let worker = self.workers.get(id)?;
                match name {
                    "role" => text(match worker.role {
                        Role::Prefill => "prefill",
                        Role::Decode => "decode",
                        Role::Encode => "encode",
                    }),
                    "model" => text(worker.model.clone()),
                    // The machine it serves from: workers sharing one share
                    // a node's links.
                    "host" => worker
                        .addr
                        .rsplit_once(':')
                        .map(|(host, _)| host.trim_start_matches("tcp://"))
                        .and_then(text),
                    "engine" if !worker.engine.is_empty() => text(worker.engine.clone()),
                    "draining" => Some(Value::Boolean(worker.health != Health::Healthy)),
                    "pie/kv-pressure" => {
                        Some(Value::Int(i64::from(worker.coarse_load.kv_pressure_bucket)))
                    }
                    "pie/inflight" => Some(Value::Int(i64::from(worker.coarse_load.inflight))),
                    // As of the worker's last report to the controller.
                    "running" => Some(Value::Int(i64::from(worker.coarse_load.running))),
                    "waiting" => Some(Value::Int(i64::from(worker.coarse_load.waiting))),
                    "capacity-tokens" | "used-tokens"
                        if worker.coarse_load.capacity_tokens == 0 =>
                    {
                        None
                    }
                    "capacity-tokens" => {
                        Some(Value::Int(worker.coarse_load.capacity_tokens as i64))
                    }
                    "used-tokens" => Some(Value::Int(worker.coarse_load.used_tokens as i64)),
                    // Its persistent disk tier (`disk_kv_persistent`): kept
                    // across runs, and the tokens it keeps a copy of.
                    "tier/disk/persistent" => self
                        .disk_tiers
                        .get(id)
                        .map(|(_, persistent, _)| Value::Boolean(*persistent)),
                    "tier/disk/used" => self
                        .disk_tiers
                        .get(id)
                        .filter(|(_, persistent, _)| *persistent)
                        .map(|(page, _, pages)| Value::Int(i64::from(*page) * *pages as i64)),
                    _ => None,
                }
            }
            Subject::WorkingSet(_) | Subject::Link(_) => None,
        }
    }

    fn prop(&self, of: &Subject, name: &str) -> Option<Value> {
        match of {
            Subject::Process(id) => {
                let proc = self.procs.get(id).or_else(|| self.exited.get(id))?;
                proc.props.get(name).cloned()
            }
            Subject::Session(id) => {
                let value = self.sessions.get(id)?.get(name)?;
                prop_value((name.to_string(), value.clone())).map(|(_, value)| value)
            }
            _ => None,
        }
    }

    fn capability(&self, name: &str) -> bool {
        match name {
            "cluster.scale" => self.scales,
            // Some worker keeps a disk tier across runs.
            "kv.persist" => self
                .disk_tiers
                .values()
                .any(|(_, persistent, _)| *persistent),
            name => capability(name),
        }
    }

    fn terminable(&self, process: &str) -> bool {
        self.procs.contains_key(process)
    }
}

#[cfg(test)]
mod tests {
    use controller_api::WorkerStatus;

    use super::*;

    fn table(kv: u8) -> RoutingTable {
        RoutingTable {
            epoch: 1,
            scales: false,
            author: None,
            workers: vec![RoutableWorker {
                id: WorkerId(1),
                addr: "w1".into(),
                role: Role::Decode,
                model: "m".into(),
                engine: String::new(),
                health: Health::Healthy,
                coarse_load: WorkerStatus {
                    kv_pressure_bucket: kv,
                    inflight: 0,
                    running: 2,
                    waiting: 5,
                    capacity_tokens: 1000,
                    used_tokens: 250,
                    draining: false,
                },
            }],
        }
    }

    fn placed(worker: &str, props: &[(&str, &str)]) -> wit::Placement {
        wit::Placement {
            process: String::new(),
            worker: Some(worker.into()),
            executors: Vec::new(),
            props: props
                .iter()
                .map(|(n, v)| (n.to_string(), Value::Text(v.to_string())))
                .collect(),
        }
    }

    fn carried_kinds(registry: &mut Registry) -> Vec<(String, String)> {
        std::mem::take(&mut registry.carried)
            .into_iter()
            .map(|(author, edge)| {
                let what = match edge {
                    Edge::PlacementChanged(id) => format!("placement {id}"),
                    Edge::ReplicationChanged(id) => format!("replication {id}"),
                    Edge::RetentionChanged(id) => format!("retention {id}"),
                    Edge::Published(subject) => format!("published {}", subject_key(&subject)),
                    other => format!("{other:?}"),
                };
                (author, what)
            })
            .collect()
    }

    fn group(sites: &[u64]) -> Group {
        Group {
            keys: Arc::new(vec![[7; 32]]),
            sites: sites.iter().copied().map(WorkerId).collect(),
            ..Default::default()
        }
    }

    fn retention(entries: &[(&str, i64)]) -> Table {
        Table {
            keys: entries.iter().map(|(g, k)| (g.to_string(), *k)).collect(),
            floor: None,
        }
    }

    #[test]
    fn a_pending_process_is_sized_from_its_prompt_text_and_its_limit() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let props = serde_json::from_value(serde_json::json!({
            "prompt": "x".repeat(640),
            "max-tokens": 64,
        }))
        .unwrap();
        registry.apply_local(
            Local::Arrived {
                id: "1".into(),
                tenant: "t".into(),
                session: None,
                props,
                prompt: Some(client_api::props::prompt_blocks(&"x".repeat(640)).into()),
            },
            &mut edges,
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        let of = |name| format!("{:?}", plane.attr(&Subject::Process("1".into()), name));
        assert_eq!(of("prompt-tokens"), format!("{:?}", Some(Value::Int(160))));
        assert_eq!(of("max-tokens"), format!("{:?}", Some(Value::Int(64))));
    }

    #[test]
    fn a_worker_holds_the_prompts_dispatched_to_it() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let text = "word ".repeat(200);
        let arrived = |id: &str, prompt: Option<String>| Local::Arrived {
            id: id.into(),
            tenant: "t".into(),
            session: None,
            props: Props::new(),
            prompt: prompt.map(|prompt| client_api::props::prompt_blocks(&prompt).into()),
        };
        for event in [
            arrived("1", Some(text.clone())),
            Local::Dispatched {
                id: "1".into(),
                worker: "1".into(),
            },
            arrived("2", Some(format!("{text}and more"))),
            arrived("3", None),
        ] {
            registry.apply_local(event, &mut edges);
        }
        registry.workers = BTreeSet::from(["1".to_string()]);
        let plane = registry.snapshot(&table(0), &Tables::default());
        let of = |id: &str, name| format!("{:?}", plane.attr(&Subject::Process(id.into()), name));
        // 1000 bytes: fifteen whole 64-byte blocks, at a guessed 16 tokens each.
        assert_eq!(
            of("2", "cached-tokens/1"),
            format!("{:?}", Some(Value::Int(240)))
        );
        assert_eq!(of("2", "cached-tokens/9"), "None", "no such worker");
        assert_eq!(of("3", "cached-tokens/1"), "None", "no prompt");
        assert_eq!(of("2", "cached-tokens"), "None");
    }

    #[test]
    fn a_session_s_props_reach_the_cluster_program_between_launches() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let changes: Props =
            serde_json::from_value(serde_json::json!({"tool-eta-ms": 900})).unwrap();
        registry.apply_local(
            Local::SessionProps {
                id: "default/s".into(),
                changes,
            },
            &mut edges,
        );
        assert!(matches!(&edges[..], [Edge::PropChanged(c)] if c.names == ["tool-eta-ms"]));
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert_eq!(plane.candidates(SubjectKind::Session), ["default/s"]);
        assert!(matches!(
            plane.prop(&Subject::Session("default/s".into()), "tool-eta-ms"),
            Some(Value::Int(900))
        ));
        edges.clear();
        registry.apply_local(
            Local::SessionClosed {
                id: "default/s".into(),
            },
            &mut edges,
        );
        assert!(matches!(&edges[..], [Edge::Closed(s)] if s == "default/s"));
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(
            plane
                .prop(&Subject::Session("default/s".into()), "tool-eta-ms")
                .is_none()
        );
    }

    #[test]
    fn cluster_scale_is_there_while_the_controller_can_act() {
        let registry = Registry::default();
        let tables = Tables::default();
        let mut routing = table(0);
        assert!(
            !registry
                .snapshot(&routing, &tables)
                .capability("cluster.scale")
        );
        routing.scales = true;
        assert!(
            registry
                .snapshot(&routing, &tables)
                .capability("cluster.scale")
        );
        assert!(
            registry
                .snapshot(&routing, &tables)
                .capability("placement.migrate"),
            "the rest as before"
        );
    }

    #[test]
    fn a_committed_placement_reads_as_attrs_and_its_change_is_carried_once() {
        let mut registry = Registry::default();
        let mut tables = Tables::default();
        let before = placed_all(&tables);
        tables
            .placement
            .insert("7".into(), placed("3", &[("tier", "gold")]));
        tables
            .executors
            .insert("7".into(), vec!["5".into(), "4".into()]);
        registry.carry_placements("a", &before, &placed_all(&tables));
        assert_eq!(
            carried_kinds(&mut registry),
            [("a".to_string(), "placement 7".to_string())],
            "one edge for the one process that changed, carried by its author"
        );

        let plane = registry.snapshot(&table(0), &tables);
        let process = Subject::Process("7".into());
        assert!(
            matches!(plane.attr(&process, "placement/worker"), Some(Value::Text(w)) if w == "3")
        );
        assert!(
            matches!(plane.attr(&process, "placement/executors"), Some(Value::Json(j)) if j == r#"["5","4"]"#)
        );
        assert!(
            matches!(plane.attr(&process, "placement/props/tier"), Some(Value::Text(v)) if v == "gold")
        );
        assert!(plane.attr(&process, "placement/props/none").is_none());

        // The same table again is no change.
        let before = placed_all(&tables);
        registry.carry_placements("b", &before, &placed_all(&tables));
        assert!(carried_kinds(&mut registry).is_empty());
    }

    #[test]
    fn a_published_value_reads_under_its_author_s_name_and_is_carried_only_when_it_changes() {
        let mut registry = Registry::default();
        let worker = Subject::Worker("1".into());
        registry.published.insert(
            "1".into(),
            HashMap::from([("mark".to_string(), serde_json::json!("own"))]),
        );
        registry.say(
            "a",
            vec![(worker.clone(), "mark".into(), Some(Value::Int(3)))],
        );
        registry.say(
            "b",
            vec![(worker.clone(), "mark".into(), Some(Value::Int(4)))],
        );
        assert_eq!(
            carried_kinds(&mut registry),
            [
                ("a".to_string(), "published worker:1".to_string()),
                ("b".to_string(), "published worker:1".to_string()),
            ]
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(matches!(
            plane.attr(&worker, "published/a/mark"),
            Some(Value::Int(3))
        ));
        assert!(matches!(
            plane.attr(&worker, "published/b/mark"),
            Some(Value::Int(4))
        ));
        assert!(
            matches!(plane.attr(&worker, "published/mark"), Some(Value::Text(t)) if t == "own"),
            "the worker's own policy's value stands beside the programs'"
        );

        registry.say(
            "a",
            vec![(worker.clone(), "mark".into(), Some(Value::Int(3)))],
        );
        assert!(
            carried_kinds(&mut registry).is_empty(),
            "the same value again is no change"
        );

        registry.say("a", vec![(worker.clone(), "mark".into(), None)]);
        assert_eq!(
            carried_kinds(&mut registry),
            [("a".to_string(), "published worker:1".to_string())]
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(
            plane.attr(&worker, "published/a/mark").is_none(),
            "withdrawn"
        );
        assert!(matches!(
            plane.attr(&worker, "published/b/mark"),
            Some(Value::Int(4))
        ));
    }

    #[test]
    fn a_group_s_committed_replication_reads_as_its_sites() {
        let mut registry = Registry::default();
        let before = replicated(&registry.replication);
        registry.replication = vec![wit::Replicate {
            group: "g".into(),
            upto: None,
            at: vec![
                wit::Site {
                    worker: "2".into(),
                    tier: wit::Tier::Host,
                },
                wit::Site {
                    worker: "1".into(),
                    tier: wit::Tier::Device,
                },
            ],
        }];
        registry.carry_replications("r", &before);
        assert_eq!(
            carried_kinds(&mut registry),
            [("r".to_string(), "replication g".to_string())]
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(
            matches!(plane.attr(&Subject::Group("g".into()), "replication/sites"), Some(Value::Json(j)) if j == r#"["1","2"]"#)
        );
    }

    #[test]
    fn a_group_s_committed_retention_reads_as_its_key_and_only_a_changed_key_is_carried() {
        let mut registry = Registry::default();
        registry.replace_retention("r", retention(&[("g", 2), ("h", 0)]));
        assert_eq!(
            carried_kinds(&mut registry),
            [("r".to_string(), "retention g".to_string())],
            "h at 0 is where a group no entry names already is"
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        let key = |id: &str| plane.attr(&Subject::Group(id.into()), "retention/key");
        assert!(matches!(key("g"), Some(Value::Int(2))));
        assert!(matches!(key("h"), Some(Value::Int(0))));
        assert!(key("other").is_none());

        registry.replace_retention("r", retention(&[("g", 2)]));
        assert!(carried_kinds(&mut registry).is_empty(), "no key changed");
        registry.replace_retention("r", retention(&[("g", 3)]));
        assert_eq!(
            carried_kinds(&mut registry),
            [("r".to_string(), "retention g".to_string())]
        );
    }

    #[test]
    fn a_retained_group_no_pool_holds_is_asked_of_a_pool_highest_key_first() {
        let mut registry = Registry {
            groups: HashMap::from([
                ("a".to_string(), group(&[1])),
                ("b".to_string(), group(&[1])),
                ("held".to_string(), group(&[1, 9])),
                ("zero".to_string(), group(&[1])),
            ]),
            retention: retention(&[("a", 1), ("b", 5), ("held", 3), ("zero", 0), ("gone", 9)]),
            ..Default::default()
        };
        assert!(
            registry.retained().is_empty(),
            "with no pool, the table is taken and nothing is asked"
        );
        registry.pools = vec!["9".into()];
        assert_eq!(
            registry.retained(),
            [
                ("b".to_string(), WorkerId(9)),
                ("a".to_string(), WorkerId(9))
            ],
            "a pool already holds `held`, `zero` is at 0, `gone` is in no directory"
        );
    }

    #[test]
    fn a_cached_run_asked_kept_on_disk_later_tells_the_programs() {
        let mut registry = Registry::default();
        let state = |registry: &mut Registry, tokens| {
            let groups = HashMap::from([(
                "g".to_string(),
                Group {
                    persist_tokens: tokens,
                    ..group(&[1])
                },
            )]);
            let mut edges = Vec::new();
            registry.apply_state(groups, Vec::new(), HashMap::new(), &mut edges);
            edges
                .into_iter()
                .filter(|edge| matches!(edge, Edge::PropChanged(_)))
                .count()
        };
        assert_eq!(state(&mut registry, None), 0, "cached, nothing asked");
        assert_eq!(
            state(&mut registry, Some(40)),
            1,
            "a later process asks it kept"
        );
        assert_eq!(state(&mut registry, Some(40)), 0, "said once");
    }

    #[test]
    fn a_copy_at_tier_disk_is_asked_apart_with_its_tokens_and_reads_as_a_disk_site() {
        let shared = host_shared();
        let mut registry = Registry {
            groups: HashMap::from([(
                "g".to_string(),
                Group {
                    disk_sites: vec![WorkerId(1)],
                    persist_tokens: Some(40),
                    ..group(&[1])
                },
            )]),
            ..Default::default()
        };
        let site = |worker: &str, tier| wit::Site {
            worker: worker.into(),
            tier,
        };
        let commit = Commit {
            state: super::super::StateCommit {
                replication: Some(vec![wit::Replicate {
                    group: "g".into(),
                    upto: Some(40),
                    at: vec![site("1", wit::Tier::Disk), site("2", wit::Tier::Host)],
                }]),
                ..Default::default()
            },
            ..Default::default()
        };
        enact(&shared, &mut registry, commit, "persist");
        assert_eq!(
            registry.wanted(),
            [("g".to_string(), WorkerId(2), "replication")],
            "a disk copy is no host copy"
        );
        assert_eq!(
            registry.wanted_on_disk(),
            [("g".to_string(), WorkerId(1), Some(40))]
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        let attr = |name| plane.attr(&Subject::Group("g".into()), name);
        assert!(matches!(attr("persist-tokens"), Some(Value::Int(40))));
        assert!(matches!(attr("disk-sites"), Some(Value::Json(json)) if json == "[\"1\"]"));
    }

    #[test]
    fn a_retention_table_is_committed_with_its_call_and_withdrawn_with_its_program() {
        let shared = host_shared();
        let mut registry = Registry {
            groups: HashMap::from([("g".to_string(), group(&[1]))]),
            pools: vec!["9".into()],
            ..Default::default()
        };
        let commit = Commit {
            state: super::super::StateCommit {
                retention: Some(retention(&[("g", 4)])),
                ..Default::default()
            },
            ..Default::default()
        };
        enact(&shared, &mut registry, commit, "keeper");
        assert!(registry.replaced, "enacted on the next round");
        assert_eq!(registry.retained(), [("g".to_string(), WorkerId(9))]);
        assert_eq!(
            carried_kinds(&mut registry),
            [("keeper".to_string(), "retention g".to_string())]
        );

        // A call that installs no retention table leaves the one standing.
        enact(&shared, &mut registry, Commit::default(), "keeper");
        assert_eq!(registry.retention.key("g"), 4);

        // Its death, or a reload, withdraws it, and the others hear it.
        let manifest: super::super::Manifest =
            toml::from_str("plane = \"cluster\"\ngrants = [\"retention\"]").unwrap();
        withdraw(&shared, &mut registry, "keeper", &manifest);
        assert!(registry.retained().is_empty());
        assert_eq!(
            carried_kinds(&mut registry),
            [("keeper".to_string(), "retention g".to_string())]
        );
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(
            plane
                .attr(&Subject::Group("g".into()), "retention/key")
                .is_none()
        );

        // A program not granted retention leaves the table when it goes.
        registry.replace_retention("keeper", retention(&[("g", 4)]));
        let other: super::super::Manifest =
            toml::from_str("plane = \"cluster\"\ngrants = [\"replication\"]").unwrap();
        withdraw(&shared, &mut registry, "other", &other);
        assert_eq!(registry.retention.key("g"), 4);
    }

    #[test]
    fn an_exited_process_is_read_with_its_counts_by_the_call_about_its_exit() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let arrived = |id: &str| Local::Arrived {
            id: id.into(),
            tenant: "t".into(),
            session: None,
            props: Props::new(),
            prompt: None,
        };
        for event in [
            arrived("1"),
            arrived("2"),
            Local::Dispatched {
                id: "1".into(),
                worker: "7".into(),
            },
            Local::Counted {
                id: "1".into(),
                prompt: 120,
                generated: 30,
            },
            Local::Exited {
                id: "1".into(),
                outcome: Outcome::Completed,
            },
        ] {
            registry.apply_local(event, &mut edges);
        }
        registry.workers = BTreeSet::from(["1".to_string()]);
        let plane = registry.snapshot(&table(0), &Tables::default());
        let of = |id: &str, name| format!("{:?}", plane.attr(&Subject::Process(id.into()), name));
        let int = |n| format!("{:?}", Some(Value::Int(n)));
        assert_eq!(of("1", "generated-tokens"), int(30));
        assert_eq!(of("1", "prompt-tokens"), int(120));
        assert_eq!(of("1", "worker"), format!("{:?}", text("7")));
        assert_eq!(of("1", "state"), "None");
        assert_eq!(
            of("2", "prompt-tokens"),
            "None",
            "the router never sees a live process's counts"
        );
        assert_eq!(plane.candidates(SubjectKind::Process), ["2"]);
        let worker = |name| format!("{:?}", plane.attr(&Subject::Worker("1".into()), name));
        assert_eq!(worker("waiting"), int(5));
        assert_eq!(worker("running"), int(2));
        assert_eq!(worker("capacity-tokens"), int(1000));
        assert_eq!(worker("used-tokens"), int(250));

        registry.exited.clear();
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(
            plane
                .attr(&Subject::Process("1".into()), "worker")
                .is_none()
        );
    }

    fn host_shared() -> Shared {
        Shared {
            tables: RwLock::default(),
            scale: watch::Sender::new(None),
            heard: Mutex::default(),
            dead: AtomicBool::new(false),
            gate_ms: AtomicU64::new(0),
            terminated: Mutex::default(),
            orders: Mutex::default(),
            courier: Mutex::default(),
            views: Mutex::default(),
        }
    }

    #[test]
    fn the_state_tables_are_taken_and_their_copies_ordered_from_a_site_that_holds_them() {
        let shared = host_shared();
        let mut registry = Registry {
            groups: HashMap::from([("g".to_string(), group(&[1]))]),
            pools: vec!["9".into()],
            directory: Arc::new(DirectoryView {
                peers: vec![(WorkerId(1), "w:1".into())],
                ..DirectoryView::default()
            }),
            ..Default::default()
        };
        let commit = Commit {
            state: super::super::StateCommit {
                replication: Some(vec![wit::Replicate {
                    group: "g".into(),
                    upto: None,
                    at: vec![wit::Site {
                        worker: "2".into(),
                        tier: wit::Tier::Host,
                    }],
                }]),
                retention: Some(retention(&[("g", 4)])),
                ..Default::default()
            },
            ..Default::default()
        };
        enact(&shared, &mut registry, commit, "keeper");
        let plane = registry.snapshot(&table(0), &Tables::default());
        assert!(
            matches!(
                plane.attr(&Subject::Group("g".into()), "retention/key"),
                Some(Value::Int(4))
            ),
            "taken: the programs read it"
        );
        let orders: Vec<(String, WorkerId, String, i64, bool)> = registry
            .orders()
            .into_iter()
            .map(|o| (o.group, o.worker, o.from, o.retention, o.disk))
            .collect();
        assert_eq!(
            orders,
            [
                ("g".to_string(), WorkerId(2), "w:1".to_string(), 4, false),
                ("g".to_string(), WorkerId(9), "w:1".to_string(), 4, false)
            ],
            "both tables' copies, each from the site that holds it"
        );
    }

    #[test]
    fn each_gateway_reads_the_decisions_about_its_own_processes() {
        let shared = host_shared();
        let mut registry = Registry::default();
        let (one, two) = (GatewayId(1), GatewayId(2));
        for (gateway, req) in [(one, 7), (two, 7)] {
            registry.apply(
                Event::Process(
                    gateway,
                    ProcessEvent::Arrived {
                        req,
                        seq: 1,
                        tenant: "t".into(),
                        session: None,
                        props: Vec::new(),
                        prompt: None,
                    },
                ),
                &mut Vec::new(),
                &shared,
            );
        }
        let mut placement = placed("3", &[("tier", "\"gold\"")]);
        placement.props = vec![("tier".into(), Value::Text("gold".into()))];
        let commit = Commit {
            schedule: Some(Table {
                keys: [("1.7".to_string(), 5), ("2.7".to_string(), 1)].into(),
                floor: Some(2),
            }),
            placement: Some(HashMap::from([("1.7".to_string(), placement)])),
            terminate: vec![("2.7".into(), "not today".into())],
            ..Default::default()
        };
        enact(&shared, &mut registry, commit, "p");
        shared.heard.lock().unwrap().insert(one, 1);
        *shared.courier.lock().unwrap() = Some(two);
        *shared.orders.lock().unwrap() = vec![ReplicateOrder {
            group: "g".into(),
            worker: WorkerId(4),
            keys: Vec::new(),
            from: String::new(),
            retention: 0,
            disk: true,
            table: "replication".into(),
        }];

        let first = decisions_of(&shared, one);
        assert!(first.active);
        assert_eq!(first.heard, 1);
        assert_eq!(
            first.schedule,
            Some(ScheduleView {
                keys: vec![(7, 5)],
                floor: Some(2)
            })
        );
        assert_eq!(first.placements.len(), 1);
        assert_eq!(first.placements[0].1.worker, Some(WorkerId(3)));
        assert_eq!(
            first.placements[0].1.props,
            [("tier".to_string(), "\"gold\"".to_string())]
        );
        assert!(first.terminated.is_empty());
        assert!(first.replicate.is_empty(), "not the courier");

        let second = decisions_of(&shared, two);
        assert_eq!(second.heard, 0, "none of its arrivals heard yet");
        assert_eq!(
            second.schedule,
            Some(ScheduleView {
                keys: vec![(7, 1)],
                floor: Some(2)
            })
        );
        assert!(second.placements.is_empty());
        assert_eq!(second.terminated, [(7, "not today".to_string())]);
        assert_eq!(second.replicate.len(), 1, "the courier carries the copies");

        // Its exit ends the termination.
        registry.apply(
            Event::Process(
                two,
                ProcessEvent::Exited {
                    req: 7,
                    outcome: controller_api::Outcome::Cancelled,
                },
            ),
            &mut Vec::new(),
            &shared,
        );
        assert!(decisions_of(&shared, two).terminated.is_empty());

        shared.dead.store(true, Ordering::Release);
        assert_eq!(
            decisions_of(&shared, two).heard,
            u64::MAX,
            "no program: every gate open"
        );
    }

    #[test]
    fn groups_whose_prefixes_cut_to_the_same_pages_are_one_copy_on_disk() {
        let run = |n: u8| Group {
            keys: Arc::new((1..=n).map(|k| [k; 32]).collect()),
            sites: vec![WorkerId(1)],
            ..Group::default()
        };
        let mut registry = Registry {
            groups: HashMap::from([
                ("prompt".to_string(), run(4)),
                ("answer".to_string(), run(6)),
            ]),
            directory: Arc::new(DirectoryView {
                disk_tiers: vec![(WorkerId(1), (16, true))],
                ..DirectoryView::default()
            }),
            ..Default::default()
        };
        let disk = |group: &str| wit::Replicate {
            group: group.into(),
            upto: Some(64),
            at: vec![wit::Site {
                worker: "1".into(),
                tier: wit::Tier::Disk,
            }],
        };
        registry.replication = vec![disk("answer"), disk("prompt")];
        let orders = registry.orders();
        assert_eq!(orders.len(), 1, "the system turn, once");
        assert_eq!(orders[0].keys.len(), 4);
    }

    #[test]
    fn a_worker_s_persistent_disk_tier_reads_as_attrs_and_a_capability() {
        let mut registry = Registry {
            directory: Arc::new(DirectoryView {
                disk_tiers: vec![(WorkerId(1), (16, true))],
                disk: vec![(WorkerId(1), vec![[1; 32], [2; 32]])],
                ..DirectoryView::default()
            }),
            ..Default::default()
        };
        registry.workers = BTreeSet::from(["1".to_string()]);
        let plane = registry.snapshot(&table(0), &Tables::default());
        let worker = Subject::Worker("1".into());
        assert!(matches!(
            plane.attr(&worker, "tier/disk/persistent"),
            Some(Value::Boolean(true))
        ));
        assert!(matches!(
            plane.attr(&worker, "tier/disk/used"),
            Some(Value::Int(32))
        ));
        assert!(plane.capability("kv.persist"));
        let none = Registry::default().snapshot(&table(0), &Tables::default());
        assert!(!none.capability("kv.persist"), "no worker keeps one");
    }

    #[test]
    fn a_process_is_named_by_its_gateway_and_its_request() {
        assert_eq!(pid(GatewayId(3), 41), "3.41");
        assert_eq!(at_gateway("3.41"), Some((GatewayId(3), 41)));
        assert_eq!(at_gateway("41"), None);
    }

    fn programs(names: &[&str]) -> Vec<(String, Vec<u8>)> {
        names
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    builtins::policy(name).unwrap().component.to_vec(),
                )
            })
            .collect()
    }

    fn start(programs: Vec<(String, Vec<u8>)>) -> Result<Arc<Host>, LoadError> {
        Host::start(
            programs,
            watch::channel(table(0)).1,
            watch::channel(DirectoryView::default()).1,
        )
    }

    /// A run cached with no ask and asked kept on disk later is ordered
    /// copied on the round that hears the ask, with nothing else happening.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_asked_kept_after_it_was_cached_is_ordered_at_once() {
        let key = |n: u8| [n; 32];
        let view = |epoch, tokens| DirectoryView {
            epoch,
            groups: vec![(
                key(2),
                controller_api::GroupView {
                    keys: vec![key(1), key(2)],
                    sites: vec![WorkerId(1)],
                    disk_sites: Vec::new(),
                    persist_tokens: tokens,
                },
            )],
            peers: vec![(WorkerId(1), "w1:peer".into())],
            disk_tiers: vec![(WorkerId(1), (16, true))],
            ..DirectoryView::default()
        };
        let (directory, directory_rx) = watch::channel(view(1, None));
        let routing = RoutingTable {
            author: Some(GatewayId(0)),
            ..table(0)
        };
        let host = Host::start(
            programs(&["persist-prefix"]),
            watch::channel(routing).1,
            directory_rx,
        )
        .unwrap();
        let mut decisions = host.decisions(GatewayId(0));
        tokio::time::sleep(STATE_TICK * 2).await;
        assert!(
            decisions.borrow_and_update().replicate.is_empty(),
            "nothing asked"
        );

        directory.send_replace(view(2, Some(32)));
        let ordered = tokio::time::timeout(STATE_TICK * 3, async {
            loop {
                decisions.changed().await.unwrap();
                let orders = decisions.borrow_and_update().replicate.clone();
                if !orders.is_empty() {
                    return orders;
                }
            }
        })
        .await
        .expect("ordered without another event");
        assert_eq!(ordered.len(), 1);
        assert!(ordered[0].disk);
        assert_eq!(ordered[0].keys, vec![key(1), key(2)]);
    }

    #[tokio::test]
    async fn a_placement_s_parts_have_an_author_each() {
        assert!(
            start(programs(&["cache-aware", "disaggregate"])).is_ok(),
            "one places processes, the other names their executors"
        );
    }

    #[tokio::test]
    async fn two_policies_granted_the_same_write_do_not_load() {
        match start(programs(&["test-route", "cache-aware"])) {
            Err(LoadError::Shared {
                grant,
                first,
                second,
            }) => {
                assert_eq!(
                    (grant.as_str(), first.as_str(), second.as_str()),
                    ("placement", "test-route", "cache-aware")
                );
            }
            Err(other) => panic!("refused for sharing a write, got {other}"),
            Ok(_) => panic!("refused for sharing a write, but it loaded"),
        }
    }

    #[tokio::test]
    async fn any_number_of_programs_may_publish() {
        let share = builtins::policy("test-share").unwrap().component.to_vec();
        assert!(
            start(vec![("one".into(), share.clone()), ("two".into(), share)]).is_ok(),
            "each publishes under its own name"
        );
    }
}
