//! The cluster's PLEX 1.1 port: the gateways' process events and the
//! directory into the PLEX host (`plex-host`), and the rows its policies
//! commit back out as each gateway's [`GatewayDecisions`].
//!
//! A process is `"<gateway>.<req>"` to the policies: each gateway counts its
//! own requests from 0. A worker is its id in decimal.
//!
//! The gate: a gateway does not dispatch a launch before every policy has
//! heard it, or its gate bound has passed. Each arrival's host sequence
//! number is kept beside the gateway's own, and a gateway's `heard` is its
//! highest arrival whose host number the host has heard
//! ([`plex_host::host::Host::heard`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use client_api::props::{PROMPT_BLOCK_BYTES, TOKENS_PER_BYTE};
use controller_api::{
    DirectoryView, GatewayDecisions, Health, PlacementView, ProcessEvent, Role, RoutingTable,
};
use ids::{GatewayId, WorkerId};
use plex_host::host::{Config, Delta, Host as PlexHost};
use plex_host::runtime::wit::Tier;
use plex_host::tables::{self, Refusal, Subject, Value};
use tokio::sync::watch;

/// The columns a cluster of pie gateways enacts. Every worker that links to
/// a controller can reach its peers, so helpers are always enactable.
const SUPPORTED: &[&str] = &[tables::PLACEMENT_WORKERS, tables::PLACEMENT_HELPERS];

/// How often the policies hear what the directory and the routing table
/// hold.
const STATE_TICK: Duration = Duration::from_secs(1);

/// How often the port looks at the host while no event comes: the host
/// commits off this thread.
const LOOK: Duration = Duration::from_millis(2);

/// Why the policies did not load.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct LoadError(pub String);

fn hex(key: &[u8; 32]) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}

/// A process's name to the policies.
fn pid(gateway: GatewayId, req: u64) -> String {
    format!("{}.{req}", gateway.0)
}

/// The gateway a process is at, and its id there.
fn at_gateway(pid: &str) -> Option<(GatewayId, u64)> {
    let (gateway, req) = pid.split_once('.')?;
    Some((GatewayId(gateway.parse().ok()?), req.parse().ok()?))
}

/// A policy is named by its built-in name, or by its file's stem.
fn policy_name(spec: &str) -> String {
    std::path::Path::new(spec)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map_or_else(|| spec.to_string(), str::to_string)
}

/// A client's prop (JSON text) as a hint value: lists and objects stay text.
fn hint_value(json: &str) -> Option<Value> {
    use serde_json::Value as Json;
    Some(match serde_json::from_str::<Json>(json).ok()? {
        Json::Null => return None,
        Json::Bool(b) => Value::Boolean(b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64()?),
        },
        Json::String(s) => Value::Text(s),
        json @ (Json::Array(_) | Json::Object(_)) => Value::Text(json.to_string()),
    })
}

fn value_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::Value::from(*i),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        Value::Text(s) => serde_json::Value::String(s.clone()),
    }
}

/// The rows in force, kept for the gateways.
#[derive(Default)]
struct Tables {
    /// `placement/workers` rows: a process's workers, best first.
    workers: HashMap<String, Vec<WorkerId>>,
    /// The column's default, if a policy set one.
    workers_default: Option<Vec<WorkerId>>,
    /// `placement/helpers` rows: the workers a process's prefill may run
    /// on, best first.
    helpers: HashMap<String, Vec<WorkerId>>,
    helpers_default: Option<Vec<WorkerId>>,
    /// Notes on a process, passed on with it: the worker plane reads them
    /// as `cluster/note/<writer>/<name>`.
    notes: HashMap<String, BTreeMap<String, String>>,
}

struct Shared {
    tables: RwLock<Tables>,
    /// Each gateway's highest arrival every policy has heard.
    heard: Mutex<HashMap<GatewayId, u64>>,
    /// No policy runs: every gate is open.
    dead: AtomicBool,
    gate_ms: AtomicU64,
    /// Processes the policies ended, and why, until they exit.
    terminated: Mutex<BTreeMap<String, String>>,
    /// Each gateway's live processes: a default row reaches them.
    live: Mutex<HashMap<GatewayId, HashSet<u64>>>,
    views: Mutex<HashMap<GatewayId, watch::Sender<GatewayDecisions>>>,
}

/// The cluster policies, running.
pub struct Host {
    events: Sender<Event>,
    shared: Arc<Shared>,
}

enum Event {
    Process(GatewayId, ProcessEvent),
    /// Swap every policy for these (name, component): the answer is why
    /// they did not load, if they did not.
    Reload {
        policies: Vec<(String, Vec<u8>)>,
        answer: tokio::sync::oneshot::Sender<Result<(), LoadError>>,
    },
    /// The directory and the routing table as of now.
    State(Arc<DirectoryView>, Arc<RoutingTable>),
}

/// A launch's gate time: one call's deadline per policy, the longest a
/// round that heard the arrival can take before a policy is stuck.
fn gate_ms(config: &Config, policies: usize) -> u64 {
    config.deadline_ms * policies as u64
}

impl Host {
    /// Starts the cluster policies `policies` names, (spec, component), in
    /// one PLEX host: each writes only the columns it imports, which no
    /// other may.
    pub fn start(
        policies: Vec<(String, Vec<u8>)>,
        mut routing: watch::Receiver<RoutingTable>,
        mut directory: watch::Receiver<DirectoryView>,
    ) -> Result<Arc<Self>, LoadError> {
        let config = Config {
            supported: SUPPORTED.iter().map(|c| c.to_string()).collect(),
            ..Config::default()
        };
        let mut host = PlexHost::new(config.clone());
        let mut names = Vec::new();
        for (spec, component) in &policies {
            let name = policy_name(spec);
            host.load(&name, component).map_err(LoadError)?;
            names.push(name);
        }
        host.start();
        let shared = Arc::new(Shared {
            tables: RwLock::default(),
            heard: Mutex::default(),
            dead: AtomicBool::new(names.is_empty()),
            gate_ms: AtomicU64::new(gate_ms(&config, names.len())),
            terminated: Mutex::default(),
            live: Mutex::default(),
            views: Mutex::default(),
        });
        let (events, rx) = channel();
        std::thread::Builder::new()
            .name("pie-cluster-plex".into())
            .spawn({
                let shared = shared.clone();
                let loaded = policies
                    .into_iter()
                    .map(|(spec, component)| (policy_name(&spec), component))
                    .collect();
                move || run(host, config, loaded, rx, shared)
            })
            .map_err(|e| LoadError(format!("spawning the cluster port: {e}")))?;
        // The workers as the policies first run, again whenever the routing
        // table changes (a row naming a worker the host does not know is
        // ignored), and the directory every tick.
        let ticks = events.clone();
        tokio::spawn(async move {
            loop {
                let view = Arc::new(directory.borrow_and_update().clone());
                let table = Arc::new(routing.borrow_and_update().clone());
                if ticks.send(Event::State(view, table)).is_err() {
                    return;
                }
                tokio::select! {
                    _ = tokio::time::sleep(STATE_TICK) => {}
                    changed = routing.changed() => if changed.is_err() { return },
                }
            }
        });
        tracing::info!(policies = ?names, "cluster policies started (PLEX 1.1)");
        Ok(Arc::new(Self { events, shared }))
    }

    /// What happened to `gateway`'s processes, in order.
    pub fn events(&self, gateway: GatewayId, events: Vec<ProcessEvent>) {
        for event in events {
            let _ = self.events.send(Event::Process(gateway, event));
        }
    }

    /// The policies' decisions about `gateway`'s processes, as they change.
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

    /// Swaps every running policy for `policies` (spec, component), as a
    /// SIGHUP asks: the old ones are unloaded (their rows withdrawn) and
    /// the new ones loaded; if one does not load, the old set is loaded
    /// again.
    pub async fn reload(&self, policies: Vec<(String, Vec<u8>)>) -> Result<(), LoadError> {
        if policies.is_empty() {
            return Err(LoadError(
                "a reload names at least one cluster policy; a controller with none starts \
                 without them"
                    .into(),
            ));
        }
        let policies = policies
            .into_iter()
            .map(|(spec, component)| (policy_name(&spec), component))
            .collect();
        let (answer, answered) = tokio::sync::oneshot::channel();
        self.events
            .send(Event::Reload { policies, answer })
            .map_err(|_| LoadError("the cluster port has ended".into()))?;
        answered
            .await
            .map_err(|_| LoadError("the cluster port has ended".into()))??;
        publish(&self.shared);
        Ok(())
    }
}

/// `gateway`'s view as the rows leave it now, epoch 0.
fn decisions_of(shared: &Shared, gateway: GatewayId) -> GatewayDecisions {
    let mine = |id: &str| {
        at_gateway(id)
            .filter(|(g, _)| *g == gateway)
            .map(|(_, req)| req)
    };
    let tables = shared.tables.read().unwrap();
    let mut placements: BTreeMap<u64, PlacementView> = BTreeMap::new();
    // The default row reaches every live process without a row of its own.
    if let Some(default) = &tables.workers_default {
        for &req in shared
            .live
            .lock()
            .unwrap()
            .get(&gateway)
            .into_iter()
            .flatten()
        {
            placements.entry(req).or_default().workers = Some(default.clone());
        }
    }
    for (id, workers) in &tables.workers {
        if let Some(req) = mine(id) {
            placements.entry(req).or_default().workers = Some(workers.clone());
        }
    }
    if let Some(default) = &tables.helpers_default {
        for &req in shared
            .live
            .lock()
            .unwrap()
            .get(&gateway)
            .into_iter()
            .flatten()
        {
            placements.entry(req).or_default().executors = default.clone();
        }
    }
    for (id, helpers) in &tables.helpers {
        if let Some(req) = mine(id) {
            placements.entry(req).or_default().executors = helpers.clone();
        }
    }
    for (id, notes) in &tables.notes {
        if let Some(req) = mine(id) {
            placements.entry(req).or_default().props = notes
                .iter()
                .map(|(name, json)| (name.clone(), json.clone()))
                .collect();
        }
    }
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
    GatewayDecisions {
        epoch: 0,
        active: true,
        gate_ms: shared.gate_ms.load(Ordering::Acquire),
        heard,
        placements: placements.into_iter().collect(),
        terminated,
        replicate: Vec::new(),
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

/// The port's own record.
struct Port {
    host: PlexHost,
    config: Config,
    shared: Arc<Shared>,
    loaded: Vec<(String, Vec<u8>)>,
    /// Arrivals not yet heard: (host sequence, gateway, gateway sequence).
    unheard: Vec<(u64, GatewayId, u64)>,
    /// The workers last put to the host.
    workers: HashSet<WorkerId>,
    /// The working sets last put to the host, each with a worker that held
    /// it (named when it goes).
    sets: HashMap<String, String>,
    version: u64,
}

fn run(
    host: PlexHost,
    config: Config,
    loaded: Vec<(String, Vec<u8>)>,
    events: Receiver<Event>,
    shared: Arc<Shared>,
) {
    let mut port = Port {
        host,
        config,
        shared,
        loaded,
        unheard: Vec::new(),
        workers: HashSet::new(),
        sets: HashMap::new(),
        version: 0,
    };
    loop {
        let first = match events.recv_timeout(LOOK) {
            Ok(event) => Some(event),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        let mut changed = false;
        for event in first.into_iter().chain(events.try_iter()) {
            match event {
                Event::Process(gateway, event) => changed |= port.apply(gateway, event),
                Event::State(view, table) => port.state(&view, &table),
                Event::Reload { policies, answer } => {
                    let _ = answer.send(port.reload(policies));
                    changed = true;
                }
            }
        }
        if port.host.due() {
            port.host.tick(Vec::new());
        }
        // What the host has heard is read before its rows: every row a round
        // committed is in force before a gate that round opens.
        let heard = port.host.heard();
        let version = port.host.version();
        if version != port.version {
            port.version = version;
            let delta = port.host.poll_delta();
            port.enact(delta);
            changed = true;
        }
        changed |= port.hear(heard);
        if changed {
            publish(&port.shared);
        }
    }
}

impl Port {
    fn reload(&mut self, policies: Vec<(String, Vec<u8>)>) -> Result<(), LoadError> {
        for (name, _) in &self.loaded {
            self.host.unload(name);
        }
        for (i, (name, component)) in policies.iter().enumerate() {
            if let Err(error) = self.host.load(name, component) {
                tracing::error!(%error, policy = %name, "cluster policies not reloaded; the running ones go on");
                for (name, _) in &policies[..i] {
                    self.host.unload(name);
                }
                for (name, component) in &self.loaded {
                    let _ = self.host.load(name, component);
                }
                return Err(LoadError(error));
            }
        }
        tracing::info!(policies = ?policies.iter().map(|(n, _)| n).collect::<Vec<_>>(), "cluster policies reloaded (PLEX 1.1)");
        self.shared
            .dead
            .store(policies.is_empty(), Ordering::Release);
        self.shared
            .gate_ms
            .store(gate_ms(&self.config, policies.len()), Ordering::Release);
        self.loaded = policies;
        Ok(())
    }

    /// Whether the event changed a gateway's view.
    fn apply(&mut self, gateway: GatewayId, event: ProcessEvent) -> bool {
        let host = &self.host;
        match event {
            ProcessEvent::Arrived {
                req,
                seq,
                tenant,
                session,
                props,
                prompt,
            } => {
                let id = pid(gateway, req);
                // Tokens a text block makes, guessed from its bytes: the
                // prompt reaches the controller as blocks, not tokens.
                let per_block = TOKENS_PER_BYTE * PROMPT_BLOCK_BYTES as f64;
                let prompt_tokens = prompt.map(|blocks| (blocks.len() as f64 * per_block) as u32);
                let max_tokens = props
                    .iter()
                    .find(|(name, _)| name == client_api::props::MAX_TOKENS)
                    .and_then(|(_, json)| json.parse::<u32>().ok());
                let hints = props
                    .iter()
                    .filter_map(|(name, json)| Some((name.clone(), hint_value(json)?)))
                    .collect();
                let session = session.map(|session| format!("{tenant}/{session}"));
                let at = host.arrived(
                    &id,
                    session,
                    Some(tenant),
                    None,
                    prompt_tokens,
                    max_tokens,
                    hints,
                    0,
                );
                self.unheard.push((at, gateway, seq));
                self.shared
                    .live
                    .lock()
                    .unwrap()
                    .entry(gateway)
                    .or_default()
                    .insert(req);
                false
            }
            ProcessEvent::Dispatched { req, worker } => {
                host.dispatched(&pid(gateway, req), &worker.0.to_string());
                false
            }
            ProcessEvent::Started { req } => {
                host.started(&pid(gateway, req));
                false
            }
            ProcessEvent::Moved { req, worker, why } => {
                let id = pid(gateway, req);
                match worker {
                    Some(worker) => host.dispatched(&id, &worker.0.to_string()),
                    None => {
                        tracing::info!(process = %id, %why, "a placement that moves a running process is refused");
                        host.refuse(
                            tables::PLACEMENT_WORKERS,
                            &id,
                            None,
                            Some(Refusal::NotSupported),
                        );
                    }
                }
                false
            }
            ProcessEvent::Exited { req, outcome } => {
                let id = pid(gateway, req);
                let outcome = match outcome {
                    controller_api::Outcome::Completed => "completed",
                    controller_api::Outcome::Failed => "failed",
                    controller_api::Outcome::Cancelled => "cancelled",
                };
                host.exited(&id, outcome, None, None, 0);
                if let Some(live) = self.shared.live.lock().unwrap().get_mut(&gateway) {
                    live.remove(&req);
                }
                self.shared.terminated.lock().unwrap().remove(&id).is_some()
            }
            ProcessEvent::SessionProps { id, changes } => {
                for (name, json) in changes {
                    host.hint(Subject::Session(id.clone()), &name, hint_value(&json));
                }
                false
            }
            ProcessEvent::SessionClosed { id } => {
                host.closed(&id);
                false
            }
            // Which workers a gateway reaches is read from the routing table.
            ProcessEvent::Workers(_) => false,
        }
    }

    /// The workers and the working sets, as the routing table and the
    /// directory have them now.
    fn state(&mut self, view: &DirectoryView, table: &RoutingTable) {
        let pools: HashSet<WorkerId> = view.pools.iter().copied().collect();
        let page: HashMap<WorkerId, u64> = view
            .disk_tiers
            .iter()
            .map(|(w, (page, _))| (*w, u64::from(*page)))
            .collect();
        let mut now = HashSet::new();
        for worker in &table.workers {
            if worker.health == Health::Unreachable {
                continue;
            }
            let load = &worker.coarse_load;
            let tiers = if pools.contains(&worker.id) {
                vec![(Tier::Host, None, 0)]
            } else {
                let capacity = (load.capacity_tokens > 0).then_some(load.capacity_tokens);
                vec![(Tier::Device, capacity, load.used_tokens)]
            };
            let id = worker.id.0.to_string();
            self.host
                .put_worker(&id, load.draining, tiers, load.running, load.waiting, None);
            let role = match worker.role {
                Role::Prefill => "prefill",
                Role::Decode => "decode",
                Role::Encode => "encode",
            };
            self.host.attr(
                Subject::Worker(id.clone()),
                "pie/role",
                Some(Value::Text(role.into())),
            );
            self.host.attr(
                Subject::Worker(id),
                "pie/model",
                Some(Value::Text(worker.model.clone())),
            );
            now.insert(worker.id);
        }
        for gone in self.workers.difference(&now) {
            self.host.remove_worker(&gone.0.to_string());
        }
        self.workers = now;

        // A group (a run kept whole from token 0) is a working set, named by
        // its last key; its copies are where the directory has it.
        let mut sets = HashMap::new();
        for (key, group) in &view.groups {
            let id = hex(key);
            let page = group
                .sites
                .iter()
                .chain(&group.disk_sites)
                .find_map(|w| page.get(w).copied())
                .unwrap_or(1);
            let tokens = u32::try_from(group.keys.len() as u64 * page).unwrap_or(u32::MAX);
            let mut copies: Vec<(String, Tier, u32)> = group
                .sites
                .iter()
                .map(|w| {
                    let tier = if pools.contains(w) {
                        Tier::Host
                    } else {
                        Tier::Device
                    };
                    (w.0.to_string(), tier, tokens)
                })
                .collect();
            copies.extend(
                group
                    .disk_sites
                    .iter()
                    .map(|w| (w.0.to_string(), Tier::Disk, tokens)),
            );
            let held = copies
                .first()
                .map(|(w, _, _)| w.clone())
                .unwrap_or_default();
            self.host
                .put_working_set(&id, None, None, None, tokens, 0, copies);
            sets.insert(id, held);
        }
        for (gone, held) in &self.sets {
            if !sets.contains_key(gone) {
                self.host.evicted(gone, held, "host", true);
            }
        }
        self.sets = sets;
    }

    /// The rows a round committed.
    fn enact(&mut self, delta: Delta) {
        let mut tables = self.shared.tables.write().unwrap();
        let parse = |list: Vec<String>| -> Vec<WorkerId> {
            list.into_iter()
                .filter_map(|w| w.parse().ok().map(WorkerId))
                .collect()
        };
        for (id, row) in delta.placement_workers.rows {
            match row {
                Some(list) => {
                    tables.workers.insert(id, parse(list));
                }
                None => {
                    tables.workers.remove(&id);
                }
            }
        }
        if let Some(default) = delta.placement_workers.default {
            tables.workers_default = default.map(parse);
        }
        for (id, row) in delta.placement_helpers.rows {
            match row {
                Some(list) => {
                    tables.helpers.insert(id, parse(list));
                }
                None => {
                    tables.helpers.remove(&id);
                }
            }
        }
        if let Some(default) = delta.placement_helpers.default {
            tables.helpers_default = default.map(parse);
        }
        for (writer, subject, name, value) in delta.notes {
            tracing::debug!(%writer, subject = %format!("{}/{}", subject.kind(), subject.id()), %name, ?value, "cluster policy note");
            let Subject::Process(id) = subject else {
                continue;
            };
            let name = format!("{}note/{writer}/{name}", client_api::props::CLUSTER_PREFIX);
            let notes = tables.notes.entry(id).or_default();
            match value {
                Some(value) => {
                    notes.insert(name, value_json(&value).to_string());
                }
                None => {
                    notes.remove(&name);
                }
            }
        }
        drop(tables);
        let mut terminated = self.shared.terminated.lock().unwrap();
        for (id, reason) in delta.terminate {
            terminated.insert(id, reason);
        }
    }

    /// Moves each gateway's `heard` up to its highest arrival the host has
    /// heard. Whether one moved.
    fn hear(&mut self, heard: u64) -> bool {
        if self.unheard.is_empty() {
            return false;
        }
        let mut moved = false;
        let mut gateways = self.shared.heard.lock().unwrap();
        self.unheard.retain(|&(at, gateway, seq)| {
            if at > heard {
                return true;
            }
            let h = gateways.entry(gateway).or_insert(0);
            if seq > *h {
                *h = seq;
                moved = true;
            }
            false
        });
        moved
    }
}
