//! The router policy on its own thread, with its own record of processes built
//! from the same events as its edges.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use client_api::props::{PROMPT_BLOCK_BYTES, PromptIndex, TOKENS_PER_BYTE};
use client_api::{ClientMessage, Props};
use controller_api::{Health, Role, RoutableWorker, RoutingTable};
use ids::{ReqId, WorkerId};
use tokio::sync::{Notify, mpsc, watch};
use worker_api::Request;

use super::program::{self, Commit, Fault, LoadError, Plane, Program, Table};
use super::wit::{self, Edge, Outcome, Subject, SubjectKind, Value};
use crate::admission::{AdmissionDecision, RejectReason};
use crate::route::RoutingHandle;
use crate::session::AdmitReject;

fn capability(name: &str) -> bool {
    name == "router.hold"
}

#[derive(Default)]
struct Tables {
    schedule: Option<Table>,
    placement: HashMap<String, wit::Placement>,
}

struct Shared {
    tables: RwLock<Tables>,
    waiting: Mutex<BTreeMap<u64, String>>,
    refused: Mutex<HashMap<String, Option<String>>>,
    cancels: mpsc::UnboundedSender<ReqId>,
    /// The last arrival a call has heard, or would have had it listened.
    heard: watch::Sender<u64>,
    changed: Notify,
}

pub struct Router {
    events: Sender<Event>,
    shared: Arc<Shared>,
    gate: Duration,
    next_seq: AtomicU64,
    dead: Arc<AtomicBool>,
}

enum Event {
    Arrived {
        id: String,
        seq: u64,
        tenant: String,
        session: Option<String>,
        props: Props,
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
    Workers(BTreeSet<String>),
    SessionProps {
        id: String,
        changes: Props,
    },
    SessionClosed {
        id: String,
    },
}

impl Router {
    pub fn start(
        name: &str,
        component: &[u8],
        routing: watch::Receiver<RoutingTable>,
        mut connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
    ) -> Result<(Arc<Self>, mpsc::UnboundedReceiver<ReqId>), LoadError> {
        let (cancels, cancelled) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared {
            tables: RwLock::default(),
            waiting: Mutex::default(),
            refused: Mutex::default(),
            cancels,
            heard: watch::Sender::new(0),
            changed: Notify::new(),
        });
        let (events, rx) = channel();
        let dead = Arc::new(AtomicBool::new(false));
        let manifest = program::spawn(name, component, capability, "pie-router-policy", {
            let (shared, dead) = (shared.clone(), dead.clone());
            move |program| run(program, rx, shared, routing, dead)
        })?;
        let gate = Duration::from_millis(manifest.gate_ms);

        let workers = events.clone();
        tokio::spawn(async move {
            loop {
                let ids = connected
                    .borrow_and_update()
                    .iter()
                    .map(|w| w.0.to_string())
                    .collect();
                if workers.send(Event::Workers(ids)).is_err() || connected.changed().await.is_err()
                {
                    return;
                }
            }
        });
        tracing::info!(policy = name, "router policy started");
        Ok((
            Arc::new(Self {
                events,
                shared,
                gate,
                next_seq: AtomicU64::new(0),
                dead,
            }),
            cancelled,
        ))
    }

    fn send(&self, event: Event) {
        let _ = self.events.send(event);
    }

    pub async fn admit(&self, req: &Request, routing: &RoutingHandle) -> Result<(), AdmitReject> {
        let id = req.req_id.0.to_string();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let (props, session) = match &req.message {
            ClientMessage::LaunchProcess { props, session, .. } => (props.clone(), session.clone()),
            _ => Default::default(),
        };
        self.shared.refused.lock().unwrap().insert(id.clone(), None);
        self.send(Event::Arrived {
            id: id.clone(),
            seq,
            tenant: req.tenant.0.clone(),
            session,
            props,
        });
        let admitted = {
            let mut waiting = Waiting {
                router: self,
                seq,
                id: &id,
                left: false,
            };
            self.pass(&mut waiting, req, routing).await
        };
        if admitted.is_err() {
            self.send(Event::Exited {
                id,
                outcome: Outcome::Cancelled,
            });
        }
        admitted
    }

    async fn pass(
        &self,
        waiting: &mut Waiting<'_>,
        req: &Request,
        routing: &RoutingHandle,
    ) -> Result<(), AdmitReject> {
        let (id, seq) = (waiting.id, waiting.seq);
        if !self.gate.is_zero() && !self.dead.load(Ordering::Acquire) {
            // Unanswered, the launch goes by the router's default.
            let mut heard = self.shared.heard.subscribe();
            let _ = tokio::time::timeout(self.gate, heard.wait_for(|&h| h >= seq)).await;
        }

        {
            let mut queue = self.shared.waiting.lock().unwrap();
            if queue.len() >= routing.max_queued_launches() {
                drop(queue);
                return waiting.leave(Err(AdmitReject::Full(RejectReason::QueueFull.to_string())));
            }
            queue.insert(seq, id.to_string());
        }
        let mut table = routing.table_watch();
        loop {
            let changed = self.shared.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if matches!(self.shared.refused.lock().unwrap().get(id), Some(Some(_))) {
                return waiting.leave(Ok(()));
            }
            if self.next(seq) {
                match routing.admit(req) {
                    AdmissionDecision::Admit => return waiting.leave(Ok(())),
                    AdmissionDecision::Reject(RejectReason::ClusterSaturated) => {}
                    AdmissionDecision::Reject(reason) => {
                        return waiting.leave(Err(AdmitReject::Full(reason.to_string())));
                    }
                }
            }
            tokio::select! {
                _ = changed => {}
                closed = table.changed() => if closed.is_err() {
                    return waiting.leave(Err(AdmitReject::Full(
                        RejectReason::ClusterSaturated.to_string(),
                    )));
                },
            }
        }
    }

    /// The highest key the schedule does not hold, the earliest among equals.
    fn next(&self, seq: u64) -> bool {
        let tables = self.shared.tables.read().unwrap();
        let schedule = tables.schedule.as_ref();
        let key = |id: &str| schedule.map_or(0, |table| table.key(id));
        let held = |id: &str| schedule.is_some_and(|t| t.floor.is_some_and(|f| t.key(id) < f));
        self.shared
            .waiting
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, id)| !held(id))
            .max_by_key(|(s, id)| (key(id), Reverse(**s)))
            .is_some_and(|(s, _)| *s == seq)
    }

    pub fn placement(&self, req: &Request) -> (Option<WorkerId>, Props) {
        let tables = self.shared.tables.read().unwrap();
        let Some(placement) = tables.placement.get(&req.req_id.0.to_string()) else {
            return (None, Props::new());
        };
        let worker = placement
            .worker
            .as_deref()
            .and_then(|w| w.parse().ok())
            .map(WorkerId);
        let props = placement
            .props
            .iter()
            .filter_map(|(name, value)| Some((name.clone(), json(value)?)))
            .collect();
        (worker, props)
    }

    pub fn dispatched(&self, req: &Request, worker: WorkerId) {
        self.send(Event::Dispatched {
            id: req.req_id.0.to_string(),
            worker: worker.0.to_string(),
        });
    }

    pub fn counted(&self, req: &Request, prompt: u64, generated: u64) {
        self.send(Event::Counted {
            id: req.req_id.0.to_string(),
            prompt,
            generated,
        });
    }

    pub fn session_props(&self, id: String, changes: Props) {
        self.send(Event::SessionProps { id, changes });
    }

    pub fn session_closed(&self, id: String) {
        self.send(Event::SessionClosed { id });
    }

    pub fn exited(&self, req: &Request, outcome: Outcome) {
        self.send(Event::Exited {
            id: req.req_id.0.to_string(),
            outcome,
        });
    }
}

struct Waiting<'a> {
    router: &'a Router,
    seq: u64,
    id: &'a str,
    left: bool,
}

impl Waiting<'_> {
    fn leave(&mut self, outcome: Result<(), AdmitReject>) -> Result<(), AdmitReject> {
        self.left = true;
        let shared = &self.router.shared;
        let refused = {
            let mut waiting = shared.waiting.lock().unwrap();
            waiting.remove(&self.seq);
            shared.refused.lock().unwrap().remove(self.id).flatten()
        };
        shared.changed.notify_waiters();
        refused.map_or(outcome, |reason| Err(AdmitReject::Refused(reason)))
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        if self.left {
            return;
        }
        let _ = self.leave(Ok(()));
        self.router.send(Event::Exited {
            id: self.id.to_string(),
            outcome: Outcome::Cancelled,
        });
    }
}

fn run(
    mut program: Program,
    events: Receiver<Event>,
    shared: Arc<Shared>,
    routing: watch::Receiver<RoutingTable>,
    dead: Arc<AtomicBool>,
) {
    let mut registry = Registry::default();
    loop {
        let first = match program.alarm() {
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
        };
        let mut edges = Vec::new();
        let mut heard = None;
        registry.exited.clear();
        for event in first.into_iter().chain(events.try_iter()) {
            if let Event::Arrived { seq, .. } = &event {
                heard = heard.max(Some(*seq));
            }
            registry.apply(event, &mut edges);
        }
        let now = now_ms();
        if program.alarm().is_some_and(|at_ms| at_ms <= now) {
            edges.push(Edge::Timer);
        }

        if !edges.is_empty() {
            tracing::debug!(policy = program.name(), ?edges, "router policy edges");
            let plane = registry.snapshot(&routing.borrow());
            match program.react(Box::new(plane), edges, now) {
                Ok(commit) => {
                    tracing::debug!(
                        policy = program.name(),
                        schedule = commit.schedule.as_ref().map(|t| t.keys.len()),
                        floor = commit.schedule.as_ref().and_then(|t| t.floor),
                        placement = commit.placement.as_ref().map(|t| t.len()),
                        "router policy call"
                    );
                    enact(&shared, &mut registry, commit);
                }
                Err(Fault::Dead(error)) => {
                    tracing::error!(policy = program.name(), %error, "router policy is dead; its tables are withdrawn");
                    dead.store(true, Ordering::Release);
                    *shared.tables.write().unwrap() = Tables::default();
                    shared.heard.send_replace(u64::MAX);
                    shared.changed.notify_waiters();
                    return;
                }
                Err(fault) => {
                    tracing::warn!(policy = program.name(), %fault, "router policy call discarded");
                }
            }
        }
        // Whether or not a call ran, the gate has nothing more to wait for.
        if let Some(seq) = heard {
            shared.heard.send_modify(|h| *h = (*h).max(seq));
        }
    }
}

fn enact(shared: &Shared, registry: &mut Registry, commit: Commit) {
    let mut changed = false;
    {
        let mut tables = shared.tables.write().unwrap();
        if let Some(schedule) = commit.schedule {
            tables.schedule = Some(schedule);
            changed = true;
        }
        if let Some(placement) = commit.placement {
            tables.placement = placement;
        }
    }
    for (id, reason) in commit.terminate {
        if !registry.procs.contains_key(&id) {
            continue;
        }
        // Not yet admitted, it is refused; admitted, the gateway cancels it.
        if let Some(refused) = shared.refused.lock().unwrap().get_mut(&id) {
            *refused = Some(reason);
            changed = true;
        } else if let Ok(req) = id.parse() {
            registry.terminated.insert(id);
            let _ = shared.cancels.send(ReqId(req));
        }
    }
    if changed {
        shared.changed.notify_waiters();
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
}

/// The prompt blocks the router counts a worker as holding: about 4M tokens
/// of text.
const HELD_BLOCKS: usize = 1 << 18;

impl Registry {
    fn apply(&mut self, event: Event, edges: &mut Vec<Edge>) {
        match event {
            Event::Arrived {
                id,
                tenant,
                session,
                props,
                ..
            } => {
                let prompt = client_api::props::prompt_of(props.iter())
                    .map(|text| client_api::props::prompt_blocks(text).into());
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
            Event::Dispatched { id, worker } => {
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
            Event::Counted {
                id,
                prompt,
                generated,
            } => {
                tracing::debug!(process = %id, prompt, generated, "router policy: a process counted its tokens");
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.counted = Some((prompt, generated));
                }
            }
            Event::Exited { id, outcome } => {
                self.terminated.remove(&id);
                if let Some(proc) = self.procs.remove(&id) {
                    self.exited.insert(id.clone(), proc);
                    edges.push(Edge::Exited(wit::Exit {
                        process: id,
                        outcome,
                    }));
                }
            }
            Event::SessionProps { id, changes } => {
                let current = self.sessions.remove(&id).unwrap_or_default();
                let (merged, names) = match client_api::props::merge(&current, changes) {
                    Ok(merged) => merged,
                    Err(why) => {
                        tracing::warn!(session = %id, %why, "router policy: session props refused");
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
            Event::SessionClosed { id } => {
                self.sessions.remove(&id);
                edges.push(Edge::Closed(id));
            }
            Event::Workers(now) => {
                for removed in self.workers.difference(&now) {
                    edges.push(Edge::WorkerRemoved(removed.clone()));
                    self.held.remove(removed);
                }
                self.workers = now;
            }
        }
    }

    fn snapshot(&self, routing: &RoutingTable) -> Snapshot {
        Snapshot {
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
    procs: BTreeMap<String, Proc>,
    exited: BTreeMap<String, Proc>,
    sessions: HashMap<String, Props>,
    held: HashMap<String, Arc<PromptIndex>>,
    workers: BTreeMap<String, RoutableWorker>,
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
            SubjectKind::Worker => self.workers.keys().cloned().collect(),
            SubjectKind::WorkingSet => Vec::new(),
        }
    }

    fn attr(&self, of: &Subject, name: &str) -> Option<Value> {
        match of {
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
            Subject::Worker(id) => {
                let worker = self.workers.get(id)?;
                match name {
                    "role" => text(match worker.role {
                        Role::Prefill => "prefill",
                        Role::Decode => "decode",
                        Role::Encode => "encode",
                    }),
                    "model" => text(worker.model.clone()),
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
                    _ => None,
                }
            }
            Subject::WorkingSet(_) => None,
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
        capability(name)
    }

    fn terminable(&self, process: &str) -> bool {
        self.procs.contains_key(process)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use controller_api::WorkerStatus;
    use ids::{SessionId, TenantId};
    use worker_api::Priority;

    use super::*;

    fn table(kv: u8) -> RoutingTable {
        RoutingTable {
            epoch: 1,
            workers: vec![RoutableWorker {
                id: WorkerId(1),
                addr: "w1".into(),
                role: Role::Decode,
                model: "m".into(),
                health: Health::Healthy,
                coarse_load: WorkerStatus {
                    kv_pressure_bucket: kv,
                    inflight: 0,
                    running: 2,
                    waiting: 5,
                    capacity_tokens: 1000,
                    used_tokens: 250,
                },
            }],
        }
    }

    fn launch(req: u64, props: serde_json::Value) -> Request {
        Request {
            req_id: ReqId(req),
            session: SessionId(1),
            tenant: TenantId("t".into()),
            priority: Priority::Normal,
            blobs: Vec::new(),
            message: ClientMessage::LaunchProcess {
                corr_id: 1,
                inferlet: "chat".into(),
                input: String::new(),
                capture_outputs: true,
                props: serde_json::from_value(props).unwrap(),
                session: None,
                router_props: Props::new(),
            },
            router_props: Props::new(),
        }
    }

    #[test]
    fn an_exited_process_is_read_with_its_counts_by_the_call_about_its_exit() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let arrived = |id: &str| Event::Arrived {
            id: id.into(),
            seq: 0,
            tenant: "t".into(),
            session: None,
            props: Props::new(),
        };
        for event in [
            arrived("1"),
            arrived("2"),
            Event::Dispatched {
                id: "1".into(),
                worker: "7".into(),
            },
            Event::Counted {
                id: "1".into(),
                prompt: 120,
                generated: 30,
            },
            Event::Exited {
                id: "1".into(),
                outcome: Outcome::Completed,
            },
            Event::Workers(BTreeSet::from(["1".to_string()])),
        ] {
            registry.apply(event, &mut edges);
        }
        let plane = registry.snapshot(&table(0));
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
        let plane = registry.snapshot(&table(0));
        assert!(
            plane
                .attr(&Subject::Process("1".into()), "worker")
                .is_none()
        );
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
        registry.apply(
            Event::Arrived {
                id: "1".into(),
                seq: 0,
                tenant: "t".into(),
                session: None,
                props,
            },
            &mut edges,
        );
        let plane = registry.snapshot(&table(0));
        let of = |name| format!("{:?}", plane.attr(&Subject::Process("1".into()), name));
        assert_eq!(of("prompt-tokens"), format!("{:?}", Some(Value::Int(160))));
        assert_eq!(of("max-tokens"), format!("{:?}", Some(Value::Int(64))));
    }

    #[test]
    fn a_worker_holds_the_prompts_dispatched_to_it() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let text = "word ".repeat(200);
        let arrived = |id: &str, prompt: Option<String>| Event::Arrived {
            id: id.into(),
            seq: 0,
            tenant: "t".into(),
            session: None,
            props: prompt
                .map(|prompt| {
                    serde_json::from_value(serde_json::json!({ "prompt": prompt })).unwrap()
                })
                .unwrap_or_default(),
        };
        for event in [
            arrived("1", Some(text.clone())),
            Event::Dispatched {
                id: "1".into(),
                worker: "1".into(),
            },
            arrived("2", Some(format!("{text}and more"))),
            arrived("3", None),
            Event::Workers(BTreeSet::from(["1".to_string()])),
        ] {
            registry.apply(event, &mut edges);
        }
        let plane = registry.snapshot(&table(0));
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
    fn a_session_s_props_reach_the_router_program_between_launches() {
        let mut registry = Registry::default();
        let mut edges = Vec::new();
        let changes: Props =
            serde_json::from_value(serde_json::json!({"tool-eta-ms": 900})).unwrap();
        registry.apply(
            Event::SessionProps {
                id: "default/s".into(),
                changes,
            },
            &mut edges,
        );
        assert!(matches!(&edges[..], [Edge::PropChanged(c)] if c.names == ["tool-eta-ms"]));
        let plane = registry.snapshot(&table(0));
        assert_eq!(plane.candidates(SubjectKind::Session), ["default/s"]);
        assert!(matches!(
            plane.prop(&Subject::Session("default/s".into()), "tool-eta-ms"),
            Some(Value::Int(900))
        ));
        edges.clear();
        registry.apply(
            Event::SessionClosed {
                id: "default/s".into(),
            },
            &mut edges,
        );
        assert!(matches!(&edges[..], [Edge::Closed(s)] if s == "default/s"));
        let plane = registry.snapshot(&table(0));
        assert!(
            plane
                .prop(&Subject::Session("default/s".into()), "tool-eta-ms")
                .is_none()
        );
    }

    struct Rig {
        router: Arc<Router>,
        routing: RoutingHandle,
        table: watch::Sender<RoutingTable>,
        cancelled: mpsc::UnboundedReceiver<ReqId>,
        _connected: watch::Sender<Arc<HashSet<WorkerId>>>,
    }

    fn rig(kv: u8) -> Rig {
        let (table, table_rx) = watch::channel(table(kv));
        let (connected, connected_rx) = watch::channel(Arc::new(HashSet::from([WorkerId(1)])));
        let routing = RoutingHandle::new(table_rx.clone(), connected_rx.clone());
        let component = builtins::policy("test-route").unwrap().component;
        let (router, cancelled) =
            Router::start("test-route", component, table_rx, connected_rx).unwrap();
        Rig {
            router,
            routing,
            table,
            cancelled,
            _connected: connected,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_launch_is_held_below_the_floor_and_placed_where_its_policy_says() {
        let rig = rig(0);
        let held = launch(1, serde_json::json!({"key": 0, "worker": "1"}));
        let free = launch(2, serde_json::json!({}));

        let waiting = tokio::spawn({
            let (router, routing, held) = (rig.router.clone(), rig.routing.clone(), held.clone());
            async move { router.admit(&held, &routing).await }
        });
        rig.router.admit(&free, &rig.routing).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!waiting.is_finished(), "key 0 is below the floor of 1");

        let (worker, props) = rig.router.placement(&held);
        assert_eq!(worker, Some(WorkerId(1)));
        assert_eq!(props["from"], serde_json::json!("1"));
        assert_eq!(rig.router.placement(&free).0, None, "the router's default");
        waiting.abort();
    }

    // One thread, so each records its release as it is released.
    #[tokio::test]
    async fn waiting_launches_leave_in_key_order_when_room_opens() {
        let rig = rig(255);
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut waits = Vec::new();
        for (req, key) in [(1, 1), (2, 5), (3, 3)] {
            let (router, routing, order) = (rig.router.clone(), rig.routing.clone(), order.clone());
            waits.push(tokio::spawn(async move {
                router
                    .admit(&launch(req, serde_json::json!({ "key": key })), &routing)
                    .await
                    .unwrap();
                order.lock().unwrap().push(req);
            }));
            // Each is heard (the gate is 50 ms) before the next arrives.
            tokio::time::sleep(Duration::from_millis(60)).await;
        }
        assert!(order.lock().unwrap().is_empty(), "the cluster is saturated");
        rig.table.send_replace(table(0));
        for wait in waits {
            wait.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), [2, 3, 1]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_terminated_launch_is_refused_while_it_waits_and_cancelled_once_it_runs() {
        let mut rig = rig(0);
        let refused = launch(1, serde_json::json!({"refuse": "not today"}));
        match rig.router.admit(&refused, &rig.routing).await {
            Err(AdmitReject::Refused(why)) => assert_eq!(why, "not today"),
            other => panic!("refused while it waits, got {other:?}"),
        }

        let stopped = launch(2, serde_json::json!({"stop": true}));
        rig.router.admit(&stopped, &rig.routing).await.unwrap();
        rig.router.dispatched(&stopped, WorkerId(1));
        let cancelled = tokio::time::timeout(Duration::from_secs(5), rig.cancelled.recv())
            .await
            .expect("the gateway is told to cancel it");
        assert_eq!(cancelled, Some(ReqId(2)));
    }
}
