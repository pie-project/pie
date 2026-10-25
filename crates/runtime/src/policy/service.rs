//! The worker policy on its own thread, with its own record of processes built
//! from the same events as its edges.

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::str::FromStr;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use client_api::Props;
use client_api::props::{PROMPT_BLOCK_BYTES, PromptIndex, TOKENS_PER_BYTE};

use super::program::{self, Commit, Fault, LoadError, Plane, Program, Retention, Table};
use super::retention;
use super::schedule;
use super::tokens;
use super::wit::{self, Edge, Subject, SubjectKind, Value};
use super::{By, Event, Outcome};
use crate::inferlet::process::{self, ProcessId};
use crate::store::kv::CachedWorkingSet;
use crate::store::registry as store_registry;

/// The worker as its own engine program sees it.
const WORKER: &str = "self";

static EVENTS: OnceLock<Sender<Event>> = OnceLock::new();

/// A worker has at most one.
pub fn start(name: &str, component: &[u8]) -> Result<(), LoadError> {
    let (tx, rx) = channel();
    program::spawn(name, component, capability, "pie-policy", move |program| {
        run(program, rx)
    })?;
    // A second one's thread ends with its channel.
    if EVENTS.set(tx).is_err() {
        return Err(LoadError::Instantiate(
            "this worker already runs a worker policy".into(),
        ));
    }
    tracing::info!(policy = name, "worker policy started");
    Ok(())
}

/// Swaps the worker policy for `next` (name, component), or for none, as a
/// SIGHUP asks. It loads whole before anything changes; on an error the
/// running one goes on. The old one stops as a dead one does: its tables and
/// what it published are withdrawn. The new one starts with no state of its
/// own, the same policy too. Blocks while it loads.
pub fn reload(next: Option<(&str, &[u8])>) -> Result<(), String> {
    let next = next.map(|(name, component)| (name.to_string(), component.to_vec()));
    let Some(events) = EVENTS.get() else {
        // None ran from boot: this starts one.
        return match next {
            Some((name, component)) => start(&name, &component).map_err(|e| e.to_string()),
            None => Ok(()),
        };
    };
    let (answer, answered) = std::sync::mpsc::sync_channel(1);
    events
        .send(Event::Reload(next, answer))
        .map_err(|_| "the worker policy's thread has ended".to_string())?;
    answered
        .recv()
        .map_err(|_| "the worker policy's thread has ended".to_string())?
}

/// A stopped policy's tables revert to the configuration's, and what it
/// published is withdrawn.
fn withdraw() {
    schedule::install(None);
    retention::install(None);
    super::offload::install(None);
    super::withdraw_published();
}

pub(super) fn report(event: Event) {
    if let Some(events) = EVENTS.get() {
        let _ = events.send(event);
    }
}

fn capability(name: &str) -> bool {
    match name {
        // Pausing evicts to the host or disk tier: without one, preemption
        // and the floor hold pending work only.
        "schedule.admit" | "schedule.preempt" | "schedule.resume" | "schedule.floor"
        | "retention.order" => true,
        // Idle cache goes to the host in key order.
        "retention.tier" => crate::planner::planner().is_some_and(|p| p.can_pause()),
        // Passes go to a peer while one is linked; `cost` reads only the
        // rule's threshold.
        "offload.remote" => crate::offload::executors::reachable(),
        // The worker's own values reach the cluster plane with its state
        // report; a process's do not yet.
        "publish" => true,
        _ => false,
    }
}

/// What a program did, logged every minute, so a run whose policy never
/// acted shows it.
#[derive(Default)]
struct Activity {
    since_ms: u64,
    calls: u64,
    call_us: u64,
    discarded: u64,
    schedules: u64,
    retentions: u64,
    keyed: u64,
    terminated: u64,
    discards: u64,
}

impl Activity {
    fn log(&mut self, policy: &str, now: u64) {
        if now.saturating_sub(self.since_ms) < 60_000 {
            return;
        }
        self.since_ms = now;
        tracing::info!(
            policy,
            calls = self.calls,
            call_ms = self.call_us / 1000,
            discarded = self.discarded,
            schedules = self.schedules,
            retentions = self.retentions,
            keyed = self.keyed,
            terminated = self.terminated,
            discards = self.discards,
            "worker policy activity"
        );
    }
}

fn run(program: Program, events: Receiver<Event>) {
    let mut registry = Registry::new(STORES);
    let mut activity = Activity {
        since_ms: now_ms(),
        ..Activity::default()
    };
    // None once it died, or a reload named none, until a reload names one.
    let mut current = Some(program);
    loop {
        let first = match current.as_ref().and_then(Program::alarm) {
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
        let mut reloads = Vec::new();
        registry.exited.clear();
        for event in first.into_iter().chain(events.try_iter()) {
            match event {
                Event::Reload(next, answer) => reloads.push((next, answer)),
                event => edges.extend(registry.apply(event)),
            }
        }
        for (next, answer) in reloads {
            let loaded = match &next {
                Some((name, component)) => Program::load(name, component, &capability).map(Some),
                None => Ok(None),
            };
            match loaded {
                Ok(next) => {
                    tracing::info!(
                        old = current.as_ref().map(Program::name),
                        new = next.as_ref().map(Program::name),
                        "worker policy reloaded"
                    );
                    if current.is_some() {
                        withdraw();
                    }
                    current = next;
                    let _ = answer.send(Ok(()));
                }
                Err(error) => {
                    tracing::error!(%error, "worker policy not reloaded; the running one goes on");
                    let _ = answer.send(Err(error.to_string()));
                }
            }
        }
        let Some(program) = current.as_mut() else {
            continue;
        };
        let now = now_ms();
        if program.alarm().is_some_and(|at_ms| at_ms <= now) {
            edges.push(Edge::Timer);
        }
        // Some events only update what a call reads: no edge, no call.
        if edges.is_empty() {
            continue;
        }

        tracing::debug!(policy = program.name(), ?edges, "worker policy edges");
        let started = Instant::now();
        let result = program.react(Box::new(registry.snapshot()), edges, now);
        activity.calls += 1;
        activity.call_us += started.elapsed().as_micros() as u64;
        match result {
            Ok(commit) => {
                activity.schedules += u64::from(commit.schedule.is_some());
                if let Some(table) = &commit.retention {
                    activity.retentions += 1;
                    activity.keyed += table.steps.len() as u64;
                }
                activity.terminated += commit.terminate.len() as u64;
                activity.discards += commit.discard.len() as u64;
                tracing::debug!(
                    policy = program.name(),
                    schedule = commit.schedule.as_ref().map(|t| t.keys.len()),
                    floor = commit.schedule.as_ref().and_then(|t| t.floor),
                    retention = commit.retention.as_ref().map(|t| t.steps.len()),
                    "worker policy call"
                );
                enact(&mut registry, commit);
            }
            Err(Fault::Dead(error)) => {
                tracing::error!(policy = program.name(), %error, "worker policy is dead; its tables are withdrawn");
                withdraw();
                current = None;
                continue;
            }
            Err(fault) => {
                activity.discarded += 1;
                tracing::warn!(policy = program.name(), %fault, "worker policy call discarded");
            }
        }
        activity.log(program.name(), now);
    }
}

fn by_process(table: Table) -> schedule::Table {
    let keys = table
        .keys
        .into_iter()
        .filter_map(|(id, key)| Some((ProcessId::from_str(&id).ok()?, key)))
        .collect();
    schedule::Table::new(keys, table.floor)
}

fn by_owner(table: Retention) -> retention::Table {
    let steps = table
        .steps
        .into_iter()
        .filter_map(|(id, steps)| {
            let steps = steps
                .into_iter()
                .map(|(upto, key)| (upto.map(u64::from), key))
                .collect();
            Some((ProcessId::from_str(&id).ok()?, steps))
        })
        .collect();
    retention::Table::new(steps)
}

fn by_rule(offload: wit::Offload) -> super::offload::Rule {
    super::offload::Rule {
        mode: match offload.mode {
            wit::OffloadMode::Never => super::offload::Mode::Never,
            wit::OffloadMode::Cost => super::offload::Mode::Cost,
            wit::OffloadMode::Always => super::offload::Mode::Always,
        },
        min_tokens: offload.min_tokens,
        executors: offload.executors,
    }
}

fn enact(registry: &mut Registry, commit: Commit) {
    if let Some(table) = commit.schedule {
        schedule::install(Some(by_process(table)));
    }
    if let Some(table) = commit.offload {
        super::offload::install(Some(super::offload::Table {
            entries: table
                .entries
                .into_iter()
                .filter_map(|(id, rule)| Some((ProcessId::from_str(&id).ok()?, by_rule(rule))))
                .collect(),
            default: table.default.map(by_rule),
        }));
    }
    if let Some(table) = commit.retention {
        retention::install(Some(by_owner(table)));
    }
    for (id, reason) in commit.terminate {
        if let Ok(pid) = ProcessId::from_str(&id) {
            registry.terminated.insert(pid);
            process::terminate_by(pid, Err(reason), By::Policy);
        }
    }
    for id in commit.discard {
        if let Ok(owner) = ProcessId::from_str(&id) {
            (registry.cache.discard)(owner);
        }
    }
    for (of, name, value) in commit.publish {
        let of = match of {
            Subject::Worker(_) => "worker".to_string(),
            Subject::Process(id) => format!("process:{id}"),
            _ => continue,
        };
        super::publish(of, name, value.and_then(|v| wit_json(&v)));
    }
}

#[derive(Clone, Copy)]
struct Cache {
    owns: fn(ProcessId) -> bool,
    list: fn() -> (Vec<CachedWorkingSet>, u32),
    discard: fn(ProcessId),
    memory: fn() -> (u64, u64),
    swap: fn() -> (u64, u64),
}

const STORES: Cache = Cache {
    owns: |owner| {
        store_registry::all_for_model(0).iter().any(|stores| {
            store_registry::with_kv_lock(&stores.kv, "policy-cache", |kv| kv.owns_cache(owner))
        })
    },
    list: || {
        let stores = store_registry::all_for_model(0);
        let page_size = stores.first().map_or(1, |s| s.kv_page_size);
        let mut sets: HashMap<ProcessId, CachedWorkingSet> = HashMap::new();
        for stores in &stores {
            let listed = store_registry::with_kv_lock(&stores.kv, "policy-cache", |kv| {
                kv.cached_working_sets()
            });
            for set in listed {
                sets.entry(set.owner)
                    .and_modify(|merged| {
                        merged.entries += set.entries;
                        merged.pages = merged.pages.max(set.pages);
                        merged.last_use = merged.last_use.max(set.last_use);
                    })
                    .or_insert(set);
            }
        }
        (sets.into_values().collect(), page_size)
    },
    memory: || {
        store_registry::all_for_model(0)
            .iter()
            .map(|stores| {
                store_registry::with_kv_lock(&stores.kv, "policy-memory", |kv| {
                    (u64::from(kv.capacity_pages()), kv.available_pages() as u64)
                })
            })
            .fold((0, 0), |(c, a), (cc, aa)| (c + cc, a + aa))
    },
    swap: || {
        store_registry::all_for_model(0)
            .iter()
            .map(|stores| {
                store_registry::with_kv_lock(&stores.kv, "policy-memory", |kv| {
                    let capacity = kv.host_swap_capacity() + kv.disk_swap_capacity();
                    let available = kv.host_swap_available() + kv.disk_swap_available();
                    (u64::from(capacity), available as u64)
                })
            })
            .fold((0, 0), |(c, a), (cc, aa)| (c + cc, a + aa))
    },
    discard: |owner| {
        let mut freed = 0;
        for stores in store_registry::all_for_model(0) {
            freed += store_registry::with_kv_lock(&stores.kv, "policy-discard", |kv| {
                kv.discard_cache(owner, By::Policy)
            });
            freed += stores.rs.lock().unwrap().discard_cache(owner);
        }
        if freed > 0 {
            for planner in crate::planner::planners() {
                planner.pages_freed();
            }
        }
    },
};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Pending,
    Running,
    Paused,
}

impl State {
    fn name(self) -> &'static str {
        match self {
            State::Pending => "pending",
            State::Running => "running",
            State::Paused => "paused",
        }
    }
}

#[derive(Debug, Clone)]
struct Proc {
    arrival_ms: u64,
    state: State,
    since_ms: u64,
    service_ms: u64,
    waited_ms: u64,
    restarts: u64,
    cached_tokens: Option<u64>,
    tenant: String,
    /// Scoped by tenant.
    session: Option<String>,
    /// Shared with every call's view: a `prompt` can be long.
    props: Arc<HashMap<String, Value>>,
    prompt: Option<Arc<[u64]>>,
}

impl Proc {
    /// Running is service; pending or paused is waiting.
    fn settle(&mut self, now_ms: u64) {
        let spent = now_ms.saturating_sub(self.since_ms);
        if self.state == State::Running {
            self.service_ms += spent;
        } else {
            self.waited_ms += spent;
        }
        self.since_ms = now_ms;
    }

    fn times(&self, now_ms: u64, live: bool) -> (u64, u64) {
        let open = if live {
            now_ms.saturating_sub(self.since_ms)
        } else {
            0
        };
        match self.state {
            State::Running => (self.service_ms + open, self.waited_ms),
            _ => (self.service_ms, self.waited_ms + open),
        }
    }
}

fn wit_json(value: &Value) -> Option<serde_json::Value> {
    use serde_json::Value as Json;
    Some(match value {
        Value::Boolean(b) => Json::Bool(*b),
        Value::Int(i) => Json::from(*i),
        Value::Float(f) => serde_json::Number::from_f64(*f).map(Json::Number)?,
        Value::Text(s) => Json::String(s.clone()),
        Value::Json(text) => serde_json::from_str(text).ok()?,
    })
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

#[derive(Debug, Clone)]
struct Retained {
    tenant: String,
    session: Option<String>,
}

struct Registry {
    /// Ordered, so `candidates` is stable.
    procs: BTreeMap<ProcessId, Proc>,
    /// Exited in the events a call is about: it can still read them.
    exited: HashMap<ProcessId, Proc>,
    terminated: HashSet<ProcessId>,
    retained: HashMap<ProcessId, Retained>,
    sessions: HashMap<String, Props>,
    cache: Cache,
    prompts: HashMap<ProcessId, (Arc<[u64]>, usize)>,
    /// `cached-tokens` before a process has run.
    held: Arc<PromptIndex>,
}

impl Registry {
    fn new(cache: Cache) -> Self {
        Self {
            procs: BTreeMap::new(),
            exited: HashMap::new(),
            terminated: HashSet::new(),
            retained: HashMap::new(),
            sessions: HashMap::new(),
            cache,
            prompts: HashMap::new(),
            held: Arc::new(PromptIndex::default()),
        }
    }

    fn apply(&mut self, event: Event) -> Vec<Edge> {
        let edge = match event {
            Event::Arrived(pid, arrival) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    // A restart: the same process, waiting again.
                    proc.settle(now_ms());
                    proc.state = State::Pending;
                    proc.restarts += 1;
                    return Vec::new();
                }
                tracing::debug!(
                    process = %id,
                    props = ?arrival.props.keys().collect::<Vec<_>>(),
                    "worker policy: a process arrived"
                );
                let now = now_ms();
                let prompt: Option<Arc<[u64]>> = client_api::props::prompt_of(arrival.props.iter())
                    .map(|text| client_api::props::prompt_blocks(text).into());
                if let Some(blocks) = &prompt {
                    self.prompts.insert(id, (blocks.clone(), 0));
                }
                self.procs.insert(
                    id,
                    Proc {
                        arrival_ms: now,
                        state: State::Pending,
                        since_ms: now,
                        service_ms: 0,
                        waited_ms: 0,
                        restarts: 0,
                        cached_tokens: None,
                        session: arrival
                            .session
                            .map(|session| format!("{}/{session}", arrival.tenant)),
                        tenant: arrival.tenant,
                        props: Arc::new(arrival.props.into_iter().filter_map(prop_value).collect()),
                        prompt,
                    },
                );
                Some(Edge::Arrived(id.to_string()))
            }
            Event::PropsChanged(pid, props, names) => {
                let id = schedule::plex_id(pid);
                self.procs.get_mut(&id).map(|proc| {
                    proc.props = Arc::new(props.into_iter().filter_map(prop_value).collect());
                    Edge::PropChanged(wit::Change {
                        subject: Subject::Process(id.to_string()),
                        names,
                    })
                })
            }
            Event::Cached(pid, tokens) => {
                let id = schedule::plex_id(pid);
                tracing::debug!(process = %id, tokens, "worker policy: cached tokens looked up");
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.cached_tokens = Some(proc.cached_tokens.unwrap_or(0).max(tokens));
                }
                None
            }
            Event::Started(pid) => self
                .set(pid, State::Running)
                .map(|id| Edge::Started(id.to_string())),
            Event::Paused(pid) => self
                .set(pid, State::Paused)
                .map(|id| Edge::Paused(id.to_string())),
            Event::Resumed(pid) => {
                self.set(pid, State::Running);
                None
            }
            Event::Offloaded(pid, worker, tokens) => {
                let id = schedule::plex_id(pid);
                self.procs.contains_key(&id).then(|| {
                    Edge::Offloaded(wit::OffloadEvent {
                        process: id.to_string(),
                        worker,
                        tokens,
                    })
                })
            }
            Event::OffloadFailed(pid, reason) => {
                let id = schedule::plex_id(pid);
                tracing::debug!(process = %id, %reason, "worker policy: a pass meant for a peer ran here");
                self.procs
                    .contains_key(&id)
                    .then(|| Edge::OffloadFailed(id.to_string()))
            }
            Event::Exited(pid, outcome) => {
                let id = schedule::plex_id(pid);
                self.terminated.remove(&id);
                let Some(mut proc) = self.procs.remove(&id) else {
                    return Vec::new();
                };
                proc.settle(now_ms());
                tracing::debug!(
                    process = %id,
                    counts = ?tokens::get(id),
                    "worker policy: a process exited"
                );
                self.exited.insert(id, proc.clone());
                let exited = Edge::Exited(wit::Exit {
                    process: id.to_string(),
                    outcome: match outcome {
                        Outcome::Completed => wit::Outcome::Completed,
                        Outcome::Failed => wit::Outcome::Failed,
                        Outcome::Cancelled => wit::Outcome::Cancelled,
                    },
                });
                if !(self.cache.owns)(id) {
                    self.prompts.remove(&id);
                    return vec![exited];
                }
                // What it left in the cache outlives it.
                self.retained.insert(
                    id,
                    Retained {
                        tenant: proc.tenant,
                        session: proc.session,
                    },
                );
                return vec![exited, Edge::Retained(id.to_string())];
            }
            Event::SessionProps(session, changes) => {
                let current = self.sessions.remove(&session).unwrap_or_default();
                let (merged, names) = match client_api::props::merge(&current, changes) {
                    Ok(merged) => merged,
                    Err(why) => {
                        tracing::warn!(%session, %why, "worker policy: session props refused");
                        (current, Vec::new())
                    }
                };
                if !merged.is_empty() {
                    self.sessions.insert(session.clone(), merged);
                }
                (!names.is_empty()).then_some(Edge::PropChanged(wit::Change {
                    subject: Subject::Session(session),
                    names,
                }))
            }
            Event::SessionClosed(session) => {
                self.sessions.remove(&session);
                Some(Edge::Closed(session))
            }
            // The policy thread swaps programs before a round applies its
            // events; one here has none to swap.
            Event::Reload(_, answer) => {
                let _ = answer.send(Err("a reload reached the registry".into()));
                None
            }
            Event::Kept(owner, tokens) => {
                if let Some((blocks, held)) = self.prompts.get_mut(&owner) {
                    // Tokens a block makes: the owner's own count where it
                    // gave one, else a guess.
                    let per_block = tokens::get(owner)
                        .map(|counts| counts.prompt)
                        .filter(|&prompt| prompt > 0 && !blocks.is_empty())
                        .map_or(TOKENS_PER_BYTE * PROMPT_BLOCK_BYTES as f64, |prompt| {
                            prompt as f64 / blocks.len() as f64
                        });
                    let covered = blocks.len().min((tokens as f64 / per_block) as usize);
                    if covered > *held {
                        *held = covered;
                        Arc::make_mut(&mut self.held).put(
                            &owner.to_string(),
                            &blocks[..covered],
                            per_block,
                        );
                    }
                }
                None
            }
            Event::CacheEmptied(owner, by) => {
                Arc::make_mut(&mut self.held).remove(&owner.to_string());
                if self.procs.contains_key(&owner) {
                    if let Some((_, held)) = self.prompts.get_mut(&owner) {
                        *held = 0;
                    }
                } else {
                    self.prompts.remove(&owner);
                }
                // A live process's working-set ends with the process.
                self.retained.remove(&owner).map(|_| {
                    Edge::Discarded(wit::Transition {
                        subject: owner.to_string(),
                        by: wit_by(by),
                    })
                })
            }
        };
        edge.into_iter().collect()
    }

    fn set(&mut self, pid: ProcessId, state: State) -> Option<ProcessId> {
        let id = schedule::plex_id(pid);
        let proc = self.procs.get_mut(&id)?;
        proc.settle(now_ms());
        proc.state = state;
        Some(id)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            procs: self
                .procs
                .iter()
                .map(|(id, proc)| (id.to_string(), proc.clone()))
                .collect(),
            exited: self
                .exited
                .iter()
                .map(|(id, proc)| (id.to_string(), proc.clone()))
                .collect(),
            retained: self
                .retained
                .iter()
                .map(|(id, retained)| (id.to_string(), retained.clone()))
                .collect(),
            sessions: self.sessions.clone(),
            held: self.held.clone(),
            cache: self.cache,
            cached: OnceCell::new(),
        }
    }
}

fn wit_by(by: By) -> wit::By {
    match by {
        By::Policy => wit::By::Policy,
        By::Host => wit::By::Host,
        By::Client => wit::By::Client,
    }
}

struct Snapshot {
    procs: HashMap<String, Proc>,
    exited: HashMap<String, Proc>,
    retained: HashMap<String, Retained>,
    sessions: HashMap<String, Props>,
    held: Arc<PromptIndex>,
    cache: Cache,
    /// Read once, when the call first asks.
    cached: OnceCell<(HashMap<String, CachedWorkingSet>, u32)>,
}

impl Snapshot {
    fn count(&self, state: State) -> i64 {
        self.procs.values().filter(|p| p.state == state).count() as i64
    }

    fn session(&self, id: &str) -> Vec<(&String, &Proc)> {
        let mut procs: Vec<_> = self
            .procs
            .iter()
            .filter(|(_, proc)| proc.session.as_deref() == Some(id))
            .collect();
        procs.sort_unstable_by_key(|(pid, _)| *pid);
        procs
    }

    fn cached(&self) -> &(HashMap<String, CachedWorkingSet>, u32) {
        self.cached.get_or_init(|| {
            let (sets, page_size) = (self.cache.list)();
            let sets = sets
                .into_iter()
                .map(|set| (set.owner.to_string(), set))
                .collect();
            (sets, page_size)
        })
    }

    fn whose(&self, id: &str) -> (Option<&String>, Option<&String>) {
        match (self.procs.get(id), self.retained.get(id)) {
            (Some(proc), _) => (Some(&proc.tenant), proc.session.as_ref()),
            (None, Some(retained)) => (Some(&retained.tenant), retained.session.as_ref()),
            (None, None) => (None, None),
        }
    }
}

fn text(value: Option<&String>) -> Option<Value> {
    value.map(|v| Value::Text(v.clone()))
}

impl Plane for Snapshot {
    fn candidates(&self, kind: SubjectKind) -> Vec<String> {
        match kind {
            SubjectKind::Process => {
                let mut ids: Vec<String> = self.procs.keys().cloned().collect();
                ids.sort_unstable();
                ids
            }
            SubjectKind::Session => {
                let sessions: BTreeSet<&String> = self
                    .procs
                    .values()
                    .filter_map(|p| p.session.as_ref())
                    .chain(self.sessions.keys())
                    .collect();
                sessions.into_iter().cloned().collect()
            }
            SubjectKind::Worker => vec![WORKER.to_string()],
            // Cluster subjects: the worker plane sees none.
            SubjectKind::Group | SubjectKind::Link => Vec::new(),
            SubjectKind::WorkingSet => {
                let mut ids: Vec<String> = self.cached().0.keys().cloned().collect();
                ids.sort_unstable();
                ids
            }
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
                let counts = || ProcessId::from_str(id).ok().and_then(tokens::get);
                let int = |n: Option<u64>| n.map(|n| Value::Int(n as i64));
                match name {
                    "state" if live => Some(Value::Text(proc.state.name().into())),
                    "state" => None,
                    "prompt-tokens" => int(counts().map(|c| c.prompt)),
                    "generated-tokens" => int(counts().map(|c| c.generated)),
                    "arrival-ms" => Some(Value::Int(proc.arrival_ms as i64)),
                    "service-ms" => int(Some(proc.times(now_ms(), live).0)),
                    "waited-ms" => int(Some(proc.times(now_ms(), live).1)),
                    "first-token-ms" => int(ProcessId::from_str(id)
                        .ok()
                        .and_then(tokens::first_token_ms)),
                    "worker" => Some(Value::Text(WORKER.into())),
                    "restarts" => Some(Value::Int(proc.restarts as i64)),
                    // What the runtime mapped in once it has run; before,
                    // what the cache holds of its prompt text.
                    "cached-tokens" => int(proc.cached_tokens.or_else(|| {
                        proc.prompt
                            .as_ref()
                            .map(|prompt| self.held.held_tokens(prompt))
                    })),
                    "tenant" => Some(Value::Text(proc.tenant.clone())),
                    "session" => proc.session.clone().map(Value::Text),
                    _ => None,
                }
            }
            Subject::Session(id) => {
                let procs = self.session(id);
                let tenant = match procs.first() {
                    Some((_, first)) => first.tenant.clone(),
                    None if self.sessions.contains_key(id) => id
                        .split_once('/')
                        .map_or(id.as_str(), |(t, _)| t)
                        .to_string(),
                    None => return None,
                };
                match name {
                    "tenant" => Some(Value::Text(tenant)),
                    _ => None,
                }
            }
            Subject::WorkingSet(id) => {
                let (sets, page_size) = self.cached();
                let set = sets.get(id)?;
                let (tenant, session) = self.whose(id);
                match name {
                    // Its owner while the owner runs; a retained one has none.
                    "owner" if self.procs.contains_key(id) => Some(Value::Text(id.clone())),
                    "owner" => None,
                    "tenant" => text(tenant),
                    "session" => text(session),
                    "tokens" => Some(Value::Int((set.pages * u64::from(*page_size)) as i64)),
                    "last-use-ms" => {
                        let ago = set.last_use.elapsed().as_millis() as u64;
                        Some(Value::Int(now_ms().saturating_sub(ago) as i64))
                    }
                    _ => None,
                }
            }
            Subject::Worker(id) if id == WORKER => match name {
                "running" => Some(Value::Int(self.count(State::Running))),
                "waiting" => Some(Value::Int(self.count(State::Pending))),
                "paused" => Some(Value::Int(self.count(State::Paused))),
                "step-ms" => {
                    crate::scheduler::last_batch_us().map(|us| Value::Int(us.div_ceil(1000) as i64))
                }
                "batch-size" => Some(Value::Int(crate::scheduler::last_batch_size() as i64)),
                // Pages cached for later processes are reclaimable, so
                // they do not count as used, as in vLLM and SGLang.
                "capacity-tokens" | "used-tokens" => {
                    let (capacity, available) = (self.cache.memory)();
                    let (sets, page_size) = self.cached();
                    let pages = if name == "capacity-tokens" {
                        capacity
                    } else {
                        let cached: u64 = sets.values().map(|set| set.pages).sum();
                        capacity.saturating_sub(available).saturating_sub(cached)
                    };
                    Some(Value::Int((pages * u64::from(*page_size)) as i64))
                }
                // Where a pause or a preemption puts kv: pie cannot compute
                // it again, so this is what a pause costs.
                "swap-capacity-tokens" | "swap-used-tokens" => {
                    let (capacity, available) = (self.cache.swap)();
                    let (_, page_size) = self.cached();
                    let pages = if name == "swap-capacity-tokens" {
                        capacity
                    } else {
                        capacity.saturating_sub(available)
                    };
                    Some(Value::Int((pages * u64::from(*page_size)) as i64))
                }
                _ => None,
            },
            _ => None,
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

    fn discardable(&self, working_set: &str) -> bool {
        // A live one ends with its owner: terminate that instead.
        self.cached().0.contains_key(working_set) && !self.procs.contains_key(working_set)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::policy::Arrival;

    fn props(value: serde_json::Value) -> client_api::Props {
        serde_json::from_value(value).unwrap()
    }

    const NO_CACHE: Cache = Cache {
        owns: |_| false,
        list: || (Vec::new(), 16),
        discard: |_| {},
        memory: || (100, 40),
        swap: || (400, 300),
    };

    fn arrive(registry: &mut Registry, tenant: &str, session: Option<&str>) -> ProcessId {
        let pid = ProcessId::new_v4();
        registry.apply(Event::Arrived(
            pid,
            Arrival {
                tenant: tenant.into(),
                session: session.map(Into::into),
                props: Default::default(),
            },
        ));
        pid
    }

    fn show(lookup: Option<Value>) -> String {
        format!("{lookup:?}")
    }

    #[test]
    fn a_process_arrives_with_its_props_and_its_changes_are_edges() {
        let mut registry = Registry::new(NO_CACHE);
        let pid = ProcessId::new_v4();
        let id = pid.to_string();
        let process = Subject::Process(id.clone());
        let arrival = Arrival {
            tenant: "t".into(),
            session: None,
            props: props(json!({"priority": 1})),
        };
        let arrived = registry.apply(Event::Arrived(pid, arrival));
        assert!(matches!(arrived[..], [Edge::Arrived(ref p)] if *p == id));
        let read =
            |registry: &Registry, name| format!("{:?}", registry.snapshot().prop(&process, name));
        assert_eq!(
            read(&registry, "priority"),
            format!("{:?}", Some(Value::Int(1)))
        );

        let changed = registry.apply(Event::PropsChanged(
            pid,
            props(json!({"priority": 7})),
            vec!["priority".into()],
        ));
        let [Edge::PropChanged(change)] = &changed[..] else {
            panic!("a change is a prop-changed edge, got {changed:?}");
        };
        assert!(matches!(change.subject, Subject::Process(ref p) if *p == id));
        assert_eq!(change.names, ["priority"]);
        assert_eq!(
            read(&registry, "priority"),
            format!("{:?}", Some(Value::Int(7)))
        );
        assert_eq!(read(&registry, "tier"), format!("{:?}", None::<Value>));

        // A process the policy never heard of has no props to change.
        let stranger = Event::PropsChanged(
            ProcessId::new_v4(),
            props(json!({"a": 1})),
            vec!["a".into()],
        );
        assert!(registry.apply(stranger).is_empty());
    }

    #[test]
    fn a_session_is_scoped_by_its_tenant() {
        let mut registry = Registry::new(NO_CACHE);
        let a1 = arrive(&mut registry, "a", Some("chat")).to_string();
        arrive(&mut registry, "a", Some("chat"));
        let b1 = arrive(&mut registry, "b", Some("chat")).to_string();
        let lone = arrive(&mut registry, "a", None).to_string();
        let plane = registry.snapshot();

        assert_eq!(plane.candidates(SubjectKind::Session), ["a/chat", "b/chat"]);
        let chat = Subject::Session("a/chat".into());
        assert_eq!(
            show(plane.attr(&chat, "tenant")),
            show(Some(Value::Text("a".into())))
        );
        assert_eq!(show(plane.prop(&chat, "anything")), show(None));
        let gone = Subject::Session("c/chat".into());
        assert_eq!(show(plane.attr(&gone, "tenant")), show(None));

        let of = |id: &String, name| show(plane.attr(&Subject::Process(id.clone()), name));
        assert_eq!(of(&a1, "session"), show(Some(Value::Text("a/chat".into()))));
        assert_eq!(of(&b1, "tenant"), show(Some(Value::Text("b".into()))));
        assert_eq!(of(&lone, "session"), show(None));
    }

    static CACHED: std::sync::Mutex<Vec<(ProcessId, u64)>> = std::sync::Mutex::new(Vec::new());

    const TEST_CACHE: Cache = Cache {
        owns: |owner| CACHED.lock().unwrap().iter().any(|(o, _)| *o == owner),
        list: || {
            let now = crate::rt::Instant::now();
            let sets = CACHED
                .lock()
                .unwrap()
                .iter()
                .map(|&(owner, pages)| CachedWorkingSet {
                    owner,
                    entries: pages as usize,
                    pages,
                    last_use: now,
                })
                .collect();
            (sets, 16)
        },
        discard: |_| {},
        memory: || (100, 40),
        swap: || (400, 300),
    };

    #[test]
    fn a_session_s_props_change_between_its_processes_and_it_closes() {
        let mut registry = Registry::new(TEST_CACHE);
        let session = || "t/chat".to_string();
        let edges = registry.apply(Event::SessionProps(
            session(),
            props(json!({"tool-eta-ms": 5000, "tool": "grep"})),
        ));
        let [Edge::PropChanged(change)] = &edges[..] else {
            panic!("{edges:?}");
        };
        assert!(matches!(&change.subject, Subject::Session(s) if *s == session()));
        assert_eq!(change.names, ["tool", "tool-eta-ms"]);

        let plane = registry.snapshot();
        assert_eq!(plane.candidates(SubjectKind::Session), [session()]);
        let read = |plane: &Snapshot, name| show(plane.prop(&Subject::Session(session()), name));
        assert_eq!(read(&plane, "tool-eta-ms"), show(Some(Value::Int(5000))));
        assert_eq!(read(&plane, "other"), show(None));
        assert_eq!(
            show(plane.attr(&Subject::Session(session()), "tenant")),
            show(Some(Value::Text("t".into())))
        );

        let same = registry.apply(Event::SessionProps(
            session(),
            props(json!({"tool": "grep"})),
        ));
        assert!(same.is_empty(), "nothing changed");
        let edges = registry.apply(Event::SessionProps(session(), props(json!({"tool": null}))));
        assert!(matches!(&edges[..], [Edge::PropChanged(c)] if c.names == ["tool"]));

        let edges = registry.apply(Event::SessionClosed(session()));
        assert!(matches!(&edges[..], [Edge::Closed(s)] if *s == session()));
        assert_eq!(read(&registry.snapshot(), "tool-eta-ms"), show(None));
    }

    #[test]
    fn a_process_counts_its_own_tokens_and_the_call_about_its_exit_still_reads_them() {
        let mut registry = Registry::new(TEST_CACHE);
        let id = arrive(&mut registry, "t", None);
        let of = |registry: &Registry, name| {
            show(
                registry
                    .snapshot()
                    .attr(&Subject::Process(id.to_string()), name),
            )
        };
        let int = |n| show(Some(Value::Int(n)));
        assert_eq!(of(&registry, "prompt-tokens"), show(None));
        tokens::set(
            id,
            tokens::Counts {
                prompt: 100,
                generated: 7,
            },
        );
        assert_eq!(of(&registry, "prompt-tokens"), int(100));
        assert_eq!(of(&registry, "generated-tokens"), int(7));

        tokens::exit(id);
        registry.apply(Event::Exited(id, Outcome::Completed));
        assert_eq!(of(&registry, "generated-tokens"), int(7));
        assert_eq!(of(&registry, "tenant"), show(Some(Value::Text("t".into()))));
        assert_eq!(of(&registry, "state"), show(None), "not live");
        let plane = registry.snapshot();
        assert!(plane.candidates(SubjectKind::Process).is_empty());
        assert!(!plane.terminable(&id.to_string()));

        registry.exited.clear();
        assert_eq!(
            of(&registry, "tenant"),
            show(None),
            "the next call does not"
        );
    }

    #[test]
    fn a_pending_process_reads_what_the_cache_holds_of_its_prompt() {
        let mut registry = Registry::new(NO_CACHE);
        let text = "word ".repeat(200);
        let with_prompt = |registry: &mut Registry, prompt: &str| {
            let pid = ProcessId::new_v4();
            registry.apply(Event::Arrived(
                pid,
                Arrival {
                    tenant: "t".into(),
                    session: None,
                    props: props(json!({ "prompt": prompt })),
                },
            ));
            schedule::plex_id(pid)
        };
        let cached = |registry: &Registry, id: ProcessId| {
            show(
                registry
                    .snapshot()
                    .attr(&Subject::Process(id.to_string()), "cached-tokens"),
            )
        };
        let tokens = |n: i64| show(Some(Value::Int(n)));
        let first = with_prompt(&mut registry, &text);
        let second = with_prompt(&mut registry, &format!("{text}and more"));
        assert_eq!(cached(&registry, second), tokens(0), "nothing held yet");
        // Ten 64-byte blocks' worth, at a guessed 16 tokens a block.
        registry.apply(Event::Kept(first, 160));
        assert_eq!(cached(&registry, second), tokens(160));
        registry.apply(Event::Cached(second, 96));
        assert_eq!(
            cached(&registry, second),
            tokens(96),
            "once it ran, what it got"
        );
        let third = with_prompt(&mut registry, &text);
        registry.apply(Event::CacheEmptied(first, By::Host));
        assert_eq!(cached(&registry, third), tokens(0), "gone with the cache");
        let bare = arrive(&mut registry, "t", None);
        assert_eq!(
            cached(&registry, bare),
            show(None),
            "no prompt, no estimate"
        );
    }

    #[test]
    fn a_working_set_is_retained_when_its_owner_exits_until_its_last_entry_goes() {
        let mut registry = Registry::new(TEST_CACHE);
        let gone = arrive(&mut registry, "t", Some("chat"));
        let live = arrive(&mut registry, "t", None);
        let uncached = arrive(&mut registry, "t", None);
        registry.apply(Event::Cached(gone, 32));
        registry.apply(Event::Cached(gone, 0));
        CACHED.lock().unwrap().extend([(gone, 4), (live, 2)]);

        let of = |registry: &Registry, id: ProcessId| {
            show(
                registry
                    .snapshot()
                    .attr(&Subject::Process(id.to_string()), "cached-tokens"),
            )
        };
        assert_eq!(
            of(&registry, gone),
            show(Some(Value::Int(32))),
            "its longest hit"
        );
        assert_eq!(of(&registry, live), show(None), "it never looked");

        let edges = registry.apply(Event::Exited(gone, Outcome::Completed));
        assert!(
            matches!(&edges[..], [Edge::Exited(_), Edge::Retained(ws)] if *ws == gone.to_string()),
            "{edges:?}"
        );
        let edges = registry.apply(Event::Exited(uncached, Outcome::Completed));
        assert!(
            matches!(&edges[..], [Edge::Exited(_)]),
            "nothing of it outlives it"
        );

        let plane = registry.snapshot();
        let mut ids = vec![gone.to_string(), live.to_string()];
        ids.sort();
        assert_eq!(plane.candidates(SubjectKind::WorkingSet), ids);
        let ws = |id: ProcessId, name| show(plane.attr(&Subject::WorkingSet(id.to_string()), name));
        assert_eq!(ws(gone, "owner"), show(None));
        assert_eq!(ws(live, "owner"), show(Some(Value::Text(live.to_string()))));
        assert_eq!(
            ws(gone, "session"),
            show(Some(Value::Text("t/chat".into())))
        );
        assert_eq!(ws(gone, "tokens"), show(Some(Value::Int(64))));
        let worker = |name| show(plane.attr(&Subject::Worker(WORKER.to_string()), name));
        assert_eq!(worker("capacity-tokens"), show(Some(Value::Int(1600))));
        // 60 of 100 pages are taken, 6 of them by what is cached.
        assert_eq!(worker("used-tokens"), show(Some(Value::Int(864))));
        // 100 of 400 pages below the device hold suspended kv.
        assert_eq!(worker("swap-capacity-tokens"), show(Some(Value::Int(6400))));
        assert_eq!(worker("swap-used-tokens"), show(Some(Value::Int(1600))));
        assert!(plane.discardable(&gone.to_string()));
        assert!(!plane.discardable(&live.to_string()), "live");
        assert!(!plane.discardable(&uncached.to_string()), "not cached");

        let edges = registry.apply(Event::CacheEmptied(live, By::Client));
        assert!(edges.is_empty(), "a live working-set ends with its owner");
        let edges = registry.apply(Event::CacheEmptied(gone, By::Host));
        let [Edge::Discarded(discarded)] = &edges[..] else {
            panic!("the last entry gone discards it, got {edges:?}");
        };
        assert_eq!(discarded.subject, gone.to_string());
        assert!(matches!(discarded.by, wit::By::Host));
        CACHED.lock().unwrap().clear();
    }

    #[test]
    fn a_client_prop_becomes_the_contracts_value() {
        let props: Vec<_> = [
            ("flag", json!(true)),
            ("priority", json!(3)),
            ("weight", json!(0.5)),
            ("tier", json!("gold")),
            ("tags", json!(["a", "b"])),
            ("gone", json!(null)),
        ]
        .into_iter()
        .filter_map(|(name, value)| prop_value((name.to_string(), value)))
        .collect();
        // The bound `value` has no `PartialEq`; its `Debug` names each case.
        let expected = [
            ("flag", Value::Boolean(true)),
            ("priority", Value::Int(3)),
            ("weight", Value::Float(0.5)),
            ("tier", Value::Text("gold".into())),
            ("tags", Value::Json(r#"["a","b"]"#.into())),
        ]
        .map(|(name, value)| (name.to_string(), value));
        assert_eq!(format!("{props:?}"), format!("{expected:?}"));
    }
}
