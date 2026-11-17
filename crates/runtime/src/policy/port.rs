//! The worker's PLEX 1.1 port: the runtime's reports into the PLEX host
//! (`plex-host`), and the host's committed rows back out.
//!
//! Three threads:
//! - any runtime thread reports an [`Event`] over a channel, as it did to the
//!   1.0 policy thread (every call site goes through [`super::report`]);
//! - this port thread turns each event into a host report, keeps the record
//!   the host cannot see (cached working sets, resting handles, tier
//!   pressure), sends the worker's counters when a round waits for them, and
//!   takes the rows a round committed;
//! - the host's own thread calls the policies, off every runtime thread.
//!
//! The worker is `self` to its policies. A process's PLEX id is the id its
//! client knows (`schedule::plex_id`): one restarted after a KV reclaim keeps
//! it, and is reported paused by the host, then started again.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use client_api::Props;
use plex_host::host::{Config, Counter, Delta, Host};
use plex_host::runtime::wit::Tier as WitTier;
use plex_host::tables::{Subject, Value};

use super::pressure::{self, Reading, Tier};
use super::{By, Event, Outcome, retention, schedule};
use crate::inferlet::process::{self, ProcessId};
use crate::store::kv::CachedWorkingSet;
use crate::store::registry as store_registry;

/// The worker, as its own policies see it.
const WORKER: &str = "self";

/// How often the port looks at the host while no report comes: the host
/// commits off this thread, so the port reads its version this often, as an
/// engine reads it once a step.
const LOOK: Duration = Duration::from_millis(2);

/// The columns the worker enacts (`Config.supported`).
const SUPPORTED: &[&str] = &[
    plex_host::tables::SCHEDULE_PRIORITY,
    plex_host::tables::SCHEDULE_HOLD,
    plex_host::tables::RETENTION_PRIORITY,
    plex_host::tables::RETENTION_TIERS,
];

static EVENTS: OnceLock<Sender<Event>> = OnceLock::new();

/// Errors starting the port.
#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LoadError {}

fn host() -> Host {
    Host::new(Config {
        supported: SUPPORTED.iter().map(|c| c.to_string()).collect(),
        ..Config::default()
    })
}

/// A policy's name in the host (the writer of its notes): a built-in's
/// name, or a component file's stem.
fn policy_name(spec: &str) -> String {
    match spec.strip_suffix(".wasm") {
        Some(path) => std::path::Path::new(path)
            .file_name()
            .map_or(spec.to_string(), |n| n.to_string_lossy().into_owned()),
        None => spec.to_string(),
    }
}

/// Starts the port with the policy `component` loaded as `name`. A worker
/// has one port.
pub fn start(name: &str, component: &[u8]) -> Result<(), LoadError> {
    let name = &policy_name(name);
    let host = host();
    let imports = host.load(name, component).map_err(LoadError)?;
    tracing::info!(policy = name, ?imports, "worker policy loaded (PLEX 1.1)");
    spawn(host, Some((name.to_string(), component.to_vec())))
}

/// Starts the port with no policy: the host keeps its record for one a
/// reload loads, and the default responder answers tier pressure. One
/// already running stands.
pub fn start_default() -> Result<(), LoadError> {
    if EVENTS.get().is_some() {
        return Ok(());
    }
    spawn(host(), None)
}

fn spawn(mut host: Host, loaded: Option<(String, Vec<u8>)>) -> Result<(), LoadError> {
    let (tx, rx) = channel();
    if EVENTS.set(tx).is_err() {
        return Err(LoadError("this worker already runs a PLEX port".into()));
    }
    host.start();
    std::thread::Builder::new()
        .name("pie-plex".into())
        .spawn(move || run(host, loaded, rx))
        .map_err(|e| LoadError(format!("spawning the port thread: {e}")))?;
    Ok(())
}

/// Swaps the worker policy for `next` (name, component), or for none, as a
/// SIGHUP asks. The old one is unloaded (its rows withdrawn) and the new one
/// loaded; if it does not load, the old one is loaded again. Blocks while it
/// loads.
pub fn reload(next: Option<(&str, &[u8])>) -> Result<(), String> {
    let next = next.map(|(name, component)| (policy_name(name), component.to_vec()));
    let Some(events) = EVENTS.get() else {
        return match next {
            Some((name, component)) => start(&name, &component).map_err(|e| e.to_string()),
            None => Ok(()),
        };
    };
    let (answer, answered) = std::sync::mpsc::sync_channel(1);
    events
        .send(Event::Reload(next, answer))
        .map_err(|_| "the worker's PLEX port has ended".to_string())?;
    answered
        .recv()
        .map_err(|_| "the worker's PLEX port has ended".to_string())?
}

pub(super) fn report(event: Event) {
    if let Some(events) = EVENTS.get() {
        let _ = events.send(event);
    }
}

/// Per tier, the episode in which the default responder finds nothing left
/// to discard: said once as it starts and once as it ends.
static NOTHING_LEFT: std::sync::Mutex<[process::WaitEpisode; 2]> =
    std::sync::Mutex::new([process::WaitEpisode::new(), process::WaitEpisode::new()]);

/// Ends `tier`'s nothing-left episode, if one is under way, saying how long
/// it ran and why it ended.
fn nothing_left_ends(tier: Tier, why: &'static str) {
    if let Some(lasted) = NOTHING_LEFT.lock().unwrap()[tier.index()].end() {
        tracing::info!(
            tier = tier.name(),
            lasted_ms = lasted.as_millis() as u64,
            why,
            "tier pressure: the waiters no longer wait on the governor alone"
        );
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn by_name(by: By) -> &'static str {
    match by {
        By::Policy => "policy",
        By::Host => "host",
        By::Client => "client",
    }
}

/// A client's prop as a hint value: lists and objects travel as JSON text.
fn hint_value(value: &serde_json::Value) -> Option<Value> {
    use serde_json::Value as Json;
    Some(match value {
        Json::Null => return None,
        Json::Bool(b) => Value::Boolean(*b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64()?),
        },
        Json::String(s) => Value::Text(s.clone()),
        json @ (Json::Array(_) | Json::Object(_)) => Value::Text(json.to_string()),
    })
}

fn hints(props: &Props) -> Vec<(String, Value)> {
    props
        .iter()
        .filter_map(|(name, value)| Some((name.clone(), hint_value(value)?)))
        .collect()
}

/// What the stores hold, read on the port thread.
#[derive(Clone, Copy)]
struct Cache {
    owns: fn(ProcessId) -> bool,
    list: fn() -> (Vec<CachedWorkingSet>, u32),
    discard: fn(ProcessId),
    /// Pages: device capacity and free; host capacity and free; disk
    /// capacity and free.
    rooms: fn() -> [(u64, u64); 3],
    resident: fn(ProcessId) -> Option<bool>,
    scale: fn() -> Option<pressure::Scale>,
}

const STORES: Cache = Cache {
    owns: |owner| {
        store_registry::all_for_model(0).iter().any(|stores| {
            store_registry::with_kv_lock(&stores.kv, "plex-cache", |kv| kv.owns_cache(owner))
        })
    },
    list: || {
        let stores = store_registry::all_for_model(0);
        let page_size = stores.first().map_or(1, |s| s.kv_page_size);
        let mut sets: HashMap<ProcessId, CachedWorkingSet> = HashMap::new();
        for stores in &stores {
            let listed = store_registry::with_kv_lock(&stores.kv, "plex-cache", |kv| {
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
    rooms: || {
        let mut rooms = [(0u64, 0u64); 3];
        for stores in store_registry::all_for_model(0) {
            store_registry::with_kv_lock(&stores.kv, "plex-rooms", |kv| {
                let capacity = u64::from(kv.capacity_pages());
                rooms[0].0 += capacity;
                rooms[0].1 += capacity - u64::from(kv.pages_out());
                rooms[1].0 += u64::from(kv.host_swap_capacity());
                rooms[1].1 += kv.host_swap_available() as u64;
                rooms[2].0 += u64::from(kv.disk_swap_capacity());
                rooms[2].1 += kv.disk_swap_available() as u64;
            });
        }
        rooms
    },
    resident: |holder| crate::planner::planner_of(holder)?.holder_resident(holder),
    scale: pressure::common_scale,
    discard: |owner| {
        let mut freed = 0;
        for stores in store_registry::all_for_model(0) {
            freed += store_registry::with_kv_lock(&stores.kv, "plex-discard", |kv| {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Waiting,
    Running,
    Paused,
}

/// What the port keeps of a live process.
#[derive(Debug, Clone)]
struct Proc {
    /// The runtime's process (a restart's is new).
    pid: ProcessId,
    state: State,
    session: Option<String>,
    /// What its fires mapped in from the cache, once one has looked.
    cached_tokens: Option<u64>,
    /// The rows its fires wrote before the first readout; known once it
    /// read one out.
    prompt_rows: u64,
    prompt_tokens: Option<u64>,
    /// The rows its fires read out.
    generated_tokens: u64,
    restarts: u64,
}

/// A handle at rest: a working set its client holds.
#[derive(Debug, Clone)]
struct Rest {
    resting: super::Resting,
    since_ms: u64,
}

/// The port's own record: what the host is told comes from it, and the
/// default tier-pressure responder reads it.
struct Port {
    host: Host,
    cache: Cache,
    procs: HashMap<ProcessId, Proc>,
    /// Exited processes' sessions, while their working sets last.
    retained: HashMap<ProcessId, Option<String>>,
    handles: HashMap<ProcessId, Rest>,
    sessions: HashMap<String, Props>,
    /// The working sets last put to the host: one gone from the stores
    /// without a report is evicted then.
    put: HashSet<ProcessId>,
    /// Where each tier's waiters stand, by [`Tier::index`].
    pressure: [Reading; 2],
    pressed: [bool; 2],
    /// Offloaded tokens by process, as `pie/offloaded-tokens`.
    offloaded: HashMap<ProcessId, u64>,
    /// The policy loaded, to load again if its successor does not load.
    loaded: Option<(String, Vec<u8>)>,
    version: u64,
    /// The schedule and retention columns as the rows left them.
    columns: Columns,
}

/// The worker's columns: each process's row (by its client's id), and the
/// column's default.
#[derive(Default)]
struct Columns {
    priority: HashMap<ProcessId, i64>,
    priority_default: Option<i64>,
    hold: HashMap<ProcessId, bool>,
    hold_default: Option<bool>,
    retention: HashMap<(ProcessId, Option<u32>), i64>,
    retention_default: Option<i64>,
}

/// A column's rows and default into `rows` and `default`; keys that name
/// no process this worker could hold are passed over.
fn apply<K, V, R: std::hash::Hash + Eq>(
    delta: plex_host::host::ColumnDelta<K, V>,
    rows: &mut HashMap<R, V>,
    default: &mut Option<V>,
    key: impl Fn(K) -> Option<R>,
) -> bool {
    let changed = !delta.rows.is_empty() || delta.default.is_some();
    for (k, row) in delta.rows {
        let Some(k) = key(k) else { continue };
        match row {
            Some(v) => {
                rows.insert(k, v);
            }
            None => {
                rows.remove(&k);
            }
        }
    }
    if let Some(d) = delta.default {
        *default = d;
    }
    changed
}

fn run(host: Host, loaded: Option<(String, Vec<u8>)>, events: Receiver<Event>) {
    let mut port = Port {
        host,
        cache: STORES,
        procs: HashMap::new(),
        retained: HashMap::new(),
        handles: HashMap::new(),
        sessions: HashMap::new(),
        put: HashSet::new(),
        pressure: [Reading::default(); 2],
        pressed: [false; 2],
        offloaded: HashMap::new(),
        loaded,
        version: 0,
        columns: Columns::default(),
    };
    loop {
        let first = match events.recv_timeout(LOOK) {
            Ok(event) => Some(event),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        for event in first.into_iter().chain(events.try_iter()) {
            match event {
                Event::Reload(next, answer) => {
                    let _ = answer.send(port.reload(next));
                }
                event => port.apply(event),
            }
        }
        port.pressure_round();
        if port.host.due() {
            port.put_worker();
            port.put_working_sets();
            port.host.tick(port.counters());
        }
        let version = port.host.version();
        if version != port.version {
            port.version = version;
            let delta = port.host.poll_delta();
            port.enact(delta);
        }
    }
}

impl Port {
    fn reload(&mut self, next: Option<(String, Vec<u8>)>) -> Result<(), String> {
        if let Some((name, _)) = &self.loaded {
            self.host.unload(name);
            // What it noted for the cluster goes with it.
            super::withdraw_published();
        }
        let Some((name, component)) = next else {
            tracing::info!("worker policy unloaded");
            self.loaded = None;
            return Ok(());
        };
        match self.host.load(&name, &component) {
            Ok(imports) => {
                tracing::info!(policy = %name, ?imports, "worker policy reloaded (PLEX 1.1)");
                self.loaded = Some((name, component));
                Ok(())
            }
            Err(error) => {
                tracing::error!(%error, "worker policy not reloaded; the running one goes on");
                if let Some((old, component)) = &self.loaded {
                    let _ = self.host.load(old, component);
                }
                Err(error)
            }
        }
    }

    fn apply(&mut self, event: Event) {
        let host = &self.host;
        match event {
            Event::Arrived(pid, arrival) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    // A restart after a reclaim: the same process, computed
                    // again, as a preemption by the host is.
                    proc.pid = pid;
                    proc.restarts += 1;
                    proc.state = State::Paused;
                    host.paused(&id.to_string(), "host");
                    host.attr(
                        Subject::Process(id.to_string()),
                        "pie/restarts",
                        Some(Value::Int(proc.restarts as i64)),
                    );
                    return;
                }
                let session = arrival
                    .session
                    .map(|session| format!("{}/{session}", arrival.tenant));
                let max_tokens = arrival
                    .props
                    .get(client_api::props::MAX_TOKENS)
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok());
                host.arrived(
                    &id.to_string(),
                    session.clone(),
                    Some(arrival.tenant),
                    Some(WORKER.to_string()),
                    None,
                    max_tokens,
                    hints(&arrival.props),
                    0,
                );
                self.procs.insert(
                    id,
                    Proc {
                        state: State::Waiting,
                        session,
                        pid,
                        cached_tokens: None,
                        prompt_rows: 0,
                        prompt_tokens: None,
                        generated_tokens: 0,
                        restarts: 0,
                    },
                );
            }
            Event::PropsChanged(pid, props, names) => {
                let id = schedule::plex_id(pid).to_string();
                for name in names {
                    let value = props.get(&name).and_then(hint_value);
                    host.hint(Subject::Process(id.clone()), &name, value);
                }
            }
            Event::Fired(pid, rows, readouts) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    if proc.prompt_tokens.is_none() {
                        proc.prompt_rows += rows;
                        if readouts > 0 {
                            proc.prompt_tokens = Some(proc.prompt_rows);
                        }
                    }
                    proc.generated_tokens += readouts;
                }
            }
            Event::Cached(pid, tokens) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.cached_tokens = Some(proc.cached_tokens.unwrap_or(0) + tokens);
                }
            }
            Event::Started(pid) | Event::Resumed(pid) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.state = State::Running;
                    host.started(&id.to_string());
                }
            }
            Event::Paused(pid) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.state = State::Paused;
                    // A held process was paused for its row; any other, for
                    // the host's own room.
                    let by = if schedule::held(id) { "policy" } else { "host" };
                    host.paused(&id.to_string(), by);
                }
            }
            Event::Exited(pid, outcome) => {
                let id = schedule::plex_id(pid);
                let Some(proc) = self.procs.remove(&id) else {
                    return;
                };
                let outcome = match outcome {
                    Outcome::Completed => "completed",
                    Outcome::Failed => "failed",
                    Outcome::Cancelled => "cancelled",
                };
                let count = |n: u64| u32::try_from(n).ok();
                host.exited(
                    &id.to_string(),
                    outcome,
                    proc.prompt_tokens.and_then(count),
                    proc.cached_tokens.and_then(count),
                    count(proc.generated_tokens).unwrap_or(u32::MAX),
                );
                self.offloaded.remove(&id);
                if (self.cache.owns)(id) {
                    // What it left in the cache outlives it.
                    self.retained.insert(id, proc.session);
                    self.put_working_sets();
                    self.host.retained(&id.to_string(), WORKER);
                }
            }
            Event::Offloaded(pid, worker, tokens) => {
                let id = schedule::plex_id(pid);
                let total = self.offloaded.entry(id).or_default();
                *total += u64::from(tokens);
                host.attr(
                    Subject::Process(id.to_string()),
                    "pie/offloaded-tokens",
                    Some(Value::Int(*total as i64)),
                );
                host.attr(
                    Subject::Process(id.to_string()),
                    "pie/offloaded-to",
                    Some(Value::Text(worker)),
                );
            }
            Event::OffloadFailed(pid, reason) => {
                let id = schedule::plex_id(pid);
                host.attr(
                    Subject::Process(id.to_string()),
                    "pie/offload-failed",
                    Some(Value::Text(reason)),
                );
            }
            Event::HandleRested(id, resting) => {
                let set = Subject::WorkingSet(id.to_string());
                host.attr(
                    set.clone(),
                    "pie/tenant",
                    Some(Value::Text(resting.tenant.clone())),
                );
                host.attr(
                    set,
                    "pie/folded-tokens",
                    Some(Value::Int(resting.folded as i64)),
                );
                for (name, value) in hints(&resting.props) {
                    host.hint(Subject::WorkingSet(id.to_string()), &name, Some(value));
                }
                self.handles.insert(
                    id,
                    Rest {
                        resting,
                        since_ms: now_ms(),
                    },
                );
                self.put_working_sets();
                self.host.retained(&id.to_string(), WORKER);
            }
            Event::HandleProps(id, props, names) => {
                if let Some(rest) = self.handles.get_mut(&id) {
                    for name in names {
                        let value = props.get(&name).and_then(hint_value);
                        host.hint(Subject::WorkingSet(id.to_string()), &name, value);
                    }
                    rest.resting.props = props;
                }
            }
            Event::HandleLent(id) => {
                // A process holds it now, out of the policy's reach.
                if self.handles.remove(&id).is_some() {
                    self.forget(id, By::Client);
                }
            }
            Event::HandleGone(id, by) => {
                if self.handles.remove(&id).is_some() {
                    self.forget(id, by);
                }
            }
            Event::SessionProps(session, changes) => {
                let current = self.sessions.remove(&session).unwrap_or_default();
                let (merged, names) = match client_api::props::merge(&current, changes) {
                    Ok(merged) => merged,
                    Err(why) => {
                        tracing::warn!(%session, %why, "session props refused");
                        (current, Vec::new())
                    }
                };
                for name in &names {
                    let value = merged.get(name).and_then(hint_value);
                    host.hint(Subject::Session(session.clone()), name, value);
                }
                if !merged.is_empty() {
                    self.sessions.insert(session, merged);
                }
            }
            Event::SessionClosed(session) => {
                self.sessions.remove(&session);
                host.closed(&session);
            }
            // The port thread answers a reload before it applies a round's
            // other events.
            Event::Reload(_, answer) => {
                let _ = answer.send(Err("a reload reached the port's record".into()));
            }
            Event::TierPressure(tier, reading) => {
                let at = tier.index();
                self.pressure[at] = reading;
                self.pressed[at] |= reading.short > 0;
                host.attr(
                    Subject::Worker(WORKER.to_string()),
                    &format!("pie/{}-band", tier.name()),
                    Some(Value::Text(reading.band.name().into())),
                );
            }
            Event::CacheEmptied(owner, by) => {
                // A live process's working set ends with the process.
                if self.retained.remove(&owner).is_some() || self.put.contains(&owner) {
                    self.forget(owner, by);
                }
            }
        }
    }

    /// A working set left every tier here.
    fn forget(&mut self, id: ProcessId, by: By) {
        self.put.remove(&id);
        self.host
            .evicted(&id.to_string(), WORKER, by_name(by), true);
    }

    /// The worker's rooms and load, as the host's `worker` record has them.
    fn put_worker(&self) {
        let page = u64::from((self.cache.list)().1.max(1));
        let [device, host, disk] = (self.cache.rooms)();
        let mut tiers = vec![(
            WitTier::Device,
            Some(device.0 * page),
            device.0.saturating_sub(device.1) * page,
        )];
        if host.0 > 0 {
            tiers.push((
                WitTier::Host,
                Some(host.0 * page),
                host.0.saturating_sub(host.1) * page,
            ));
        }
        if disk.0 > 0 {
            tiers.push((
                WitTier::Disk,
                Some(disk.0 * page),
                disk.0.saturating_sub(disk.1) * page,
            ));
        }
        let count = |state| self.procs.values().filter(|p| p.state == state).count() as u32;
        let step_ns = crate::scheduler::last_batch_us().map(|us| us * 1000);
        self.host.attr(
            Subject::Worker(WORKER.to_string()),
            "pie/batch-size",
            Some(Value::Int(crate::scheduler::last_batch_size() as i64)),
        );
        self.host.put_worker(
            WORKER,
            false,
            tiers,
            count(State::Running),
            count(State::Waiting) + count(State::Paused),
            step_ns,
        );
    }

    /// Every working set the stores hold, cached or resting, as the host's
    /// tree has them; one gone without a report is evicted now.
    fn put_working_sets(&mut self) {
        let (sets, page) = (self.cache.list)();
        let page = u64::from(page.max(1));
        let mut now: HashSet<ProcessId> = HashSet::new();
        for set in sets {
            let id = set.owner;
            let live = self.procs.contains_key(&id);
            let session = match self.procs.get(&id) {
                Some(proc) => proc.session.clone(),
                None => self.retained.get(&id).cloned().flatten(),
            };
            let tokens = u32::try_from(set.pages * page).unwrap_or(u32::MAX);
            let idle_ns = set.last_use.elapsed().as_nanos() as u64;
            self.host.put_working_set(
                &id.to_string(),
                None,
                live.then(|| id.to_string()),
                session,
                tokens,
                idle_ns,
                vec![(WORKER.to_string(), WitTier::Device, tokens)],
            );
            now.insert(id);
        }
        for (id, rest) in &self.handles {
            let tokens = u32::try_from(rest.resting.tokens).unwrap_or(u32::MAX);
            let tier = match (self.cache.resident)(*id) {
                Some(false) => WitTier::Host,
                _ => WitTier::Device,
            };
            let idle_ns = now_ms().saturating_sub(rest.since_ms) * 1_000_000;
            self.host.put_working_set(
                &id.to_string(),
                None,
                None,
                rest.resting.session.clone(),
                tokens,
                idle_ns,
                vec![(WORKER.to_string(), tier, tokens)],
            );
            now.insert(*id);
        }
        for gone in self.put.difference(&now).copied().collect::<Vec<_>>() {
            if !self.procs.contains_key(&gone) {
                self.retained.remove(&gone);
                self.host
                    .evicted(&gone.to_string(), WORKER, by_name(By::Host), true);
            }
        }
        self.put = now;
    }

    /// The live processes' counts.
    fn counters(&self) -> Vec<Counter> {
        self.procs
            .iter()
            .map(|(id, proc)| {
                let count = |n: u64| u32::try_from(n).ok();
                (
                    id.to_string(),
                    proc.prompt_tokens.and_then(count),
                    proc.cached_tokens.and_then(count),
                    count(proc.generated_tokens),
                )
            })
            .collect()
    }

    /// The rows a round committed. Notes for the cluster go out with the
    /// worker's state report.
    fn enact(&mut self, delta: Delta) {
        for (writer, subject, name, value) in delta.notes {
            let key = format!("note/{writer}/{}/{}/{name}", subject.kind(), subject.id());
            let value = value.map(|v| match v {
                Value::Boolean(b) => serde_json::Value::Bool(b),
                Value::Int(i) => serde_json::Value::from(i),
                Value::Float(f) => serde_json::Number::from_f64(f)
                    .map_or(serde_json::Value::Null, serde_json::Value::Number),
                Value::Text(s) => serde_json::Value::String(s),
            });
            tracing::debug!(%key, ?value, "worker policy note");
            super::publish(key, value);
        }
        let pid = |id: String| ProcessId::from_str(&id).ok();
        let c = &mut self.columns;
        let scheduled = apply(
            delta.schedule_priority,
            &mut c.priority,
            &mut c.priority_default,
            pid,
        ) | apply(delta.schedule_hold, &mut c.hold, &mut c.hold_default, pid);
        if scheduled {
            tracing::debug!(
                priority = c.priority.len(),
                hold = c.hold.len(),
                hold_default = ?c.hold_default,
                "schedule rows installed"
            );
            schedule::install(Some(schedule::Table::new(
                c.priority.clone(),
                c.priority_default,
                c.hold.clone(),
                c.hold_default,
            )));
        }
        let kept = apply(
            delta.retention_priority,
            &mut c.retention,
            &mut c.retention_default,
            |(id, tokens): (String, Option<u32>)| Some((pid(id)?, tokens)),
        );
        if kept {
            let mut steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>> = HashMap::new();
            for (&(owner, tokens), &key) in &c.retention {
                steps
                    .entry(owner)
                    .or_default()
                    .push((tokens.map(u64::from), key));
            }
            retention::install(Some(retention::Table::new(steps, c.retention_default)));
        }
        for (id, reason) in delta.terminate {
            let Some(proc) = pid(id).and_then(|id| self.procs.get(&id)) else {
                continue;
            };
            process::terminate_by(proc.pid, Err(reason), By::Policy);
        }
    }

    /// Tiers whose waiters are short: the default responder answers them,
    /// discarding working sets lowest retention key first until what they
    /// free covers the shortfall (`pressure::choose`). Under PLEX 1.1 no
    /// policy hears tier pressure: this is the host's own behaviour.
    fn pressure_round(&mut self) {
        let pressed: Vec<Tier> = Tier::ALL
            .into_iter()
            .filter(|tier| self.pressed[tier.index()] && self.pressure[tier.index()].short > 0)
            .collect();
        self.pressed = [false; 2];
        if pressed.is_empty() {
            return;
        }
        let keys = retention::Installed;
        let mut discarded: Vec<ProcessId> = Vec::new();
        for tier in pressed {
            let short = self.pressure[tier.index()].short;
            if short == 0 {
                nothing_left_ends(tier, "the tier is short no more");
                continue;
            }
            let candidates = self
                .candidates_for(tier, &keys)
                .into_iter()
                .filter(|c| ProcessId::from_str(&c.id).is_ok_and(|id| !discarded.contains(&id)))
                .collect();
            let chosen = pressure::choose(candidates, short);
            if chosen.is_empty() {
                // Said once an episode, as it starts; its repeats at DEBUG.
                if NOTHING_LEFT.lock().unwrap()[tier.index()].seen() {
                    tracing::info!(
                        tier = tier.name(),
                        short,
                        "tier pressure: nothing is left to discard; the waiters wait on the governor"
                    );
                } else {
                    tracing::debug!(
                        tier = tier.name(),
                        short,
                        "tier pressure: still nothing left to discard"
                    );
                }
                pressure::answered_nothing(tier);
            } else {
                nothing_left_ends(tier, "a working set is discarded");
            }
            for id in chosen {
                let Ok(owner) = ProcessId::from_str(&id) else {
                    continue;
                };
                tracing::info!(
                    tier = tier.name(),
                    short,
                    working_set = %owner,
                    "tier pressure: the default responder discards a working set"
                );
                match self.handles.get(&owner) {
                    Some(handle) => crate::server::discard_handle(
                        handle.resting.client,
                        owner,
                        "tier pressure: the default responder discarded it".to_string(),
                    ),
                    None => (self.cache.discard)(owner),
                }
                discarded.push(owner);
            }
        }
    }

    /// What a working set holds, kv bytes then state bytes, where known.
    fn held_bytes(
        &self,
        id: ProcessId,
        pages: Option<u64>,
        page: u64,
    ) -> (Option<u64>, Option<u64>) {
        let Some(scale) = (self.cache.scale)() else {
            return (None, None);
        };
        let known = |n: u64| (n > 0).then_some(n);
        if let Some(Rest { resting, .. }) = self.handles.get(&id) {
            let pages = resting.tokens.div_ceil(page);
            let state = u64::from(resting.slots).saturating_mul(scale.state_slot_bytes);
            return (
                known(scale.kv_page_bytes).map(|b| b.saturating_mul(pages)),
                (resting.slots == 0 || scale.state_slot_bytes > 0).then_some(state),
            );
        }
        (
            pages
                .zip(known(scale.kv_page_bytes))
                .map(|(p, b)| p.saturating_mul(b)),
            None,
        )
    }

    /// The working sets the default responder may discard to make room on
    /// `tier`, with their keys and what discarding each frees there.
    fn candidates_for(&self, tier: Tier, keys: &dyn retention::Keys) -> Vec<pressure::Candidate> {
        let now = now_ms();
        let (sets, page) = (self.cache.list)();
        let page = u64::from(page.max(1));
        let mut out = Vec::new();
        for set in sets {
            let owner = set.owner;
            // A live process's working set is not the responder's.
            if self.procs.contains_key(&owner) {
                continue;
            }
            let (kv, _) = self.held_bytes(owner, Some(set.pages), page);
            let ago = set.last_use.elapsed().as_millis() as u64;
            // Cached kv may sit on either tier: how much on the host is not
            // told.
            let bytes = match tier {
                Tier::Device => kv,
                Tier::Host => None,
            };
            out.push(pressure::Candidate {
                key: keys.key_at(Some(owner), 0),
                id: owner.to_string(),
                last_use_ms: now.saturating_sub(ago),
                bytes,
            });
        }
        for (&owner, rest) in &self.handles {
            let (kv, state) = self.held_bytes(owner, None, page);
            let whole = kv.map(|kv| kv.saturating_add(state.unwrap_or(0)));
            // A handle frees room only on the tier it rests on.
            let bytes = match (tier, (self.cache.resident)(owner)) {
                (Tier::Device, Some(false)) | (Tier::Host, Some(true)) => Some(0),
                _ => whole,
            };
            out.push(pressure::Candidate {
                key: keys.key_at(Some(owner), 0),
                id: owner.to_string(),
                last_use_ms: rest.since_ms,
                bytes,
            });
        }
        out
    }
}
