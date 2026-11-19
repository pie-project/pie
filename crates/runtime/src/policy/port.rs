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
use std::sync::atomic::Ordering;
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
    spawn(host(), None)?;
    // How long idle cache stays is a policy's to state, with its retention
    // keys; with none, the default is said here rather than left implied.
    tracing::info!(
        "no worker policy: idle cache has no time-based expiry; it stays until a demand needs its room, \
         then goes lowest retention key, then least recently used, first"
    );
    Ok(())
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
/// Per tier: an episode of the default responder keeping sets keyed above
/// the demand, said once at INFO as it starts.
static KEPT: std::sync::Mutex<[process::WaitEpisode; 2]> =
    std::sync::Mutex::new([process::WaitEpisode::new(), process::WaitEpisode::new()]);

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
    /// Moves `owner`'s idle cache down a tier (device to host, then disk),
    /// never dropping it: whether a move is on its way.
    demote: fn(ProcessId) -> bool,
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
    demote: |owner| {
        crate::planner::planners()
            .iter()
            .fold(false, |moved, planner| planner.demote_owner(owner) || moved)
    },
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
    /// Waiting on its client: paused by the client, whatever else moves
    /// its KV meanwhile.
    idle: bool,
    session: Option<String>,
    /// Its `cached-tokens`, two meanings by state. Waiting: the leading
    /// tokens of its prompt some kv store holds now, kept up as the cache
    /// changes (store::waiting); only a prompt the runtime has at arrival
    /// (`prompt-tokens`, or a completion's `raw-prompt`) is looked up, so a
    /// chat request reads none while it waits and a policy ranking waiting
    /// work by it (lpm) passes chat over. Running and after: what its fires
    /// mapped in from the cache, once one has looked.
    cached_tokens: Option<u64>,
    /// The rows its fires wrote before the first readout; known once it
    /// read one out.
    prompt_rows: u64,
    prompt_tokens: Option<u64>,
    /// The rows its fires read out.
    generated_tokens: u64,
    /// The rows its fires wrote: its working set's length.
    rows: u64,
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
    tiers: HashMap<(ProcessId, Option<u32>), u8>,
    tiers_default: Option<u8>,
}

/// A column's rows and default into `rows` and `default`; keys that name
/// no process this worker could hold are passed over.
fn apply<K, V, R: std::hash::Hash + Eq>(
    column: usize,
    delta: plex_host::host::ColumnDelta<K, V>,
    rows: &mut HashMap<R, V>,
    default: &mut Option<V>,
    key: impl Fn(K) -> Option<R>,
) -> bool {
    let changed = !delta.rows.is_empty() || delta.default.is_some();
    for (k, row) in delta.rows {
        let Some(k) = key(k) else {
            super::ROWS_PASSED_OVER[column].fetch_add(1, Ordering::Relaxed);
            continue;
        };
        super::ROWS_ENACTED[column].fetch_add(1, Ordering::Relaxed);
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
    let mut port = Port::new(host, STORES, loaded);
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
        let _ = port.pressure_round(&retention::Installed);
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
    fn new(host: Host, cache: Cache, loaded: Option<(String, Vec<u8>)>) -> Self {
        Port {
            host,
            cache,
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
        }
    }

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
                        idle: false,
                        cached_tokens: None,
                        prompt_rows: 0,
                        prompt_tokens: None,
                        generated_tokens: 0,
                        rows: 0,
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
                    proc.rows += rows;
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
            Event::Waiting(pid, tokens) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id)
                    && proc.state == State::Waiting
                {
                    proc.cached_tokens = Some(tokens);
                }
            }
            Event::Started(pid) | Event::Resumed(pid) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    if proc.state == State::Waiting {
                        // What it found while waiting is its fires' to count
                        // from its start.
                        proc.cached_tokens = None;
                    }
                    proc.state = State::Running;
                    host.started(&id.to_string());
                }
            }
            Event::Idle(pid, idle) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id) {
                    proc.idle = idle;
                    if idle {
                        proc.state = State::Paused;
                        host.paused(&id.to_string(), "client");
                    } else {
                        proc.state = State::Running;
                        host.started(&id.to_string());
                    }
                }
            }
            Event::Paused(pid) => {
                let id = schedule::plex_id(pid);
                if let Some(proc) = self.procs.get_mut(&id)
                    && !proc.idle
                {
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
        // A live process's own working set, before any of it is cached: a
        // policy's rows on it (its retention while it waits) need it known.
        for (id, proc) in &self.procs {
            if now.contains(id) || proc.rows == 0 {
                continue;
            }
            let tokens = u32::try_from(proc.rows).unwrap_or(u32::MAX);
            let tier = match (self.cache.resident)(proc.pid) {
                Some(false) => WitTier::Host,
                _ => WitTier::Device,
            };
            self.host.put_working_set(
                &id.to_string(),
                None,
                Some(id.to_string()),
                proc.session.clone(),
                tokens,
                0,
                vec![(WORKER.to_string(), tier, tokens)],
            );
            now.insert(*id);
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
            0,
            delta.schedule_priority,
            &mut c.priority,
            &mut c.priority_default,
            pid,
        ) | apply(
            1,
            delta.schedule_hold,
            &mut c.hold,
            &mut c.hold_default,
            pid,
        );
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
        let prefix = |(id, tokens): (String, Option<u32>)| Some((pid(id)?, tokens));
        let kept = apply(
            2,
            delta.retention_priority,
            &mut c.retention,
            &mut c.retention_default,
            prefix,
        ) | apply(
            3,
            delta.retention_tiers,
            &mut c.tiers,
            &mut c.tiers_default,
            prefix,
        );
        if kept {
            tracing::debug!(
                priority = c.retention.len(),
                tiers = c.tiers.len(),
                "retention rows installed"
            );
            let mut steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>> = HashMap::new();
            for (&(owner, tokens), &key) in &c.retention {
                steps
                    .entry(owner)
                    .or_default()
                    .push((tokens.map(u64::from), key));
            }
            let mut tiers: HashMap<ProcessId, Vec<(Option<u64>, u8)>> = HashMap::new();
            for (&(owner, tokens), &set) in &c.tiers {
                tiers
                    .entry(owner)
                    .or_default()
                    .push((tokens.map(u64::from), set));
            }
            retention::install(Some(
                retention::Table::new(steps, c.retention_default)
                    .with_tiers(tiers, c.tiers_default),
            ));
        }
        for (id, reason) in delta.terminate {
            let Some(proc) = pid(id).and_then(|id| self.procs.get(&id)) else {
                continue;
            };
            // A running process ends its reply at its next fire; one not
            // started yet is refused.
            match proc.state {
                State::Running => super::stop(proc.pid, reason),
                _ => process::terminate_by(proc.pid, Err(reason), By::Policy),
            }
        }
    }

    /// Tiers whose waiters are short: the default responder answers them,
    /// discarding working sets lowest retention key first until what they
    /// free covers the shortfall, keyed by `keys`. Under PLEX 1.1 no
    /// policy hears tier pressure: this is the host's own behaviour. The
    /// working sets it discarded.
    fn pressure_round(&mut self, keys: &dyn retention::Keys) -> Vec<ProcessId> {
        let pressed: Vec<Tier> = Tier::ALL
            .into_iter()
            .filter(|tier| self.pressed[tier.index()] && self.pressure[tier.index()].short > 0)
            .collect();
        self.pressed = [false; 2];
        if pressed.is_empty() {
            return Vec::new();
        }
        let mut discarded: Vec<ProcessId> = Vec::new();
        for tier in pressed {
            let short = self.pressure[tier.index()].short;
            let demand = self.pressure[tier.index()].demand;
            if short == 0 {
                nothing_left_ends(tier, "the tier is short no more");
                continue;
            }
            let rooms = (self.cache.rooms)();
            let below = tier == Tier::Device && rooms[1].1 + rooms[2].1 > 0;
            let mut candidates: Vec<pressure::Candidate> = self
                .candidates_for(tier, keys)
                .into_iter()
                // The runtime drops no state on its own: only idle cache
                // (the cached sets of finished processes) goes. A resting
                // handle is a live process's state its client will resume,
                // and discarding it fails that client's next call; under
                // PLEX 1.1 no policy hears tier pressure to choose one.
                .filter(|c| {
                    ProcessId::from_str(&c.id)
                        .is_ok_and(|id| !self.handles.contains_key(&id) && !discarded.contains(&id))
                })
                .filter(|c| c.bytes != Some(0) && c.key != pressure::PINNED)
                .collect();
            candidates
                .sort_by(|a, b| (a.key, a.last_use_ms, &a.id).cmp(&(b.key, b.last_use_ms, &b.id)));
            // Lowest key first, until what was freed covers the shortfall.
            // Each candidate is judged where it stands (`take_for`): moved
            // down a tier where one below has room and the demand may move
            // it; else discarded where the demand may discard it; else kept,
            // and not counted.
            let mut freed: u64 = 0;
            let mut took = false;
            for candidate in candidates {
                if freed >= short {
                    break;
                }
                let Ok(owner) = ProcessId::from_str(&candidate.id) else {
                    continue;
                };
                let key = candidate.key;
                let movable = below;
                match take_for(demand, key, movable, || (self.cache.demote)(owner)) {
                    Take::MovedDown => {
                        tracing::info!(
                            tier = tier.name(),
                            short,
                            working_set = %owner,
                            key,
                            up_to = ?demand.up_to,
                            "tier pressure: the default responder moves a working set down a tier"
                        );
                    }
                    Take::Discard => {
                        tracing::info!(
                            tier = tier.name(),
                            short,
                            working_set = %owner,
                            key,
                            up_to = ?demand.up_to,
                            "tier pressure: the default responder discards a working set"
                        );
                        (self.cache.discard)(owner);
                        discarded.push(owner);
                    }
                    Take::Kept => {
                        let why = if !movable {
                            "keyed above the demand, and no tier below takes it"
                        } else if !demand.may_demote(key) {
                            "keyed above the demand, which running work will serve"
                        } else {
                            "keyed above the demand, and it could not move down"
                        };
                        // Said once an episode at INFO, its repeats at DEBUG.
                        if KEPT.lock().unwrap()[tier.index()].seen() {
                            tracing::info!(tier = tier.name(), working_set = %owner, key, up_to = ?demand.up_to, why, "tier pressure: the default responder keeps a working set");
                        } else {
                            tracing::debug!(tier = tier.name(), working_set = %owner, key, up_to = ?demand.up_to, why, "tier pressure: the default responder keeps a working set");
                        }
                        continue;
                    }
                }
                took = true;
                match candidate.bytes {
                    Some(bytes) => freed = freed.saturating_add(bytes),
                    // Its size unknown: the waiter's next report says
                    // whether that was enough.
                    None => break,
                }
            }
            if took {
                nothing_left_ends(tier, "a working set is moved down or discarded");
                let _ = KEPT.lock().unwrap()[tier.index()].end();
            } else {
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
            }
        }
        discarded
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

/// What the default responder does with a chosen set.
#[derive(Debug, PartialEq, Eq)]
enum Take {
    /// Moved down a tier: kept, not lost.
    MovedDown,
    /// Left where it is: it could not move down, and its key is above what
    /// the demand may discard.
    Kept,
    Discard,
}

/// A set keyed `key`, judged where it stands: moved down a tier where one
/// below takes it (`movable`) and the demand may move it (`demote` starts the
/// move and says whether it did); else discarded where the demand's key
/// allows; else kept, and not counted toward the shortfall.
fn take_for(
    demand: pressure::Demand,
    key: i64,
    movable: bool,
    demote: impl FnOnce() -> bool,
) -> Take {
    if movable && demand.may_demote(key) && demote() {
        Take::MovedDown
    } else if demand.may_discard(key) {
        Take::Discard
    } else {
        Take::Kept
    }
}

#[cfg(test)]
mod responder_tests {
    use super::*;
    use crate::policy::pressure::Band;

    /// The 1.1 port's default responder takes for a demand only what its
    /// key lets it, as the 1.0 host's does: within the key, moved down a tier
    /// where one below has room and discarded where none has; above it, only
    /// moved down, and only where the demand is stalled. A6000's conform:
    /// three key-15 sets did not cover the shortfall, and the key-20 set was
    /// discarded next for a lower-keyed ask.
    #[test]
    fn the_default_responder_takes_only_what_the_demand_s_key_lets_it() {
        let bounded = pressure::Demand {
            up_to: Some(15),
            stalled: false,
        };
        let stalled = pressure::Demand {
            up_to: Some(15),
            stalled: true,
        };
        // Within the key: moved down where it can, else discarded.
        assert_eq!(take_for(bounded, 15, true, || true), Take::MovedDown);
        assert_eq!(
            take_for(bounded, 15, true, || false),
            Take::Discard,
            "a move refused is a discard"
        );
        assert_eq!(
            take_for(bounded, 5, false, || unreachable!()),
            Take::Discard
        );
        // Above it: kept unless stalled, and then only moved down; a move
        // is not even asked for where the demand may not move it.
        assert_eq!(take_for(bounded, 20, true, || unreachable!()), Take::Kept);
        assert_eq!(take_for(bounded, 20, false, || unreachable!()), Take::Kept);
        assert_eq!(take_for(stalled, 20, true, || true), Take::MovedDown);
        assert_eq!(take_for(stalled, 20, true, || false), Take::Kept);
        assert_eq!(take_for(stalled, 20, false, || unreachable!()), Take::Kept);
        // Unbounded (no table installed): anything, as before.
        let unbounded = pressure::Demand::default();
        assert_eq!(
            take_for(unbounded, 20, false, || unreachable!()),
            Take::Discard
        );
    }

    /// The idle cache a burst left behind: its owners, how many pages each
    /// kept, and how long ago each was last used, in ms.
    static IDLE: std::sync::Mutex<Vec<(ProcessId, u64, u64)>> = std::sync::Mutex::new(Vec::new());
    /// What the stores were told to discard, in order.
    static SHED: std::sync::Mutex<Vec<ProcessId>> = std::sync::Mutex::new(Vec::new());
    /// The two tests share [`IDLE`] and [`SHED`]: one at a time.
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    /// A kv page, in bytes.
    const PAGE: u64 = 1 << 20;

    /// Stores holding [`IDLE`], with no tier below the device to move to.
    const IDLE_CACHE: Cache = Cache {
        owns: |owner| IDLE.lock().unwrap().iter().any(|(o, _, _)| *o == owner),
        list: || {
            let now = crate::rt::Instant::now();
            let sets = IDLE
                .lock()
                .unwrap()
                .iter()
                .map(|&(owner, pages, ago_ms)| CachedWorkingSet {
                    owner,
                    entries: 1,
                    pages,
                    last_use: now
                        .checked_sub(Duration::from_millis(ago_ms))
                        .unwrap_or(now),
                })
                .collect();
            (sets, 16)
        },
        discard: |owner| {
            IDLE.lock().unwrap().retain(|(o, _, _)| *o != owner);
            SHED.lock().unwrap().push(owner);
        },
        demote: |_| false,
        rooms: || [(100, 40), (0, 0), (0, 0)],
        resident: |_| Some(true),
        scale: || {
            Some(pressure::Scale {
                kv_page_bytes: PAGE,
                state_slot_bytes: 0,
            })
        },
    };

    fn idle_port() -> Port {
        Port::new(host(), IDLE_CACHE, None)
    }

    fn press(port: &mut Port, short: u64, total: u64) {
        port.apply(Event::TierPressure(
            Tier::Device,
            Reading {
                band: Band::of(Band::Clear, short, total),
                short,
                demand: pressure::Demand::default(),
            },
        ));
    }

    /// Tier pressure never discards a resting handle: it is a live
    /// process's state its client will resume (Mac Pro's restore-5 runs
    /// failed requests on "the default responder discards a working set").
    /// Only idle cache goes, and under PLEX 1.1 no policy hears tier
    /// pressure to choose a handle instead.
    #[test]
    fn the_default_responder_never_discards_a_resting_handle() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let mut port = idle_port();
        let (handle, idle) = (ProcessId::new_v4(), ProcessId::new_v4());
        IDLE.lock().unwrap().clear();
        IDLE.lock().unwrap().push((idle, 4, 10));
        SHED.lock().unwrap().clear();
        port.apply(Event::HandleRested(
            handle,
            crate::policy::Resting {
                tenant: "t".into(),
                session: None,
                props: Props::default(),
                tokens: 4096,
                folded: 0,
                slots: 0,
            },
        ));
        assert!(
            port.candidates_for(Tier::Device, &retention::Installed)
                .iter()
                .any(|c| c.id == handle.to_string()),
            "a resting handle is a working set the port reads"
        );
        press(&mut port, 1 << 30, 1 << 31);
        let discarded = port.pressure_round(&retention::Installed);
        assert_eq!(discarded, [idle], "idle cache goes");
        assert!(
            !discarded.contains(&handle),
            "the default responder discarded a live process's state"
        );
        IDLE.lock().unwrap().clear();
        SHED.lock().unwrap().clear();
    }

    /// A pool of chunks of two kv pages each, in segments of four chunks,
    /// whose pages are the burst's sets' interleaved as concurrent decodes
    /// lay them: page `i` of every set before page `i + 1` of any. A chunk
    /// comes free only once both of its pages' sets are gone; an evacuation
    /// moves whole chunks, so the pool maps no fewer segments than its used
    /// chunks fill.
    struct InterleavedPool {
        /// Per chunk, the owners of its two pages.
        chunks: Vec<[ProcessId; 2]>,
        mapped: u64,
    }

    impl InterleavedPool {
        const CHUNKS_PER_SEGMENT: u64 = 4;

        fn used_chunks(&self) -> u64 {
            let idle = IDLE.lock().unwrap();
            let held = |owner: &ProcessId| idle.iter().any(|(o, _, _)| o == owner);
            self.chunks
                .iter()
                .filter(|pages| pages.iter().any(held))
                .count() as u64
        }
    }

    impl crate::governor::PoolTarget for InterleavedPool {
        fn segment_bytes(&self) -> u64 {
            2 * PAGE * Self::CHUNKS_PER_SEGMENT
        }
        fn floor(&self) -> u64 {
            self.segment_bytes()
        }
        fn hard_limit(&self) -> u64 {
            self.chunks.len() as u64 / Self::CHUNKS_PER_SEGMENT * self.segment_bytes()
        }
        fn held(&self) -> u64 {
            self.mapped * self.segment_bytes()
        }
        fn growth(&self) -> crate::governor::Growth {
            crate::governor::Growth::Elastic
        }
        fn retarget(&mut self, segments: u64) -> crate::governor::Retarget {
            let least = self.used_chunks().div_ceil(Self::CHUNKS_PER_SEGMENT);
            self.mapped = self.mapped.min(segments.max(least));
            if self.mapped <= segments {
                crate::governor::Retarget::Reached
            } else {
                crate::governor::Retarget::NoRoom { held: self.mapped }
            }
        }
    }

    /// Yield 1 after a burst, no worker policy: the finished processes'
    /// cached working sets fill the pool, their pages interleaved, and the
    /// system's pressure brings the target under what the pool maps. The
    /// governor's tier pressure reaches the port's default responder, which
    /// discards idle cache lowest retention key first, then least recently
    /// used; one round's discards free pages inside chunks other sets still
    /// use and no segment, so the shortfall stands, the governor says it
    /// again on its next turn, and the responder looks again until the pool
    /// can give the segments back. A live process's working set is never
    /// chosen, and idle cache keyed higher goes last.
    #[test]
    fn device_pressure_with_no_policy_sheds_idle_cache_until_the_pool_reaches() {
        use crate::governor::run::Steer;
        use crate::governor::{PoolTarget, Retarget, Signal, Yield, steered};

        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        const SETS: usize = 7;
        const PAGES: usize = 4;
        let mut port = idle_port();
        let live = ProcessId::new_v4();
        port.apply(Event::Arrived(
            live,
            crate::policy::Arrival {
                tenant: "t".into(),
                session: None,
                props: Props::default(),
            },
        ));
        let idle: Vec<ProcessId> = (0..SETS).map(|_| ProcessId::new_v4()).collect();
        // The burst: each set kept four pages; the lower its index, the
        // longer ago it was last used. The live process's set holds pages
        // too, last used longest ago of all.
        {
            let mut cached = IDLE.lock().unwrap();
            cached.clear();
            cached.extend(
                idle.iter()
                    .enumerate()
                    .map(|(at, &owner)| (owner, PAGES as u64, 1000 - 100 * at as u64)),
            );
            cached.push((schedule::plex_id(live), PAGES as u64, 5000));
        }
        SHED.lock().unwrap().clear();
        // A policy keyed the oldest set above the others: it goes last.
        let kept = idle[0];
        let keys = move |owner: Option<ProcessId>| i64::from(owner == Some(kept)) * 5;

        // Pages laid as the burst decoded them: page `i` of every set, the
        // live one's included, before page `i + 1` of any, in an order that
        // pairs the least recently used sets with later ones.
        let row = [
            idle[1],
            idle[3],
            idle[2],
            idle[4],
            idle[5],
            idle[6],
            idle[0],
            schedule::plex_id(live),
        ];
        let laid: Vec<ProcessId> = (0..PAGES).flat_map(|_| row).collect();
        let chunks: Vec<[ProcessId; 2]> = laid.chunks(2).map(|pair| [pair[0], pair[1]]).collect();
        let mut pool = InterleavedPool {
            mapped: chunks.len() as u64 / InterleavedPool::CHUNKS_PER_SEGMENT,
            chunks,
        };
        assert_eq!(pool.mapped, 4, "the burst fills the pool");
        let mut steer = Steer::new(&pool, Yield::new(1.0).unwrap());
        let total = pool.hard_limit();

        // Pressure 0.6 at yield 1 brings the target to three segments of
        // the four mapped: 8 MiB short, two sets' worth. The first round
        // discards the two least recently used, whose pages share every
        // chunk with sets still kept: no chunk comes free, the pool maps
        // what it did, and the shortfall stands.
        let mut rounds = 0;
        let mut shed_per_round = Vec::new();
        let answer = loop {
            rounds += 1;
            assert!(rounds <= 8, "the pool never reached its target");
            let mut signals = Vec::new();
            let turn = steer.turn(&mut pool, 0.6, &mut |signal| signals.push(signal));
            assert_eq!(turn.segments, 3);
            if turn.answer == Retarget::Reached {
                break turn.answer;
            }
            assert!(
                matches!(turn.answer, Retarget::NoRoom { .. }),
                "{:?}",
                turn.answer
            );
            let [signal @ Signal::TierPressure { .. }] = signals[..] else {
                panic!("a standing shortfall is said every turn: {signals:?}");
            };
            press(&mut port, steered::short_bytes(signal), total);
            let before = SHED.lock().unwrap().len();
            let discarded = port.pressure_round(&keys);
            assert!(!discarded.is_empty(), "idle cache is left to discard");
            shed_per_round.push(SHED.lock().unwrap().len() - before);
        };
        assert_eq!(answer, Retarget::Reached);
        assert!(pool.mapped <= 3);
        assert_eq!(
            shed_per_round,
            [2, 2],
            "one round's discards freed no segment, and the responder looked again"
        );

        // Floor-keyed idle cache least recently used first; the set keyed
        // above it after every floor-keyed one; the live process's never.
        let shed = SHED.lock().unwrap().clone();
        assert_eq!(shed, idle[1..5], "floor-keyed, least recently used first");
        assert!(
            !shed.contains(&schedule::plex_id(live)),
            "a live working set is never chosen"
        );
        assert!(
            !shed.contains(&kept),
            "keyed above the floor, it outlasts them"
        );
        IDLE.lock().unwrap().clear();
        SHED.lock().unwrap().clear();
    }
}
