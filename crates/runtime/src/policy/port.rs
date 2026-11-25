//! The worker's PLEX 1.1 port: the runtime's reports into the PLEX host
//! (`plex-host`), and the host's committed rows back out.
//!
//! Three threads:
//! - any runtime thread reports an [`Event`] over a channel, as it did to the
//!   1.0 policy thread (every call site goes through [`super::report`]);
//! - this port thread turns each event into a host report, keeps the record
//!   the host cannot see (cached working sets, resting handles), sends the worker's counters when a round waits for them, and
//!   takes the rows a round committed;
//! - the host's own thread calls the policies, off every runtime thread.
//!
//! The worker is `self` to its policies. A process's PLEX id is the id its
//! client knows (`schedule::plex_id`): one restarted after a KV reclaim keeps
//! it, and is reported paused by the host, then started again.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use client_api::Props;
use plex_host::host::{Config, Counter, Delta, Host};
use plex_host::runtime::wit::Tier as WitTier;
use plex_host::tables::{Subject, Value};

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

static EVENTS: crate::slot::Slot<Sender<Event>> = crate::slot::Slot::new();

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
    spawn(host, (name.to_string(), component.to_vec()))
}

fn spawn(mut host: Host, loaded: (String, Vec<u8>)) -> Result<(), LoadError> {
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

/// Swaps the worker policy for `name`'s `component`, as a SIGHUP asks. The
/// old one is unloaded (its rows withdrawn) and the new one loaded; if it
/// does not load, the old one is loaded again. Blocks while it loads.
pub fn reload(name: &str, component: &[u8]) -> Result<(), String> {
    let next = (policy_name(name), component.to_vec());
    let Some(events) = EVENTS.get() else {
        return start(&next.0, &next.1).map_err(|e| e.to_string());
    };
    let (answer, answered) = std::sync::mpsc::sync_channel(1);
    events
        .send(Event::Reload(next, answer))
        .map_err(|_| "the worker's PLEX port has ended".to_string())?;
    answered
        .recv()
        .map_err(|_| "the worker's PLEX port has ended".to_string())?
}

/// Ends the port (its thread stops once its events' sender goes) and
/// withdraws what the policy set, for [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    EVENTS.clear();
    super::schedule::install(None);
    super::retention::reset();
    super::grants::withdraw();
    super::withdraw_published();
}

pub(super) fn report(event: Event) {
    if let Some(events) = EVENTS.get() {
        let _ = events.send(event);
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
    /// Pages: device capacity and free; host capacity and free; disk
    /// capacity and free.
    rooms: fn() -> [(u64, u64); 3],
    resident: fn(ProcessId) -> Option<bool>,
}

const STORES: Cache = Cache {
    owns: |owner| {
        store_registry::all_for_model(0).iter().any(|stores| {
            store_registry::with_kv_lock(&stores.kv, "plex-cache", |kv| kv.holds_kv_of(owner))
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
                        merged.on_device |= set.on_device;
                        merged.on_host |= set.on_host;
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
                rooms[2].1 += kv.disk_swap_available(crate::policy::retention::Installed) as u64;
            });
        }
        rooms
    },
    resident: |holder| crate::planner::planner_of(holder)?.holder_resident(holder),
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
    /// What its fires mapped in from the cache, once one has looked.
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

/// The port's own record: what the host is told comes from it.
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
    /// Offloaded tokens by process, as `pie/offloaded-tokens`.
    offloaded: HashMap<ProcessId, u64>,
    /// The policy loaded, to load again if its successor does not load.
    loaded: (String, Vec<u8>),
    version: u64,
}

/// A column's rows and default as a table change: rows by the keys this
/// worker can name, `None` where deleted; keys naming nothing it could hold
/// are passed over.
fn rows<K, V, R, W>(
    column: usize,
    delta: plex_host::host::ColumnDelta<K, V>,
    key: impl Fn(K) -> Option<R>,
    value: impl Fn(V) -> W,
) -> (Vec<(R, Option<W>)>, Option<Option<W>>) {
    let mut out = Vec::with_capacity(delta.rows.len());
    for (k, row) in delta.rows {
        let Some(k) = key(k) else {
            super::ROWS_PASSED_OVER[column].fetch_add(1, Ordering::Relaxed);
            continue;
        };
        super::ROWS_ENACTED[column].fetch_add(1, Ordering::Relaxed);
        out.push((k, row.map(&value)));
    }
    (out, delta.default.map(|d| d.map(&value)))
}

fn run(host: Host, loaded: (String, Vec<u8>), events: Receiver<Event>) {
    let mut port = Port {
        host,
        cache: STORES,
        procs: HashMap::new(),
        retained: HashMap::new(),
        handles: HashMap::new(),
        sessions: HashMap::new(),
        put: HashSet::new(),
        offloaded: HashMap::new(),
        loaded,
        version: 0,
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
    fn reload(&mut self, (name, component): (String, Vec<u8>)) -> Result<(), String> {
        let (old, old_component) = &self.loaded;
        self.host.unload(old);
        // What it noted for the cluster goes with it.
        super::withdraw_published();
        match self.host.load(&name, &component) {
            Ok(imports) => {
                tracing::info!(policy = %name, ?imports, "worker policy reloaded (PLEX 1.1)");
                self.loaded = (name, component);
                Ok(())
            }
            Err(error) => {
                tracing::error!(%error, "worker policy not reloaded; the running one goes on");
                let _ = self.host.load(old, old_component);
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
                let held = (self.cache.owns)(id);
                tracing::debug!(target: "pie::plex", %id, held, "exited");
                if held {
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
        // Runtime-derived facts a policy may pace starts by (`default`'s
        // seat-to-end); the lanes absent (not 0) until learned.
        let (lanes, spread) = crate::planner::to_end_facts();
        let lanes = lanes.map(|n| Value::Int(i64::try_from(n).unwrap_or(i64::MAX)));
        self.host.attr(Subject::Worker(WORKER.to_string()), "pie/lanes-to-end", lanes);
        self.host.attr(
            Subject::Worker(WORKER.to_string()),
            "pie/starts-spread",
            Some(Value::Boolean(spread)),
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
        // A set its exited owner's stores still hold (its last fire not
        // settled, so what it leaves is not cached yet) is not gone: its
        // rows stay for the cache it becomes.
        for gone in self.put.difference(&now).copied().collect::<Vec<_>>() {
            if self.procs.contains_key(&gone) || (self.cache.owns)(gone) {
                now.insert(gone);
                continue;
            }
            tracing::debug!(target: "pie::plex", id = %gone, "working set gone");
            self.retained.remove(&gone);
            self.host
                .evicted(&gone.to_string(), WORKER, by_name(By::Host), true);
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
        let (keys, key_default) = rows(0, delta.schedule_priority, pid, |v| v);
        let (holds, hold_default) = rows(1, delta.schedule_hold, pid, |v| v);
        schedule::apply(schedule::Change {
            keys,
            key_default,
            holds,
            hold_default,
        });
        let prefix = |(id, tokens): (String, Option<u32>)| Some((pid(id)?, tokens.map(u64::from)));
        let (keys, key_default) = rows(2, delta.retention_priority, prefix, |v| v);
        let (tiers, tiers_default) = rows(3, delta.retention_tiers, prefix, |v| v);
        retention::apply(retention::Change {
            keys,
            key_default,
            tiers,
            tiers_default,
        });
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
}

/// A port with no stores behind it, for tests that run a built-in policy
/// over reports and read what it committed from the installed tables.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    const NO_STORES: Cache = Cache {
        // What an exited process leaves is held: it is retained.
        owns: |_| true,
        list: || (Vec::new(), 1),
        rooms: || [(0, 0); 3],
        resident: |_| None,
    };

    /// The worker as a test puts it to the host: its attrs, and its time
    /// per step.
    #[derive(Default)]
    pub(crate) struct Worker {
        pub attrs: Vec<(&'static str, Value)>,
        pub step_ns: Option<u64>,
    }

    /// [`enact`] on a worker with a device tier and `worker`'s attrs, put
    /// before the first event.
    pub(crate) fn enact_on(name: &str, component: &[u8], worker: &Worker, events: Vec<Event>) {
        let mut host = host();
        host.put_worker(
            WORKER,
            false,
            vec![(WitTier::Device, Some(1 << 20), 0)],
            0,
            0,
            worker.step_ns,
        );
        for (attr, value) in &worker.attrs {
            host.attr(Subject::Worker(WORKER.to_string()), attr, Some(value.clone()));
        }
        host.load(name, component).expect("the policy loads");
        host.start();
        let mut port = Port {
            host,
            cache: NO_STORES,
            procs: HashMap::new(),
            retained: HashMap::new(),
            handles: HashMap::new(),
            sessions: HashMap::new(),
            put: HashSet::new(),
            offloaded: HashMap::new(),
            loaded: (name.to_string(), component.to_vec()),
            version: 0,
        };
        for event in events {
            port.apply(event);
            // The host's tree has each live working set once it wrote.
            port.put_working_sets();
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while port.host.heard() < port.host.reported() {
            assert!(
                std::time::Instant::now() < deadline,
                "the policy heard every report"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let delta = port.host.poll_delta();
        port.enact(delta);
        port.host.stop();
    }
}

/// What the port tells its policies, read off the `test-watch` policy's
/// note, and how the host takes a faulting policy.
#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::policy::{Arrival, Resting};

    fn props(value: serde_json::Value) -> Props {
        serde_json::from_value(value).unwrap()
    }

    const NO_CACHE: Cache = Cache {
        owns: |_| false,
        list: || (Vec::new(), 16),
        rooms: || [(100, 40), (400, 300), (0, 0)],
        resident: |_| None,
    };

    fn manual() -> Host {
        Host::new(Config {
            supported: SUPPORTED.iter().map(|c| c.to_string()).collect(),
            manual: true,
            ..Config::default()
        })
    }

    fn component(name: &str) -> &'static [u8] {
        builtins::policy(name)
            .unwrap_or_else(|| panic!("no built-in policy {name}"))
            .component
    }

    /// A port whose rounds a test runs itself, `test-watch` loaded.
    fn port(cache: Cache) -> Port {
        let host = manual();
        host.load("test-watch", component("test-watch")).unwrap();
        let mut port = Port {
            host,
            cache,
            procs: HashMap::new(),
            retained: HashMap::new(),
            handles: HashMap::new(),
            sessions: HashMap::new(),
            put: HashSet::new(),
            offloaded: HashMap::new(),
            loaded: ("test-watch".to_string(), Vec::new()),
            version: 0,
        };
        seen(&mut port);
        port
    }

    /// What the policy saw in a round run now, a line each.
    fn seen(port: &mut Port) -> Vec<String> {
        port.put_working_sets();
        assert!(port.host.run(true), "{:?}", port.host.stats().last_fault);
        let notes = port.host.poll_delta().notes;
        let Some((.., Some(Value::Text(text)))) =
            notes.into_iter().find(|(_, _, name, _)| name == "seen")
        else {
            panic!("the policy notes what it saw");
        };
        text.lines().map(String::from).collect()
    }

    fn has(seen: &[String], line: &str) -> bool {
        seen.iter().any(|l| l == line)
    }

    fn arrive(port: &mut Port, tenant: &str, session: Option<&str>) -> ProcessId {
        let pid = ProcessId::new_v4();
        port.apply(Event::Arrived(
            pid,
            Arrival {
                tenant: tenant.into(),
                session: session.map(Into::into),
                ..Arrival::default()
            },
        ));
        pid
    }

    fn process(id: ProcessId, seen: &[String]) -> Option<String> {
        seen.iter()
            .find_map(|l| l.strip_prefix(&format!("process {id} ")))
            .map(String::from)
    }

    #[test]
    fn a_process_arrives_with_its_props_and_its_changes_are_edges() {
        let mut port = port(NO_CACHE);
        let pid = ProcessId::new_v4();
        port.apply(Event::Arrived(
            pid,
            Arrival {
                tenant: "t".into(),
                props: props(json!({"priority": 1})),
                ..Arrival::default()
            },
        ));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("arrived {pid}")), "{saw:?}");
        assert!(process(pid, &saw).unwrap().ends_with("priority=Value::Int(1)"), "{saw:?}");

        port.apply(Event::PropsChanged(
            pid,
            props(json!({"priority": 7})),
            vec!["priority".into()],
        ));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("hint process:{pid} priority")), "{saw:?}");
        assert!(process(pid, &saw).unwrap().ends_with("priority=Value::Int(7)"), "{saw:?}");
        assert!(!saw.iter().any(|l| l.contains("tier")));

        // A process the policy never heard of has no props to change.
        port.apply(Event::PropsChanged(
            ProcessId::new_v4(),
            props(json!({"a": 1})),
            vec!["a".into()],
        ));
        let saw = seen(&mut port);
        assert!(!saw.iter().any(|l| l.starts_with("hint ")), "{saw:?}");
    }

    #[test]
    fn a_session_is_scoped_by_its_tenant() {
        let mut port = port(NO_CACHE);
        let a1 = arrive(&mut port, "a", Some("chat"));
        arrive(&mut port, "a", Some("chat"));
        let b1 = arrive(&mut port, "b", Some("chat"));
        let lone = arrive(&mut port, "a", None);
        let saw = seen(&mut port);
        let of = |id| process(id, &saw).unwrap();
        assert!(of(a1).starts_with("session=a/chat tenant=a "), "{saw:?}");
        assert!(of(b1).starts_with("session=b/chat tenant=b "), "{saw:?}");
        assert!(of(lone).starts_with("session= tenant=a "), "{saw:?}");
    }

    #[test]
    fn a_session_s_props_change_between_its_processes_and_it_closes() {
        let mut port = port(NO_CACHE);
        let session = || "t/chat".to_string();
        let hints = |saw: &[String]| -> Vec<String> {
            let mut names: Vec<_> = saw
                .iter()
                .filter_map(|l| l.strip_prefix("hint session:t/chat "))
                .map(String::from)
                .collect();
            names.sort();
            names
        };
        port.apply(Event::SessionProps(
            session(),
            props(json!({"tool-eta-ms": 5000, "tool": "grep"})),
        ));
        let saw = seen(&mut port);
        assert_eq!(hints(&saw), ["tool", "tool-eta-ms"]);
        assert!(has(&saw, "has session:t/chat tool-eta-ms=Value::Int(5000)"), "{saw:?}");

        port.apply(Event::SessionProps(session(), props(json!({"tool": "grep"}))));
        assert!(hints(&seen(&mut port)).is_empty(), "nothing changed");
        port.apply(Event::SessionProps(session(), props(json!({"tool": null}))));
        assert_eq!(hints(&seen(&mut port)), ["tool"]);

        port.apply(Event::SessionClosed(session()));
        let saw = seen(&mut port);
        assert!(has(&saw, "closed t/chat"), "{saw:?}");
        assert!(has(&saw, "has session:t/chat tool-eta-ms=none"), "{saw:?}");
    }

    #[test]
    fn the_call_about_a_process_exit_still_reads_it() {
        let mut port = port(NO_CACHE);
        let id = arrive(&mut port, "t", None);
        seen(&mut port);
        port.apply(Event::Exited(id, Outcome::Completed));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("exited {id}")), "{saw:?}");
        let of = process(id, &saw).expect("the call about its exit reads it");
        assert!(of.contains("tenant=t ") && of.contains("state=ProcessState::Exited"), "{of}");
        assert_eq!(process(id, &seen(&mut port)), None, "the next call does not");
    }

    #[test]
    fn a_resting_handle_is_a_working_set_the_policy_reads_until_it_is_lent() {
        let mut port = port(NO_CACHE);
        let id = ProcessId::new_v4();
        let rest = Event::HandleRested(
            id,
            Resting {
                tenant: "t".into(),
                session: Some("chat".into()),
                props: props(json!({"pico/frozen-tokens": 512})),
                tokens: 640,
                folded: 512,
            },
        );
        port.apply(rest.clone());
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("retained {id}")), "{saw:?}");
        let set =
            format!("working-set {id} owner= session=chat tokens=640 folded=Value::Int(512)");
        assert!(has(&saw, &set), "{saw:?}");
        let frozen = format!("has working-set:{id} pico/frozen-tokens=Value::Int(512)");
        assert!(has(&saw, &frozen), "{saw:?}");

        port.apply(Event::HandleProps(id, props(json!({"a": 1})), vec!["a".into()]));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("hint working-set:{id} a")), "{saw:?}");
        assert!(has(&saw, &format!("has working-set:{id} a=Value::Int(1)")), "{saw:?}");

        // Lent to a process, it is out of the policy's reach.
        port.apply(Event::HandleLent(id));
        assert!(!has(&seen(&mut port), &set));
        port.apply(rest);
        assert!(has(&seen(&mut port), &set), "at rest again");
        port.apply(Event::HandleGone(id, By::Client));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("evicted {id} by By::Client")), "{saw:?}");
        assert!(!has(&saw, &set), "{saw:?}");
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
                    on_device: true,
                    on_host: false,
                })
                .collect();
            (sets, 16)
        },
        rooms: || [(100, 40), (400, 300), (0, 0)],
        resident: |_| None,
    };

    #[test]
    fn a_working_set_is_retained_when_its_owner_exits_until_its_last_entry_goes() {
        let mut port = port(TEST_CACHE);
        let gone = arrive(&mut port, "t", Some("chat"));
        let live = arrive(&mut port, "t", None);
        let uncached = arrive(&mut port, "t", None);
        port.apply(Event::Cached(gone, 32));
        port.apply(Event::Cached(gone, 0));
        assert_eq!(port.procs[&gone].cached_tokens, Some(32), "what its fires mapped in");
        assert_eq!(port.procs[&live].cached_tokens, None, "it never looked");
        CACHED.lock().unwrap().extend([(gone, 4), (live, 2)]);

        port.apply(Event::Exited(gone, Outcome::Completed));
        port.apply(Event::Exited(uncached, Outcome::Completed));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("exited {gone}")) && has(&saw, &format!("retained {gone}")));
        assert!(has(&saw, &format!("exited {uncached}")));
        assert!(!has(&saw, &format!("retained {uncached}")), "nothing of it outlives it");
        let set = |id, owner: &str, session, tokens| {
            format!("working-set {id} owner={owner} session={session} tokens={tokens} folded=none")
        };
        assert!(has(&saw, &set(gone, "", "t/chat", 64)), "{saw:?}");
        assert!(has(&saw, &set(live, &live.to_string(), "", 32)), "{saw:?}");
        assert!(!saw.iter().any(|l| l.starts_with(&format!("working-set {uncached}"))));

        CACHED.lock().unwrap().retain(|(owner, _)| *owner != live);
        port.apply(Event::CacheEmptied(live, By::Client));
        let saw = seen(&mut port);
        assert!(
            !saw.iter().any(|l| l.starts_with("evicted ")),
            "a live working set ends with its owner: {saw:?}"
        );
        CACHED.lock().unwrap().retain(|(owner, _)| *owner != gone);
        port.apply(Event::CacheEmptied(gone, By::Host));
        let saw = seen(&mut port);
        assert!(has(&saw, &format!("evicted {gone} by By::Host")), "{saw:?}");
        assert!(!saw.iter().any(|l| l.starts_with("working-set ")), "{saw:?}");
        CACHED.lock().unwrap().clear();
    }

    #[test]
    fn a_client_prop_becomes_the_contracts_value() {
        let mut got = hints(&props(json!({
            "flag": true, "priority": 3, "weight": 0.5, "tier": "gold", "tags": ["a", "b"],
            "gone": null,
        })));
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let expected = [
            ("flag", Value::Boolean(true)),
            ("priority", Value::Int(3)),
            ("tags", Value::Text(r#"["a","b"]"#.into())),
            ("tier", Value::Text("gold".into())),
            ("weight", Value::Float(0.5)),
        ]
        .map(|(name, value)| (name.to_string(), value));
        assert_eq!(format!("{got:?}"), format!("{expected:?}"));
    }

    /// Every built-in is a worker's or a cluster's: a worker host loads the
    /// first, which write no placement, replication or share column, and
    /// refuses the second by name.
    #[test]
    fn every_built_in_policy_loads_and_imports_no_cluster_writes() {
        use plex_host::tables::COLUMNS;
        let mut loaded = Vec::new();
        for builtin in builtins::policies() {
            match manual().load(builtin.name, builtin.component) {
                Ok(imports) => {
                    let writes: Vec<_> = imports
                        .iter()
                        .filter(|i| COLUMNS.contains(&i.as_str()))
                        .collect();
                    assert!(
                        writes.iter().all(|i| SUPPORTED.contains(&i.as_str())),
                        "{}: {writes:?}",
                        builtin.name
                    );
                    loaded.push(builtin.name);
                }
                Err(why) => {
                    assert!(why.contains("which this host cannot enact"), "{}: {why}", builtin.name);
                    let mut columns = COLUMNS.map(String::from).to_vec();
                    columns.retain(|c| !SUPPORTED.contains(&c.as_str()));
                    let cluster = Host::new(Config {
                        supported: columns,
                        manual: true,
                        ..Config::default()
                    });
                    cluster
                        .load(builtin.name, builtin.component)
                        .unwrap_or_else(|why| panic!("{}: {why}", builtin.name));
                }
            }
        }
        for worker in ["default", "noop"] {
            assert!(loaded.contains(&worker), "{worker} is a worker's: {loaded:?}");
        }
        assert!(!loaded.contains(&"default-cluster"), "{loaded:?}");
    }

    #[test]
    fn a_trap_discards_the_call_and_the_program_runs_again() {
        let host = manual();
        host.load("test-trap", component("test-trap")).unwrap();
        for calls in 1..=2 {
            host.run(true);
            let stats = host.stats();
            assert_eq!((stats.calls, stats.traps, stats.commits), (calls, calls, 0));
            let default = host.poll_delta().schedule_priority.default;
            assert_eq!(default, None, "its row is discarded");
        }
    }

    #[test]
    fn an_overrun_is_cut_off_at_the_deadline() {
        let host = manual();
        host.load("test-spin", component("test-spin")).unwrap();
        let deadline = Duration::from_millis(Config::default().deadline_ms);
        for calls in 1..=2 {
            let started = std::time::Instant::now();
            host.run(true);
            let spent = started.elapsed();
            let stats = host.stats();
            assert_eq!((stats.calls, stats.deadlines, stats.commits), (calls, calls, 0));
            assert!(spent < deadline * 20, "cut off in {spent:?}");
            let default = host.poll_delta().schedule_priority.default;
            assert_eq!(default, None, "its row is discarded");
        }
    }
}
