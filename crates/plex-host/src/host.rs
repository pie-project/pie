//! The host: what the engine has reported, the policies, and the thread
//! that calls them.
//!
//! The engine's thread only ever appends an event, updates a counter, or
//! reads a version number; it never waits for a policy. The policy thread
//! takes the events, freezes a snapshot, calls each policy, and commits
//! each call's writes whole. The port then enacts what changed, at its own
//! pace.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value as Json, json};

use crate::runtime::{Call, Fault, Proc, Program, Snapshot, snap, wit, wit_subject};
use crate::tables::{self, COLUMNS, Column, Prefix, Refusal, Subject, Tables, TierSet, Value};

/// How long the policy thread waits for the engine's counters before it
/// calls on what it has: an idle engine reports nothing, and its counters
/// are current.
const GRACE: Duration = Duration::from_millis(5);

/// How soon a policy whose call failed is called again, doubled for each
/// failure in a row.
const RETRY_NS: u64 = 1_000_000;

#[derive(Clone)]
pub struct Config {
    /// The columns this host's engine or router can enact.
    pub supported: Vec<String>,
    /// CPU time one call may take.
    pub deadline_ms: u64,
    /// The least time between two rounds of calls.
    pub interval_ms: u64,
    /// Failed calls in a row before a policy's rows are withdrawn.
    pub lease_calls: u32,
    /// Time is set by the caller, and rounds run when it says (`run`):
    /// for scenario tests.
    pub manual: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            supported: COLUMNS.iter().map(|c| c.to_string()).collect(),
            deadline_ms: 50,
            interval_ms: 0,
            lease_calls: 3,
            manual: false,
        }
    }
}

/// What a snapshot is built from. The maps are persistent (`imbl`): a
/// round clones them under the lock in O(1) and builds its snapshot after
/// the lock is released, so the engine's appends never wait on it.
type Map<K, V> = imbl::HashMap<K, V>;

/// Names this host's round implementation in its stats, so results from
/// before and after a change to it are told apart.
pub const HOST_BUILD: &str = "lazy-snapshot";
/// A subject's hints or attributes: small, shared with a snapshot until the
/// next change copies them.
type Named = Arc<HashMap<String, Value>>;

#[derive(Clone)]
struct Parts {
    procs: Map<String, Proc>,
    workers: Map<String, snap::Worker>,
    sets: Map<String, snap::WorkingSet>,
    hints: Map<Subject, Named>,
    attrs: Map<Subject, Named>,
}

#[derive(Default)]
struct State {
    procs: Map<String, Proc>,
    workers: Map<String, snap::Worker>,
    sets: Map<String, snap::WorkingSet>,
    hints: Map<Subject, Named>,
    attrs: Map<Subject, Named>,
    events: Vec<wit::Event>,
    /// Columns with a row newly refused: their writers hear `refused`.
    refused: Vec<&'static str>,
    /// Sessions closed since the last round: their notes go after the
    /// round that hears `closed`.
    closed: Vec<String>,
    /// Processes that exited since the last round: that round sees them
    /// and reaps them, so it needs no scan of every process.
    exited: Vec<String>,
    kicked: bool,
    stop: bool,
    /// The sequence number of the last event reported (`Host::reported`).
    seq: u64,
}

/// A running process's counts at a step: (id, prompt tokens, cached tokens,
/// generated tokens), each `None` where the engine has not counted it.
pub type Counter = (String, Option<u32>, Option<u32>, Option<u32>);

/// One column's changes since the last poll.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDelta<K, V> {
    /// A key and its row now; `None`: no row (the default applies).
    pub rows: Vec<(K, Option<V>)>,
    /// `Some(v)` if the default changed: `Some(None)` is the column's own.
    pub default: Option<Option<V>>,
}

impl<K, V> ColumnDelta<K, V> {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.default.is_none()
    }
}

/// What changed in the tables since the last poll (`Host::poll_delta`).
#[derive(Debug, Clone)]
pub struct Delta {
    pub schedule_priority: ColumnDelta<String, i64>,
    pub schedule_hold: ColumnDelta<String, bool>,
    pub retention_priority: ColumnDelta<Prefix, i64>,
    pub retention_tiers: ColumnDelta<Prefix, TierSet>,
    pub placement_workers: ColumnDelta<String, Vec<String>>,
    pub placement_helpers: ColumnDelta<String, Vec<String>>,
    pub replication_workers: ColumnDelta<Prefix, Vec<String>>,
    /// pie's extension: group -> tenants, sorted.
    pub share_groups: ColumnDelta<String, Vec<String>>,
    /// Processes to end, and why.
    pub terminate: Vec<(String, String)>,
    /// Notes written for the other scope: (writer, subject, name, value).
    pub notes: Vec<(String, Subject, String, Option<Value>)>,
}

#[derive(Default, Clone)]
pub struct Stats {
    pub rounds: u64,
    pub calls: u64,
    pub traps: u64,
    pub deadlines: u64,
    pub withdrawn: u64,
    pub call_ns: u64,
    pub max_call_ns: u64,
    pub commits: u64,
    pub last_fault: Option<String>,
}

struct Shared {
    config: Config,
    start: Instant,
    manual_now: AtomicU64,
    state: Mutex<State>,
    wake: Condvar,
    tables: Arc<Mutex<Tables>>,
    programs: Mutex<Vec<Program>>,
    /// Events for the next round that one policy must not hear (its own
    /// note): (the policy's index, the event).
    carry: Mutex<Vec<(usize, wit::Event)>>,
    /// Bumped by every commit that changed something a port enacts.
    version: AtomicU64,
    /// The last event sequence number a completed round has taken
    /// (`Host::heard`).
    heard: AtomicU64,
    /// Events are waiting for a round.
    pending: AtomicBool,
    /// The earliest alarm, or `u64::MAX`.
    alarm_at: AtomicU64,
    stats: Mutex<Stats>,
}

pub struct Host {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn state_of(name: &str) -> Option<wit::ProcessState> {
    Some(match name {
        "pending" => wit::ProcessState::Pending,
        "waiting" => wit::ProcessState::Waiting,
        "running" => wit::ProcessState::Running,
        "paused" => wit::ProcessState::Paused,
        "exited" => wit::ProcessState::Exited,
        _ => return None,
    })
}

fn by_of(name: &str) -> wit::By {
    match name {
        "policy" => wit::By::Policy,
        "client" => wit::By::Client,
        _ => wit::By::Host,
    }
}

pub fn tier_of(name: &str) -> Option<wit::Tier> {
    Some(match name {
        "device" => wit::Tier::Device,
        "host" => wit::Tier::Host,
        "disk" => wit::Tier::Disk,
        _ => return None,
    })
}

fn tier_name(t: wit::Tier) -> &'static str {
    match t {
        wit::Tier::Device => "device",
        wit::Tier::Host => "host",
        wit::Tier::Disk => "disk",
    }
}

impl Shared {
    fn now(&self) -> u64 {
        if self.config.manual {
            self.manual_now.load(Ordering::Relaxed)
        } else {
            self.start.elapsed().as_nanos() as u64
        }
    }

    /// Queues an event. The first of a batch wakes the policy thread, which
    /// then waits a moment for the engine's counters.
    /// Returns the event's sequence number.
    fn push(&self, state: &mut State, event: wit::Event) -> u64 {
        state.events.push(event);
        state.seq += 1;
        if !self.pending.swap(true, Ordering::Release) {
            self.wake.notify_one();
        }
        state.seq
    }

    fn move_to(&self, proc: &mut Proc, to: wit::ProcessState, now: u64) {
        let spent = now.saturating_sub(proc.since);
        if proc.rec.state == wit::ProcessState::Running {
            proc.rec.service += spent;
        } else {
            proc.rec.waited += spent;
        }
        proc.since = proc.since.max(now);
        proc.rec.state = to;
    }

    /// The state's maps as they are now: O(1), for under the lock.
    fn parts(state: &State) -> Parts {
        Parts {
            procs: state.procs.clone(),
            workers: state.workers.clone(),
            sets: state.sets.clone(),
            hints: state.hints.clone(),
            attrs: state.attrs.clone(),
        }
    }

    /// A round's snapshot: the maps as `parts` took them, read lazily.
    fn snapshot(parts: Parts, now: u64) -> Snapshot {
        Snapshot::new(
            now,
            parts.procs,
            parts.workers,
            parts.sets,
            parts.hints,
            parts.attrs,
        )
    }

    /// One round: every policy that has something to hear is called once,
    /// on one snapshot, and each call's writes are committed as it returns.
    fn round(&self, force: bool) -> bool {
        let now = self.now();
        let (events, parts, refused, exited, closed, upto) = {
            let mut state = self.state.lock().unwrap();
            let upto = state.seq;
            state.kicked = false;
            self.pending.store(false, Ordering::Release);
            let events = std::mem::take(&mut state.events);
            let refused = std::mem::take(&mut state.refused);
            let parts = Self::parts(&state);
            // An exited process is seen by this round and no other (one
            // that arrived again under the same id since is live).
            let exited: Vec<String> = std::mem::take(&mut state.exited)
                .into_iter()
                .filter(|id| {
                    state
                        .procs
                        .get(id)
                        .is_some_and(|p| p.rec.state == wit::ProcessState::Exited)
                })
                .collect();
            for id in &exited {
                state.procs.remove(id);
                state.hints.remove(&Subject::Process(id.clone()));
                state.attrs.remove(&Subject::Process(id.clone()));
            }
            let closed = std::mem::take(&mut state.closed);
            (events, parts, refused, exited, closed, upto)
        };
        let snapshot = Arc::new(Self::snapshot(parts, now));
        let carried = std::mem::take(&mut *self.carry.lock().unwrap());
        let mut programs = self.programs.lock().unwrap();
        let mut changed = false;
        let mut noted: Vec<(usize, wit::Event)> = Vec::new();
        let mut called = false;
        for index in 0..programs.len() {
            let program = &mut programs[index];
            let mut heard = events.clone();
            heard.extend(
                carried
                    .iter()
                    .filter(|(writer, _)| *writer > index)
                    .map(|(_, e)| e.clone()),
            );
            heard.extend(noted.iter().map(|(_, e)| e.clone()));
            if refused.iter().any(|column| program.writes(column)) {
                heard.push(wit::Event::Refused);
            }
            if program.alarm.is_some_and(|at| at <= now) {
                program.alarm = None;
                heard.push(wit::Event::Alarm);
            }
            // A policy's first call may hear nothing: it plans from the
            // snapshot as it starts. So may a forced round's.
            if heard.is_empty() && !program.fresh && !force {
                continue;
            }
            program.fresh = false;
            called = true;
            let call = Call {
                policy: program.name.clone(),
                snapshot: snapshot.clone(),
                tables: self.tables.clone(),
                staged: Default::default(),
            };
            let started = Instant::now();
            let result = program.react(call, &heard, now, self.config.deadline_ms);
            let spent = started.elapsed().as_nanos() as u64;
            let mut stats = self.stats.lock().unwrap();
            stats.calls += 1;
            stats.call_ns += spent;
            stats.max_call_ns = stats.max_call_ns.max(spent);
            match result {
                Ok(staged) => {
                    program.failures = 0;
                    if let Some(at) = staged.alarm {
                        program.alarm = Some(at);
                    }
                    let mut tables = self.tables.lock().unwrap();
                    let before = tables.notes_out.len();
                    if tables.commit(&program.name, staged) {
                        changed = true;
                        stats.commits += 1;
                    }
                    // A note for the other scope is a change a port carries.
                    if tables.notes_out.len() > before {
                        changed = true;
                    }
                    for (_, subject, name, _) in &tables.notes_out[before..] {
                        noted.push((
                            index,
                            wit::Event::NoteChanged(wit::Change {
                                subject: wit_subject(subject),
                                name: name.clone(),
                            }),
                        ));
                    }
                }
                Err(fault) => {
                    program.failures += 1;
                    match &fault {
                        Fault::Deadline => stats.deadlines += 1,
                        Fault::Trap(_) => stats.traps += 1,
                    }
                    stats.last_fault = Some(format!("{}: {fault:?}", program.name));
                    // A fresh instance takes the next call, and is called
                    // soon whether or not anything happens: a policy that
                    // traps once recovers at once, and one that keeps
                    // failing runs out its lease in a few milliseconds
                    // rather than waiting for events that may never come.
                    program.fresh = true;
                    let retry = now + RETRY_NS * (1 << program.failures.min(6)) as u64;
                    program.alarm = Some(program.alarm.map_or(retry, |at| at.min(retry)));
                    if program.failures >= self.config.lease_calls {
                        // The lease ran out: the columns' own defaults.
                        let mut tables = self.tables.lock().unwrap();
                        if tables.withdraw(&program.imports) {
                            changed = true;
                            stats.withdrawn += 1;
                        }
                        // Called again at its next event, as a fresh
                        // instance: one that recovers writes its rows anew.
                        program.alarm = None;
                    }
                }
            }
        }
        // A note reaches the policies after its writer in this round, and
        // the ones before it in the next.
        noted.retain(|(writer, _)| *writer > 0);
        if !noted.is_empty() {
            self.carry.lock().unwrap().extend(noted);
            self.pending.store(true, Ordering::Release);
        }
        {
            let mut tables = self.tables.lock().unwrap();
            for id in &exited {
                changed |= tables.reap_process(id);
            }
            for session in closed {
                tables.notes.remove(&Subject::Session(session));
            }
        }
        let alarm = programs
            .iter()
            .filter_map(|p| p.alarm)
            .min()
            .unwrap_or(u64::MAX);
        self.alarm_at.store(alarm, Ordering::Release);
        self.stats.lock().unwrap().rounds += called as u64;
        if changed {
            self.version.fetch_add(1, Ordering::Release);
        }
        // Every policy has now been called on what was reported up to
        // `upto` (or had nothing to hear), its writes committed: a gate
        // waiting on an arrival can open.
        self.heard.fetch_max(upto, Ordering::Release);
        changed
    }

    fn serve(self: Arc<Self>) {
        let interval = Duration::from_millis(self.config.interval_ms);
        let mut last: Option<Instant> = None;
        loop {
            let mut state = self.state.lock().unwrap();
            let mut waited_since: Option<Instant> = None;
            loop {
                if state.stop {
                    return;
                }
                let now = self.now();
                let alarm = self.alarm_at.load(Ordering::Acquire);
                let alarm_due = alarm != u64::MAX && now >= alarm;
                let pending = self.pending.load(Ordering::Acquire);
                if state.kicked && (pending || alarm_due) {
                    break;
                }
                if pending || alarm_due {
                    // Something to hear: give the engine a moment to send
                    // its counters, then go on without them.
                    let since = *waited_since.get_or_insert_with(Instant::now);
                    let left = GRACE.saturating_sub(since.elapsed());
                    if left.is_zero() {
                        break;
                    }
                    state = self.wake.wait_timeout(state, left).unwrap().0;
                    continue;
                }
                state.kicked = false;
                let sleep = if alarm == u64::MAX {
                    Duration::from_secs(3600)
                } else {
                    Duration::from_nanos(alarm - now)
                };
                state = self.wake.wait_timeout(state, sleep).unwrap().0;
            }
            drop(state);
            if let Some(last) = last
                && last.elapsed() < interval
            {
                std::thread::sleep(interval - last.elapsed());
            }
            last = Some(Instant::now());
            self.round(false);
        }
    }
}

impl Host {
    pub fn new(config: Config) -> Host {
        let shared = Arc::new(Shared {
            config,
            start: Instant::now(),
            manual_now: AtomicU64::new(0),
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            tables: Arc::new(Mutex::new(Tables::default())),
            programs: Mutex::new(Vec::new()),
            carry: Mutex::new(Vec::new()),
            version: AtomicU64::new(0),
            heard: AtomicU64::new(0),
            pending: AtomicBool::new(false),
            alarm_at: AtomicU64::new(u64::MAX),
            stats: Mutex::new(Stats::default()),
        });
        Host {
            shared,
            thread: None,
        }
    }

    /// Loads a policy. Its imports are what it may write: a column this
    /// host cannot enact, or one another policy writes, refuses the load.
    pub fn load(&self, name: &str, wasm: &[u8]) -> Result<Vec<String>, String> {
        let program = Program::load(name, wasm)?;
        let mut programs = self.shared.programs.lock().unwrap();
        if programs.iter().any(|p| p.name == name) {
            return Err(format!("a policy named {name} is loaded"));
        }
        for import in &program.imports {
            if !COLUMNS.contains(&import.as_str()) {
                continue;
            }
            if !self.shared.config.supported.iter().any(|c| c == import) {
                return Err(format!(
                    "{name} writes {import}, which this host cannot enact"
                ));
            }
            if let Some(other) = programs.iter().find(|p| p.writes(import)) {
                return Err(format!("{name} and {} both write {import}", other.name));
            }
        }
        let imports = program.imports.clone();
        programs.push(program);
        drop(programs);
        // Its first call: a policy plans from the snapshot when it starts.
        let mut state = self.shared.state.lock().unwrap();
        self.shared.pending.store(true, Ordering::Release);
        state.kicked = true;
        self.shared.wake.notify_one();
        Ok(imports)
    }

    pub fn unload(&self, name: &str) -> bool {
        let mut programs = self.shared.programs.lock().unwrap();
        let Some(index) = programs.iter().position(|p| p.name == name) else {
            return false;
        };
        let program = programs.remove(index);
        if self
            .shared
            .tables
            .lock()
            .unwrap()
            .withdraw(&program.imports)
        {
            self.shared.version.fetch_add(1, Ordering::Release);
        }
        true
    }

    pub fn start(&mut self) {
        if self.thread.is_none() && !self.shared.config.manual {
            let shared = self.shared.clone();
            self.thread = Some(
                std::thread::Builder::new()
                    .name("plex-policy".into())
                    .spawn(move || shared.serve())
                    .expect("spawn the policy thread"),
            );
        }
    }

    pub fn stop(&mut self) {
        self.shared.state.lock().unwrap().stop = true;
        self.shared.wake.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    // Time.

    pub fn now(&self) -> u64 {
        self.shared.now()
    }

    pub fn set_now(&self, now: u64) {
        self.shared.manual_now.store(now, Ordering::Relaxed);
    }

    /// Runs a round now, if there is anything to hear: for a host without
    /// a policy thread (`manual`).
    pub fn run(&self, force: bool) -> bool {
        let alarm = self.shared.alarm_at.load(Ordering::Acquire);
        if force
            || self.shared.pending.load(Ordering::Acquire)
            || (alarm != u64::MAX && self.shared.now() >= alarm)
        {
            self.shared.round(force)
        } else {
            false
        }
    }

    /// The earliest time a policy asked to be called again.
    pub fn alarm(&self) -> Option<u64> {
        match self.shared.alarm_at.load(Ordering::Acquire) {
            u64::MAX => None,
            at => Some(at),
        }
    }

    // What the engine's thread asks on every step: two atomic reads.

    /// Whether a round is waiting on the engine's counters.
    pub fn due(&self) -> bool {
        if self.shared.pending.load(Ordering::Acquire) {
            return true;
        }
        let alarm = self.shared.alarm_at.load(Ordering::Acquire);
        alarm != u64::MAX && self.shared.now() >= alarm
    }

    pub fn version(&self) -> u64 {
        self.shared.version.load(Ordering::Acquire)
    }

    /// The sequence number of the last event reported. `arrived` returns
    /// its own.
    pub fn reported(&self) -> u64 {
        self.shared.state.lock().unwrap().seq
    }

    /// The last sequence number every policy has heard: the events up to
    /// it were taken by a round that has ended, with every policy called
    /// on them (a call that trapped heard them too) and its writes
    /// committed. A gate that must not dispatch before the policies have
    /// seen an arrival waits until `heard() >= arrived(..)`.
    pub fn heard(&self) -> u64 {
        self.shared.heard.load(Ordering::Acquire)
    }

    // What the engine reports.

    #[allow(clippy::too_many_arguments)]
    pub fn arrived(
        &self,
        id: &str,
        session: Option<String>,
        tenant: Option<String>,
        worker: Option<String>,
        prompt_tokens: Option<u32>,
        max_tokens: Option<u32>,
        hints: Vec<(String, Value)>,
        age_ns: u64,
    ) -> u64 {
        let now = self.shared.now();
        let arrival = now.saturating_sub(age_ns);
        let mut state = self.shared.state.lock().unwrap();
        let rec = snap::Process {
            id: id.to_string(),
            session,
            tenant,
            state: if worker.is_some() {
                wit::ProcessState::Waiting
            } else {
                wit::ProcessState::Pending
            },
            worker,
            arrival,
            first_token: None,
            service: 0,
            waited: 0,
            prompt_tokens,
            cached_tokens: None,
            generated_tokens: 0,
            max_tokens,
        };
        state.procs.insert(
            id.to_string(),
            Proc {
                rec,
                since: arrival,
            },
        );
        if !hints.is_empty() {
            Arc::make_mut(
                state
                    .hints
                    .entry(Subject::Process(id.to_string()))
                    .or_default(),
            )
            .extend(hints);
        }
        self.shared
            .push(&mut state, wit::Event::Arrived(id.to_string()))
    }

    pub fn dispatched(&self, id: &str, worker: &str) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        let Some(proc) = state.procs.get_mut(id) else {
            return;
        };
        proc.rec.worker = Some(worker.to_string());
        if proc.rec.state == wit::ProcessState::Pending {
            self.shared.move_to(proc, wit::ProcessState::Waiting, now);
        }
        self.shared.push(
            &mut state,
            wit::Event::Dispatched(wit::Dispatch {
                process: id.to_string(),
                worker: worker.to_string(),
            }),
        );
    }

    /// Started or resumed.
    pub fn started(&self, id: &str) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        let Some(proc) = state.procs.get_mut(id) else {
            return;
        };
        if proc.rec.state == wit::ProcessState::Running {
            return;
        }
        self.shared.move_to(proc, wit::ProcessState::Running, now);
        self.shared
            .push(&mut state, wit::Event::Started(id.to_string()));
    }

    pub fn paused(&self, id: &str, by: &str) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        let Some(proc) = state.procs.get_mut(id) else {
            return;
        };
        if proc.rec.state == wit::ProcessState::Paused {
            return;
        }
        self.shared.move_to(proc, wit::ProcessState::Paused, now);
        self.shared.push(
            &mut state,
            wit::Event::Paused(wit::Pause {
                process: id.to_string(),
                by: by_of(by),
            }),
        );
    }

    pub fn first_token(&self, id: &str) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        if let Some(proc) = state.procs.get_mut(id)
            && proc.rec.first_token.is_none()
        {
            proc.rec.first_token = Some(now);
        }
    }

    pub fn exited(
        &self,
        id: &str,
        outcome: &str,
        prompt_tokens: Option<u32>,
        cached_tokens: Option<u32>,
        generated_tokens: u32,
    ) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        let Some(proc) = state.procs.get_mut(id) else {
            return;
        };
        if proc.rec.state == wit::ProcessState::Exited {
            return;
        }
        self.shared.move_to(proc, wit::ProcessState::Exited, now);
        state.exited.push(id.to_string());
        let Some(proc) = state.procs.get_mut(id) else {
            return;
        };
        proc.rec.prompt_tokens = prompt_tokens.or(proc.rec.prompt_tokens);
        proc.rec.cached_tokens = cached_tokens.or(proc.rec.cached_tokens);
        proc.rec.generated_tokens = generated_tokens.max(proc.rec.generated_tokens);
        if proc.rec.first_token.is_none() && generated_tokens > 0 {
            proc.rec.first_token = Some(now);
        }
        let outcome = match outcome {
            "completed" => wit::Outcome::Completed,
            "cancelled" => wit::Outcome::Cancelled,
            _ => wit::Outcome::Failed,
        };
        self.shared.push(
            &mut state,
            wit::Event::Exited(wit::Exit {
                process: id.to_string(),
                outcome,
            }),
        );
    }

    /// The counters of live processes, as of now: (id, prompt tokens,
    /// cached tokens, generated tokens). Wakes the policy thread.
    pub fn tick(&self, counters: Vec<Counter>) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        for (id, prompt, cached, generated) in counters {
            if let Some(proc) = state.procs.get_mut(&id) {
                proc.rec.prompt_tokens = prompt.or(proc.rec.prompt_tokens);
                proc.rec.cached_tokens = cached.or(proc.rec.cached_tokens);
                if let Some(generated) = generated {
                    if generated > 0 && proc.rec.first_token.is_none() {
                        proc.rec.first_token = Some(now);
                    }
                    proc.rec.generated_tokens = generated;
                }
            }
        }
        state.kicked = true;
        self.shared.wake.notify_one();
    }

    /// Wakes the policy thread without new counters.
    pub fn kick(&self) {
        self.shared.state.lock().unwrap().kicked = true;
        self.shared.wake.notify_one();
    }

    pub fn closed(&self, session: &str) {
        let mut state = self.shared.state.lock().unwrap();
        state.hints.remove(&Subject::Session(session.to_string()));
        state.closed.push(session.to_string());
        self.shared
            .push(&mut state, wit::Event::Closed(session.to_string()));
    }

    /// A client set a hint (some) or removed it (none).
    pub fn hint(&self, subject: Subject, name: &str, value: Option<Value>) {
        let mut state = self.shared.state.lock().unwrap();
        let changed = match value {
            Some(v) => {
                Arc::make_mut(state.hints.entry(subject.clone()).or_default())
                    .insert(name.to_string(), v.clone())
                    != Some(v)
            }
            None => state
                .hints
                .get_mut(&subject)
                .is_some_and(|h| Arc::make_mut(h).remove(name).is_some()),
        };
        if changed {
            self.shared.push(
                &mut state,
                wit::Event::HintChanged(wit::Change {
                    subject: wit_subject(&subject),
                    name: name.to_string(),
                }),
            );
        }
    }

    /// A measured value beyond the records (`get-attr`). No event.
    pub fn attr(&self, subject: Subject, name: &str, value: Option<Value>) {
        let mut state = self.shared.state.lock().unwrap();
        match value {
            Some(v) => {
                Arc::make_mut(state.attrs.entry(subject).or_default()).insert(name.to_string(), v);
            }
            None => {
                if let Some(attrs) = state.attrs.get_mut(&subject) {
                    Arc::make_mut(attrs).remove(name);
                }
            }
        }
    }

    /// A note from the other scope's policies.
    pub fn note(&self, writer: &str, subject: Subject, name: &str, value: Option<Value>) {
        let mut tables = self.shared.tables.lock().unwrap();
        let key = (writer.to_string(), name.to_string());
        let changed = match value {
            Some(v) => {
                tables
                    .notes
                    .entry(subject.clone())
                    .or_default()
                    .insert(key, v.clone())
                    != Some(v)
            }
            None => tables
                .notes
                .get_mut(&subject)
                .is_some_and(|n| n.remove(&key).is_some()),
        };
        drop(tables);
        if changed {
            let mut state = self.shared.state.lock().unwrap();
            self.shared.push(
                &mut state,
                wit::Event::NoteChanged(wit::Change {
                    subject: wit_subject(&subject),
                    name: name.to_string(),
                }),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn put_worker(
        &self,
        id: &str,
        draining: bool,
        tiers: Vec<(wit::Tier, Option<u64>, u64)>,
        running: u32,
        waiting: u32,
        step_ns: Option<u64>,
    ) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        let rec = snap::Worker {
            id: id.to_string(),
            draining,
            tiers: tiers
                .into_iter()
                .map(|(tier, capacity_tokens, used_tokens)| snap::Room {
                    tier,
                    capacity_tokens,
                    used_tokens,
                })
                .collect(),
            running,
            waiting,
            step: step_ns,
            as_of: now,
        };
        let news = state
            .workers
            .get(id)
            .is_none_or(|old| old.draining != draining);
        state.workers.insert(id.to_string(), rec);
        if news {
            self.shared
                .push(&mut state, wit::Event::WorkerChanged(id.to_string()));
        }
    }

    pub fn remove_worker(&self, id: &str) {
        let mut state = self.shared.state.lock().unwrap();
        if state.workers.remove(id).is_some() {
            state.hints.remove(&Subject::Worker(id.to_string()));
            self.shared
                .push(&mut state, wit::Event::WorkerChanged(id.to_string()));
        }
    }

    /// A working set as it is now. `idle_ns` is how long since its last
    /// use. No event: `retained` and `evicted` are reported apart.
    #[allow(clippy::too_many_arguments)]
    pub fn put_working_set(
        &self,
        id: &str,
        parent: Option<String>,
        owner: Option<String>,
        session: Option<String>,
        tokens: u32,
        idle_ns: u64,
        copies: Vec<(String, wit::Tier, u32)>,
    ) {
        let now = self.shared.now();
        let mut state = self.shared.state.lock().unwrap();
        state.sets.insert(
            id.to_string(),
            snap::WorkingSet {
                id: id.to_string(),
                parent,
                owner,
                session,
                tokens,
                last_use: now.saturating_sub(idle_ns),
                copies: copies
                    .into_iter()
                    .map(|(worker, tier, tokens)| snap::Replica {
                        worker,
                        tier,
                        tokens,
                    })
                    .collect(),
            },
        );
    }

    pub fn retained(&self, working_set: &str, worker: &str) {
        let mut state = self.shared.state.lock().unwrap();
        if let Some(set) = state.sets.get_mut(working_set) {
            set.owner = None;
        }
        self.shared.push(
            &mut state,
            wit::Event::Retained(wit::Retention {
                working_set: working_set.to_string(),
                worker: worker.to_string(),
            }),
        );
    }

    /// The last of a working set left `worker`. With `gone`, it is nowhere
    /// any more: its rows and notes go with it.
    pub fn evicted(&self, working_set: &str, worker: &str, by: &str, gone: bool) {
        let mut state = self.shared.state.lock().unwrap();
        if gone {
            state.sets.remove(working_set);
            state
                .hints
                .remove(&Subject::WorkingSet(working_set.to_string()));
            state
                .attrs
                .remove(&Subject::WorkingSet(working_set.to_string()));
            if self
                .shared
                .tables
                .lock()
                .unwrap()
                .reap_working_set(working_set)
            {
                self.shared.version.fetch_add(1, Ordering::Release);
            }
        }
        self.shared.push(
            &mut state,
            wit::Event::Evicted(wit::Eviction {
                working_set: working_set.to_string(),
                worker: worker.to_string(),
                by: by_of(by),
            }),
        );
    }

    /// The port cannot enact a row (some), or now can (none). `key` is a
    /// process id, or a working set and its token count.
    pub fn refuse(
        &self,
        column: &str,
        key: &str,
        tokens: Option<u32>,
        why: Option<Refusal>,
    ) -> bool {
        fn mark<K: std::hash::Hash + Eq + Clone, V>(
            column: &mut Column<K, V>,
            key: K,
            why: Option<Refusal>,
        ) -> bool {
            match why {
                Some(why) if column.rows.contains_key(&key) => {
                    column.refused.insert(key, why) != Some(why)
                }
                Some(_) => false,
                None => {
                    column.refused.remove(&key);
                    false
                }
            }
        }
        let Some(name) = COLUMNS.iter().find(|c| **c == column) else {
            return false;
        };
        let mut tables = self.shared.tables.lock().unwrap();
        let prefix: Prefix = (key.to_string(), tokens);
        let news = match *name {
            tables::SCHEDULE_PRIORITY => mark(&mut tables.schedule_priority, key.to_string(), why),
            tables::SCHEDULE_HOLD => mark(&mut tables.schedule_hold, key.to_string(), why),
            tables::RETENTION_PRIORITY => mark(&mut tables.retention_priority, prefix, why),
            tables::RETENTION_TIERS => mark(&mut tables.retention_tiers, prefix, why),
            tables::PLACEMENT_WORKERS => mark(&mut tables.placement_workers, key.to_string(), why),
            tables::PLACEMENT_HELPERS => mark(&mut tables.placement_helpers, key.to_string(), why),
            tables::REPLICATION_WORKERS => mark(&mut tables.replication_workers, prefix, why),
            _ => false,
        };
        drop(tables);
        if news {
            let mut state = self.shared.state.lock().unwrap();
            state.refused.push(name);
            if !self.shared.pending.swap(true, Ordering::Release) {
                self.shared.wake.notify_one();
            }
        }
        news
    }

    // What the port enacts.

    /// What changed in the tables since the last `poll`, as JSON: for each
    /// column `{"rows": [[key, value-or-null], ...], "default": [value]}`
    /// (`default` only if it changed; a prefix key is `[working-set,
    /// tokens-or-null]`), `terminate` as `[[process, reason]]`, and `notes`
    /// as `[[writer, subject-kind, subject-id, name, value-or-null]]`.
    pub fn poll(&self) -> Json {
        let d = self.poll_delta();
        fn column<K, V>(
            out: &mut serde_json::Map<String, Json>,
            name: &str,
            c: ColumnDelta<K, V>,
            key: impl Fn(&K) -> Json,
            value: impl Fn(&V) -> Json,
        ) {
            if c.rows.is_empty() && c.default.is_none() {
                return;
            }
            let rows: Vec<Json> = c
                .rows
                .iter()
                .map(|(k, v)| json!([key(k), v.as_ref().map(&value)]))
                .collect();
            let mut entry = json!({ "rows": rows });
            if let Some(default) = c.default {
                entry["default"] = json!([default.as_ref().map(&value)]);
            }
            out.insert(name.to_string(), entry);
        }
        let id = |k: &String| json!(k);
        let prefix = |k: &Prefix| json!([k.0, k.1]);
        let tiers = |v: &TierSet| {
            json!(
                [wit::Tier::Device, wit::Tier::Host, wit::Tier::Disk]
                    .into_iter()
                    .filter(|t| v & crate::runtime::tier_bit(*t) != 0)
                    .map(tier_name)
                    .collect::<Vec<_>>()
            )
        };
        let mut out = serde_json::Map::new();
        column(
            &mut out,
            tables::SCHEDULE_PRIORITY,
            d.schedule_priority,
            id,
            |v| json!(v),
        );
        column(&mut out, tables::SCHEDULE_HOLD, d.schedule_hold, id, |v| {
            json!(v)
        });
        column(
            &mut out,
            tables::RETENTION_PRIORITY,
            d.retention_priority,
            prefix,
            |v| json!(v),
        );
        column(
            &mut out,
            tables::RETENTION_TIERS,
            d.retention_tiers,
            prefix,
            tiers,
        );
        column(
            &mut out,
            tables::PLACEMENT_WORKERS,
            d.placement_workers,
            id,
            |v| json!(v),
        );
        column(
            &mut out,
            tables::PLACEMENT_HELPERS,
            d.placement_helpers,
            id,
            |v| json!(v),
        );
        column(
            &mut out,
            tables::REPLICATION_WORKERS,
            d.replication_workers,
            prefix,
            |v| json!(v),
        );
        column(&mut out, tables::SHARE_GROUPS, d.share_groups, id, |v| json!(v));
        if !d.terminate.is_empty() {
            out.insert("terminate".into(), json!(d.terminate));
        }
        if !d.notes.is_empty() {
            let notes: Vec<Json> = d
                .notes
                .into_iter()
                .map(|(writer, subject, name, value)| {
                    json!([
                        writer,
                        subject.kind(),
                        subject.id(),
                        name,
                        value.as_ref().map(value_json)
                    ])
                })
                .collect();
            out.insert("notes".into(), Json::Array(notes));
        }
        Json::Object(out)
    }

    /// What changed in the tables since the last `poll` or `poll_delta`,
    /// typed: per column the rows that changed (`None`: no row now) and
    /// the default if it changed (`Some(None)`: the column's own again),
    /// then the processes to terminate and the notes for the other scope.
    /// Coalesced: a port enacts only the difference.
    pub fn poll_delta(&self) -> Delta {
        fn take<K: Clone + Eq + std::hash::Hash, V: Clone>(
            c: &mut Column<K, V>,
        ) -> ColumnDelta<K, V> {
            let rows = c.changed.drain().collect();
            let default = std::mem::take(&mut c.default_changed).then(|| c.default.clone());
            ColumnDelta { rows, default }
        }
        let mut tables = self.shared.tables.lock().unwrap();
        let t = &mut *tables;
        Delta {
            schedule_priority: take(&mut t.schedule_priority),
            schedule_hold: take(&mut t.schedule_hold),
            retention_priority: take(&mut t.retention_priority),
            retention_tiers: take(&mut t.retention_tiers),
            placement_workers: take(&mut t.placement_workers),
            placement_helpers: take(&mut t.placement_helpers),
            replication_workers: take(&mut t.replication_workers),
            share_groups: take(&mut t.share_groups),
            terminate: std::mem::take(&mut t.terminate),
            notes: std::mem::take(&mut t.notes_out),
        }
    }

    pub fn stats(&self) -> Stats {
        self.shared.stats.lock().unwrap().clone()
    }

    pub fn policies(&self) -> Vec<(String, Vec<String>)> {
        self.shared
            .programs
            .lock()
            .unwrap()
            .iter()
            .map(|p| (p.name.clone(), p.imports.clone()))
            .collect()
    }

    /// Drops every policy's instance, as a trap would: for fault tests.
    pub fn reset(&self) {
        for program in self.shared.programs.lock().unwrap().iter_mut() {
            program.reset();
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn value_json(v: &Value) -> Json {
    match v {
        Value::Boolean(b) => json!(b),
        Value::Int(i) => json!(i),
        Value::Float(f) => json!(f),
        Value::Text(t) => json!(t),
    }
}

pub fn parse_state(name: &str) -> Option<wit::ProcessState> {
    state_of(name)
}

#[cfg(test)]
mod lock_time {
    use super::*;

    /// How long a round's snapshot holds the state lock, by size.
    #[test]
    #[ignore]
    fn snapshot_lock_time() {
        for &n in &[100usize, 1000, 5000] {
            let host = Host::new(Config {
                manual: true,
                ..Default::default()
            });
            for i in 0..n {
                host.arrived(
                    &format!("p{i}"),
                    Some(format!("s{}", i / 4)),
                    Some(format!("t{}", i % 8)),
                    Some("w0".into()),
                    Some(4000),
                    Some(512),
                    vec![
                        ("tool".into(), Value::Text("shell".into())),
                        ("deadline-ms".into(), Value::Int(3000)),
                    ],
                    0,
                );
                host.put_working_set(
                    &format!("ws{i}"),
                    None,
                    Some(format!("p{i}")),
                    Some(format!("s{}", i / 4)),
                    4000,
                    0,
                    vec![("w0".into(), wit::Tier::Device, 4000)],
                );
            }
            let reps = 50;
            let (mut held, mut built) = (0f64, 0f64);
            for _ in 0..reps {
                let t = Instant::now();
                let state = host.shared.state.lock().unwrap();
                let parts = Shared::parts(&state);
                drop(state);
                held += t.elapsed().as_secs_f64();
                let t = Instant::now();
                // The lazy snapshot, then a policy reading every process.
                let snapshot = Shared::snapshot(parts.clone(), 1);
                std::hint::black_box(snapshot.processes().len());
                built += t.elapsed().as_secs_f64();
            }
            println!(
                "n={n}: {:.2} us under the lock, {:.0} us to build after it",
                held * 1e6 / reps as f64,
                built * 1e6 / reps as f64
            );
        }
    }
}

#[cfg(test)]
mod append_time {
    use super::*;

    /// What the engine's thread pays per report, with n processes live.
    #[test]
    #[ignore]
    fn append_cost() {
        for &n in &[100usize, 1000, 5000] {
            let host = Host::new(Config {
                manual: true,
                ..Default::default()
            });
            let t = Instant::now();
            for i in 0..n {
                host.arrived(
                    &format!("p{i}"),
                    Some(format!("s{}", i / 4)),
                    None,
                    Some("w0".into()),
                    Some(4000),
                    Some(512),
                    vec![("tool".into(), Value::Text("shell".into()))],
                    0,
                );
            }
            let arrive = t.elapsed().as_secs_f64() * 1e9 / n as f64;
            let t = Instant::now();
            for i in 0..n {
                host.started(&format!("p{i}"));
                host.hint(
                    Subject::Process(format!("p{i}")),
                    "x",
                    Some(Value::Int(i as i64)),
                );
            }
            let other = t.elapsed().as_secs_f64() * 1e9 / (2 * n) as f64;
            println!("n={n}: arrived {arrive:.0} ns, started/hint {other:.0} ns");
        }
    }
}
