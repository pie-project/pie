//! The worker plane of PLEX 1.1 (`plex:policy@1.1.0`, the `plex-host`
//! crate): when a worker's passes run, and where its state sits within it.
//!
//! A process's PLEX id is the id its client knows: one restarted after a KV
//! reclaim keeps it. The browser build runs no policy: the runtime still
//! reports here, and nothing listens.
#![cfg_attr(
    target_arch = "wasm32",
    allow(
        dead_code,
        reason = "no policy runs in the browser: its reports are formed and dropped"
    )
)]

pub mod grants;
pub mod pressure;
pub mod retention;
pub mod schedule;
pub mod share;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod port;
#[cfg(test)]
mod seat_to_end_tests;

#[cfg(not(target_arch = "wasm32"))]
pub use port::{LoadError, reload, start};

/// Ends the policy thread and forgets what the boot's policy set, for
/// [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    #[cfg(not(target_arch = "wasm32"))]
    port::reset();
    pressure::reset();
    schedule::reset();
    for count in ROWS_ENACTED.iter().chain(&ROWS_PASSED_OVER) {
        count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    withdraw_published();
    *STARTS.lock().unwrap() = Starts::default();
    STOPS.lock().unwrap().clear();
}

use client_api::Props;

/// The worker columns a port enacts, as `/metrics` names them.
pub const COLUMNS: [&str; 4] = [
    "schedule/priority",
    "schedule/hold",
    "retention/priority",
    "retention/tiers",
];

/// Per column of [`COLUMNS`]: the rows enacted (set or cleared) since start.
pub static ROWS_ENACTED: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Per column of [`COLUMNS`]: the rows passed over, their key naming no
/// process this worker could hold.
pub static ROWS_PASSED_OVER: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

use crate::inferlet::process::ProcessId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum By {
    Policy,
    Host,
    Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Default)]
pub struct Arrival {
    /// The one its gateway authenticated, or [`crate::server::DEFAULT_TENANT`].
    pub tenant: String,
    pub session: Option<String>,
    pub props: Props,
    /// Set by the runtime alone, for a pass a peer offloaded here: the
    /// asking set's domain, which the pass keys under whatever scope it
    /// asks ([`share`]). Never from a client.
    pub keys_under: Option<crate::store::kv::hash::Hash256>,
}

/// A resting handle, for the worker policy to read.
#[derive(Debug, Clone)]
pub(crate) struct Resting {
    pub tenant: String,
    pub session: Option<String>,
    pub props: Props,
    pub tokens: u64,
    pub folded: u64,
}

/// Each is a no-op when no policy is loaded.
#[derive(Debug, Clone)]
pub(crate) enum Event {
    Arrived(ProcessId, Arrival),
    PropsChanged(ProcessId, Props, Vec<String>),
    /// One of its fires found this many tokens cached and mapped them in.
    Cached(ProcessId, u64),
    /// While it waits for admission: the tokens of its prompt's prefix the
    /// stores hold now ([`crate::store::waiting`]), in place of the last.
    Waiting(ProcessId, u64),
    /// One of its fires wrote `rows` rows and read out `readouts` of them:
    /// the rows before its first readout are its prompt, each readout a
    /// token it generates.
    Fired(ProcessId, u64, u64),
    CacheEmptied(ProcessId, By),
    Started(ProcessId),
    Paused(ProcessId),
    /// It waits on its client (`true`), or no longer does: paused by the
    /// client, then running again.
    Idle(ProcessId, bool),
    Resumed(ProcessId),
    Exited(ProcessId, Outcome),
    /// A pass of the process ran on another worker: `tokens` of it.
    Offloaded(ProcessId, String, u32),
    /// A pass meant for another worker ran here instead, and why.
    OffloadFailed(ProcessId, String),
    /// A working set a client holds came to rest, under an id it keeps for
    /// life.
    HandleRested(ProcessId, Resting),
    /// All its props, and which of them changed.
    HandleProps(ProcessId, Props, Vec<String>),
    /// A process has it now: out of the policy's reach until it rests again.
    HandleLent(ProcessId),
    HandleGone(ProcessId, By),
    /// The session id is `<tenant>/<session>`.
    SessionProps(String, Props),
    SessionClosed(String),
    /// Where a tier's waiters stand now ([`pressure::raise`]): sent when its
    /// shortfall changes, or stands a while. The policy hears an edge only
    /// when the band changes.
    TierPressure(pressure::Tier, pressure::Reading),
    /// Swap the policy for this one (name, component), loaded whole first:
    /// the answer is why it did not load.
    Reload(
        (String, Vec<u8>),
        std::sync::mpsc::SyncSender<Result<(), String>>,
    ),
}

/// What the worker policy published about this worker for the cluster plane,
/// by name.
static PUBLISHED: std::sync::LazyLock<
    std::sync::RwLock<std::collections::BTreeMap<String, serde_json::Value>>,
> = std::sync::LazyLock::new(Default::default);

/// What a policy that stopped published is withdrawn.
pub(crate) fn withdraw_published() {
    PUBLISHED.write().unwrap().clear();
}

pub(crate) fn publish(name: String, value: Option<serde_json::Value>) {
    let mut published = PUBLISHED.write().unwrap();
    match value {
        Some(value) => {
            published.insert(name, value);
        }
        None => {
            published.remove(&name);
        }
    }
}

/// Everything the worker policy has published, as `(name, value)`.
pub fn published() -> Vec<(String, serde_json::Value)> {
    PUBLISHED
        .read()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

pub fn session_props(session: String, changes: Props) {
    report(Event::SessionProps(session, changes));
}

pub fn session_closed(session: String) {
    report(Event::SessionClosed(session));
}

/// Who waits to hear a process start (a worker forwards it to its
/// gateway), and which have started: a start that comes before its waiter
/// is not lost.
#[derive(Default)]
struct Starts {
    waiting: std::collections::HashMap<ProcessId, Vec<tokio::sync::oneshot::Sender<()>>>,
    started: std::collections::HashSet<ProcessId>,
}

static STARTS: std::sync::LazyLock<std::sync::Mutex<Starts>> =
    std::sync::LazyLock::new(Default::default);

/// Resolves once `pid` has started (at once if it already has); never if
/// it exits first.
pub fn started(pid: ProcessId) -> tokio::sync::oneshot::Receiver<()> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut starts = STARTS.lock().unwrap();
    if starts.started.contains(&pid) {
        let _ = tx.send(());
    } else {
        starts.waiting.entry(pid).or_default().push(tx);
    }
    rx
}

/// How a fire refused because a policy stopped its process reads: this,
/// then the reason. Compat's `STOPPED`.
pub const STOPPED: &str = "stopped by policy: ";

/// Policy stops asked of running processes: the reason, and whether a fire
/// has refused for it yet.
static STOPS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<ProcessId, (String, bool)>>,
> = std::sync::LazyLock::new(Default::default);

/// A policy stops running `pid`: its next fire refuses with the reason, so
/// the program can end its reply; a fire after that ends the process.
pub(crate) fn stop(pid: ProcessId, reason: String) {
    STOPS.lock().unwrap().entry(pid).or_insert((reason, false));
}

/// At a fire of `pid`: `Some(refusal)` if a policy stopped it.
pub(crate) fn stopped(pid: ProcessId) -> Option<String> {
    let mut stops = STOPS.lock().unwrap();
    let (reason, told) = stops.get_mut(&pid)?;
    let refusal = format!("{STOPPED}{reason}");
    if *told {
        let reason = reason.clone();
        drop(stops);
        crate::inferlet::process::terminate_by(pid, Err(reason), By::Policy);
    } else {
        *told = true;
    }
    Some(refusal)
}

fn note_start(event: &Event) {
    match event {
        Event::Started(pid) => {
            let mut starts = STARTS.lock().unwrap();
            starts.started.insert(*pid);
            for tx in starts.waiting.remove(pid).into_iter().flatten() {
                let _ = tx.send(());
            }
        }
        Event::Exited(pid, _) => {
            let mut starts = STARTS.lock().unwrap();
            starts.started.remove(pid);
            starts.waiting.remove(pid);
            drop(starts);
            STOPS.lock().unwrap().remove(pid);
        }
        _ => {}
    }
}

pub(crate) fn report(event: Event) {
    note_start(&event);
    #[cfg(not(target_arch = "wasm32"))]
    port::report(event);
    #[cfg(target_arch = "wasm32")]
    let _ = event;
}
