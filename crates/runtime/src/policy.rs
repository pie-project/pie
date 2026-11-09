//! The worker plane of PLEX (`plex:policy@1.0.0`): when a worker's passes run,
//! and where its state sits within it.
//!
//! A process's PLEX id is the id its client knows: one restarted after a KV
//! reclaim keeps it. The browser build runs no policy: the runtime still
//! reports here, and nothing listens.
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

pub mod offload;
pub mod pressure;
pub mod retention;
pub mod schedule;

#[cfg(not(target_arch = "wasm32"))]
mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod program;
#[cfg(not(target_arch = "wasm32"))]
mod service;

#[cfg(not(target_arch = "wasm32"))]
pub use manifest::{Manifest, ManifestError};
#[cfg(not(target_arch = "wasm32"))]
pub use program::{Commit, Fault, LoadError, Offloads, Plane, Program, Retention, Table};
#[cfg(not(target_arch = "wasm32"))]
pub use service::{reload, start, start_default};

use client_api::Props;

use crate::inferlet::process::ProcessId;

#[cfg(not(target_arch = "wasm32"))]
pub mod wit {
    pub use super::program::bindings::plex::policy::types::*;
}

pub const CAPABILITIES: &[&str] = &[
    "schedule.admit",
    "schedule.preempt",
    "schedule.preempt-for-admit",
    "schedule.resume",
    "schedule.floor",
    "retention.order",
    "retention.tier",
    "offload.remote",
    "publish",
];

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
}

/// A resting handle, for the worker policy to read.
#[derive(Debug, Clone)]
pub(crate) struct Resting {
    pub client: crate::server::ClientId,
    pub tenant: String,
    pub session: Option<String>,
    pub props: Props,
    pub tokens: u64,
    pub folded: u64,
    pub slots: u32,
}

/// Each is a no-op when no policy is loaded.
#[derive(Debug, Clone)]
pub(crate) enum Event {
    Arrived(ProcessId, Arrival),
    PropsChanged(ProcessId, Props, Vec<String>),
    /// One of its fires found this many tokens cached and mapped them in.
    Cached(ProcessId, u64),
    Kept(ProcessId, u64),
    CacheEmptied(ProcessId, By),
    Started(ProcessId),
    Paused(ProcessId),
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
    /// Swap the policy for this one (name, component), or for none, loaded
    /// whole first: the answer is why it did not load.
    Reload(
        Option<(String, Vec<u8>)>,
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

pub(crate) fn report(event: Event) {
    #[cfg(not(target_arch = "wasm32"))]
    service::report(event);
    #[cfg(target_arch = "wasm32")]
    let _ = event;
}
