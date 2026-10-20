//! The worker plane of PLEX (`plex:policy@1.0.0`): when a worker's passes run,
//! and where its state sits within it.
//!
//! A process's PLEX id is the id its client knows: one restarted after a KV
//! reclaim keeps it. The browser build runs no policy: the runtime still
//! reports here, and nothing listens.
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

pub mod offload;
pub mod retention;
pub mod schedule;
pub mod tokens;

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
pub use service::start;

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
    /// The resume log a move relaunched it with (`session.resumed`).
    pub resumed: Option<String>,
}

/// Each is a no-op when no policy is loaded.
#[derive(Debug, Clone)]
pub(crate) enum Event {
    Arrived(ProcessId, Arrival),
    PropsChanged(ProcessId, Props, Vec<String>),
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
    /// The session id is `<tenant>/<session>`.
    SessionProps(String, Props),
    SessionClosed(String),
}

/// What the worker policy published for the cluster plane, by subject
/// (`worker`, `process:<id>`) and name.
static PUBLISHED: std::sync::LazyLock<
    std::sync::RwLock<std::collections::BTreeMap<(String, String), serde_json::Value>>,
> = std::sync::LazyLock::new(Default::default);

pub(crate) fn publish(of: String, name: String, value: Option<serde_json::Value>) {
    let mut published = PUBLISHED.write().unwrap();
    match value {
        Some(value) => {
            published.insert((of, name), value);
        }
        None => {
            published.remove(&(of, name));
        }
    }
}

/// Everything the worker policy has published, as `(subject, name, value)`.
pub fn published() -> Vec<(String, String, serde_json::Value)> {
    PUBLISHED
        .read()
        .unwrap()
        .iter()
        .map(|((of, name), value)| (of.clone(), name.clone(), value.clone()))
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
