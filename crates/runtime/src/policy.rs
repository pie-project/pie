//! The engine plane of PLEX (`plex:policy@1.0.0`).
//!
//! A process's PLEX id is the id its client knows: one restarted after a KV
//! reclaim keeps it. The browser build runs no policy: the runtime still
//! reports here, and nothing listens.
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

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
pub use program::{Commit, Fault, LoadError, Plane, Program, Retention, Table};
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
    /// The session id is `<tenant>/<session>`.
    SessionProps(String, Props),
    SessionClosed(String),
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
