//! Handles: state a session owns between processes.
//!
//! A process is short and its state long: a working set a client names lives
//! in the session, moves into a process for a launch, and comes back when the
//! process gives it. It has one owner at a time, so nothing here shares a
//! mutable set; a client that wants to keep one forks it first.
//!
//! The planner schedules residency, not programs: each handle out of a
//! process stands in it as a [holder](Held), idle from the start, and the
//! eviction and restore every process gets are the ones it gets.

mod held;
mod lease;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod snapshot;
mod table;

pub(crate) use lease::{Bag, Kind, Lease, Moved, enter};
pub(crate) use table::{Landing, Table};

use client_api::{HandleErrorCode, HandleName};

/// How long a fork holds up its session for a move to end, before it is
/// refused as `in_use`: the session has output to deliver.
const FORK_WAIT: std::time::Duration = std::time::Duration::from_millis(250);

/// The most handles one session names at once: a name costs almost nothing,
/// but not nothing, and a client that never closes one should hear so.
const MAX_HANDLES: usize = 4096;

/// The most pictures and sound one session holds, by what they decode to.
/// Nothing moves these off the host, so this is all that bounds them.
const MAX_MEDIA_BYTES: usize = 8 << 30;

/// Why a handle op or launch was refused: its code is what a client branches
/// on, and every layer's text starts with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HandleError {
    pub(crate) code: HandleErrorCode,
    pub(crate) message: String,
}

impl HandleError {
    fn new(code: HandleErrorCode, message: impl Into<String>) -> Self {
        HandleError {
            code,
            message: message.into(),
        }
    }

    fn quota(message: impl Into<String>) -> Self {
        Self::new(HandleErrorCode::Quota, message)
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(HandleErrorCode::Invalid, message)
    }

    /// What held still would not: it stays, and a retry may find it settled.
    fn moving(message: String) -> Self {
        Self::new(HandleErrorCode::InUse, message)
    }

    fn unknown(name: HandleName) -> Self {
        Self::new(
            HandleErrorCode::Unknown,
            format!("no handle {name} in this session"),
        )
    }
}

impl std::fmt::Display for HandleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for HandleError {}

impl From<HandleError> for String {
    fn from(error: HandleError) -> String {
        error.to_string()
    }
}
