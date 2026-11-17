//! No table is the engine's default: every key 0, nothing held, so each order
//! falls back to arrival.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::inferlet::process::ProcessId;

/// The 1.1 schedule columns as they stand: `priority` (highest first) and
/// `hold`, each by process with the column's default for the rest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    keys: HashMap<ProcessId, i64>,
    key_default: Option<i64>,
    holds: HashMap<ProcessId, bool>,
    hold_default: Option<bool>,
}

impl Table {
    pub fn new(
        keys: HashMap<ProcessId, i64>,
        key_default: Option<i64>,
        holds: HashMap<ProcessId, bool>,
        hold_default: Option<bool>,
    ) -> Self {
        Self {
            keys,
            key_default,
            holds,
            hold_default,
        }
    }

    pub fn key(&self, pid: ProcessId) -> i64 {
        let id = plex_id(pid);
        self.keys
            .get(&id)
            .copied()
            .or(self.key_default)
            .unwrap_or(0)
    }

    /// Its row, or the default, says so.
    pub fn names(&self, pid: ProcessId) -> bool {
        self.keys.contains_key(&plex_id(pid)) || self.key_default.is_some()
    }

    /// Not to be started, and to be paused if running.
    pub fn held(&self, pid: ProcessId) -> bool {
        let id = plex_id(pid);
        self.holds
            .get(&id)
            .copied()
            .or(self.hold_default)
            .unwrap_or(false)
    }
}

static CURRENT: RwLock<Option<Arc<Table>>> = RwLock::new(None);

#[cfg(test)]
pub(crate) static TEST_LOCK: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

pub fn current() -> Option<Arc<Table>> {
    CURRENT.read().clone()
}

pub fn key(pid: ProcessId) -> i64 {
    CURRENT
        .read()
        .as_ref()
        .map_or(0, |schedule| schedule.key(pid))
}

pub fn held(pid: ProcessId) -> bool {
    CURRENT
        .read()
        .as_ref()
        .is_some_and(|schedule| schedule.held(pid))
}

pub(crate) fn install(schedule: Option<Table>) {
    *CURRENT.write() = schedule.map(Arc::new);
    crate::inferlet::process::regrant_admission();
    for planner in crate::planner::planners() {
        planner.reschedule();
    }
    crate::scheduler::worker::notify_schedule_changed();
}

static ALIASES: RwLock<Option<HashMap<ProcessId, ProcessId>>> = RwLock::new(None);

pub fn plex_id(pid: ProcessId) -> ProcessId {
    ALIASES
        .read()
        .as_ref()
        .and_then(|aliases| aliases.get(&pid).copied())
        .unwrap_or(pid)
}

pub(crate) fn alias(pid: ProcessId, client_pid: ProcessId) {
    if pid != client_pid {
        ALIASES
            .write()
            .get_or_insert_with(HashMap::new)
            .insert(pid, client_pid);
    }
}

pub(crate) fn forget(pid: ProcessId) {
    if let Some(aliases) = ALIASES.write().as_mut() {
        aliases.remove(&pid);
    }
}
