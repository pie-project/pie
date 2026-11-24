//! The schedule table (PLEX 1.1 `schedule/priority` and `schedule/hold`).
//!
//! No row is the column's default: key 0, not held, so each order falls back
//! to the engine's own, arrival (first come, first served). That tie rule is
//! the only order the runtime keeps of its own.

use std::collections::{HashMap, HashSet};
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

#[cfg(test)]
pub(crate) fn install(schedule: Option<Table>) {
    *CURRENT.write() = schedule.map(Arc::new);
    changed(None);
}

/// What a policy round changed: rows set (`Some`) or deleted (`None`) by
/// PLEX id, and defaults changed (`Some(None)`: back to the column's own).
#[derive(Debug, Default)]
pub struct Change {
    pub keys: Vec<(ProcessId, Option<i64>)>,
    pub key_default: Option<Option<i64>>,
    pub holds: Vec<(ProcessId, Option<bool>)>,
    pub hold_default: Option<Option<bool>>,
}

impl Change {
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
            && self.key_default.is_none()
            && self.holds.is_empty()
            && self.hold_default.is_none()
    }
}

/// Enacts a round's changes in place: only the rows named change, and the
/// planners read again only the processes they name (all of them where a
/// default changed). Ties keep the engine's order: first come, first served.
pub(crate) fn apply(change: Change) {
    if change.is_empty() {
        return;
    }
    tracing::debug!(target: "pie::schedule", keys = ?change.keys, holds = ?change.holds, "rows enacted");
    let all = change.key_default.is_some() || change.hold_default.is_some();
    let touched: HashSet<ProcessId> = change
        .keys
        .iter()
        .map(|(id, _)| *id)
        .chain(change.holds.iter().map(|(id, _)| *id))
        .collect();
    {
        let mut current = CURRENT.write();
        // A reader holding the last table keeps it; this one is the next.
        let table = Arc::make_mut(current.get_or_insert_with(Default::default));
        for (id, row) in change.keys {
            match row {
                Some(key) => table.keys.insert(id, key),
                None => table.keys.remove(&id),
            };
        }
        for (id, row) in change.holds {
            match row {
                Some(hold) => table.holds.insert(id, hold),
                None => table.holds.remove(&id),
            };
        }
        if let Some(default) = change.key_default {
            table.key_default = default;
        }
        if let Some(default) = change.hold_default {
            table.hold_default = default;
        }
    }
    changed((!all).then_some(&touched));
}

/// The table changed for the processes `only` names, or all.
fn changed(only: Option<&HashSet<ProcessId>>) {
    crate::inferlet::process::regrant_admission();
    for planner in crate::planner::planners() {
        planner.reschedule_only(only);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_change_sets_and_deletes_only_the_rows_it_names() {
        let (a, b, c) = (
            ProcessId::from_u128(0x5a),
            ProcessId::from_u128(0x5b),
            ProcessId::from_u128(0x5c),
        );
        let _lock = TEST_LOCK.blocking_write();
        install(None);
        apply(Change {
            keys: vec![(a, Some(7)), (b, Some(-1))],
            holds: vec![(b, Some(true))],
            ..Change::default()
        });
        assert_eq!((key(a), held(a)), (7, false));
        assert_eq!((key(b), held(b)), (-1, true));
        assert_eq!((key(c), held(c)), (0, false), "no row: the column's own");
        let before = current().unwrap();
        apply(Change {
            keys: vec![(a, None)],
            hold_default: Some(Some(true)),
            ..Change::default()
        });
        assert_eq!((key(a), held(a)), (0, true), "row gone, the default holds");
        assert_eq!(before.key(a), 7, "a reader's table is not changed under it");
        apply(Change {
            keys: vec![(b, None)],
            holds: vec![(b, None)],
            hold_default: Some(None),
            ..Change::default()
        });
        assert_eq!(*current().unwrap(), Table::default(), "a withdrawal leaves no row");
        install(None);
    }
}
