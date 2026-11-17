//! The tables: one column per interface, committed rows and the writes a
//! call has staged over them.

use std::collections::HashMap;
use std::hash::Hash;

/// The most rows one column holds.
pub const MAX_ROWS: usize = 65_536;

/// Why a write was not taken or a row not enacted (the contract's
/// `error-code`, without its text).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NotSupported,
    InvalidArgument,
    LimitExceeded,
    Other,
}

impl Refusal {
    pub fn name(self) -> &'static str {
        match self {
            Refusal::NotSupported => "not-supported",
            Refusal::InvalidArgument => "invalid-argument",
            Refusal::LimitExceeded => "limit-exceeded",
            Refusal::Other => "other",
        }
    }

    pub fn parse(name: &str) -> Refusal {
        match name {
            "not-supported" => Refusal::NotSupported,
            "invalid-argument" => Refusal::InvalidArgument,
            "limit-exceeded" => Refusal::LimitExceeded,
            _ => Refusal::Other,
        }
    }
}

/// A column as committed: what the engine reads.
pub struct Column<K, V> {
    pub rows: HashMap<K, V>,
    pub default: Option<V>,
    /// Rows the port could not enact.
    pub refused: HashMap<K, Refusal>,
    /// What changed since the port last looked: a key and its row now
    /// (none: no row). Coalesced, so a port enacts only the difference.
    pub changed: HashMap<K, Option<V>>,
    pub default_changed: bool,
}

impl<K, V> Default for Column<K, V> {
    fn default() -> Self {
        Column {
            rows: HashMap::new(),
            default: None,
            refused: HashMap::new(),
            changed: HashMap::new(),
            default_changed: false,
        }
    }
}

/// The writes of one call to one column, not yet committed.
pub struct Stage<K, V> {
    cleared: bool,
    /// A key written in this call: its value, or none for a delete.
    rows: HashMap<K, Option<V>>,
    /// The default, if this call wrote it (none inside: unset by `clear`).
    default: Option<Option<V>>,
}

impl<K, V> Default for Stage<K, V> {
    fn default() -> Self {
        Stage {
            cleared: false,
            rows: HashMap::new(),
            default: None,
        }
    }
}

impl<K: Hash + Eq + Clone, V: Clone + PartialEq> Stage<K, V> {
    pub fn get(&self, committed: &Column<K, V>, key: &K) -> Option<V> {
        match self.rows.get(key) {
            Some(row) => row.clone(),
            None if self.cleared => None,
            None => committed.rows.get(key).cloned(),
        }
    }

    /// Rows this call would leave, at most: the bound a write is checked
    /// against.
    pub fn len(&self, committed: &Column<K, V>) -> usize {
        if self.cleared {
            self.rows.len()
        } else {
            committed.rows.len() + self.rows.len()
        }
    }

    pub fn set(&mut self, key: K, value: V) {
        self.rows.insert(key, Some(value));
    }

    pub fn delete(&mut self, key: K) {
        self.rows.insert(key, None);
    }

    pub fn set_default(&mut self, value: V) {
        self.default = Some(Some(value));
    }

    pub fn clear(&mut self) {
        self.cleared = true;
        self.rows.clear();
        self.default = Some(None);
    }

    pub fn is_empty(&self) -> bool {
        !self.cleared && self.rows.is_empty() && self.default.is_none()
    }
}

impl<K: Hash + Eq + Clone, V: Clone + PartialEq> Column<K, V> {
    /// Takes a call's writes. Only what differs from the rows in force is
    /// recorded as changed: a plan written again as it was changes nothing.
    pub fn commit(&mut self, stage: Stage<K, V>) -> bool {
        let mut any = false;
        if stage.cleared {
            let gone: Vec<K> = self
                .rows
                .keys()
                .filter(|k| !matches!(stage.rows.get(*k), Some(Some(_))))
                .cloned()
                .collect();
            for key in gone {
                self.rows.remove(&key);
                self.refused.remove(&key);
                self.changed.insert(key, None);
                any = true;
            }
        }
        for (key, row) in stage.rows {
            match row {
                Some(value) => {
                    if self.rows.get(&key) != Some(&value) {
                        self.rows.insert(key.clone(), value.clone());
                        self.refused.remove(&key);
                        self.changed.insert(key, Some(value));
                        any = true;
                    }
                }
                None => {
                    if self.rows.remove(&key).is_some() {
                        self.refused.remove(&key);
                        self.changed.insert(key, None);
                        any = true;
                    }
                }
            }
        }
        if let Some(default) = stage.default
            && self.default != default
        {
            self.default = default;
            self.default_changed = true;
            any = true;
        }
        any
    }

    /// Removes the rows whose subject is gone, and says so to the port
    /// like any other change. True if a row went.
    pub fn reap(&mut self, gone: impl Fn(&K) -> bool) -> bool {
        let keys: Vec<K> = self.rows.keys().filter(|k| gone(k)).cloned().collect();
        self.refused.retain(|k, _| !gone(k));
        for key in &keys {
            self.rows.remove(key);
            self.changed.insert(key.clone(), None);
        }
        !keys.is_empty()
    }

    /// Withdraws every row and the default: the lease ran out.
    pub fn withdraw(&mut self) -> bool {
        let any = !self.rows.is_empty() || self.default.is_some();
        for (key, _) in self.rows.drain() {
            self.changed.insert(key, None);
        }
        self.refused.clear();
        if self.default.take().is_some() {
            self.default_changed = true;
        }
        any
    }
}

/// A retention or replication key: a working set and how many of its
/// first tokens (none: all).
pub type Prefix = (String, Option<u32>);

/// Tiers as a set: bit 0 device, 1 host, 2 disk.
pub type TierSet = u8;

/// Who or what a hint or a note is about.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Subject {
    Global,
    Process(String),
    Session(String),
    Worker(String),
    WorkingSet(String),
}

impl Subject {
    pub fn kind(&self) -> &'static str {
        match self {
            Subject::Global => "global",
            Subject::Process(_) => "process",
            Subject::Session(_) => "session",
            Subject::Worker(_) => "worker",
            Subject::WorkingSet(_) => "working-set",
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Subject::Global => "",
            Subject::Process(id)
            | Subject::Session(id)
            | Subject::Worker(id)
            | Subject::WorkingSet(id) => id,
        }
    }

    pub fn parse(kind: &str, id: &str) -> Option<Subject> {
        Some(match kind {
            "global" => Subject::Global,
            "process" => Subject::Process(id.to_string()),
            "session" => Subject::Session(id.to_string()),
            "worker" => Subject::Worker(id.to_string()),
            "working-set" => Subject::WorkingSet(id.to_string()),
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Boolean(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

/// The names of the columns, as a policy imports them.
pub const SCHEDULE_PRIORITY: &str = "plex:schedule/priority";
pub const SCHEDULE_HOLD: &str = "plex:schedule/hold";
pub const RETENTION_PRIORITY: &str = "plex:retention/priority";
pub const RETENTION_TIERS: &str = "plex:retention/tiers";
pub const PLACEMENT_WORKERS: &str = "plex:placement/workers";
pub const PLACEMENT_HELPERS: &str = "plex:placement/helpers";
pub const REPLICATION_WORKERS: &str = "plex:replication/workers";
pub const TERMINATE: &str = "plex:core/terminate";

pub const COLUMNS: [&str; 7] = [
    SCHEDULE_PRIORITY,
    SCHEDULE_HOLD,
    RETENTION_PRIORITY,
    RETENTION_TIERS,
    PLACEMENT_WORKERS,
    PLACEMENT_HELPERS,
    REPLICATION_WORKERS,
];

/// Every column, and what policies have said beside them.
#[derive(Default)]
pub struct Tables {
    pub schedule_priority: Column<String, i64>,
    pub schedule_hold: Column<String, bool>,
    pub retention_priority: Column<Prefix, i64>,
    pub retention_tiers: Column<Prefix, TierSet>,
    pub placement_workers: Column<String, Vec<String>>,
    pub placement_helpers: Column<String, Vec<String>>,
    pub replication_workers: Column<Prefix, Vec<String>>,
    /// Notes, by subject, then by (writer, name).
    pub notes: HashMap<Subject, HashMap<(String, String), Value>>,
    /// Processes to end, and why, until the port takes them.
    pub terminate: Vec<(String, String)>,
    /// Notes written or deleted since the port last looked, for the other
    /// scope: (writer, subject, name, value).
    pub notes_out: Vec<(String, Subject, String, Option<Value>)>,
}

/// One call's writes to every column, and beside them.
#[derive(Default)]
pub struct Staged {
    pub schedule_priority: Stage<String, i64>,
    pub schedule_hold: Stage<String, bool>,
    pub retention_priority: Stage<Prefix, i64>,
    pub retention_tiers: Stage<Prefix, TierSet>,
    pub placement_workers: Stage<String, Vec<String>>,
    pub placement_helpers: Stage<String, Vec<String>>,
    pub replication_workers: Stage<Prefix, Vec<String>>,
    /// A note set (some) or deleted (none), in order.
    pub notes: Vec<(Subject, String, Option<Value>)>,
    pub terminate: Vec<(String, String)>,
    /// The alarm, if this call set it.
    pub alarm: Option<u64>,
}

impl Tables {
    /// Takes a call's writes, all of them. True if a column changed.
    pub fn commit(&mut self, writer: &str, staged: Staged) -> bool {
        let mut any = false;
        any |= self.schedule_priority.commit(staged.schedule_priority);
        any |= self.schedule_hold.commit(staged.schedule_hold);
        any |= self.retention_priority.commit(staged.retention_priority);
        any |= self.retention_tiers.commit(staged.retention_tiers);
        any |= self.placement_workers.commit(staged.placement_workers);
        any |= self.placement_helpers.commit(staged.placement_helpers);
        any |= self.replication_workers.commit(staged.replication_workers);
        for (subject, name, value) in staged.notes {
            let key = (writer.to_string(), name.clone());
            let changed = match &value {
                Some(v) => {
                    self.notes
                        .entry(subject.clone())
                        .or_default()
                        .insert(key, v.clone())
                        .as_ref()
                        != Some(v)
                }
                None => self
                    .notes
                    .get_mut(&subject)
                    .is_some_and(|notes| notes.remove(&key).is_some()),
            };
            if changed {
                self.notes_out
                    .push((writer.to_string(), subject, name, value));
            }
        }
        if !staged.terminate.is_empty() {
            self.terminate.extend(staged.terminate);
            any = true;
        }
        any
    }

    /// Withdraws the named columns' rows and defaults.
    pub fn withdraw(&mut self, columns: &[String]) -> bool {
        let mut any = false;
        for column in columns {
            any |= match column.as_str() {
                SCHEDULE_PRIORITY => self.schedule_priority.withdraw(),
                SCHEDULE_HOLD => self.schedule_hold.withdraw(),
                RETENTION_PRIORITY => self.retention_priority.withdraw(),
                RETENTION_TIERS => self.retention_tiers.withdraw(),
                PLACEMENT_WORKERS => self.placement_workers.withdraw(),
                PLACEMENT_HELPERS => self.placement_helpers.withdraw(),
                REPLICATION_WORKERS => self.replication_workers.withdraw(),
                _ => false,
            };
        }
        any
    }

    pub fn reap_process(&mut self, id: &str) -> bool {
        self.notes.remove(&Subject::Process(id.to_string()));
        self.schedule_priority.reap(|k| k == id)
            | self.schedule_hold.reap(|k| k == id)
            | self.placement_workers.reap(|k| k == id)
            | self.placement_helpers.reap(|k| k == id)
    }

    pub fn reap_working_set(&mut self, id: &str) -> bool {
        self.notes.remove(&Subject::WorkingSet(id.to_string()));
        self.retention_priority.reap(|k| k.0 == id)
            | self.retention_tiers.reap(|k| k.0 == id)
            | self.replication_workers.reap(|k| k.0 == id)
    }
}
