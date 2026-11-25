//! A built-in policy on a PLEX host whose clock and rounds the test runs.

use std::collections::BTreeMap;
use std::hash::Hash;

use plex_host::host::{ColumnDelta, Config, Host};
use plex_host::runtime::wit::Tier;
use plex_host::tables::{PLACEMENT_WORKERS, Refusal, Subject, Value};

const MS: u64 = 1_000_000;

/// The rows and the default in force.
#[derive(Debug)]
pub struct Table<K, V> {
    pub rows: BTreeMap<K, V>,
    pub default: Option<V>,
}

impl<K: Ord, V> Default for Table<K, V> {
    fn default() -> Self {
        Self {
            rows: BTreeMap::new(),
            default: None,
        }
    }
}

impl<K: Ord + Hash, V> Table<K, V> {
    fn take(&mut self, delta: ColumnDelta<K, V>) {
        for (key, row) in delta.rows {
            match row {
                Some(row) => self.rows.insert(key, row),
                None => self.rows.remove(&key),
            };
        }
        if let Some(default) = delta.default {
            self.default = default;
        }
    }
}

type Prefix = (String, Option<u32>);

pub struct Rig {
    pub host: Host,
    /// What the policy may write.
    pub imports: Vec<String>,
    pub keys: Table<String, i64>,
    pub holds: Table<String, bool>,
    pub priority: Table<Prefix, i64>,
    /// Tier bits: the device 1, the host 2, the disk 4.
    pub tiers: Table<Prefix, u8>,
    pub placed: Table<String, Vec<String>>,
    pub helpers: Table<String, Vec<String>>,
    pub copies: Table<Prefix, Vec<String>>,
}

pub fn whole(set: &str) -> Prefix {
    (set.to_string(), None)
}

impl Rig {
    pub fn new(policy: &str) -> Self {
        let host = Host::new(Config {
            manual: true,
            deadline_ms: 60 * 60 * 1000,
            ..Config::default()
        });
        let imports = host
            .load(policy, builtins::policy(policy).unwrap().component)
            .unwrap();
        Self {
            host,
            imports,
            keys: Table::default(),
            holds: Table::default(),
            priority: Table::default(),
            tiers: Table::default(),
            placed: Table::default(),
            helpers: Table::default(),
            copies: Table::default(),
        }
    }

    /// One call of the policy at `ms`, where it has something to hear.
    pub fn at(&mut self, ms: u64) {
        self.host.set_now(ms * MS);
        self.host.run(false);
        let delta = self.host.poll_delta();
        self.keys.take(delta.schedule_priority);
        self.holds.take(delta.schedule_hold);
        self.priority.take(delta.retention_priority);
        self.tiers.take(delta.retention_tiers);
        self.placed.take(delta.placement_workers);
        self.helpers.take(delta.placement_helpers);
        self.copies.take(delta.replication_workers);
    }

    /// When the policy asked to be called again.
    pub fn alarm(&self) -> Option<u64> {
        self.host.alarm().map(|at| at / MS)
    }

    pub fn attr(&self, subject: Subject, name: &str, value: Value) {
        self.host.attr(subject, name, Some(value));
    }

    // A worker's own plane.

    /// A process arrives at its worker, its prompt 100 tokens.
    pub fn at_worker(&self, id: &str, tenant: &str, session: Option<&str>, hints: &[(&str, i64)]) {
        self.host.arrived(
            id,
            session.map(str::to_string),
            Some(tenant.into()),
            Some("self".into()),
            Some(100),
            None,
            ints(hints),
            0,
        );
    }

    /// A process completes, what it leaves cached: 300 tokens.
    pub fn completed(&self, id: &str, session: Option<&str>) {
        self.host
            .put_working_set(id, None, None, session.map(str::to_string), 300, 0, {
                vec![("self".into(), Tier::Device, 300)]
            });
        self.host.exited(id, "completed", None, None, 0);
        self.host.retained(id, "self");
    }

    // The cluster's plane.

    /// A decode worker with a device tier.
    pub fn decode(&self, id: &str, draining: bool) {
        self.host
            .put_worker(id, draining, vec![(Tier::Device, None, 0)], 0, 0, None);
        self.attr(worker(id), "pie/role", Value::Text("decode".into()));
    }

    /// A launch arrives at its gateway.
    pub fn launch(&self, id: &str, session: Option<&str>, hints: &[(&str, i64)]) {
        self.host.arrived(
            id,
            session.map(str::to_string),
            Some("t".into()),
            None,
            None,
            None,
            ints(hints),
            0,
        );
    }

    /// A launch arrives, and runs on `at`.
    pub fn turn(&self, id: &str, at: &str) {
        self.launch(id, None, &[]);
        self.host.dispatched(id, at);
        self.host.started(id);
    }

    /// The gateway cannot move `id`.
    pub fn refuse(&self, id: &str) {
        self.host
            .refuse(PLACEMENT_WORKERS, id, None, Some(Refusal::NotSupported));
    }

    /// The row of `id`, best first.
    pub fn row(&self, id: &str) -> Option<Vec<&str>> {
        let row = self.placed.rows.get(id)?;
        Some(row.iter().map(String::as_str).collect())
    }
}

fn ints(hints: &[(&str, i64)]) -> Vec<(String, Value)> {
    hints
        .iter()
        .map(|(name, value)| (name.to_string(), Value::Int(*value)))
        .collect()
}

pub fn worker(id: &str) -> Subject {
    Subject::Worker(id.into())
}

pub fn process(id: &str) -> Subject {
    Subject::Process(id.into())
}
