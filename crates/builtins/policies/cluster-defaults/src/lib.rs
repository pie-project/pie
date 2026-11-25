//! What every built-in cluster policy does beside its own rows: a row for
//! each pending process ([`Placement`]), the helpers every process may use
//! ([`Helpers`]) and the pools every run is kept in ([`Pools`]). None sets
//! the alarm: each says when it wants the next look, and a policy sets the
//! earliest ([`wake`]).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use plex::prelude::*;

/// The kv pressure (of 255) at which a worker takes no new process: the
/// last sixteenth is room for the running ones to grow into.
const FULL_PRESSURE: i64 = 240;

/// The processes at which a worker takes no new one: past it a new one
/// only waits at the worker behind the rest.
const FULL_PROCESSES: i64 = 1024;

/// While a process is held, the next look for room: a worker's load is an
/// attr, which wakes no policy, and the controller restates it each second.
const HELD_RECHECK: Duration = SECOND;

/// Sets the alarm at the earliest of `marks`, if any.
pub fn wake(marks: impl IntoIterator<Item = Option<Mark>>) {
    if let Some(at) = marks.into_iter().flatten().min() {
        alarm::set(at);
    }
}

/// A decode worker's load, compared least first.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Load {
    pub pressure: i64,
    pub processes: i64,
    pub id: WorkerId,
}

impl Load {
    pub fn full(&self) -> bool {
        self.pressure >= FULL_PRESSURE || self.processes >= FULL_PROCESSES
    }
}

/// The decode workers that may take a new process, each with its load.
pub fn decode_workers() -> Vec<Load> {
    snapshot::get_workers()
        .into_iter()
        .filter(|worker| !worker.draining && worker.device().is_some())
        .filter_map(|worker| {
            let subject = Subject::Worker(worker.id.clone());
            let attr = |name| snapshot::get_attr(&subject, name);
            (attr("pie/role") == Some(Value::Text("decode".into()))).then(|| Load {
                pressure: attr("pie/kv-pressure").and_then(|v| v.int()).unwrap_or(0),
                processes: attr("pie/inflight").and_then(|v| v.int()).unwrap_or(0),
                id: worker.id,
            })
        })
        .collect()
}

/// The session a process is of: the one its client names, else its
/// connection.
fn session_of(process: &snapshot::Process) -> Option<String> {
    process.session.clone().or_else(|| {
        match snapshot::get_attr(&Subject::Process(process.id.clone()), "pie/connection") {
            // Never spelled as a named session is: those hold a `/`.
            Some(Value::Text(connection)) => Some(connection),
            _ => None,
        }
    })
}

/// Where `session` ranks `worker` (rendezvous hashing: FNV-1a over both,
/// mixed): the highest is its worker. One that joins or leaves moves only
/// the sessions it ranks first for.
pub fn rank(session: &str, worker: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in session.bytes().chain([0]).chain(worker.bytes()) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

/// The `placement/workers` rows of the pending processes.
#[derive(Default)]
pub struct Placement {
    /// The rows last written, by process.
    placed: BTreeMap<ProcessId, Vec<WorkerId>>,
}

impl Placement {
    /// Writes a row for each pending process, and `running`'s rows for the
    /// processes it names; the rows of the rest go. A process `named` names
    /// a worker for goes to it first, where that takes new processes at
    /// all, draining excepted. A tied launch goes to its worker, whatever
    /// is named. When to look again, while one is held.
    pub fn plan(
        &mut self,
        now: Mark,
        named: impl Fn(&snapshot::Process) -> Option<WorkerId>,
        running: BTreeMap<ProcessId, Vec<WorkerId>>,
    ) -> Option<Mark> {
        let mut loads = decode_workers();
        let known: BTreeSet<WorkerId> = snapshot::get_workers()
            .into_iter()
            .map(|worker| worker.id)
            .collect();
        let processes = snapshot::get_processes();
        let mut homes: HashMap<String, WorkerId> = HashMap::new();
        for process in &processes {
            if let (Some(worker), Some(session)) = (&process.worker, session_of(process)) {
                homes.insert(session, worker.clone());
            }
        }
        let mut pending: Vec<&snapshot::Process> = processes
            .iter()
            .filter(|process| process.state == ProcessState::Pending)
            .collect();
        // First come, first placed.
        pending.sort_by(|a, b| (a.arrival, &a.id).cmp(&(b.arrival, &b.id)));
        let mut rows = running;
        for process in pending {
            loads.sort();
            let mut row: Vec<WorkerId> = loads
                .iter()
                .filter(|load| !load.full())
                .map(|load| load.id.clone())
                .collect();
            let tied = snapshot::get_attr(&Subject::Process(process.id.clone()), "pie/tied");
            if let Some(Value::Text(tied)) = &tied {
                // Gone, it has no row: its gateway says its handles went.
                if !known.contains(tied) {
                    continue;
                }
                row = vec![tied.clone()];
            } else if let Some(session) = session_of(process) {
                let home = homes.get(&session);
                row.sort_by_key(|w| (home != Some(w), Reverse(rank(&session, w))));
            }
            // With every worker full it is held, whatever is named.
            if tied.is_none()
                && !row.is_empty()
                && let Some(to) = named(process)
                && loads.iter().any(|load| load.id == to)
            {
                row.retain(|w| *w != to);
                row.insert(0, to);
            }
            if let Some(first) = row.first()
                && let Some(load) = loads.iter_mut().find(|load| load.id == *first)
            {
                load.processes = load.processes.saturating_add(1);
            }
            rows.insert(process.id.clone(), row);
        }
        let again = rows
            .values()
            .any(Vec::is_empty)
            .then(|| now.saturating_add(HELD_RECHECK));
        for gone in self.placed.keys().filter(|id| !rows.contains_key(*id)) {
            placement::workers::delete(gone);
        }
        let changed: Vec<(ProcessId, Vec<WorkerId>)> = rows
            .iter()
            .filter(|(id, row)| self.placed.get(*id) != Some(row))
            .map(|(id, row)| (id.clone(), row.clone()))
            .collect();
        if !changed.is_empty() && placement::workers::set_list(&changed).is_err() {
            // Refused whole: nothing of it is written, so it goes again
            // next round.
            self.placed.retain(|id, _| rows.contains_key(id));
            for (id, _) in &changed {
                self.placed.remove(id);
            }
            return again;
        }
        self.placed = rows;
        again
    }
}

/// `placement/helpers`: every worker that serves other workers' passes.
#[derive(Default)]
pub struct Helpers {
    /// What the default was last written as; empty is the column's own.
    written: Vec<WorkerId>,
}

impl Helpers {
    pub fn plan(&mut self) {
        let mut helpers: Vec<WorkerId> = snapshot::get_workers()
            .into_iter()
            .map(|worker| worker.id)
            .filter(|id| {
                snapshot::get_attr(&Subject::Worker(id.clone()), "pie/prefill-listen").is_some()
            })
            .collect();
        helpers.sort();
        if helpers == self.written {
            return;
        }
        let written = match helpers.as_slice() {
            [] => {
                placement::helpers::clear();
                Ok(())
            }
            some => placement::helpers::set_default(some),
        };
        if written.is_ok() {
            self.written = helpers;
        }
    }
}

/// The KV pools, the workers with no device tier, in order.
pub fn pools() -> Vec<WorkerId> {
    let mut pools: Vec<WorkerId> = snapshot::get_workers()
        .into_iter()
        .filter(|worker| worker.device().is_none())
        .map(|worker| worker.id)
        .collect();
    pools.sort();
    pools
}

/// `replication/workers`: every pool keeps a copy of what the workers cache.
#[derive(Default)]
pub struct Pools {
    /// What the default was last written as; empty is the column's own.
    written: Vec<WorkerId>,
}

impl Pools {
    pub fn plan(&mut self) {
        let pools = pools();
        if pools == self.written {
            return;
        }
        let written = match pools.as_slice() {
            [] => {
                replication::workers::clear();
                Ok(())
            }
            some => replication::workers::set_default(some),
        };
        if written.is_ok() {
            self.written = pools;
        }
    }
}

/// All three, for a policy whose own rows are the placement's.
#[derive(Default)]
pub struct Defaults {
    placement: Placement,
    helpers: Helpers,
    pools: Pools,
}

impl Defaults {
    /// [`Placement::plan`], then the helpers and the pools.
    pub fn plan(
        &mut self,
        now: Mark,
        named: impl Fn(&snapshot::Process) -> Option<WorkerId>,
        running: BTreeMap<ProcessId, Vec<WorkerId>>,
    ) -> Option<Mark> {
        let again = self.placement.plan(now, named, running);
        self.helpers.plan();
        self.pools.plan();
        again
    }
}
