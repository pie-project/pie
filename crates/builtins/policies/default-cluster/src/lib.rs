//! The cluster policy pie runs when `[cluster] cluster_policies` names none.
//! It writes each row only when it differs from what it last wrote:
//!
//! - `replication/workers`: every KV pool (a worker with no `device` tier)
//!   keeps a copy of what the workers cache. Each pool's own retention
//!   says how long.
//! - `placement/helpers`: a process's prefill may run on any worker that
//!   serves other workers' passes (`pie/prefill-listen`, its
//!   `offload.listen`). Whether a pass is worth sending is its worker's
//!   configuration (`offload.prefill_min_suffix_tokens`).
//! - `placement/workers`: each pending process gets a row of the decode
//!   workers it may run on, best first. A worker its session already runs
//!   on goes first; the rest go least loaded first (the worker's kv
//!   pressure, `pie/kv-pressure`, then the processes it holds,
//!   `pie/inflight`, then its id), a process placed this round counting on
//!   the worker it was placed at the head of. A worker that is draining or
//!   full ([`FULL_PRESSURE`], [`FULL_PROCESSES`]) is left off; with none
//!   left the row is empty, which holds the process at its gateway until a
//!   round finds room (it looks again each second while one is held). A
//!   row goes once its process is dispatched, so a running process never
//!   moves for its worker's load. Importing this column puts every launch
//!   behind the dispatch gate: a gateway dispatches it once the policy has
//!   heard it.

use std::collections::{BTreeMap, HashMap};

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

#[derive(Default)]
struct DefaultCluster {
    /// What each default was last written as; empty is the column's own.
    pools: Vec<WorkerId>,
    helpers: Vec<WorkerId>,
    /// The placement rows last written, by process.
    placed: BTreeMap<ProcessId, Vec<WorkerId>>,
}

/// A decode worker's load, compared least first.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Load {
    pressure: i64,
    processes: i64,
    id: WorkerId,
}

impl Load {
    fn full(&self) -> bool {
        self.pressure >= FULL_PRESSURE || self.processes >= FULL_PROCESSES
    }
}

/// The decode workers that may take a new process, each with its load.
fn decode_workers() -> Vec<Load> {
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

impl DefaultCluster {
    /// A row for each pending process; the rows of the rest go.
    fn place(&mut self, now: Mark) {
        let mut loads = decode_workers();
        let processes = snapshot::get_processes();
        let mut sessions: HashMap<&str, WorkerId> = HashMap::new();
        for process in &processes {
            if let (Some(session), Some(worker)) = (&process.session, &process.worker) {
                sessions.insert(session.as_str(), worker.clone());
            }
        }
        let mut pending: Vec<&snapshot::Process> = processes
            .iter()
            .filter(|process| process.state == ProcessState::Pending)
            .collect();
        // First come, first placed.
        pending.sort_by(|a, b| (a.arrival, &a.id).cmp(&(b.arrival, &b.id)));
        let mut rows = BTreeMap::new();
        for process in pending {
            loads.sort();
            let mut row: Vec<WorkerId> = loads
                .iter()
                .filter(|load| !load.full())
                .map(|load| load.id.clone())
                .collect();
            let home = process
                .session
                .as_deref()
                .and_then(|session| sessions.get(session));
            if let Some(at) = home.and_then(|home| row.iter().position(|w| w == home)) {
                let home = row.remove(at);
                row.insert(0, home);
            }
            if let Some(first) = row.first()
                && let Some(load) = loads.iter_mut().find(|load| load.id == *first)
            {
                load.processes = load.processes.saturating_add(1);
            }
            rows.insert(process.id.clone(), row);
        }
        if rows.values().any(Vec::is_empty) {
            alarm::set(now.saturating_add(HELD_RECHECK));
        }
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
            return;
        }
        self.placed = rows;
    }
}

impl Policy for DefaultCluster {
    fn plan(&mut self, now: Mark) {
        self.place(now);
        let mut pools = Vec::new();
        let mut helpers = Vec::new();
        for worker in snapshot::get_workers() {
            if worker.device().is_none() {
                pools.push(worker.id.clone());
            }
            let listen =
                snapshot::get_attr(&Subject::Worker(worker.id.clone()), "pie/prefill-listen");
            if listen.is_some() {
                helpers.push(worker.id);
            }
        }
        pools.sort();
        helpers.sort();
        if pools != self.pools {
            let written = match pools.as_slice() {
                [] => {
                    replication::workers::clear();
                    Ok(())
                }
                some => replication::workers::set_default(some),
            };
            if written.is_ok() {
                self.pools = pools;
            }
        }
        if helpers != self.helpers {
            let written = match helpers.as_slice() {
                [] => {
                    placement::helpers::clear();
                    Ok(())
                }
                some => placement::helpers::set_default(some),
            };
            if written.is_ok() {
                self.helpers = helpers;
            }
        }
    }
}

plex::export!(DefaultCluster);
