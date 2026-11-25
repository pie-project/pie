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
//!   workers it may run on, best first. A session's (the one its client
//!   names, else its connection, `pie/connection`) go first to the worker
//!   one of its processes is at, then in the order the session hashes the
//!   workers to, so its next turn finds what its last left cached, with
//!   nothing remembered here. A launch of no session goes least loaded
//!   first (the worker's kv pressure, `pie/kv-pressure`, then the processes
//!   it holds, `pie/inflight`, then its id), a process placed this round
//!   counting on the worker it was placed at the head of. A launch that can
//!   run on one worker alone (`pie/tied`: it holds its session's handles)
//!   gets that one, full or draining, and waits there. Any other worker that is draining or
//!   full ([`FULL_PRESSURE`], [`FULL_PROCESSES`]) is left off; with none
//!   left the row is empty, which holds the process at its gateway until a
//!   round finds room (it looks again each second while one is held). A
//!   row goes once its process is dispatched, so a running process never
//!   moves for its worker's load. Importing this column puts every launch
//!   behind the dispatch gate: a gateway dispatches it once the policy has
//!   heard it.

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
fn rank(session: &str, worker: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in session.bytes().chain([0]).chain(worker.bytes()) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

impl DefaultCluster {
    /// A row for each pending process; the rows of the rest go.
    fn place(&mut self, now: Mark) {
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
        let mut rows = BTreeMap::new();
        for process in pending {
            loads.sort();
            let mut row: Vec<WorkerId> = loads
                .iter()
                .filter(|load| !load.full())
                .map(|load| load.id.clone())
                .collect();
            let tied = snapshot::get_attr(&Subject::Process(process.id.clone()), "pie/tied");
            if let Some(Value::Text(tied)) = tied {
                // Gone, it has no row: its gateway says its handles went.
                if !known.contains(&tied) {
                    continue;
                }
                row = vec![tied];
            } else if let Some(session) = session_of(process) {
                let home = homes.get(&session);
                row.sort_by_key(|w| (home != Some(w), std::cmp::Reverse(rank(&session, w))));
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
