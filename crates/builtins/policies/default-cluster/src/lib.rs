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

use std::collections::BTreeMap;

use cluster_defaults::{Defaults, wake};
use plex::prelude::*;

#[derive(Default)]
struct DefaultCluster {
    defaults: Defaults,
}

impl Policy for DefaultCluster {
    fn plan(&mut self, now: Mark) {
        wake([self.defaults.plan(now, |_| None, BTreeMap::new())]);
    }
}

plex::export!(DefaultCluster);
