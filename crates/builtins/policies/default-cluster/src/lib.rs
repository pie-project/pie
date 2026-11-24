//! The cluster policy pie runs when `[cluster] cluster_policies` names none.
//! It writes only defaults, each only when it differs from the column's own:
//!
//! - `replication/workers`: every KV pool (a worker with no `device` tier)
//!   keeps a copy of what the workers cache. Each pool's own retention
//!   says how long.
//! - `placement/helpers`: a process's prefill may run on any worker that
//!   serves other workers' passes (`pie/prefill-listen`, its
//!   `offload.listen`). Whether a pass is worth sending is its worker's
//!   configuration (`offload.prefill_min_suffix_tokens`).
//!
//! It writes no placement rows: the router's own order places processes.

use plex::prelude::*;

#[derive(Default)]
struct DefaultCluster {
    /// What each default was last written as; empty is the column's own.
    pools: Vec<WorkerId>,
    helpers: Vec<WorkerId>,
}

impl Policy for DefaultCluster {
    fn plan(&mut self, _now: Mark) {
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
