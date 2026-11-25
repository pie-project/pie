//! Keeps every run some worker caches on two decode workers, over what
//! `default-cluster` does: the two a rendezvous hash of the run picks among
//! those not draining, so a run's copies stay where they are as workers
//! come and go, and losing a worker leaves each of its runs on another.
//!
//! No event says a run was cached, so the policy looks each second; the
//! controller reads what the workers hold each second too. Which tier a
//! copy sits in, and whether it stays, is the receiving worker's retention.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use cluster_defaults::{Helpers, Placement, decode_workers, pools, rank, wake};
use plex::prelude::*;

/// Workers each run is kept on; the one that computed it may be one of them.
const COPIES: usize = 2;

/// How often it looks for runs cached since.
const LOOK: Duration = SECOND;

/// The rows a column takes.
const MAX_ROWS: usize = 65_536;

#[derive(Default)]
struct Replicate {
    placement: Placement,
    helpers: Helpers,
    /// The next look.
    next: Mark,
}

impl Replicate {
    /// A run's row stands in for the default, so it names the pools too.
    fn copies(&self) {
        let pools = pools();
        let workers: Vec<WorkerId> = decode_workers().into_iter().map(|load| load.id).collect();
        replication::workers::clear();
        if !pools.is_empty() {
            replication::workers::set_default(&pools).ok();
        }
        if workers.len() < COPIES {
            return;
        }
        let rows: Vec<(Prefix, Vec<WorkerId>)> = snapshot::get_working_sets()
            .into_iter()
            .take(MAX_ROWS)
            .map(|set| {
                let mut at = workers.clone();
                at.sort_by_key(|worker| Reverse(rank(&set.id, worker)));
                at.truncate(COPIES);
                at.extend(pools.iter().cloned());
                (Prefix::all(set.id), at)
            })
            .collect();
        replication::workers::set_list(&rows).ok();
    }
}

impl Policy for Replicate {
    fn on_worker_changed(&mut self, _worker: &WorkerId, now: Mark) {
        self.next = now;
    }

    fn plan(&mut self, now: Mark) {
        let again = self.placement.plan(now, |_| None, BTreeMap::new());
        self.helpers.plan();
        if now >= self.next {
            self.copies();
            self.next = now.saturating_add(LOOK);
        }
        wake([again, Some(self.next)]);
    }
}

plex::export!(Replicate);
