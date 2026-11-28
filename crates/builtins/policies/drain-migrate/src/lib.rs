//! Moves the turns on a draining worker to the decode workers that are not,
//! spread over the least loaded, over what `default-cluster` does.
//!
//! A turn moves while it has sent no output, by running again from its
//! input, or after some where it can go on from what it sent. One the
//! gateway refuses finishes where it runs, which the drain waits for, and
//! is not asked again; a tied launch is not asked at all.
//!
//! - Hint `worker` (int or text): where to start it, as under `pin`.

use std::collections::{BTreeMap, BTreeSet};

use cluster_defaults::{Defaults, decode_workers, move_row, named, tied, wake};
use plex::prelude::*;

#[derive(Default)]
struct DrainMigrate {
    defaults: Defaults,
    /// Where each turn asked to move is to go.
    moving: BTreeMap<ProcessId, WorkerId>,
    /// Moves the gateway refused: those turns stay where they run.
    refused: BTreeSet<ProcessId>,
}

impl DrainMigrate {
    fn moves(&mut self) -> BTreeMap<ProcessId, Vec<WorkerId>> {
        let draining: BTreeSet<WorkerId> = snapshot::get_workers()
            .into_iter()
            .filter(|worker| worker.draining)
            .map(|worker| worker.id)
            .collect();
        let mut serving = decode_workers();
        serving.sort_by(|a, b| (a.processes, &a.id).cmp(&(b.processes, &b.id)));
        let serving: Vec<WorkerId> = serving.into_iter().map(|load| load.id).collect();
        let moving = std::mem::take(&mut self.moving);
        let mut rows = BTreeMap::new();
        if serving.is_empty() {
            return rows;
        }
        for process in snapshot::get_processes() {
            let Some(at) = process.worker.as_ref().filter(|at| draining.contains(*at)) else {
                continue;
            };
            if !process.running() || self.refused.contains(&process.id) || tied(&process).is_some()
            {
                continue;
            }
            let to = moving
                .get(&process.id)
                .filter(|to| serving.contains(to))
                .unwrap_or(&serving[self.moving.len() % serving.len()])
                .clone();
            rows.insert(process.id.clone(), move_row(&to, at, &serving));
            self.moving.insert(process.id, to);
        }
        rows
    }
}

impl Policy for DrainMigrate {
    fn on_refused(&mut self, _now: Mark) {
        for (process, _) in placement::workers::get_refused() {
            self.refused.insert(process);
        }
    }

    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.refused.remove(&exit.process);
    }

    fn plan(&mut self, now: Mark) {
        let moves = self.moves();
        wake([self.defaults.plan(now, named, moves)]);
    }
}

plex::export!(DrainMigrate);
