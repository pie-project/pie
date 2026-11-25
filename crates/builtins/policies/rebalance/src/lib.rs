//! Keeps running turns even across the decode workers (after Llumnix), over
//! what `default-cluster` does: while one runs at least two more than
//! another, the youngest of its turns moves to the idlest, one a call and
//! each turn at most once.
//!
//! A turn moves while it has sent no output, by running again from its
//! input, or after some where it can go on from what it sent. One the
//! gateway refuses stays where it runs; a tied launch is not asked at all.
//!
//! - Hint `worker` (int or text): where to start it, as under `pin`.

use std::collections::{BTreeMap, BTreeSet};

use cluster_defaults::{Defaults, decode_workers, move_row, named, tied, wake};
use plex::prelude::*;

#[derive(Default)]
struct Rebalance {
    defaults: Defaults,
    /// Turns on their way: from where, to where.
    moving: BTreeMap<ProcessId, (WorkerId, WorkerId)>,
    /// Turns asked to move, and those refused: none is asked again.
    asked: BTreeSet<ProcessId>,
}

impl Rebalance {
    fn moves(&mut self) -> BTreeMap<ProcessId, Vec<WorkerId>> {
        let serving: Vec<WorkerId> = decode_workers().into_iter().map(|load| load.id).collect();
        let mut turns: BTreeMap<WorkerId, Vec<(Mark, ProcessId)>> =
            serving.iter().map(|id| (id.clone(), Vec::new())).collect();
        let running: Vec<snapshot::Process> = snapshot::get_processes()
            .into_iter()
            .filter(|process| process.running())
            .collect();
        // Arrived elsewhere, exited or with nowhere to go: no more a move.
        self.moving.retain(|id, (from, to)| {
            serving.contains(to)
                && running
                    .iter()
                    .any(|process| process.id == *id && process.worker.as_ref() == Some(from))
        });
        for process in running {
            // A turn on its way counts where it is going.
            let at = match self.moving.get(&process.id) {
                Some((_, to)) => Some(to),
                None => process.worker.as_ref(),
            };
            if let Some(turns) = at.and_then(|at| turns.get_mut(at)) {
                turns.push((process.arrival, process.id));
            }
        }
        let busiest = turns.iter().max_by_key(|(_, turns)| turns.len());
        let idlest = turns.iter().min_by_key(|(_, turns)| turns.len());
        if let (Some((from, turns)), Some((to, idle))) = (busiest, idlest)
            && turns.len() >= idle.len() + 2
            // The youngest has the least to carry.
            && let Some((_, id)) = turns
                .iter()
                .filter(|(_, id)| !self.asked.contains(id))
                .filter(|(_, id)| snapshot::get_process(id).is_some_and(|p| tied(&p).is_none()))
                .max_by_key(|(arrival, _)| *arrival)
        {
            self.asked.insert(id.clone());
            self.moving.insert(id.clone(), (from.clone(), to.clone()));
        }
        self.moving
            .iter()
            .map(|(id, (from, to))| (id.clone(), move_row(to, from, &serving)))
            .collect()
    }
}

impl Policy for Rebalance {
    fn on_refused(&mut self, _now: Mark) {
        for (process, _) in placement::workers::get_refused() {
            self.moving.remove(&process);
        }
    }

    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.asked.remove(&exit.process);
    }

    fn plan(&mut self, now: Mark) {
        let moves = self.moves();
        wake([self.defaults.plan(now, named, moves)]);
    }
}

plex::export!(Rebalance);
