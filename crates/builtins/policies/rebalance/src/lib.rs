//! Keeps running turns even across the serving workers by moving them.

use std::collections::{BTreeMap, HashSet};

use policy::{Dispatch, Id, Placement, Subject, SubjectKind, attr, candidates, cluster, prop};

#[derive(Default)]
struct Rebalance {
    moved: HashSet<Id>,
}

impl cluster::Policy for Rebalance {
    fn on_migrated(&mut self, dispatch: &Dispatch) {
        self.moved.insert(dispatch.process.clone());
    }

    fn on_exited(&mut self, exit: &policy::Exit) {
        self.moved.remove(&exit.process);
    }

    fn plan(&mut self, _now_ms: u64) {
        let is = |worker: &str, name: &str| {
            attr(&Subject::Worker(worker.to_string()), name).boolean() == Some(true)
        };
        let mut running: BTreeMap<String, Vec<(i64, Id)>> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|w| !is(w, "draining") && !is(w, "pool"))
            .map(|w| (w, Vec::new()))
            .collect();
        let mut placements = Vec::new();
        for process in candidates(SubjectKind::Process) {
            let of = Subject::Process(process.clone());
            if attr(&of, "state").text() != Some("running") {
                // A launch names where it starts, or goes by the default.
                let named = prop(&of, "worker");
                let at = named
                    .int()
                    .map(|n| n.to_string())
                    .or(named.text().map(str::to_string));
                if let Some(at) = at {
                    placements.push(Placement {
                        process,
                        worker: Some(at),
                        executors: Vec::new(),
                        props: Vec::new(),
                    });
                }
                continue;
            }
            let Some(worker) = attr(&of, "worker").text().map(str::to_string) else {
                continue;
            };
            let arrived = attr(&of, "arrival-ms").int().unwrap_or(0);
            if let Some(turns) = running.get_mut(&worker) {
                turns.push((arrived, process));
            }
        }
        let busiest = running.iter().max_by_key(|(_, t)| t.len());
        let idlest = running.iter().min_by_key(|(_, t)| t.len());
        if let (Some((from, turns)), Some((to, idle))) = (busiest, idlest)
            && turns.len() >= idle.len() + 2
        {
            // The youngest has the least to carry.
            if let Some((_, process)) = turns
                .iter()
                .filter(|(_, p)| !self.moved.contains(p))
                .max_by_key(|(arrived, _)| *arrived)
            {
                let _ = from;
                placements.push(Placement {
                    process: process.clone(),
                    worker: Some(to.clone()),
                    executors: Vec::new(),
                    props: Vec::new(),
                });
                self.moved.insert(process.clone());
            }
        }
        let _ = cluster::install_placement(&placements);
    }
}

policy::cluster!(Rebalance);
