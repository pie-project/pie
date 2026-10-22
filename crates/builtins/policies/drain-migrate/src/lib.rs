//! Moves the turns on a draining worker to the least loaded worker that is
//! not draining. The gateway moves a turn only while it has sent no output;
//! one it refuses, for having sent some, finishes where it runs, which the
//! drain waits for, and is not asked again. A launch's `worker` prop places
//! it before it runs, as `pin` does, so a pinned launch still moves when its
//! worker drains.

use std::collections::HashSet;

use policy::{Id, Placement, Subject, SubjectKind, attr, candidates, cluster, prop};

#[derive(Default)]
struct DrainMigrate {
    /// Moves the gateway refused: those turns stay where they run.
    refused: HashSet<Id>,
}

impl cluster::Policy for DrainMigrate {
    fn on_migration_failed(&mut self, process: &Id) {
        self.refused.insert(process.clone());
    }

    fn on_exited(&mut self, exit: &policy::Exit) {
        self.refused.remove(&exit.process);
    }

    fn plan(&mut self, _now_ms: u64) {
        let workers = candidates(SubjectKind::Worker);
        let is = |worker: &str, name: &str| {
            attr(&Subject::Worker(worker.to_string()), name).boolean() == Some(true)
        };
        let draining: Vec<&String> = workers.iter().filter(|w| is(w, "draining")).collect();
        let mut serving: Vec<(i64, String)> = workers
            .iter()
            .filter(|w| !is(w, "draining") && !is(w, "pool"))
            .map(|w| {
                let load = attr(&Subject::Worker(w.clone()), "pie/inflight")
                    .int()
                    .unwrap_or(0);
                (load, w.clone())
            })
            .collect();
        serving.sort();
        let mut placements = Vec::new();
        for process in candidates(SubjectKind::Process) {
            let of = Subject::Process(process.clone());
            // Only before it runs: once it has run, where it runs is its own
            // until a drain moves it.
            if attr(&of, "worker").is_some() {
                continue;
            }
            let named = prop(&of, "worker");
            let Some(pinned) = named
                .int()
                .map(|n| n.to_string())
                .or(named.text().map(str::to_string))
            else {
                continue;
            };
            if draining.contains(&&pinned) {
                continue;
            }
            placements.push(Placement {
                process,
                worker: Some(pinned),
                executors: Vec::new(),
                props: Vec::new(),
            });
        }
        if !draining.is_empty() && !serving.is_empty() {
            let mut moved = 0usize;
            for process in candidates(SubjectKind::Process) {
                if self.refused.contains(&process) {
                    continue;
                }
                let of = Subject::Process(process.clone());
                let Some(worker) = attr(&of, "worker").text().map(str::to_string) else {
                    continue;
                };
                if attr(&of, "state").text() != Some("running") || !draining.contains(&&worker) {
                    continue;
                }
                // Spread over the least loaded, in turn.
                let (_, to) = &serving[moved % serving.len()];
                moved += 1;
                placements.push(Placement {
                    process,
                    worker: Some(to.clone()),
                    executors: Vec::new(),
                    props: Vec::new(),
                });
            }
        }
        let _ = cluster::install_placement(&placements);
    }
}

policy::cluster!(DrainMigrate);
