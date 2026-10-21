//! Moves the turns on a draining worker to the least loaded worker that is
//! not draining. The gateway moves a turn only while it has sent no output;
//! one that has finishes where it runs, which the drain waits for. A
//! launch's `worker` prop places it before it runs, as `pin` does, so a
//! pinned launch still moves when its worker drains.

use policy::{Placement, Subject, SubjectKind, attr, candidates, cluster, prop};

#[derive(Default)]
struct DrainMigrate;

impl cluster::Policy for DrainMigrate {
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
            let Some(pinned) = prop(&of, "worker").int().map(|w| w.to_string()) else {
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
            for (at, process) in candidates(SubjectKind::Process).into_iter().enumerate() {
                let of = Subject::Process(process.clone());
                let Some(worker) = attr(&of, "worker").text().map(str::to_string) else {
                    continue;
                };
                if attr(&of, "state").text() != Some("running") || !draining.contains(&&worker) {
                    continue;
                }
                // Spread over the least loaded, in turn.
                let (_, to) = &serving[at % serving.len()];
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
