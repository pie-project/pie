//! Moves every turn running on a draining worker to the least loaded worker
//! that is not draining.

use policy::{Placement, Subject, SubjectKind, attr, candidates, cluster};

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
