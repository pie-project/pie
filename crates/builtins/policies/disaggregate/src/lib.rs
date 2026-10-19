//! Prefill and decode on workers of their own role.

use std::collections::BTreeMap;

use policy::{Placement, Subject, SubjectKind, attr, candidates, cluster, prop};

#[derive(Default)]
struct Disaggregate;

impl cluster::Policy for Disaggregate {
    fn plan(&mut self, _now_ms: u64) {
        let of_worker = |worker: &str, name: &str| attr(&Subject::Worker(worker.to_string()), name);
        let mut load: BTreeMap<String, (bool, usize)> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|w| {
                of_worker(w, "draining").boolean() != Some(true)
                    && of_worker(w, "pool").boolean() != Some(true)
            })
            .map(|w| {
                let prefills = of_worker(&w, "role").text() == Some("prefill");
                (w, (prefills, 0))
            })
            .collect();
        let mut waiting = Vec::new();
        for process in candidates(SubjectKind::Process) {
            let of = Subject::Process(process.clone());
            if attr(&of, "state").text() == Some("running") {
                if let Some(worker) = attr(&of, "worker").text()
                    && let Some((_, running)) = load.get_mut(worker)
                {
                    *running += 1;
                }
            } else {
                waiting.push(process);
            }
        }
        let mut prefills: Vec<(usize, String)> = load
            .iter()
            .filter(|(_, (prefills, _))| *prefills)
            .map(|(w, (_, running))| (*running, w.clone()))
            .collect();
        prefills.sort();
        let executors: Vec<String> = prefills.into_iter().map(|(_, w)| w).collect();
        let mut placements = Vec::new();
        for process in waiting {
            let named = prop(&Subject::Process(process.clone()), "worker");
            let named = named
                .int()
                .map(|n| n.to_string())
                .or(named.text().map(str::to_string));
            let worker = named.or_else(|| {
                let (worker, (_, running)) = load
                    .iter_mut()
                    .filter(|(_, (prefills, _))| !*prefills)
                    .min_by_key(|(_, (_, running))| *running)?;
                *running += 1;
                Some(worker.clone())
            });
            placements.push(Placement {
                process,
                worker,
                executors: executors.clone(),
                props: Vec::new(),
            });
        }
        let _ = cluster::install_placement(&placements);
    }
}

policy::cluster!(Disaggregate);
