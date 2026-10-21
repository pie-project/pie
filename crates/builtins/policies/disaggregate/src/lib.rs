//! Names the cluster's prefill workers as every process's executors, the
//! least loaded first, and leaves where the process runs to the gateway's
//! default: its passes go to them as the worker plane takes them up.

use policy::{Placement, Subject, SubjectKind, attr, candidates, cluster};

#[derive(Default)]
struct Disaggregate;

impl cluster::Policy for Disaggregate {
    fn plan(&mut self, _now_ms: u64) {
        let mut prefill: Vec<(i64, String)> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|w| {
                let worker = Subject::Worker(w.clone());
                attr(&worker, "role").text() == Some("prefill")
                    && attr(&worker, "draining").boolean() != Some(true)
            })
            .map(|w| {
                let load = attr(&Subject::Worker(w.clone()), "running")
                    .int()
                    .unwrap_or(0);
                (load, w)
            })
            .collect();
        prefill.sort();
        let executors: Vec<String> = prefill.into_iter().map(|(_, w)| w).collect();
        let placements: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .map(|process| Placement {
                process,
                worker: None,
                executors: executors.clone(),
                props: Vec::new(),
            })
            .collect();
        cluster::install_placement(&placements);
    }
}

policy::cluster!(Disaggregate);
