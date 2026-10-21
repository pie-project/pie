//! Names the cluster's prefill workers as every process's executors: those on
//! the host where the process is placed first, then the least loaded. Where
//! the process runs is another program's write (`placement/worker`, read as
//! it was committed), or the gateway's default; its passes go to these as the
//! worker plane takes them up.

use policy::{Placement, Subject, SubjectKind, attr, candidates, cluster};

#[derive(Default)]
struct Disaggregate;

impl cluster::Policy for Disaggregate {
    fn plan(&mut self, _now_ms: u64) {
        let mut prefill: Vec<(i64, String, Option<String>)> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|w| {
                let worker = Subject::Worker(w.clone());
                attr(&worker, "role").text() == Some("prefill")
                    && attr(&worker, "draining").boolean() != Some(true)
            })
            .map(|w| {
                let worker = Subject::Worker(w.clone());
                let load = attr(&worker, "running").int().unwrap_or(0);
                let host = attr(&worker, "host").text().map(str::to_string);
                (load, w, host)
            })
            .collect();
        prefill.sort();
        let placements: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .map(|process| {
                let home = attr(&Subject::Process(process.clone()), "placement/worker")
                    .text()
                    .and_then(|worker| {
                        attr(&Subject::Worker(worker.to_string()), "host")
                            .text()
                            .map(str::to_string)
                    });
                // Stable: within each side, least loaded first.
                let mut executors: Vec<&(i64, String, Option<String>)> = prefill.iter().collect();
                executors.sort_by_key(|(_, _, host)| home.is_none() || *host != home);
                Placement {
                    process,
                    worker: None,
                    executors: executors.into_iter().map(|(_, w, _)| w.clone()).collect(),
                    props: Vec::new(),
                }
            })
            .collect();
        cluster::install_placement(&placements);
    }
}

policy::cluster!(Disaggregate);
