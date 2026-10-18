//! Sends a launch to the worker its `worker` prop names.

use policy::{Placement, Subject, SubjectKind, candidates, cluster, prop};

#[derive(Default)]
struct Pin;

impl cluster::Policy for Pin {
    fn plan(&mut self, _now_ms: u64) {
        let placements: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|process| {
                let named = prop(&Subject::Process(process.clone()), "worker");
                let worker = match named.int() {
                    Some(n) => n.to_string(),
                    None => named.text()?.to_string(),
                };
                Some(Placement {
                    process,
                    worker: Some(worker),
                    executors: Vec::new(),
                    props: Vec::new(),
                })
            })
            .collect();
        let _ = cluster::install_placement(&placements);
    }
}

policy::cluster!(Pin);
