//! Orders launches by their client's `priority` and passes it on as
//! `cluster/priority`.

use policy::{Placement, Subject, SubjectKind, Value, candidates, cluster, prop};

#[derive(Default)]
struct ClusterPriority;

impl cluster::Policy for ClusterPriority {
    fn plan(&mut self, _now_ms: u64) {
        let ranked: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|p| {
                let priority = prop(&Subject::Process(p.clone()), "priority").int()?;
                Some((p, priority))
            })
            .collect();
        let placements: Vec<_> = ranked
            .iter()
            .map(|(p, priority)| Placement {
                process: p.clone(),
                worker: None,
                executors: Vec::new(),
                props: vec![("priority".to_string(), Value::Int(*priority))],
            })
            .collect();
        cluster::install_schedule(&ranked, None);
        cluster::install_placement(&placements);
    }
}

policy::cluster!(ClusterPriority);
