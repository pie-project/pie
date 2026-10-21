//! Orders, holds, places and ends launches by their props, so a test can see
//! the cluster host enact each table and verb.

use policy::{Placement, Subject, SubjectKind, Value, attr, candidates, cluster, prop, terminate};

#[derive(Default)]
struct Route;

impl cluster::Policy for Route {
    fn plan(&mut self, _now_ms: u64) {
        let processes = candidates(SubjectKind::Process);
        let keys: Vec<_> = processes
            .iter()
            .map(|p| {
                let key = prop(&Subject::Process(p.clone()), "key").int();
                (p.clone(), key.unwrap_or(1))
            })
            .collect();
        let placements: Vec<_> = processes
            .iter()
            .map(|p| Placement {
                process: p.clone(),
                worker: prop(&Subject::Process(p.clone()), "worker")
                    .text()
                    .map(str::to_string),
                // Stated, or what `test-share` published of the worker
                // this program placed it on.
                executors: prop(&Subject::Process(p.clone()), "executors")
                    .text()
                    .map(str::to_string)
                    .or_else(|| {
                        attr(&Subject::Process(p.clone()), "published/test-share/placed")
                            .text()
                            .map(str::to_string)
                    })
                    .map(|ids| ids.split(',').map(str::to_string).collect())
                    .unwrap_or_default(),
                props: vec![("from".to_string(), Value::Text(p.clone()))],
            })
            .collect();
        cluster::install_schedule(&keys, Some(1));
        cluster::install_placement(&placements);
        for p in &processes {
            let process = Subject::Process(p.clone());
            if let Some(reason) = prop(&process, "refuse").text() {
                terminate(p, reason);
            }
            if let Some(mark) = prop(&process, "publish").text() {
                cluster::publish(&process, "mark", Some(&Value::Text(mark.to_string())));
            }
            let running = attr(&process, "state").text() == Some("running");
            if running && prop(&process, "stop").boolean() == Some(true) {
                terminate(p, "stopped");
            }
        }
    }
}

policy::cluster!(Route);
