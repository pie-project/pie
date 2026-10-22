//! Publishes, for every process, the worker another program's committed
//! placement names, so a test can see one cluster program read another's
//! decision and answer it under its own name.

use policy::{Subject, SubjectKind, Value, attr, candidates, cluster};

#[derive(Default)]
struct Share;

impl cluster::Policy for Share {
    fn plan(&mut self, _now_ms: u64) {
        for process in candidates(SubjectKind::Process) {
            let subject = Subject::Process(process);
            let placed = attr(&subject, "placement/worker")
                .text()
                .map(|worker| Value::Text(worker.to_string()));
            cluster::publish(&subject, "placed", placed.as_ref());
        }
    }
}

policy::cluster!(Share);
