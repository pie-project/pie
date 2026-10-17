//! Sends every process of a session to the worker its session's last process
//! went to.

use std::collections::HashMap;

use policy::{Dispatch, Exit, Id, Placement, Subject, SubjectKind, attr, candidates, router};

const IDLE_MS: u64 = 60 * 60 * 1000;

#[derive(Default)]
struct SessionAffinity {
    sessions: HashMap<String, (Id, u64)>,
    processes: HashMap<Id, String>,
    dispatched: Vec<(String, Id)>,
}

impl router::Policy for SessionAffinity {
    fn on_arrived(&mut self, process: &Id) {
        if let Some(session) = attr(&Subject::Process(process.clone()), "session").text() {
            self.processes.insert(process.clone(), session.to_string());
        }
    }

    fn on_dispatched(&mut self, dispatch: &Dispatch) {
        if let Some(session) = self.processes.get(&dispatch.process) {
            self.dispatched
                .push((session.clone(), dispatch.worker.clone()));
        }
    }

    fn on_exited(&mut self, exit: &Exit) {
        self.processes.remove(&exit.process);
    }

    fn on_worker_removed(&mut self, worker: &Id) {
        self.sessions.retain(|_, (w, _)| w != worker);
    }

    fn plan(&mut self, now_ms: u64) {
        for (session, worker) in self.dispatched.drain(..) {
            self.sessions.insert(session, (worker, now_ms));
        }
        self.sessions
            .retain(|_, (_, at)| now_ms.saturating_sub(*at) < IDLE_MS);

        let placements: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|p| {
                let session = self.processes.get(&p)?;
                let (worker, _) = self.sessions.get(session)?;
                Some(Placement {
                    process: p,
                    worker: Some(worker.clone()),
                    props: Vec::new(),
                })
            })
            .collect();
        router::install_placement(&placements);
    }
}

policy::router!(SessionAffinity);
