//! Places a launch by its `worker` prop, and moves it to `move-to` once it
//! has run `move-after-ms`.

use std::collections::HashMap;

use policy::{Id, Placement, Subject, SubjectKind, candidates, cluster, prop, set_alarm};

#[derive(Default)]
struct Relocate {
    /// When each process arrived: stamped by the call that heard it, whose
    /// clock is `plan`'s.
    arrived: HashMap<Id, u64>,
}

fn worker(of: &Subject, name: &str) -> Option<String> {
    let named = prop(of, name);
    match named.int() {
        Some(n) => Some(n.to_string()),
        None => named.text().map(str::to_string),
    }
}

impl cluster::Policy for Relocate {
    fn on_exited(&mut self, exit: &policy::Exit) {
        self.arrived.remove(&exit.process);
    }

    fn plan(&mut self, now_ms: u64) {
        let mut next_alarm: Option<u64> = None;
        let placements: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|process| {
                let of = Subject::Process(process.clone());
                let arrived = *self.arrived.entry(process.clone()).or_insert(now_ms);
                let mut at = worker(&of, "worker");
                if let (Some(to), Some(after)) =
                    (worker(&of, "move-to"), prop(&of, "move-after-ms").int())
                {
                    let due = arrived + after.max(0) as u64;
                    if now_ms >= due {
                        at = Some(to);
                    } else {
                        next_alarm = Some(next_alarm.map_or(due, |a| a.min(due)));
                    }
                }
                Some(Placement {
                    process,
                    worker: Some(at?),
                    executors: Vec::new(),
                    props: Vec::new(),
                })
            })
            .collect();
        let _ = cluster::install_placement(&placements);
        if let Some(at) = next_alarm {
            set_alarm(at);
        }
    }
}

policy::cluster!(Relocate);
