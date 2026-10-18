//! Earliest deadline first, `deadline-ms` after arrival; processes without one
//! run after those with one.

use policy::{Subject, SubjectKind, attr, candidates, prop, worker};

#[derive(Default)]
struct Deadline;

impl worker::Policy for Deadline {
    fn plan(&mut self, _now_ms: u64) {
        let entries: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|p| {
                let process = Subject::Process(p.clone());
                let budget = prop(&process, "deadline-ms").int()?;
                let due = attr(&process, "arrival-ms").int()?.saturating_add(budget);
                // Sooner is larger, and every deadline outranks none (0).
                Some((p, i64::MAX - due.max(0)))
            })
            .collect();
        worker::install_schedule(&entries, None);
    }
}

policy::worker!(Deadline);
