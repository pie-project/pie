//! Keys every process by `router/priority`, else its client's `priority`, else
//! 0.

use policy::{Subject, SubjectKind, candidates, engine, prop};

#[derive(Default)]
struct Priority;

impl engine::Policy for Priority {
    fn plan(&mut self, _now_ms: u64) {
        let entries: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .filter_map(|process| {
                let of = Subject::Process(process.clone());
                let key = prop(&of, "router/priority")
                    .int()
                    .or_else(|| prop(&of, "priority").int())?;
                Some((process, key))
            })
            .collect();
        // Only an ungranted write is refused, and the manifest grants it.
        engine::install_schedule(&entries, None);
    }
}

policy::engine!(Priority);
