//! Orders launches by their client's `priority` and passes it on as
//! `router/priority`.

use policy::{Placement, Subject, SubjectKind, Value, candidates, prop, router};

#[derive(Default)]
struct RouterPriority;

impl router::Policy for RouterPriority {
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
                props: vec![("priority".to_string(), Value::Int(*priority))],
            })
            .collect();
        router::install_schedule(&ranked, None);
        router::install_placement(&placements);
    }
}

policy::router!(RouterPriority);
