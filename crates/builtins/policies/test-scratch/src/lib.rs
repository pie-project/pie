//! Counts its calls in scratch and keys every process by the count, so a
//! test can see scratch commit, survive a restart, and a table follow it.

use policy::{SubjectKind, Value, candidates, engine, get, put, set_alarm};

#[derive(Default)]
struct Scratch;

impl engine::Policy for Scratch {
    fn plan(&mut self, now_ms: u64) {
        let calls = get("calls").and_then(|v| v.int()).unwrap_or(0) + 1;
        put("calls", Some(&Value::Int(calls)));
        let entries: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .map(|process| (process, calls))
            .collect();
        let _ = engine::install_schedule(&entries, None);
        set_alarm(now_ms + 1000);
    }
}

policy::engine!(Scratch);
