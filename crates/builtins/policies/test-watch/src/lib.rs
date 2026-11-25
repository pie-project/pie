//! Writes what each call saw as the worker's note `seen`: the events it
//! heard, then the snapshot's processes and working sets, and every hint a
//! change has named, a line each. For tests of what a port reports.

use plex::prelude::*;

#[derive(Default)]
struct Watch {
    calls: u64,
    heard: Vec<String>,
    hints: Vec<(Subject, String)>,
}

fn spelled(of: &Subject) -> String {
    match of {
        Subject::Global => "global".to_string(),
        Subject::Session(id) => format!("session:{id}"),
        Subject::Process(id) => format!("process:{id}"),
        Subject::WorkingSet(id) => format!("working-set:{id}"),
        Subject::Worker(id) => format!("worker:{id}"),
    }
}

fn shown(value: Option<Value>) -> String {
    match value {
        None => "none".to_string(),
        Some(value) => format!("{value:?}"),
    }
}

impl Policy for Watch {
    fn on_arrived(&mut self, process: &ProcessId, _now: Mark) {
        self.heard.push(format!("arrived {process}"));
    }

    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.heard.push(format!("exited {}", exit.process));
    }

    fn on_closed(&mut self, session: &SessionId, _now: Mark) {
        self.heard.push(format!("closed {session}"));
    }

    fn on_retained(&mut self, retention: &Retention, _now: Mark) {
        self.heard.push(format!("retained {}", retention.working_set));
    }

    fn on_evicted(&mut self, eviction: &Eviction, _now: Mark) {
        self.heard
            .push(format!("evicted {} by {:?}", eviction.working_set, eviction.by));
    }

    fn on_hint_changed(&mut self, change: &Change, _now: Mark) {
        self.heard
            .push(format!("hint {} {}", spelled(&change.subject), change.name));
        let named = (change.subject.clone(), change.name.clone());
        if !self.hints.contains(&named) {
            self.hints.push(named);
        }
    }

    fn plan(&mut self, _now: Mark) {
        self.calls += 1;
        let mut seen = vec![format!("call {}", self.calls)];
        seen.append(&mut self.heard);
        for p in snapshot::get_processes() {
            seen.push(format!(
                "process {} session={} tenant={} state={:?} priority={}",
                p.id,
                p.session.clone().unwrap_or_default(),
                p.tenant(),
                p.state,
                shown(plex::hint(&p.id, "priority")),
            ));
        }
        for ws in snapshot::get_working_sets() {
            let of = Subject::WorkingSet(ws.id.clone());
            seen.push(format!(
                "working-set {} owner={} session={} tokens={} folded={}",
                ws.id,
                ws.owner.unwrap_or_default(),
                ws.session.unwrap_or_default(),
                ws.tokens,
                shown(snapshot::get_attr(&of, "pie/folded-tokens")),
            ));
        }
        for (of, name) in &self.hints {
            seen.push(format!("has {} {name}={}", spelled(of), shown(hints::get(of, name))));
        }
        let worker = Subject::Worker("self".to_string());
        notes::set(&worker, "seen", &Value::Text(seen.join("\n"))).ok();
    }
}

plex::export!(Watch);
