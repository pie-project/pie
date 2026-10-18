//! Keeps a completed process's working set for its `keep-ms`, between a
//! session's turns (after Continuum).

use std::collections::HashMap;

use policy::{By, Change, Exit, Id, Outcome, Retain, Subject, attr, prop, set_alarm, worker};

/// Above any key the worker gives a working set this policy does not name.
const KEEP: i64 = 1 << 40;

struct Kept {
    session: Option<String>,
    due_ms: Option<u64>,
    keep_ms: u64,
}

#[derive(Default)]
struct KeepSession {
    keep_ms: HashMap<Id, u64>,
    completed: HashMap<Id, u64>,
    kept: HashMap<Id, Kept>,
}

impl KeepSession {
    fn read(&mut self, process: &Id) {
        match prop(&Subject::Process(process.clone()), "keep-ms").int() {
            Some(ms) if ms > 0 => {
                self.keep_ms.insert(process.clone(), ms as u64);
            }
            _ => {
                self.keep_ms.remove(process);
            }
        }
    }
}

impl worker::Policy for KeepSession {
    fn on_arrived(&mut self, process: &Id) {
        self.read(process);
    }

    fn on_prop_changed(&mut self, change: &Change) {
        if let Subject::Process(process) = &change.subject
            && change.names.iter().any(|n| n == "keep-ms")
        {
            self.read(process);
        }
    }

    fn on_exited(&mut self, exit: &Exit) {
        if let Some(ms) = self.keep_ms.remove(&exit.process)
            && exit.outcome == Outcome::Completed
        {
            self.completed.insert(exit.process.clone(), ms);
        }
    }

    fn on_retained(&mut self, working_set: &Id) {
        // A working-set has its owner's id.
        if let Some(keep_ms) = self.completed.remove(working_set) {
            let session = attr(&Subject::WorkingSet(working_set.clone()), "session")
                .text()
                .map(str::to_string);
            self.kept.insert(
                working_set.clone(),
                Kept {
                    session,
                    due_ms: None,
                    keep_ms,
                },
            );
        }
    }

    fn on_discarded(&mut self, working_set: &Id, _by: By) {
        self.kept.remove(working_set);
    }

    fn plan(&mut self, now_ms: u64) {
        self.completed.clear();
        // A session keeps only its latest turn.
        let mut latest: HashMap<String, (u64, Id)> = HashMap::new();
        for (ws, kept) in &mut self.kept {
            let due = *kept.due_ms.get_or_insert(now_ms + kept.keep_ms);
            if let Some(session) = &kept.session {
                let newest = latest.entry(session.clone()).or_insert((due, ws.clone()));
                if due > newest.0 {
                    *newest = (due, ws.clone());
                }
            }
        }
        self.kept.retain(|ws, kept| {
            kept.session
                .as_ref()
                .is_none_or(|session| latest[session].1 == *ws)
        });

        let (due, kept): (Vec<_>, Vec<_>) = self
            .kept
            .iter()
            .partition(|(_, kept)| kept.due_ms.is_some_and(|due| due <= now_ms));
        for (ws, _) in &due {
            worker::discard(ws);
        }
        let entries: Vec<_> = kept
            .iter()
            .map(|(ws, _)| Retain::all((*ws).clone(), KEEP))
            .collect();
        let next = kept.iter().filter_map(|(_, k)| k.due_ms).min();
        worker::install_retention(&entries);
        if let Some(next) = next {
            set_alarm(next);
        }
        let discarded: Vec<Id> = due.into_iter().map(|(ws, _)| ws.clone()).collect();
        for ws in discarded {
            self.kept.remove(&ws);
        }
    }
}

policy::worker!(KeepSession);
