//! A cluster probe: places each process as its hints say, so a test can
//! drive the router's placement column, its gate and its notes from a
//! client.
//!
//! - `to`, on the process's session, else on the process: the workers of
//!   its `placement/workers` row. A worker id, or several separated by
//!   commas (brackets and quotes are ignored, so a JSON list reads too);
//!   empty text is the empty row, which holds it at the gateway. Without
//!   `to` the process has no row. A session's `to` set between requests
//!   changes the row of a process already there.
//! - `refuse` (text): terminate it, with that reason, while it is pending.
//! - `stop` (bool): terminate it once it runs, with reason `stopped`.
//! - `say`, `rank`: noted on the process as `said` and `rank`.
//! - `say-session`, `say-global`: noted as `said` on its session, on
//!   `global`.
//!
//! It notes on each pending process, under `seen`, the working set it saw
//! (`<prompt-tokens>/<tokens>:<worker>=<tokens held>,...`), and on
//! `global`, under `refused`, how many of its rows the router refuses.

use plex::prelude::*;

#[derive(Default)]
struct Route {
    refused: usize,
}

fn workers(value: &Value) -> Option<Vec<WorkerId>> {
    match value {
        Value::Int(i) => Some(vec![i.to_string()]),
        Value::Text(text) => Some(
            text.split(',')
                .map(|w| w.trim_matches(|c| " []\"".contains(c)))
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect(),
        ),
        _ => None,
    }
}

impl Policy for Route {
    fn plan(&mut self, _now: Mark) {
        let mut rows = Vec::new();
        for p in snapshot::get_processes().iter().filter(|p| p.live()) {
            let process = Subject::Process(p.id.clone());
            let to = p
                .session
                .as_deref()
                .and_then(|s| session_hint(s, "to"))
                .or_else(|| hint(&p.id, "to"));
            if let Some(list) = to.as_ref().and_then(workers) {
                rows.push((p.id.clone(), list));
            }
            if p.state == ProcessState::Pending {
                if let Some(Value::Text(reason)) = hint(&p.id, "refuse") {
                    terminate::terminate(&p.id, &reason).ok();
                }
                if let Some(set) = snapshot::get_working_set(&p.id) {
                    let copies: Vec<String> = set
                        .copies
                        .iter()
                        .map(|c| format!("{}={}", c.worker, c.tokens))
                        .collect();
                    let seen = format!(
                        "{}/{}:{}",
                        p.prompt_tokens.unwrap_or(0),
                        set.tokens,
                        copies.join(",")
                    );
                    notes::set(&process, "seen", &Value::Text(seen)).ok();
                }
            }
            if p.running() && hint(&p.id, "stop").boolean() == Some(true) {
                terminate::terminate(&p.id, "stopped").ok();
            }
            if let Some(said) = hint(&p.id, "say") {
                notes::set(&process, "said", &said).ok();
            }
            if let Some(rank) = hint(&p.id, "rank").int() {
                notes::set(&process, "rank", &Value::Int(rank)).ok();
            }
            if let (Some(session), Some(said)) = (&p.session, hint(&p.id, "say-session")) {
                notes::set(&Subject::Session(session.clone()), "said", &said).ok();
            }
            if let Some(said) = hint(&p.id, "say-global") {
                notes::set(&Subject::Global, "said", &said).ok();
            }
        }
        placement::workers::clear();
        placement::workers::set_list(&rows).ok();
        let refused = placement::workers::get_refused().len();
        if refused != self.refused {
            self.refused = refused;
            notes::set(&Subject::Global, "refused", &Value::Int(refused as i64)).ok();
        }
    }
}

plex::export!(Route);
