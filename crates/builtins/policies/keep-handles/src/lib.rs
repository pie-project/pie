//! Orders resting handles by their `keep` prop, with the first `head` tokens
//! of one kept above that, and discards those marked `weak`, lowest `keep`
//! first, while fewer than [`ROOM`] more can rest.

use policy::{Retain, Subject, SubjectKind, attr, candidates, prop, worker};

/// Recurrent slots kept free at rest, so a launch is not refused for want of
/// them before this has run.
const ROOM: i64 = 2;
/// Past any `keep`: a head stays while its tail goes.
const HEAD: i64 = i64::MAX;

#[derive(Default)]
struct KeepHandles;

impl worker::Policy for KeepHandles {
    fn plan(&mut self, _now_ms: u64) {
        let mut sets: Vec<_> = candidates(SubjectKind::WorkingSet)
            .into_iter()
            .map(|ws| {
                let of = Subject::WorkingSet(ws.clone());
                let keep = prop(&of, "keep").int().unwrap_or(0);
                let weak = prop(&of, "weak").boolean() == Some(true);
                let head = prop(&of, "head").int().unwrap_or(0) as u32;
                (keep, ws, weak, head)
            })
            .collect();
        let table: Vec<_> = sets
            .iter()
            .flat_map(|(keep, ws, _, head)| {
                let whole = Retain::all(ws.clone(), *keep);
                let kept = (*head > 0).then(|| Retain::upto(ws.clone(), HEAD, *head));
                std::iter::once(whole).chain(kept)
            })
            .collect();
        worker::install_retention(&table);

        let worker = |name| {
            candidates(SubjectKind::Worker)
                .first()
                .and_then(|id| attr(&Subject::Worker(id.clone()), name).int())
                .unwrap_or(0)
        };
        let free = worker("swap-capacity-slots") - worker("swap-used-slots");
        if worker("swap-capacity-slots") == 0 || free >= ROOM {
            return;
        }
        sets.sort();
        for (_, ws, ..) in sets
            .into_iter()
            .filter(|(_, _, weak, _)| *weak)
            .take((ROOM - free) as usize)
        {
            worker::discard(&ws);
        }
    }
}

policy::worker!(KeepHandles);
