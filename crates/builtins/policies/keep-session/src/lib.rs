//! Keeps a completed process's KV for its session's next turn (after
//! Continuum), over what `default` does.
//!
//! - Hint `keep-ms` (int, above 0): the KV the process leaves as it
//!   completes is kept above every other for that long, on any tier, and
//!   dropped then. A session keeps only its latest turn: the one before is
//!   cache a process left, as any other.

use std::collections::HashMap;

use plex::prelude::*;
use worker_defaults::{Defaults, IdleCache, wake};

/// Above any key this worker gives a working set named nowhere here.
const KEEP: i64 = 1 << 40;

struct Kept {
    session: Option<SessionId>,
    /// Its prompt tokens, for the rows it has once it is not kept.
    prompt: Option<u32>,
    due: Mark,
    /// Which of the sets kept came first.
    nth: u64,
}

#[derive(Default)]
struct KeepSession {
    defaults: Defaults,
    /// Completed processes not yet retained, and how long to keep them.
    completed: HashMap<ProcessId, Duration>,
    kept: HashMap<WorkingSetId, Kept>,
    nth: u64,
}

impl Policy for KeepSession {
    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.defaults.cache.on_exited(exit);
        let keep = hint(&exit.process, "keep-ms")
            .int()
            .and_then(|ms| u64::try_from(ms).ok())
            .filter(|&ms| ms > 0 && exit.outcome == Outcome::Completed);
        if let Some(ms) = keep {
            self.completed
                .insert(exit.process.clone(), ms.saturating_mul(MILLISECOND));
        }
    }

    fn on_retained(&mut self, retention: &Retention, now: Mark) {
        let set = &retention.working_set;
        let Some(keep) = self.completed.remove(set) else {
            self.defaults.cache.on_retained(retention);
            return;
        };
        let prompt = self.defaults.cache.left(retention).flatten();
        self.nth += 1;
        self.kept.insert(
            set.clone(),
            Kept {
                session: snapshot::get_working_set(set).and_then(|set| set.session),
                prompt,
                due: now.saturating_add(keep),
                nth: self.nth,
            },
        );
        retention::priority::set(&Prefix::all(set.as_str()), KEEP).ok();
    }

    fn on_evicted(&mut self, eviction: &Eviction, _now: Mark) {
        self.kept.remove(&eviction.working_set);
    }

    fn plan(&mut self, now: Mark) {
        let again = self.defaults.plan(now, |_| 0);
        self.completed
            .retain(|process, _| self.defaults.cache.awaits(process));

        let mut latest: HashMap<SessionId, u64> = HashMap::new();
        for kept in self.kept.values() {
            if let Some(session) = &kept.session {
                let nth = latest.entry(session.clone()).or_default();
                *nth = (*nth).max(kept.nth);
            }
        }
        self.kept.retain(|set, kept| {
            let whole = Prefix::all(set.as_str());
            if kept.due <= now {
                retention::priority::delete(&whole);
                retention::tiers::set(&whole, &[]).ok();
                return false;
            }
            let newest = kept.session.as_ref().is_none_or(|s| latest[s] == kept.nth);
            if !newest {
                IdleCache::write(set, kept.prompt);
            }
            newest
        });
        wake([again, self.kept.values().map(|kept| kept.due).min()]);
    }
}

plex::export!(KeepSession);
