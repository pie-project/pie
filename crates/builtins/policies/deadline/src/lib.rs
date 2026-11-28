//! Earliest deadline first, over what `default` does.
//!
//! - Hint `deadline-ms` (int): the process is due that long after it
//!   arrives, and the soonest due runs first. A process without one runs
//!   after every process with one.
//!
//! Starts that seat-to-end holds are let go in the same order.

use plex::prelude::*;
use worker_defaults::{Defaults, wake};

#[derive(Default)]
struct Deadline {
    defaults: Defaults,
}

/// Sooner is larger, and any is larger than 0, the key of no deadline.
fn key(process: &snapshot::Process) -> Option<i64> {
    let ms = hint(&process.id, "deadline-ms").int()?;
    let budget = u64::try_from(ms).unwrap_or(0).saturating_mul(MILLISECOND);
    let due = process.arrival.saturating_add(budget);
    Some(i64::MAX - i64::try_from(due).unwrap_or(i64::MAX - 1))
}

impl Policy for Deadline {
    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.defaults.cache.on_exited(exit);
    }

    fn on_retained(&mut self, retention: &Retention, _now: Mark) {
        self.defaults.cache.on_retained(retention);
    }

    fn plan(&mut self, now: Mark) {
        let rows: Vec<(ProcessId, i64)> = snapshot::get_processes()
            .into_iter()
            .filter_map(|process| key(&process).map(|key| (process.id, key)))
            .collect();
        schedule::priority::clear();
        schedule::priority::set_list(&rows).ok();
        wake([self.defaults.plan(now, |process| key(process).unwrap_or(0))]);
    }
}

plex::export!(Deadline);
