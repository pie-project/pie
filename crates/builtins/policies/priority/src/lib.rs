//! Orders a worker's processes by the priority their client states, over
//! what `default` does.
//!
//! - Hint `priority` (int): larger runs first. A process without one is 0.
//!
//! Starts that seat-to-end holds are let go in the same order.

use plex::prelude::*;
use worker_defaults::{Defaults, wake};

#[derive(Default)]
struct Priority {
    defaults: Defaults,
}

fn key(process: &snapshot::Process) -> Option<i64> {
    hint(&process.id, "priority").int()
}

impl Policy for Priority {
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

plex::export!(Priority);
