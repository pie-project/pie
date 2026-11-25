//! The worker policy pie runs when `[runtime] worker_policy` names none. It
//! writes only what differs from the columns' own defaults:
//!
//! - The cache a process leaves when it exits is kept on the device and
//!   nowhere else: `retention/tiers {ws, all} = [device]`. When the device
//!   wants its room it is dropped, not moved down, unless another policy
//!   keys it there. Live processes and resting handles keep the column's
//!   default (every tier), so a suspended or parked set still moves down.
//! - Among that cache, its decoded tail goes before its prompt:
//!   `retention/priority {ws, all} = -1` and `{ws, prompt} = 0`. Equal
//!   priorities fall to the engine's own order, least recently used.
//! - Seat-to-end: while the device room cannot hold the running processes
//!   to their ends (more want to run than the worker's attr
//!   `pie/lanes-to-end`, the full-context lanes its pools hold at once), a
//!   process waits to start: `schedule/hold` defaults to true, every
//!   started process has a `false` row, and the waiting ones are let go
//!   first come, first served as lanes free. Where the attr
//!   `pie/starts-spread` says starting together runs short (a pool with no
//!   host tier, where a lane run short restarts its process), they are let
//!   go one a step apart. Below the bound no hold is written: arrivals
//!   start at once.
//!
//! It writes no schedule priority rows: processes run in the engine's own
//! order (first come, first served). `noop` is the policy that writes no
//! rows.

use plex::prelude::*;
use worker_defaults::{Defaults, wake};

#[derive(Default)]
struct Default {
    defaults: Defaults,
}

impl Policy for Default {
    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.defaults.cache.on_exited(exit);
    }

    fn on_retained(&mut self, retention: &Retention, _now: Mark) {
        self.defaults.cache.on_retained(retention);
    }

    fn plan(&mut self, now: Mark) {
        wake([self.defaults.plan(now, |_| 0)]);
    }
}

plex::export!(Default);
