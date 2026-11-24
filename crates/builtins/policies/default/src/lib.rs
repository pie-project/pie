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
//!
//! It writes no schedule rows: processes run in the engine's own order
//! (first come, first served). `noop` is the policy that writes no rows.

use std::collections::HashMap;

use plex::prelude::*;

#[derive(Default)]
struct Default {
    /// Processes that exited and may yet be retained: their prompt tokens,
    /// where the host counted them. A working set retained without an exit
    /// (a resting handle) is not cache a process left, and keeps every tier.
    exited: HashMap<ProcessId, Option<u32>>,
}

impl Policy for Default {
    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        let prompt = snapshot::get_process(&exit.process).and_then(|p| p.prompt_tokens);
        self.exited.insert(exit.process.clone(), prompt);
    }

    fn on_retained(&mut self, retention: &Retention, _now: Mark) {
        // A process's working set has the process's id.
        let Some(prompt) = self.exited.remove(&retention.working_set) else {
            return;
        };
        let set = &retention.working_set;
        let mut keys = vec![(Prefix::all(set.as_str()), -1)];
        if let Some(prompt) = prompt.filter(|&p| p > 0) {
            keys.push((Prefix::upto(set.as_str(), prompt), 0));
        }
        retention::priority::set_list(&keys).ok();
        retention::tiers::set(&Prefix::all(set.as_str()), &[Tier::Device]).ok();
    }

    fn plan(&mut self, _now: Mark) {
        // A process is in the snapshot through the call that delivers its
        // exit, and its retention comes no later than the next call's: one
        // gone that left nothing cached is forgotten.
        self.exited
            .retain(|process, _| snapshot::get_process(process).is_some());
    }
}

plex::export!(Default);
