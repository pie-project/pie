//! Fair share across tenants by running time (VTC, OSDI '24), over what
//! `default` does: the tenant that has had the least runs first, and one
//! newly active starts at the least an active one has had, so idle time is
//! no credit.
//!
//! Starts that seat-to-end holds are let go in the same order.

use std::collections::{HashMap, HashSet};

use plex::prelude::*;
use worker_defaults::{Defaults, wake};

/// While a process runs, how often the shares are counted again.
const TICK: Duration = 250 * MILLISECOND;

#[derive(Default)]
struct Fair {
    defaults: Defaults,
    /// Each process's running time as last counted.
    seen: HashMap<ProcessId, Duration>,
    /// Each tenant's running time.
    served: HashMap<String, Duration>,
    /// The tenants with a process, as of the last call.
    active: HashSet<String>,
}

impl Fair {
    fn count(&mut self, processes: &[snapshot::Process]) {
        let mut seen = HashMap::new();
        for process in processes {
            let before = self.seen.get(&process.id).copied().unwrap_or(0);
            *self.served.entry(process.tenant().to_string()).or_default() +=
                process.service.saturating_sub(before);
            if process.live() {
                seen.insert(process.id.clone(), process.service);
            }
        }
        self.seen = seen;

        let active: HashSet<String> = processes
            .iter()
            .filter(|process| process.live())
            .map(|process| process.tenant().to_string())
            .collect();
        let least = self
            .active
            .intersection(&active)
            .map(|tenant| self.served.get(tenant).copied().unwrap_or(0))
            .min();
        if let Some(least) = least {
            for tenant in active.difference(&self.active) {
                let served = self.served.entry(tenant.clone()).or_default();
                *served = (*served).max(least);
            }
        }
        self.active = active;
    }

    fn key(&self, process: &snapshot::Process) -> i64 {
        let served = self.served.get(process.tenant()).copied().unwrap_or(0);
        -i64::try_from(served).unwrap_or(i64::MAX)
    }
}

impl Policy for Fair {
    fn on_exited(&mut self, exit: &Exit, _now: Mark) {
        self.defaults.cache.on_exited(exit);
    }

    fn on_retained(&mut self, retention: &Retention, _now: Mark) {
        self.defaults.cache.on_retained(retention);
    }

    fn plan(&mut self, now: Mark) {
        let processes = snapshot::get_processes();
        self.count(&processes);
        let keys: HashMap<ProcessId, i64> = processes
            .iter()
            .filter(|process| process.live())
            .map(|process| (process.id.clone(), self.key(process)))
            .collect();
        let rows: Vec<(ProcessId, i64)> = keys.clone().into_iter().collect();
        schedule::priority::clear();
        schedule::priority::set_list(&rows).ok();
        let again = self
            .defaults
            .plan(now, |process| keys.get(&process.id).copied().unwrap_or(0));
        let running = processes.iter().any(|process| process.running());
        wake([again, running.then(|| now.saturating_add(TICK))]);
    }
}

plex::export!(Fair);
