//! Fair share across tenants by running time (VTC, OSDI '24); a returning
//! tenant starts at the least active one's service.

use std::collections::{HashMap, HashSet};

use policy::{Id, Subject, SubjectKind, attr, candidates, engine, set_alarm};

const TICK_MS: u64 = 250;

#[derive(Default)]
struct Fair {
    served: HashMap<String, u64>,
    active: HashSet<String>,
    last_ms: Option<u64>,
}

impl engine::Policy for Fair {
    fn plan(&mut self, now_ms: u64) {
        let processes: Vec<(Id, String, bool)> = candidates(SubjectKind::Process)
            .into_iter()
            .map(|p| {
                let of = |name| attr(&Subject::Process(p.clone()), name);
                let tenant = of("tenant").text().unwrap_or_default().to_string();
                let running = of("state").text() == Some("running");
                (p, tenant, running)
            })
            .collect();

        let elapsed = self.last_ms.map_or(0, |last| now_ms.saturating_sub(last));
        self.last_ms = Some(now_ms);
        for (_, tenant, running) in &processes {
            if *running {
                *self.served.entry(tenant.clone()).or_default() += elapsed;
            }
        }

        // A tenant newly active starts no lower than the least active one.
        let active: HashSet<String> = processes.iter().map(|(_, t, _)| t.clone()).collect();
        let least = self
            .active
            .intersection(&active)
            .map(|t| self.served.get(t).copied().unwrap_or(0))
            .min();
        if let Some(least) = least {
            for tenant in active.difference(&self.active) {
                let served = self.served.entry(tenant.clone()).or_default();
                *served = (*served).max(least);
            }
        }
        self.active = active;

        let entries: Vec<_> = processes
            .iter()
            .map(|(p, tenant, _)| {
                let served = self.served.get(tenant).copied().unwrap_or(0);
                (p.clone(), -(served.min(i64::MAX as u64) as i64))
            })
            .collect();
        engine::install_schedule(&entries, None);
        if processes.iter().any(|(_, _, running)| *running) {
            set_alarm(now_ms + TICK_MS);
        }
    }
}

policy::engine!(Fair);
