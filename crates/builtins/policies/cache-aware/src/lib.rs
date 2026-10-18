//! Sends a launch where its prompt is cached, unless that piles turns on
//! one worker (after SGLang's cache-aware router).
//!
//! The gateway knows what it sent where: `cached-tokens/<worker>` is how
//! much of a launch's prompt went to that worker before. A launch goes to
//! the worker holding the most of it, if that is at least
//! [`CACHE_THRESHOLD`] of the prompt; else to the least loaded. When the
//! busiest worker runs [`BALANCE_ABS`] more turns than the idlest and
//! [`BALANCE_REL`] times as many, load wins over cache. A `worker` prop
//! names where a launch starts.

use std::collections::{BTreeMap, HashMap};

use policy::{Exit, Id, Placement, Subject, SubjectKind, attr, candidates, cluster, prop};

/// Of a prompt, the share a worker must hold for the launch to follow it.
const CACHE_THRESHOLD: f64 = 0.3;
/// Turns more on the busiest worker than on the idlest, and ...
const BALANCE_ABS: i64 = 8;
/// ... times as many, past which a launch goes to the idlest.
const BALANCE_REL: f64 = 1.5;

#[derive(Default)]
struct CacheAware {
    /// Where each waiting launch was placed: kept, so it does not flip
    /// between plans.
    placed: HashMap<Id, String>,
}

impl cluster::Policy for CacheAware {
    fn on_dispatched(&mut self, dispatch: &policy::Dispatch) {
        self.placed.remove(&dispatch.process);
    }

    fn on_exited(&mut self, exit: &Exit) {
        self.placed.remove(&exit.process);
    }

    fn on_worker_removed(&mut self, worker: &Id) {
        self.placed.retain(|_, w| w != worker);
    }

    fn plan(&mut self, _now_ms: u64) {
        let is = |worker: &str, name: &str| {
            attr(&Subject::Worker(worker.to_string()), name).boolean() == Some(true)
        };
        let mut load: BTreeMap<String, i64> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|w| !is(w, "draining") && !is(w, "pool"))
            .map(|w| (w, 0))
            .collect();
        if load.is_empty() {
            return;
        }
        let mut waiting = Vec::new();
        for process in candidates(SubjectKind::Process) {
            let of = Subject::Process(process.clone());
            if attr(&of, "state").text() == Some("running") {
                if let Some(n) = attr(&of, "worker").text().and_then(|w| load.get_mut(w)) {
                    *n += 1;
                }
            } else {
                waiting.push(process);
            }
        }
        // Launches placed before count where they go.
        self.placed.retain(|p, _| waiting.contains(p));
        for worker in self.placed.values() {
            if let Some(n) = load.get_mut(worker) {
                *n += 1;
            }
        }

        let mut placements = Vec::new();
        for process in waiting {
            let of = Subject::Process(process.clone());
            let worker = match self.placed.get(&process) {
                Some(worker) if load.contains_key(worker) => worker.clone(),
                _ => {
                    let named = prop(&of, "worker");
                    let worker = named
                        .int()
                        .map(|n| n.to_string())
                        .or(named.text().map(str::to_string))
                        .unwrap_or_else(|| choose(&of, &load));
                    *load.entry(worker.clone()).or_default() += 1;
                    self.placed.insert(process.clone(), worker.clone());
                    worker
                }
            };
            placements.push(Placement {
                process,
                worker: Some(worker),
                executors: Vec::new(),
                props: Vec::new(),
            });
        }
        let _ = cluster::install_placement(&placements);
    }
}

fn choose(of: &Subject, load: &BTreeMap<String, i64>) -> String {
    let idlest = || {
        load.iter()
            .min_by_key(|(w, n)| (**n, (*w).clone()))
            .map(|(w, _)| w.clone())
            .expect("a worker")
    };
    let (lo, hi) = (
        *load.values().min().unwrap_or(&0),
        *load.values().max().unwrap_or(&0),
    );
    if hi - lo > BALANCE_ABS && hi as f64 > BALANCE_REL * lo as f64 {
        return idlest();
    }
    let prompt = attr(of, "prompt-tokens").int().unwrap_or(0);
    let best = load
        .iter()
        .map(|(w, n)| {
            let held = attr(of, &format!("cached-tokens/{w}")).int().unwrap_or(0);
            (held, -n, w.clone())
        })
        .max();
    match best {
        Some((held, _, w)) if prompt > 0 && held as f64 >= CACHE_THRESHOLD * prompt as f64 => w,
        _ => idlest(),
    }
}

policy::cluster!(CacheAware);
