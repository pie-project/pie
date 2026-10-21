//! The controller's actuator for `cluster.scale`: it launches workers into
//! the slots its config lists, and stops the ones it launched, until each
//! shape a scale table names runs as many workers as the table says.
//!
//! Level-triggered: every tick it counts again and acts on the difference,
//! so a launch that died, a worker that left or a table that changed are
//! all the same case. A shape the table does not name keeps what it has,
//! and a worker the actuator did not launch is never stopped by it.
//!
//! A worker is stopped by SIGTERM: pie-worker's signal path drains it (no
//! new turns, the cluster policy moves its turns to others by their resume
//! logs) before it leaves. Which worker of a shape goes is the host's choice
//! for now, not a policy's: the most recently launched.

use std::collections::{HashMap, HashSet};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use controller_api::{Health, LAUNCH_ENV, ScaleEntry, ScaleStatus};
use ids::WorkerId;
use serde::Deserialize;

use crate::state::Worker;

/// A worker the actuator may launch: its shape, and the command that starts
/// it (a pie-worker with its own config: device, ports).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slot {
    /// None: a worker that loads no model.
    #[serde(default)]
    pub model: Option<String>,
    pub engine: String,
    pub command: Vec<String>,
}

type Shape = (Option<String>, String);

fn shape_of(slot: &Slot) -> Shape {
    (slot.model.clone(), slot.engine.clone())
}

fn serves(worker: &Worker, shape: &Shape) -> bool {
    let model = shape.0.as_deref().unwrap_or("");
    worker.model == model && worker.engine == shape.1
}

struct Launched {
    child: Child,
    token: String,
    at: Instant,
    stopping: bool,
}

pub struct Actuator {
    slots: Vec<Slot>,
    launched: Vec<Option<Launched>>,
    desired: Vec<ScaleEntry>,
    /// Shapes told unreachable, so each is told once until it is not.
    short: HashSet<Shape>,
    /// Tokens of launches that exited: a worker still registered under one
    /// is gone, only not yet evicted.
    gone: HashSet<String>,
    /// Makes this controller's tokens its own, apart from another's.
    epoch: u128,
    next_token: u64,
}

impl Actuator {
    #[must_use]
    pub fn new(slots: Vec<Slot>) -> Self {
        let launched = slots.iter().map(|_| None).collect();
        Self {
            slots,
            launched,
            desired: Vec::new(),
            short: HashSet::new(),
            gone: HashSet::new(),
            epoch: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
            next_token: 0,
        }
    }

    /// It has slots to launch into.
    #[must_use]
    pub fn scales(&self) -> bool {
        !self.slots.is_empty()
    }

    /// The table to converge on, whole.
    pub fn desire(&mut self, entries: Vec<ScaleEntry>) {
        self.desired = entries;
    }

    /// One step toward the table: reaps exited launches, then launches or
    /// stops what each named shape is short or over.
    pub fn converge(&mut self, workers: &HashMap<WorkerId, Worker>) {
        self.reap();
        for entry in self.desired.clone() {
            let shape = (entry.model.clone(), entry.engine.clone());
            let status = self.count(&shape, entry.workers, workers);
            let have = status.serving + status.launching;
            if have < entry.workers {
                let want = entry.workers - have;
                let launched = self.launch(&shape, want);
                if launched < want {
                    if self.short.insert(shape.clone()) {
                        tracing::warn!(
                            model = ?shape.0,
                            engine = %shape.1,
                            desired = entry.workers,
                            have = have + launched,
                            "the scale table asks for more workers of a shape than the controller has slots for"
                        );
                    }
                } else {
                    self.short.remove(&shape);
                }
            } else {
                self.short.remove(&shape);
                if have > entry.workers {
                    self.stop(&shape, have - entry.workers);
                }
            }
        }
    }

    /// Each shape the table names, desired against current.
    #[must_use]
    pub fn status(&self, workers: &HashMap<WorkerId, Worker>) -> Vec<ScaleStatus> {
        self.desired
            .iter()
            .map(|entry| {
                self.count(
                    &(entry.model.clone(), entry.engine.clone()),
                    entry.workers,
                    workers,
                )
            })
            .collect()
    }

    /// Stops every worker it launched: the controller is going.
    pub fn stop_all(&mut self) {
        for launched in self.launched.iter_mut().flatten() {
            if !launched.stopping {
                terminate(&launched.child);
                launched.stopping = true;
            }
        }
    }

    fn count(
        &self,
        shape: &Shape,
        desired: u32,
        workers: &HashMap<WorkerId, Worker>,
    ) -> ScaleStatus {
        let stopping_tokens: HashSet<&str> = self
            .launched
            .iter()
            .flatten()
            .filter(|l| l.stopping)
            .map(|l| l.token.as_str())
            .collect();
        let registered: HashSet<&str> = workers
            .values()
            .filter_map(|w| w.launch.as_deref())
            .collect();
        let serving = workers
            .values()
            .filter(|w| serves(w, shape))
            .filter(|w| !w.load.draining && w.health != Health::Unreachable)
            .filter(|w| {
                w.launch
                    .as_deref()
                    .is_none_or(|t| !stopping_tokens.contains(t) && !self.gone.contains(t))
            })
            .count() as u32;
        let ours = self
            .slots
            .iter()
            .zip(&self.launched)
            .filter(|(slot, _)| shape_of(slot) == *shape);
        let (mut launching, mut stopping, mut free) = (0, 0, 0);
        for (_, launched) in ours {
            match launched {
                None => free += 1,
                Some(l) if l.stopping => stopping += 1,
                Some(l) if !registered.contains(l.token.as_str()) => launching += 1,
                Some(_) => {}
            }
        }
        ScaleStatus {
            model: shape.0.clone(),
            engine: shape.1.clone(),
            desired,
            serving,
            launching,
            stopping,
            free,
        }
    }

    /// Launches up to `want` workers of `shape` into free slots; how many
    /// it did.
    fn launch(&mut self, shape: &Shape, want: u32) -> u32 {
        let mut done = 0;
        for at in 0..self.slots.len() {
            if done == want {
                break;
            }
            if self.launched[at].is_some() || shape_of(&self.slots[at]) != *shape {
                continue;
            }
            let slot = &self.slots[at];
            let Some((program, args)) = slot.command.split_first() else {
                continue;
            };
            self.next_token += 1;
            let token = format!("{:x}-{at}-{}", self.epoch, self.next_token);
            match Command::new(program)
                .args(args)
                .env(LAUNCH_ENV, &token)
                .stdin(Stdio::null())
                .spawn()
            {
                Ok(child) => {
                    tracing::info!(slot = at, pid = child.id(), %token, model = ?shape.0, engine = %shape.1, "launched a worker to meet the scale table");
                    self.launched[at] = Some(Launched {
                        child,
                        token,
                        at: Instant::now(),
                        stopping: false,
                    });
                    done += 1;
                }
                Err(error) => {
                    tracing::warn!(slot = at, command = ?slot.command, %error, "a slot's worker would not launch");
                }
            }
        }
        done
    }

    /// Stops `count` of the workers of `shape` it launched, the most recently
    /// launched first.
    fn stop(&mut self, shape: &Shape, count: u32) {
        let mut ours: Vec<usize> = (0..self.slots.len())
            .filter(|&at| shape_of(&self.slots[at]) == *shape)
            .filter(|&at| self.launched[at].as_ref().is_some_and(|l| !l.stopping))
            .collect();
        ours.sort_by_key(|&at| std::cmp::Reverse(self.launched[at].as_ref().map(|l| l.at)));
        for at in ours.into_iter().take(count as usize) {
            if let Some(launched) = self.launched[at].as_mut() {
                tracing::info!(slot = at, pid = launched.child.id(), token = %launched.token, "stopping a worker the scale table no longer wants; it drains first");
                terminate(&launched.child);
                launched.stopping = true;
            }
        }
    }

    /// Frees the slots whose worker exited.
    fn reap(&mut self) {
        for (at, slot) in self.launched.iter_mut().enumerate() {
            let exited = slot
                .as_mut()
                .is_some_and(|l| !matches!(l.child.try_wait(), Ok(None)));
            if exited && let Some(l) = slot.take() {
                tracing::info!(slot = at, token = %l.token, "a launched worker exited; its slot is free");
                self.gone.insert(l.token);
            }
        }
    }
}

impl Drop for Actuator {
    fn drop(&mut self) {
        self.stop_all();
    }
}

/// SIGTERM to `child`, through `kill`, so the worker drains before it goes.
#[cfg(unix)]
fn terminate(child: &Child) {
    let pid = child.id().to_string();
    match Command::new("kill").args(["-TERM", &pid]).status() {
        Ok(status) if status.success() => {}
        Ok(status) => tracing::warn!(%pid, %status, "kill -TERM refused"),
        Err(error) => tracing::warn!(%pid, %error, "kill -TERM did not run"),
    }
}

#[cfg(not(unix))]
fn terminate(child: &Child) {
    tracing::warn!(
        pid = child.id(),
        "stopping a worker needs a signal it can drain on; this host has none"
    );
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use controller_api::{Role, WorkerStatus};

    use super::*;

    /// A worker that drains forever: it ignores the SIGTERM a stop sends.
    fn slot(model: &str) -> Slot {
        Slot {
            model: Some(model.into()),
            engine: "cuda_native".into(),
            command: vec![
                "sh".into(),
                "-c".into(),
                "trap '' TERM; exec sleep 1000".into(),
            ],
        }
    }

    /// Kills what the test launched, past the drain a stop would wait for.
    fn kill_all(actuator: &mut Actuator) {
        for launched in actuator.launched.iter_mut().flatten() {
            let _ = launched.child.kill();
            let _ = launched.child.wait();
        }
        actuator.launched.iter_mut().for_each(|l| *l = None);
    }

    fn entry(model: &str, workers: u32) -> ScaleEntry {
        ScaleEntry {
            model: Some(model.into()),
            engine: "cuda_native".into(),
            workers,
        }
    }

    fn worker(model: &str, launch: Option<String>) -> Worker {
        Worker {
            role: Role::Decode,
            model: model.into(),
            engine: "cuda_native".into(),
            addr: String::new(),
            launch,
            neighbors: Vec::new(),
            last_hb: Instant::now(),
            load: WorkerStatus::default(),
            health: Health::Healthy,
        }
    }

    fn tokens(actuator: &Actuator) -> Vec<Option<String>> {
        actuator
            .launched
            .iter()
            .map(|l| l.as_ref().filter(|l| !l.stopping).map(|l| l.token.clone()))
            .collect()
    }

    #[test]
    fn it_launches_to_the_table_and_stops_what_it_launched_past_it() {
        let mut actuator = Actuator::new(vec![slot("m"), slot("m"), slot("m")]);
        let mut workers = HashMap::new();
        // A worker it did not launch serves the shape already.
        workers.insert(WorkerId(9), worker("m", None));
        actuator.desire(vec![entry("m", 3)]);
        actuator.converge(&workers);
        let launched: Vec<String> = tokens(&actuator).into_iter().flatten().collect();
        assert_eq!(launched.len(), 2, "two short of three, one already serving");
        let status = actuator.status(&workers);
        assert_eq!(
            (status[0].serving, status[0].launching, status[0].free),
            (1, 2, 1)
        );

        // The launches register.
        for (n, token) in launched.iter().enumerate() {
            workers.insert(WorkerId(n as u64), worker("m", Some(token.clone())));
        }
        actuator.converge(&workers);
        assert_eq!(
            tokens(&actuator).iter().flatten().count(),
            2,
            "met: nothing more"
        );

        // Down to one: its own two go, the one it did not launch stays.
        actuator.desire(vec![entry("m", 1)]);
        actuator.converge(&workers);
        assert_eq!(tokens(&actuator).iter().flatten().count(), 0);
        let status = actuator.status(&workers);
        assert_eq!((status[0].serving, status[0].stopping), (1, 2));
        actuator.converge(&workers);
        assert_eq!(
            actuator.status(&workers)[0].stopping,
            2,
            "stopped once, not again"
        );

        // Down to none: the one it did not launch is never stopped.
        actuator.desire(vec![entry("m", 0)]);
        actuator.converge(&workers);
        assert_eq!(actuator.status(&workers)[0].serving, 1);
        kill_all(&mut actuator);
    }

    #[test]
    fn a_slot_whose_worker_exited_is_free_again() {
        let mut actuator = Actuator::new(vec![Slot {
            command: vec!["true".into()],
            ..slot("m")
        }]);
        actuator.desire(vec![entry("m", 1)]);
        actuator.converge(&HashMap::new());
        assert!(actuator.launched[0].is_some());
        std::thread::sleep(Duration::from_millis(200));
        actuator.reap();
        assert!(actuator.launched[0].is_none(), "reaped");
        actuator.converge(&HashMap::new());
        assert!(
            actuator.launched[0].is_some(),
            "launched again: the table still wants it"
        );
        actuator.stop_all();
    }

    #[test]
    fn a_shape_past_its_slots_is_unreachable_and_one_not_named_keeps_what_it_has() {
        let mut actuator = Actuator::new(vec![slot("m"), slot("n")]);
        let mut workers = HashMap::new();
        workers.insert(WorkerId(1), worker("n", None));
        workers.insert(WorkerId(2), worker("n", None));
        actuator.desire(vec![entry("m", 3)]);
        actuator.converge(&workers);
        let status = actuator.status(&workers);
        assert_eq!(status.len(), 1, "only the named shape is reported");
        assert_eq!((status[0].launching, status[0].free), (1, 0));
        assert!(
            actuator
                .short
                .contains(&(Some("m".into()), "cuda_native".into()))
        );
        assert!(
            actuator.launched[1].is_none(),
            "n is not named: nothing launched for it"
        );
        assert_eq!(workers.len(), 2, "and nothing of n stopped");
        kill_all(&mut actuator);
    }

    #[test]
    fn a_stopped_worker_still_registered_after_it_exited_does_not_count() {
        // One kept, one stopped: the stopped one exits on its signal, and
        // stays registered until its heartbeats lapse.
        let mut actuator = Actuator::new(vec![
            slot("m"),
            Slot {
                command: vec!["sleep".into(), "1000".into()],
                ..slot("m")
            },
        ]);
        actuator.desire(vec![entry("m", 2)]);
        actuator.converge(&HashMap::new());
        let mut workers = HashMap::new();
        for (n, token) in tokens(&actuator).into_iter().flatten().enumerate() {
            workers.insert(WorkerId(n as u64), worker("m", Some(token)));
        }
        actuator.desire(vec![entry("m", 1)]);
        actuator.converge(&workers);
        let kept: Vec<String> = tokens(&actuator).into_iter().flatten().collect();
        assert_eq!(kept.len(), 1, "the most recent one is stopped");
        std::thread::sleep(Duration::from_millis(300));
        actuator.converge(&workers);
        assert_eq!(
            tokens(&actuator).into_iter().flatten().collect::<Vec<_>>(),
            kept,
            "its exit does not make it count again, so the kept one stays"
        );
        assert_eq!(actuator.status(&workers)[0].serving, 1);
        kill_all(&mut actuator);
    }

    #[test]
    fn a_draining_worker_does_not_count_as_serving() {
        let mut actuator = Actuator::new(vec![slot("m")]);
        let mut workers = HashMap::new();
        let mut draining = worker("m", None);
        draining.load.draining = true;
        workers.insert(WorkerId(1), draining);
        actuator.desire(vec![entry("m", 1)]);
        actuator.converge(&workers);
        assert_eq!(
            actuator.status(&workers)[0].launching,
            1,
            "a replacement is launched"
        );
        kill_all(&mut actuator);
    }
}
