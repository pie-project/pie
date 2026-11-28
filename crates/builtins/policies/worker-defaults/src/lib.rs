//! What every built-in worker policy does beside its own rows, where
//! nothing else says so: the device-only cache ([`IdleCache`]) and the
//! holds of seat-to-end ([`SeatToEnd`]). Neither sets the alarm: each says
//! when it wants the next look, and a policy sets the earliest ([`wake`]).

use std::cmp::Reverse;
use std::collections::HashMap;

use plex::prelude::*;

/// Sets the alarm at the earliest of `marks`, if any.
pub fn wake(marks: impl IntoIterator<Item = Option<Mark>>) {
    if let Some(at) = marks.into_iter().flatten().min() {
        alarm::set(at);
    }
}

/// The rows of the cache a process leaves when it exits: on the device and
/// nowhere else, its decoded tail before its prompt.
#[derive(Default)]
pub struct IdleCache {
    /// Processes that exited and may yet be retained: their prompt tokens,
    /// where the host counted them. A working set retained without an exit
    /// (a resting handle) is not cache a process left, and keeps every tier.
    exited: HashMap<ProcessId, Option<u32>>,
}

impl IdleCache {
    pub fn on_exited(&mut self, exit: &Exit) {
        let prompt = snapshot::get_process(&exit.process).and_then(|p| p.prompt_tokens);
        self.exited.insert(exit.process.clone(), prompt);
    }

    /// The exited process `retention`'s cache is, with its prompt tokens,
    /// once: `None` for a set no process left.
    pub fn left(&mut self, retention: &Retention) -> Option<Option<u32>> {
        // A process's working set has the process's id.
        self.exited.remove(&retention.working_set)
    }

    pub fn on_retained(&mut self, retention: &Retention) {
        if let Some(prompt) = self.left(retention) {
            Self::write(&retention.working_set, prompt);
        }
    }

    /// The rows of `set`, whose first `prompt` tokens were its prompt.
    pub fn write(set: &WorkingSetId, prompt: Option<u32>) {
        let mut keys = vec![(Prefix::all(set.as_str()), -1)];
        if let Some(prompt) = prompt.filter(|&p| p > 0) {
            keys.push((Prefix::upto(set.as_str(), prompt), 0));
        }
        retention::priority::set_list(&keys).ok();
        retention::tiers::set(&Prefix::all(set.as_str()), &[Tier::Device]).ok();
    }

    /// Whether `process` exited and may yet be retained.
    pub fn awaits(&self, process: &ProcessId) -> bool {
        self.exited.contains_key(process)
    }

    /// Forgets the exited that left nothing cached: a process is in the
    /// snapshot through the call that delivers its exit, and its retention
    /// comes no later than the next call's.
    pub fn plan(&mut self) {
        self.exited
            .retain(|process, _| snapshot::get_process(process).is_some());
    }
}

/// The holds seat-to-end keeps.
#[derive(Default)]
pub struct SeatToEnd {
    /// The hold rows as last written, sorted; `None` while no hold is in
    /// force (the column's own default, false).
    written: Option<Vec<(ProcessId, bool)>>,
    /// Waiting processes let go past the bound, still waiting to start.
    released: Vec<ProcessId>,
    /// When the last one was let go: the spread's stagger runs from it.
    last_release: Option<Mark>,
}

impl SeatToEnd {
    /// Holds nothing: the column's own default again.
    fn lift(&mut self) {
        if self.written.take().is_some() {
            schedule::hold::clear();
        }
        self.released.clear();
        self.last_release = None;
    }

    /// Writes the holds. `key` orders who is let go, larger first; equal
    /// keys go first come, first served. When to look again, if a start is
    /// due later.
    pub fn plan(&mut self, now: Mark, key: impl Fn(&snapshot::Process) -> i64) -> Option<Mark> {
        let Some(worker) = snapshot::get_workers()
            .into_iter()
            .find(|worker| worker.device().is_some())
        else {
            self.lift();
            return None;
        };
        let subject = Subject::Worker(worker.id.clone());
        // Unknown until a process has finished: nothing to hold against.
        let Some(lanes) = snapshot::get_attr(&subject, "pie/lanes-to-end")
            .and_then(|lanes| lanes.int())
            .map(|lanes| usize::try_from(lanes).unwrap_or(0))
        else {
            self.lift();
            return None;
        };
        let spread = snapshot::get_attr(&subject, "pie/starts-spread")
            .and_then(|spread| spread.boolean())
            .unwrap_or(false);
        let processes = snapshot::get_processes();
        let here = |process: &&snapshot::Process| {
            process.worker.as_ref().is_none_or(|on| *on == worker.id)
        };
        let running = processes
            .iter()
            .filter(here)
            .filter(|process| process.running())
            .count();
        let mut waiting: Vec<&snapshot::Process> = processes
            .iter()
            .filter(here)
            .filter(|process| process.state == ProcessState::Waiting)
            .collect();
        waiting.sort_by_cached_key(|process| {
            (Reverse(key(process)), process.arrival, process.id.clone())
        });
        // At least one runs, however large a lane is.
        let lanes = lanes.max(1);
        if running + waiting.len() <= lanes {
            self.lift();
            return None;
        }
        self.released
            .retain(|id| waiting.iter().any(|process| process.id == *id));
        let mut room = lanes
            .saturating_sub(running)
            .saturating_sub(self.released.len());
        let step = worker.step.unwrap_or(0);
        let mut next_due = None;
        if spread && room > 0 {
            // One start a step: lanes started together run short together.
            match self.last_release {
                Some(at) if now < at.saturating_add(step) => {
                    next_due = Some(at.saturating_add(step));
                    room = 0;
                }
                _ => room = 1,
            }
        }
        for process in &waiting {
            if room == 0 {
                break;
            }
            if !self.released.contains(&process.id) {
                self.released.push(process.id.clone());
                self.last_release = Some(now);
                room -= 1;
            }
        }
        let more = waiting
            .iter()
            .any(|process| !self.released.contains(&process.id));
        let again =
            (spread && more && step > 0).then(|| next_due.unwrap_or(now.saturating_add(step)));
        let mut rows: Vec<(ProcessId, bool)> = processes
            .iter()
            .filter(|process| process.started())
            .map(|process| (process.id.clone(), false))
            .chain(self.released.iter().map(|id| (id.clone(), false)))
            .collect();
        rows.sort();
        rows.dedup();
        if self.written.as_ref() == Some(&rows) {
            return again;
        }
        schedule::hold::clear();
        if schedule::hold::set_default(true).is_err() || schedule::hold::set_list(&rows).is_err() {
            schedule::hold::clear();
            self.written = None;
            return again;
        }
        self.written = Some(rows);
        again
    }
}

/// Both, for a policy whose own rows are the schedule's.
#[derive(Default)]
pub struct Defaults {
    pub cache: IdleCache,
    pub seats: SeatToEnd,
}

impl Defaults {
    /// After the call's events: [`SeatToEnd::plan`].
    pub fn plan(&mut self, now: Mark, key: impl Fn(&snapshot::Process) -> i64) -> Option<Mark> {
        self.cache.plan();
        self.seats.plan(now, key)
    }
}
