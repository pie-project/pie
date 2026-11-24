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

use std::collections::HashMap;

use plex::prelude::*;

#[derive(Default)]
struct Default {
    /// Processes that exited and may yet be retained: their prompt tokens,
    /// where the host counted them. A working set retained without an exit
    /// (a resting handle) is not cache a process left, and keeps every tier.
    exited: HashMap<ProcessId, Option<u32>>,
    seats: SeatToEnd,
}

/// The holds seat-to-end keeps.
#[derive(Default)]
struct SeatToEnd {
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

    fn plan(&mut self, now: Mark) {
        let Some(worker) = snapshot::get_workers()
            .into_iter()
            .find(|worker| worker.device().is_some())
        else {
            self.lift();
            return;
        };
        let subject = Subject::Worker(worker.id.clone());
        // Unknown until a process has finished: nothing to hold against.
        let Some(lanes) = snapshot::get_attr(&subject, "pie/lanes-to-end")
            .and_then(|lanes| lanes.int())
            .map(|lanes| usize::try_from(lanes).unwrap_or(0))
        else {
            self.lift();
            return;
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
        // First come, first served: the engine's own order.
        waiting.sort_by(|a, b| (a.arrival, &a.id).cmp(&(b.arrival, &b.id)));
        // At least one runs, however large a lane is.
        let lanes = lanes.max(1);
        if running + waiting.len() <= lanes {
            self.lift();
            return;
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
        if spread && more && step > 0 {
            alarm::set(next_due.unwrap_or(now.saturating_add(step)));
        }
        let mut rows: Vec<(ProcessId, bool)> = processes
            .iter()
            .filter(|process| process.started())
            .map(|process| (process.id.clone(), false))
            .chain(self.released.iter().map(|id| (id.clone(), false)))
            .collect();
        rows.sort();
        rows.dedup();
        if self.written.as_ref() == Some(&rows) {
            return;
        }
        schedule::hold::clear();
        if schedule::hold::set_default(true).is_err() || schedule::hold::set_list(&rows).is_err() {
            schedule::hold::clear();
            self.written = None;
            return;
        }
        self.written = Some(rows);
    }
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

    fn plan(&mut self, now: Mark) {
        // A process is in the snapshot through the call that delivers its
        // exit, and its retention comes no later than the next call's: one
        // gone that left nothing cached is forgotten.
        self.exited
            .retain(|process, _| snapshot::get_process(process).is_some());
        self.seats.plan(now);
    }
}

plex::export!(Default);
