//! Holds each process that states the hint `pause-at-ms` from that long
//! after its arrival for `pause-for-ms` (1000 without one), then lifts it:
//! the host pauses it (its state parked on the tiers below the device) and
//! resumes it. What a host's test of state that must survive the move uses
//! (a beam group's page lists, BEAM-PAGE-LISTS §9). Named by `[runtime]
//! worker_policy`; the client states the hints as the process's props at
//! launch.
//!
//! It writes one column, `schedule/hold`: a process inside its window is
//! held, every other one is not. It wakes itself at the next window's edge
//! with the alarm.
//!
//! A hint stated as anything but an integer (a string `"300"`, a float) is
//! refused by name: the process is terminated with a reason that names it,
//! which its client reads as the refusal of its next step, never a pause
//! silently skipped. (A trap would say nothing: a PLEX 1.1 policy has no
//! WASI, so its panic message reaches no one.)

use plex::prelude::*;

#[derive(Default)]
struct Pause;

impl Policy for Pause {
    fn plan(&mut self, now: Mark) {
        let mut next: Option<Mark> = None;
        let mut held: Vec<(ProcessId, bool)> = Vec::new();
        for p in snapshot::get_processes().iter().filter(|p| p.live()) {
            let (at, length) = match (whole(&p.id, "pause-at-ms"), whole(&p.id, "pause-for-ms")) {
                (Ok(None), _) => continue,
                (Ok(Some(at)), Ok(length)) => (at, length.unwrap_or(1000)),
                (Err(why), _) | (_, Err(why)) => {
                    terminate::terminate(&p.id, &why).ok();
                    continue;
                }
            };
            let from = p.arrival.saturating_add(ms(at));
            let to = from.saturating_add(ms(length));
            let edge = if now < from {
                Some(from)
            } else if now < to {
                held.push((p.id.clone(), true));
                Some(to)
            } else {
                None
            };
            if let Some(edge) = edge {
                next = Some(next.map_or(edge, |n| n.min(edge)));
            }
        }
        // No row is the column's default, not held: a process whose window
        // passed (or that states none) runs.
        schedule::hold::clear();
        schedule::hold::set_list(&held).expect("test-pause: the host refused its hold rows");
        if let Some(at) = next {
            alarm::set(at);
        }
    }
}

/// `n` ms as a length of time (ns), none below zero.
fn ms(n: i64) -> Duration {
    u64::try_from(n).unwrap_or(0).saturating_mul(MILLISECOND)
}

/// The integer hint `name` of `process`: none where it states none; one it
/// states as anything but an integer is refused, by name.
fn whole(process: &str, name: &str) -> Result<Option<i64>, String> {
    match hint(process, name) {
        None => Ok(None),
        Some(Value::Int(ms)) => Ok(Some(ms)),
        Some(_) => Err(format!(
            "test-pause: the hint {name} must be an integer (ms), as `{name}: 300`"
        )),
    }
}

plex::export!(Pause);
