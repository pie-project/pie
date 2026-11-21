//! Holds each process that states `pause-at-ms` below the schedule floor
//! from that long after its arrival for `pause-for-ms`, then lifts it: the
//! host pauses it (its state parked on the tiers below the device) and
//! resumes it.

use policy::{Subject, SubjectKind, attr, candidates, prop, set_alarm, worker};

#[derive(Default)]
struct Pause;

impl worker::Policy for Pause {
    fn plan(&mut self, now_ms: u64) {
        let now = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let mut next: Option<i64> = None;
        let mut held = false;
        let entries: Vec<_> = candidates(SubjectKind::Process)
            .into_iter()
            .map(|p| {
                let process = Subject::Process(p.clone());
                let window = whole(&process, "pause-at-ms").and_then(|at| {
                    let from = attr(&process, "arrival-ms").int()?.saturating_add(at);
                    let length = whole(&process, "pause-for-ms").unwrap_or(1000);
                    Some((from, from.saturating_add(length)))
                });
                let paused = match window {
                    Some((from, _)) if now < from => {
                        next = Some(next.map_or(from, |n| n.min(from)));
                        false
                    }
                    Some((_, to)) if now < to => {
                        next = Some(next.map_or(to, |n| n.min(to)));
                        true
                    }
                    _ => false,
                };
                held |= paused;
                // Below the floor (1) is paused; every other process runs.
                (p, i64::from(!paused))
            })
            .collect();
        worker::install_schedule(&entries, held.then_some(1));
        if let Some(at) = next {
            set_alarm(u64::try_from(at).unwrap_or(0));
        }
    }
}

/// The integer prop `name` of `process`: none where it states none; one it
/// states as anything but an integer (a string `"300"`, a float) is refused
/// by name, a trap the host reports, never a pause silently skipped.
fn whole(process: &Subject, name: &str) -> Option<i64> {
    let stated = prop(process, name);
    if !stated.is_some() {
        return None;
    }
    match stated.int() {
        Some(ms) => Some(ms),
        None => panic!("test-pause: the prop {name} must be an integer (ms), as `{name}: 300`"),
    }
}

policy::worker!(Pause);
