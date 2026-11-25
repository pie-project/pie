//! Seat-to-end in the built-in `default` policy: the runtime gates no start
//! on how many lanes its pools hold to their end; the policy holds starts
//! with `schedule/hold` while more would run than the worker's attr
//! `pie/lanes-to-end`, and lets them go first come, first served.

use plex_host::tables::Value;

use super::port::testing::{Worker, enact_on};
use super::{Arrival, Event, Outcome, schedule};
use crate::inferlet::process::ProcessId;

fn pids<const N: usize>(base: u128) -> [ProcessId; N] {
    std::array::from_fn(|i| ProcessId::from_u128(base + i as u128))
}

fn worker(lanes: i64, spread: bool, step_ns: Option<u64>) -> Worker {
    Worker {
        attrs: vec![
            ("pie/lanes-to-end", Value::Int(lanes)),
            ("pie/starts-spread", Value::Boolean(spread)),
        ],
        step_ns,
    }
}

/// `running` arrive and start, then `waiting` arrive; then `exits` of the
/// running exit.
fn events(running: &[ProcessId], waiting: &[ProcessId], exits: &[ProcessId]) -> Vec<Event> {
    let mut events: Vec<Event> = running
        .iter()
        .flat_map(|&pid| [Event::Arrived(pid, Arrival::default()), Event::Started(pid)])
        .collect();
    events.extend(waiting.iter().map(|&pid| Event::Arrived(pid, Arrival::default())));
    events.extend(exits.iter().map(|&pid| Event::Exited(pid, Outcome::Completed)));
    events
}

fn run(worker: &Worker, events: Vec<Event>) {
    let policy = builtins::policy("default").expect("the built-in default policy");
    schedule::install(None);
    enact_on("default", policy.component, worker, events);
}

/// The schedule lock, the table cleared again as it goes: a hold default
/// left installed would hold every other test's processes.
struct Alone(#[allow(dead_code)] tokio::sync::RwLockWriteGuard<'static, ()>);

impl Drop for Alone {
    fn drop(&mut self) {
        schedule::install(None);
    }
}

fn alone() -> Alone {
    Alone(schedule::TEST_LOCK.blocking_write())
}

fn held(pid: ProcessId) -> bool {
    schedule::held(pid)
}

/// Two lanes to the end, two running: the two that arrive after wait to
/// start (held), the running ones run on (a `false` row each, so the
/// default `true` pauses none of them).
#[test]
fn past_the_lanes_to_the_end_a_start_is_held_and_the_running_run_on() {
    let _alone = alone();
    let [a, b, c, d] = pids::<4>(0x5e00);
    run(&worker(2, false, None), events(&[a, b], &[c, d], &[]));
    assert!(!held(a) && !held(b), "the running are not held");
    assert!(held(c) && held(d), "the waiting wait for a lane to finish");
    // One that arrives after the policy's last call is held by the default.
    assert!(held(ProcessId::from_u128(0x5eff)), "the hold's default is true");
}

/// Where starting together runs short (`pie/starts-spread`), two lanes
/// freed at once let one start, the next a step later.
#[test]
fn spread_starts_go_one_a_step() {
    let _alone = alone();
    let [a, b, c, d, e] = pids::<5>(0x5e30);
    let hour = 3_600_000_000_000;
    run(&worker(2, true, Some(hour)), events(&[a, b], &[c, d, e], &[a, b]));
    assert!(!held(c), "the oldest starts");
    assert!(held(d) && held(e), "the rest a step apart");
    // Not spread: both freed lanes start at once.
    run(&worker(2, false, Some(hour)), events(&[a, b], &[c, d, e], &[a, b]));
    assert!(!held(c) && !held(d));
    assert!(held(e));
}
