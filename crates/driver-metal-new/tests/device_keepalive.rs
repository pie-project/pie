//! The keepalive thread: that it submits, that it clamps, and that it joins.
//!
//! The claim worth testing is not "the thread is alive". A thread parked
//! forever on its first event wait is alive and keeps the GPU exactly as idle
//! as no keepalive at all, which is why every meaningful assertion here is
//! about the committed counter having MOVED -- past the in-flight depth
//! specifically, since that is the point at which the depth bound has been
//! reached, released by a retiring command buffer, and passed.
//!
//! The rest is about the hazards the C++ leaves open: a failed or repeated
//! start that leaves half-built state behind, and a thread that outlives the
//! objects it commits through.

#![allow(clippy::print_stdout)]

use std::time::{Duration, Instant};

use driver_metal_new::{Context, Error, Keepalive, MIN_DEPTH, MIN_THREADGROUPS};

/// Small enough that a dispatch retires quickly -- the tests below want many
/// submissions in a short window, not a busy GPU.
const SPIN_ITERS: u32 = 64;

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

/// Poll until `keepalive` has committed more than `target`, or give up.
///
/// Polling rather than one long sleep so a passing run is fast and a failing
/// one still gets the full budget before it is called a failure.
fn wait_for_commits(keepalive: &Keepalive, target: u64, budget: Duration) -> u64 {
    let deadline = Instant::now() + budget;
    loop {
        let committed = keepalive.committed();
        if committed > target || Instant::now() >= deadline {
            return committed;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn starting_a_keepalive_and_dropping_it_immediately_neither_hangs_nor_crashes() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let keepalive = Keepalive::start(&context, SPIN_ITERS, 1, 2).expect("keepalive starts");
    // Dropped before the thread has necessarily reached its first commit,
    // which is the race the join has to survive: the flag may be cleared
    // between the loop check and the commit, or before either.
    drop(keepalive);
}

#[test]
fn the_committed_counter_advances_past_the_in_flight_depth() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let depth = 3;
    let keepalive = Keepalive::start(&context, SPIN_ITERS, 2, depth).expect("keepalive starts");
    let target = u64::from(keepalive.depth());

    let committed = wait_for_commits(&keepalive, target, Duration::from_secs(5));
    assert!(
        committed > target,
        "the thread committed {committed} buffers in five seconds, and anything at or below the \
         depth of {target} means it reached the first in-flight wait and stopped there -- the \
         keepalive would be a live thread submitting nothing"
    );
}

#[test]
fn a_depth_below_two_and_a_grid_of_no_threadgroups_are_clamped_up() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    for depth in [0, 1] {
        let keepalive = Keepalive::start(&context, SPIN_ITERS, 0, depth).expect("keepalive starts");
        assert_eq!(
            keepalive.depth(),
            MIN_DEPTH,
            "a depth of {depth} drains the queue between dispatches, so it is raised to the \
             smallest depth that keeps one buffer in flight"
        );
        assert_eq!(
            keepalive.threadgroups(),
            MIN_THREADGROUPS,
            "a grid of no threadgroups dispatches nothing at all"
        );
        assert_eq!(
            keepalive.spin_iters(),
            SPIN_ITERS,
            "spin_iters is not clamped"
        );
    }
}

#[test]
fn two_keepalives_can_be_started_and_dropped_in_sequence() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // This is where the C++'s half-initialised state would bite: its second
    // `start_keepalive` takes the branch that writes through a buffer the
    // first call may never have created. Here the second start builds its own
    // objects from nothing, so it either works or fails, and the first one's
    // outcome cannot reach it.
    let first = Keepalive::start(&context, SPIN_ITERS, 1, 2).expect("first keepalive starts");
    let advanced = wait_for_commits(&first, u64::from(first.depth()), Duration::from_secs(5));
    assert!(advanced > u64::from(first.depth()), "{advanced}");
    drop(first);

    let second = Keepalive::start(&context, SPIN_ITERS, 1, 2).expect("second keepalive starts");
    let advanced = wait_for_commits(&second, u64::from(second.depth()), Duration::from_secs(5));
    assert!(
        advanced > u64::from(second.depth()),
        "the second keepalive submits as freely as the first: {advanced}"
    );
    drop(second);
}

#[test]
fn dropping_a_running_keepalive_joins_within_a_bounded_time() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let keepalive = Keepalive::start(&context, SPIN_ITERS, 4, 4).expect("keepalive starts");
    let committed = wait_for_commits(
        &keepalive,
        u64::from(keepalive.depth()),
        Duration::from_secs(5),
    );
    assert!(committed > u64::from(keepalive.depth()), "{committed}");

    let start = Instant::now();
    drop(keepalive);
    let elapsed = start.elapsed();
    // The thread checks the flag once an iteration and its only blocking
    // calls are bounded at five seconds each, so a drop that took longer than
    // the two of them would mean it is not observing the flag at all.
    assert!(
        elapsed < Duration::from_secs(11),
        "the drop took {elapsed:?}; the join is supposed to be bounded by the thread's own wait \
         timeouts"
    );
    // Reaching here at all is the other half of the claim: the process is
    // still running after every Metal object the thread owned was released on
    // the thread, which is what would fault if the join had not happened.
}
