//! `PIE_TURN_TRACE=1`: where a guest's turnaround goes. Every 100 ms a line
//! with the processes instantiating, started and finished in the window, the
//! gaps from a guest's last take to its next submit (over 20, 50 and 100 ms,
//! and the longest), the longest submit call, and how late a 1 ms tick on
//! the runtime's workers ran: a late tick is a worker pool too busy to poll
//! the guests.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

static ON: LazyLock<bool> =
    LazyLock::new(|| std::env::var("PIE_TURN_TRACE").is_ok_and(|v| v == "1"));

static INSTANTIATING: AtomicU64 = AtomicU64::new(0);
static STARTED: AtomicU64 = AtomicU64::new(0);
static FINISHED: AtomicU64 = AtomicU64::new(0);
static GAPS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static GAP_MAX_US: AtomicU64 = AtomicU64::new(0);
static SUBMIT_MAX_US: AtomicU64 = AtomicU64::new(0);
static FIRST_SUBMIT_MAX_US: AtomicU64 = AtomicU64::new(0);
static TICK_LATE_MAX_US: AtomicU64 = AtomicU64::new(0);

#[must_use]
pub fn on() -> bool {
    *ON
}

pub fn instantiating(delta: i64) {
    if on() {
        if delta > 0 {
            INSTANTIATING.fetch_add(1, Relaxed);
            STARTED.fetch_add(1, Relaxed);
        } else {
            INSTANTIATING.fetch_sub(1, Relaxed);
        }
    }
}

pub fn finished() {
    if on() {
        FINISHED.fetch_add(1, Relaxed);
    }
}

/// A submit that began `gap` after the guest's last take returned, and took
/// `took`.
pub fn submit(gap: Option<Duration>, took: Duration, first: bool) {
    let took = took.as_micros() as u64;
    SUBMIT_MAX_US.fetch_max(took, Relaxed);
    if first {
        FIRST_SUBMIT_MAX_US.fetch_max(took, Relaxed);
    }
    if let Some(gap) = gap {
        let us = gap.as_micros() as u64;
        GAP_MAX_US.fetch_max(us, Relaxed);
        for (i, ms) in [20u64, 50, 100].into_iter().enumerate() {
            if us >= ms * 1000 {
                GAPS[i].fetch_add(1, Relaxed);
            }
        }
    }
}

/// The ticker and the reporter, on the runtime's workers.
#[cfg(target_arch = "wasm32")]
pub fn spawn() {}

/// The ticker and the reporter, on the runtime's workers.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn() {
    if !on() || !crate::rt::has_runtime() {
        return;
    }
    crate::rt::spawn(async {
        loop {
            let at = Instant::now();
            tokio::time::sleep(Duration::from_millis(1)).await;
            let late = at.elapsed().saturating_sub(Duration::from_millis(1));
            TICK_LATE_MAX_US.fetch_max(late.as_micros() as u64, Relaxed);
        }
    });
    crate::rt::spawn(async {
        let origin = Instant::now();
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let started = STARTED.swap(0, Relaxed);
            let finished = FINISHED.swap(0, Relaxed);
            let gaps: Vec<u64> = GAPS.iter().map(|g| g.swap(0, Relaxed)).collect();
            let gap_max = GAP_MAX_US.swap(0, Relaxed);
            let submit_max = SUBMIT_MAX_US.swap(0, Relaxed);
            let first_max = FIRST_SUBMIT_MAX_US.swap(0, Relaxed);
            let late = TICK_LATE_MAX_US.swap(0, Relaxed);
            if started + finished + gaps[0] == 0 && late < 5_000 {
                continue;
            }
            println!(
                "[turn] t_ms={} instantiating={} started={started} finished={finished} \
                 gaps>20/50/100ms={}/{}/{} gap_max_us={gap_max} submit_max_us={submit_max} \
                 first_submit_max_us={first_max} tick_late_max_us={late}",
                origin.elapsed().as_millis(),
                INSTANTIATING.load(Relaxed),
                gaps[0],
                gaps[1],
                gaps[2],
            );
        }
    });
}
