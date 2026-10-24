use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ResidencyPlanner;
use super::inner::Inner;

struct LockCensus {
    n: AtomicU64,
    wait_ns: AtomicU64,
    hold_ns: AtomicU64,
    wait_max_ns: AtomicU64,
    hold_max_ns: AtomicU64,
}
static LOCK_CENSUS: LockCensus = LockCensus {
    n: AtomicU64::new(0),
    wait_ns: AtomicU64::new(0),
    hold_ns: AtomicU64::new(0),
    wait_max_ns: AtomicU64::new(0),
    hold_max_ns: AtomicU64::new(0),
};
fn lock_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("PIE_PLANNER_LOCK_TRACE").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}
pub(super) struct TimedGuard<'a> {
    guard: parking_lot::MutexGuard<'a, Inner>,
    held_from: Option<crate::rt::Instant>,
}
impl<'a> std::ops::Deref for TimedGuard<'a> {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.guard
    }
}
impl<'a> std::ops::DerefMut for TimedGuard<'a> {
    fn deref_mut(&mut self) -> &mut Inner {
        &mut self.guard
    }
}
impl<'a> Drop for TimedGuard<'a> {
    fn drop(&mut self) {
        if let Some(t) = self.held_from {
            let ns = t.elapsed().as_nanos() as u64;
            LOCK_CENSUS.hold_ns.fetch_add(ns, Ordering::Relaxed);
            LOCK_CENSUS.hold_max_ns.fetch_max(ns, Ordering::Relaxed);
        }
    }
}

impl ResidencyPlanner {
    pub(super) fn lock_inner(&self) -> TimedGuard<'_> {
        if !lock_trace_enabled() {
            return TimedGuard {
                guard: self.inner.lock(),
                held_from: None,
            };
        }
        let t0 = crate::rt::Instant::now();
        let guard = self.inner.lock();
        let waited = t0.elapsed().as_nanos() as u64;
        LOCK_CENSUS.n.fetch_add(1, Ordering::Relaxed);
        LOCK_CENSUS.wait_ns.fetch_add(waited, Ordering::Relaxed);
        LOCK_CENSUS.wait_max_ns.fetch_max(waited, Ordering::Relaxed);
        TimedGuard {
            guard,
            held_from: Some(crate::rt::Instant::now()),
        }
    }

    pub fn lock_census(&self) -> (u64, u64, u64, u64, u64) {
        let c = &LOCK_CENSUS;
        (
            c.n.swap(0, Ordering::Relaxed),
            c.wait_ns.swap(0, Ordering::Relaxed) / 1000,
            c.hold_ns.swap(0, Ordering::Relaxed) / 1000,
            c.wait_max_ns.swap(0, Ordering::Relaxed) / 1000,
            c.hold_max_ns.swap(0, Ordering::Relaxed) / 1000,
        )
    }
}
