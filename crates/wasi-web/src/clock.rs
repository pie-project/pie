//! Time source and timers.
//!
//! A browser tab has no `Instant`, so the host plugs its own reading (e.g.
//! `performance.now()` scaled to nanoseconds) via [`set_clock`]. Guest sleeps
//! never block: they park a waker keyed by deadline, and the host calls
//! [`fire_due_timers`] from its event loop (a `setTimeout` scheduled for the
//! returned next deadline) to wake them.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

use wasmtime_wasi_io::async_trait;
use wasmtime_wasi_io::poll::Pollable;

static CLOCK: Mutex<fn() -> u64> = Mutex::new(default_now);
static WALL_CLOCK: Mutex<fn() -> u64> = Mutex::new(default_wall_now);
static TIMERS: Mutex<Vec<(u64, Waker)>> = Mutex::new(Vec::new());

/// Install the monotonic clock: nanoseconds from an arbitrary fixed origin,
/// never decreasing. Call before running guests on wasm32.
pub fn set_clock(f: fn() -> u64) {
    *CLOCK.lock().unwrap() = f;
}

/// Install the wall clock: nanoseconds since the Unix epoch (`Date.now()`).
pub fn set_wall_clock(f: fn() -> u64) {
    *WALL_CLOCK.lock().unwrap() = f;
}

/// Current monotonic reading in nanoseconds.
pub fn now() -> u64 {
    let f = *CLOCK.lock().unwrap();
    f()
}

pub(crate) fn wall_now() -> u64 {
    let f = *WALL_CLOCK.lock().unwrap();
    f()
}

#[cfg(not(target_arch = "wasm32"))]
fn default_now() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Without a host clock, advance one nanosecond per reading: monotone and
/// distinct, which is what `Instant`-style guest code relies on.
#[cfg(target_arch = "wasm32")]
fn default_now() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(not(target_arch = "wasm32"))]
fn default_wall_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn default_wall_now() -> u64 {
    0
}

/// Wake every timer whose deadline has passed; returns the earliest deadline
/// still pending so the host can schedule its next tick.
pub fn fire_due_timers() -> Option<u64> {
    let now = now();
    let due: Vec<Waker> = {
        let mut timers = TIMERS.lock().unwrap();
        let (due, pending): (Vec<_>, Vec<_>) = timers.drain(..).partition(|(d, _)| *d <= now);
        *timers = pending;
        due.into_iter().map(|(_, w)| w).collect()
    };
    for waker in due {
        waker.wake();
    }
    TIMERS.lock().unwrap().iter().map(|(d, _)| *d).min()
}

fn register(deadline: u64, waker: &Waker) {
    let mut timers = TIMERS.lock().unwrap();
    if let Some(slot) = timers
        .iter_mut()
        .find(|(d, w)| *d == deadline && w.will_wake(waker))
    {
        slot.1 = waker.clone();
        return;
    }
    timers.push((deadline, waker.clone()));
    drop(timers);
    // Native has no host event loop to tick us; a sleeper thread stands in.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let wait = std::time::Duration::from_nanos(deadline.saturating_sub(now()));
        std::thread::spawn(move || {
            std::thread::sleep(wait);
            fire_due_timers();
        });
    }
}

/// Resolves once the monotonic clock reaches `deadline`; `None` never resolves.
pub(crate) struct Wait {
    pub(crate) deadline: Option<u64>,
}

impl Wait {
    pub(crate) fn until(deadline: u64) -> Self {
        Wait {
            deadline: Some(deadline),
        }
    }

    pub(crate) fn duration(nanos: u64) -> Self {
        Wait {
            deadline: now().checked_add(nanos),
        }
    }
}

impl Future for Wait {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let Some(deadline) = self.deadline else {
            return Poll::Pending;
        };
        if now() >= deadline {
            return Poll::Ready(());
        }
        register(deadline, cx.waker());
        Poll::Pending
    }
}

/// The `wasi:io/poll` pollable behind `subscribe-instant`/`subscribe-duration`.
pub(crate) struct Deadline(pub(crate) Option<u64>);

#[async_trait]
impl Pollable for Deadline {
    async fn ready(&mut self) {
        Wait { deadline: self.0 }.await
    }
}
