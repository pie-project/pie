//! A monotonic clock and timers over whatever the page provides.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

/// Milliseconds since an arbitrary origin, as `performance.now()` reports.
pub type ClockFn = fn() -> f64;

static CLOCK: OnceLock<ClockFn> = OnceLock::new();

/// Install the clock. The first installer wins; the runtime never changes
/// clocks once it has read one.
pub fn set_clock(clock: ClockFn) {
    let _ = CLOCK.set(clock);
}

fn now_ns() -> u64 {
    match CLOCK.get() {
        Some(clock) => (clock() * 1_000_000.0) as u64,
        #[cfg(not(target_arch = "wasm32"))]
        None => {
            static START: OnceLock<std::time::Instant> = OnceLock::new();
            let start = *START.get_or_init(std::time::Instant::now);
            u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }
        // Before the page installs a clock nothing has happened yet.
        #[cfg(target_arch = "wasm32")]
        None => 0,
    }
}

/// `std::time::Instant`'s surface over the page clock. Nanoseconds from an
/// origin nobody cares about: every consumer reads deltas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(u64);

impl Instant {
    pub fn now() -> Self {
        Instant(now_ns())
    }

    /// Milliseconds since the clock's origin (`performance.now()` on a page).
    pub fn elapsed_since_origin_ms(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    pub fn elapsed(&self) -> Duration {
        Self::now().saturating_duration_since(*self)
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.saturating_duration_since(earlier)
    }

    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }

    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        u64::try_from(duration.as_nanos())
            .ok()
            .and_then(|ns| self.0.checked_add(ns))
            .map(Instant)
    }

    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        u64::try_from(duration.as_nanos())
            .ok()
            .and_then(|ns| self.0.checked_sub(ns))
            .map(Instant)
    }
}

impl std::ops::Add<Duration> for Instant {
    type Output = Instant;
    fn add(self, rhs: Duration) -> Instant {
        self.checked_add(rhs)
            .expect("overflow when adding duration to instant")
    }
}

impl std::ops::AddAssign<Duration> for Instant {
    fn add_assign(&mut self, rhs: Duration) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub<Duration> for Instant {
    type Output = Instant;
    fn sub(self, rhs: Duration) -> Instant {
        self.checked_sub(rhs)
            .expect("overflow when subtracting duration from instant")
    }
}

impl std::ops::Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, rhs: Instant) -> Duration {
        self.saturating_duration_since(rhs)
    }
}

// ---- timers ------------------------------------------------------------------

struct Timers {
    heap: BinaryHeap<Reverse<(u64, u64)>>,
    wakers: HashMap<u64, Waker>,
    next_seq: u64,
}

thread_local! {
    static TIMERS: RefCell<Timers> = RefCell::new(Timers {
        heap: BinaryHeap::new(),
        wakers: HashMap::new(),
        next_seq: 0,
    });
}

fn register(deadline: Instant, waker: Waker) -> u64 {
    TIMERS.with(|t| {
        let mut t = t.borrow_mut();
        let seq = t.next_seq;
        t.next_seq += 1;
        t.heap.push(Reverse((deadline.0, seq)));
        t.wakers.insert(seq, waker);
        seq
    })
}

fn cancel(seq: u64) {
    TIMERS.with(|t| {
        t.borrow_mut().wakers.remove(&seq);
    });
}

/// Wake every timer whose deadline has passed. Returns how many fired.
pub fn fire_due() -> usize {
    let now = now_ns();
    let mut due = Vec::new();
    TIMERS.with(|t| {
        let mut t = t.borrow_mut();
        while let Some(Reverse((deadline, seq))) = t.heap.peek().copied() {
            if deadline > now {
                break;
            }
            t.heap.pop();
            if let Some(waker) = t.wakers.remove(&seq) {
                due.push(waker);
            }
        }
    });
    let fired = due.len();
    for waker in due {
        waker.wake();
    }
    fired
}

/// How long until the earliest live timer, if any.
pub fn next_deadline() -> Option<Duration> {
    let now = now_ns();
    TIMERS.with(|t| {
        let mut t = t.borrow_mut();
        // Drop cancelled entries off the top so a stale seq can't keep the
        // page polling.
        while let Some(Reverse((deadline, seq))) = t.heap.peek().copied() {
            if t.wakers.contains_key(&seq) {
                return Some(Duration::from_nanos(deadline.saturating_sub(now)));
            }
            t.heap.pop();
        }
        None
    })
}

/// Resolves once its deadline has passed.
pub struct Sleep {
    deadline: Instant,
    registered: Option<u64>,
}

impl Sleep {
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

impl Future for Sleep {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            if let Some(seq) = self.registered.take() {
                cancel(seq);
            }
            return Poll::Ready(());
        }
        if let Some(seq) = self.registered.take() {
            cancel(seq);
        }
        self.registered = Some(register(self.deadline, cx.waker().clone()));
        Poll::Pending
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {
        if let Some(seq) = self.registered.take() {
            cancel(seq);
        }
    }
}

pub fn sleep(duration: Duration) -> Sleep {
    sleep_until(Instant::now() + duration)
}

pub fn sleep_until(deadline: Instant) -> Sleep {
    Sleep {
        deadline,
        registered: None,
    }
}

/// The deadline passed before the future resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Elapsed;

impl std::fmt::Display for Elapsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("deadline has elapsed")
    }
}

impl std::error::Error for Elapsed {}

pub struct Timeout<F> {
    future: Pin<Box<F>>,
    sleep: Sleep,
}

impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, Elapsed>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Poll::Ready(value) = self.future.as_mut().poll(cx) {
            return Poll::Ready(Ok(value));
        }
        match Pin::new(&mut self.sleep).poll(cx) {
            Poll::Ready(()) => Poll::Ready(Err(Elapsed)),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub fn timeout<F: Future>(duration: Duration, future: F) -> Timeout<F> {
    Timeout {
        future: Box::pin(future),
        sleep: sleep(duration),
    }
}

/// tokio's `Interval`: the first tick completes immediately, later ticks
/// each wait one period from the previous deadline.
pub struct Interval {
    period: Duration,
    next: Instant,
    sleep: Option<Sleep>,
}

impl Interval {
    pub async fn tick(&mut self) -> Instant {
        let deadline = self.next;
        let sleep = self.sleep.get_or_insert_with(|| sleep_until(deadline));
        Pin::new(sleep).await;
        self.sleep = None;
        self.next = deadline + self.period;
        deadline
    }
}

pub fn interval(period: Duration) -> Interval {
    assert!(period > Duration::ZERO, "interval period must be non-zero");
    Interval {
        period,
        next: Instant::now(),
        sleep: None,
    }
}
