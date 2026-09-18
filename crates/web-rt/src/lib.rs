//! What a browser tab offers in place of an operating system.
//!
//! pie's runtime was written for a machine with threads, a monotonic clock
//! and a multi-threaded tokio runtime. A tab has one thread, an event loop it
//! must keep returning to, `performance.now()`, and — through JSPI-backed
//! fibers — the ability to park an execution and resume it later. This crate
//! is the thin layer that expresses the former in terms of the latter:
//!
//! * [`executor`] — a task queue the page drives one `tick` at a time; wakers
//!   ask the page for another tick through a hook;
//! * [`time`] — an `Instant` over a page-provided clock, plus `sleep`,
//!   `timeout` and `interval` on a timer heap the tick services;
//! * [`channel`] — crossbeam's channel surface where a blocking receive parks
//!   the calling green thread instead of an OS thread;
//! * [`thread`] — green threads: a closure that runs on its own fiber and may
//!   block through [`block_on_poll`], which parks it until a waker fires.
//!
//! On native targets only [`time::Instant`] and the hooks compile; the
//! runtime uses tokio, std and crossbeam there and never reaches this crate.

pub mod channel;
pub mod executor;
pub mod time;

#[cfg(target_arch = "wasm32")]
pub mod thread;

#[cfg(target_arch = "wasm32")]
pub use thread::block_on_poll;

pub use executor::{JoinError, JoinHandle, spawn, spawn_blocking, tick, yield_now};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn a_tick_runs_ready_tasks_and_reports_the_next_timer() {
        let hits = Arc::new(AtomicUsize::new(0));
        let h = Arc::clone(&hits);
        let handle = spawn(async move {
            h.fetch_add(1, Ordering::SeqCst);
            yield_now().await;
            h.fetch_add(1, Ordering::SeqCst);
            7
        });
        let next = tick(1000);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "a yield re-queues within the tick"
        );
        assert!(next.is_none());
        let mut handle = Box::pin(handle);
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        assert!(matches!(
            handle.as_mut().poll(&mut cx),
            std::task::Poll::Ready(Ok(7))
        ));
    }

    #[test]
    fn a_sleep_parks_until_its_deadline() {
        let done = Arc::new(AtomicUsize::new(0));
        let d = Arc::clone(&done);
        spawn(async move {
            time::sleep(Duration::from_millis(20)).await;
            d.store(1, Ordering::SeqCst);
        });
        let next = tick(1000).expect("a timer is armed");
        assert!(next <= Duration::from_millis(20));
        assert_eq!(done.load(Ordering::SeqCst), 0);
        std::thread::sleep(Duration::from_millis(25));
        tick(1000);
        assert_eq!(done.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn timeout_and_channels_agree_on_readiness() {
        let (tx, rx) = channel::unbounded::<u32>();
        let seen = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&seen);
        spawn(async move {
            let first = time::timeout(Duration::from_millis(5), rx.recv_async()).await;
            assert!(first.is_err(), "nothing was sent yet");
            let second = rx.recv_async().await.unwrap();
            s.store(second as usize, Ordering::SeqCst);
        });
        tick(1000);
        std::thread::sleep(Duration::from_millis(8));
        tick(1000);
        tx.send(42).unwrap();
        tick(1000);
        assert_eq!(seen.load(Ordering::SeqCst), 42);

        let (tx, rx) = channel::unbounded::<u32>();
        assert_eq!(rx.try_recv(), Err(channel::TryRecvError::Empty));
        drop(tx);
        assert_eq!(rx.try_recv(), Err(channel::TryRecvError::Disconnected));
    }
}
