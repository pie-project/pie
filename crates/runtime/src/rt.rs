#[cfg(not(target_arch = "wasm32"))]
mod imp {
    #[cfg(test)]
    pub use super::test_clock::Instant;
    #[cfg(not(test))]
    pub use std::time::Instant;
    pub use tokio::task::{JoinHandle, spawn, spawn_blocking, yield_now};

    pub mod time {
        pub use tokio::time::{Instant, interval, sleep, timeout};
    }

    pub use crossbeam::channel;
    pub use crossbeam::queue::SegQueue;

    pub mod thread {
        pub use std::thread::{Builder, JoinHandle};
    }

    pub fn has_runtime() -> bool {
        tokio::runtime::Handle::try_current().is_ok()
    }

    pub fn block_on<F: core::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build current-thread runtime")
            .block_on(f)
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    pub use web_std::time::Instant;
    pub use web_std::{JoinHandle, spawn, spawn_blocking, yield_now};

    pub mod time {
        pub use web_std::time::{Instant, interval, sleep, timeout};
    }

    pub use web_std::channel;
    pub use web_std::channel::SegQueue;
    pub use web_std::thread;

    pub fn has_runtime() -> bool {
        true
    }

    pub fn block_on<F: core::future::Future>(f: F) -> F::Output {
        let mut f = core::pin::pin!(f);
        web_std::block_on_poll(|cx| f.as_mut().poll(cx))
    }
}

pub use imp::*;

/// The wall clock, never the test clock: std's, or on wasm32, where std's
/// panics, the page's.
pub mod wall {
    #[cfg(not(target_arch = "wasm32"))]
    pub use std::time::Instant;
    #[cfg(target_arch = "wasm32")]
    pub use web_std::time::Instant;
}

/// The clock the runtime's `Instant` reads under test: the wall clock, or,
/// on a thread that turned it on ([`test_clock::start`]), a virtual clock
/// the test advances itself ([`test_clock::advance`]), so a seeded
/// simulation reads the same times on every run.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub mod test_clock {
    use std::cell::Cell;
    use std::time::Duration;

    thread_local! {
        /// The virtual clock's origin and how far it has been advanced.
        static VIRTUAL: Cell<Option<(std::time::Instant, Duration)>> = const { Cell::new(None) };
    }

    /// Turns the virtual clock on for this thread, at a fixed origin.
    pub fn start() {
        VIRTUAL.with(|clock| clock.set(Some((std::time::Instant::now(), Duration::ZERO))));
    }

    /// Back to the wall clock.
    pub fn stop() {
        VIRTUAL.with(|clock| clock.set(None));
    }

    /// Moves the virtual clock on by `by`.
    pub fn advance(by: Duration) {
        VIRTUAL.with(|clock| {
            if let Some((origin, at)) = clock.get() {
                clock.set(Some((origin, at + by)));
            }
        });
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Instant(std::time::Instant);

    impl Instant {
        pub fn now() -> Self {
            Self(VIRTUAL.with(|clock| match clock.get() {
                Some((origin, at)) => origin + at,
                None => std::time::Instant::now(),
            }))
        }

        pub fn elapsed(&self) -> Duration {
            Self::now().0.saturating_duration_since(self.0)
        }

        pub fn duration_since(&self, earlier: Instant) -> Duration {
            self.0.saturating_duration_since(earlier.0)
        }

        pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
            self.0.saturating_duration_since(earlier.0)
        }

        pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
            self.0.checked_duration_since(earlier.0)
        }

        pub fn checked_add(&self, by: Duration) -> Option<Instant> {
            self.0.checked_add(by).map(Self)
        }

        pub fn checked_sub(&self, by: Duration) -> Option<Instant> {
            self.0.checked_sub(by).map(Self)
        }
    }

    impl std::ops::Add<Duration> for Instant {
        type Output = Instant;
        fn add(self, by: Duration) -> Instant {
            Self(self.0 + by)
        }
    }

    impl std::ops::AddAssign<Duration> for Instant {
        fn add_assign(&mut self, by: Duration) {
            self.0 += by;
        }
    }

    impl std::ops::Sub<Duration> for Instant {
        type Output = Instant;
        fn sub(self, by: Duration) -> Instant {
            Self(self.0 - by)
        }
    }

    impl std::ops::SubAssign<Duration> for Instant {
        fn sub_assign(&mut self, by: Duration) {
            self.0 -= by;
        }
    }

    impl std::ops::Sub<Instant> for Instant {
        type Output = Duration;
        fn sub(self, earlier: Instant) -> Duration {
            self.0.saturating_duration_since(earlier.0)
        }
    }
}
