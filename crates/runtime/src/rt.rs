//! What the platform provides: threads, a clock, an executor, channels.
//!
//! The runtime was written against tokio, `std::thread`, `crate::rt::Instant`
//! and crossbeam. A browser tab has none of those — one thread, an event loop
//! that must be returned to, and `performance.now()` — so every use goes
//! through this module, which names the native thing on native targets and
//! the `web-rt` equivalent on wasm32. The two arms export the same names so
//! the rest of the crate reads the same either way.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    pub use std::time::Instant;
    pub use tokio::task::{JoinHandle, spawn, spawn_blocking, yield_now};

    pub mod time {
        pub use tokio::time::{Instant, interval, timeout};
    }

    pub use crossbeam::channel;
    pub use crossbeam::queue::SegQueue;

    pub mod thread {
        pub use std::thread::{Builder, JoinHandle};
    }

    /// Whether spawning is possible right now: tokio needs an ambient runtime.
    pub fn has_runtime() -> bool {
        tokio::runtime::Handle::try_current().is_ok()
    }

    /// Drive a future to completion on the calling OS thread, using a private
    /// current-thread tokio runtime (timers and I/O enabled). The scheduler
    /// owns one dedicated thread; this is where its async loop runs, so the
    /// future need not be `Send`.
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
    pub use web_rt::time::Instant;
    pub use web_rt::{JoinHandle, spawn, spawn_blocking, yield_now};

    pub mod time {
        pub use web_rt::time::{Instant, interval, timeout};
    }

    pub use web_rt::channel;
    pub use web_rt::channel::SegQueue;
    pub use web_rt::thread;

    /// The page's executor is always there.
    pub fn has_runtime() -> bool {
        true
    }

    /// Drive a future to completion by parking on the page executor until its
    /// waker fires. The scheduler runs as a green thread, so this cooperatively
    /// yields to the event loop the same way the old sync loop did.
    pub fn block_on<F: core::future::Future>(f: F) -> F::Output {
        let mut f = core::pin::pin!(f);
        web_rt::block_on_poll(|cx| f.as_mut().poll(cx))
    }
}

pub use imp::*;
