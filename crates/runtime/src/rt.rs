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
        pub use web_std::time::{Instant, interval, timeout};
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
