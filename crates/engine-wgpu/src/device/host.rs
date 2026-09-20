use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::error::{Fault, Result};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Instant = std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) type Instant = web_std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn can_block() -> bool {
    true
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn can_block() -> bool {
    web_std::thread::in_green_thread()
}

static PARKED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

pub fn parked() -> Vec<&'static str> {
    PARKED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

struct Parking(&'static str);

impl Drop for Parking {
    fn drop(&mut self) {
        let mut parked = PARKED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(at) = parked.iter().rposition(|name| *name == self.0) {
            parked.remove(at);
        }
    }
}

pub(crate) fn block_on_poll<T>(
    call: &'static str,
    f: impl FnMut(&mut Context<'_>) -> Poll<T>,
) -> Result<T> {
    if !can_block() {
        return Err(Fault::Blocking { call });
    }
    PARKED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(call);
    let _parking = Parking(call);
    Ok(block_on_poll_here(f))
}

#[cfg(target_arch = "wasm32")]
fn block_on_poll_here<T>(f: impl FnMut(&mut Context<'_>) -> Poll<T>) -> T {
    web_std::block_on_poll(f)
}

#[cfg(not(target_arch = "wasm32"))]
fn block_on_poll_here<T>(mut f: impl FnMut(&mut Context<'_>) -> Poll<T>) -> T {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(value) = f(&mut cx) {
            return value;
        }
        std::thread::park();
    }
}

pub(crate) fn block_on<F: Future>(call: &'static str, future: F) -> Result<F::Output> {
    let mut future = pin!(future);
    block_on_poll(call, |cx| future.as_mut().poll(cx))
}

pub(crate) struct Signal<T> {
    inner: Arc<Mutex<(Option<T>, Option<Waker>)>>,
}

impl<T> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Signal {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Signal<T> {
    pub(crate) fn new() -> Self {
        Signal {
            inner: Arc::new(Mutex::new((None, None))),
        }
    }

    pub(crate) fn notify(&self, value: T) {
        let waker = {
            let mut slot = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.0 = Some(value);
            slot.1.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub(crate) fn park(&self, call: &'static str) -> Result<T> {
        let inner = Arc::clone(&self.inner);
        block_on_poll(call, |cx| {
            let mut slot = inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match slot.0.take() {
                Some(value) => Poll::Ready(value),
                None => {
                    slot.1 = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        })
    }
}

pub(crate) mod thread {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn spawn(name: String, f: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
        std::thread::Builder::new().name(name).spawn(f).map(|_| ())
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn spawn(name: String, f: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
        web_std::thread::Builder::new()
            .name(name)
            .spawn(f)
            .map(|_| ())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod channel {
    pub(crate) use std::sync::mpsc::{SendError, Sender, channel as unbounded};
}

#[cfg(target_arch = "wasm32")]
pub(crate) mod channel {
    pub(crate) use web_std::channel::{SendError, Sender, unbounded};
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub(crate) struct Poller {
    state: Arc<(Mutex<PollerState>, std::sync::Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[derive(Default)]
struct PollerState {
    kicked: bool,
    stop: bool,
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl Poller {
    const WAIT: std::time::Duration = std::time::Duration::from_millis(500);

    pub(crate) fn start(device: wgpu::Device) -> Poller {
        let state: Arc<(Mutex<PollerState>, std::sync::Condvar)> = Arc::default();
        let shared = Arc::clone(&state);
        let thread = std::thread::Builder::new()
            .name("wgpu poll".into())
            .spawn(move || {
                let (lock, cv) = &*shared;
                loop {
                    {
                        let mut state = lock
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        while !state.kicked && !state.stop {
                            state = cv
                                .wait(state)
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                        }
                        if state.stop {
                            return;
                        }
                        state.kicked = false;
                    }
                    let polled = device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Self::WAIT),
                    });
                    if matches!(polled, Err(wgpu::PollError::Timeout)) {
                        lock.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .kicked = true;
                    }
                }
            })
            .ok();
        Poller { state, thread }
    }

    pub(crate) fn kick(&self) {
        let (lock, cv) = &*self.state;
        lock.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .kicked = true;
        cv.notify_one();
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl Drop for Poller {
    fn drop(&mut self) {
        let (lock, cv) = &*self.state;
        lock.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stop = true;
        cv.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
pub(crate) struct Poller;

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
impl Poller {
    pub(crate) fn start(_device: wgpu::Device) -> Poller {
        Poller
    }

    pub(crate) fn kick(&self) {}
}
