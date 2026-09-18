//! What the two hosts give the engine, behind one shape.
//!
//! The engine is written for the browser: nothing in it blocks on the GPU,
//! every answer arrives by callback, and a wait is a park on a [`Signal`].
//! Native runs the same code. The differences are all here:
//!
//! - **where a wait parks.** In a tab the engine lane is a green thread and
//!   parks by handing the executor a waker; off a green thread there is
//!   nothing to park and the wait is refused as [`Fault::Blocking`]. Native
//!   parks the OS thread and is woken by whoever delivers the value.
//! - **who fires the callbacks.** WebGPU fires them from the page's event
//!   loop. wgpu on Vulkan, Metal or DX12 fires them only from `Device::poll`,
//!   so native keeps one [`Poller`] thread that polls whenever work or a map
//!   is outstanding; every submit and every map [kicks](Poller::kick) it.
//! - **threads and channels.** The landing worker and the guest runners are
//!   green threads over `web-rt` channels in a tab, OS threads over `mpsc`
//!   channels natively.
//! - **the clock.** `std::time::Instant` panics on wasm32-unknown-unknown.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::error::{Fault, Result};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Instant = std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) type Instant = web_rt::time::Instant;

/// Whether the calling thread may park: native always, a tab only from a
/// green thread.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn can_block() -> bool {
    true
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn can_block() -> bool {
    web_rt::thread::in_green_thread()
}

/// The waits currently parked, by name: what a stalled host is waiting on.
static PARKED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// The names of every wait parked right now (a name repeats per waiter).
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

/// Polls `f` until it is ready, parking the calling thread between polls.
/// `call` names the wait in the refusal a caller that cannot park gets, and
/// in [`parked`] while it waits.
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
    web_rt::block_on_poll(f)
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

/// Drives `future` to completion the way [`block_on_poll`] does.
pub(crate) fn block_on<F: Future>(call: &'static str, future: F) -> Result<F::Output> {
    let mut future = pin!(future);
    block_on_poll(call, |cx| future.as_mut().poll(cx))
}

/// A one-shot value delivered by a callback and awaited by a parked caller.
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

    /// Deliver the value and wake whoever parked on it.
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

    /// Park the caller until the value arrives. `call` names the wait in the
    /// refusal a caller that cannot park gets.
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

/// Threads the engine starts for itself: the landing worker and the guest
/// runners.
pub(crate) mod thread {
    /// Starts a named thread running `f`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn spawn(name: String, f: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
        std::thread::Builder::new().name(name).spawn(f).map(|_| ())
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn spawn(name: String, f: impl FnOnce() + Send + 'static) -> std::io::Result<()> {
        web_rt::thread::Builder::new()
            .name(name)
            .spawn(f)
            .map(|_| ())
    }
}

/// Unbounded channels with the `mpsc` shape.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod channel {
    pub(crate) use std::sync::mpsc::{SendError, Sender, channel as unbounded};
}

#[cfg(target_arch = "wasm32")]
pub(crate) mod channel {
    pub(crate) use web_rt::channel::{SendError, Sender, unbounded};
}

/// Native's poll thread: `Device::poll` is what fires wgpu's callbacks there,
/// and it blocks, so one thread does nothing else. It sleeps until kicked,
/// then polls (waiting for the latest submission) until the queue is empty
/// and no kick arrived meanwhile.
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
                    // Callbacks registered against an empty queue fire on
                    // the next poll, and a wait on an empty queue returns at
                    // once, so one poll per kick is always enough; a wait
                    // that times out is re-run by re-kicking.
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

    /// Something was submitted or a map was asked for: poll again.
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

/// The browser's queue fires its callbacks itself; there is no one to kick.
#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
pub(crate) struct Poller;

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
impl Poller {
    pub(crate) fn start(_device: wgpu::Device) -> Poller {
        Poller
    }

    pub(crate) fn kick(&self) {}
}
