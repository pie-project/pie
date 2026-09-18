//! Green threads: `std::thread`'s surface over a fiber.
//!
//! The runtime's scheduler and engine lanes are ordinary loops that block in
//! a channel receive. On a tab they run on a fiber each — the same
//! stack-switching wasmtime uses for guest calls, so the page's JSPI
//! implementation serves both — and a block becomes a suspend. The executor
//! owns one task per green thread; polling the task resumes the fiber, and
//! the fiber suspending is the task returning `Pending`.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use wasmtime_fiber::{Fiber, FiberStack, Suspend};

type GreenSuspend = Suspend<(), (), ()>;

thread_local! {
    /// The suspend handle of the green thread that is running right now.
    /// Installed by the task that resumes a thread and restored when the
    /// resume returns, so a thread that spawned another (the scheduler
    /// spawns its engine lane) never finds the child's handle here.
    static CURRENT: Cell<*mut GreenSuspend> = const { Cell::new(std::ptr::null_mut()) };
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Where a green thread's body publishes its suspend handle on first entry,
/// for the task that owns it to install on every later resume.
type SuspendSlot = Rc<Cell<*mut GreenSuspend>>;

/// Default shadow-stack size. Rust on wasm32 keeps only address-taken locals
/// on this stack; the engine lane's frames are deep but not wide.
const DEFAULT_STACK: usize = 4 << 20;

pub struct Builder {
    name: Option<String>,
    stack_size: usize,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            name: None,
            stack_size: DEFAULT_STACK,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = size;
        self
    }

    pub fn spawn<F, T>(self, f: F) -> std::io::Result<JoinHandle<T>>
    where
        F: FnOnce() -> T + 'static,
        T: 'static,
    {
        let state = Arc::new(Mutex::new(JoinState {
            result: None,
            waker: None,
        }));
        let stack = FiberStack::new(self.stack_size, false)
            .map_err(|error| std::io::Error::other(format!("green thread stack: {error}")))?;
        let task_state = Arc::clone(&state);
        let slot: SuspendSlot = Rc::new(Cell::new(std::ptr::null_mut()));
        let published = Rc::clone(&slot);
        let fiber = Fiber::new(stack, move |(), suspend: &mut GreenSuspend| {
            let handle = suspend as *mut GreenSuspend;
            published.set(handle);
            CURRENT.with(|c| c.set(handle));
            let value = f();
            let mut s = task_state.lock().unwrap();
            s.result = Some(value);
            if let Some(w) = s.waker.take() {
                w.wake();
            }
        })
        .map_err(|(error, _stack)| std::io::Error::other(format!("green thread: {error}")))?;
        let _ = self.name;
        crate::executor::spawn(GreenTask {
            fiber: Some(fiber),
            suspend: slot,
        });
        Ok(JoinHandle { state })
    }
}

/// `std::thread::spawn` with the default stack.
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + 'static,
    T: 'static,
{
    Builder::new().spawn(f).expect("spawn green thread")
}

struct JoinState<T> {
    result: Option<T>,
    waker: Option<Waker>,
}

pub struct JoinHandle<T> {
    state: Arc<Mutex<JoinState<T>>>,
}

impl<T> JoinHandle<T> {
    /// From a green thread: block until the thread finishes. From anywhere
    /// else there is nothing to park, so the result is taken if it is already
    /// there and `Err` otherwise — a caller on the page's own stack cannot
    /// wait for anything.
    pub fn join(self) -> Result<T, Box<dyn Any + Send + 'static>> {
        if in_green_thread() {
            let state = Arc::clone(&self.state);
            return Ok(block_on_poll(|cx| {
                let mut s = state.lock().unwrap();
                if let Some(value) = s.result.take() {
                    Poll::Ready(value)
                } else {
                    s.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }));
        }
        self.state
            .lock()
            .unwrap()
            .result
            .take()
            .ok_or_else(|| Box::new("green thread still running") as Box<dyn Any + Send>)
    }

    pub fn is_finished(&self) -> bool {
        self.state.lock().unwrap().result.is_some()
    }
}

/// From a task: await the thread's result. The executor task that owns the
/// thread wakes the handle when the body returns.
impl<T> Future for JoinHandle<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut s = self.state.lock().unwrap();
        match s.result.take() {
            Some(value) => Poll::Ready(value),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

struct GreenTask {
    fiber: Option<Fiber<'static, (), (), ()>>,
    suspend: SuspendSlot,
}

impl Future for GreenTask {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let Some(fiber) = self.fiber.as_ref() else {
            return Poll::Ready(());
        };
        // Install this thread's handle and waker for the duration of the
        // resume, and put back whatever was there (a resume can nest inside
        // another green thread's poll only through the executor, but the
        // restore keeps that honest too). On first entry the body publishes
        // the handle itself.
        let previous_waker = PENDING_WAKER.with(|w| w.replace(Some(cx.waker().clone())));
        let previous = CURRENT.with(|c| c.replace(self.suspend.get()));
        DEPTH.with(|d| d.set(d.get() + 1));
        let outcome = fiber.resume(());
        DEPTH.with(|d| d.set(d.get() - 1));
        CURRENT.with(|c| c.set(previous));
        PENDING_WAKER.with(|w| *w.borrow_mut() = previous_waker);
        match outcome {
            Ok(()) => {
                self.fiber = None;
                Poll::Ready(())
            }
            Err(()) => Poll::Pending,
        }
    }
}

thread_local! {
    static PENDING_WAKER: RefCell<Option<Waker>> = const { RefCell::new(None) };
}

/// Whether the caller runs on a green thread, i.e. may block through
/// [`block_on_poll`].
pub fn in_green_thread() -> bool {
    DEPTH.with(|d| d.get() > 0)
}

/// Block the calling green thread until `f` reports `Ready`.
///
/// `f` is polled with the executor's waker for this thread; when it returns
/// `Pending` the fiber suspends, and it is polled again once that waker
/// fires. Calling this from anywhere other than a green thread panics: the
/// page's own stack has nowhere to go.
pub fn block_on_poll<T>(mut f: impl FnMut(&mut Context<'_>) -> Poll<T>) -> T {
    assert!(
        in_green_thread(),
        "blocking outside a green thread would stall the browser's event loop"
    );
    loop {
        let waker = PENDING_WAKER
            .with(|w| w.borrow().clone())
            .expect("a green thread is resumed with a waker");
        let mut cx = Context::from_waker(&waker);
        if let Poll::Ready(value) = f(&mut cx) {
            return value;
        }
        let suspend = CURRENT.with(|c| c.get());
        assert!(
            !suspend.is_null(),
            "a running green thread owns its suspend handle"
        );
        // SAFETY: the pointer was taken from the `&mut Suspend` wasmtime-fiber
        // handed the fiber body, which is live for the whole fiber; only the
        // running fiber reaches its own handle, and one call is in flight at
        // a time on one thread.
        unsafe { (*suspend).suspend(()) };
    }
}

/// Yield to the executor from a green thread without waiting on anything;
/// the thread is re-queued immediately.
pub fn yield_now() {
    block_on_poll(|cx| {
        cx.waker().wake_by_ref();
        Poll::Pending
    })
}
