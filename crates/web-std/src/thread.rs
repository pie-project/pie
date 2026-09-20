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
    static CURRENT: Cell<*mut GreenSuspend> = const { Cell::new(std::ptr::null_mut()) };
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

type SuspendSlot = Rc<Cell<*mut GreenSuspend>>;

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

pub fn in_green_thread() -> bool {
    DEPTH.with(|d| d.get() > 0)
}

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

pub fn yield_now() {
    block_on_poll(|cx| {
        cx.waker().wake_by_ref();
        Poll::Pending
    })
}
