use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

type Task = Pin<Box<dyn Future<Output = ()>>>;

pub type WakeHook = fn();

static WAKE_HOOK: OnceLock<WakeHook> = OnceLock::new();

pub fn set_wake_hook(hook: WakeHook) {
    let _ = WAKE_HOOK.set(hook);
}

struct Shared {
    ready: Mutex<VecDeque<usize>>,
    armed: AtomicBool,
}

impl Shared {
    fn push(&self, id: usize) {
        self.ready.lock().unwrap().push_back(id);
        if !self.armed.swap(true, Ordering::AcqRel)
            && let Some(hook) = WAKE_HOOK.get()
        {
            hook();
        }
    }
}

struct TaskWaker {
    id: usize,
    queued: AtomicBool,
    shared: Arc<Shared>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            self.shared.push(self.id);
        }
    }
}

struct Slot {
    task: Option<Task>,
    waker: Arc<TaskWaker>,
}

struct Executor {
    slots: slab::Slab<Slot>,
    shared: Arc<Shared>,
}

thread_local! {
    static EXECUTOR: RefCell<Executor> = RefCell::new(Executor {
        slots: slab::Slab::new(),
        shared: Arc::new(Shared {
            ready: Mutex::new(VecDeque::new()),
            armed: AtomicBool::new(false),
        }),
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinError;

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("task was aborted")
    }
}

impl std::error::Error for JoinError {}

impl JoinError {
    pub fn is_panic(&self) -> bool {
        false
    }
    pub fn is_cancelled(&self) -> bool {
        true
    }
}

struct JoinState<T> {
    result: Option<T>,
    waker: Option<Waker>,
    aborted: bool,
    finished: bool,
}

pub struct JoinHandle<T> {
    state: Arc<Mutex<JoinState<T>>>,
}

impl<T> JoinHandle<T> {
    pub fn abort(&self) {
        self.state.lock().unwrap().aborted = true;
    }

    pub fn is_finished(&self) -> bool {
        self.state.lock().unwrap().finished
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, JoinError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap();
        if let Some(value) = state.result.take() {
            return Poll::Ready(Ok(value));
        }
        if state.finished {
            return Poll::Ready(Err(JoinError));
        }
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + 'static,
    F::Output: 'static,
{
    let state = Arc::new(Mutex::new(JoinState {
        result: None,
        waker: None,
        aborted: false,
        finished: false,
    }));
    let task_state = Arc::clone(&state);
    let mut future = Box::pin(future);
    let task: Task = Box::pin(std::future::poll_fn(move |cx| {
        if task_state.lock().unwrap().aborted {
            let mut s = task_state.lock().unwrap();
            s.finished = true;
            if let Some(w) = s.waker.take() {
                w.wake();
            }
            return Poll::Ready(());
        }
        match future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(value) => {
                let mut s = task_state.lock().unwrap();
                s.result = Some(value);
                s.finished = true;
                if let Some(w) = s.waker.take() {
                    w.wake();
                }
                Poll::Ready(())
            }
        }
    }));
    EXECUTOR.with(|e| {
        let mut e = e.borrow_mut();
        let shared = Arc::clone(&e.shared);
        let entry = e.slots.vacant_entry();
        let id = entry.key();
        let waker = Arc::new(TaskWaker {
            id,
            queued: AtomicBool::new(false),
            shared,
        });
        entry.insert(Slot {
            task: Some(task),
            waker: Arc::clone(&waker),
        });
        waker.wake_by_ref();
    });
    JoinHandle { state }
}

pub fn spawn_blocking<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + 'static,
    T: 'static,
{
    spawn(async move { f() })
}

pub async fn yield_now() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;
}

pub fn tick(budget: usize) -> Option<Duration> {
    let shared = EXECUTOR.with(|e| Arc::clone(&e.borrow().shared));
    let mut polls = 0;
    loop {
        shared.armed.store(true, Ordering::Release);
        while polls < budget {
            let Some(id) = shared.ready.lock().unwrap().pop_front() else {
                break;
            };
            polls += 1;
            let taken = EXECUTOR.with(|e| {
                let mut e = e.borrow_mut();
                e.slots.get_mut(id).and_then(|slot| {
                    slot.waker.queued.store(false, Ordering::Release);
                    slot.task.take().map(|task| (task, Arc::clone(&slot.waker)))
                })
            });
            let Some((mut task, waker)) = taken else {
                continue;
            };
            let waker_ref = Waker::from(Arc::clone(&waker));
            let mut cx = Context::from_waker(&waker_ref);
            match task.as_mut().poll(&mut cx) {
                Poll::Ready(()) => {
                    EXECUTOR.with(|e| {
                        let mut e = e.borrow_mut();
                        if e.slots.contains(id) {
                            e.slots.remove(id);
                        }
                    });
                }
                Poll::Pending => {
                    EXECUTOR.with(|e| {
                        if let Some(slot) = e.borrow_mut().slots.get_mut(id) {
                            slot.task = Some(task);
                        }
                    });
                }
            }
        }
        let fired = crate::time::fire_due();
        let more = !shared.ready.lock().unwrap().is_empty();
        if fired == 0 && !more {
            break;
        }
        if polls >= budget {
            break;
        }
    }
    let pending = !shared.ready.lock().unwrap().is_empty();
    shared.armed.store(false, Ordering::Release);
    if pending {
        return Some(Duration::ZERO);
    }
    crate::time::next_deadline()
}

pub fn task_count() -> usize {
    EXECUTOR.with(|e| e.borrow().slots.len())
}
