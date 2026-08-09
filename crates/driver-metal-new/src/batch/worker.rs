//! The executor worker: one thread that owns the device state, and the only
//! thread that touches it.
//!
//! Metal command queues and MTL4 command allocators are not safe to drive
//! from several threads, and a control op (a K/V copy, a pool resize, a
//! close) must never run while a forward is mid-flight. The C++
//! `ExecutorWorker` answers both with one dedicated FIFO thread: every
//! executor touch is a job, jobs run in submission order, and the two
//! hazards become impossible *as long as everyone remembers to go through
//! the worker* — the context pointer is still right there, and nothing
//! stops a caller from using it directly.
//!
//! The Rust worker closes that gap by ownership: [`Worker::spawn`] takes a
//! factory, the state is **constructed on the worker thread and never
//! leaves it**, and a job is a closure that receives `&mut S`. There is no
//! other handle to the state, so "some other thread touched the context"
//! is not a discipline — it is unrepresentable. This is also what lets the
//! state hold the crate's single-threaded types (`Rc`, the `Runtime`, a
//! `Stepper`): `S` need not be `Send`, because it never crosses a thread;
//! only the *factory* and the *jobs* do.
//!
//! Two submission modes, as in the C++: [`run`](Worker::run) is
//! synchronous and returns the job's value (a panic inside the job crosses
//! back to the caller, and the worker survives); [`post`](Worker::post) is
//! fire-and-forget for the launch path, whose job owns its own error
//! handling — a stray panic is contained so one bad job cannot tear the
//! worker down. [`drain`](Worker::drain) is a synchronous no-op job, which
//! in a FIFO is a barrier.
//!
//! What did not survive: the C++ runs a same-thread `run` inline so a job
//! may re-enter the worker. A Rust job already holds `&mut S`; an inline
//! nested job would alias it, so re-entry is a contract violation and
//! panics with instructions rather than deadlocking silently.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::thread::{JoinHandle, ThreadId};

/// One queued unit: a job over the state, or the stop signal.
enum Message<S> {
    Job(Box<dyn FnOnce(&mut S) + Send>),
    Stop,
}

/// The single-owner serializer for one device's executor state.
///
/// Cloneable handles are deliberately absent: wrap the worker in an [`Arc`]
/// to share it, so the drop that stops the thread is the last one.
pub struct Worker<S> {
    sender: Mutex<Option<Sender<Message<S>>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    thread_id: ThreadId,
    submitted: AtomicU64,
}

impl<S: 'static> Worker<S> {
    /// Start the worker and build its state on the worker thread.
    ///
    /// The factory runs first, before any job; the state it returns lives
    /// on the thread until the worker drops. `S` itself need not be `Send`
    /// — that is the point.
    #[must_use]
    pub fn spawn<F>(factory: F) -> Worker<S>
    where
        F: FnOnce() -> S + Send + 'static,
    {
        let (sender, receiver) = channel::<Message<S>>();
        let (id_out, id_in) = channel();
        let thread = std::thread::spawn(move || {
            let _ = id_out.send(std::thread::current().id());
            let mut state = factory();
            while let Ok(message) = receiver.recv() {
                match message {
                    Message::Job(job) => job(&mut state),
                    Message::Stop => break,
                }
            }
        });
        let thread_id = id_in.recv().expect("the worker thread reports its id");
        Worker {
            sender: Mutex::new(Some(sender)),
            thread: Mutex::new(Some(thread)),
            thread_id,
            submitted: AtomicU64::new(0),
        }
    }

    /// Run `job` on the worker thread and return its value.
    ///
    /// Blocks until the job finishes, in FIFO order behind everything
    /// already queued. A panic inside the job resumes on this caller — the
    /// C++ rethrows the captured exception the same way — and the worker
    /// thread survives to take the next job.
    ///
    /// # Panics
    ///
    /// Re-entry: a job calling `run` on its own worker would deadlock (and
    /// alias its own `&mut S`), so it panics with instructions instead.
    pub fn run<R, F>(&self, job: F) -> R
    where
        R: Send + 'static,
        F: FnOnce(&mut S) -> R + Send + 'static,
    {
        assert!(
            std::thread::current().id() != self.thread_id,
            "a worker job re-entered Worker::run; it already holds &mut S — compose \
             functions over the state instead of re-submitting"
        );
        let (done, wait) = channel();
        self.send(Box::new(move |state| {
            let outcome = catch_unwind(AssertUnwindSafe(|| job(state)));
            let _ = done.send(outcome);
        }));
        match wait.recv().expect("the worker completes every job") {
            Ok(value) => value,
            Err(panic) => resume_unwind(panic),
        }
    }

    /// Queue `job` and return immediately.
    ///
    /// The launch path's mode: the ABI call returns after acceptance,
    /// before the forward and its settlement run. The job owns its own
    /// error handling by contract; a stray panic is contained so a single
    /// bad job cannot tear down the worker thread.
    pub fn post<F>(&self, job: F)
    where
        F: FnOnce(&mut S) + Send + 'static,
    {
        assert!(
            std::thread::current().id() != self.thread_id,
            "a worker job re-entered Worker::post; it already holds &mut S"
        );
        self.send(Box::new(move |state| {
            let _ = catch_unwind(AssertUnwindSafe(|| job(state)));
        }));
    }

    /// Block until every job submitted before this call has finished.
    ///
    /// FIFO makes a synchronous no-op a barrier.
    pub fn drain(&self) {
        self.run(|_| {});
    }

    /// How many jobs have been submitted.
    #[must_use]
    pub fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Relaxed)
    }

    fn send(&self, job: Box<dyn FnOnce(&mut S) + Send>) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
        let guard = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .as_ref()
            .expect("the worker accepts jobs until it drops")
            .send(Message::Job(job))
            .expect("the worker thread outlives its handle");
    }
}

impl<S> Drop for Worker<S> {
    /// Stop after the queue empties, then join.
    ///
    /// Jobs already queued still run — dropping the worker is a barrier,
    /// not an abort — matching the C++ destructor's drain-then-stop.
    fn drop(&mut self) {
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(sender) = sender {
            let _ = sender.send(Message::Stop);
        }
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }
}

impl<S> std::fmt::Debug for Worker<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("submitted", &self.submitted.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;

    /// The ownership fence: the state may be `!Send` (an `Rc`, like the
    /// runtime's), because it is built on the worker thread and never
    /// leaves. The C++ design cannot state this at all.
    #[test]
    fn the_state_never_leaves_its_thread_and_need_not_be_send() {
        let worker = Worker::spawn(|| Rc::new(std::cell::Cell::new(0u32)));
        let born_on = worker.run(|_| std::thread::current().id());
        assert_ne!(born_on, std::thread::current().id());
        worker.run(|state| state.set(state.get() + 41));
        assert_eq!(worker.run(|state| state.get() + 1), 42);
    }

    #[test]
    fn jobs_run_in_submission_order_and_drain_is_a_barrier() {
        let worker = Worker::spawn(Vec::new);
        for value in 0..64u32 {
            worker.post(move |state: &mut Vec<u32>| state.push(value));
        }
        worker.drain();
        let seen = worker.run(std::mem::take);
        assert_eq!(seen, (0..64).collect::<Vec<_>>());
        assert_eq!(worker.submitted(), 64 + 2);
    }

    #[test]
    fn a_jobs_panic_resumes_on_the_caller_and_the_worker_survives() {
        let worker = Worker::spawn(|| 7u32);
        let caught = std::panic::catch_unwind(AssertUnwindSafe(|| {
            worker.run(|_: &mut u32| panic!("the job is at fault"));
        }));
        assert!(
            caught.is_err(),
            "run rethrows the job's panic on the caller"
        );
        assert_eq!(
            worker.run(|state| *state),
            7,
            "the worker took the next job"
        );
    }

    #[test]
    fn a_posted_jobs_panic_is_contained() {
        let worker = Worker::spawn(|| 0u32);
        worker.post(|_: &mut u32| panic!("fire-and-forget owns its errors"));
        worker.post(|state: &mut u32| *state += 1);
        worker.drain();
        assert_eq!(worker.run(|state| *state), 1);
    }

    #[test]
    fn dropping_the_worker_finishes_the_queue_first() {
        let seen = Arc::new(AtomicU64::new(0));
        let worker = Worker::spawn(|| ());
        for _ in 0..32 {
            let seen = Arc::clone(&seen);
            worker.post(move |(): &mut ()| {
                seen.fetch_add(1, Ordering::Relaxed);
            });
        }
        drop(worker);
        assert_eq!(
            seen.load(Ordering::Relaxed),
            32,
            "drop is a barrier, not an abort"
        );
    }
}
