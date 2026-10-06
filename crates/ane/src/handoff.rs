use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSharedEvent;

use crate::private::{Binding, Program};

const BOUND: Duration = Duration::from_secs(2);

struct Job {
    ready: u64,
    started: Option<Instant>,
}

struct State {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    jobs: Mutex<HashMap<u64, Job>>,
    value: AtomicU64,
    ids: AtomicU64,
    failures: AtomicU64,
    retired: AtomicBool,
    stop: AtomicBool,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}

impl State {
    fn retire(&self, why: &str) {
        if !self.retired.swap(true, Ordering::AcqRel) {
            eprintln!(
                "PIE_ANE: the Neural Engine split stopped ({why}); the GPU runs the MLP alone"
            );
        }
        self.event.setSignaledValue(
            self.value
                .load(Ordering::Acquire)
                .max(self.event.signaledValue()),
        );
    }

    fn fail(&self, why: &str) {
        self.failures.fetch_add(1, Ordering::AcqRel);
        self.retire(why);
    }

    fn reported(&self, id: u64, ok: bool) {
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.remove(&id);
        }
        if !ok {
            self.fail("an evaluation failed");
        }
    }

    fn watch(&self) {
        while !self.stop.load(Ordering::Acquire) {
            let now = Instant::now();
            let signaled = self.event.signaledValue();
            let expired = self.jobs.lock().ok().is_some_and(|mut jobs| {
                let mut expired = false;
                jobs.retain(|_, job| {
                    if job.started.is_none() && signaled >= job.ready {
                        job.started = Some(now);
                    }
                    let late = job.started.is_some_and(|at| now - at > BOUND);
                    expired |= late;
                    !late
                });
                expired
            });
            if expired {
                self.fail("an evaluation did not finish within 2 s");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

pub struct Monitor {
    state: Arc<State>,
}

impl Monitor {
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.state.failures.load(Ordering::Acquire)
    }

    pub fn retire(&self, why: &str) {
        self.state.fail(why);
    }
}

pub struct Handoff {
    state: Arc<State>,
    watchdog: Option<std::thread::JoinHandle<()>>,
}

impl Handoff {
    pub fn new(event: Retained<ProtocolObject<dyn MTLSharedEvent>>) -> std::io::Result<Handoff> {
        let state = Arc::new(State {
            event,
            jobs: Mutex::new(HashMap::new()),
            value: AtomicU64::new(0),
            ids: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            retired: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let watched = state.clone();
        let watchdog = std::thread::Builder::new()
            .name("pie-ane-watch".into())
            .spawn(move || watched.watch())?;
        Ok(Handoff {
            state,
            watchdog: Some(watchdog),
        })
    }

    #[must_use]
    pub fn event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.state.event
    }

    pub fn next(&self) -> u64 {
        let value = self.state.value.fetch_add(1, Ordering::AcqRel) + 1;
        if self.retired() {
            self.state.event.setSignaledValue(value);
        }
        value
    }

    #[must_use]
    pub fn retired(&self) -> bool {
        self.state.retired.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn failures(&self) -> u64 {
        self.state.failures.load(Ordering::Acquire)
    }

    pub fn retire(&self, why: &str) {
        self.state.retire(why);
    }

    #[must_use]
    pub fn monitor(&self) -> Monitor {
        Monitor {
            state: self.state.clone(),
        }
    }

    pub fn start(&self, program: &Program, binding: &Binding, ready: u64, done: u64) {
        if self.retired() {
            return;
        }
        let id = self.state.ids.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut jobs) = self.state.jobs.lock() {
            jobs.insert(
                id,
                Job {
                    ready,
                    started: None,
                },
            );
        }
        let state = self.state.clone();
        let fault = std::env::var_os("PIE_ANE_FAULT").is_some() && id == 0;
        let event = Retained::as_ptr(&self.state.event) as *mut std::ffi::c_void;
        let started = unsafe {
            program.enqueue(
                binding,
                event,
                ready,
                done,
                Box::new(move |ok| state.reported(id, ok && !fault)),
            )
        };
        if let Err(why) = started {
            self.state.reported(id, false);
            eprintln!("PIE_ANE: {why}");
        }
    }
}

impl Drop for Handoff {
    fn drop(&mut self) {
        self.state.retire("the model unloaded");
        let deadline = Instant::now() + BOUND;
        while Instant::now() < deadline
            && self
                .state
                .jobs
                .lock()
                .ok()
                .is_some_and(|jobs| !jobs.is_empty())
        {
            std::thread::sleep(Duration::from_millis(2));
        }
        self.state.stop.store(true, Ordering::Release);
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}
