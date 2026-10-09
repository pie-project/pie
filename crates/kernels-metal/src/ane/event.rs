//! The shared event the GPU and the Neural Engine hand work across. Its
//! value only rises: each hand-off takes two fresh values, one the GPU
//! signals when the inputs are ready and one the Neural Engine signals when
//! its partial is written.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSharedEvent;

use super::program::{Binding, Program};

struct State {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: AtomicU64,
    failures: AtomicU64,
    retired: AtomicBool,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}

#[derive(Clone)]
pub struct Handoff(Arc<State>);

impl Handoff {
    pub fn new(event: Retained<ProtocolObject<dyn MTLSharedEvent>>) -> Handoff {
        Handoff(Arc::new(State {
            event,
            value: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            retired: AtomicBool::new(false),
        }))
    }

    #[must_use]
    pub fn event(&self) -> &Retained<ProtocolObject<dyn MTLSharedEvent>> {
        &self.0.event
    }

    /// The next value to hand off on. Once retired, every value is signaled
    /// at once, so a GPU wait on it never blocks.
    pub fn next(&self) -> u64 {
        let value = self.0.value.fetch_add(1, Ordering::AcqRel) + 1;
        if self.retired() {
            self.0.event.setSignaledValue(value);
        }
        value
    }

    /// Whether the Neural Engine has failed and the split is off for good.
    #[must_use]
    pub fn retired(&self) -> bool {
        self.0.retired.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn failures(&self) -> u64 {
        self.0.failures.load(Ordering::Acquire)
    }

    /// Retires the hand-off: says why once, and signals every value taken so
    /// far so no GPU command buffer waits on a partial that will never come.
    pub fn fail(&self, why: &str) {
        self.0.failures.fetch_add(1, Ordering::AcqRel);
        if !self.0.retired.swap(true, Ordering::AcqRel) {
            eprintln!(
                "PIE_ANE: the Neural Engine split stopped ({why}); the GPU runs the MLP alone"
            );
        }
        let at = self.0.value.load(Ordering::Acquire);
        self.0
            .event
            .setSignaledValue(at.max(self.0.event.signaledValue()));
    }

    /// Queues `binding` to run once the GPU signals `ready`, signaling `done`
    /// after. A failure, now or on completion, retires the hand-off.
    pub fn start(&self, program: &Program, binding: &Binding, ready: u64, done: u64) {
        if self.retired() {
            return;
        }
        let this = self.clone();
        let report = Box::new(move |ok: bool| {
            if !ok {
                this.fail("an evaluation failed");
            }
        });
        if let Err(why) = unsafe { program.enqueue(binding, &self.0.event, ready, done, report) } {
            self.fail(&why);
        }
    }
}
