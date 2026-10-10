//! The shared event the GPU and the Neural Engine hand work across. Its
//! value only rises: each hand-off takes two fresh values, one the GPU
//! signals when the inputs are ready and one the Neural Engine signals when
//! its partial is written.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLSharedEvent, MTLSharedEventListener};

use super::program::{Binding, Program};

/// The two event values of one hand-off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allotment {
    /// The GPU raises it when the inputs are packed.
    pub ready: u64,
    /// The Neural Engine raises it when the partial is written.
    pub done: u64,
}

struct State {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Where the event tells us the GPU has raised a `ready` value, so the
    /// request can be submitted then and not before.
    listener: Retained<MTLSharedEventListener>,
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
            listener: MTLSharedEventListener::new(),
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

    /// One hand-off's values.
    pub fn allot(&self) -> Allotment {
        Allotment {
            ready: self.next(),
            done: self.next(),
        }
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
                "PIE_ANE: the Neural Engine split stopped ({why}); the GPU runs everything alone"
            );
        }
        let at = self.0.value.load(Ordering::Acquire);
        self.0
            .event
            .setSignaledValue(at.max(self.0.event.signaledValue()));
    }

    /// Arranges for `binding` to run once the GPU raises `ready`, signaling
    /// `done` after. The request is submitted from the event's listener at
    /// that moment, not before, so the Neural Engine's queue never holds a
    /// request behind its wait. A failure, at submission or on completion,
    /// retires the hand-off.
    pub fn start(&self, program: &Program, binding: &Binding, ready: u64, done: u64) {
        if self.retired() {
            return;
        }
        let state = self.0.clone();
        let (program, binding) = (program.clone(), binding.clone());
        let block = block2::RcBlock::new(
            move |_: core::ptr::NonNull<ProtocolObject<dyn MTLSharedEvent>>, _: u64| {
                // Retired since: the event sits at its ceiling, and a request
                // signaling `done` now would pull it back down.
                if state.retired.load(Ordering::Acquire) {
                    return;
                }
                let this = Handoff(state.clone());
                let report = Box::new(move |ok: bool| {
                    if !ok {
                        this.fail("an evaluation failed");
                    }
                });
                if let Err(why) =
                    unsafe { program.enqueue(&binding, &state.event, ready, done, report) }
                {
                    Handoff(state.clone()).fail(&why);
                }
            },
        );
        // SAFETY: `notifyListener:atValue:block:` copies the block; the
        // listener and the event are the hand-off's own and outlive it.
        unsafe {
            self.0.event.notifyListener_atValue_block(
                &self.0.listener,
                ready,
                block2::RcBlock::as_ptr(&block),
            );
        }
    }
}
