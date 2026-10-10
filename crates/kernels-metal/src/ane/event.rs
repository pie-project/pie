//! The two shared events the GPU and the Neural Engine hand work across:
//! `ready`, which only the GPU raises, and `done`, which only the Neural
//! Engine raises (and this side, for a request that came back without). The
//! order that keeps either from falling is the [`ledger`](crate::ledger)'s.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLSharedEvent, MTLSharedEventListener};

use super::program::{Binding, Program};
pub use crate::ledger::Allotment;
use crate::ledger::Ledger;

type Event = Retained<ProtocolObject<dyn MTLSharedEvent>>;

struct State {
    ready: Event,
    done: Event,
    /// Where the `ready` event tells us the GPU has raised the front's
    /// value, so its request is submitted then and not before.
    listener: Retained<MTLSharedEventListener>,
    ledger: Ledger<(Program, Binding)>,
}

unsafe impl Send for State {}
unsafe impl Sync for State {}

#[derive(Clone)]
pub struct Handoff(Arc<State>);

impl Handoff {
    pub fn new(ready: Event, done: Event) -> Handoff {
        Handoff(Arc::new(State {
            ready,
            done,
            listener: MTLSharedEventListener::new(),
            ledger: Ledger::new(),
        }))
    }

    /// The event the GPU signals once a hand-off's inputs are packed.
    #[must_use]
    pub fn ready_event(&self) -> &Event {
        &self.0.ready
    }

    /// The event the GPU waits on for a hand-off's partial.
    #[must_use]
    pub fn done_event(&self) -> &Event {
        &self.0.done
    }

    /// The next value to hand off on.
    pub fn next(&self) -> u64 {
        self.0.ledger.next()
    }

    /// One hand-off's values.
    pub fn allot(&self) -> Allotment {
        self.0.ledger.allot()
    }

    /// Whether the Neural Engine has failed and no new split is planned.
    #[must_use]
    pub fn retired(&self) -> bool {
        self.0.ledger.retired()
    }

    #[must_use]
    pub fn failures(&self) -> u64 {
        self.0.ledger.failures()
    }

    /// Stops new splits being planned, and says why once. Hand-offs already
    /// committed still run.
    pub fn fail(&self, why: &str) {
        if self.0.ledger.retire() {
            stopped(why);
        }
    }

    /// Stages `binding` to run once the GPU raises `ready`, signaling `done`
    /// after. It reaches the Neural Engine only if this frame commits.
    pub fn start(&self, program: &Program, binding: &Binding, ready: u64, done: u64) {
        self.0.ledger.stage(
            Allotment { ready, done },
            (program.clone(), binding.clone()),
        );
    }

    /// Drops what an encode that stopped short staged.
    pub fn discard_staged(&self) {
        self.0.ledger.discard_staged();
    }

    /// The frame being encoded committed: its hand-offs join the queue.
    pub fn commit_staged(&self) {
        if let Some(head) = self.0.ledger.commit_staged() {
            listen(&self.0, head);
        }
    }
}

/// Submits `at`'s request once the `ready` event reaches it. One request is
/// out at a time: the next is listened for only once this one is back.
fn listen(state: &Arc<State>, at: Allotment) {
    let owner = state.clone();
    let block = block2::RcBlock::new(
        move |_: core::ptr::NonNull<ProtocolObject<dyn MTLSharedEvent>>, _: u64| {
            let Some((program, binding)) = owner.ledger.take(at) else {
                return;
            };
            let state = owner.clone();
            let report = Box::new(move |ok: bool| {
                back(&state, at, ok, "an evaluation failed");
            });
            if let Err(why) = unsafe {
                program.enqueue(
                    &binding,
                    &owner.ready,
                    &owner.done,
                    at.ready,
                    at.done,
                    report,
                )
            } {
                back(&owner, at, false, &why);
            }
        },
    );
    // SAFETY: `notifyListener:atValue:block:` copies the block; the
    // listener and the event are the hand-off's own and outlive it.
    unsafe {
        state.ready.notifyListener_atValue_block(
            &state.listener,
            at.ready,
            block2::RcBlock::as_ptr(&block),
        );
    }
}

/// `at`'s request is back. One that failed has its `done` raised here
/// before the next request is listened for, so the Neural Engine's next
/// raise always lands after it.
fn back(state: &Arc<State>, at: Allotment, ok: bool, why: &str) {
    let back = state.ledger.back(at, ok);
    if let Some(value) = back.raise
        && state.done.signaledValue() < value
    {
        state.done.setSignaledValue(value);
    }
    if back.retired {
        stopped(why);
    }
    if let Some(next) = back.next {
        listen(state, next);
    }
}

fn stopped(why: &str) {
    eprintln!("PIE_ANE: the Neural Engine split stopped ({why}); the GPU runs everything alone");
}
