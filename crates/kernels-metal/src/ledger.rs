//! How the GPU and the Neural Engine hand work across without either ever
//! waiting on a value nobody will raise, kept apart from Metal so it can be
//! held to account off Apple hardware.
//!
//! Each hand-off takes two fresh values: `ready`, which the GPU raises once
//! the inputs are packed, and `done`, which the Neural Engine raises once its
//! partial is written and which the GPU waits on. They live on two events,
//! one per direction, so each has one writer that raises its values in the
//! order they were taken: the GPU raises `ready` in queue order, and the
//! Neural Engine is handed one request at a time, in commit order, so it
//! raises `done` in that order too. A Metal shared event takes whatever it
//! is set to, lower values included; with one writer per event and that
//! writer in order, neither ever falls.
//!
//! Only a committed frame's hand-offs reach the Neural Engine (the frame
//! pushes them as it commits): a frame whose encode stopped short never runs
//! on the GPU, and a request for it would answer a `done` out of turn. A
//! request that comes back refused or failed
//! has its `done` raised here, and only then is the next one listened for,
//! so this side's raise can never land after the Neural Engine's next one.
//! Retiring the split stops new hand-offs being planned; the ones already
//! committed still go to the Neural Engine, which answers or fails each.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The two event values of one hand-off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allotment {
    /// The GPU raises it on the `ready` event when the inputs are packed.
    pub ready: u64,
    /// The Neural Engine raises it on the `done` event when the partial is
    /// written.
    pub done: u64,
}

/// What the caller does once a request is back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Back {
    /// Raise the `done` event to this first, if it stands below it: the
    /// Neural Engine did not.
    pub raise: Option<u64>,
    /// This failure retired the split.
    pub retired: bool,
    /// Then listen for this hand-off's `ready`.
    pub next: Option<Allotment>,
}

pub struct Ledger<P> {
    taken: AtomicU64,
    failures: AtomicU64,
    retired: AtomicBool,
    /// Committed hand-offs, in order; the front is listened for or out with
    /// the Neural Engine.
    queued: Mutex<VecDeque<(Allotment, P)>>,
}

impl<P> Default for Ledger<P> {
    fn default() -> Self {
        Ledger {
            taken: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            retired: AtomicBool::new(false),
            queued: Mutex::new(VecDeque::new()),
        }
    }
}

impl<P> Ledger<P> {
    #[must_use]
    pub fn new() -> Ledger<P> {
        Ledger::default()
    }

    fn queued(&self) -> std::sync::MutexGuard<'_, VecDeque<(Allotment, P)>> {
        self.queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The next value to hand off on.
    pub fn next(&self) -> u64 {
        self.taken.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// One hand-off's values.
    pub fn allot(&self) -> Allotment {
        Allotment {
            ready: self.next(),
            done: self.next(),
        }
    }

    /// Whether the Neural Engine has failed and no new hand-off is planned.
    #[must_use]
    pub fn retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures.load(Ordering::Acquire)
    }

    /// Stops new hand-offs being planned. True the first time, so the caller
    /// says why once.
    pub fn retire(&self) -> bool {
        self.failures.fetch_add(1, Ordering::AcqRel);
        !self.retired.swap(true, Ordering::AcqRel)
    }

    /// A committed hand-off joins the queue: `at` itself, to listen for, when
    /// the queue was idle.
    pub fn push(&self, at: Allotment, payload: P) -> Option<Allotment> {
        let mut queued = self.queued();
        queued.push_back((at, payload));
        (queued.len() == 1).then_some(at)
    }

    /// The `ready` event reached the front's `ready`: hands its payload over
    /// to be submitted. The front stays queued until it is back.
    pub fn take(&self, at: Allotment) -> Option<P>
    where
        P: Clone,
    {
        let queued = self.queued();
        let (front, payload) = queued.front()?;
        (*front == at).then(|| payload.clone())
    }

    /// The front is back from the Neural Engine, `ok` or not.
    pub fn back(&self, at: Allotment, ok: bool) -> Back {
        let retired = !ok && self.retire();
        let mut queued = self.queued();
        if queued.front().map(|(front, _)| *front) == Some(at) {
            queued.pop_front();
        }
        let next = queued.front().map(|(front, _)| *front);
        Back {
            raise: (!ok).then_some(at.done),
            retired,
            next,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Metal shared event: set means set, lower values included. Every set
    /// is kept, so a fall shows.
    #[derive(Default)]
    struct Event(Vec<u64>);

    impl Event {
        fn value(&self) -> u64 {
            self.0.last().copied().unwrap_or(0)
        }
        fn set(&mut self, value: u64) {
            self.0.push(value);
        }
        fn never_fell(&self) -> bool {
            self.0.windows(2).all(|pair| pair[0] <= pair[1])
        }
    }

    /// Both engines and both events, stepped by hand. The Neural Engine runs
    /// what it is handed at once, unless told to fail it.
    struct World {
        ledger: Ledger<u32>,
        ready: Event,
        done: Event,
        listening: Option<Allotment>,
        ran: Vec<u32>,
    }

    impl World {
        fn new() -> World {
            World {
                ledger: Ledger::new(),
                ready: Event::default(),
                done: Event::default(),
                listening: None,
                ran: Vec::new(),
            }
        }

        /// A frame commits: its hand-offs join the queue, in order.
        fn commit(&mut self, hand_offs: &[(Allotment, u32)]) {
            for &(at, payload) in hand_offs {
                if let Some(head) = self.ledger.push(at, payload) {
                    self.listening = Some(head);
                }
            }
            self.wake(true);
        }

        /// The GPU raises `at.ready`.
        fn gpu_ready(&mut self, at: Allotment, engine_ok: bool) {
            self.ready.set(at.ready);
            self.wake(engine_ok);
        }

        /// Fires the listener while the `ready` event stands at or past it,
        /// and runs what it hands over.
        fn wake(&mut self, engine_ok: bool) {
            while let Some(at) = self.listening
                && self.ready.value() >= at.ready
            {
                self.listening = None;
                let payload = self.ledger.take(at).expect("the listened-for front");
                if engine_ok {
                    self.done.set(at.done);
                    self.ran.push(payload);
                }
                let back = self.ledger.back(at, engine_ok);
                if let Some(value) = back.raise
                    && self.done.value() < value
                {
                    self.done.set(value);
                }
                self.listening = back.next;
            }
        }

        fn gpu_passes(&self, at: Allotment) -> bool {
            self.done.value() >= at.done
        }
    }

    #[test]
    fn a_committed_hand_off_is_answered_and_the_gpu_passes() {
        let mut w = World::new();
        let at = w.ledger.allot();
        w.commit(&[(at, 1)]);
        w.gpu_ready(at, true);
        assert!(w.gpu_passes(at));
        assert_eq!(w.ran, [1]);
    }

    #[test]
    fn a_failed_request_is_answered_here_and_the_split_retires() {
        let mut w = World::new();
        let (first, second) = (w.ledger.allot(), w.ledger.allot());
        w.commit(&[(first, 1)]);
        w.commit(&[(second, 2)]);

        w.gpu_ready(first, false);
        assert!(w.gpu_passes(first), "the GPU waits on a done nobody raises");
        assert!(w.ledger.retired());

        // Committed before the failure, so it still goes to the engine.
        w.gpu_ready(second, true);
        assert!(w.gpu_passes(second));
        assert_eq!(w.ran, [2]);
        assert!(w.done.never_fell() && w.ready.never_fell());
    }

    #[test]
    fn retiring_leaves_committed_hand_offs_to_the_neural_engine() {
        let mut w = World::new();
        let at = w.ledger.allot();
        w.commit(&[(at, 1)]);
        assert!(w.ledger.retire(), "a partial that was not finite");
        w.gpu_ready(at, true);
        assert!(w.gpu_passes(at));
        assert_eq!(w.ran, [1]);
    }

    #[test]
    fn the_next_request_waits_for_the_one_before_to_come_back() {
        let mut w = World::new();
        let (first, second) = (w.ledger.allot(), w.ledger.allot());
        w.commit(&[(first, 1), (second, 2)]);
        assert_eq!(w.listening, Some(first));
        assert_eq!(w.ledger.take(second), None, "only the front is handed over");
        w.gpu_ready(first, true);
        assert_eq!(w.listening, Some(second));
        w.gpu_ready(second, true);
        assert_eq!(w.ran, [1, 2]);
        assert_eq!(w.listening, None);
    }

    #[test]
    fn a_frame_the_gpu_dropped_is_run_behind_the_next_ready_and_nothing_falls() {
        let mut w = World::new();
        let (dropped, next) = (w.ledger.allot(), w.ledger.allot());
        w.commit(&[(dropped, 1)]);
        w.commit(&[(next, 2)]);
        // The first frame's command buffer failed before its signal; the next
        // frame raises its own ready, past the first's.
        w.gpu_ready(next, true);
        assert!(w.gpu_passes(next));
        assert_eq!(w.ran, [1, 2]);
        assert!(w.done.never_fell() && w.ready.never_fell());
    }

    #[test]
    fn a_long_run_of_frames_with_failures_never_lets_an_event_fall() {
        let mut w = World::new();
        for frame in 0..64u32 {
            let hand_offs: Vec<(Allotment, u32)> =
                (0..3).map(|i| (w.ledger.allot(), frame * 3 + i)).collect();
            if frame % 7 == 3 {
                continue; // its encode stopped short: never committed
            }
            w.commit(&hand_offs);
            for (i, &(at, _)) in hand_offs.iter().enumerate() {
                let ok = !(frame + i as u32).is_multiple_of(5);
                w.gpu_ready(at, ok);
                assert!(w.gpu_passes(at), "frame {frame} hand-off {i}");
            }
        }
        assert!(w.done.never_fell() && w.ready.never_fell());
        assert_eq!(w.listening, None);
    }
}
