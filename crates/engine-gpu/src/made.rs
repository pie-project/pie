//! What a backend creates on its device once its load serves: every layout,
//! descriptor pool or set, bind group and buffer a serving fire, a bind or a
//! pool operation makes (a kernel's or a guest step's pipeline, and its
//! layout, is counted where it is built, by the kernel it serves). The rule
//! is that the load makes every object serving uses, so each one noted here
//! is one it did not make ([`engine::live::Tally::BuiltAfterReady`]).

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// One object made after ready: what it is, and whose (a buffer's owner in
/// the device's books, the kernel of a pipeline). `session` where a guest
/// session's construction made it ([`in_session`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Creation {
    pub what: &'static str,
    pub by: &'static str,
    pub session: bool,
}

/// A device's note of what it made after ready.
#[derive(Debug, Default)]
pub struct Made {
    ready: AtomicBool,
    after: Mutex<Vec<Creation>>,
}

thread_local! {
    static SESSION: Cell<u32> = const { Cell::new(0) };
}

/// Runs `f` as a guest session's construction: what it makes is noted as
/// the session's, which a guest's per-bind buffers are.
pub fn in_session<R>(f: impl FnOnce() -> R) -> R {
    SESSION.with(|depth| depth.set(depth.get() + 1));
    let answer = f();
    SESSION.with(|depth| depth.set(depth.get() - 1));
    answer
}

impl Made {
    /// From here on the load serves: what is made is noted.
    pub fn ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Notes a `what` of `by` made, where the load serves.
    pub fn note(&self, what: &'static str, by: &'static str) {
        if !self.is_ready() {
            return;
        }
        let session = SESSION.with(|depth| depth.get() > 0);
        self.after
            .lock()
            .expect("made lock poisoned")
            .push(Creation { what, by, session });
    }

    /// What was made since the last take.
    pub fn take(&self) -> Vec<Creation> {
        std::mem::take(&mut *self.after.lock().expect("made lock poisoned"))
    }
}

#[cfg(test)]
mod tests {
    use super::{Made, in_session};

    /// Nothing is noted before ready; after it each object is, a session's
    /// construction's as the session's.
    #[test]
    fn what_is_made_after_ready_is_noted_and_a_session_s_apart() {
        let made = Made::default();
        made.note("pipeline", "load");
        assert!(made.take().is_empty());
        made.ready();
        made.note("buffer", "readout");
        in_session(|| made.note("buffer", "guest"));
        let taken = made.take();
        assert_eq!(taken.len(), 2);
        assert!(!taken[0].session);
        assert!(taken[1].session);
        assert!(made.take().is_empty());
    }
}
