//! What the GPU says about a step after it has run.
//!
//! # The event says the step ended, not that it worked
//!
//! [`Stepper`](super::encoder::Stepper) waits on a shared event, and the
//! queue signals that event as a separate operation after the commit. A
//! command buffer that FAULTED still reaches the signal: the wait returns,
//! the step reports success, and the output buffer holds whatever it held.
//! Commit feedback is the only place Metal 4 says otherwise -- there is no
//! status to read off the command buffer and no error out-parameter on
//! `commit:count:options:`.
//!
//! So this is not instrumentation. Without it a GPU fault is indistinguishable
//! from a model that answers badly.
//!
//! # It lands late, and out of order
//!
//! Metal invokes the handler on a queue of its own, after the fence the step
//! waited on. So a step's own feedback is usually NOT available when
//! [`Stepper::run`](super::encoder::Stepper::run) returns; it arrives during
//! the next one. Two consequences shape this module:
//!
//! * The state is an [`Arc`], not a field. Handlers can fire after everything
//!   that created them is dropped, and the block is what has to keep the
//!   state alive.
//! * Only the newest is kept, by event value rather than by arrival, because
//!   two handlers can land in either order.
//!
//! The error path is therefore deferred by one step: a fault is raised at the
//! start of the step after the one that caused it. That is later than one
//! would like and still the earliest it can be known.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use block2::RcBlock;
use objc2_metal::{MTL4CommitFeedback, MTL4CommitOptions};

/// What the GPU reported about one committed step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Feedback {
    /// The event value of the step this describes.
    pub step: u64,
    /// Host time the GPU began the step, in seconds.
    pub gpu_start: f64,
    /// Host time the GPU finished the step, in seconds.
    pub gpu_end: f64,
    /// What went wrong, if anything did.
    pub error: Option<String>,
}

impl Feedback {
    /// How long the GPU spent on the step.
    ///
    /// The two timestamps are host times on the same clock, so the difference
    /// is real wall time on the device -- and it excludes the encode and the
    /// wait, which is what makes it worth reading separately from the host's
    /// own measurement of the step.
    #[must_use]
    pub fn gpu_time(&self) -> Duration {
        Duration::from_secs_f64((self.gpu_end - self.gpu_start).max(0.0))
    }

    /// Whether the GPU reported a fault.
    #[must_use]
    pub const fn failed(&self) -> bool {
        self.error.is_some()
    }
}

/// The newest feedback the GPU has delivered.
///
/// Cloneable, and a clone observes the same state: the handler writes through
/// whichever clone the block captured.
#[derive(Debug, Clone, Default)]
pub struct Feedbacks {
    latest: Arc<Mutex<Option<Feedback>>>,
}

impl Feedbacks {
    /// An observer with nothing in it yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The newest feedback delivered so far, if any.
    ///
    /// Does not wait. See the module docs: the step that just returned has
    /// usually not been reported on yet.
    #[must_use]
    pub fn latest(&self) -> Option<Feedback> {
        self.locked().clone()
    }

    /// Wait up to `timeout` for feedback describing step `step` or later.
    ///
    /// Polls, because Metal offers nothing to block on. Intended for tracing
    /// and for tests; the step path itself does not wait, and reads whatever
    /// has landed by the next step instead.
    #[must_use]
    pub fn await_step(&self, step: u64, timeout: Duration) -> Option<Feedback> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(f) = self.latest()
                && f.step >= step
            {
                return Some(f);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Take the newest feedback if it reports a fault newer than `surfaced`.
    ///
    /// Used by the step path to raise a fault exactly once, at the start of
    /// the step after the one that caused it.
    pub(crate) fn take_error_after(&self, surfaced: u64) -> Option<Feedback> {
        self.locked()
            .clone()
            .filter(|f| f.failed() && f.step > surfaced)
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, Option<Feedback>> {
        // A poisoned lock here means a handler panicked, which cannot happen:
        // the handler does no allocation-fallible work and cannot unwind into
        // Objective-C. Recovering is still better than a panic in a Drop path.
        self.latest.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Build the commit options for one commit, with the handler attached.
    ///
    /// A FRESH options object every time. `addFeedbackHandler:` appends, so a
    /// shared instance accumulates one handler per step forever -- and the
    /// class is documented as not thread-safe.
    pub(crate) fn options(
        &self,
        step: u64,
    ) -> (RcBlockHandler, objc2::rc::Retained<MTL4CommitOptions>) {
        let options = MTL4CommitOptions::new();
        let latest = Arc::clone(&self.latest);
        let handler = RcBlock::new(
            move |feedback: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn MTL4CommitFeedback>,
            >| {
                // SAFETY: Metal hands the handler a live feedback object for
                // the duration of the call.
                let feedback = unsafe { feedback.as_ref() };
                let got = Feedback {
                    step,
                    gpu_start: feedback.GPUStartTime(),
                    gpu_end: feedback.GPUEndTime(),
                    error: feedback.error().map(|e| describe_gpu_error(&e)),
                };
                let mut slot = latest.lock().unwrap_or_else(|e| e.into_inner());
                // Handlers can land in either order; keep the newest.
                if slot.as_ref().is_none_or(|old| got.step >= old.step) {
                    *slot = Some(got);
                }
            },
        );
        // SAFETY: the block is kept alive by the returned handle, which the
        // caller holds across the commit. `addFeedbackHandler:` copies the
        // block, but the copy is only guaranteed after the call returns.
        unsafe { options.addFeedbackHandler(RcBlock::as_ptr(&handler)) };
        (handler, options)
    }
}

/// Keeps the feedback block alive across the commit that installs it.
pub(crate) type RcBlockHandler =
    RcBlock<dyn Fn(std::ptr::NonNull<objc2::runtime::ProtocolObject<dyn MTL4CommitFeedback>>)>;

/// Spell out an `NSError` from the queue, because its description does not.
///
/// The localized description of an MTL4 queue error is "the operation
/// couldn't be completed", which names nothing. The domain, the code and the
/// underlying error are what say what went wrong -- and a step that faults
/// nineteen gigabytes into a model gives no other clue.
fn describe_gpu_error(error: &objc2_foundation::NSError) -> String {
    let mut out = format!(
        "{} [{} code {}]",
        error.localizedDescription(),
        error.domain(),
        error.code()
    );
    if let Some(underlying) = error
        .userInfo()
        .objectForKey(unsafe { objc2_foundation::NSUnderlyingErrorKey })
    {
        out.push_str(&format!(" underlying: {underlying:?}"));
    }
    out
}
