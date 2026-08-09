//! What one step cost, split into the two halves that have different causes.
//!
//! A step that takes too long is either the host building command buffers or
//! the GPU running them, and those are fixed by different work. One number
//! for the pair cannot say which, so [`Timing`] carries both -- which is what
//! the C++ header means by "the manager wants BOTH reported separately".
//!
//! # Three clocks, not two
//!
//! `encode` and `gpu_exec` are both read from the host's clock, either side
//! of the commit. `gpu` is the GPU's own report of the same submission,
//! delivered asynchronously through commit feedback, and it is smaller: it
//! starts when the work starts on the device rather than when the host handed
//! it over, and it ends when the work ends rather than when the host woke up
//! and noticed. The difference between `gpu_exec` and `gpu` is therefore the
//! part of a step that is neither encoding nor computing -- queue latency and
//! host wake-up -- and [`Timing::overhead`] is that subtraction.
//!
//! # What this type deliberately does not carry
//!
//! The C++ `StepTiming` also carries `completed`, `timed_out`, `gpu_error`
//! and `gpu_error_text`. Those are not timings, they are the step's outcome,
//! and a struct that reports both invites the caller that reads the numbers
//! and forgets the flags -- which is how a step that never ran came to be
//! reported as one that took zero milliseconds. Here the outcome is the
//! `Result` the runner returns, and a `Timing` exists only for a step that
//! completed.
//!
//! # `None` is not zero
//!
//! `gpu` is an [`Option`], because commit feedback is delivered on Metal's
//! own schedule and may not have landed by the time the wait returns. The C++
//! writes 0.0 in that case, which is indistinguishable from a step the GPU
//! reported as taking no time at all. A caller that needs the calibrated
//! number rather than the host-observed one can wait for it explicitly with
//! [`super::Feedbacks::await_step`] and [`Timing::step`].

use std::time::Duration;

/// What one step cost.
///
/// Returned by every runner on [`super::Stepper`]. Every field is a real
/// measurement of a step that finished; see the module docs for why nothing
/// here reports failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Timing {
    /// Building the command buffers, on the host.
    ///
    /// From the allocator reset to the last `endCommandBuffer`. This is the
    /// cost of the encoder itself and of whatever the caller's closure does,
    /// and it is pure host time -- the GPU has not been told about any of it
    /// yet.
    pub encode: Duration,

    /// Commit to completion, as the host sees it.
    ///
    /// Includes the time the submission spent queued behind other work, the
    /// execution itself, and the host's own wake-up after the event was
    /// signalled. It is the number that matters for throughput, because it is
    /// the time the caller actually waited.
    pub gpu_exec: Duration,

    /// Execution, as the GPU reports it.
    ///
    /// `GPUEndTime - GPUStartTime` from the commit feedback handler, so it
    /// excludes queueing and host wake-up. `None` means the feedback had not
    /// arrived when the step returned, which is common and is not an error --
    /// see the module docs on why that is not folded into a zero.
    pub gpu: Option<Duration>,

    /// The timeline value this step signalled.
    ///
    /// Carried so that a caller who wants [`Self::gpu`] and got `None` can
    /// ask for it again later: the feedback slot is keyed by exactly this
    /// number.
    pub step: u64,
}

impl Timing {
    /// Everything the caller waited for.
    ///
    /// Encode plus host-observed execution, matching the C++ `total_ms`. Not
    /// `gpu`, even when it is known: the GPU's number excludes time the
    /// caller nonetheless spent.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.encode + self.gpu_exec
    }

    /// The part of the wait that was neither encoding nor computing.
    ///
    /// Queue latency and host wake-up: what the host observed minus what the
    /// GPU says it spent. `None` when the GPU has not reported, and saturating
    /// rather than signed because the two clocks are calibrated separately and
    /// a tiny negative difference is a calibration artefact rather than a
    /// measurement.
    #[must_use]
    pub fn overhead(&self) -> Option<Duration> {
        Some(self.gpu_exec.saturating_sub(self.gpu?))
    }

    /// Fold `later` into this timing, for a run made of several submissions.
    ///
    /// The two host halves add, because the caller waited for both. `gpu`
    /// does NOT add: the feedback slot holds one report, so what is available
    /// after the last segment describes the last segment. Taking the newest
    /// rather than summing is the honest answer -- a sum of one segment's
    /// GPU time and another's absence is neither.
    pub(crate) fn extend(&mut self, later: Self) {
        self.encode += later.encode;
        self.gpu_exec += later.gpu_exec;
        self.gpu = later.gpu;
        self.step = later.step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    #[test]
    fn total_is_what_the_caller_waited_for_and_not_what_the_gpu_reported() {
        let timing = Timing {
            encode: ms(2),
            gpu_exec: ms(10),
            gpu: Some(ms(7)),
            step: 1,
        };
        assert_eq!(timing.total(), ms(12));
    }

    #[test]
    fn overhead_is_the_wait_the_gpu_does_not_account_for() {
        let timing = Timing {
            encode: ms(2),
            gpu_exec: ms(10),
            gpu: Some(ms(7)),
            step: 1,
        };
        assert_eq!(timing.overhead(), Some(ms(3)));
    }

    #[test]
    fn overhead_is_unknown_rather_than_the_whole_wait_when_the_gpu_is_silent() {
        let timing = Timing {
            encode: ms(2),
            gpu_exec: ms(10),
            gpu: None,
            step: 1,
        };
        assert_eq!(
            timing.overhead(),
            None,
            "an absent GPU report must not read as a step that was all overhead"
        );
    }

    #[test]
    fn a_gpu_report_longer_than_the_host_wait_is_zero_overhead_not_a_wrap() {
        // The two clocks are calibrated separately, so this happens. The
        // subtraction must not wrap into an enormous positive duration.
        let timing = Timing {
            encode: ms(1),
            gpu_exec: ms(5),
            gpu: Some(ms(6)),
            step: 1,
        };
        assert_eq!(timing.overhead(), Some(Duration::ZERO));
    }

    #[test]
    fn extending_adds_the_host_halves_and_takes_the_newest_gpu_report() {
        let mut first = Timing {
            encode: ms(2),
            gpu_exec: ms(10),
            gpu: Some(ms(7)),
            step: 1,
        };
        first.extend(Timing {
            encode: ms(3),
            gpu_exec: ms(20),
            gpu: None,
            step: 2,
        });
        assert_eq!(first.encode, ms(5));
        assert_eq!(first.gpu_exec, ms(30));
        assert_eq!(
            first.gpu, None,
            "the slot holds one report, so a summed GPU time would be fiction"
        );
        assert_eq!(first.step, 2, "the timing names the last submission");
    }
}
