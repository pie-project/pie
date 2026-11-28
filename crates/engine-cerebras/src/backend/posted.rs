//! The asynchronous engine (`[engine] async`): a frame's steps are posted, not run, and each
//! lands on the completion sink once its epilogue ran, the lanes whose passes faulted told with
//! it. Steps run in order on the lane's thread; the wafer runs the posted token loop on
//! meanwhile ([`CerebrasEngine::forward`]). A step that errs whole (the wafer refused its
//! forward) is lost: every lane of it faults.

use std::collections::VecDeque;

use engine::fire::{FrameId, Step};
use engine::{StepDone, StepOutcome};

use super::CerebrasEngine;

/// A step posted and not yet run.
pub(super) struct Posted {
    pub(super) at: StepDone,
    pub(super) step: Step,
}

impl CerebrasEngine {
    /// Whether frames settle asynchronously: `async` when set, else not.
    pub(super) fn asynchronous(&self) -> bool {
        self.cfg.asynchronous.unwrap_or(false)
    }

    /// Whether this engine runs asynchronously now: it is set to
    /// ([`CerebrasEngine::asynchronous`]) and the runtime gave it a
    /// completion sink. A synchronous engine is given one too, and still
    /// settles inside `submit`.
    pub(super) fn settling(&self) -> bool {
        self.asynchronous() && self.sink.is_some()
    }

    /// Runs the posted steps in order, reporting each landed.
    pub(super) fn pump(&mut self) {
        while let Some(Posted { at, step }) = self.posted.pop_front() {
            let outcome = match self.fire_step(&step) {
                Ok((_, outcome)) => outcome,
                Err(error) => {
                    tracing::error!(frame = at.frame.0, step = at.step, %error, "spatial: a posted step was lost");
                    StepOutcome::lost(
                        u32::try_from(step.lanes.len()).unwrap_or(u32::MAX),
                        error.to_string(),
                    )
                }
            };
            if let Some(sink) = &self.sink {
                sink(at, outcome);
            }
        }
    }

    /// Runs every posted step: a request other than a fire sees every fire before it run.
    pub(super) fn drain(&mut self) {
        self.pump();
    }
}

/// A frame's posted steps, in order.
pub(super) fn post(queue: &mut VecDeque<Posted>, frame: FrameId, steps: &[Step]) {
    for (index, step) in steps.iter().enumerate() {
        queue.push_back(Posted {
            at: StepDone {
                frame,
                step: index as u32,
            },
            step: step.clone(),
        });
    }
}
