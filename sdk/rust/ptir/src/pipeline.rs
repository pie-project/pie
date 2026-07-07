//! `Pipeline` — submits passes run-ahead (overview §3). Ordering is carried by
//! the channels' bits, not by host code.
//!
//! For P1 `submit` traces the pass once (memoized), lints + binds it; the async
//! run-ahead enqueue + host harvest land with the channel store (P3).

use alloc::rc::Rc;

use crate::error::TraceErrors;
use crate::forward::{ForwardPass, TracedForward};

/// A run-ahead submission pipeline (overview §3).
pub struct Pipeline {
    last: core::cell::RefCell<Option<Rc<TracedForward>>>,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline { last: core::cell::RefCell::new(None) }
    }

    /// `submit(&fwd)` — enqueue a pass. Traces the pass once (trace-once
    /// memoization), lints + binds it; a malformed pass panics with the trace
    /// error. Use [`Pipeline::try_submit`] / [`ForwardPass::trace`] for the
    /// fallible form.
    pub fn submit(&self, fwd: &ForwardPass<'_>) {
        match fwd.trace() {
            Ok(t) => *self.last.borrow_mut() = Some(t),
            Err(e) => panic!("submit of a malformed pass:\n{e}"),
        }
    }

    /// The fallible submit: trace + lint + bind, returning the trace or its errors.
    pub fn try_submit(&self, fwd: &ForwardPass<'_>) -> Result<Rc<TracedForward>, TraceErrors> {
        let t = fwd.trace()?;
        *self.last.borrow_mut() = Some(t.clone());
        Ok(t)
    }

    /// `close()` — signal no further submissions (implied by drop; overview §1).
    pub fn close(&self) {
        *self.last.borrow_mut() = None;
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}
