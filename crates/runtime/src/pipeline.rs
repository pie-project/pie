pub mod channel;
pub(crate) mod coalesce;
pub mod fire;
pub mod instance;
pub mod media;
pub mod program;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fire::{PendingFireQueue, PendingFires, PipelineFailure};

/// The pipelines a batch may hold right now: open, neither parked nor
/// closed. What `pipeline.speculation-advised` reads.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

pub struct Pipeline {
    pub fires: PendingFires,
    pub(crate) failure: PipelineFailure,
    pub(crate) scope: crate::store::PipelineScope,
    pub(crate) frame_seq: std::sync::atomic::AtomicU64,
    /// Counted in [`ACTIVE`]: from its making, and again from each submit,
    /// until it parks or closes.
    active: AtomicBool,
}

impl Pipeline {
    pub fn new() -> Self {
        let fires = Arc::new(PendingFireQueue::new());
        let weak_fires = Arc::downgrade(&fires);
        let failure: PipelineFailure = Arc::new(Mutex::new(None));
        ACTIVE.fetch_add(1, Ordering::Relaxed);
        Self {
            fires,
            scope: crate::store::PipelineScope::failing_as(failure.clone(), move || {
                weak_fires
                    .upgrade()
                    .is_none_or(|fires| fires.lock().unwrap().is_empty())
            }),
            failure,
            frame_seq: std::sync::atomic::AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Whether it is one a batch may hold: it submits again after a park;
    /// a park or a close takes it out.
    pub(crate) fn set_active(&self, active: bool) {
        if self.active.swap(active, Ordering::Relaxed) != active {
            if active {
                ACTIVE.fetch_add(1, Ordering::Relaxed);
            } else {
                ACTIVE.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Whether drafting pays here: no other pipeline is active beside it, so
    /// its fires batch with nobody's.
    #[must_use]
    pub fn speculation_advised(&self) -> bool {
        let own = usize::from(self.active.load(Ordering::Relaxed));
        ACTIVE.load(Ordering::Relaxed).saturating_sub(own) == 0
    }

    pub(crate) fn next_frame_seq(&self) -> u64 {
        self.frame_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.set_active(false);
        self.scope.close();
    }
}
