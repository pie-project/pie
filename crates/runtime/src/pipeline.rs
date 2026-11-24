pub mod channel;
pub(crate) mod coalesce;
pub mod fire;
pub mod instance;
pub mod media;
pub mod program;
pub mod standard;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fire::{PendingFireQueue, PendingFires, PipelineFailure};

/// The pipelines a batch may hold right now: open, neither parked nor
/// closed. What `pipeline.speculation-advised` reads.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Whether no process has work outstanding: no pipeline open (each parks
/// or closes once its process has nothing in flight). What the engines'
/// work behind serving waits for.
#[must_use]
pub fn none_active() -> bool {
    ACTIVE.load(Ordering::Relaxed) == 0
}

pub struct Pipeline {
    pub fires: PendingFires,
    pub(crate) failure: PipelineFailure,
    pub(crate) scope: crate::store::PipelineScope,
    pub(crate) frame_seq: std::sync::atomic::AtomicU64,
    /// One past the sequence of the frame a cut pass truncated to go in
    /// frames of its own (`pipeline::fire::submit_pass_stamped`): the
    /// frame's later slots follow in frames of their own too. 0: none.
    reframed: std::sync::atomic::AtomicU64,
    /// Counted in [`ACTIVE`]: from its making, and again from each submit,
    /// until it parks or closes.
    active: AtomicBool,
    /// Where each of the last fires submitted on it began writing each
    /// working set, in submission order: what `retract` takes back. No more
    /// than the run-ahead window are kept, the most a program runs ahead
    /// of what it reads.
    pub(crate) written: Mutex<std::collections::VecDeque<fire::WrittenFire>>,
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
                weak_fires.upgrade().is_none_or(|fires| fires.drained())
            }),
            failure,
            frame_seq: std::sync::atomic::AtomicU64::new(0),
            reframed: std::sync::atomic::AtomicU64::new(0),
            active: AtomicBool::new(true),
            written: Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Whether it is one a batch may hold: it submits again after a park;
    /// a park or a close takes it out.
    pub(crate) fn set_active(&self, active: bool) {
        if self.active.swap(active, Ordering::Relaxed) != active {
            if active {
                ACTIVE.fetch_add(1, Ordering::Relaxed);
            } else if ACTIVE.fetch_sub(1, Ordering::Relaxed) == 1 {
                // The last one parked or closed: nothing is live, and what
                // the engines measure behind serving may run now.
                crate::scheduler::lane::turn_behind();
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

    /// Frame `seq` was truncated before one of its slots, a cut pass, so
    /// that pass and the slots after it go in frames of their own.
    pub(crate) fn mark_reframed(&self, seq: u64) {
        self.reframed
            .store(seq + 1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether frame `seq` was truncated so ([`Self::mark_reframed`]).
    pub(crate) fn reframed(&self, seq: u64) -> bool {
        self.reframed.load(std::sync::atomic::Ordering::Relaxed) == seq + 1
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
