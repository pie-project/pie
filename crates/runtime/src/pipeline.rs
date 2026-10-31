pub mod channel;
pub(crate) mod coalesce;
pub mod fire;
pub mod instance;
pub mod media;
pub mod program;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fire::{PendingFireQueue, PendingFires, PipelineFailure};

/// The pipelines open right now (`pipeline.live`).
static LIVE: AtomicUsize = AtomicUsize::new(0);

pub struct Pipeline {
    pub fires: PendingFires,
    pub(crate) failure: PipelineFailure,
    pub(crate) scope: crate::store::PipelineScope,
    pub(crate) frame_seq: std::sync::atomic::AtomicU64,
    /// The frames it posted and whether it closed, shared with every fire it
    /// hands an engine (`engine::ahead::Ahead`).
    pub(crate) posts: Arc<::engine::ahead::Posts>,
}

impl Pipeline {
    pub fn new() -> Self {
        let fires = Arc::new(PendingFireQueue::new());
        let weak_fires = Arc::downgrade(&fires);
        LIVE.fetch_add(1, Ordering::Relaxed);
        Self {
            fires,
            failure: Arc::new(Mutex::new(None)),
            scope: crate::store::PipelineScope::new(move || {
                weak_fires
                    .upgrade()
                    .is_none_or(|fires| fires.lock().unwrap().is_empty())
            }),
            frame_seq: std::sync::atomic::AtomicU64::new(0),
            posts: Arc::default(),
        }
    }

    #[must_use]
    pub fn live() -> usize {
        LIVE.load(Ordering::Relaxed)
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
        LIVE.fetch_sub(1, Ordering::Relaxed);
        self.scope.close();
    }
}
