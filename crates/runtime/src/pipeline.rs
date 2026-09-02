//! The forward path: guest-programmed pipelines. The wire format lives in
//! `eta_ir`; this module is the runtime domain that binds/instantiates/fires
//! it — [`program`] (bind/price/cache), [`instance`], [`channel`],
//! [`Pipeline`] (the in-flight fire FIFO) and [`fire`] (one fire's
//! prepare/submit/finalize).
//!
//! This module imports only `scheduler`/`store`/`engine` plus `eta_ir` and
//! external leaf crates, never `inferlet`/`server`.

pub mod channel;
pub mod fire;
pub mod instance;
/// The media door's runtime half: the run scan, its refusals, and the shape
/// the contract's `Step` wants.
pub mod media;
pub mod program;

use std::sync::{Arc, Mutex};

use fire::{PendingFireQueue, PendingFires, PipelineFailure};

/// A run-ahead submission pipeline. Owns the in-flight fire FIFO; submission
/// order rides the scheduler queue, completion order rides this FIFO.
///
/// Every pass binding a shared channel must submit on the same pipeline
/// (enforced by [`fire::wire_channels_to_pipeline`]), since fire t's
/// epilogue channel puts happen-before fire t+1's descriptor reads.
pub struct Pipeline {
    /// This pipeline's in-flight fires, oldest first — the FIFO above.
    pub fires: PendingFires,
    pub(crate) failure: PipelineFailure,
    pub(crate) scope: crate::store::PipelineScope,
    /// Per-lane frame sequence for Vesuvius frame submission (k > 1): each
    /// `forward.submit` frame on this pipeline takes the next number.
    pub(crate) frame_seq: std::sync::atomic::AtomicU64,
}

impl Pipeline {
    /// A fresh pipeline: an empty FIFO, no failure recorded yet.
    pub fn new() -> Self {
        let fires = Arc::new(PendingFireQueue::new());
        let weak_fires = Arc::downgrade(&fires);
        Self {
            fires,
            failure: Arc::new(Mutex::new(None)),
            scope: crate::store::PipelineScope::new(move || {
                weak_fires
                    .upgrade()
                    .is_none_or(|fires| fires.lock().unwrap().is_empty())
            }),
            frame_seq: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The next frame sequence number for this lane (frame mode).
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
        self.scope.close();
    }
}
