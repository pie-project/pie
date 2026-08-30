//! THE forward path: guest-programmed pipelines. The wire format (the IR
//! itself) lives in `eta_ir`; this module is the runtime domain that
//! binds/instantiates/fires it.
//!
//! - [`program`]: container bytes -> bind -> price -> cache; absorbs
//!   `model_profile()`.
//! - [`instance`]: program + seeds -> [`instance::Instance`], plus
//!   [`instance::ForwardPass`] (the WIT `forward-pass` resource's domain
//!   state).
//! - [`channel`]: [`channel::ChannelCell`] host endpoint, SPSC roles, plus
//!   [`channel::Channel`] (the WIT `channel` resource's domain state).
//! - [`Pipeline`]: the ordering-domain resource (the WIT `pipeline` resource's
//!   domain state) — owns the in-flight fire FIFO.
//! - [`fire`]: one fire: prepare -> run-ahead submit -> finalize/poison, plus
//!   `geometry`/`kv`/`rs`/`lease`.
//!
//! Layering: this module imports only `scheduler`/`store`/`engine` plus the
//! `eta_ir` IR crate and external leaf crates — never `inferlet`/`server`.
//! The WIT resource *types* live here because they hold domain state;
//! `inferlet::host` owns only the thin `Host*` trait impls that push/get/
//! delete them from the WASM component resource table.

pub mod channel;
pub mod fire;
pub mod instance;
/// The media door's runtime half: the run scan, its refusals, and the shape
/// the contract's `Step` wants (`.wiki/alto/media-door.md` §3/§6).
pub mod media;
pub mod offload;
pub mod program;

use std::sync::{Arc, Mutex};

use fire::{PendingFireQueue, PendingFires, PipelineFailure};

/// A run-ahead submission pipeline (overview §3): the ORDERING domain (W3.1,
/// WIT `pie:inferlet/pipeline.pipeline`). Owns the in-flight fire FIFO;
/// submission order rides the scheduler queue, completion order rides this
/// FIFO.
///
/// **FIFO INVARIANT (B3, mandatory).** Every pass binding a shared channel
/// MUST submit on the SAME pipeline (enforced by
/// [`fire::wire_channels_to_pipeline`]) — the entire correctness argument for
/// run-ahead + multi-pass chaining, since fire t's epilogue channel puts
/// happen-before fire t+1's descriptor reads. Domain state, not WIT glue:
/// `inferlet::host::pipeline` only holds the thin `Host`/`HostPipeline`
/// impls that push/get/delete it from the WASM component resource table.
pub struct Pipeline {
    /// This pipeline's in-flight fires, oldest first — the FIFO above.
    pub fires: PendingFires,
    pub(crate) failure: PipelineFailure,
    pub(crate) scope: crate::store::PipelineScope,
    /// Per-lane frame sequence for Vesuvius frame submission (k > 1): each
    /// `forward.submit` frame on this pipeline takes the next number.
    pub(crate) frame_seq: std::sync::atomic::AtomicU64,
    // This branch does not carry upstream's `PIE_DEFER_ALLOC` handle: its
    // frame-grant queuing is superseded by this branch's kv-contention
    // rewrite of `prepare_submission`. Default behaviour is unaffected
    // (upstream's path is opt-in and off unless asked for).
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
