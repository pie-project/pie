//! THE forward path: guest-programmed pipelines. The wire format (the IR
//! itself) lives in the `pie_ptir` crate; this module is the runtime domain
//! that binds/instantiates/fires it.
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
//! Layering: this module imports only `scheduler`/`store`/`driver` (strictly
//! below it) plus the `pie_ptir` IR crate and other external leaf crates
//! (`wasmtime::component::ResourceTable`, `uuid`) — never `inferlet`/
//! `server`. The WIT resource *types* (`Channel`, `ForwardPass`,
//! `Pipeline`) live here because they hold domain state (cells, FIFO,
//! working-set reps, cursor, failure, devgeo); `inferlet::host` (L4) owns
//! only the thin `Host*` trait impls that push/get/delete them from the WASM
//! component resource table, via the [`fire::FireContext`] trait it
//! implements for `ProcessCtx` — see `fire`'s module doc.

pub mod channel;
pub mod fire;
pub mod instance;
pub mod offload;
pub mod program;

use std::sync::{Arc, Mutex};

use fire::{PendingFireQueue, PendingFires, PipelineFailure};

/// A run-ahead submission pipeline (overview §3): the ORDERING domain (W3.1)
/// — the WIT `pie:inferlet/pipeline.pipeline` resource. Owns the in-flight
/// fire FIFO — fires submitted here are issued in submission order, so fire
/// t's epilogue channel puts happen-before fire t+1's descriptor reads,
/// EXTENDED ACROSS PASSES (draft→verify chaining). `take`/`read` await this
/// FIFO via each channel's recorded pipeline. Submission order rides the
/// scheduler queue; completion order rides this FIFO.
///
/// **FIFO INVARIANT (B3, mandatory).** Fires of one pipeline keep submission
/// order through the scheduler onto one stream, and every pass binding a
/// shared channel MUST submit on the SAME pipeline (enforced by
/// [`fire::wire_channels_to_pipeline`]). This is the ENTIRE correctness
/// argument for run-ahead + multi-pass chaining: because all interacting
/// fires funnel onto one ordered FIFO, fire t's epilogue puts happen-before
/// fire t+1's descriptor reads. `push_back` at submit + `pop_front` at
/// finalize preserve that order; the same-pipeline check makes it an explicit
/// invariant, not an accident.
///
/// Domain state (not WIT glue), so it lives here rather than in
/// `inferlet::host::pipeline`, which only holds the `Host`/`HostPipeline`
/// impls that push/get/delete it from the WASM component resource table.
pub struct Pipeline {
    pub fires: PendingFires,
    pub(crate) failure: PipelineFailure,
    pub(crate) scope: crate::store::PipelineScope,
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
        }
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
