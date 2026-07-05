//! # Driver Subsystem
//!
//! One channel per driver — every request travels through the same
//! [`DriverChannel`]. Forward batches, page copies, and adapter loads
//! all share the transport; there is no separate "cold path."
//!
//! Requests and responses are the wire-canonical
//! [`pie_driver_abi::RequestPayload`] / [`pie_driver_abi::ResponsePayload`]
//! enums directly — no pie-internal mirror. The runtime pairs each
//! payload with a routing [`DriverId`] (the same value the wire
//! [`pie_driver_abi::Frame`] carries at its top level) to pick a channel.
//!
//! [`DriverChannel`] has two implementations:
//!   - [`InProcChannel`] — embedded drivers (cuda + metal + dummy
//!     linked into `pie-worker`). Heap-backed queue + condvar wakeup;
//!     the FFI hands the C++ driver a typed view of the request data.
//!   - [`InProcPollingChannel`] — low-latency embedded-driver channel
//!     with the same FFI surface but fixed slots and polling waits
//!     instead of the heap-backed pending map/inbox.
//!   - [`ShmemChannel`] — subprocess drivers (Python `dev`, `vllm`,
//!     `sglang`). POSIX-shmem ring carrying rkyv-encoded frames.

mod channel;
mod inproc;
mod inproc_polling;
mod ops;
mod prefetch;
mod shmem;

/// X0 — the tensor-waker substrate (Runtime–Driver Boundary B9–B12): the
/// Rust-owned waker slot table + `pie_wake` FFI the direct-call transport
/// (X1–X4) parks on. Lives in the leaf crate `pie-waker` (so the
/// register/commit race is loom-model-checked without the runtime's
/// dependency graph); re-exported here behind the `ptir` flag.
#[cfg(feature = "ptir")]
pub use pie_waker as waker;

use anyhow::Result;
use async_trait::async_trait;

/// Stable per-driver identifier. Indexes into the runtime's driver registry
/// (see `pie::driver`). Used by request routing to pick which `DriverChannel`
/// services the forward. Pie-internal — the wire schema carries this as the
/// `driver_id: uint32` field on `Frame` / `ResponseFrame`.
pub type DriverId = usize;

/// Static driver configuration (capacity limits). Populated at
/// [`install_channel`] time from the caps handshake.
#[derive(Debug, Clone)]
pub struct DriverSpec {
    pub num_kv_pages: usize,
    pub limits: SchedulerLimits,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerLimits {
    pub max_forward_requests: usize,
    pub max_forward_tokens: usize,
    pub max_page_refs: usize,
    pub max_logit_rows: usize,
    pub max_prob_rows: usize,
    pub max_sampler_rows: usize,
    pub max_custom_mask_bytes: usize,
    pub max_logprob_labels: usize,
}

impl DriverSpec {
    /// Scheduler-facing limits published by the driver's capacity handshake.
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

/// Driver-bound request. The payload is the wire-canonical
/// [`pie_driver_abi::RequestPayload`] (Forward / Copy / Adapter / Health),
/// with `driver_id` paired alongside for routing (the on-wire frame
/// carries it at the top level via [`pie_driver_abi::Frame.driver_id`]).
///
/// The Save / ZoInitialize / ZoUpdate adapter ops are wired end-to-end
/// but the per-driver dispatchers treat them as no-ops returning
/// `Status(0)` — no current driver implements them. The plumbing
/// exists so the runtime adapter API stays uniform and a future driver
/// can opt in by replacing the no-op with real logic.
#[derive(Debug)]
pub struct DriverRequest {
    pub driver_id: DriverId,
    pub payload: pie_driver_abi::RequestPayload,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_limits_use_forward_limits() {
        let spec = DriverSpec {
            num_kv_pages: 1024,
            limits: SchedulerLimits {
                max_forward_requests: 64,
                max_forward_tokens: 2048,
                max_page_refs: 262144,
                max_logit_rows: 2048,
                max_prob_rows: 2048,
                max_sampler_rows: 2048,
                max_custom_mask_bytes: 8 * 1024 * 1024,
                max_logprob_labels: 512,
            },
        };

        let limits = spec.scheduler_limits();
        assert_eq!(limits.max_forward_requests, 64);
        assert_eq!(limits.max_forward_tokens, 2048);
        assert_eq!(limits.max_page_refs, 262144);
        assert_eq!(limits.max_logit_rows, 2048);
        assert_eq!(limits.max_prob_rows, 2048);
        assert_eq!(limits.max_sampler_rows, 2048);
        assert_eq!(limits.max_logprob_labels, 512);
    }
}

/// Driver response. The payload is the wire-canonical
/// [`pie_driver_abi::ResponsePayload`] (Forward or Status); `aborted`
/// mirrors the same flag on [`pie_driver_abi::ResponseFrame`] (set by the
/// driver on transport-level failures).
#[derive(Debug)]
pub struct DriverResponse {
    pub aborted: bool,
    pub payload: pie_driver_abi::ResponsePayload,
}

/// A response whose submission has already been ordered (the request is
/// enqueued) but whose blocking wait is deferred. Calling the boxed closure
/// blocks for the driver response. The run-ahead scheduler uses this to fix
/// fire order on its own thread (the enqueue happens at `submit_deferred` call
/// time, in fire order) and then await the GPU off-thread, so building and
/// enqueuing the next batch overlaps the in-flight forward.
pub type DeferredResponse = Box<dyn FnOnce() -> Result<DriverResponse> + Send>;

#[async_trait]
pub trait DriverChannel: Send + Sync {
    /// Submit a request and resolve with the typed response.
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse>;

    /// Synchronous submit. Used by the hot fire path, which runs on a
    /// dedicated OS thread (the batch scheduler loop) — calling the
    /// async `submit` from there would require either `block_on` or
    /// dragging tokio's runtime onto the scheduler thread. Every
    /// production impl already has a sync inner (`submit_sync_for_state`,
    /// `submit_blocking`, `roundtrip_sync`); this is the trait-level
    /// entry point.
    fn submit_sync(&self, req: DriverRequest) -> Result<DriverResponse>;

    /// Enqueue `req` now (fixing its submission order at the call site) and
    /// return a closure that blocks for the response. Lets the run-ahead
    /// scheduler enqueue fires in-order on its own thread, then await each
    /// off-thread so the GPU wait overlaps building the next batch. The default
    /// is fully synchronous — it submits and waits inline, returning an
    /// already-resolved closure (correct and order-preserving, just no
    /// overlap). [`InProcChannel`] (the embedded-driver hot path) overrides
    /// this to defer only the response wait.
    fn submit_deferred(&self, req: DriverRequest) -> Result<DeferredResponse> {
        let resp = self.submit_sync(req)?;
        Ok(Box::new(move || Ok(resp)))
    }

    /// Fire-and-forget submission. The driver still processes the
    /// request; the caller doesn't wait for the response. Errors at
    /// enqueue time (channel closed) are returned synchronously.
    fn notify(&self, req: DriverRequest) -> Result<()>;

    /// Mark the channel as aborted so any in-flight `submit` returns
    /// promptly with an error. Idempotent. Called by the supervisor's
    /// watchdog when it observes that the driver has exited.
    fn abort(&self);

    /// Fire-and-forget JIT **prefetch** (the #11 prefetch seam): warm the
    /// driver's compile cache for a sampling program so the later real fire
    /// finds it `Ready` (the NVRTC compile overlaps the in-flight run-ahead
    /// steps, off the TTFT path). The compile is keyed on
    /// `program_identity_hash(bytecode, manifest)` — the SAME key as the #10
    /// distinct-count / #11 compile-cache / M-batch grouping — so it dedups
    /// against the in-flight compile pool (idempotent; duplicate or
    /// already-compiled programs collapse). Never blocks, never reports errors.
    ///
    /// Default **no-op**: drivers without a JIT sampling backend, and the
    /// out-of-proc/IPC path until its additive `DriverRequest::Prefetch` oneway
    /// fast-follow lands. The embedded [`InProcChannel`] overrides this to drive
    /// the C++ `IProgramBackend::prefetch_compile` over the in-proc FFI.
    fn prefetch_compile(&self, _bytecode: &[u8], _manifest: &[pie_sampling_ir::Binding]) {}
}

pub use channel::{
    abort_all_driver_channels, fire_batch, fire_batch_deferred, fire_batch_sync, get_spec,
    install_channel, install_spec, prefetch_compile, register_driver, FireHandle,
};
pub use inproc::{InProcChannel, InProcVTable};
pub use inproc_polling::InProcPollingChannel;

pub use ops::{
    copy_d2d, copy_d2h, copy_h2d, copy_h2h, copy_rs_d2d, generate_audio, load_adapter,
    save_adapter, zo_initialize_adapter, zo_update_adapter,
};
pub use shmem::ShmemChannel;
