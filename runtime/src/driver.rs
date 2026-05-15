//! # Driver Subsystem
//!
//! One channel per driver — every request travels through the same
//! [`DriverChannel`]. Forward batches, page copies, and adapter loads
//! all share the transport; there is no separate "cold path."
//!
//! Requests and responses are the wire-canonical
//! [`pie_bridge::RequestPayload`] / [`pie_bridge::ResponsePayload`]
//! enums directly — no pie-internal mirror. The runtime pairs each
//! payload with a routing [`DriverId`] (the same value the wire
//! [`pie_bridge::Frame`] carries at its top level) to pick a channel.
//!
//! [`DriverChannel`] has two implementations:
//!   - [`InProcChannel`] — embedded drivers (cuda + portable + dummy
//!     linked into `pie-server`). Heap-backed queue + condvar wakeup;
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
mod shmem;

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
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
}

/// Driver-bound request. The payload is the wire-canonical
/// [`pie_bridge::RequestPayload`] (Forward / Copy / Adapter / Health),
/// with `driver_id` paired alongside for routing (the on-wire frame
/// carries it at the top level via [`pie_bridge::Frame.driver_id`]).
///
/// The Save / ZoInitialize / ZoUpdate adapter ops are wired end-to-end
/// but the per-driver dispatchers treat them as no-ops returning
/// `Status(0)` — no current driver implements them. The plumbing
/// exists so the runtime adapter API stays uniform and a future driver
/// can opt in by replacing the no-op with real logic.
#[derive(Debug)]
pub struct DriverRequest {
    pub driver_id: DriverId,
    pub payload: pie_bridge::RequestPayload,
}

/// Driver response. The payload is the wire-canonical
/// [`pie_bridge::ResponsePayload`] (Forward or Status); `aborted`
/// mirrors the same flag on [`pie_bridge::ResponseFrame`] (set by the
/// driver on transport-level failures).
#[derive(Debug)]
pub struct DriverResponse {
    pub aborted: bool,
    pub payload: pie_bridge::ResponsePayload,
}

#[async_trait]
pub trait DriverChannel: Send + Sync {
    /// Submit a request and resolve with the typed response.
    async fn submit(&self, req: DriverRequest) -> Result<DriverResponse>;

    /// Fire-and-forget submission. The driver still processes the
    /// request; the caller doesn't wait for the response. Errors at
    /// enqueue time (channel closed) are returned synchronously.
    fn notify(&self, req: DriverRequest) -> Result<()>;

    /// Mark the channel as aborted so any in-flight `submit` returns
    /// promptly with an error. Idempotent. Called by the supervisor's
    /// watchdog when it observes that the driver has exited.
    fn abort(&self);
}

pub use channel::{
    abort_all_driver_channels, fire_batch, get_spec, install_channel, install_spec, register_driver,
};
pub use inproc::{InProcChannel, InProcVTable};
pub use inproc_polling::InProcPollingChannel;

pub use ops::{
    copy_d2d, copy_d2h, copy_h2d, copy_h2h, load_adapter, save_adapter, zo_initialize_adapter,
    zo_update_adapter,
};
pub use shmem::ShmemChannel;
