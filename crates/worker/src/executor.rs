//! The executor role — a named hole where the transport was.
//!
//! # `palo B-remote`: this module WAS the envelope's server half
//!
//! What stood here was 3,371 lines of tarpc service: `ExecutorRpc` and its
//! `ExecutorRequest`/`ExecutorResponse` dispatch, a per-client hello
//! handshake with a `REMOTE_WIRE_VERSION` check and a scratch-grant lease, a
//! blob-fetch budget, an inline KV push that `cudaMemcpy`'d pages out of the
//! local pool, a NIXL registration path, and a FIFO driver actor that
//! translated `RemoteLaunch` back into a `FrameSubmission`. Every message
//! type it named lived in `driver_api::remote`, and the palo contract rewrite
//! deleted that module whole (design §7, decision 19):
//!
//! > Remote is a property, not an encoding: every noun serde, trait
//! > object-safe; wire versioning is the transport's concern, not the
//! > contract's.
//!
//! Redesigning the envelope is not this wave. What is here is the surface the
//! rest of the worker names — the role's boot path, its stats, and the dial —
//! with every verb refusing by name. A prefill worker that boots into this
//! role is told, at boot, that it cannot serve one.
//!
//! # What the future envelope has to carry
//!
//! * **the hello** — a wire version (on the transport, not on the contract),
//!   the peer's [`ModelIdentity`], its
//!   [`Capabilities`](driver_api::Capabilities), its
//!   [`KvLayout`](driver_api::KvLayout), and the scratch grant (base page +
//!   count) the caller may address inside the peer's pool.
//! * **the verbs** — `register_program`, `register_channel`, `bind_instance`,
//!   `fire`, `copy_kv`, `encode`, `close_*`. All seven take serde nouns now,
//!   so the message set is the trait; what the envelope adds is id mapping
//!   (the peer mints program/channel/instance ids), a frame-size ceiling, and
//!   an asynchronous [`FireTicket`](driver_api::FireTicket) path.
//! * **the transfer** — inline bytes or an RDMA handle, and a way to say
//!   which was used.
//!
//! `crate::link::partner` carries the same marker on the client half, and
//! `engine::pipeline::offload` on the admission half.

use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::translate::ModelDrivers;

/// Which half of a model a load carries.
///
/// **WORKER VOCABULARY NOW.** It was `driver_api::ModelComponent`, and the
/// contract dropped it because it says WHICH GRAPH to load by enum, and that
/// is now which `Plan` you hand over (`driver-api::load`'s header: "the
/// encoder is a traced plan like any other"). What it still means here is
/// which of a deployment's two loads a worker is: a decode worker's full
/// model, or an encode partner's encoder.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelComponent {
    /// The whole model.
    Full,
    /// The text half only.
    Text,
    /// The multimodal encoder only.
    Encode,
}

/// The token two workers compare before they trade KV pages.
///
/// Worker vocabulary for the same reason [`ModelComponent`] is: it is a hash
/// of what the two DRIVERS loaded plus which component each is, and neither
/// half of that is a statement a driver makes about itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// blake3 over the model name, the checkpoint digest, and the driver's
    /// own answers about what it loaded.
    pub hash: [u8; 32],
    /// Which half.
    pub component: ModelComponent,
}

/// What a controller reads off an executor to schedule against it.
#[derive(Default)]
pub(crate) struct ExecutorStats {
    inflight: AtomicU32,
    leased_pages: AtomicU32,
}

impl ExecutorStats {
    pub(crate) fn inflight(&self) -> u32 {
        self.inflight.load(Ordering::Relaxed)
    }

    pub(crate) fn kv_pressure_bucket(&self, total_pages: u32) -> u8 {
        if total_pages == 0 {
            return 0;
        }
        let used = self.leased_pages.load(Ordering::Relaxed) as u64;
        ((used.saturating_mul(u8::MAX as u64) / total_pages as u64).min(u8::MAX as u64)) as u8
    }
}

/// The executor role's server.
///
/// Every field the running server had — the accept task, the actor task, the
/// core handle — went with the service. What is left is what its callers read
/// while it is up, so that the shape of `boot_executor` does not have to
/// change when the envelope returns.
pub(crate) struct ExecutorServer {
    endpoint: String,
    stats: Arc<ExecutorStats>,
    total_pages: u32,
}

impl ExecutorServer {
    /// Serve the executor role at `addr`.
    ///
    /// # Errors
    ///
    /// Always. See the module header: there is no transport, and a server
    /// that bound a port and answered nothing would leave a controller
    /// dispatching prefills into silence.
    pub(crate) async fn bind_with_transfer(
        addr: &str,
        drivers: ModelDrivers,
        model: ModelIdentity,
        max_clients: usize,
        transfer: crate::config::OffloadTransfer,
    ) -> Result<Self> {
        let _ = (drivers, model, max_clients, transfer);
        Err(anyhow!(
            "this build cannot serve the executor role at {addr}: the remote \
             envelope `driver_api::remote` carried was deleted by the palo \
             contract rewrite and its successor is palo B-remote. Boot this \
             worker in the standalone role."
        ))
    }

    /// Where peers would dial it.
    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Its live counters.
    pub(crate) fn stats(&self) -> Arc<ExecutorStats> {
        Arc::clone(&self.stats)
    }

    /// How many KV pages its pool holds.
    pub(crate) fn total_pages(&self) -> u32 {
        self.total_pages
    }

    /// Stop serving.
    pub(crate) async fn shutdown(self) {}
}

/// Dial a peer executor, answering the client and the local address the peer
/// sees this worker at.
///
/// # Errors
///
/// Always, for the reason [`ExecutorServer::bind_with_transfer`] gives. The
/// local-IP half is what the peer's blob fetches would have been routed to.
pub(crate) async fn connect_with_local_ip(addr: &str) -> Result<((), IpAddr)> {
    Err(anyhow!(
        "cannot dial executor {addr}: the remote envelope is palo B-remote"
    ))
}
