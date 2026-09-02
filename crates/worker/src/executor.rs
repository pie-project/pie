//! The executor role — a named hole where the transport goes. What is here is
//! the surface the rest of the worker names: the role's boot path, its stats,
//! and the dial, with every verb refusing by name, so a prefill worker that
//! boots into this role is told at boot that it cannot serve one.

use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::backend::ModelEngines;

/// Which half of a model a load carries: a decode worker's full model, or an
/// encode partner's encoder.
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

/// The token two workers compare before they trade KV pages: a hash of what
/// the two engines loaded plus which component each is.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// blake3 over the model name, the checkpoint digest, and the engine's
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

/// The executor role's server. What's left is what its callers read while it
/// is up, so `boot_executor`'s shape need not change when the envelope returns.
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
    /// Always: there is no transport.
    pub(crate) async fn bind_with_transfer(
        addr: &str,
        engines: ModelEngines,
        model: ModelIdentity,
        max_clients: usize,
        transfer: crate::config::OffloadTransfer,
    ) -> Result<Self> {
        let _ = (engines, model, max_clients, transfer);
        Err(anyhow!(
            "this build cannot serve the executor role at {addr}: remote \
             executors are not supported in this release. Boot this worker \
             in the standalone role."
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
/// Always: no transport. The local-IP half is what the peer's blob fetches
/// would have been routed to.
pub(crate) async fn connect_with_local_ip(addr: &str) -> Result<((), IpAddr)> {
    Err(anyhow!(
        "cannot dial executor {addr}: remote executors are not supported in this release"
    ))
}
