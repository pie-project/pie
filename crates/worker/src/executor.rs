#[cfg(feature = "net")]
use std::net::IpAddr;
#[cfg(feature = "net")]
use std::sync::Arc;
#[cfg(feature = "net")]
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "net")]
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[cfg(feature = "net")]
use crate::backend::ModelEngines;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelComponent {
    Full,
    Text,
    Encode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub hash: [u8; 32],
    pub component: ModelComponent,
}

#[cfg(feature = "net")]
#[derive(Default)]
pub(crate) struct ExecutorStats {
    inflight: AtomicU32,
    leased_pages: AtomicU32,
}

#[cfg(feature = "net")]
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

#[cfg(feature = "net")]
pub(crate) struct ExecutorServer {
    endpoint: String,
    stats: Arc<ExecutorStats>,
    total_pages: u32,
}

#[cfg(feature = "net")]
impl ExecutorServer {
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

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(crate) fn stats(&self) -> Arc<ExecutorStats> {
        Arc::clone(&self.stats)
    }

    pub(crate) fn total_pages(&self) -> u32 {
        self.total_pages
    }

    pub(crate) async fn shutdown(self) {}
}

#[cfg(feature = "net")]
pub(crate) async fn connect_with_local_ip(addr: &str) -> Result<((), IpAddr)> {
    Err(anyhow!(
        "cannot dial executor {addr}: remote executors are not supported in this release"
    ))
}
