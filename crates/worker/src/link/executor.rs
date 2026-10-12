//! The prefill and encode executor roles, which this release refuses.

use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow};

use crate::backend::ModelEngines;
use crate::executor::ModelIdentity;

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

pub(crate) struct ExecutorServer {
    endpoint: String,
    stats: Arc<ExecutorStats>,
    total_pages: u32,
}

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

pub(crate) async fn connect_with_local_ip(addr: &str) -> Result<((), IpAddr)> {
    Err(anyhow!(
        "cannot dial executor {addr}: remote executors are not supported in this release"
    ))
}
