//! IPC bridge for cross-process Python communication.
//!
//! This module provides an async wrapper for IPC-based communication with
//! Python worker processes in the symmetric worker architecture.

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

/// Async IPC client for cross-process communication.
///
/// Uses ipc-channel to communicate with Python processes in other PIDs.
#[derive(Clone)]
pub struct AsyncIpcClient {
    backend: std::sync::Arc<crate::model::ffi_ipc::FfiIpcBackend>,
}

impl AsyncIpcClient {
    /// Create a new IPC client from an FfiIpcBackend.
    pub fn new(backend: std::sync::Arc<crate::model::ffi_ipc::FfiIpcBackend>) -> Self {
        Self { backend }
    }

    /// Call a Python method asynchronously via IPC.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        #[cfg(feature = "ipc-profiling")]
        let t0 = std::time::Instant::now();

        // Serialize arguments
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow::anyhow!("Failed to serialize args: {}", e))?;
        #[cfg(feature = "ipc-profiling")]
        let payload_len = payload.len();
        #[cfg(feature = "ipc-profiling")]
        let t_serialized = std::time::Instant::now();

        // Send via IPC
        let response = self.backend.call(method, payload).await?;
        #[cfg(feature = "ipc-profiling")]
        let t_ipc_done = std::time::Instant::now();

        // Deserialize response
        let result: R = rmp_serde::from_slice(&response)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {}", e))?;

        #[cfg(feature = "ipc-profiling")]
        if crate::model::ffi_ipc::ipc_timing_enabled() {
            let t_deserialized = std::time::Instant::now();
            eprintln!(
                "[BRIDGE-SERDE] serialize={:.3}ms ipc={:.3}ms deserialize={:.3}ms total={:.3}ms req_bytes={} resp_bytes={}",
                t_serialized.duration_since(t0).as_micros() as f64 / 1000.0,
                t_ipc_done.duration_since(t_serialized).as_micros() as f64 / 1000.0,
                t_deserialized.duration_since(t_ipc_done).as_micros() as f64 / 1000.0,
                t_deserialized.duration_since(t0).as_micros() as f64 / 1000.0,
                payload_len,
                response.len(),
            );
        }

        Ok(result)
    }

    /// Fire-and-forget notification.
    pub async fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize,
    {
        let _: () = self.call(method, args).await?;
        Ok(())
    }

    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: std::time::Duration,
    ) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        tokio::time::timeout(timeout, self.call(method, args))
            .await
            .map_err(|_| anyhow::anyhow!("IPC call timed out"))?
    }
}
