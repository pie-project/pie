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
        static FINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fine = *FINE.get_or_init(|| std::env::var("PIE_FINE_TIMING").is_ok());
        let t0 = std::time::Instant::now();

        // Serialize arguments
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow::anyhow!("Failed to serialize args: {}", e))?;
        let t1 = std::time::Instant::now();
        let payload_len = payload.len();

        // Send via IPC
        let response = self.backend.call(method, payload).await?;
        let t2 = std::time::Instant::now();
        let resp_len = response.len();

        // Deserialize response
        let result = rmp_serde::from_slice(&response)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {}", e))?;
        let t3 = std::time::Instant::now();

        if fine && method == "fire_batch" {
            eprintln!(
                "[IPC-DETAIL] serialize={:.2}ms ipc={:.2}ms deserialize={:.2}ms total={:.2}ms payload={}B resp={}B",
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t3.duration_since(t2).as_secs_f64() * 1000.0,
                t3.duration_since(t0).as_secs_f64() * 1000.0,
                payload_len, resp_len,
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
