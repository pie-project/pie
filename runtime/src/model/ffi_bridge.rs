//! IPC bridge for cross-process Python communication.
//!
//! This module provides an async wrapper for IPC-based communication with
//! Python worker processes in the symmetric worker architecture.

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;

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
        // Serialize arguments
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow::anyhow!("Failed to serialize args: {}", e))?;
        
        // Send via IPC
        let response = self.backend.call(method, payload).await?;
        
        // Deserialize response — detect Python-side exceptions
        match rmp_serde::from_slice(&response) {
            Ok(result) => Ok(result),
            Err(deser_err) => {
                // Python error handler sends {"__error__": msg, "__traceback__": tb}
                if let Ok(map) = rmp_serde::from_slice::<HashMap<String, String>>(&response)
                    && let Some(msg) = map.get("__error__")
                {
                    let tb = map.get("__traceback__").map(|s| s.as_str()).unwrap_or("");
                    return Err(anyhow::anyhow!(
                        "Python error in `{}`: {}\n{}",
                        method, msg, tb
                    ));
                }
                Err(anyhow::anyhow!("Failed to deserialize `{}` response: {}", method, deser_err))
            }
        }
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
