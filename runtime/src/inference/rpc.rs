//! RPC backend for Python IPC communication.
//!
//! This module provides an async wrapper for IPC-based communication with
//! Python worker processes.

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Backend for the IPC communication layer.
/// This is a placeholder - will be connected to actual IPC implementation.
#[derive(Clone, Debug)]
pub struct IpcBackend {
    // TODO: Connect to actual IPC implementation
}

impl IpcBackend {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn call(&self, _method: &str, _payload: Vec<u8>) -> Result<Vec<u8>> {
        // Placeholder - will be implemented with actual IPC
        Ok(Vec::new())
    }
}

/// Async IPC client for cross-process communication.
///
/// Uses IPC to communicate with Python processes.
#[derive(Clone, Debug)]
pub struct RpcClient {
    backend: Arc<IpcBackend>,
}

impl RpcClient {
    /// Create a new RPC client from an IPC backend.
    pub fn new(backend: Arc<IpcBackend>) -> Self {
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
        
        // Deserialize response
        rmp_serde::from_slice(&response)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {}", e))
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
        timeout: Duration,
    ) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        tokio::time::timeout(timeout, self.call(method, args))
            .await
            .map_err(|_| anyhow::anyhow!("RPC call timed out"))?
    }
}
