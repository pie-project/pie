//! RPC/IPC infrastructure for cross-process communication.
//!
//! This module provides the IPC backend and client types used for
//! communication between the Rust runtime and Python worker processes.

use anyhow::Result;
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcSender};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

// =============================================================================
// IPC Message Types
// =============================================================================

/// Request message sent from Rust to Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    pub request_id: u64,
    pub method: String,
    pub payload: Vec<u8>,
}

/// Response message sent from Python to Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    pub request_id: u64,
    pub payload: Vec<u8>,
}

/// Channel endpoints sent from Rust to Python during connection setup
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcChannels {
    /// Python receives requests on this
    pub request_rx: IpcReceiver<IpcRequest>,
    /// Python sends responses on this (cloned from Rust's sender)
    pub response_tx: IpcSender<IpcResponse>,
}

// =============================================================================
// FfiIpcBackend - Rust side IPC server
// =============================================================================

/// Server-side IPC backend (used by Rust runtime in Rank 0 process).
///
/// Exposes an async `call()` interface compatible with the existing `RpcBackend`.
pub struct FfiIpcBackend {
    /// Sender for Rust → Python requests (wrapped in Mutex for Sync safety)
    request_tx: Mutex<IpcSender<IpcRequest>>,
    /// Receiver for Python → Rust responses  
    response_rx: Arc<Mutex<IpcReceiver<IpcResponse>>>,
    /// Pending response channels
    pending: Arc<dashmap::DashMap<u64, oneshot::Sender<Vec<u8>>>>,
    /// Counter for request IDs
    next_id: AtomicU64,
    /// Server name for Python to connect
    server_name: String,
    /// Group ID
    group_id: usize,
    /// Whether connection is established
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl FfiIpcBackend {
    /// Create a new IPC backend with proper channel exchange.
    ///
    /// This uses a two-stage handshake:
    /// 1. Rust creates channels and a one-shot server
    /// 2. Python connects and sends an empty ack
    /// 3. Rust sends the channel pair to Python
    pub fn new_with_handshake(group_id: usize) -> Result<(Self, String)> {
        // Create both channel pairs
        let (request_tx, request_rx) = ipc::channel::<IpcRequest>()?;
        let (response_tx, response_rx) = ipc::channel::<IpcResponse>()?;
        
        // Create one-shot server for Python to connect
        let (ack_server, server_name) = IpcOneShotServer::<IpcSender<IpcChannels>>::new()?;
        
        let connected = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let connected_clone = Arc::clone(&connected);
        
        // Prepare channels to send to Python
        let channels = IpcChannels { request_rx, response_tx };
        
        // Clone response_rx and pending for the response handler
        let response_rx_for_handler = Arc::new(Mutex::new(response_rx));
        let response_rx_clone = Arc::clone(&response_rx_for_handler);
        let pending: Arc<dashmap::DashMap<u64, oneshot::Sender<Vec<u8>>>> = Arc::new(dashmap::DashMap::new());
        let pending_clone = Arc::clone(&pending);
        
        // Spawn thread to handle connection AND start response handler
        std::thread::spawn(move || {
            // Wait for Python to connect and send us a sender
            if let Ok((_, channels_tx)) = ack_server.accept() {
                // Python sent us a sender, use it to send the channels
                if channels_tx.send(channels).is_ok() {
                    connected_clone.store(true, Ordering::SeqCst);
                    
                    // Start response handler in this thread (blocking receive loop)
                    loop {
                        let rx = response_rx_clone.lock().unwrap();
                        match rx.recv() {
                            Ok(response) => {
                                drop(rx); // Release lock before processing
                                if let Some((_, tx)) = pending_clone.remove(&response.request_id) {
                                    let _ = tx.send(response.payload);
                                }
                            }
                            Err(_) => break, // Channel closed
                        }
                    }
                }
            }
        });
        
        let backend = Self {
            request_tx: Mutex::new(request_tx),
            response_rx: response_rx_for_handler,
            pending,
            next_id: AtomicU64::new(1),
            server_name: server_name.clone(),
            group_id,
            connected,
        };
        
        Ok((backend, server_name))
    }
    
    /// Check if Python is connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
    
    /// Wait for connection to be established
    pub async fn wait_for_connection(&self) -> Result<()> {
        let connected = Arc::clone(&self.connected);
        tokio::task::spawn_blocking(move || {
            while !connected.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }).await?;
        Ok(())
    }
    
    /// Send a request to Python and await response (async)
    pub async fn call(&self, method: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        let request = IpcRequest {
            request_id,
            method: method.to_string(),
            payload,
        };
        
        // Create response channel
        let (response_tx, response_rx) = oneshot::channel();
        self.pending.insert(request_id, response_tx);
        
        // Send request (scope ensures lock is dropped before await)
        {
            let tx = self.request_tx.lock().unwrap();
            tx.send(request)?;
        }
        
        // Await response
        response_rx.await.map_err(|_| anyhow::anyhow!("Response channel closed"))
    }
    
    /// Get server name for Python to connect
    pub fn server_name(&self) -> &str {
        &self.server_name
    }
    
    /// Get group ID
    pub fn group_id(&self) -> usize {
        self.group_id
    }
    
    /// Broadcast shutdown message to Python worker.
    pub fn broadcast_shutdown(&self) -> Result<()> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        let request = IpcRequest {
            request_id,
            method: "shutdown".to_string(),
            payload: vec![],
        };
        
        // Send request - don't wait for response
        let tx = self.request_tx.lock().unwrap();
        tx.send(request)?;
        
        Ok(())
    }
}

// =============================================================================
// AsyncIpcClient - Rust async wrapper
// =============================================================================

/// Async IPC client for cross-process communication.
///
/// Uses ipc-channel to communicate with Python processes in other PIDs.
#[derive(Clone)]
pub struct AsyncIpcClient {
    backend: Arc<FfiIpcBackend>,
}

impl AsyncIpcClient {
    /// Create a new IPC client from an FfiIpcBackend.
    pub fn new(backend: Arc<FfiIpcBackend>) -> Self {
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
            .map_err(|_| anyhow::anyhow!("IPC call timed out"))?
    }
}

// =============================================================================
// RPC Backend
// =============================================================================

/// RPC backend for Python IPC communication.
///
/// Thin wrapper around AsyncIpcClient with stricter type bounds for use
/// in model handshake and backend communication.
#[derive(Clone)]
pub struct RpcBackend {
    client: AsyncIpcClient,
}

impl RpcBackend {
    /// Create a new RPC backend from an IPC client.
    pub fn new(client: AsyncIpcClient) -> Self {
        Self { client }
    }

    /// Call a Python method asynchronously via IPC.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: DeserializeOwned + Send + 'static,
    {
        self.client.call(method, args).await
    }

    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: Duration,
    ) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: DeserializeOwned + Send + 'static,
    {
        self.client.call_with_timeout(method, args, timeout).await
    }
}

impl std::fmt::Debug for RpcBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcBackend").finish()
    }
}
