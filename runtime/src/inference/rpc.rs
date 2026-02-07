//! RPC infrastructure for cross-process IPC communication.
//!
//! Provides both server and client types for IPC-based RPC:
//!
//! - **`RpcServer`**: Creates an `IpcOneShotServer`, accepts a client
//!   connection, and exposes `poll()` / `respond()` for request handling.
//!
//! - **`RpcClient`**: Connects to an existing `IpcOneShotServer` and
//!   exposes async `call()` / `notify()` for sending requests.
//!
//! In the inference pattern, Python wraps `RpcServer` (via `PyRpcServer`)
//! and Rust connects via `RpcClient`.

use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcSender};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

// =============================================================================
// Wire Types
// =============================================================================

/// Request message sent over IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    pub request_id: u64,
    pub method: String,
    pub payload: Vec<u8>,
}

/// Response message sent over IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    pub request_id: u64,
    pub payload: Vec<u8>,
}

// =============================================================================
// Channel Exchange
// =============================================================================

/// Channel endpoints exchanged during the handshake.
///
/// Bundled from the **client's perspective**:
/// - `request_tx`: client sends requests on this
/// - `response_rx`: client receives responses on this
#[derive(Debug, Serialize, Deserialize)]
struct IpcChannels {
    request_tx: IpcSender<IpcRequest>,
    response_rx: IpcReceiver<IpcResponse>,
}

// =============================================================================
// RpcServer — receives requests, sends responses
// =============================================================================

/// IPC server that receives requests and sends responses.
///
/// Creates an `IpcOneShotServer` for a single client to connect to.
/// After the handshake, the server can poll for incoming requests and
/// send responses back.
///
/// This is transport-only — no serialization is performed.
pub struct RpcServer {
    /// Receiver for incoming requests from the client
    request_rx: Mutex<IpcReceiver<IpcRequest>>,
    /// Sender for outgoing responses to the client
    response_tx: Mutex<IpcSender<IpcResponse>>,
    /// Server name for clients to connect
    server_name: String,
    /// Whether the server has been closed
    closed: AtomicBool,
}

impl RpcServer {
    /// Create a new IPC server.
    ///
    /// Sets up channel pairs and an `IpcOneShotServer`. A background thread
    /// waits for a client to connect and completes the channel exchange.
    pub fn create() -> Result<Self> {
        // Create both channel pairs
        let (request_tx, request_rx) = ipc::channel::<IpcRequest>()?;
        let (response_tx, response_rx) = ipc::channel::<IpcResponse>()?;

        // Create one-shot server for the client to connect
        let (ack_server, server_name) = IpcOneShotServer::<IpcSender<IpcChannels>>::new()?;

        // Bundle the client-side endpoints
        let channels = IpcChannels {
            request_tx,
            response_rx,
        };

        // Spawn thread to handle the handshake (blocking accept)
        std::thread::spawn(move || {
            match ack_server.accept() {
                Ok((_, channels_tx)) => {
                    if let Err(e) = channels_tx.send(channels) {
                        tracing::error!("RPC handshake: failed to send channels: {e}");
                    }
                }
                Err(e) => {
                    tracing::error!("RPC handshake: accept failed: {e}");
                }
            }
        });

        Ok(Self {
            request_rx: Mutex::new(request_rx),
            response_tx: Mutex::new(response_tx),
            server_name,
            closed: AtomicBool::new(false),
        })
    }

    /// Get the server name for clients to connect to.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Poll for the next request (blocking with timeout).
    ///
    /// Returns `Ok(Some(request))` if a request arrived,
    /// `Ok(None)` on timeout, or `Err` if the server is closed
    /// or the channel errored.
    pub fn poll(&self, timeout: Duration) -> Result<Option<IpcRequest>> {
        if self.closed.load(Ordering::SeqCst) {
            bail!("Server closed");
        }

        let rx = self.request_rx.lock().unwrap();
        match rx.try_recv_timeout(timeout) {
            Ok(request) => Ok(Some(request)),
            Err(ipc_channel::ipc::TryRecvError::Empty) => {
                if self.closed.load(Ordering::SeqCst) {
                    bail!("Server closed");
                }
                Ok(None)
            }
            Err(ipc_channel::ipc::TryRecvError::IpcError(e)) => {
                bail!("IPC error: {:?}", e);
            }
        }
    }

    /// Send a response back to the client.
    pub fn respond(&self, request_id: u64, payload: Vec<u8>) -> Result<()> {
        let tx = self.response_tx.lock().unwrap();
        tx.send(IpcResponse {
            request_id,
            payload,
        })?;
        Ok(())
    }

    /// Close the server, causing pending/future polls to error.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Check if the server has been closed.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

// =============================================================================
// RpcClient — sends requests, receives responses
// =============================================================================

/// IPC client that connects to an `RpcServer` (or `PyRpcServer`).
///
/// After the handshake, the client sends `IpcRequest` messages and
/// receives `IpcResponse` messages. A background thread routes incoming
/// responses to their corresponding oneshot channels.
#[derive(Clone)]
pub struct RpcClient {
    /// Sender for outgoing requests
    request_tx: Arc<Mutex<IpcSender<IpcRequest>>>,
    /// Pending response channels indexed by request ID
    pending: Arc<DashMap<u64, oneshot::Sender<Vec<u8>>>>,
    /// Counter for request IDs
    next_id: Arc<AtomicU64>,
}

impl RpcClient {
    /// Connect to an `RpcServer` and perform the handshake.
    ///
    /// 1. Create a temporary `ipc::channel<IpcChannels>` carrier
    /// 2. Connect to the server's `IpcOneShotServer` and send the carrier sender
    /// 3. Receive the channel endpoints from the server
    /// 4. Spawn a background thread to route responses
    pub fn connect(server_name: &str) -> Result<Self> {
        // Create carrier channel to receive IpcChannels from the server
        let (channels_tx, channels_rx) = ipc::channel::<IpcChannels>()?;

        // Connect to the server's one-shot server and send our carrier
        let sender = IpcSender::connect(server_name.to_string())?;
        sender.send(channels_tx)?;

        // Receive the channel endpoints
        let channels = channels_rx.recv()?;

        // Set up response routing
        let pending: Arc<DashMap<u64, oneshot::Sender<Vec<u8>>>> =
            Arc::new(DashMap::new());
        let pending_clone = Arc::clone(&pending);

        let response_rx = channels.response_rx;

        // Spawn blocking thread to route responses to pending oneshot channels
        std::thread::spawn(move || {
            loop {
                match response_rx.recv() {
                    Ok(response) => {
                        if let Some((_, tx)) = pending_clone.remove(&response.request_id) {
                            let _ = tx.send(response.payload);
                        }
                    }
                    Err(_) => break, // Channel closed
                }
            }
        });

        Ok(Self {
            request_tx: Arc::new(Mutex::new(channels.request_tx)),
            pending,
            next_id: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Send a raw request and await the response.
    async fn raw_call(&self, method: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let request = IpcRequest {
            request_id,
            method: method.to_string(),
            payload,
        };

        // Register pending response
        let (response_tx, response_rx) = oneshot::channel();
        self.pending.insert(request_id, response_tx);

        // Send request (scope ensures lock is dropped before await)
        if let Err(e) = {
            let tx = self.request_tx.lock().unwrap();
            tx.send(request)
        } {
            self.pending.remove(&request_id);
            return Err(e.into());
        }

        // Await response
        response_rx
            .await
            .map_err(|_| anyhow!("Response channel closed"))
    }

    /// Call a remote method, serializing args and deserializing the response.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow!("Failed to serialize args: {}", e))?;

        let response = self.raw_call(method, payload).await?;

        rmp_serde::from_slice(&response)
            .map_err(|e| anyhow!("Failed to deserialize response: {}", e))
    }

    /// Fire-and-forget notification (no response expected).
    pub fn notify<T>(&self, method: &str, args: &T) -> Result<()>
    where
        T: Serialize,
    {
        let payload = rmp_serde::to_vec_named(args)
            .map_err(|e| anyhow!("Failed to serialize args: {}", e))?;

        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = IpcRequest {
            request_id,
            method: method.to_string(),
            payload,
        };

        let tx = self.request_tx.lock().unwrap();
        tx.send(request)?;
        Ok(())
    }

    /// Call with a timeout.
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
            .map_err(|_| anyhow!("RPC call timed out"))?
    }

}

impl std::fmt::Debug for RpcClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcClient")
            .field("pending_requests", &self.pending.len())
            .finish()
    }
}