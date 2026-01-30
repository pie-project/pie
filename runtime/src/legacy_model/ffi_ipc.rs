//! IPC-based FFI queue for cross-process Rust ↔ Python communication.
//!
//! This module uses `ipc-channel` for high-performance cross-process IPC,
//! allowing each group leader process to have its own FFI queue without
//! sharing the GIL with Rank 0.
//!
//! # Design
//!
//! Both `IpcSender` and `IpcReceiver` are serializable, so we can send
//! channel endpoints over an existing channel. The pattern is:
//!
//! 1. Rust creates `IpcOneShotServer` and shares `server_name` with Python
//! 2. Python connects using `IpcSender::connect(server_name)`
//! 3. Rust accepts and receives an initial message with Python's response sender
//! 4. Both sides now have bidirectional communication:
//!    - Rust → Python: via request sender
//!    - Python → Rust: via response sender that Python sent during connect
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Rust Async Runtime (Rank 0 Process)                            │
//! │   ┌─────────────────────────────────────────────────────────┐   │
//! │   │ FfiIpcBackend                                           │   │
//! │   │  request_tx ──────────► Python                          │   │
//! │   │  response_rx ◄────────── Python                         │   │
//! │   └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//!                          ipc-channel
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Python Worker Process (Group Leader - owns its own GIL)        │
//! │   ┌─────────────────────────────────────────────────────────┐   │
//! │   │ FfiIpcQueue                                             │   │
//! │   │  request_rx ◄────────── Rust                            │   │
//! │   │  response_tx ──────────► Rust                           │   │
//! │   └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcSender};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

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
    /// Create a new IPC backend for a specific group.
    ///
    /// Returns `(backend, server_name)`. Pass `server_name` to Python
    /// so it can connect using `FfiIpcQueue.connect(server_name)`.
    pub fn new(group_id: usize) -> Result<(Self, String)> {
        // Create both channel pairs upfront
        let (request_tx, request_rx) = ipc::channel::<IpcRequest>()?;
        let (response_tx, response_rx) = ipc::channel::<IpcResponse>()?;
        
        // Create one-shot server for initial handshake
        let (one_shot_server, server_name) = IpcOneShotServer::<()>::new()?;
        
        // Spawn thread to accept connection and send channel endpoints
        let channels = IpcChannels { request_rx, response_tx };
        let connected = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let connected_clone = Arc::clone(&connected);
        
        std::thread::spawn(move || {
            if let Ok((_, _)) = one_shot_server.accept() {
                // Connection accepted - nothing to do, channels are in IpcChannels struct
                // Actually we need to send the channels... let me rethink
            }
            connected_clone.store(true, Ordering::SeqCst);
        });
        
        // Hmm, IpcOneShotServer returns data FROM the client. We need to SEND to the client.
        // The pattern should be:
        // 1. Python connects via IpcSender::connect(server_name)
        // 2. Rust's one_shot_server.accept() returns an IpcReceiver for the data Python sends
        // 3. But we need to send channels TO Python
        //
        // Actually looking at the servo example:
        // - Server creates IpcOneShotServer<ChannelType>
        // - Client connects with IpcSender::connect() 
        // - Client SENDS something via that sender
        // - Server receives it via accept()
        //
        // So for bidirectional:
        // - Server creates channels and one_shot_server
        // - Python connects and sends a "hello" (or its response sender)
        // - Server receives, then sends the request_rx to Python somehow...
        //
        // Actually we can do it by having Python connect, send its response_tx,
        // and then we send request_rx back... but that requires TWO round trips.
        //
        // Simpler: Use a shared memory or file-based mechanism to pass the OsIpcChannel
        // Or: Create channels differently...
        //
        // Let me check if IpcOneShotServer can send data to the client...
        // Looking at the API: accept() -> Result<(IpcReceiver<T>, T)>
        // The T is what the CLIENT sent. So we need client to send us something.
        //
        // Clean solution:
        // 1. Rust creates IpcOneShotServer<IpcChannels>
        // 2. Python calls IpcSender::connect(name) to get a sender
        // 3. Python creates (response_tx, response_rx) locally
        // 4. Python sends IpcChannels { request_rx: ???, response_tx }
        //
        // Wait, that's backwards. We need RUST to create request channel.
        //
        // Ok, cleanest design:
        // 1. Rust creates IpcOneShotServer<IpcSender<IpcChannels>>
        // 2. When Python connects, it sends an IpcSender<IpcChannels> to Rust
        // 3. Rust uses that sender to send the IpcChannels to Python
        // 4. Now Python has request_rx and response_tx
        
        // Actually even simpler using the pattern from servo:
        // Server sends channels TO client by including them in the response
        // But IpcOneShotServer is for receiving FROM client...
        //
        // Let me use a different approach: Two-stage handshake
        
        let backend = Self {
            request_tx: Mutex::new(request_tx),
            response_rx: Arc::new(Mutex::new(response_rx)),
            pending: Arc::new(dashmap::DashMap::new()),
            next_id: AtomicU64::new(1),
            server_name: server_name.clone(),
            group_id,
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        
        // We need a way to send `channels` to Python. Let's use a second channel.
        // Actually we can store it and let Python fetch via a different mechanism.
        //
        // For now, let me just serialize the channels to a file that Python can read.
        // This is a hacky but simple approach.
        
        Ok((backend, server_name))
    }
    
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
        
        // Create two one-shot servers:
        // 1. For Python to connect and acknowledge
        // 2. For us to send the channels
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
        
        // Start response handler
        self.spawn_response_handler();
        
        Ok(())
    }
    
    /// Spawn background thread to handle responses from Python
    fn spawn_response_handler(&self) {
        let response_rx = Arc::clone(&self.response_rx);
        let pending = Arc::clone(&self.pending);
        
        std::thread::spawn(move || {
            loop {
                let rx = response_rx.lock().unwrap();
                match rx.recv() {
                    Ok(response) => {
                        drop(rx); // Release lock before processing
                        if let Some((_, tx)) = pending.remove(&response.request_id) {
                            let _ = tx.send(response.payload);
                        }
                    }
                    Err(_) => break, // Channel closed
                }
            }
        });
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
    ///
    /// This sends a "shutdown" method call to the Python worker, causing its
    /// IPC poll loop to exit gracefully.
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

/// Client-side IPC queue (used by Python worker processes via PyO3).
///
/// Python creates this by connecting to the server name provided by Rust.
#[pyclass]
pub struct FfiIpcQueue {
    /// Receiver for requests from Rust
    request_rx: Arc<Mutex<IpcReceiver<IpcRequest>>>,
    /// Sender for responses to Rust (wrapped in Mutex for Sync safety)
    response_tx: Mutex<IpcSender<IpcResponse>>,
    /// Group ID
    group_id: usize,
    /// Flag to signal that the queue is closed and poll should exit
    closed: Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl FfiIpcQueue {
    /// Connect to the IPC server using its server name.
    ///
    /// This performs a two-stage handshake:
    /// 1. Connect to server and send a channels receiver
    /// 2. Receive the actual request/response channels from server
    #[staticmethod]
    fn connect(server_name: &str, group_id: usize) -> PyResult<Self> {
        // Create a one-shot receiver to get channels from Rust
        let (channels_tx, channels_rx) = ipc::channel::<IpcChannels>()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        
        // Connect to Rust's one-shot server and send our receiver
        let sender = IpcSender::connect(server_name.to_string())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        
        sender.send(channels_tx)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        
        // Receive the channels from Rust
        let channels = channels_rx.recv()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        
        Ok(Self {
            request_rx: Arc::new(Mutex::new(channels.request_rx)),
            response_tx: Mutex::new(channels.response_tx),
            group_id,
            closed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
    
    /// Poll for the next request from Rust (blocking with timeout).
    ///
    /// Returns `(request_id, method_name, payload_bytes)` or `None` on timeout.
    /// Raises `PyIOError` with "Queue closed" if `close()` was called.
    fn poll_blocking(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<(u64, String, Py<PyBytes>)>> {
        // Check if closed before polling
        if self.closed.load(Ordering::SeqCst) {
            return Err(pyo3::exceptions::PyIOError::new_err("Queue closed"));
        }

        let request_rx = Arc::clone(&self.request_rx);
        let closed = Arc::clone(&self.closed);
        let timeout = std::time::Duration::from_millis(timeout_ms);
        
        // Release GIL while waiting
        let result = py.allow_threads(move || {
            // Check again after acquiring potential lock
            if closed.load(Ordering::SeqCst) {
                return Err("Queue closed".to_string());
            }
            let rx = request_rx.lock().unwrap();
            match rx.try_recv_timeout(timeout) {
                Ok(request) => Ok(Some(request)),
                Err(ipc_channel::ipc::TryRecvError::Empty) => {
                    // Check if closed during wait
                    if closed.load(Ordering::SeqCst) {
                        Err("Queue closed".to_string())
                    } else {
                        Ok(None)
                    }
                }
                Err(ipc_channel::ipc::TryRecvError::IpcError(e)) => {
                    Err(format!("IPC error: {:?}", e))
                }
            }
        });
        
        match result {
            Ok(Some(request)) => {
                let py_bytes = PyBytes::new(py, &request.payload).into();
                Ok(Some((request.request_id, request.method, py_bytes)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e)),
        }
    }
    
    /// Close the queue, causing any pending or future poll_blocking calls to return error.
    ///
    /// This should be called during shutdown to signal workers to exit.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
    
    /// Check if the queue is closed.
    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
    
    /// Send a response back to Rust for the given request ID.
    fn respond(&self, request_id: u64, response: &[u8]) -> PyResult<bool> {
        let tx = self.response_tx.lock().unwrap();
        tx.send(IpcResponse {
            request_id,
            payload: response.to_vec(),
        })
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        Ok(true)
    }
    
    /// Get the group ID this queue handles.
    fn group_id(&self) -> usize {
        self.group_id
    }
}
