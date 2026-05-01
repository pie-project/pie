//! # Device Module
//!
//! Device abstraction and RPC communication for inference backends.
//!
//! Each physical device runs as a service actor that owns its RPC connection.
//! Public functions provide a typed interface over message-passing:
//! [`call()`], [`notify()`], and [`get_spec()`].
//!
//! `fire_batch` always takes the shared-memory zero-copy fast path; all other
//! RPC methods (adapter ops, page copies, query/ping) use the ipc-channel
//! transport. There are no env-var toggles.

use std::sync::LazyLock;
use std::time::Duration;
use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcSender};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

use crate::service::{ServiceArray, ServiceHandler};

/// Default shmem region name (one worker per host). Geometry of the region
/// — slot count, request/response buffer sizes — is owned by the driver
/// and read out of the header at attach time; see `shmem_ipc::ShmemClient`.
const SHMEM_NAME: &str = "/pie_shmem";
/// Busy-spin window (µs) before yielding while waiting on resp_seq.
const SHMEM_SPIN_US: u64 = 10_000;

/// Device identifier.
pub type DeviceId = usize;

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);


/// Spawns a device service and connects to its RPC backend.
/// Returns the device index used to address it in subsequent calls.
pub fn spawn(
    hostname: &str,
    num_kv_pages: usize,
    max_batch_size: usize,
    max_batch_tokens: usize,
) -> usize {
    SERVICES.spawn(move || {
        let device = DeviceSpec {
            hostname: hostname.to_string(),
            num_kv_pages,
            max_batch_size,
            max_batch_tokens,
        };
        let rpc = RpcClient::connect(hostname)
            .unwrap_or_else(|e| panic!("Failed to connect to device {hostname}: {e}"));
        Device { config: device, rpc }
    }).expect("Failed to spawn device service")
}

/// Calls a remote method, serializing `args` and deserializing the response.
pub async fn call<T: Serialize, R: DeserializeOwned>(
    device_idx: usize,
    method: &str,
    args: &T,
) -> Result<R> {
    let payload = rmp_serde::to_vec_named(args)
        .map_err(|e| anyhow!("Failed to serialize args: {e}"))?;
    let (tx, rx) = oneshot::channel();
    SERVICES.send(device_idx, Message::Call {
        method: method.to_string(),
        payload,
        response: tx,
    })?;
    let response = rx.await
        .map_err(|_| anyhow!("Device service channel closed"))??;
    rmp_serde::from_slice(&response)
        .map_err(|e| anyhow!("Failed to deserialize response: {e}"))
}

/// Calls a remote method with a timeout.
pub async fn call_with_timeout<T: Serialize, R: DeserializeOwned>(
    device_idx: usize,
    method: &str,
    args: &T,
    timeout: Duration,
) -> Result<R> {
    tokio::time::timeout(timeout, call(device_idx, method, args))
        .await
        .map_err(|_| anyhow!("Device call '{method}' timed out"))?
}



/// Sends a fire-and-forget notification, serializing `args`.
pub fn notify<T: Serialize>(device_idx: usize, method: &str, args: &T) -> Result<()> {
    let payload = rmp_serde::to_vec_named(args)
        .map_err(|e| anyhow!("Failed to serialize args: {e}"))?;
    SERVICES.send(device_idx, Message::Notify {
        method: method.to_string(),
        payload,
    })
}

// =============================================================================
// Convenience Wrappers
// =============================================================================

/// Fires a batched forward pass on the given device via the zero-copy
/// shared-memory fast path.
pub async fn fire_batch(
    device_idx: usize,
    batch: &crate::inference::request::BatchedForwardPassRequest,
) -> Result<crate::inference::request::BatchedForwardPassResponse> {
    let client = get_or_init_shmem_client(device_idx)?;
    let req_id = SHMEM_REQ_ID.fetch_add(1, Ordering::Relaxed) as u32;
    let send_walltime_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);

    let (resp_bytes, _ts) = tokio::task::block_in_place(|| {
        client.call_with(
            req_id,
            crate::shmem_ipc::METHOD_TAG_FIRE_BATCH,
            send_walltime_us,
            |buf| crate::shmem_schema::write_request(batch, buf),
        )
    })?;
    crate::shmem_schema::read_response(&resp_bytes)
}

/// Lazy-initialized shmem clients keyed by device index. All devices share
/// one client today (one shmem region per worker).
static SHMEM_CLIENTS: LazyLock<DashMap<usize, Arc<crate::shmem_ipc::ShmemClient>>> =
    LazyLock::new(DashMap::new);

static SHMEM_REQ_ID: AtomicU64 = AtomicU64::new(1);

fn get_or_init_shmem_client(_device_idx: usize) -> Result<Arc<crate::shmem_ipc::ShmemClient>> {
    if let Some(c) = SHMEM_CLIENTS.get(&0) {
        return Ok(c.clone());
    }
    let client = crate::shmem_ipc::ShmemClient::open(SHMEM_NAME, SHMEM_SPIN_US)?;
    let arc = Arc::new(client);
    SHMEM_CLIENTS.insert(0, arc.clone());
    Ok(arc)
}

/// GPU → CPU page copy (fire-and-forget).
/// `gpu_phys_ids`: source GPU physical page IDs.
/// `cpu_pages`: destination CPU swap pool page IDs.
pub fn copy_d2h(device_idx: DeviceId, gpu_phys_ids: &[u32], cpu_pages: &[u32]) -> Result<()> {
    #[derive(Serialize)]
    struct Req { phys_ids: Vec<u32>, slots: Vec<u32> }
    notify(device_idx, "copy_d2h", &Req {
        phys_ids: gpu_phys_ids.to_vec(),
        slots: cpu_pages.to_vec(),
    })
}

/// CPU → GPU page copy (fire-and-forget).
/// `gpu_phys_ids`: destination GPU physical page IDs.
/// `cpu_pages`: source CPU swap pool page IDs.
pub fn copy_h2d(device_idx: DeviceId, gpu_phys_ids: &[u32], cpu_pages: &[u32]) -> Result<()> {
    #[derive(Serialize)]
    struct Req { phys_ids: Vec<u32>, slots: Vec<u32> }
    notify(device_idx, "copy_h2d", &Req {
        phys_ids: gpu_phys_ids.to_vec(),
        slots: cpu_pages.to_vec(),
    })
}

/// GPU → GPU page copy (fire-and-forget).
/// `src_phys_ids`: source GPU physical page IDs.
/// `dst_phys_ids`: destination GPU physical page IDs.
pub fn copy_d2d(device_idx: DeviceId, src_phys_ids: &[u32], dst_phys_ids: &[u32]) -> Result<()> {
    #[derive(Serialize)]
    struct Req { src_phys_ids: Vec<u32>, dst_phys_ids: Vec<u32> }
    notify(device_idx, "copy_d2d", &Req {
        src_phys_ids: src_phys_ids.to_vec(),
        dst_phys_ids: dst_phys_ids.to_vec(),
    })
}

/// CPU → CPU page copy (fire-and-forget).
/// `src_slots`: source CPU swap pool page IDs.
/// `dst_slots`: destination CPU swap pool page IDs.
pub fn copy_h2h(device_idx: DeviceId, src_slots: &[u32], dst_slots: &[u32]) -> Result<()> {
    #[derive(Serialize)]
    struct Req { src_slots: Vec<u32>, dst_slots: Vec<u32> }
    notify(device_idx, "copy_h2h", &Req {
        src_slots: src_slots.to_vec(),
        dst_slots: dst_slots.to_vec(),
    })
}

/// Returns the device's static configuration.
pub async fn get_spec(device_idx: usize) -> Result<DeviceSpec> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(device_idx, Message::GetSpec { response: tx })?;
    rx.await.map_err(|_| anyhow!("Device service channel closed"))
}


// =============================================================================
// Internal Types
// =============================================================================


/// Static device configuration (hostname, capacity limits).
///
/// Populated at spawn time from `bootstrap::DeviceConfig`.
#[derive(Debug, Clone)]
pub struct DeviceSpec {
    pub hostname: String,
    pub num_kv_pages: usize,
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
}



/// Actor messages for the device service.
#[derive(Debug)]
enum Message {
    /// Request-response RPC call.
    Call {
        method: String,
        payload: Vec<u8>,
        response: oneshot::Sender<Result<Vec<u8>>>,
    },
    /// Fire-and-forget RPC notification.
    Notify {
        method: String,
        payload: Vec<u8>,
    },
    /// Returns the device's configuration.
    GetSpec {
        response: oneshot::Sender<DeviceSpec>,
    },
}

/// Per-device actor that owns the RPC connection.
struct Device {
    config: DeviceSpec,
    rpc: RpcClient,
}


impl ServiceHandler for Device {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Call { method, payload, response } => {
                let result = self.rpc.call(&method, payload).await;
                let _ = response.send(result);
            }
            Message::Notify { method, payload } => {
                if let Err(e) = self.rpc.notify(&method, payload) {
                    tracing::error!("Device notify '{}' failed: {e}", method);
                }
            }
            Message::GetSpec { response } => {
                let _ = response.send(self.config.clone());
            }
        }
    }
}



// =============================================================================
// RPC infrastructure for cross-process IPC communication.
//
// Provides both server and client types for IPC-based RPC:
//
// - `RpcServer`: Creates an `IpcOneShotServer`, accepts a client
//   connection, and exposes `poll()` / `respond()` for request handling.
//
// - `RpcClient`: Connects to an existing `IpcOneShotServer` and
//   exposes async `call()` / `notify()` for sending requests.
//
// In the inference pattern, Python wraps `RpcServer`
// and Rust connects via `RpcClient`.
// =============================================================================

/// Request message sent over IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcRequest {
    pub request_id: u64,
    pub method: String,
    pub payload: Vec<u8>,
    /// Wall-clock micros (UNIX epoch) when sender prepared this request.
    /// Default 0 means unset.
    #[serde(default)]
    pub send_walltime_us: u64,
}

/// Response message sent over IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponse {
    pub request_id: u64,
    pub payload: Vec<u8>,
    /// Wall-clock micros when worker dispatched the response.
    #[serde(default)]
    pub respond_walltime_us: u64,
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
        self.respond_with_ts(request_id, payload, 0)
    }

    /// Same as `respond` but allows the caller to embed a wall-clock micros
    /// timestamp that the client can use to measure backward IPC transit.
    pub fn respond_with_ts(
        &self,
        request_id: u64,
        payload: Vec<u8>,
        respond_walltime_us: u64,
    ) -> Result<()> {
        let tx = self.response_tx.lock().unwrap();
        tx.send(IpcResponse {
            request_id,
            payload,
            respond_walltime_us,
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

/// IPC client that connects to an `RpcServer`.
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

        // Blocking-recv router for cold-path RPCs (adapter ops, page copies,
        // query/ping). The hot-path `fire_batch` goes via shared memory and
        // never reaches this thread, so a busy-spin would just burn CPU.
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

    /// Send a request and await the response.
    pub async fn call(&self, method: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let send_walltime_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        let request = IpcRequest {
            request_id,
            method: method.to_string(),
            payload,
            send_walltime_us,
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

        response_rx
            .await
            .map_err(|_| anyhow!("Response channel closed"))
    }

    /// Fire-and-forget notification (no response expected).
    pub fn notify(&self, method: &str, payload: Vec<u8>) -> Result<()> {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = IpcRequest {
            request_id,
            method: method.to_string(),
            payload,
            send_walltime_us: 0,
        };

        let tx = self.request_tx.lock().unwrap();
        tx.send(request)?;
        Ok(())
    }

}
