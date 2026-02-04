//! PyO3 bindings for Python interop.
//!
//! This module contains all Python-exposed types and functions,
//! including server configuration, handles, and IPC queue types.

use anyhow::Result;
use ipc_channel::ipc::{self, IpcReceiver, IpcSender};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

use crate::auth::AuthorizedUsers;
use crate::engine::{self, Config as EngineConfig};
use crate::telemetry::TelemetryConfig;
use super::rpc::{FfiIpcBackend, IpcChannels, IpcRequest, IpcResponse, AsyncIpcClient, RpcBackend};

// =============================================================================
// FfiIpcQueue - Python side IPC client (PyO3)
// =============================================================================

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

// =============================================================================
// Server Configuration
// =============================================================================

/// Configuration for the PIE server, exposed to Python.
#[pyclass]
#[derive(Clone)]
pub struct ServerConfig {
    #[pyo3(get, set)]
    pub host: String,
    #[pyo3(get, set)]
    pub port: u16,
    #[pyo3(get, set)]
    pub enable_auth: bool,
    #[pyo3(get, set)]
    pub cache_dir: String,
    #[pyo3(get, set)]
    pub verbose: bool,
    #[pyo3(get, set)]
    pub log_dir: Option<String>,
    #[pyo3(get, set)]
    pub registry: String,
    // Telemetry configuration
    #[pyo3(get, set)]
    pub telemetry_enabled: bool,
    #[pyo3(get, set)]
    pub telemetry_endpoint: String,
    #[pyo3(get, set)]
    pub telemetry_service_name: String,
}

#[pymethods]
impl ServerConfig {
    #[new]
    #[pyo3(signature = (host, port, enable_auth, cache_dir, verbose, log_dir, registry, telemetry_enabled=false, telemetry_endpoint="http://localhost:4317".to_string(), telemetry_service_name="pie-runtime".to_string()))]
    fn new(
        host: String,
        port: u16,
        enable_auth: bool,
        cache_dir: String,
        verbose: bool,
        log_dir: String,
        registry: String,
        telemetry_enabled: bool,
        telemetry_endpoint: String,
        telemetry_service_name: String,
    ) -> Self {
        ServerConfig {
            host,
            port,
            enable_auth,
            cache_dir,
            verbose,
            log_dir: Some(log_dir),
            registry,
            telemetry_enabled,
            telemetry_endpoint,
            telemetry_service_name,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ServerConfig(host='{}', port={}, enable_auth={}, cache_dir='{}', verbose={}, log_dir={:?}, registry='{}', telemetry_enabled={})",
            self.host, self.port, self.enable_auth, self.cache_dir, self.verbose, self.log_dir, self.registry, self.telemetry_enabled
        )
    }
}

impl From<ServerConfig> for EngineConfig {
    fn from(cfg: ServerConfig) -> Self {
        EngineConfig {
            host: cfg.host,
            port: cfg.port,
            enable_auth: cfg.enable_auth,
            cache_dir: PathBuf::from(cfg.cache_dir),
            verbose: cfg.verbose,
            log_dir: cfg.log_dir.map(PathBuf::from),
            registry: cfg.registry,
            telemetry: TelemetryConfig {
                enabled: cfg.telemetry_enabled,
                endpoint: cfg.telemetry_endpoint,
                service_name: cfg.telemetry_service_name,
            },
        }
    }
}

// =============================================================================
// Server Handles
// =============================================================================

/// Handle to a running server, allowing graceful shutdown.
#[pyclass]
pub struct ServerHandle {
    internal_token: String,
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl ServerHandle {
    /// Get the internal authentication token.
    #[getter]
    fn internal_token(&self) -> String {
        self.internal_token.clone()
    }

    /// Gracefully shut down the server.
    fn shutdown(&self) -> PyResult<()> {
        let mut guard = self.shutdown_tx.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire lock: {}", e))
        })?;

        if let Some(tx) = guard.take() {
            tx.send(()).map_err(|_| {
                PyRuntimeError::new_err("Failed to send shutdown signal (server may already be stopped)")
            })?;
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Server already shut down"))
        }
    }

    /// Check if the server is still running.
    fn is_running(&self) -> bool {
        self.shutdown_tx.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Get list of registered model names (backends that have connected).
    fn registered_models(&self) -> Vec<String> {
        crate::model::registered_models()
    }

    fn __repr__(&self) -> String {
        let running = if self.is_running() { "running" } else { "stopped" };
        format!("ServerHandle(status={}, token={}...)", running, &self.internal_token[..8])
    }
}

/// Partial handle for two-phase server initialization.
///
/// Phase 1 creates IPC backends and returns this handle + server names.
/// Python spawns worker threads and connects to the IPC servers.
/// Phase 2 calls `complete()` to perform handshake and register the model.
#[pyclass]
pub struct PartialServerHandle {
    internal_token: String,
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    runtime: Arc<tokio::runtime::Runtime>,
    /// IPC backends (not yet handshaked)
    backends: Arc<Mutex<Option<Vec<RpcBackend>>>>,
}

#[pymethods]
impl PartialServerHandle {
    /// Complete initialization by performing handshake and registering the model.
    ///
    /// This should be called after Python IPC workers have connected.
    /// Blocks until handshake completes successfully.
    fn complete(&self, py: Python<'_>) -> PyResult<ServerHandle> {
        use crate::model;
        
        py.allow_threads(|| {
            let backends = {
                let mut guard = self.backends.lock().map_err(|e| {
                    PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
                })?;
                guard.take().ok_or_else(|| {
                    PyRuntimeError::new_err("complete() already called")
                })?
            };
            
            self.runtime.block_on(async {
                // Install each backend as a model
                for backend in backends {
                    let model_id = model::install_model_with_backend(backend).await
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to install model: {}", e)))?;
                    tracing::info!("Installed model with ID: {}", model_id);
                }
                
                Ok::<(), PyErr>(())
            })?;
            
            Ok(ServerHandle {
                internal_token: self.internal_token.clone(),
                shutdown_tx: Arc::clone(&self.shutdown_tx),
                runtime: Arc::clone(&self.runtime),
            })
        })
    }
    
    /// Get the internal authentication token.
    #[getter]
    fn internal_token(&self) -> String {
        self.internal_token.clone()
    }
    
    fn __repr__(&self) -> String {
        format!("PartialServerHandle(token={}...)", &self.internal_token[..8])
    }
}

// =============================================================================
// Phase 1 Server Initialization
// =============================================================================

/// Phase 1: Start the Pie server and create IPC backends WITHOUT handshaking.
///
/// This returns immediately after creating IPC backends, allowing Python to
/// spawn worker threads and connect to the IPC servers before Phase 2.
///
/// Args:
///     config: ServerConfig with host/port settings
///     authorized_users_path: Optional path to authorized users file
///     num_groups: Total number of groups (all will use IPC)
///
/// Returns:
///     Tuple of (PartialServerHandle, list of IPC server names for all groups)
#[pyfunction]
#[pyo3(signature = (config, authorized_users_path, num_groups))]
fn start_server_phase1(
    py: Python<'_>,
    config: ServerConfig,
    authorized_users_path: Option<String>,
    num_groups: usize,
) -> PyResult<(PartialServerHandle, Vec<String>)> {
    
    // Allow other Python threads to run while we block
    py.allow_threads(|| {
        let rt = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?);

        let result = rt.block_on(async {
            let authorized_users = match authorized_users_path {
                Some(path) => AuthorizedUsers::load(&PathBuf::from(path)).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load authorized users: {}", e))
                })?,
                None => AuthorizedUsers::default(),
            };

            let (ready_tx, ready_rx) = oneshot::channel();
            let (shutdown_tx, shutdown_rx) = oneshot::channel();

            let engine_config: EngineConfig = config.into();

            // Spawn the server in a background task
            tokio::spawn(async move {
                if let Err(e) = engine::run_server(engine_config, authorized_users, ready_tx, shutdown_rx).await {
                    eprintln!("[PIE] Server error: {}", e);
                }
            });

            // Wait for server to be ready
            let internal_token = ready_rx.await.map_err(|e| {
                PyRuntimeError::new_err(format!("Server failed to start: {}", e))
            })?;

            // Create IPC backends for ALL groups (no handshake yet - just channel setup)
            let mut backends = Vec::with_capacity(num_groups);
            let mut ipc_server_names = Vec::with_capacity(num_groups);
            
            for group_id in 0..num_groups {
                let (ipc_backend, server_name) = FfiIpcBackend::new_with_handshake(group_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create IPC backend for group {}: {}", group_id, e)))?;
                
                let ipc_client = AsyncIpcClient::new(Arc::new(ipc_backend));
                backends.push(RpcBackend::new(ipc_client));
                ipc_server_names.push(server_name);
            }

            Ok::<(String, oneshot::Sender<()>, Vec<RpcBackend>, Vec<String>), PyErr>(
                (internal_token, shutdown_tx, backends, ipc_server_names)
            )
        })?;

        let (token, shutdown_tx, backends, server_names) = result;

        let handle = PartialServerHandle {
            internal_token: token,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            runtime: rt,
            backends: Arc::new(Mutex::new(Some(backends))),
        };
        
        Ok((handle, server_names))
    })
}

// =============================================================================
// Python Module
// =============================================================================

/// Python module definition for _pie
#[pymodule]
pub fn _pie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ServerConfig>()?;
    m.add_class::<ServerHandle>()?;
    m.add_class::<PartialServerHandle>()?;
    m.add_class::<FfiIpcQueue>()?;
    m.add_function(wrap_pyfunction!(start_server_phase1, m)?)?;
    Ok(())
}
