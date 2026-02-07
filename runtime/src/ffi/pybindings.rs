//! PyO3 bindings for Python interop.
//!
//! This module contains all Python-exposed types and functions,
//! including server configuration, handles, and IPC RPC types.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

use crate::auth;
use crate::bootstrap::{self, Config as BootstrapConfig};
use crate::inference::rpc::{RpcClient, RpcServer};
use crate::bootstrap::{TelemetryConfig, AuthConfig as BootstrapAuthConfig};

// =============================================================================
// PyRpcServer - Thin PyO3 wrapper around RpcServer
// =============================================================================

/// Python-hosted IPC server (thin wrapper around `RpcServer`).
///
/// Usage from Python:
/// ```python
/// server = PyRpcServer.create()
/// name = server.server_name()  # give this to Rust's RpcClient
/// while True:
///     req = server.poll_blocking(timeout_ms=1000)
///     if req is not None:
///         request_id, method, payload = req
///         result = handle(method, payload)
///         server.respond(request_id, result)
/// ```
#[pyclass]
pub struct PyRpcServer {
    inner: RpcServer,
}

#[pymethods]
impl PyRpcServer {
    /// Create a new IPC server.
    #[staticmethod]
    fn create() -> PyResult<Self> {
        let inner = RpcServer::create()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        Ok(Self { inner })
    }

    /// Get the server name for Rust clients to connect to.
    fn server_name(&self) -> String {
        self.inner.server_name().to_string()
    }

    /// Poll for the next request from Rust (blocking with timeout).
    ///
    /// Returns `(request_id, method_name, payload_bytes)` or `None` on timeout.
    fn poll_blocking(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Option<(u64, String, Py<PyBytes>)>> {
        let timeout = std::time::Duration::from_millis(timeout_ms);

        // Release GIL while waiting
        let result = py.allow_threads(|| {
            self.inner.poll(timeout)
        });

        match result {
            Ok(Some(request)) => {
                let py_bytes = PyBytes::new(py, &request.payload).into();
                Ok(Some((request.request_id, request.method, py_bytes)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("{:?}", e))),
        }
    }

    /// Send a response back to Rust for the given request ID.
    fn respond(&self, request_id: u64, response: &[u8]) -> PyResult<bool> {
        self.inner
            .respond(request_id, response.to_vec())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
        Ok(true)
    }

    /// Close the server.
    fn close(&self) {
        self.inner.close();
    }

    /// Check if the server is closed.
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
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

impl From<ServerConfig> for BootstrapConfig {
    fn from(cfg: ServerConfig) -> Self {
        BootstrapConfig {
            host: cfg.host,
            port: cfg.port,
            auth: BootstrapAuthConfig {
                enabled: cfg.enable_auth,
                authorized_users_dir: PathBuf::new(),
            },
            cache_dir: PathBuf::from(cfg.cache_dir),
            verbose: cfg.verbose,
            log_dir: cfg.log_dir.map(PathBuf::from),
            registry_url: cfg.registry,
            telemetry: TelemetryConfig {
                enabled: cfg.telemetry_enabled,
                endpoint: cfg.telemetry_endpoint,
                service_name: cfg.telemetry_service_name,
            },
            models: vec![],
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
                PyRuntimeError::new_err(
                    "Failed to send shutdown signal (server may already be stopped)",
                )
            })?;
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Server already shut down"))
        }
    }

    /// Check if the server is still running.
    fn is_running(&self) -> bool {
        self.shutdown_tx
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    /// Get list of registered model names (backends that have connected).
    fn registered_models(&self) -> Vec<String> {
        crate::model::registered_models()
    }

    fn __repr__(&self) -> String {
        let running = if self.is_running() {
            "running"
        } else {
            "stopped"
        };
        format!(
            "ServerHandle(status={}, token={}...)",
            running,
            &self.internal_token[..8]
        )
    }
}

/// Partial handle for two-phase server initialization.
///
/// Phase 1 starts the server and returns this handle.
/// Python creates `PyRpcServer` instances and passes the server names.
/// Phase 2 calls `complete()` with the server names — Rust connects via `RpcClient`.
#[pyclass]
pub struct PartialServerHandle {
    internal_token: String,
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PartialServerHandle {
    /// Complete initialization by connecting to Python-hosted RPC servers.
    ///
    /// Python should create `PyRpcServer` instances first, then pass the
    /// server names here. Rust connects via `RpcClient` and performs
    /// the handshake to register each model.
    fn complete(&self, py: Python<'_>, server_names: Vec<String>) -> PyResult<ServerHandle> {
        use crate::model;

        py.allow_threads(|| {
            self.runtime.block_on(async {
                for server_name in server_names {
                    let client = RpcClient::connect(&server_name).map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to connect to RPC server '{}': {}",
                            server_name, e
                        ))
                    })?;

                    let model_id = model::install_model_with_backend(client)
                        .await
                        .map_err(|e| {
                            PyRuntimeError::new_err(format!("Failed to install model: {}", e))
                        })?;
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
        format!(
            "PartialServerHandle(token={}...)",
            &self.internal_token[..8]
        )
    }
}

// =============================================================================
// Phase 1 Server Initialization
// =============================================================================

/// Phase 1: Start the Pie server.
///
/// Returns a `PartialServerHandle`. Python then creates `PyRpcServer`
/// instances and calls `handle.complete(server_names)` to finalize.
#[pyfunction]
#[pyo3(signature = (config, authorized_users_path))]
fn start_server_phase1(
    py: Python<'_>,
    config: ServerConfig,
    authorized_users_path: Option<String>,
) -> PyResult<PartialServerHandle> {
    py.allow_threads(|| {
        let rt = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?,
        );

        let result = rt.block_on(async {
            let (shutdown_tx, shutdown_rx) = oneshot::channel();

            let mut bootstrap_config: BootstrapConfig = config.into();

            // Set the authorized users path if provided
            if let Some(path) = authorized_users_path {
                bootstrap_config.auth.authorized_users_dir = PathBuf::from(path);
            }

            let internal_token = bootstrap::bootstrap(bootstrap_config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Server failed to start: {}", e)))?;

            Ok::<(String, oneshot::Sender<()>), PyErr>((internal_token, shutdown_tx))
        })?;

        let (token, shutdown_tx) = result;

        Ok(PartialServerHandle {
            internal_token: token,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            runtime: rt,
        })
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
    m.add_class::<PyRpcServer>()?;
    m.add_function(wrap_pyfunction!(start_server_phase1, m)?)?;
    Ok(())
}
