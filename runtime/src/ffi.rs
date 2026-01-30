//! FFI module containing all PyO3 bindings.
//!
//! This module serves as the boundary between Rust and Python.
//! All Python-exposed types and functions are defined here.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

use crate::auth::AuthorizedUsers;
use crate::engine::{self, Config as EngineConfig};
use crate::telemetry::TelemetryConfig;

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
        crate::legacy_model::registered_models()
    }

    fn __repr__(&self) -> String {
        let running = if self.is_running() { "running" } else { "stopped" };
        format!("ServerHandle(status={}, token={}...)", running, &self.internal_token[..8])
    }
}

/// Initialize the Python backend in-process and return a dispatcher.
///
/// This creates a Runtime and dispatcher for direct FFI calls, enabling\n/// low-latency communication with Python.
///
/// Args:
///     model_config: Dictionary containing model configuration
///
/// Returns:
///     A Runtime instance for use with the FFI queue worker
#[pyfunction]
fn initialize_backend(py: Python<'_>, model_config: &Bound<'_, pyo3::types::PyDict>) -> PyResult<PyObject> {
    // Import pie_worker modules
    let config_mod = py.import("pie_worker.config")?;
    let runtime_mod = py.import("pie_worker.runtime")?;
    
    // Create RuntimeConfig using from_args with **kwargs
    let config_cls = config_mod.getattr("RuntimeConfig")?;
    let from_args = config_cls.getattr("from_args")?;
    let config = from_args.call((), Some(model_config))?;
    
    // Create and return Runtime instance directly
    let runtime_cls = runtime_mod.getattr("Runtime")?;
    let runtime = runtime_cls.call1((config,))?;
    
    Ok(runtime.unbind())
}


/// Starts the PIE server and returns a handle for management.
///
/// This function blocks until the server is ready, then returns a ServerHandle
/// that can be used for graceful shutdown and status checks.
///
/// Args:
///     config: Server configuration
///     authorized_users_path: Optional path to authorized_users.toml
///
/// Returns:
///     A ServerHandle object with the internal token and shutdown capability.
#[pyfunction]
#[pyo3(signature = (config, authorized_users_path = None))]
fn start_server(
    py: Python<'_>,
    config: ServerConfig,
    authorized_users_path: Option<String>,
) -> PyResult<ServerHandle> {
    // Allow other Python threads to run while we block
    py.allow_threads(|| {
        let rt = Arc::new(tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?);

        let internal_token = rt.block_on(async {
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

            // Wait for server to be ready and get the internal auth token
            let internal_token = ready_rx.await.map_err(|e| {
                PyRuntimeError::new_err(format!("Server failed to start: {}", e))
            })?;

            Ok::<(String, oneshot::Sender<()>), PyErr>((internal_token, shutdown_tx))
        })?;

        let (token, shutdown_tx) = internal_token;

        Ok(ServerHandle {
            internal_token: token,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            runtime: rt,
        })
    })
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
    backends: Arc<Mutex<Option<Vec<crate::legacy_model::RpcBackend>>>>,
    /// Scheduler config for model creation
    scheduler_config: crate::legacy_model::SchedulerConfig,
}

#[pymethods]
impl PartialServerHandle {
    /// Complete initialization by performing handshake and registering the model.
    ///
    /// This should be called after Python IPC workers have connected.
    /// Blocks until handshake completes successfully.
    fn complete(&self, py: Python<'_>) -> PyResult<ServerHandle> {
        use crate::legacy_model::{self, Model};
        
        py.allow_threads(|| {
            let backends = {
                let mut guard = self.backends.lock().map_err(|e| {
                    PyRuntimeError::new_err(format!("Lock poisoned: {}", e))
                })?;
                guard.take().ok_or_else(|| {
                    PyRuntimeError::new_err("complete() already called")
                })?
            };
            
            let scheduler_config = self.scheduler_config.clone();
            
            self.runtime.block_on(async {
                // Now Python workers are connected, handshake will succeed
                let model = Model::new_with_backends(backends, scheduler_config).await
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create model: {}", e)))?;
                
                let model_name = model.name().to_string();
                
                // Register the model with the engine
                let model_id = legacy_model::install_model(model_name.clone(), model);
                if model_id.is_none() {
                    return Err(PyRuntimeError::new_err(format!(
                        "Failed to register model '{}': already exists", model_name
                    )));
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
    use crate::legacy_model::{SchedulerConfig, RpcBackend};
    use crate::legacy_model::ffi_bridge::AsyncIpcClient;
    use crate::legacy_model::ffi_ipc::FfiIpcBackend;
    
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
                
                let ipc_client = AsyncIpcClient::new(std::sync::Arc::new(ipc_backend));
                backends.push(RpcBackend::new_ipc(ipc_client));
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
            scheduler_config: SchedulerConfig::default(),
        };
        
        Ok((handle, server_names))
    })
}


/// Python module definition for _pie
#[pymodule]
pub fn _pie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ServerConfig>()?;
    m.add_class::<ServerHandle>()?;
    m.add_class::<PartialServerHandle>()?;
    m.add_class::<crate::legacy_model::ffi_ipc::FfiIpcQueue>()?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_function(wrap_pyfunction!(start_server_phase1, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_backend, m)?)?;
    Ok(())
}

