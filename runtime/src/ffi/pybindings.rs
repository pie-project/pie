//! PyO3 bindings for Python interop.
//!
//! This module contains all Python-exposed types and functions,
//! including configuration, handles, and IPC RPC types.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;
use std::sync::Arc;

use crate::bootstrap::{
    AuthConfig, Config as BootstrapConfig, DeviceConfig as BootstrapDeviceConfig,
    ModelConfig as BootstrapModelConfig, SchedulerConfig as BootstrapSchedulerConfig,
    TelemetryConfig,
};
use crate::device::RpcServer as InternalRpcServer;

// =============================================================================
// RpcServer - Thin PyO3 wrapper around device::RpcServer
// =============================================================================

/// Python-hosted IPC server (thin wrapper around `RpcServer`).
///
/// Usage from Python:
/// ```python
/// server = RpcServer.create()
/// name = server.server_name()  # give this to Rust's RpcClient
/// while True:
///     req = server.poll_blocking(timeout_ms=1000)
///     if req is not None:
///         request_id, method, payload = req
///         result = handle(method, payload)
///         server.respond(request_id, result)
/// ```
#[pyclass(name = "RpcServer")]
pub struct PyRpcServer {
    inner: InternalRpcServer,
}

#[pymethods]
impl PyRpcServer {
    /// Create a new IPC server.
    #[staticmethod]
    fn create() -> PyResult<Self> {
        let inner = InternalRpcServer::create()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create RPC server: {}", e)))?;
        Ok(PyRpcServer { inner })
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
        let result = py.allow_threads(|| {
            self.inner
                .poll(std::time::Duration::from_millis(timeout_ms))
        });

        match result {
            Ok(Some(req)) => {
                let py_bytes = PyBytes::new(py, &req.payload).into();
                Ok(Some((req.request_id, req.method, py_bytes)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!("Poll error: {}", e))),
        }
    }

    /// Send a response back to Rust for the given request ID.
    fn respond(&self, request_id: u64, response: &[u8]) -> PyResult<bool> {
        self.inner
            .respond(request_id, response.to_vec())
            .map(|_| true)
            .map_err(|e| PyRuntimeError::new_err(format!("Respond error: {}", e)))
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
// Configuration Types
// =============================================================================

/// Scheduler configuration for a model.
#[pyclass(name = "SchedulerConfig")]
#[derive(Clone)]
pub struct SchedulerConfig {
    #[pyo3(get, set)]
    pub max_in_flight_batches: usize,
    #[pyo3(get, set)]
    pub request_timeout_secs: u64,
    #[pyo3(get, set)]
    pub max_wait_ms: u64,
    #[pyo3(get, set)]
    pub min_batch_for_optimization: usize,
}

#[pymethods]
impl SchedulerConfig {
    #[new]
    #[pyo3(signature = (
        max_in_flight_batches = 4,
        request_timeout_secs = 120,
        max_wait_ms = 50,
        min_batch_for_optimization = 8,
    ))]
    fn new(
        max_in_flight_batches: usize,
        request_timeout_secs: u64,
        max_wait_ms: u64,
        min_batch_for_optimization: usize,
    ) -> Self {
        SchedulerConfig {
            max_in_flight_batches,
            request_timeout_secs,
            max_wait_ms,
            min_batch_for_optimization,
        }
    }
}

#[pyclass(name = "DeviceConfig")]
#[derive(Clone)]
pub struct DeviceConfig {
    /// IPC server name from `RpcServer.server_name()`
    #[pyo3(get, set)]
    pub hostname: String,
    /// Total KV cache pages available on this device group
    #[pyo3(get, set)]
    pub total_pages: usize,
    /// Maximum batch tokens this device can handle
    #[pyo3(get, set)]
    pub max_batch_tokens: usize,
    /// Maximum batch size this device can handle
    #[pyo3(get, set)]
    pub max_batch_size: usize,
}

#[pymethods]
impl DeviceConfig {
    #[new]
    fn new(hostname: String, total_pages: usize, max_batch_tokens: usize, max_batch_size: usize) -> Self {
        DeviceConfig {
            hostname,
            total_pages,
            max_batch_tokens,
            max_batch_size,
        }
    }
}

#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
pub struct ModelConfig {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub chat_template: String,
    #[pyo3(get, set)]
    pub stop_tokens: Vec<u32>,
    #[pyo3(get, set)]
    pub kv_page_size: usize,
    #[pyo3(get, set)]
    pub tokenizer_path: String,
    #[pyo3(get, set)]
    pub devices: Vec<DeviceConfig>,
    #[pyo3(get, set)]
    pub scheduler: SchedulerConfig,
}

#[pymethods]
impl ModelConfig {
    #[new]
    #[pyo3(signature = (
        name,
        chat_template,
        stop_tokens,
        kv_page_size,
        tokenizer_path,
        devices,
        scheduler = None,
    ))]
    fn new(
        name: String,
        chat_template: String,
        stop_tokens: Vec<u32>,
        kv_page_size: usize,
        tokenizer_path: String,
        devices: Vec<DeviceConfig>,
        scheduler: Option<SchedulerConfig>,
    ) -> Self {
        ModelConfig {
            name,
            chat_template,
            stop_tokens,
            kv_page_size,
            tokenizer_path,
            devices,
            scheduler: scheduler.unwrap_or_else(|| SchedulerConfig::new(4, 120, 50, 8)),
        }
    }
}

/// Top-level server configuration exposed to Python.
/// Maps directly to `bootstrap::Config`.
#[pyclass]
#[derive(Clone)]
pub struct Config {
    #[pyo3(get, set)]
    pub host: String,
    #[pyo3(get, set)]
    pub port: u16,
    #[pyo3(get, set)]
    pub verbose: bool,
    #[pyo3(get, set)]
    pub registry: String,
    // Auth
    #[pyo3(get, set)]
    pub auth_enabled: bool,
    #[pyo3(get, set)]
    pub auth_dir: String,
    // Program/cache directory
    #[pyo3(get, set)]
    pub program_dir: String,
    // Log directory
    #[pyo3(get, set)]
    pub log_dir: String,
    // Telemetry
    #[pyo3(get, set)]
    pub telemetry_enabled: bool,
    #[pyo3(get, set)]
    pub telemetry_endpoint: String,
    #[pyo3(get, set)]
    pub telemetry_service_name: String,
    // Models
    #[pyo3(get, set)]
    pub models: Vec<ModelConfig>,
}

#[pymethods]
impl Config {
    #[new]
    #[pyo3(signature = (
        host,
        port,
        verbose = false,
        registry = "https://registry.pie-project.org/".to_string(),
        auth_enabled = false,
        auth_dir = "".to_string(),
        program_dir = "".to_string(),
        log_dir = "".to_string(),
        telemetry_enabled = false,
        telemetry_endpoint = "http://localhost:4317".to_string(),
        telemetry_service_name = "pie".to_string(),
        models = vec![],
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        host: String,
        port: u16,
        verbose: bool,
        registry: String,
        auth_enabled: bool,
        auth_dir: String,
        program_dir: String,
        log_dir: String,
        telemetry_enabled: bool,
        telemetry_endpoint: String,
        telemetry_service_name: String,
        models: Vec<ModelConfig>,
    ) -> Self {
        Config {
            host,
            port,
            verbose,
            registry,
            auth_enabled,
            auth_dir,
            program_dir,
            log_dir,
            telemetry_enabled,
            telemetry_endpoint,
            telemetry_service_name,
            models,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(host='{}', port={}, verbose={}, models={})",
            self.host,
            self.port,
            self.verbose,
            self.models.len()
        )
    }
}

impl From<Config> for BootstrapConfig {
    fn from(cfg: Config) -> Self {
        BootstrapConfig {
            host: cfg.host,
            port: cfg.port,
            auth: AuthConfig {
                enabled: cfg.auth_enabled,
                authorized_users_dir: PathBuf::from(cfg.auth_dir),
            },
            cache_dir: PathBuf::from(cfg.program_dir),
            verbose: cfg.verbose,
            log_dir: if cfg.log_dir.is_empty() {
                None
            } else {
                Some(PathBuf::from(cfg.log_dir))
            },
            registry_url: cfg.registry,
            telemetry: TelemetryConfig {
                enabled: cfg.telemetry_enabled,
                endpoint: cfg.telemetry_endpoint,
                service_name: cfg.telemetry_service_name,
            },
            models: cfg
                .models
                .into_iter()
                .map(|m| BootstrapModelConfig {
                    name: m.name,
                    chat_template: m.chat_template,
                    stop_tokens: m.stop_tokens,
                    kv_page_size: m.kv_page_size,
                    tokenizer_path: PathBuf::from(m.tokenizer_path),
                    devices: m
                        .devices
                        .into_iter()
                        .map(|d| BootstrapDeviceConfig {
                            hostname: d.hostname,
                            total_pages: d.total_pages,
                            max_batch_tokens: d.max_batch_tokens,
                            max_batch_size: d.max_batch_size,
                        })
                        .collect(),
                    scheduler: BootstrapSchedulerConfig {
                        max_in_flight_batches: m.scheduler.max_in_flight_batches,
                        request_timeout_secs: m.scheduler.request_timeout_secs,
                        max_wait_ms: m.scheduler.max_wait_ms,
                        min_batch_for_optimization: m.scheduler.min_batch_for_optimization,
                    },
                })
                .collect(),
            skip_tracing: false,
        }
    }
}

// =============================================================================
// Runtime Handle
// =============================================================================

/// Handle to a running Pie runtime.
///
/// Holds the internal auth token and the tokio runtime that
/// keeps background services alive.
#[pyclass]
pub struct RuntimeHandle {
    internal_token: String,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl RuntimeHandle {
    /// Get the internal authentication token.
    #[getter]
    fn internal_token(&self) -> String {
        self.internal_token.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RuntimeHandle(token={}...)",
            &self.internal_token[..self.internal_token.len().min(8)]
        )
    }
}

// =============================================================================
// Bootstrap
// =============================================================================

/// Bootstrap the Pie runtime with the given configuration.
///
/// This creates the tokio runtime, initializes all services (auth, program
/// manager, linker, server, models, devices, schedulers), and returns a
/// `RuntimeHandle` that keeps everything alive.
///
/// Call this AFTER Python workers have been spawned and their RPC servers
/// are ready. The `Config.models` should include `DeviceConfig` entries
/// with the `hostname` from each worker's `RpcServer.server_name()`.
#[pyfunction]
#[pyo3(name = "bootstrap")]
fn py_bootstrap(py: Python<'_>, config: Config) -> PyResult<RuntimeHandle> {
    py.allow_threads(|| {
        let rt = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?,
        );

        let bootstrap_config: BootstrapConfig = config.into();

        let internal_token = rt.block_on(async {
            crate::bootstrap::bootstrap(bootstrap_config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Bootstrap failed: {}", e)))
        })?;

        Ok(RuntimeHandle {
            internal_token,
            runtime: rt,
        })
    })
}

// =============================================================================
// Python Module
// =============================================================================

/// Python module definition for pie_runtime
#[pymodule]
pub fn pie_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<DeviceConfig>()?;
    m.add_class::<SchedulerConfig>()?;
    m.add_class::<RuntimeHandle>()?;
    m.add_class::<PyRpcServer>()?;
    m.add_function(wrap_pyfunction!(py_bootstrap, m)?)?;
    Ok(())
}
