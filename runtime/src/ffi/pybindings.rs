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
    ModelConfig as BootstrapModelConfig, RuntimeConfig as BootstrapRuntimeConfig,
    SchedulerConfig as BootstrapSchedulerConfig, TelemetryConfig,
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
#[pyclass]
pub struct RpcServer {
    inner: InternalRpcServer,
}

#[pymethods]
impl RpcServer {
    /// Create a new IPC server.
    #[staticmethod]
    fn create() -> PyResult<Self> {
        let inner = InternalRpcServer::create()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create RPC server: {}", e)))?;
        Ok(RpcServer { inner })
    }

    /// Get the server name for Rust clients to connect to.
    fn server_name(&self) -> String {
        self.inner.server_name().to_string()
    }

    /// Poll for the next request from Rust (blocking with timeout).
    ///
    /// Returns `(request_id, method_name, payload_bytes, send_walltime_us)`
    /// or `None` on timeout.
    fn poll_blocking(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Option<(u64, String, Py<PyBytes>, u64)>> {
        let result = py.allow_threads(|| {
            self.inner
                .poll(std::time::Duration::from_millis(timeout_ms))
        });

        match result {
            Ok(Some(req)) => {
                let py_bytes = PyBytes::new(py, &req.payload).into();
                Ok(Some((req.request_id, req.method, py_bytes, req.send_walltime_us)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!("Poll error: {}", e))),
        }
    }

    /// Busy-spin for up to `spin_us` microseconds doing zero-timeout polls,
    /// then fall back to a blocking poll with `fallback_timeout_ms`.
    /// Eliminates cross-process wakeup latency on the hot path.
    fn poll_busy(
        &self,
        py: Python<'_>,
        spin_us: u64,
        fallback_timeout_ms: u64,
    ) -> PyResult<Option<(u64, String, Py<PyBytes>, u64)>> {
        let result = py.allow_threads(|| -> Result<_, _> {
            let deadline = std::time::Instant::now() + std::time::Duration::from_micros(spin_us);
            loop {
                match self.inner.poll(std::time::Duration::ZERO) {
                    Ok(Some(req)) => return Ok::<_, anyhow::Error>(Some(req)),
                    Ok(None) => {
                        if std::time::Instant::now() >= deadline {
                            return self
                                .inner
                                .poll(std::time::Duration::from_millis(fallback_timeout_ms));
                        }
                        std::hint::spin_loop();
                    }
                    Err(e) => return Err(e),
                }
            }
        });

        match result {
            Ok(Some(req)) => {
                let py_bytes = PyBytes::new(py, &req.payload).into();
                Ok(Some((req.request_id, req.method, py_bytes, req.send_walltime_us)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!("Poll error: {}", e))),
        }
    }

    /// Block for the first request, then drain up to `max_n - 1` additional
    /// pending requests non-blocking. Reduces syscall + GIL overhead vs
    /// calling `poll_blocking` once per request.
    fn drain_blocking(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
        max_n: usize,
    ) -> PyResult<Vec<(u64, String, Py<PyBytes>, u64)>> {
        let first = py.allow_threads(|| {
            self.inner.poll(std::time::Duration::from_millis(timeout_ms))
        });
        let mut out: Vec<(u64, String, Py<PyBytes>, u64)> = Vec::new();
        match first {
            Ok(Some(req)) => {
                let py_bytes = PyBytes::new(py, &req.payload).into();
                out.push((req.request_id, req.method, py_bytes, req.send_walltime_us));
            }
            Ok(None) => return Ok(out),
            Err(e) => return Err(PyRuntimeError::new_err(format!("Poll error: {}", e))),
        }
        // Drain whatever's already buffered.
        while out.len() < max_n {
            let nb = py.allow_threads(|| self.inner.poll(std::time::Duration::from_millis(0)));
            match nb {
                Ok(Some(req)) => {
                    let py_bytes = PyBytes::new(py, &req.payload).into();
                    out.push((req.request_id, req.method, py_bytes, req.send_walltime_us));
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
        Ok(out)
    }

    /// Send a response back to Rust for the given request ID.
    /// `respond_walltime_us` is the wall-clock micros at the time of dispatch
    /// (used for IPC transit profiling); pass 0 to disable.
    #[pyo3(signature = (request_id, response, respond_walltime_us = 0))]
    fn respond(&self, request_id: u64, response: &[u8], respond_walltime_us: u64) -> PyResult<bool> {
        self.inner
            .respond_with_ts(request_id, response.to_vec(), respond_walltime_us)
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

/// Runtime tuning — tokio worker pool + wasmtime engine pool +
/// per-instance security policies.
///
/// Every field is required from Python. Python (`pie.config.RuntimeConfig`)
/// is the source of truth for defaults; this pyclass is a transport.
#[pyclass(name = "RuntimeConfig")]
#[derive(Clone)]
pub struct RuntimeConfig {
    #[pyo3(get, set)] pub worker_threads: usize,
    #[pyo3(get, set)] pub wasm_max_instances: u32,
    #[pyo3(get, set)] pub wasm_max_memory_mb: usize,
    #[pyo3(get, set)] pub wasm_warm_memory_mb: usize,
    #[pyo3(get, set)] pub wasm_warm_slots: u32,
    #[pyo3(get, set)] pub allow_fs: bool,
    #[pyo3(get, set)] pub fs_scratch_dir: String,
    #[pyo3(get, set)] pub allow_network: bool,
    #[pyo3(get, set)] pub network_allowed_hosts: Vec<String>,
    #[pyo3(get, set)] pub max_upload_mb: usize,
}

#[pymethods]
impl RuntimeConfig {
    #[new]
    #[pyo3(signature = (
        worker_threads,
        wasm_max_instances,
        wasm_max_memory_mb,
        wasm_warm_memory_mb,
        wasm_warm_slots,
        allow_fs,
        fs_scratch_dir,
        allow_network,
        network_allowed_hosts,
        max_upload_mb,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_threads: usize,
        wasm_max_instances: u32,
        wasm_max_memory_mb: usize,
        wasm_warm_memory_mb: usize,
        wasm_warm_slots: u32,
        allow_fs: bool,
        fs_scratch_dir: String,
        allow_network: bool,
        network_allowed_hosts: Vec<String>,
        max_upload_mb: usize,
    ) -> Self {
        RuntimeConfig {
            worker_threads,
            wasm_max_instances,
            wasm_max_memory_mb,
            wasm_warm_memory_mb,
            wasm_warm_slots,
            allow_fs,
            fs_scratch_dir,
            allow_network,
            network_allowed_hosts,
            max_upload_mb,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RuntimeConfig(worker_threads={:?}, wasm_max_instances={:?}, \
             wasm_max_memory_mb={:?}, wasm_warm_memory_mb={:?}, \
             wasm_warm_slots={:?}, allow_fs={}, fs_scratch_dir={:?}, \
             allow_network={}, network_allowed_hosts={:?}, \
             max_upload_mb={:?})",
            self.worker_threads,
            self.wasm_max_instances,
            self.wasm_max_memory_mb,
            self.wasm_warm_memory_mb,
            self.wasm_warm_slots,
            self.allow_fs,
            self.fs_scratch_dir,
            self.allow_network,
            self.network_allowed_hosts,
            self.max_upload_mb,
        )
    }
}

/// Scheduler configuration for a model — batch policy + per-process
/// admission/market knobs. Every field is required from Python; Python
/// (`pie.config.SchedulerConfig`) is the source of truth for defaults.
#[pyclass(name = "SchedulerConfig")]
#[derive(Clone)]
pub struct SchedulerConfig {
    #[pyo3(get, set)] pub batch_policy: String,
    #[pyo3(get, set)] pub request_timeout_secs: u64,
    #[pyo3(get, set)] pub default_token_limit: Option<usize>,
    #[pyo3(get, set)] pub default_endowment_pages: usize,
    #[pyo3(get, set)] pub admission_oversubscription_factor: f64,
    #[pyo3(get, set)] pub restore_pause_at_utilization: f64,
}

#[pymethods]
impl SchedulerConfig {
    #[new]
    #[pyo3(signature = (
        batch_policy,
        request_timeout_secs,
        default_token_limit,
        default_endowment_pages,
        admission_oversubscription_factor,
        restore_pause_at_utilization,
    ))]
    fn new(
        batch_policy: String,
        request_timeout_secs: u64,
        default_token_limit: Option<usize>,
        default_endowment_pages: usize,
        admission_oversubscription_factor: f64,
        restore_pause_at_utilization: f64,
    ) -> Self {
        SchedulerConfig {
            batch_policy,
            request_timeout_secs,
            default_token_limit,
            default_endowment_pages,
            admission_oversubscription_factor,
            restore_pause_at_utilization,
        }
    }
}

/// Per-device capabilities published by the worker handshake.
/// `hostname` is the IPC server name from `RpcServer.server_name()`;
/// the rest are the device's own admission/batch limits.
#[pyclass(name = "DeviceConfig")]
#[derive(Clone)]
pub struct DeviceConfig {
    #[pyo3(get, set)] pub hostname: String,
    #[pyo3(get, set)] pub total_pages: usize,
    #[pyo3(get, set)] pub cpu_pages: usize,
    #[pyo3(get, set)] pub max_batch_tokens: usize,
    #[pyo3(get, set)] pub max_batch_size: usize,
}

#[pymethods]
impl DeviceConfig {
    #[new]
    #[pyo3(signature = (hostname, total_pages, max_batch_tokens, max_batch_size, cpu_pages = 0))]
    fn new(hostname: String, total_pages: usize, max_batch_tokens: usize, max_batch_size: usize, cpu_pages: usize) -> Self {
        DeviceConfig {
            hostname,
            total_pages,
            cpu_pages,
            max_batch_tokens,
            max_batch_size,
        }
    }
}

#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
pub struct ModelConfig {
    #[pyo3(get, set)] pub name: String,
    #[pyo3(get, set)] pub arch_name: String,
    #[pyo3(get, set)] pub kv_page_size: usize,
    #[pyo3(get, set)] pub tokenizer_path: String,
    #[pyo3(get, set)] pub devices: Vec<DeviceConfig>,
    #[pyo3(get, set)] pub scheduler: SchedulerConfig,
}

#[pymethods]
impl ModelConfig {
    #[new]
    #[pyo3(signature = (
        name,
        arch_name,
        kv_page_size,
        tokenizer_path,
        devices,
        scheduler,
    ))]
    fn new(
        name: String,
        arch_name: String,
        kv_page_size: usize,
        tokenizer_path: String,
        devices: Vec<DeviceConfig>,
        scheduler: SchedulerConfig,
    ) -> Self {
        ModelConfig {
            name,
            arch_name,
            kv_page_size,
            tokenizer_path,
            devices,
            scheduler,
        }
    }
}

/// Top-level server configuration exposed to Python — pure transport.
/// Every field is required; Python (`pie.config.Config`) is the source
/// of truth for defaults. Maps directly to `bootstrap::Config`.
#[pyclass]
#[derive(Clone)]
pub struct Config {
    #[pyo3(get, set)] pub host: String,
    #[pyo3(get, set)] pub port: u16,
    #[pyo3(get, set)] pub verbose: bool,
    #[pyo3(get, set)] pub registry: String,
    #[pyo3(get, set)] pub auth_enabled: bool,
    #[pyo3(get, set)] pub auth_dir: String,
    #[pyo3(get, set)] pub program_dir: String,
    #[pyo3(get, set)] pub log_dir: String,
    #[pyo3(get, set)] pub telemetry_enabled: bool,
    #[pyo3(get, set)] pub telemetry_endpoint: String,
    #[pyo3(get, set)] pub telemetry_service_name: String,
    #[pyo3(get, set)] pub runtime: RuntimeConfig,
    #[pyo3(get, set)] pub models: Vec<ModelConfig>,
    #[pyo3(get, set)] pub max_concurrent_processes: Option<usize>,
    #[pyo3(get, set)] pub python_snapshot: bool,
}

#[pymethods]
impl Config {
    #[new]
    #[pyo3(signature = (
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
        runtime,
        models,
        max_concurrent_processes,
        python_snapshot,
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
        runtime: RuntimeConfig,
        models: Vec<ModelConfig>,
        max_concurrent_processes: Option<usize>,
        python_snapshot: bool,
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
            runtime,
            models,
            max_concurrent_processes,
            python_snapshot,
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
            runtime: BootstrapRuntimeConfig {
                worker_threads: cfg.runtime.worker_threads,
                wasm_max_instances: cfg.runtime.wasm_max_instances,
                wasm_max_memory_mb: cfg.runtime.wasm_max_memory_mb,
                wasm_warm_memory_mb: cfg.runtime.wasm_warm_memory_mb,
                wasm_warm_slots: cfg.runtime.wasm_warm_slots,
                allow_fs: cfg.runtime.allow_fs,
                fs_scratch_dir: PathBuf::from(cfg.runtime.fs_scratch_dir),
                allow_network: cfg.runtime.allow_network,
                network_allowed_hosts: cfg.runtime.network_allowed_hosts,
                max_upload_mb: cfg.runtime.max_upload_mb,
            },
            models: cfg
                .models
                .into_iter()
                .map(|m| BootstrapModelConfig {
                    name: m.name,
                    arch_name: m.arch_name,
                    kv_page_size: m.kv_page_size,
                    tokenizer_path: PathBuf::from(m.tokenizer_path),
                    devices: m
                        .devices
                        .into_iter()
                        .map(|d| BootstrapDeviceConfig {
                            hostname: d.hostname,
                            total_pages: d.total_pages,
                            cpu_pages: d.cpu_pages,
                            max_batch_tokens: d.max_batch_tokens,
                            max_batch_size: d.max_batch_size,
                        })
                        .collect(),
                    scheduler: BootstrapSchedulerConfig {
                        batch_policy: m.scheduler.batch_policy,
                        request_timeout_secs: m.scheduler.request_timeout_secs,
                        default_token_limit: m.scheduler.default_token_limit,
                        default_endowment_pages: m.scheduler.default_endowment_pages,
                        admission_oversubscription_factor:
                            m.scheduler.admission_oversubscription_factor,
                        restore_pause_at_utilization:
                            m.scheduler.restore_pause_at_utilization,
                    },
                })
                .collect(),
            skip_tracing: false,
            max_concurrent_processes: cfg.max_concurrent_processes,
            python_snapshot: cfg.python_snapshot,
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

    /// Force-shutdown the runtime by exiting the process.
    /// Force-shutdown the runtime by exiting the process.
    fn shutdown(&self) {
        // std::process::exit(0);
        // Do nothing - runtime shuts down when handle is dropped
        tracing::info!("RuntimeHandle::shutdown called - ignoring (shutdown on drop)");
    }

    /// Returns true if the runtime is running.
    /// Always true — there is no intermediate "shutting down" state.
    fn is_running(&self) -> bool {
        true
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
        let bootstrap_config: BootstrapConfig = config.into();

        // Python is the source of truth for `worker_threads`; it
        // resolves the default (typically `os.cpu_count()`) and validates
        // `> 0` before sending. Apply unconditionally.
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.enable_all();
        builder.worker_threads(bootstrap_config.runtime.worker_threads);
        let rt = Arc::new(
            builder
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?,
        );

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

/// Python module definition — compiled as `pie._runtime`
#[pymodule]
pub fn _runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Config>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<DeviceConfig>()?;
    m.add_class::<SchedulerConfig>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<RuntimeHandle>()?;
    m.add_class::<RpcServer>()?;
    m.add_function(wrap_pyfunction!(py_bootstrap, m)?)?;
    Ok(())
}
