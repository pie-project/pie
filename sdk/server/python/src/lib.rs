//! `pie-server` Python bindings — the embeddable counterpart to the
//! `pie` CLI binary.
//!
//! Both surfaces drive the same library (`worker`); this crate
//! is just a pyo3 wrapper around [`worker::runtime::start_runtime`]
//! plus a [`worker::runtime::RuntimeHandle`] handle. Lifecycle:
//! when the Python `EngineHandle` is dropped (or the user's interpreter
//! exits), the embedded tokio runtime + every subprocess engine are
//! torn down — combined with the `PR_SET_PDEATHSIG` hook in
//! `subprocess_engine`, this means "script ends → server is gone, no
//! orphans".

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use worker::config::Config as ServeConfig;
use worker::runtime::{self, RuntimeHandle as ServeHandle};

/// Live engine returned by `bootstrap`. Holds the tokio runtime that
/// keeps the WS scheduler + engine supervisors alive.
///
/// Methods:
///   - `url` (str)        — `ws://host:port` the engine is listening on
///   - `shutdown()`       — blocking, idempotent. Stops engines + runtime.
///   - `is_running()`     — `True` until `shutdown()` returns.
#[pyclass(name = "EngineHandle")]
struct PyEngineHandle {
    url: String,
    /// `(handle, runtime)` together — once `shutdown()` runs, both are
    /// taken to `None`. The runtime has to outlive every subprocess
    /// engine join, which `ServeHandle::shutdown` guarantees.
    inner: Mutex<Option<(ServeHandle, Arc<tokio::runtime::Runtime>)>>,
}

#[pymethods]
impl PyEngineHandle {
    #[getter]
    fn url(&self) -> String {
        self.url.clone()
    }

    /// True until `shutdown()` returns. Cheap; no blocking.
    fn is_running(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    /// Tear down the engine: signals every engine, joins them, releases
    /// the tokio runtime. Idempotent — calling twice is a no-op.
    /// Releases the GIL during the (potentially slow) join.
    fn shutdown(&self, py: Python<'_>) {
        let taken = self.inner.lock().unwrap().take();
        if let Some((handle, runtime)) = taken {
            py.detach(|| {
                runtime.block_on(handle.shutdown());
                // Drop the runtime; tokio joins worker threads.
                drop(runtime);
            });
        }
    }
}

impl Drop for PyEngineHandle {
    /// Safety net for "user forgot to call `shutdown()`" — happens on
    /// interpreter exit, when GC reclaims the handle, or when the
    /// Python `Server.__aexit__` raises before reaching `shutdown()`.
    fn drop(&mut self) {
        if let Some((handle, runtime)) = self.inner.lock().unwrap().take() {
            runtime.block_on(handle.shutdown());
            drop(runtime);
        }
    }
}

/// Install the global `tracing` subscriber, once per process.
///
/// The CLI does this in `bootstrap::observe::init_tracing`; the wheel did it
/// nowhere, so every engine-side `warn!`/`error!` was dropped on the floor and
/// `RUST_LOG` had no observable effect. That matters most exactly when
/// something goes wrong: a request that fails inside the engine reaches Python
/// as a terse channel error ("channel is poisoned", "pipeline is closed") whose
/// actual cause is only ever stated in a WARN.
///
/// Same conventions as the CLI: level from `RUST_LOG`, stderr as the writer so
/// stdout stays clean for piping, and `try_init` so a host process that has
/// already installed a subscriber keeps it rather than panicking.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Boot the engine from a TOML config string (same schema `pie serve
/// --config <path>` reads). Returns an [`EngineHandle`] that the caller
/// can query and shut down.
///
/// Blocks until the engine is fully booted (engines spawned, weights
/// loaded, WS listener bound). Releases the GIL during the wait so
/// other Python threads keep running.
#[pyfunction]
#[pyo3(text_signature = "(toml_str)")]
fn bootstrap(py: Python<'_>, toml_str: &str) -> PyResult<PyEngineHandle> {
    init_tracing();
    // The CLI does this in `bootstrap::install_crypto_provider`; the wheel does
    // not link that crate, and every HTTPS client in the engine (inferlet
    // registry fetches, blob loads) is built on `rustls-no-provider` and would
    // fail to build a client without a backend chosen first. Idempotent.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let cfg: ServeConfig = toml::from_str(toml_str)
        .map_err(|e| PyValueError::new_err(format!("parse config TOML: {e}")))?;
    cfg.validate()
        .map_err(|e| PyValueError::new_err(format!("validate config: {e:#}")))?;

    py.detach(|| -> PyResult<PyEngineHandle> {
        let runtime = runtime::build_runtime(&cfg)
            .map_err(|e| PyRuntimeError::new_err(format!("build tokio runtime: {e:#}")))?;
        let runtime = Arc::new(runtime);

        // The embedded engine wheel is always single-node: embed an in-proc
        // controller and self-register before booting the engine.
        let control_addr = format!("{}:{}", cfg.server.host, cfg.server.port);
        let coordinator = runtime::connect(&runtime::TopologyMode::SingleNode, control_addr)
            .map_err(|e| PyRuntimeError::new_err(format!("join control plane: {e:#}")))?;

        let handle = runtime
            .block_on(runtime::start_runtime(cfg, coordinator))
            .map_err(|e| PyRuntimeError::new_err(format!("start_runtime: {e:#}")))?;

        let url = handle.url.clone();
        Ok(PyEngineHandle {
            url,
            inner: Mutex::new(Some((handle, runtime))),
        })
    })
}

/// Module name is `_engine`; maturin's `module-name = "pie._engine"`
/// in `pyproject.toml` places the resulting `.so` at
/// `python/pie/_engine.so`, so `from pie import _engine` works.
#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_class::<PyEngineHandle>()?;
    Ok(())
}
