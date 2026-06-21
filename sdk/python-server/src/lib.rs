//! `pie-server` Python bindings — the embeddable counterpart to the
//! `pie` CLI binary.
//!
//! Both surfaces drive the same library (`pie-worker`); this crate
//! is just a pyo3 wrapper around [`pie_worker::serve::start_engine`]
//! plus a [`pie_worker::serve::EngineHandle`] handle. Lifecycle:
//! when the Python `EngineHandle` is dropped (or the user's interpreter
//! exits), the embedded tokio runtime + every subprocess driver are
//! torn down — combined with the `PR_SET_PDEATHSIG` hook in
//! `subprocess_driver`, this means "script ends → server is gone, no
//! orphans".

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use pie_worker::config::Config as ServeConfig;
use pie_worker::serve::{self, EngineHandle as ServeHandle};

/// Live engine returned by `bootstrap`. Holds the tokio runtime that
/// keeps the WS scheduler + driver supervisors alive.
///
/// Methods:
///   - `url` (str)        — `ws://host:port` the engine is listening on
///   - `token` (str)      — internal auth token (pass to `pie-client`'s
///                          `auth_by_token`)
///   - `shutdown()`       — blocking, idempotent. Stops drivers + runtime.
///   - `is_running()`     — `True` until `shutdown()` returns.
#[pyclass(name = "EngineHandle")]
struct PyEngineHandle {
    url: String,
    token: String,
    /// `(handle, runtime)` together — once `shutdown()` runs, both are
    /// taken to `None`. The runtime has to outlive every subprocess
    /// driver join, which `ServeHandle::shutdown` guarantees.
    inner: Mutex<Option<(ServeHandle, Arc<tokio::runtime::Runtime>)>>,
}

#[pymethods]
impl PyEngineHandle {
    #[getter]
    fn url(&self) -> String {
        self.url.clone()
    }

    #[getter]
    fn token(&self) -> String {
        self.token.clone()
    }

    /// True until `shutdown()` returns. Cheap; no blocking.
    fn is_running(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    /// Tear down the engine: signals every driver, joins them, releases
    /// the tokio runtime. Idempotent — calling twice is a no-op.
    /// Releases the GIL during the (potentially slow) join.
    fn shutdown(&self, py: Python<'_>) {
        let taken = self.inner.lock().unwrap().take();
        if let Some((handle, runtime)) = taken {
            py.allow_threads(|| {
                handle.shutdown();
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
            handle.shutdown();
            drop(runtime);
        }
    }
}

/// Boot the engine from a TOML config string (same schema `pie serve
/// --config <path>` reads). Returns an [`EngineHandle`] that the caller
/// can query and shut down.
///
/// Blocks until the engine is fully booted (drivers spawned, weights
/// loaded, WS listener bound). Releases the GIL during the wait so
/// other Python threads keep running.
#[pyfunction]
#[pyo3(text_signature = "(toml_str)")]
fn bootstrap(py: Python<'_>, toml_str: &str) -> PyResult<PyEngineHandle> {
    let cfg: ServeConfig = toml::from_str(toml_str)
        .map_err(|e| PyValueError::new_err(format!("parse config TOML: {e}")))?;
    cfg.validate()
        .map_err(|e| PyValueError::new_err(format!("validate config: {e:#}")))?;

    py.allow_threads(|| -> PyResult<PyEngineHandle> {
        let runtime = serve::build_runtime(&cfg)
            .map_err(|e| PyRuntimeError::new_err(format!("build tokio runtime: {e:#}")))?;
        let runtime = Arc::new(runtime);

        // Best-effort: install the Python WASM runtime tarball if missing,
        // mirroring `pie serve`'s startup. Failures (offline / no registry)
        // log + continue; only matters for Python inferlets.
        pie_worker::py_runtime::ensure_installed_best_effort();

        // The embedded engine wheel is always single-node: embed an in-proc
        // controller and self-register before booting the engine.
        let control_addr = format!("{}:{}", cfg.server.host, cfg.server.port);
        let coordinator = serve::coordination::connect(
            &serve::TopologyMode::SingleNode,
            control_addr,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("join control plane: {e:#}")))?;

        let handle = runtime
            .block_on(serve::start_engine(cfg, coordinator))
            .map_err(|e| PyRuntimeError::new_err(format!("start_engine: {e:#}")))?;

        let url = handle.url.clone();
        let token = handle.token.clone();

        Ok(PyEngineHandle {
            url,
            token,
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
