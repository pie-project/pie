//! `pie-server` Python bindings — the embeddable counterpart to the
//! `pie` CLI binary.
//!
//! Both surfaces drive the same library (`pie-worker`); this crate
//! is just a pyo3 wrapper around [`pie_worker::engine::start_engine`]
//! plus a [`pie_worker::engine::EngineHandle`] handle. Lifecycle:
//! when the Python `EngineHandle` is dropped (or the user's interpreter
//! exits), the embedded tokio runtime + every subprocess driver are
//! torn down — combined with the `PR_SET_PDEATHSIG` hook in
//! `subprocess_driver`, this means "script ends → server is gone, no
//! orphans".

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use pie_worker::config::Config as ServeConfig;
use pie_worker::engine::{self, EngineHandle as ServeHandle};

/// Live engine returned by `bootstrap`. Holds the tokio runtime that
/// keeps the WS scheduler + driver supervisors alive.
///
/// Methods:
///   - `url` (str)        — `ws://host:port` the engine is listening on
///   - `shutdown()`       — blocking, idempotent. Stops drivers + runtime.
///   - `is_running()`     — `True` until `shutdown()` returns.
#[pyclass(name = "EngineHandle")]
struct PyEngineHandle {
    url: String,
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

    py.detach(|| -> PyResult<PyEngineHandle> {
        let runtime = engine::build_runtime(&cfg)
            .map_err(|e| PyRuntimeError::new_err(format!("build tokio runtime: {e:#}")))?;
        let runtime = Arc::new(runtime);

        // The embedded engine wheel is always single-node: embed an in-proc
        // controller and self-register before booting the engine.
        let control_addr = format!("{}:{}", cfg.server.host, cfg.server.port);
        let coordinator = engine::connect(&engine::TopologyMode::SingleNode, control_addr)
            .map_err(|e| PyRuntimeError::new_err(format!("join control plane: {e:#}")))?;

        let handle = runtime
            .block_on(engine::start_engine(cfg, coordinator))
            .map_err(|e| PyRuntimeError::new_err(format!("start_engine: {e:#}")))?;

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
