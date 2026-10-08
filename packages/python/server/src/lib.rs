//! `pie._engine`: what `pie serve` boots, inside a Python process — the
//! Python twin of `@pie-project/server`.
//!
//! [`pie::run_standalone`] brings up the controller, the gateway with its
//! HTTP routes and the worker from the same config file the CLI reads. The
//! boot and the shutdown block with the GIL released; `pie.server.Server`
//! runs them off-thread. A handle dropped without `shutdown()` (GC,
//! interpreter exit) still tears the engine down, on a thread of its own so
//! the GIL is not held; what that thread has not released by process exit,
//! the OS reclaims.

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use pie::StandaloneHandle;
use runtime::inferlet::program;

/// The runtime outlives the handle's shutdown: every engine is joined on it.
type Live = (StandaloneHandle, tokio::runtime::Runtime);

fn stop((handle, runtime): Live) {
    runtime.block_on(handle.shutdown());
    drop(runtime);
}

#[pyclass(name = "EngineHandle")]
struct PyEngineHandle {
    url: String,
    live: Mutex<Option<Live>>,
    runtime: tokio::runtime::Handle,
}

#[pymethods]
impl PyEngineHandle {
    /// `ws://host:port` the gateway listens on; its HTTP routes share the
    /// port. With `server.port = 0` this is the port the OS handed out.
    #[getter]
    fn url(&self) -> String {
        self.url.clone()
    }

    /// True until `shutdown()` returns.
    fn is_running(&self) -> bool {
        self.live.lock().unwrap().is_some()
    }

    /// Hand the runtime a language component (`python`, `javascript`) from
    /// bytes; a script inferlet in that language can run once this returns.
    /// Blocks with the GIL released.
    fn install_language(
        &self,
        py: Python<'_>,
        language: &str,
        component: &[u8],
    ) -> PyResult<String> {
        let language = program::Language::parse(language)
            .map_err(|e| PyValueError::new_err(format!("language: {e:#}")))?;
        let runtime = self.runtime()?;
        let component = component.to_vec();
        py.detach(|| runtime.block_on(program::add_language(language, component)))
            .map_err(|e| PyRuntimeError::new_err(format!("install language: {e:#}")))?;
        Ok(language.name().to_string())
    }

    #[pyo3(signature = (bytes, file, version=None))]
    fn install(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        file: &str,
        version: Option<&str>,
    ) -> PyResult<String> {
        let runtime = self.runtime()?;
        let bytes = bytes.to_vec();
        let name = py
            .detach(|| runtime.block_on(program::add(bytes, file, version, true)))
            .map_err(|e| PyRuntimeError::new_err(format!("install: {e:#}")))?;
        Ok(name.to_string())
    }

    /// Stop every engine, join them, and release the runtime. Idempotent;
    /// blocks with the GIL released.
    fn shutdown(&self, py: Python<'_>) {
        if let Some(live) = self.live.lock().unwrap().take() {
            py.detach(|| stop(live));
        }
    }
}

impl PyEngineHandle {
    fn runtime(&self) -> PyResult<tokio::runtime::Handle> {
        if self.live.lock().unwrap().is_none() {
            return Err(PyRuntimeError::new_err("the server is shut down"));
        }
        Ok(self.runtime.clone())
    }
}

impl Drop for PyEngineHandle {
    fn drop(&mut self) {
        if let Some(live) = self.live.lock().unwrap().take() {
            std::thread::spawn(move || stop(live));
        }
    }
}

/// Level from `RUST_LOG` (default `warn`), stderr as the writer; a host
/// that already installed a subscriber keeps it.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Boot from the TOML text `pie serve --config` reads. Blocks, with the
/// GIL released, until engines are up, weights are loaded and the listener
/// is bound.
#[pyfunction]
fn bootstrap(py: Python<'_>, toml_str: &str) -> PyResult<PyEngineHandle> {
    init_tracing();
    let (controller, gateway, worker) = pie::derive::derive_standalone(toml_str)
        .map_err(|e| PyValueError::new_err(format!("config: {e:#}")))?;

    py.detach(|| {
        let runtime = worker::serve::build_runtime(&worker)
            .map_err(|e| PyRuntimeError::new_err(format!("build tokio runtime: {e:#}")))?;
        let handle = runtime
            .block_on(pie::run_standalone(controller, gateway, worker))
            .map_err(|e| PyRuntimeError::new_err(format!("start: {e:#}")))?;
        Ok(PyEngineHandle {
            url: format!("ws://{}", handle.listen_addr),
            runtime: runtime.handle().clone(),
            live: Mutex::new(Some((handle, runtime))),
        })
    })
}

/// `module-name = "pie._engine"` in pyproject.toml puts this beside the
/// package as `pie/_engine.so`.
#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_class::<PyEngineHandle>()?;
    Ok(())
}
