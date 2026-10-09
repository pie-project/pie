//! `pie._engine`: what `pie serve` boots, inside a Python process — the
//! Python twin of `@pie-project/server`.
//!
//! [`worker::Server::serve`] brings up the worker and, beside it, the
//! controller and the gateway with its HTTP routes, from the same config file
//! the CLI reads. Its calls block with the GIL released; `pie.server.Server`
//! runs them off-thread. A handle dropped without `shutdown()` (GC,
//! interpreter exit) still tears the engine down, on a thread of its own so
//! the GIL is not held.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use runtime::inferlet::program;

type Live = Arc<worker::Server>;

#[pyclass(name = "EngineHandle")]
struct PyEngineHandle {
    url: String,
    live: Mutex<Option<Live>>,
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
        let server = self.live()?;
        let component = component.to_vec();
        py.detach(|| server.install_language(language.name(), component))
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
        let server = self.live()?;
        let bytes = bytes.to_vec();
        py.detach(|| server.install(bytes, file, version))
            .map_err(|e| PyRuntimeError::new_err(format!("install: {e:#}")))
    }

    /// Stop every engine, join them, and release the runtime. Idempotent;
    /// blocks with the GIL released.
    fn shutdown(&self, py: Python<'_>) {
        if let Some(server) = self.live.lock().unwrap().take() {
            py.detach(|| server.shutdown());
        }
    }
}

impl PyEngineHandle {
    fn live(&self) -> PyResult<Live> {
        self.live
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("the server is shut down"))
    }
}

impl Drop for PyEngineHandle {
    fn drop(&mut self) {
        if let Some(server) = self.live.lock().unwrap().take() {
            std::thread::spawn(move || server.shutdown());
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
    let config = worker::Config::parse(toml_str)
        .map_err(|e| PyValueError::new_err(format!("config: {e:#}")))?;
    let server = py
        .detach(|| worker::Server::serve(config))
        .map_err(|e| PyRuntimeError::new_err(format!("start: {e:#}")))?;
    let addr = server.listen_addr().expect("a served worker listens");
    Ok(PyEngineHandle {
        url: format!("ws://{addr}"),
        live: Mutex::new(Some(Arc::new(server))),
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
