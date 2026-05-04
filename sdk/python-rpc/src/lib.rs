//! `pie_rpc` — Python bindings for pie's [`RpcServer`] IPC primitive.
//!
//! Each Python driver (native / vllm / sglang) creates one `RpcServer`
//! per leader rank and publishes its `server_name()` so the Rust runtime
//! can dial in via `RpcClient`. The actual transport is `ipc-channel`
//! (POSIX message queue + `IpcOneShotServer`) — see
//! [`pie::device::RpcServer`] for the implementation.
//!
//! Two channels run in parallel per driver: this one (cold path —
//! capability negotiation, device control, KV swap RPCs) and the
//! shmem fast path that each worker mounts at `/pie_shmem_g{group_id}`
//! for `fire_batch`. They are independent; a driver can use both, only
//! shmem, or only RPC depending on its needs.

use pie::device::RpcServer as InternalRpcServer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Python-hosted IPC server (thin wrapper around [`pie::device::RpcServer`]).
///
/// Usage from Python:
/// ```python
/// from pie_rpc import RpcServer
///
/// server = RpcServer.create()
/// name = server.server_name()  # publish to the Rust runtime via the handshake
/// while True:
///     req = server.poll_blocking(timeout_ms=1000)
///     if req is not None:
///         request_id, method, payload, _walltime = req
///         result = handle(method, payload)
///         server.respond(request_id, result)
/// ```
#[pyclass]
pub struct RpcServer {
    inner: InternalRpcServer,
}

#[pymethods]
impl RpcServer {
    /// Create a new IPC server. The `server_name()` returned afterwards
    /// is what the Rust runtime dials in via `RpcClient::connect`.
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

/// Python module — compiled as `pie_rpc`.
#[pymodule]
fn pie_rpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RpcServer>()?;
    Ok(())
}
