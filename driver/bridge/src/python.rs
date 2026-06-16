//! PyO3 module entry. Wires up every `Py<T>` schema class (via the
//! `schema_module!` helper) plus the shmem-ring server wrappers
//! (`PyShmemServer` / `PyLease`) that Python downstream drivers
//! (dev / sglang / vllm) need to run as subprocesses.

use std::sync::Mutex;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBytes, PyModule};

use crate::ipc::{Lease, ShmemServer};

pie_bridge_macros::schema_module! {
    Frame, ResponseFrame, RequestPayload, ResponsePayload,
    ForwardRequest, ForwardResponse, CopyRequest, AdapterRequest,
    StatusResponse, AdapterBinding,
    Sampler, CopyDir, CopyResource, AdapterOp,
}

// ---------------------------------------------------------------------------
// Zero-copy slice → NumPy helper
// ---------------------------------------------------------------------------
//
// The schema macro's `Vec<T>` getter for Python used to do
// `from_raw_parts(...).to_vec()`, which allocated and memcpy'd the
// whole slice on every access. With the parent `Py<PyBytes>` owner
// available, we can hand back a NumPy view over the same bytes via
// `numpy.frombuffer(bytes, dtype, count, offset)`. NumPy keeps a
// reference to the PyBytes object for the array's lifetime, so the
// underlying archive stays alive as long as any view does.
//
// On the Python caller side, `np.asarray(fr.token_ids)` is now a
// no-op cast (the array is already a NumPy view), saving the
// previously per-fire `Vec<u32> → np.asarray` copy.

/// Process-wide cache of the numpy callables + dtype objects we touch
/// in every slice getter. Without this, each `slice_to_numpy` call
/// would do `py.import("numpy")` + `np.getattr(dtype)` + a method
/// lookup — three hash-based Python dict ops per accessor. A typical
/// parsed ForwardRequest has ~15 slice fields, so that adds up.
/// One-time init under [`pyo3::sync::GILOnceCell`].
struct NpCache {
    frombuffer: Py<PyAny>,
    empty: Py<PyAny>,
    dt_u8: Py<PyAny>,
    dt_u16: Py<PyAny>,
    dt_u32: Py<PyAny>,
    dt_u64: Py<PyAny>,
    dt_i8: Py<PyAny>,
    dt_i16: Py<PyAny>,
    dt_i32: Py<PyAny>,
    dt_i64: Py<PyAny>,
    dt_f32: Py<PyAny>,
    dt_f64: Py<PyAny>,
    dt_bool: Py<PyAny>,
}

static NP_CACHE: GILOnceCell<NpCache> = GILOnceCell::new();

fn np_cache(py: Python<'_>) -> PyResult<&'static NpCache> {
    NP_CACHE.get_or_try_init(py, || -> PyResult<NpCache> {
        let np = py.import("numpy")?;
        let get = |name: &str| -> PyResult<Py<PyAny>> { Ok(np.getattr(name)?.unbind()) };
        Ok(NpCache {
            frombuffer: get("frombuffer")?,
            empty: get("empty")?,
            dt_u8: get("uint8")?,
            dt_u16: get("uint16")?,
            dt_u32: get("uint32")?,
            dt_u64: get("uint64")?,
            dt_i8: get("int8")?,
            dt_i16: get("int16")?,
            dt_i32: get("int32")?,
            dt_i64: get("int64")?,
            dt_f32: get("float32")?,
            dt_f64: get("float64")?,
            dt_bool: get("bool_")?,
        })
    })
}

fn dtype_of<'a>(cache: &'a NpCache, name: &str) -> &'a Py<PyAny> {
    match name {
        "uint8" => &cache.dt_u8,
        "uint16" => &cache.dt_u16,
        "uint32" => &cache.dt_u32,
        "uint64" => &cache.dt_u64,
        "int8" => &cache.dt_i8,
        "int16" => &cache.dt_i16,
        "int32" => &cache.dt_i32,
        "int64" => &cache.dt_i64,
        "float32" => &cache.dt_f32,
        "float64" => &cache.dt_f64,
        "bool_" => &cache.dt_bool,
        // Defensive: macro only emits the names above. Fall back to
        // u8 so a future schema field added without updating this
        // table still parses (it just won't be a typed view).
        _ => &cache.dt_u8,
    }
}

/// Build a zero-copy NumPy 1-D view over a slice inside the parent
/// `Py<PyBytes>`. `ptr` points into `bytes`'s buffer; `count` is the
/// element count; `dtype` is the NumPy dtype name (e.g. `"uint32"`).
///
/// Falls back to a small allocated array when the slice is empty —
/// `numpy.frombuffer` rejects `count = 0` with a `ValueError`, but
/// the natural empty-Vec result is an empty array of the right type.
pub(crate) fn slice_to_numpy<'py>(
    py: Python<'py>,
    bytes: &Py<PyBytes>,
    ptr: *const u8,
    count: usize,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cache = np_cache(py)?;
    let dt = dtype_of(cache, dtype).bind(py);
    if ptr.is_null() || count == 0 {
        return cache.empty.bind(py).call1((0usize, dt));
    }
    let buf = bytes.bind(py);
    // SAFETY: `bytes` is a `Py<PyBytes>` we hold a refcount on; its
    // buffer pointer is stable for the GIL lifetime. `ptr` was
    // produced by a schema accessor on the same archive that lives
    // inside this `PyBytes` buffer, so the offset is in-range.
    let buf_start = buf.as_bytes().as_ptr();
    let offset = (ptr as usize).wrapping_sub(buf_start as usize);
    // numpy.frombuffer(buffer, dtype, count, offset) — positional args
    // through count/offset are supported.
    cache.frombuffer.bind(py).call1((buf, dt, count, offset))
}

// ---------------------------------------------------------------------------
// Shmem ring — Python-facing wrappers around `pie_bridge::ipc`.
// ---------------------------------------------------------------------------

/// PyO3 wrapper for [`pie_bridge::ipc::ShmemServer`]. Creates
/// the POSIX shmem region and exposes `poll` / `poll_blocking` to the
/// per-driver worker loop.
#[pyclass(name = "ShmemServer")]
pub struct PyShmemServer {
    inner: ShmemServer,
}

#[pymethods]
impl PyShmemServer {
    /// Create a new shmem region. `schema_hash` defaults to the bridge
    /// crate's `SCHEMA_HASH` (the value the runtime side will compare).
    ///
    /// Wait strategy: `spin_budget_us` (default `1000`) is the
    /// busy-spin window after `poll_blocking` comes up empty —
    /// catches back-to-back requests with sub-µs wake. After the
    /// budget elapses, the server parks on the global `req_wake`
    /// atomic via the cross-process kernel primitive (Linux futex,
    /// Windows WaitOnAddress, macOS __ulock_wait); set to `0` to
    /// always park. See `pie_bridge::ipc::ShmemServer`.
    #[new]
    #[pyo3(signature = (
        name, num_slots, req_buf, resp_buf,
        schema_hash=None, spin_budget_us=1000
    ))]
    fn new(
        name: &str,
        num_slots: usize,
        req_buf: usize,
        resp_buf: usize,
        schema_hash: Option<[u8; 8]>,
        spin_budget_us: u64,
    ) -> PyResult<Self> {
        let hash = schema_hash.unwrap_or(crate::SCHEMA_HASH);
        ShmemServer::create(name, num_slots, req_buf, resp_buf, spin_budget_us, hash)
            .map(|inner| Self { inner })
            .map_err(|e| PyRuntimeError::new_err(format!("ShmemServer::create: {e}")))
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }
    #[getter]
    fn num_slots(&self) -> usize {
        self.inner.num_slots()
    }
    #[getter]
    fn req_buf_size(&self) -> usize {
        self.inner.req_buf_size()
    }
    #[getter]
    fn resp_buf_size(&self) -> usize {
        self.inner.resp_buf_size()
    }

    fn stop(&self) {
        self.inner.stop();
    }
    #[getter]
    fn stopped(&self) -> bool {
        self.inner.stopped()
    }

    /// Non-blocking poll — returns a [`PyLease`] if a slot has a new
    /// request, else `None`.
    fn poll(&self) -> Option<PyLease> {
        self.inner.poll().map(PyLease::new)
    }

    /// Block up to `timeout_ms` milliseconds for a request to arrive.
    fn poll_blocking(&self, timeout_ms: u64) -> Option<PyLease> {
        self.inner
            .poll_blocking(Duration::from_millis(timeout_ms))
            .map(PyLease::new)
    }
}

/// PyO3 wrapper for [`pie_bridge::ipc::Lease`]. RAII handle
/// over one in-flight request — call `commit(bytes)` / `commit_status(code)`
/// to publish a response, or `abort()` to mark the request aborted.
/// Dropping without commit auto-aborts (writes a `ResponseFrame
/// {aborted: true, status: -1}` to the slot).
#[pyclass(name = "Lease", unsendable)]
pub struct PyLease {
    // Option so consuming methods (commit/abort) can `take()` the inner
    // Lease. After take, all other methods raise.
    inner: Mutex<Option<Lease>>,
}

impl PyLease {
    fn new(lease: Lease) -> Self {
        Self {
            inner: Mutex::new(Some(lease)),
        }
    }

    fn with_lease<R>(&self, f: impl FnOnce(&Lease) -> R) -> PyResult<R> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Lease lock poisoned"))?;
        match guard.as_ref() {
            Some(l) => Ok(f(l)),
            None => Err(PyRuntimeError::new_err("Lease already consumed")),
        }
    }

    fn take_lease(&self) -> PyResult<Lease> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Lease lock poisoned"))?;
        guard
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Lease already consumed"))
    }
}

#[pymethods]
impl PyLease {
    /// Request payload bytes. Returns a Python `bytes` (copy) so the
    /// caller doesn't hold a slice into the shmem slot past commit.
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.with_lease(|l| PyBytes::new(py, l.payload()))
    }

    /// Publish a response payload, releasing the slot.
    fn commit(&self, bytes: &[u8]) -> PyResult<()> {
        let lease = self.take_lease()?;
        lease
            .commit(bytes)
            .map_err(|e| PyValueError::new_err(format!("commit: {e}")))
    }

    /// Publish a `StatusResponse { status }` for cold methods that
    /// don't need a full response payload.
    fn commit_status(&self, status: i32) -> PyResult<()> {
        let lease = self.take_lease()?;
        lease
            .commit_status(status)
            .map_err(|e| PyValueError::new_err(format!("commit_status: {e}")))
    }

    /// Mark the request aborted (writes `ResponseFrame { aborted: true,
    /// status: -1 }`). Releases the slot.
    fn abort(&self) -> PyResult<()> {
        let lease = self.take_lease()?;
        lease.abort();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_schema_types(m)?;
    m.add_class::<PyShmemServer>()?;
    m.add_class::<PyLease>()?;
    // Expose SCHEMA_HASH as a `bytes` so callers can compare against
    // the value baked into the runtime side at compile time.
    Python::with_gil(|py| -> PyResult<()> {
        m.add("SCHEMA_HASH", PyBytes::new(py, &crate::SCHEMA_HASH))?;
        Ok(())
    })?;
    Ok(())
}
