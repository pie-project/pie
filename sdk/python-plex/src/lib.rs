use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pie_policy::{
    Document, HostSupportV0_6, PlexError as RustPlexError, PlexRuntime, QueryError, QueryHandler,
};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_native, PlexError, PyException);
create_exception!(_native, InvalidEvent, PlexError);
create_exception!(_native, BackendError, PlexError);
create_exception!(_native, PolicyPackageError, PlexError);
create_exception!(_native, QueryCallbackError, PlexError);

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static INVOKING_RUNTIME_IDS: RefCell<BTreeSet<u64>> =
        const { RefCell::new(BTreeSet::new()) };
}

struct PythonQueryHandler {
    callback: Py<PyAny>,
}

impl QueryHandler for PythonQueryHandler {
    fn query(&self, method: &str, args: &Document) -> Result<Document, QueryError> {
        Python::attach(|py| {
            let json = py
                .import("json")
                .map_err(|error| QueryError::Handler(error.to_string()))?;
            let args_json = serde_json::to_string(args)
                .map_err(|error| QueryError::Handler(error.to_string()))?;
            let args_object = json
                .call_method1("loads", (args_json,))
                .map_err(|error| QueryError::Handler(error.to_string()))?;
            let result = self
                .callback
                .bind(py)
                .call1((method, args_object))
                .map_err(|error| QueryError::Handler(error.to_string()))?;
            let result_json: String = json
                .call_method1("dumps", (result,))
                .and_then(|encoded| encoded.extract())
                .map_err(|error| QueryError::Handler(error.to_string()))?;
            serde_json::from_str(&result_json)
                .map_err(|error| QueryError::Handler(format!("non-JSON query result: {error}")))
        })
    }
}

#[pyclass(name = "NativeRuntime")]
struct NativeRuntime {
    id: u64,
    runtime: PlexRuntime,
}

#[pymethods]
impl NativeRuntime {
    #[new]
    #[pyo3(signature = (policy, query=None, mechanics=None))]
    fn new(
        policy: &str,
        query: Option<Py<PyAny>>,
        mechanics: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let package = std::fs::read(policy)
            .map_err(|error| PolicyPackageError::new_err(error.to_string()))?;
        let query_handler = query
            .map(|callback| Arc::new(PythonQueryHandler { callback }) as Arc<dyn QueryHandler>);
        let support = HostSupportV0_6::with_standard_ids(mechanics.unwrap_or_default())
            .map_err(|error| PolicyPackageError::new_err(error.to_string()))?;
        let runtime = PlexRuntime::from_package_bytes(&package, query_handler, support)
            .map_err(map_plex_error)?;
        Ok(Self {
            id: next_runtime_id()?,
            runtime,
        })
    }

    fn invoke_json(&self, py: Python<'_>, event_json: &str) -> PyResult<String> {
        let _guard = InvocationGuard::enter(self.id)?;
        py.detach(|| self.runtime.invoke_json(event_json))
            .map_err(map_plex_error)
    }
}

struct InvocationGuard {
    runtime_id: u64,
}

impl InvocationGuard {
    fn enter(runtime_id: u64) -> PyResult<Self> {
        INVOKING_RUNTIME_IDS.with(|runtime_ids| {
            if !runtime_ids.borrow_mut().insert(runtime_id) {
                return Err(QueryCallbackError::new_err(
                    "recursive invoke on the same PLEX runtime is not allowed",
                ));
            }
            Ok(Self { runtime_id })
        })
    }
}

impl Drop for InvocationGuard {
    fn drop(&mut self) {
        INVOKING_RUNTIME_IDS.with(|runtime_ids| {
            let removed = runtime_ids.borrow_mut().remove(&self.runtime_id);
            debug_assert!(removed);
        });
    }
}

fn next_runtime_id() -> PyResult<u64> {
    NEXT_RUNTIME_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| PlexError::new_err("PLEX runtime ID space exhausted"))
}

fn map_plex_error(error: RustPlexError) -> PyErr {
    match error {
        RustPlexError::InvalidEvent(message) => InvalidEvent::new_err(message),
        RustPlexError::Backend(message) => BackendError::new_err(message),
        RustPlexError::PolicyPackage(message) => PolicyPackageError::new_err(message),
        RustPlexError::Runtime(message) => PlexError::new_err(message),
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRuntime>()?;
    module.add("PlexError", module.py().get_type::<PlexError>())?;
    module.add("InvalidEvent", module.py().get_type::<InvalidEvent>())?;
    module.add("BackendError", module.py().get_type::<BackendError>())?;
    module.add(
        "PolicyPackageError",
        module.py().get_type::<PolicyPackageError>(),
    )?;
    module.add(
        "QueryCallbackError",
        module.py().get_type::<QueryCallbackError>(),
    )?;
    Ok(())
}
