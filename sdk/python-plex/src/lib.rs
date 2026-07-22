use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pie_policy::{Document, PlexError as RustPlexError, PlexRuntime, QueryError, QueryHandler};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_native, PlexError, PyException);
create_exception!(_native, InvalidEvent, PlexError);
create_exception!(_native, BackendError, PlexError);
create_exception!(_native, PolicyPackageError, PlexError);
create_exception!(_native, QueryCallbackError, PlexError);

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
    runtime: PlexRuntime,
    invoking: AtomicBool,
}

#[pymethods]
impl NativeRuntime {
    #[new]
    #[pyo3(signature = (policy, query=None, actions=None))]
    fn new(policy: &str, query: Option<Py<PyAny>>, actions: Option<Vec<String>>) -> PyResult<Self> {
        let package = std::fs::read(policy)
            .map_err(|error| PolicyPackageError::new_err(error.to_string()))?;
        let query_handler = query
            .map(|callback| Arc::new(PythonQueryHandler { callback }) as Arc<dyn QueryHandler>);
        let runtime = PlexRuntime::from_package_bytes(
            &package,
            query_handler,
            actions
                .unwrap_or_default()
                .into_iter()
                .collect::<BTreeSet<_>>(),
        )
        .map_err(map_plex_error)?;
        Ok(Self {
            runtime,
            invoking: AtomicBool::new(false),
        })
    }

    fn invoke_json(&self, py: Python<'_>, event_json: &str) -> PyResult<String> {
        if self
            .invoking
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(QueryCallbackError::new_err(
                "recursive invoke on the same PLEX runtime is not allowed",
            ));
        }
        let _guard = InvocationGuard(&self.invoking);
        py.detach(|| self.runtime.invoke_json(event_json))
            .map_err(map_plex_error)
    }
}

struct InvocationGuard<'a>(&'a AtomicBool);

impl Drop for InvocationGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
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
