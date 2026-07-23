use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use arc_swap::ArcSwapOption;
use crossbeam_channel::{Sender, TrySendError, bounded};
use pie_policy::{Document, PlexError as RustPlexError, PlexRuntime, QueryError, QueryHandler};
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

#[derive(Clone, Copy)]
enum AsyncChannel {
    Route,
    Admit,
    Schedule,
    Evict,
    Feedback,
}

impl AsyncChannel {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "route" => Ok(Self::Route),
            "admit" => Ok(Self::Admit),
            "schedule" => Ok(Self::Schedule),
            "evict" => Ok(Self::Evict),
            "feedback" => Ok(Self::Feedback),
            _ => Err(PlexError::new_err(format!(
                "unknown async PLEX channel {value:?}"
            ))),
        }
    }
}

struct AsyncOutcome {
    epoch: u64,
    json: String,
}

#[derive(Default)]
struct AsyncSlots {
    route: ArcSwapOption<AsyncOutcome>,
    admit: ArcSwapOption<AsyncOutcome>,
    schedule: ArcSwapOption<AsyncOutcome>,
    evict: ArcSwapOption<AsyncOutcome>,
    feedback: ArcSwapOption<AsyncOutcome>,
}

impl AsyncSlots {
    fn slot(&self, channel: AsyncChannel) -> &ArcSwapOption<AsyncOutcome> {
        match channel {
            AsyncChannel::Route => &self.route,
            AsyncChannel::Admit => &self.admit,
            AsyncChannel::Schedule => &self.schedule,
            AsyncChannel::Evict => &self.evict,
            AsyncChannel::Feedback => &self.feedback,
        }
    }
}

struct AsyncCommand {
    channel: AsyncChannel,
    epoch: u64,
    payload: AsyncPayload,
}

enum AsyncPayload {
    Single(Vec<u8>),
    Batch(Vec<u8>),
}

#[pyclass(name = "NativeAsyncRuntime")]
struct NativeAsyncRuntime {
    sender: Option<Sender<AsyncCommand>>,
    slots: Arc<AsyncSlots>,
    worker: Option<JoinHandle<()>>,
    submitted: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
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

#[pymethods]
impl NativeAsyncRuntime {
    #[new]
    #[pyo3(signature = (policy, query=None, actions=None, queue_capacity=64))]
    fn new(
        policy: &str,
        query: Option<Py<PyAny>>,
        actions: Option<Vec<String>>,
        queue_capacity: usize,
    ) -> PyResult<Self> {
        if queue_capacity == 0 {
            return Err(PlexError::new_err(
                "async PLEX queue_capacity must be non-zero",
            ));
        }
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

        let (sender, receiver) = bounded::<AsyncCommand>(queue_capacity);
        let slots = Arc::new(AsyncSlots::default());
        let worker_slots = slots.clone();
        let submitted = Arc::new(AtomicU64::new(0));
        let dropped = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let worker_completed = completed.clone();
        let worker = std::thread::Builder::new()
            .name("pie-plex-async".into())
            .spawn(move || {
                while let Ok(command) = receiver.recv() {
                    let result = match command.payload {
                        AsyncPayload::Single(event_json) => std::str::from_utf8(&event_json)
                            .map_err(|error| {
                                RustPlexError::InvalidEvent(format!(
                                    "invalid UTF-8 event JSON: {error}"
                                ))
                            })
                            .and_then(|event_json| runtime.invoke_json(event_json)),
                        AsyncPayload::Batch(events_json) => std::str::from_utf8(&events_json)
                            .map_err(|error| {
                                RustPlexError::InvalidEvent(format!(
                                    "invalid UTF-8 event batch JSON: {error}"
                                ))
                            })
                            .and_then(|events_json| invoke_batch_json(&runtime, events_json)),
                    };
                    let json = match result {
                        Ok(outcome) => outcome,
                        Err(error) => serde_json::json!({
                            "status": "error",
                            "error": error.to_string(),
                        })
                        .to_string(),
                    };
                    worker_slots
                        .slot(command.channel)
                        .store(Some(Arc::new(AsyncOutcome {
                            epoch: command.epoch,
                            json,
                        })));
                    worker_completed.fetch_add(1, Ordering::Relaxed);
                }
            })
            .map_err(|error| PlexError::new_err(error.to_string()))?;

        Ok(Self {
            sender: Some(sender),
            slots,
            worker: Some(worker),
            submitted,
            dropped,
            completed,
        })
    }

    fn try_submit_json(&self, channel: &str, epoch: u64, event_json: String) -> PyResult<bool> {
        let channel = AsyncChannel::parse(channel)?;
        let Some(sender) = &self.sender else {
            return Ok(false);
        };
        match sender.try_send(AsyncCommand {
            channel,
            epoch,
            payload: AsyncPayload::Single(event_json.into_bytes()),
        }) {
            Ok(()) => {
                self.submitted.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
        }
    }

    fn try_submit_bytes(&self, channel: &str, epoch: u64, event_json: &[u8]) -> PyResult<bool> {
        let channel = AsyncChannel::parse(channel)?;
        let Some(sender) = &self.sender else {
            return Ok(false);
        };
        match sender.try_send(AsyncCommand {
            channel,
            epoch,
            payload: AsyncPayload::Single(event_json.to_vec()),
        }) {
            Ok(()) => {
                self.submitted.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
        }
    }

    fn try_submit_batch_json(
        &self,
        channel: &str,
        epoch: u64,
        events_json: String,
    ) -> PyResult<bool> {
        let channel = AsyncChannel::parse(channel)?;
        let Some(sender) = &self.sender else {
            return Ok(false);
        };
        match sender.try_send(AsyncCommand {
            channel,
            epoch,
            payload: AsyncPayload::Batch(events_json.into_bytes()),
        }) {
            Ok(()) => {
                self.submitted.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
        }
    }

    fn latest_json(&self, channel: &str, after_epoch: u64) -> PyResult<Option<(u64, String)>> {
        let channel = AsyncChannel::parse(channel)?;
        let Some(outcome) = self.slots.slot(channel).load_full() else {
            return Ok(None);
        };
        if outcome.epoch <= after_epoch {
            return Ok(None);
        }
        Ok(Some((outcome.epoch, outcome.json.clone())))
    }

    fn stats(&self) -> (u64, u64, u64) {
        (
            self.submitted.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
            self.completed.load(Ordering::Relaxed),
        )
    }

    fn shutdown(&mut self, py: Python<'_>) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = py.detach(|| worker.join());
        }
    }
}

impl Drop for NativeAsyncRuntime {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn invoke_batch_json(runtime: &PlexRuntime, events_json: &str) -> Result<String, RustPlexError> {
    let events: Vec<Document> = serde_json::from_str(events_json).map_err(|error| {
        RustPlexError::InvalidEvent(format!("invalid event batch JSON: {error}"))
    })?;
    let mut latest = serde_json::json!({"status": "unavailable"});
    for event in events {
        latest = runtime.invoke(event)?;
    }
    serde_json::to_string(&latest)
        .map_err(|error| RustPlexError::Runtime(format!("failed to serialize outcome: {error}")))
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
    module.add_class::<NativeAsyncRuntime>()?;
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
