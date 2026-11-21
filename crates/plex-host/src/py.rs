//! The host as the Python module `plex_host`, for the vLLM and SGLang
//! ports and the router.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::host::{Config, Host as Inner, tier_of};
use crate::tables::{COLUMNS, Refusal, Subject, Value};

#[derive(FromPyObject)]
enum PyValue {
    Boolean(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl From<PyValue> for Value {
    fn from(v: PyValue) -> Value {
        match v {
            PyValue::Boolean(b) => Value::Boolean(b),
            PyValue::Int(i) => Value::Int(i),
            PyValue::Float(f) => Value::Float(f),
            PyValue::Text(t) => Value::Text(t),
        }
    }
}

fn subject(kind: &str, id: &str) -> PyResult<Subject> {
    Subject::parse(kind, id).ok_or_else(|| PyValueError::new_err(format!("no subject kind {kind}")))
}

fn tier(name: &str) -> PyResult<crate::runtime::wit::Tier> {
    tier_of(name).ok_or_else(|| PyValueError::new_err(format!("no tier {name}")))
}

/// One scope's host: a worker's, or the cluster's.
#[pyclass(name = "Host")]
struct Host {
    inner: Inner,
}

#[pymethods]
impl Host {
    #[new]
    #[pyo3(signature = (supported=None, deadline_ms=50, interval_ms=0, lease_calls=3, manual=false))]
    fn new(
        supported: Option<Vec<String>>,
        deadline_ms: u64,
        interval_ms: u64,
        lease_calls: u32,
        manual: bool,
    ) -> Host {
        let supported =
            supported.unwrap_or_else(|| COLUMNS.iter().map(|c| c.to_string()).collect());
        Host {
            inner: Inner::new(Config {
                supported,
                deadline_ms,
                interval_ms,
                lease_calls,
                manual,
            }),
        }
    }

    /// Loads a policy component; the interfaces it imports.
    fn load(&self, py: Python<'_>, name: &str, wasm: &[u8]) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.load(name, wasm))
            .map_err(PyRuntimeError::new_err)
    }

    fn unload(&self, name: &str) -> bool {
        self.inner.unload(name)
    }

    fn start(&mut self) {
        self.inner.start();
    }

    fn stop(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.stop());
    }

    fn now(&self) -> u64 {
        self.inner.now()
    }

    fn set_now(&self, now: u64) {
        self.inner.set_now(now);
    }

    /// Runs a round if there is something to hear; with `force`, runs
    /// one regardless, calling every policy.
    #[pyo3(signature = (force=false))]
    fn run(&self, py: Python<'_>, force: bool) -> bool {
        py.detach(|| self.inner.run(force))
    }

    fn alarm(&self) -> Option<u64> {
        self.inner.alarm()
    }

    fn due(&self) -> bool {
        self.inner.due()
    }

    /// The sequence number of the last event reported.
    fn reported(&self) -> u64 {
        self.inner.reported()
    }

    /// The last sequence number every policy has heard (`Host::heard`).
    fn heard(&self) -> u64 {
        self.inner.heard()
    }

    fn version(&self) -> u64 {
        self.inner.version()
    }

    #[pyo3(signature = (id, session=None, tenant=None, worker=None, prompt_tokens=None, max_tokens=None, hints=None, age_ns=0))]
    #[allow(clippy::too_many_arguments)]
    fn arrived(
        &self,
        id: &str,
        session: Option<String>,
        tenant: Option<String>,
        worker: Option<String>,
        prompt_tokens: Option<u32>,
        max_tokens: Option<u32>,
        hints: Option<Vec<(String, PyValue)>>,
        age_ns: u64,
    ) -> u64 {
        let hints = hints
            .unwrap_or_default()
            .into_iter()
            .map(|(n, v)| (n, v.into()))
            .collect();
        self.inner.arrived(
            id,
            session,
            tenant,
            worker,
            prompt_tokens,
            max_tokens,
            hints,
            age_ns,
        )
    }

    fn dispatched(&self, id: &str, worker: &str) {
        self.inner.dispatched(id, worker);
    }

    fn started(&self, id: &str) {
        self.inner.started(id);
    }

    #[pyo3(signature = (id, by="host"))]
    fn paused(&self, id: &str, by: &str) {
        self.inner.paused(id, by);
    }

    fn first_token(&self, id: &str) {
        self.inner.first_token(id);
    }

    #[pyo3(signature = (id, outcome, prompt_tokens=None, cached_tokens=None, generated_tokens=0))]
    fn exited(
        &self,
        id: &str,
        outcome: &str,
        prompt_tokens: Option<u32>,
        cached_tokens: Option<u32>,
        generated_tokens: u32,
    ) {
        self.inner
            .exited(id, outcome, prompt_tokens, cached_tokens, generated_tokens);
    }

    /// `[(id, prompt-tokens, cached-tokens, generated-tokens)]` of live
    /// processes.
    fn tick(&self, counters: Vec<crate::host::Counter>) {
        self.inner.tick(counters);
    }

    fn kick(&self) {
        self.inner.kick();
    }

    fn closed(&self, session: &str) {
        self.inner.closed(session);
    }

    #[pyo3(signature = (kind, id, name, value=None))]
    fn hint(&self, kind: &str, id: &str, name: &str, value: Option<PyValue>) -> PyResult<()> {
        self.inner
            .hint(subject(kind, id)?, name, value.map(Into::into));
        Ok(())
    }

    #[pyo3(signature = (kind, id, name, value=None))]
    fn attr(&self, kind: &str, id: &str, name: &str, value: Option<PyValue>) -> PyResult<()> {
        self.inner
            .attr(subject(kind, id)?, name, value.map(Into::into));
        Ok(())
    }

    #[pyo3(signature = (writer, kind, id, name, value=None))]
    fn note(
        &self,
        writer: &str,
        kind: &str,
        id: &str,
        name: &str,
        value: Option<PyValue>,
    ) -> PyResult<()> {
        self.inner
            .note(writer, subject(kind, id)?, name, value.map(Into::into));
        Ok(())
    }

    /// `tiers` is `[(tier, capacity-tokens or None, used-tokens)]`.
    #[pyo3(signature = (id, tiers, running=0, waiting=0, draining=false, step_ns=None))]
    fn put_worker(
        &self,
        id: &str,
        tiers: Vec<(String, Option<u64>, u64)>,
        running: u32,
        waiting: u32,
        draining: bool,
        step_ns: Option<u64>,
    ) -> PyResult<()> {
        let tiers = tiers
            .into_iter()
            .map(|(t, c, u)| Ok((tier(&t)?, c, u)))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .put_worker(id, draining, tiers, running, waiting, step_ns);
        Ok(())
    }

    fn remove_worker(&self, id: &str) {
        self.inner.remove_worker(id);
    }

    /// `copies` is `[(worker, tier, tokens)]`.
    #[pyo3(signature = (id, tokens, copies, parent=None, owner=None, session=None, idle_ns=0))]
    #[allow(clippy::too_many_arguments)]
    fn put_working_set(
        &self,
        id: &str,
        tokens: u32,
        copies: Vec<(String, String, u32)>,
        parent: Option<String>,
        owner: Option<String>,
        session: Option<String>,
        idle_ns: u64,
    ) -> PyResult<()> {
        let copies = copies
            .into_iter()
            .map(|(w, t, n)| Ok((w, tier(&t)?, n)))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .put_working_set(id, parent, owner, session, tokens, idle_ns, copies);
        Ok(())
    }

    fn retained(&self, working_set: &str, worker: &str) {
        self.inner.retained(working_set, worker);
    }

    #[pyo3(signature = (working_set, worker, by="host", gone=true))]
    fn evicted(&self, working_set: &str, worker: &str, by: &str, gone: bool) {
        self.inner.evicted(working_set, worker, by, gone);
    }

    /// The port cannot enact a row (`why` an error-code name), or can
    /// again (`why` None).
    #[pyo3(signature = (column, key, tokens=None, why=None))]
    fn refuse(&self, column: &str, key: &str, tokens: Option<u32>, why: Option<&str>) -> bool {
        self.inner
            .refuse(column, key, tokens, why.map(Refusal::parse))
    }

    /// What changed since the last poll, as JSON text (`Host.poll` in the
    /// Rust documentation has its shape).
    fn poll(&self) -> String {
        self.inner.poll().to_string()
    }

    fn stats(&self) -> String {
        let s = self.inner.stats();
        serde_json::json!({
            // Which host built these numbers: "lazy-snapshot" since the
            // round stopped rebuilding the snapshot (2026-11-20).
            "host_build": crate::host::HOST_BUILD,
            "rounds": s.rounds, "calls": s.calls, "traps": s.traps, "deadlines": s.deadlines,
            "withdrawn": s.withdrawn, "call_ns": s.call_ns, "max_call_ns": s.max_call_ns,
            "commits": s.commits, "last_fault": s.last_fault,
        })
        .to_string()
    }

    fn policies(&self) -> Vec<(String, Vec<String>)> {
        self.inner.policies()
    }

    fn reset(&self) {
        self.inner.reset();
    }
}

#[pymodule]
fn plex_host(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Host>()?;
    m.add("COLUMNS", COLUMNS.to_vec())?;
    Ok(())
}
