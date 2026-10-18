//! A call is a transaction: one that traps or overruns stages nothing, and a
//! fresh instance takes the next call.
//!
//! A policy is a WASI 0.3 component, and WASI 0.3 is async: a program runs
//! its calls to completion on the thread that loaded it, which runs no other
//! async runtime (`spawn` gives it one).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;
use std::sync::OnceLock;
use std::time::Duration;

use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Store, StoreLimits, StoreLimitsBuilder, Trap};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use super::manifest::{Manifest, ManifestError};

pub(crate) mod bindings {
    wasmtime::component::bindgen!({
        path: "../policy/wit",
        world: "worker",
        exports: { default: async },
    });
}

use bindings::plex::policy::{reads, types, worker_writes, writes};
use types::{Edge, Offload, Retain, Subject, SubjectKind, Value};

const TICK: Duration = Duration::from_millis(1);
const MEMORY_BYTES: usize = 64 << 20;
const MAX_ENTRIES: usize = 1 << 16;

pub trait Plane: Send {
    /// Stable for the life of the call.
    fn candidates(&self, kind: SubjectKind) -> Vec<String>;
    fn attr(&self, of: &Subject, name: &str) -> Option<Value>;
    fn prop(&self, of: &Subject, name: &str) -> Option<Value>;
    fn capability(&self, name: &str) -> bool;
    /// Whether `terminate` applies to `process` now.
    fn terminable(&self, process: &str) -> bool;
    /// Whether `discard` applies to `working_set` now: retained, not live.
    fn discardable(&self, working_set: &str) -> bool;
}

/// Why a write was not taken. The program is not told; the host logs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error("the manifest does not grant it")]
    NotGranted,
    #[error("a table names the same subject twice")]
    DuplicateSubject,
    #[error("a table is too large")]
    TooLarge,
    #[error("its subject is not in a state it applies to")]
    WrongState,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    pub keys: HashMap<String, i64>,
    pub floor: Option<i64>,
}

impl Table {
    fn of(entries: Vec<(String, i64)>, floor: Option<i64>) -> Result<Self, Refusal> {
        if entries.len() > MAX_ENTRIES {
            return Err(Refusal::TooLarge);
        }
        let mut keys = HashMap::with_capacity(entries.len());
        for (subject, key) in entries {
            match keys.entry(subject) {
                Entry::Occupied(_) => return Err(Refusal::DuplicateSubject),
                Entry::Vacant(slot) => {
                    slot.insert(key);
                }
            }
        }
        Ok(Self { keys, floor })
    }

    pub fn key(&self, subject: &str) -> i64 {
        self.keys.get(subject).copied().unwrap_or(0)
    }
}

/// Each working-set's entries as `(upto, key)`, `None` covering all of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Retention {
    pub steps: HashMap<String, Vec<(Option<u32>, i64)>>,
}

impl Retention {
    fn of(entries: Vec<Retain>) -> Result<Self, Refusal> {
        if entries.len() > MAX_ENTRIES {
            return Err(Refusal::TooLarge);
        }
        let mut steps: HashMap<String, Vec<(Option<u32>, i64)>> = HashMap::new();
        for Retain {
            working_set,
            key,
            upto,
        } in entries
        {
            let steps = steps.entry(working_set).or_default();
            if steps.iter().any(|(at, _)| *at == upto) {
                return Err(Refusal::DuplicateSubject);
            }
            steps.push((upto, key));
        }
        Ok(Self { steps })
    }

    /// The key at token `at` of `working_set`: the highest entry covering
    /// it, else 0.
    pub fn key_at(&self, working_set: &str, at: u32) -> i64 {
        self.steps
            .get(working_set)
            .into_iter()
            .flatten()
            .filter(|(upto, _)| upto.is_none_or(|upto| at < upto))
            .map(|&(_, key)| key)
            .max()
            .unwrap_or(0)
    }

    pub fn key(&self, working_set: &str) -> i64 {
        self.key_at(working_set, 0)
    }
}

/// Where each process's passes may run: `default` for every process no entry
/// names.
#[derive(Debug, Clone, Default)]
pub struct Offloads {
    pub entries: HashMap<String, Offload>,
    pub default: Option<Offload>,
}

/// `None`: not installed by this call; the one before it still stands.
#[derive(Debug, Default)]
pub struct Commit {
    pub schedule: Option<Table>,
    pub retention: Option<Retention>,
    pub terminate: Vec<(String, String)>,
    pub discard: Vec<String>,
    pub offload: Option<Offloads>,
    /// `(subject, name, value)`, in call order; `None` withdraws.
    pub publish: Vec<(Subject, String, Option<Value>)>,
}

#[derive(Debug, thiserror::Error)]
pub enum Fault {
    #[error("overran its {0} ms deadline")]
    Deadline(u64),
    #[error("trapped: {0}")]
    Trap(String),
    /// Its tables must revert to empty.
    #[error("could not be restarted: {0}")]
    Dead(String),
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("does not compile: {0}")]
    Compile(String),
    #[error("does not instantiate: {0}")]
    Instantiate(String),
}

pub struct Program {
    name: String,
    manifest: Manifest,
    pre: bindings::WorkerPre<HostState>,
    store: Store<HostState>,
    instance: bindings::Worker,
    alarm: Option<u64>,
}

impl Program {
    pub fn load(
        name: &str,
        component: &[u8],
        capability: &dyn Fn(&str) -> bool,
    ) -> Result<Self, LoadError> {
        let manifest = Manifest::of(component, capability)?;
        let engine = engine();
        let component =
            Component::new(engine, component).map_err(|e| LoadError::Compile(format!("{e:#}")))?;

        let mut linker = Linker::new(engine);
        // Policies target wasm32-wasip3 and so import WASI 0.3; the context
        // is empty.
        wasmtime_wasi::p3::add_to_linker(&mut linker)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;
        bindings::Worker::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;
        let pre = linker
            .instantiate_pre(&component)
            .and_then(bindings::WorkerPre::new)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;

        let (store, instance) = start(&pre, name, &manifest).map_err(LoadError::Instantiate)?;
        Ok(Self {
            name: name.to_string(),
            manifest,
            pre,
            store,
            instance,
            alarm: None,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn alarm(&self) -> Option<u64> {
        self.alarm
    }

    /// A call with no such edge does not run.
    pub fn react(
        &mut self,
        plane: Box<dyn Plane>,
        edges: Vec<Edge>,
        now_ms: u64,
    ) -> Result<Commit, Fault> {
        let edges: Vec<Edge> = edges
            .into_iter()
            .filter(|edge| self.manifest.subscribes(edge_name(edge)))
            .collect();
        if edges.is_empty() {
            return Ok(Commit::default());
        }
        if edges.iter().any(|edge| matches!(edge, Edge::Timer)) {
            self.alarm = None;
        }

        self.store.data_mut().call = Some(Call::new(plane));
        self.store
            .set_epoch_deadline(ticks(self.manifest.deadline_ms));
        let result = block_on(self.instance.plex_policy_policy().call_react(
            &mut self.store,
            &edges,
            now_ms,
        ));
        let call = self
            .store
            .data_mut()
            .call
            .take()
            .expect("the call react opened");

        match result {
            Ok(()) => Ok(self.commit(call.staged)),
            Err(error) => {
                let fault = if error.downcast_ref::<Trap>() == Some(&Trap::Interrupt) {
                    Fault::Deadline(self.manifest.deadline_ms)
                } else {
                    Fault::Trap(format!("{error:#}"))
                };
                // A trapped instance may be mid-way through anything; only a
                // fresh one is safe to call again.
                self.restart()?;
                Err(fault)
            }
        }
    }

    fn commit(&mut self, staged: Staged) -> Commit {
        if let Some(at_ms) = staged.alarm {
            self.alarm = Some(at_ms);
        }
        Commit {
            schedule: staged.schedule,
            retention: staged.retention,
            terminate: staged.terminate,
            discard: staged.discard,
            offload: staged.offload,
            publish: staged.publish,
        }
    }

    fn restart(&mut self) -> Result<(), Fault> {
        let (store, instance) =
            start(&self.pre, &self.name, &self.manifest).map_err(Fault::Dead)?;
        self.store = store;
        self.instance = instance;
        Ok(())
    }
}

fn start(
    pre: &bindings::WorkerPre<HostState>,
    name: &str,
    manifest: &Manifest,
) -> Result<(Store<HostState>, bindings::Worker), String> {
    let mut store = Store::new(pre.engine(), HostState::new(name, manifest));
    store.limiter(|state| &mut state.limits);
    // A component's initializers are guest code, bounded like any call.
    store.set_epoch_deadline(ticks(manifest.deadline_ms));
    let instance = block_on(pre.instantiate_async(&mut store)).map_err(|e| format!("{e:#}"))?;
    Ok((store, instance))
}

/// Loads a program on a thread of its own named `thread`, which then runs
/// `serve` with it; the manifest, or why it did not load, comes back here.
pub fn spawn(
    name: &str,
    component: &[u8],
    capability: fn(&str) -> bool,
    thread: &str,
    serve: impl FnOnce(Program) + Send + 'static,
) -> Result<Manifest, LoadError> {
    let (name, component) = (name.to_string(), component.to_vec());
    let (loaded, outcome) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name(thread.into())
        .spawn(
            move || match Program::load(&name, &component, &capability) {
                Ok(program) => {
                    let _ = loaded.send(Ok(program.manifest().clone()));
                    serve(program);
                }
                Err(error) => {
                    let _ = loaded.send(Err(error));
                }
            },
        )
        .map_err(|e| LoadError::Instantiate(format!("spawning its thread: {e}")))?;
    outcome
        .recv()
        .map_err(|_| LoadError::Instantiate("its thread ended before it loaded".into()))?
}

/// Runs a call to completion on this thread, on a runtime of its own.
fn block_on<F: Future>(future: F) -> F::Output {
    thread_local! {
        static RUNTIME: tokio::runtime::Runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a policy thread's runtime");
    }
    RUNTIME.with(|runtime| runtime.block_on(future))
}

fn ticks(deadline_ms: u64) -> u64 {
    // One more than the deadline: the first tick can come at once.
    deadline_ms.div_ceil(TICK.as_millis() as u64) + 1
}

/// One engine for every policy in the process, apart from the inferlets':
/// epoch interruption here would put a deadline on every inferlet store.
pub(super) fn engine() -> &'static wasmtime::Engine {
    static ENGINE: OnceLock<wasmtime::Engine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let mut config = wasmtime::Config::new();
        config.wasm_component_model(true);
        config.wasm_component_model_async(true);
        config.epoch_interruption(true);
        // Relaxed SIMD may lower differently per CPU, so two workers could
        // disagree about a policy's arithmetic.
        config.relaxed_simd_deterministic(true);
        let engine = wasmtime::Engine::new(&config).expect("the policy engine's configuration");
        let ticker = engine.clone();
        std::thread::Builder::new()
            .name("pie-policy-epoch".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(TICK);
                    ticker.increment_epoch();
                }
            })
            .expect("spawn the policy epoch ticker");
        engine
    })
}

fn edge_name(edge: &Edge) -> &'static str {
    match edge {
        Edge::Arrived(_) => "arrived",
        Edge::Exited(_) => "exited",
        Edge::PropChanged(_) => "prop-changed",
        Edge::Closed(_) => "closed",
        Edge::Timer => "timer",
        Edge::Dispatched(_) => "dispatched",
        Edge::WorkerRemoved(_) => "worker-removed",
        Edge::WorkerDraining(_) => "worker-draining",
        Edge::Started(_) => "started",
        Edge::Paused(_) => "paused",
        Edge::Retained(_) => "retained",
        Edge::Discarded(_) => "discarded",
        Edge::Migrated(_) => "migrated",
        Edge::MigrationFailed(_) => "migration-failed",
        Edge::Copied(_) => "copied",
        Edge::Dropped(_) => "dropped",
        Edge::TierPressure(_) => "tier-pressure",
        Edge::LinkChanged(_) => "link-changed",
        Edge::Offloaded(_) => "offloaded",
        Edge::OffloadFailed(_) => "offload-failed",
    }
}

struct Grants {
    schedule: bool,
    retention: bool,
    terminate: bool,
    discard: bool,
    offload: bool,
    publish: bool,
}

struct HostState {
    name: String,
    wasi: WasiCtx,
    table: ResourceTable,
    limits: StoreLimits,
    grants: Grants,
    call: Option<Call>,
}

impl HostState {
    fn new(name: &str, manifest: &Manifest) -> Self {
        Self {
            name: name.to_string(),
            wasi: WasiCtxBuilder::new().build(),
            table: ResourceTable::new(),
            limits: StoreLimitsBuilder::new().memory_size(MEMORY_BYTES).build(),
            grants: Grants {
                schedule: manifest.grants("schedule"),
                retention: manifest.grants("retention"),
                terminate: manifest.grants("terminate"),
                discard: manifest.grants("discard"),
                offload: manifest.grants("offload"),
                publish: manifest.grants("publish"),
            },
            call: None,
        }
    }

    /// Imports reached outside a call (a component's initializers) read and
    /// write nothing.
    fn call(&mut self) -> Result<&mut Call, Refusal> {
        self.call.as_mut().ok_or(Refusal::WrongState)
    }

    /// Stages a write, or logs why it was not taken.
    fn write(&mut self, verb: &str, write: impl FnOnce(&mut Self) -> Result<(), Refusal>) {
        if let Err(why) = write(self) {
            tracing::warn!(policy = %self.name, verb, %why, "worker policy write refused");
        }
    }
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

struct Call {
    plane: Box<dyn Plane>,
    staged: Staged,
}

impl Call {
    fn new(plane: Box<dyn Plane>) -> Self {
        Self {
            plane,
            staged: Staged::default(),
        }
    }
}

#[derive(Default)]
struct Staged {
    schedule: Option<Table>,
    retention: Option<Retention>,
    terminate: Vec<(String, String)>,
    discard: Vec<String>,
    offload: Option<Offloads>,
    publish: Vec<(Subject, String, Option<Value>)>,
    alarm: Option<u64>,
}

impl types::Host for HostState {}

impl reads::Host for HostState {
    fn candidates(&mut self, kind: SubjectKind) -> Vec<String> {
        self.call()
            .map(|call| call.plane.candidates(kind))
            .unwrap_or_default()
    }

    fn attr(&mut self, of: Subject, name: String) -> Option<Value> {
        self.call().ok()?.plane.attr(&of, &name)
    }

    fn prop(&mut self, of: Subject, name: String) -> Option<Value> {
        self.call().ok()?.plane.prop(&of, &name)
    }

    fn capability(&mut self, name: String) -> bool {
        self.call().is_ok_and(|call| call.plane.capability(&name))
    }
}

impl writes::Host for HostState {
    fn set_alarm(&mut self, at_ms: u64) {
        if let Ok(call) = self.call() {
            call.staged.alarm = Some(at_ms);
        }
    }

    fn terminate(&mut self, process: String, reason: String) {
        self.write("terminate", |state| {
            if !state.grants.terminate {
                return Err(Refusal::NotGranted);
            }
            let call = state.call()?;
            if !call.plane.terminable(&process) {
                return Err(Refusal::WrongState);
            }
            call.staged.terminate.push((process, reason));
            Ok(())
        });
    }
}

impl worker_writes::Host for HostState {
    fn install_schedule(&mut self, entries: Vec<(String, i64)>, floor: Option<i64>) {
        self.write("install-schedule", |state| {
            if !state.grants.schedule {
                return Err(Refusal::NotGranted);
            }
            let table = Table::of(entries, floor)?;
            state.call()?.staged.schedule = Some(table);
            Ok(())
        });
    }

    fn install_retention(&mut self, entries: Vec<Retain>) {
        self.write("install-retention", |state| {
            if !state.grants.retention {
                return Err(Refusal::NotGranted);
            }
            let table = Retention::of(entries)?;
            state.call()?.staged.retention = Some(table);
            Ok(())
        });
    }

    fn discard(&mut self, working_set: String) {
        self.write("discard", |state| {
            if !state.grants.discard {
                return Err(Refusal::NotGranted);
            }
            let call = state.call()?;
            if !call.plane.discardable(&working_set) {
                return Err(Refusal::WrongState);
            }
            call.staged.discard.push(working_set);
            Ok(())
        });
    }

    fn install_offload(&mut self, entries: Vec<(String, Offload)>, default: Option<Offload>) {
        self.write("install-offload", |state| {
            if !state.grants.offload {
                return Err(Refusal::NotGranted);
            }
            if entries.len() > MAX_ENTRIES {
                return Err(Refusal::TooLarge);
            }
            let mut table = HashMap::with_capacity(entries.len());
            for (process, offload) in entries {
                match table.entry(process) {
                    Entry::Occupied(_) => return Err(Refusal::DuplicateSubject),
                    Entry::Vacant(slot) => {
                        slot.insert(offload);
                    }
                }
            }
            state.call()?.staged.offload = Some(Offloads {
                entries: table,
                default,
            });
            Ok(())
        });
    }

    fn publish(&mut self, of: Subject, name: String, val: Option<Value>) {
        self.write("publish", |state| {
            if !state.grants.publish {
                return Err(Refusal::NotGranted);
            }
            // This worker, or a process on it.
            if !matches!(of, Subject::Worker(_) | Subject::Process(_)) {
                return Err(Refusal::WrongState);
            }
            let call = state.call()?;
            if call.staged.publish.len() >= MAX_ENTRIES {
                return Err(Refusal::TooLarge);
            }
            call.staged.publish.push((of, name, val));
            Ok(())
        });
    }
}

#[cfg(test)]
mod tests;
