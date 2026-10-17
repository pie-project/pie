//! A call is a transaction: one that traps or overruns stages nothing, and the
//! instance restarts with its scratch intact.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::OnceLock;
use std::time::Duration;

use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Store, StoreLimits, StoreLimitsBuilder, Trap};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use super::manifest::{Manifest, ManifestError};

pub(crate) mod bindings {
    wasmtime::component::bindgen!({
        path: "../policy/wit",
        world: "engine",
    });
}

use bindings::plex::policy::{engine_writes, reads, types, writes};
use types::{Edge, Level, Lookup, Refusal, Retain, Subject, SubjectKind, Value};

const TICK: Duration = Duration::from_millis(1);
const MEMORY_BYTES: usize = 64 << 20;
const MAX_ENTRIES: usize = 1 << 16;

pub trait Plane: Send {
    /// Stable for the life of the call.
    fn candidates(&self, kind: SubjectKind) -> Vec<String>;
    fn attr(&self, of: &Subject, name: &str) -> Lookup;
    fn prop(&self, of: &Subject, name: &str) -> Lookup;
    fn capability(&self, name: &str) -> Level;
    fn terminable(&self, process: &str) -> Result<(), Refusal>;
    fn discardable(&self, working_set: &str) -> Result<(), Refusal>;
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
    pub floor: Option<i64>,
}

impl Retention {
    fn of(entries: Vec<Retain>, floor: Option<i64>) -> Result<Self, Refusal> {
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
        Ok(Self { steps, floor })
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

/// `None`: not installed by this call; the one before it still stands.
#[derive(Debug, Default)]
pub struct Commit {
    pub schedule: Option<Table>,
    pub retention: Option<Retention>,
    pub terminate: Vec<(String, String)>,
    pub discard: Vec<String>,
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
    pre: bindings::EnginePre<HostState>,
    store: Store<HostState>,
    instance: bindings::Engine,
    alarm: Option<u64>,
}

impl Program {
    pub fn load(
        name: &str,
        component: &[u8],
        capability: &dyn Fn(&str) -> Level,
    ) -> Result<Self, LoadError> {
        let manifest = Manifest::of(component, capability)?;
        let engine = engine();
        let component =
            Component::new(engine, component).map_err(|e| LoadError::Compile(format!("{e:#}")))?;

        let mut linker = Linker::new(engine);
        // Policies target wasm32-wasip2 and so import WASI; the context is
        // empty.
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;
        bindings::Engine::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;
        let pre = linker
            .instantiate_pre(&component)
            .and_then(bindings::EnginePre::new)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;

        let (store, instance) =
            start(&pre, &manifest, HashMap::new()).map_err(LoadError::Instantiate)?;
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
        let result = self
            .instance
            .plex_policy_policy()
            .call_react(&mut self.store, &edges, now_ms);
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
        let scratch = &mut self.store.data_mut().scratch;
        for (name, val) in staged.puts {
            match val {
                Some(val) => {
                    scratch.insert(name, val);
                }
                None => {
                    scratch.remove(&name);
                }
            }
        }
        if let Some(at_ms) = staged.alarm {
            self.alarm = Some(at_ms);
        }
        Commit {
            schedule: staged.schedule,
            retention: staged.retention,
            terminate: staged.terminate,
            discard: staged.discard,
        }
    }

    fn restart(&mut self) -> Result<(), Fault> {
        let scratch = std::mem::take(&mut self.store.data_mut().scratch);
        let (store, instance) = start(&self.pre, &self.manifest, scratch).map_err(Fault::Dead)?;
        self.store = store;
        self.instance = instance;
        Ok(())
    }
}

fn start(
    pre: &bindings::EnginePre<HostState>,
    manifest: &Manifest,
    scratch: HashMap<String, Value>,
) -> Result<(Store<HostState>, bindings::Engine), String> {
    let mut store = Store::new(pre.engine(), HostState::new(manifest, scratch));
    store.limiter(|state| &mut state.limits);
    // A component's initializers are guest code, bounded like any call.
    store.set_epoch_deadline(ticks(manifest.deadline_ms));
    let instance = pre.instantiate(&mut store).map_err(|e| format!("{e:#}"))?;
    Ok((store, instance))
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
        Edge::Diverged(_) => "diverged",
        Edge::Timer => "timer",
        Edge::Dispatched(_) => "dispatched",
        Edge::WorkerAdded(_) => "worker-added",
        Edge::WorkerRemoved(_) => "worker-removed",
        Edge::WorkerDraining(_) => "worker-draining",
        Edge::Started(_) => "started",
        Edge::Paused(_) => "paused",
        Edge::Resumed(_) => "resumed",
        Edge::Retained(_) => "retained",
        Edge::Discarded(_) => "discarded",
    }
}

struct Grants {
    schedule: bool,
    retention: bool,
    terminate: bool,
    discard: bool,
}

struct HostState {
    wasi: WasiCtx,
    table: ResourceTable,
    limits: StoreLimits,
    grants: Grants,
    scratch: HashMap<String, Value>,
    call: Option<Call>,
}

impl HostState {
    fn new(manifest: &Manifest, scratch: HashMap<String, Value>) -> Self {
        Self {
            wasi: WasiCtxBuilder::new().build(),
            table: ResourceTable::new(),
            limits: StoreLimitsBuilder::new().memory_size(MEMORY_BYTES).build(),
            grants: Grants {
                schedule: manifest.grants("schedule"),
                retention: manifest.grants("retention"),
                terminate: manifest.grants("terminate"),
                discard: manifest.grants("discard"),
            },
            scratch,
            call: None,
        }
    }

    /// Imports reached outside a call (a component's initializers) read and
    /// write nothing.
    fn call(&mut self) -> Result<&mut Call, Refusal> {
        self.call.as_mut().ok_or(Refusal::WrongState)
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
    /// Each kind's ids, read once per call so pages agree.
    candidates: [Option<Vec<String>>; 4],
    staged: Staged,
}

impl Call {
    fn new(plane: Box<dyn Plane>) -> Self {
        Self {
            plane,
            candidates: Default::default(),
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
    puts: Vec<(String, Option<Value>)>,
    alarm: Option<u64>,
}

impl types::Host for HostState {}

impl reads::Host for HostState {
    fn candidates(&mut self, kind: SubjectKind, cursor: u32, n: u32) -> Vec<String> {
        let Ok(Call {
            plane, candidates, ..
        }) = self.call()
        else {
            return Vec::new();
        };
        let slot = match kind {
            SubjectKind::Session => 0,
            SubjectKind::Process => 1,
            SubjectKind::WorkingSet => 2,
            SubjectKind::Worker => 3,
        };
        candidates[slot]
            .get_or_insert_with(|| plane.candidates(kind))
            .iter()
            .skip(cursor as usize)
            .take(n as usize)
            .cloned()
            .collect()
    }

    fn attr(&mut self, of: Subject, name: String) -> Lookup {
        match self.call() {
            Ok(call) => call.plane.attr(&of, &name),
            Err(_) => Lookup::Absent,
        }
    }

    fn prop(&mut self, of: Subject, name: String) -> Lookup {
        match self.call() {
            Ok(call) => call.plane.prop(&of, &name),
            Err(_) => Lookup::Absent,
        }
    }

    fn capability(&mut self, name: String) -> Level {
        match self.call() {
            Ok(call) => call.plane.capability(&name),
            Err(_) => Level::Absent,
        }
    }

    fn get(&mut self, name: String) -> Option<Value> {
        // A call reads its own writes; they commit only when it returns.
        if let Some(call) = self.call.as_ref()
            && let Some((_, val)) = call.staged.puts.iter().rev().find(|(n, _)| *n == name)
        {
            return val.clone();
        }
        self.scratch.get(&name).cloned()
    }
}

impl writes::Host for HostState {
    fn put(&mut self, name: String, val: Option<Value>) {
        if let Ok(call) = self.call() {
            call.staged.puts.push((name, val));
        }
    }

    fn set_alarm(&mut self, at_ms: u64) {
        if let Ok(call) = self.call() {
            call.staged.alarm = Some(at_ms);
        }
    }

    fn terminate(&mut self, process: String, reason: String) -> Result<(), Refusal> {
        if !self.grants.terminate {
            return Err(Refusal::NotGranted);
        }
        let call = self.call()?;
        call.plane.terminable(&process)?;
        call.staged.terminate.push((process, reason));
        Ok(())
    }
}

impl engine_writes::Host for HostState {
    fn install_schedule(
        &mut self,
        entries: Vec<(String, i64)>,
        floor: Option<i64>,
    ) -> Result<(), Refusal> {
        if !self.grants.schedule {
            return Err(Refusal::NotGranted);
        }
        let table = Table::of(entries, floor)?;
        self.call()?.staged.schedule = Some(table);
        Ok(())
    }

    fn install_retention(
        &mut self,
        entries: Vec<Retain>,
        floor: Option<i64>,
    ) -> Result<(), Refusal> {
        if !self.grants.retention {
            return Err(Refusal::NotGranted);
        }
        let table = Retention::of(entries, floor)?;
        self.call()?.staged.retention = Some(table);
        Ok(())
    }

    fn discard(&mut self, working_set: String) -> Result<(), Refusal> {
        if !self.grants.discard {
            return Err(Refusal::NotGranted);
        }
        let call = self.call()?;
        call.plane.discardable(&working_set)?;
        call.staged.discard.push(working_set);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
