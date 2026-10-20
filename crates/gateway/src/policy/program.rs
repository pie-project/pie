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
        world: "cluster",
        exports: { default: async },
    });
}

use bindings::plex::policy::{cluster_writes, reads, state_writes, types, writes};
use types::{
    Accept, Edge, Placement, Replicate, Scale, Subject, SubjectKind, TransferClass, Value,
};

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
    pub fn key(&self, subject: &str) -> i64 {
        self.keys.get(subject).copied().unwrap_or(0)
    }
}

fn by_subject<T>(entries: Vec<(String, T)>) -> Result<HashMap<String, T>, Refusal> {
    if entries.len() > MAX_ENTRIES {
        return Err(Refusal::TooLarge);
    }
    let mut table = HashMap::with_capacity(entries.len());
    for (subject, entry) in entries {
        match table.entry(subject) {
            Entry::Occupied(_) => return Err(Refusal::DuplicateSubject),
            Entry::Vacant(slot) => {
                slot.insert(entry);
            }
        }
    }
    Ok(table)
}

/// The state tables a call installed. Each is enacted only as far as its
/// capability (`state.*`) says, and a policy reads `capability` to know.
#[derive(Debug, Default)]
pub struct StateCommit {
    pub replication: Option<Vec<Replicate>>,
    pub retention: Option<Table>,
    pub transfer_shares: Option<Vec<(TransferClass, u32)>>,
    pub acceptance: Option<Vec<Accept>>,
    pub discard: Vec<String>,
}

/// `None`: not installed by this call; the one before it still stands.
#[derive(Debug, Default)]
pub struct Commit {
    pub schedule: Option<Table>,
    pub placement: Option<HashMap<String, Placement>>,
    pub scale: Option<Vec<Scale>>,
    pub terminate: Vec<(String, String)>,
    pub state: StateCommit,
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
    #[error("`{0}` {1}")]
    Of(String, Box<LoadError>),
    #[error("`{first}` and `{second}` are both granted `{grant}`")]
    Shared {
        grant: String,
        first: String,
        second: String,
    },
}

pub struct Program {
    name: String,
    manifest: Manifest,
    pre: bindings::ClusterPre<HostState>,
    store: Store<HostState>,
    instance: bindings::Cluster,
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
        bindings::Cluster::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|e| LoadError::Instantiate(format!("{e:#}")))?;
        let pre = linker
            .instantiate_pre(&component)
            .and_then(bindings::ClusterPre::new)
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
            placement: staged.placement,
            scale: staged.scale,
            terminate: staged.terminate,
            state: staged.state,
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
    pre: &bindings::ClusterPre<HostState>,
    name: &str,
    manifest: &Manifest,
) -> Result<(Store<HostState>, bindings::Cluster), String> {
    let mut store = Store::new(pre.engine(), HostState::new(name, manifest));
    store.limiter(|state| &mut state.limits);
    // A component's initializers are guest code, bounded like any call.
    store.set_epoch_deadline(ticks(manifest.deadline_ms));
    let instance = block_on(pre.instantiate_async(&mut store)).map_err(|e| format!("{e:#}"))?;
    Ok((store, instance))
}

/// Loads programs on a thread of their own named `thread`, which then runs
/// `serve` with them; their manifests, or why one did not load, come back
/// here. No two may be granted the same write: each table has one author.
pub fn spawn(
    programs: Vec<(String, Vec<u8>)>,
    capability: fn(&str) -> bool,
    thread: &str,
    serve: impl FnOnce(Vec<Program>) + Send + 'static,
) -> Result<Vec<Manifest>, LoadError> {
    let (loaded, outcome) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name(thread.into())
        .spawn(move || {
            let programs = programs
                .iter()
                .map(|(name, component)| {
                    Program::load(name, component, &capability)
                        .map_err(|error| LoadError::Of(name.clone(), Box::new(error)))
                })
                .collect::<Result<Vec<_>, _>>()
                .and_then(|programs| {
                    let mut authors: HashMap<&str, &str> = HashMap::new();
                    for program in &programs {
                        for grant in &program.manifest.grants {
                            if let Some(first) = authors.insert(grant, &program.name) {
                                return Err(LoadError::Shared {
                                    grant: grant.clone(),
                                    first: first.to_string(),
                                    second: program.name.clone(),
                                });
                            }
                        }
                    }
                    Ok(programs)
                });
            match programs {
                Ok(programs) => {
                    let manifests = programs.iter().map(|p| p.manifest.clone()).collect();
                    let _ = loaded.send(Ok(manifests));
                    serve(programs);
                }
                Err(error) => {
                    let _ = loaded.send(Err(error));
                }
            }
        })
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

/// The cluster plane's own engine, apart from any the process hosts inferlets on:
/// epoch interruption there would put a deadline on every inferlet store.
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
            .name("pie-cluster-epoch".into())
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
        Edge::Migrated(_) => "migrated",
        Edge::MigrationFailed(_) => "migration-failed",
        Edge::Copied(_) => "copied",
        Edge::Dropped(_) => "dropped",
        Edge::TierPressure(_) => "tier-pressure",
        Edge::LinkChanged(_) => "link-changed",
        Edge::Started(_) => "started",
        Edge::Paused(_) => "paused",
        Edge::Retained(_) => "retained",
        Edge::Discarded(_) => "discarded",
        Edge::Offloaded(_) => "offloaded",
        Edge::OffloadFailed(_) => "offload-failed",
    }
}

struct Grants {
    schedule: bool,
    placement: bool,
    scale: bool,
    terminate: bool,
    replication: bool,
    retention: bool,
    transfer_shares: bool,
    acceptance: bool,
    discard: bool,
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
                placement: manifest.grants("placement"),
                scale: manifest.grants("scale"),
                terminate: manifest.grants("terminate"),
                replication: manifest.grants("replication"),
                retention: manifest.grants("retention"),
                transfer_shares: manifest.grants("transfer-shares"),
                acceptance: manifest.grants("acceptance"),
                discard: manifest.grants("discard"),
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
            tracing::warn!(policy = %self.name, verb, %why, "cluster policy write refused");
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
    placement: Option<HashMap<String, Placement>>,
    scale: Option<Vec<Scale>>,
    terminate: Vec<(String, String)>,
    state: StateCommit,
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

impl cluster_writes::Host for HostState {
    fn install_schedule(&mut self, entries: Vec<(String, i64)>, floor: Option<i64>) {
        self.write("install-schedule", |state| {
            if !state.grants.schedule {
                return Err(Refusal::NotGranted);
            }
            let keys = by_subject(entries)?;
            state.call()?.staged.schedule = Some(Table { keys, floor });
            Ok(())
        });
    }

    fn install_placement(&mut self, entries: Vec<Placement>) {
        self.write("install-placement", |state| {
            if !state.grants.placement {
                return Err(Refusal::NotGranted);
            }
            let entries = entries.into_iter().map(|p| (p.process.clone(), p));
            let table = by_subject(entries.collect())?;
            state.call()?.staged.placement = Some(table);
            Ok(())
        });
    }

    fn install_scale(&mut self, entries: Vec<Scale>) {
        self.write("install-scale", |state| {
            if !state.grants.scale {
                return Err(Refusal::NotGranted);
            }
            if entries.len() > MAX_ENTRIES {
                return Err(Refusal::TooLarge);
            }
            let shapes = entries.iter().map(|s| (s.model.clone(), s.engine.clone()));
            if shapes.collect::<std::collections::HashSet<_>>().len() != entries.len() {
                return Err(Refusal::DuplicateSubject);
            }
            state.call()?.staged.scale = Some(entries);
            Ok(())
        });
    }
}

impl state_writes::Host for HostState {
    fn install_replication(&mut self, entries: Vec<Replicate>) {
        self.write("install-replication", |state| {
            if !state.grants.replication {
                return Err(Refusal::NotGranted);
            }
            if entries.len() > MAX_ENTRIES {
                return Err(Refusal::TooLarge);
            }
            let groups = entries.iter().map(|r| (r.group.clone(), r.upto));
            if groups.collect::<std::collections::HashSet<_>>().len() != entries.len() {
                return Err(Refusal::DuplicateSubject);
            }
            state.call()?.staged.state.replication = Some(entries);
            Ok(())
        });
    }

    fn install_retention(&mut self, entries: Vec<(String, i64)>) {
        self.write("install-retention", |state| {
            if !state.grants.retention {
                return Err(Refusal::NotGranted);
            }
            let keys = by_subject(entries)?;
            state.call()?.staged.state.retention = Some(Table { keys, floor: None });
            Ok(())
        });
    }

    fn install_transfer_shares(&mut self, entries: Vec<(TransferClass, u32)>) {
        self.write("install-transfer-shares", |state| {
            if !state.grants.transfer_shares {
                return Err(Refusal::NotGranted);
            }
            let classes = entries.iter().map(|(class, _)| *class as u8);
            if classes.collect::<std::collections::HashSet<_>>().len() != entries.len() {
                return Err(Refusal::DuplicateSubject);
            }
            state.call()?.staged.state.transfer_shares = Some(entries);
            Ok(())
        });
    }

    fn install_acceptance(&mut self, entries: Vec<Accept>) {
        self.write("install-acceptance", |state| {
            if !state.grants.acceptance {
                return Err(Refusal::NotGranted);
            }
            if entries.len() > MAX_ENTRIES {
                return Err(Refusal::TooLarge);
            }
            state.call()?.staged.state.acceptance = Some(entries);
            Ok(())
        });
    }

    fn discard(&mut self, group: String) {
        self.write("discard", |state| {
            if !state.grants.discard {
                return Err(Refusal::NotGranted);
            }
            state.call()?.staged.state.discard.push(group);
            Ok(())
        });
    }
}

#[cfg(test)]
mod tests;
