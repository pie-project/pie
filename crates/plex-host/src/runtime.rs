//! A policy as a wasm component: its imports, served from one call's
//! snapshot and staged writes, and its one export.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use wasmtime::component::{Component, HasSelf, Linker};
use wasmtime::{Engine, Store, StoreLimits, StoreLimitsBuilder, Trap};

use crate::tables::{self, MAX_ROWS, Prefix, Refusal, Staged, Subject, Tables, TierSet};

wasmtime::component::bindgen!({
    path: "wit",
    inline: "
        package plex:host;
        world policy-host {
            include plex:policy/worker@1.1.0;
            include plex:policy/cluster@1.1.0;
        }
    ",
    world: "plex:host/policy-host",
});

pub use plex::core::snapshot as snap;
pub use plex::core::types as wit;

/// The memory one policy instance may use.
pub const MEMORY_BYTES: usize = 64 << 20;

/// What one call reads: the host's state as of its start.
#[derive(Default)]
pub struct Snapshot {
    pub processes: Vec<snap::Process>,
    pub process_index: HashMap<String, usize>,
    pub workers: Vec<snap::Worker>,
    pub worker_index: HashMap<String, usize>,
    pub working_sets: Vec<snap::WorkingSet>,
    pub working_set_index: HashMap<String, usize>,
    pub hints: HashMap<Subject, HashMap<String, tables::Value>>,
    pub attrs: HashMap<Subject, HashMap<String, tables::Value>>,
}

impl Snapshot {
    pub fn index(&mut self) {
        self.process_index = self
            .processes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id.clone(), i))
            .collect();
        self.worker_index = self
            .workers
            .iter()
            .enumerate()
            .map(|(i, w)| (w.id.clone(), i))
            .collect();
        self.working_set_index = self
            .working_sets
            .iter()
            .enumerate()
            .map(|(i, w)| (w.id.clone(), i))
            .collect();
    }

    fn live_process(&self, id: &str) -> bool {
        self.process_index
            .get(id)
            .is_some_and(|i| self.processes[*i].state != wit::ProcessState::Exited)
    }
}

/// One call in progress.
pub struct Call {
    pub policy: String,
    pub snapshot: Arc<Snapshot>,
    pub tables: Arc<Mutex<Tables>>,
    pub staged: Staged,
}

pub struct Ctx {
    limits: StoreLimits,
    call: Option<Call>,
}

impl Ctx {
    fn call(&mut self) -> &mut Call {
        self.call
            .as_mut()
            .expect("an import is only called inside react")
    }
}

fn subject(s: wit::Subject) -> Subject {
    match s {
        wit::Subject::Global => Subject::Global,
        wit::Subject::Process(id) => Subject::Process(id),
        wit::Subject::Session(id) => Subject::Session(id),
        wit::Subject::Worker(id) => Subject::Worker(id),
        wit::Subject::WorkingSet(id) => Subject::WorkingSet(id),
    }
}

pub fn wit_subject(s: &Subject) -> wit::Subject {
    match s {
        Subject::Global => wit::Subject::Global,
        Subject::Process(id) => wit::Subject::Process(id.clone()),
        Subject::Session(id) => wit::Subject::Session(id.clone()),
        Subject::Worker(id) => wit::Subject::Worker(id.clone()),
        Subject::WorkingSet(id) => wit::Subject::WorkingSet(id.clone()),
    }
}

fn value(v: &tables::Value) -> wit::Value {
    match v {
        tables::Value::Boolean(b) => wit::Value::Boolean(*b),
        tables::Value::Int(i) => wit::Value::Int(*i),
        tables::Value::Float(f) => wit::Value::Float(*f),
        tables::Value::Text(t) => wit::Value::Text(t.clone()),
    }
}

fn own_value(v: wit::Value) -> tables::Value {
    match v {
        wit::Value::Boolean(b) => tables::Value::Boolean(b),
        wit::Value::Int(i) => tables::Value::Int(i),
        wit::Value::Float(f) => tables::Value::Float(f),
        wit::Value::Text(t) => tables::Value::Text(t),
    }
}

fn error(r: Refusal) -> wit::ErrorCode {
    match r {
        Refusal::NotSupported => wit::ErrorCode::NotSupported,
        Refusal::InvalidArgument => wit::ErrorCode::InvalidArgument,
        Refusal::LimitExceeded => wit::ErrorCode::LimitExceeded,
        Refusal::Other => wit::ErrorCode::Other(None),
    }
}

pub fn tier_bit(t: wit::Tier) -> TierSet {
    match t {
        wit::Tier::Device => 1,
        wit::Tier::Host => 2,
        wit::Tier::Disk => 4,
    }
}

fn tier_set(tiers: Vec<wit::Tier>) -> TierSet {
    tiers.into_iter().fold(0, |set, t| set | tier_bit(t))
}

fn tier_list(set: TierSet) -> Vec<wit::Tier> {
    [wit::Tier::Device, wit::Tier::Host, wit::Tier::Disk]
        .into_iter()
        .filter(|t| set & tier_bit(*t) != 0)
        .collect()
}

fn prefix(p: wit::Prefix) -> Prefix {
    (p.working_set, p.tokens)
}

fn wit_prefix(p: &Prefix) -> wit::Prefix {
    wit::Prefix {
        working_set: p.0.clone(),
        tokens: p.1,
    }
}

/// Whether every worker a row names is one the host knows.
fn known_workers(snapshot: &Snapshot, workers: &[String]) -> bool {
    workers
        .iter()
        .all(|w| snapshot.worker_index.contains_key(w))
}

/// A column's seven functions, over one field of the tables. `live` says
/// whether a key's subject still exists (a write to one that does not is
/// ignored); `valid` whether a value can be taken.
macro_rules! column {
    (
        $trait:path, $field:ident,
        key: $wkey:ty => |$k:ident| $key:expr, back: |$bk:ident| $back:expr,
        live: |$snap:ident, $lk:ident| $live:expr,
        value: $wval:ty => |$v:ident| $val:expr, back: |$bv:ident| $vback:expr,
        valid: |$vsnap:ident, $vv:ident| $valid:expr
    ) => {
        impl $trait for Ctx {
            fn get(&mut self, key: $wkey) -> Option<$wval> {
                let call = self.call();
                let key = {
                    let $k = key;
                    $key
                };
                let tables = call.tables.lock().unwrap();
                call.staged
                    .$field
                    .get(&tables.$field, &key)
                    .map(|$bv| $vback)
            }

            fn set(&mut self, key: $wkey, value: $wval) -> Result<(), wit::ErrorCode> {
                self.set_list(vec![(key, value)])
            }

            fn set_list(&mut self, entries: Vec<($wkey, $wval)>) -> Result<(), wit::ErrorCode> {
                let call = self.call();
                let mut rows = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    let key = {
                        let $k = key;
                        $key
                    };
                    let value = {
                        let $v = value;
                        $val
                    };
                    // A row naming something that is gone (its subject, or a
                    // worker that left) is ignored, as a stale id is.
                    let ok = {
                        let ($vsnap, $vv) = (&*call.snapshot, &value);
                        $valid
                    };
                    let ($snap, $lk) = (&*call.snapshot, &key);
                    if ok && $live {
                        rows.push((key, value));
                    }
                }
                let committed = call.tables.lock().unwrap();
                if call.staged.$field.len(&committed.$field) + rows.len() > MAX_ROWS {
                    return Err(wit::ErrorCode::LimitExceeded);
                }
                drop(committed);
                for (key, value) in rows {
                    call.staged.$field.set(key, value);
                }
                Ok(())
            }

            fn delete(&mut self, key: $wkey) {
                let key = {
                    let $k = key;
                    $key
                };
                self.call().staged.$field.delete(key);
            }

            fn set_default(&mut self, value: $wval) -> Result<(), wit::ErrorCode> {
                let call = self.call();
                let value = {
                    let $v = value;
                    $val
                };
                {
                    let ($vsnap, $vv) = (&*call.snapshot, &value);
                    if !$valid {
                        return Err(wit::ErrorCode::InvalidArgument);
                    }
                }
                call.staged.$field.set_default(value);
                Ok(())
            }

            fn clear(&mut self) {
                self.call().staged.$field.clear();
            }

            fn get_refused(&mut self) -> Vec<($wkey, wit::ErrorCode)> {
                let call = self.call();
                let tables = call.tables.lock().unwrap();
                tables
                    .$field
                    .refused
                    .iter()
                    .map(|($bk, why)| ($back, error(*why)))
                    .collect()
            }
        }
    };
}

column!(plex::schedule::priority::Host, schedule_priority,
    key: String => |k| k, back: |k| k.clone(),
    live: |s, k| s.live_process(k),
    value: i64 => |v| v, back: |v| v,
    valid: |_s, _v| true);

column!(plex::schedule::hold::Host, schedule_hold,
    key: String => |k| k, back: |k| k.clone(),
    live: |s, k| s.live_process(k),
    value: bool => |v| v, back: |v| v,
    valid: |_s, _v| true);

column!(plex::retention::priority::Host, retention_priority,
    key: wit::Prefix => |k| prefix(k), back: |k| wit_prefix(k),
    live: |s, k| s.working_set_index.contains_key(&k.0),
    value: i64 => |v| v, back: |v| v,
    valid: |_s, _v| true);

column!(plex::retention::tiers::Host, retention_tiers,
    key: wit::Prefix => |k| prefix(k), back: |k| wit_prefix(k),
    live: |s, k| s.working_set_index.contains_key(&k.0),
    value: Vec<wit::Tier> => |v| tier_set(v), back: |v| tier_list(v),
    valid: |_s, _v| true);

column!(plex::placement::workers::Host, placement_workers,
    key: String => |k| k, back: |k| k.clone(),
    live: |s, k| s.live_process(k),
    value: Vec<String> => |v| v, back: |v| v,
    valid: |s, v| known_workers(s, v));

column!(plex::placement::helpers::Host, placement_helpers,
    key: String => |k| k, back: |k| k.clone(),
    live: |s, k| s.live_process(k),
    value: Vec<String> => |v| v, back: |v| v,
    valid: |s, v| known_workers(s, v));

column!(plex::replication::workers::Host, replication_workers,
    key: wit::Prefix => |k| prefix(k), back: |k| wit_prefix(k),
    live: |s, k| s.working_set_index.contains_key(&k.0),
    value: Vec<String> => |v| v, back: |v| v,
    valid: |s, v| known_workers(s, v));

impl plex::core::types::Host for Ctx {}
impl plex::schedule::types::Host for Ctx {}
impl plex::retention::types::Host for Ctx {}

impl snap::Host for Ctx {
    fn get_processes(&mut self) -> Vec<snap::Process> {
        self.call().snapshot.processes.clone()
    }

    fn get_workers(&mut self) -> Vec<snap::Worker> {
        self.call().snapshot.workers.clone()
    }

    fn get_working_sets(&mut self) -> Vec<snap::WorkingSet> {
        self.call().snapshot.working_sets.clone()
    }

    fn get_process(&mut self, id: String) -> Option<snap::Process> {
        let s = &self.call().snapshot;
        s.process_index.get(&id).map(|i| s.processes[*i].clone())
    }

    fn get_worker(&mut self, id: String) -> Option<snap::Worker> {
        let s = &self.call().snapshot;
        s.worker_index.get(&id).map(|i| s.workers[*i].clone())
    }

    fn get_working_set(&mut self, id: String) -> Option<snap::WorkingSet> {
        let s = &self.call().snapshot;
        s.working_set_index
            .get(&id)
            .map(|i| s.working_sets[*i].clone())
    }

    fn get_attr(&mut self, of: wit::Subject, name: String) -> Option<wit::Value> {
        self.call()
            .snapshot
            .attrs
            .get(&subject(of))?
            .get(&name)
            .map(value)
    }
}

impl plex::core::hints::Host for Ctx {
    fn get(&mut self, of: wit::Subject, name: String) -> Option<wit::Value> {
        self.call()
            .snapshot
            .hints
            .get(&subject(of))?
            .get(&name)
            .map(value)
    }
}

impl plex::core::notes::Host for Ctx {
    fn get(&mut self, of: wit::Subject, writer: String, name: String) -> Option<wit::Value> {
        let call = self.call();
        let of = subject(of);
        if writer == call.policy {
            // Its own writes of this call come first.
            if let Some((_, _, staged)) = call
                .staged
                .notes
                .iter()
                .rev()
                .find(|(s, n, _)| *s == of && *n == name)
            {
                return staged.as_ref().map(value);
            }
        }
        let tables = call.tables.lock().unwrap();
        tables.notes.get(&of)?.get(&(writer, name)).map(value)
    }

    fn set(
        &mut self,
        of: wit::Subject,
        name: String,
        val: wit::Value,
    ) -> Result<(), wit::ErrorCode> {
        let call = self.call();
        if call.staged.notes.len() >= MAX_ROWS {
            return Err(wit::ErrorCode::LimitExceeded);
        }
        call.staged
            .notes
            .push((subject(of), name, Some(own_value(val))));
        Ok(())
    }

    fn delete(&mut self, of: wit::Subject, name: String) {
        self.call().staged.notes.push((subject(of), name, None));
    }
}

impl plex::core::alarm::Host for Ctx {
    fn set(&mut self, when: u64) {
        self.call().staged.alarm = Some(when);
    }
}

impl plex::core::terminate::Host for Ctx {
    fn terminate(&mut self, process: String, reason: String) -> Result<(), wit::ErrorCode> {
        let call = self.call();
        if call.snapshot.live_process(&process) {
            call.staged.terminate.push((process, reason));
        }
        Ok(())
    }
}

/// The wasm engine every policy shares, and the thread that ticks its
/// epoch every millisecond: a deadline costs a policy nothing until it is
/// passed.
pub fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let mut config = wasmtime::Config::new();
        config.wasm_component_model(true);
        config.epoch_interruption(true);
        let engine = Engine::new(&config).expect("a wasm engine");
        let ticker = engine.clone();
        std::thread::Builder::new()
            .name("plex-epoch".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_millis(1));
                    ticker.increment_epoch();
                }
            })
            .expect("spawn the epoch ticker");
        engine
    })
}

#[derive(Debug)]
pub enum Fault {
    /// The call ran past its deadline.
    Deadline,
    Trap(String),
}

pub struct Program {
    pub name: String,
    /// The interfaces it imports, without their versions.
    pub imports: Vec<String>,
    pre: PolicyHostPre<Ctx>,
    live: Option<(Store<Ctx>, PolicyHost)>,
    /// When it asked to be called again.
    pub alarm: Option<u64>,
    /// Calls that failed in a row.
    pub failures: u32,
    /// Not yet called since it was loaded.
    pub fresh: bool,
}

impl Program {
    pub fn load(name: &str, wasm: &[u8]) -> Result<Program, String> {
        let engine = engine();
        let component =
            Component::new(engine, wasm).map_err(|e| format!("not a component: {e:#}"))?;
        let mut imports = Vec::new();
        for (import, _) in component.component_type().imports(engine) {
            let bare = import.split('@').next().unwrap_or(import).to_string();
            if !bare.starts_with("plex:") {
                return Err(format!(
                    "imports {import}: a policy imports only plex interfaces"
                ));
            }
            imports.push(bare);
        }
        let mut linker: Linker<Ctx> = Linker::new(engine);
        let mut options = LinkOptions::default();
        options.placement_helpers(true);
        PolicyHost::add_to_linker::<_, HasSelf<_>>(&mut linker, &options, |ctx| ctx)
            .map_err(|e| format!("{e:#}"))?;
        let pre = linker
            .instantiate_pre(&component)
            .map_err(|e| format!("does not link: {e:#}"))?;
        let pre = PolicyHostPre::new(pre).map_err(|e| format!("exports no policy: {e:#}"))?;
        Ok(Program {
            name: name.to_string(),
            imports,
            pre,
            live: None,
            alarm: None,
            failures: 0,
            fresh: true,
        })
    }

    pub fn writes(&self, column: &str) -> bool {
        self.imports.iter().any(|i| i == column)
    }

    fn instance(&mut self, deadline_ms: u64) -> Result<&mut (Store<Ctx>, PolicyHost), Fault> {
        if self.live.is_none() {
            let ctx = Ctx {
                limits: StoreLimitsBuilder::new().memory_size(MEMORY_BYTES).build(),
                call: None,
            };
            let mut store = Store::new(engine(), ctx);
            store.limiter(|ctx| &mut ctx.limits);
            store.set_epoch_deadline(deadline_ms.max(1));
            let instance = self.pre.instantiate(&mut store).map_err(fault)?;
            self.live = Some((store, instance));
        }
        Ok(self.live.as_mut().unwrap())
    }

    /// One call. Its writes come back staged; on a fault they are dropped
    /// with the instance, and a fresh one takes the next call.
    pub fn react(
        &mut self,
        call: Call,
        events: &[wit::Event],
        now: u64,
        deadline_ms: u64,
    ) -> Result<Staged, Fault> {
        let (store, instance) = self.instance(deadline_ms)?;
        store.data_mut().call = Some(call);
        store.set_epoch_deadline(deadline_ms.max(1));
        let result = instance
            .plex_policy_policy()
            .call_react(&mut *store, events, now);
        let call = store.data_mut().call.take();
        match result {
            Ok(()) => Ok(call.map(|c| c.staged).unwrap_or_default()),
            Err(e) => {
                self.live = None;
                Err(fault(e))
            }
        }
    }

    /// Drops the instance: the next call starts without its memory.
    pub fn reset(&mut self) {
        self.live = None;
    }
}

fn fault(e: wasmtime::Error) -> Fault {
    match e.downcast_ref::<Trap>() {
        Some(Trap::Interrupt) => Fault::Deadline,
        _ => Fault::Trap(format!("{e:#}")),
    }
}
