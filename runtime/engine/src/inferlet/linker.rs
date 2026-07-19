//! Linker service.
//!
//! Singleton service that owns the pre-configured wasmtime Engine, three
//! immutable base-linker variants, and generation-keyed `InstancePre`s.
//! Dynamic dependencies clone the appropriate base before adding store-bound
//! definitions; no-dependency programs share a single-flight `InstancePre`.
//!
//! Instantiations run CONCURRENTLY (W3): the actor is only the spawn
//! point — each Instantiate message spawns an independent task over the
//! shared Engine (Send+Sync). The strictly sequential service loop used
//! to serialize ~2-3 ms of per-instance link work across a 256-process
//! fleet entry (~0.5-0.75 s of ramp with the GPU near-idle). Instances
//! are mutually independent: each builds its own Store/ProcessCtx, and
//! the component/dependency lookups are shared-read. Base-linker and
//! `InstancePre` cells are shared behind short-held map mutexes; expensive
//! construction is single-flight and happens outside those mutexes.

pub(super) mod dynamic;

use std::collections::{HashMap, hash_map::Entry};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Result, anyhow};
use tokio::sync::{OnceCell, oneshot};
use wasmtime::component::{Component, Instance, InstancePre, Linker as WasmLinker};
use wasmtime::{Engine, Store};

use crate::inferlet::host;
use crate::service::{Service, ServiceHandler};

use super::process::{OutputMode, ProcessCtx, ProcessId};
use super::program::{self, InstalledComponent, ProgramName};
use super::python::runtime as py_runtime;
use super::sandbox::{FsPolicy, InstancePolicy, NetworkPolicy};

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the linker service with the given engine.
pub fn spawn(engine: &Engine, fs: FsPolicy, network: NetworkPolicy) {
    let policy = InstancePolicy { fs, network };
    SERVICE
        .spawn(|| Linker::new(engine, policy))
        .expect("linker already spawned");
}

// ---- Public API (message wrappers) ------------------------------------------

/// Link and instantiate a program with its dependencies.
pub async fn instantiate(
    process_id: ProcessId,
    username: String,
    program_name: &ProgramName,
    output: OutputMode,
) -> Result<(Store<ProcessCtx>, Instance)> {
    let timing_started =
        crate::scheduler::fire_timing_enabled().then(crate::scheduler::fire_timing_now_us);
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        process_id,
        username,
        program_name: program_name.clone(),
        output,
        response: tx,
    })?;
    let result = rx.await?;
    if let Some(started_us) = timing_started {
        let finished_us = crate::scheduler::fire_timing_now_us();
        crate::scheduler::fire_timing_write(&serde_json::json!({
            "schema": 1,
            "source": "runtime",
            "event": "process_instantiated",
            "process_id": process_id,
            "program": program_name.to_string(),
            "success": result.is_ok(),
            "instantiate_started_us": started_us,
            "instantiate_finished_us": finished_us,
            "instantiate_us": finished_us.saturating_sub(started_us),
        }));
    }
    result
}

pub(crate) fn invalidate(program_name: &ProgramName) {
    let _ = SERVICE.send(Message::Invalidate {
        program_name: program_name.clone(),
    });
}

// ---- State ------------------------------------------------------------------

type InstancePreKey = (ProgramName, u64);
type InstancePreCell = Arc<OnceCell<InstancePre<ProcessCtx>>>;
type InstancePreCache = Arc<Mutex<HashMap<InstancePreKey, InstancePreCell>>>;
type BaseLinkerCell = Arc<OnceCell<Arc<WasmLinker<ProcessCtx>>>>;
type BaseLinkerCache = Arc<Mutex<HashMap<LinkerVariant, BaseLinkerCell>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum LinkerVariant {
    Plain,
    PythonFull,
    PythonStripped,
}

impl LinkerVariant {
    fn for_program(python_runtime: bool, any_snapshotted: bool) -> Self {
        match (python_runtime, any_snapshotted) {
            (false, _) => Self::Plain,
            (true, false) => Self::PythonFull,
            (true, true) => Self::PythonStripped,
        }
    }

    fn shared_modules(self) -> &'static [(String, wasmtime::Module)] {
        match self {
            Self::Plain => &[],
            Self::PythonFull => py_runtime::full_modules(),
            Self::PythonStripped => py_runtime::stripped_modules(),
        }
    }
}

struct Linker {
    engine: Engine,
    policy: InstancePolicy,
    base_linker_cache: BaseLinkerCache,
    instance_pre_cache: InstancePreCache,
}

impl Linker {
    fn new(engine: &Engine, policy: InstancePolicy) -> Self {
        Linker {
            engine: engine.clone(),
            policy,
            base_linker_cache: Arc::new(Mutex::new(HashMap::new())),
            instance_pre_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn build_base_linker(
        engine: &Engine,
        policy: &InstancePolicy,
        variant: LinkerVariant,
    ) -> Result<WasmLinker<ProcessCtx>> {
        let mut linker = WasmLinker::<ProcessCtx>::new(engine);

        wasmtime_wasi::p2::add_to_linker_async(&mut linker).expect("Failed to link WASI");
        wasmtime_wasi::p3::add_to_linker(&mut linker).expect("Failed to link WASI p3");
        wasmtime_wasi_http::p3::add_to_linker(&mut linker).expect("Failed to link WASI HTTP p3");

        // wasm32-wasip3 std still imports the RC-versioned insecure-seed name.
        {
            let mut root = linker.root();
            let mut random = root
                .instance("wasi:random/insecure-seed@0.3.0-rc-2026-03-15")
                .expect("Failed to add wasi:random insecure-seed rc shim");
            random
                .func_wrap_async("get-insecure-seed", |_store, (): ()| {
                    Box::new(
                        async move { Ok(((0x9e37_79b9_7f4a_7c15u64, 0xbf58_476d_1ce4_e5b9u64),)) },
                    )
                })
                .expect("Failed to shim get-insecure-seed");
        }

        // p3 HTTP is always linked and enforces policy through ProcessCtx. The
        // legacy p2 HTTP surface must be absent when networking is denied.
        if policy.network.allow {
            wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)
                .expect("Failed to link WASI HTTP");
        }

        host::add_to_linker(&mut linker)?;

        // `full_modules` and `stripped_modules` are process-global OnceLock
        // values compiled from the one runtime directory installed at startup.
        // They do not vary with a manifest's python-runtime version string.
        for (name, module) in variant.shared_modules() {
            linker.root().module(name, module).unwrap_or_else(|error| {
                panic!("Failed to register shared module '{name}': {error}")
            });
        }

        Ok(linker)
    }

    async fn base_linker(
        engine: &Engine,
        policy: &InstancePolicy,
        cache: &BaseLinkerCache,
        variant: LinkerVariant,
    ) -> Result<Arc<WasmLinker<ProcessCtx>>> {
        let cell = {
            let mut cache = cache.lock().unwrap();
            Arc::clone(
                cache
                    .entry(variant)
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            )
        };
        let linker = cell
            .get_or_try_init(|| async {
                Self::build_base_linker(engine, policy, variant).map(Arc::new)
            })
            .await?;
        Ok(Arc::clone(linker))
    }

    fn instance_pre_cell(
        cache: &InstancePreCache,
        program_name: &ProgramName,
        generation: u64,
    ) -> (InstancePreCell, bool) {
        let mut cache = cache.lock().unwrap();
        cache.retain(|(name, cached_generation), _| {
            name != program_name || *cached_generation == generation
        });
        match cache.entry((program_name.clone(), generation)) {
            Entry::Occupied(entry) => (Arc::clone(entry.get()), true),
            Entry::Vacant(entry) => {
                let cell = Arc::new(OnceCell::new());
                entry.insert(Arc::clone(&cell));
                (cell, false)
            }
        }
    }

    async fn instantiate(
        engine: Engine,
        policy: InstancePolicy,
        base_linker_cache: BaseLinkerCache,
        instance_pre_cache: InstancePreCache,
        process_id: ProcessId,
        username: String,
        program_name: &ProgramName,
        output: OutputMode,
    ) -> Result<(Store<ProcessCtx>, Instance)> {
        // Diagnostic sub-stamps (iteration 76 provenance): split the span
        // into program-service round-trips vs ctx/store vs link+instantiate.
        let timing = crate::scheduler::fire_timing_enabled();
        let mut stamps = [0u64; 4];
        let stamp_t0 = timing.then(crate::scheduler::fire_timing_now_us);
        // 1. Get the main component (with snapshot status + python-runtime decl)
        let main = program::get_wasm_component(program_name)
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;
        if timing {
            stamps[0] = crate::scheduler::fire_timing_now_us();
        }

        // 2. Resolve dependencies and reconcile the python-runtime requirement
        //    across the main program and its direct dependencies. Also tracks
        //    whether any Python component in the graph was snapshotted — that
        //    determines which shared-module variant we use for instantiation.
        let (dependency_components, python_runtime, any_snapshotted) =
            Self::resolve_dependencies_and_runtime(program_name, &main).await?;
        if timing {
            stamps[1] = crate::scheduler::fire_timing_now_us();
        }
        let generation = main.generation;
        let component = main.component;
        let cacheable_instance_pre = dependency_components.is_empty();

        // 3. Gate shared Python runtime loading by whether anything in the graph
        //    declared a python-runtime requirement. Non-Python inferlets pay no
        //    cost for the py-runtime env vars or preopens.
        //
        //    For Python inferlets, pick stripped shared modules when any
        //    component in the graph is snapshotted — their data segments and
        //    start sections have been baked into the snapshot image, so
        //    running them again would clobber it. Use full modules otherwise
        //    so CPython can initialize normally.
        let linker_variant = LinkerVariant::for_program(python_runtime.is_some(), any_snapshotted);
        let py_runtime_dir_for_ctx = python_runtime.is_some().then(py_runtime::dir).flatten();

        // 4. Create process context and store
        let process_ctx = ProcessCtx::new(
            process_id,
            username,
            output,
            &policy,
            py_runtime_dir_for_ctx,
        )
        .await?;
        let mut store = Store::new(&engine, process_ctx);
        if timing {
            stamps[2] = crate::scheduler::fire_timing_now_us();
        }

        // 5. Reuse the immutable linker configuration. Construction is lazy
        // and single-flight per Plain/PythonFull/PythonStripped variant.
        let base_linker =
            Self::base_linker(&engine, &policy, &base_linker_cache, linker_variant).await?;

        // 6. Instantiate library dependencies (dynamic linking)
        let dynamic_linker = if dependency_components.is_empty() {
            None
        } else {
            let mut linker = base_linker.as_ref().clone();
            dynamic::instantiate_libraries(&engine, &mut linker, &mut store, dependency_components)
                .await?;
            Some(linker)
        };

        // 7. Instantiate the main component
        let mut pre_hit = false;
        let instance = if cacheable_instance_pre {
            let (cell, cache_hit) =
                Self::instance_pre_cell(&instance_pre_cache, program_name, generation);
            pre_hit = cache_hit;
            let pre = cell
                .get_or_try_init(|| async {
                    base_linker
                        .instantiate_pre(&component)
                        .map_err(|error| anyhow!("Instantiation pre-link error: {error}"))
                })
                .await?;
            pre.instantiate_async(&mut store)
                .await
                .map_err(|e| anyhow!("Instantiation error: {e}"))?
        } else {
            dynamic_linker
                .expect("dynamic dependencies require a cloned linker")
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| anyhow!("Instantiation error: {e}"))?
        };
        if let Some(t0) = stamp_t0 {
            stamps[3] = crate::scheduler::fire_timing_now_us();
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "runtime",
                "event": "instantiate_breakdown",
                "process_id": process_id,
                "pre_hit": pre_hit,
                "get_component_us": stamps[0].saturating_sub(t0),
                "resolve_deps_us": stamps[1].saturating_sub(stamps[0]),
                "ctx_store_us": stamps[2].saturating_sub(stamps[1]),
                "instantiate_us": stamps[3].saturating_sub(stamps[2]),
            }));
        }

        Ok((store, instance))
    }

    /// Resolve dependency components for a program and reconcile the
    /// python-runtime version declared across the main program and its direct
    /// dependencies. Also tracks whether any Python component in the graph
    /// has been snapshotted — callers use this to pick stripped vs full
    /// shared modules at instantiate time. Returns an error if multiple
    /// python-runtime declarations conflict.
    async fn resolve_dependencies_and_runtime(
        program_name: &ProgramName,
        main: &InstalledComponent,
    ) -> Result<(Vec<Component>, Option<String>, bool)> {
        let manifest = program::fetch_manifest(program_name)
            .await
            .ok_or_else(|| anyhow!("Manifest not found for: {}", program_name))?;

        let mut python_runtime: Option<String> = main.python_runtime.clone();
        let mut any_snapshotted = main.snapshotted;

        let dep_names = manifest.dependency_names();
        let mut components = Vec::with_capacity(dep_names.len());

        for dep_name in dep_names {
            let dep = program::get_wasm_component(&dep_name)
                .await
                .ok_or_else(|| anyhow!("Dependency component not found: {}", dep_name))?;

            if dep.snapshotted {
                any_snapshotted = true;
            }

            if let Some(dep_py_rt) = dep.python_runtime.as_deref() {
                match &python_runtime {
                    Some(existing) if existing != dep_py_rt => {
                        return Err(anyhow!(
                            "Conflicting python-runtime versions among dependencies of {}: \
                             '{}' vs '{}' (from {})",
                            program_name,
                            existing,
                            dep_py_rt,
                            dep_name,
                        ));
                    }
                    None => python_runtime = Some(dep_py_rt.to_string()),
                    _ => {}
                }
            }

            components.push(dep.component);
        }

        Ok((components, python_runtime, any_snapshotted))
    }
}

// ---- Messages ---------------------------------------------------------------

enum Message {
    Instantiate {
        process_id: ProcessId,
        username: String,
        program_name: ProgramName,
        output: OutputMode,
        response: oneshot::Sender<Result<(Store<ProcessCtx>, Instance)>>,
    },
    Invalidate {
        program_name: ProgramName,
    },
}

impl ServiceHandler for Linker {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Instantiate {
                process_id,
                username,
                program_name,
                output,
                response,
            } => {
                // Spawn, do not await (W3): the actor loop remains a dispatch
                // point. Spawned tasks share immutable base linkers and
                // generation-keyed single-flight InstancePre cells; only
                // request-local stores and dependency-mutated clones differ.
                let engine = self.engine.clone();
                let policy = self.policy.clone();
                let base_cache = Arc::clone(&self.base_linker_cache);
                let pre_cache = Arc::clone(&self.instance_pre_cache);
                tokio::task::spawn(async move {
                    let link_started_us = crate::scheduler::fire_timing_enabled()
                        .then(crate::scheduler::fire_timing_now_us);
                    let result = Linker::instantiate(
                        engine,
                        policy,
                        base_cache,
                        pre_cache,
                        process_id,
                        username,
                        &program_name,
                        output,
                    )
                    .await;
                    if let Some(started_us) = link_started_us {
                        // Splits the public wrapper's `instantiate_us`
                        // (queue + link) into its link-work part — the W3
                        // gate reads this to prove the residual is work,
                        // not actor queue wait.
                        crate::scheduler::fire_timing_write(&serde_json::json!({
                            "schema": 1,
                            "source": "runtime",
                            "event": "process_link_work",
                            "process_id": process_id.to_string(),
                            "success": result.is_ok(),
                            "link_started_us": started_us,
                            "link_us": crate::scheduler::fire_timing_now_us()
                                .saturating_sub(started_us),
                        }));
                    }
                    let _ = response.send(result);
                });
            }
            Message::Invalidate { program_name } => {
                self.instance_pre_cache
                    .lock()
                    .unwrap()
                    .retain(|(name, _), _| name != &program_name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::future::join_all;

    use super::*;

    fn test_engine() -> Engine {
        let mut config = wasmtime::Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config).unwrap();
        py_runtime::init(
            &engine,
            std::path::Path::new("/nonexistent/pie-test-python"),
            false,
        );
        engine
    }

    fn test_policy() -> InstancePolicy {
        InstancePolicy {
            fs: FsPolicy {
                allow: false,
                base_dir: PathBuf::new(),
            },
            network: NetworkPolicy::parse(false, &[]).unwrap(),
        }
    }

    #[test]
    fn linker_variant_tracks_python_snapshot_mode() {
        assert_eq!(
            LinkerVariant::for_program(false, false),
            LinkerVariant::Plain
        );
        assert_eq!(
            LinkerVariant::for_program(false, true),
            LinkerVariant::Plain
        );
        assert_eq!(
            LinkerVariant::for_program(true, false),
            LinkerVariant::PythonFull
        );
        assert_eq!(
            LinkerVariant::for_program(true, true),
            LinkerVariant::PythonStripped
        );
    }

    #[tokio::test]
    async fn base_linker_and_instance_pre_are_single_flight() {
        let engine = test_engine();
        let policy = test_policy();
        let base_cache = Arc::new(Mutex::new(HashMap::new()));
        let bases = join_all(
            (0..64)
                .map(|_| Linker::base_linker(&engine, &policy, &base_cache, LinkerVariant::Plain)),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();
        assert_eq!(base_cache.lock().unwrap().len(), 1);
        assert!(bases.iter().all(|base| Arc::ptr_eq(base, &bases[0])));
        let python_full =
            Linker::base_linker(&engine, &policy, &base_cache, LinkerVariant::PythonFull)
                .await
                .unwrap();
        let python_stripped =
            Linker::base_linker(&engine, &policy, &base_cache, LinkerVariant::PythonStripped)
                .await
                .unwrap();
        assert_eq!(base_cache.lock().unwrap().len(), 3);
        assert!(!Arc::ptr_eq(&python_full, &python_stripped));

        let component = Component::new(&engine, "(component)").unwrap();
        python_full.instantiate_pre(&component).unwrap();
        python_stripped.instantiate_pre(&component).unwrap();
        let pre_cache = Arc::new(Mutex::new(HashMap::new()));
        let name = ProgramName::parse("single-flight@1.0.0").unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let results = join_all((0..64).map(|_| {
            let (cell, hit) = Linker::instance_pre_cell(&pre_cache, &name, 1);
            let builds = Arc::clone(&builds);
            let base = Arc::clone(&bases[0]);
            let component = component.clone();
            async move {
                let pre = cell
                    .get_or_try_init(|| async {
                        builds.fetch_add(1, Ordering::Relaxed);
                        tokio::task::yield_now().await;
                        base.instantiate_pre(&component)
                            .map_err(anyhow::Error::from)
                    })
                    .await;
                (hit, pre.is_ok())
            }
        }))
        .await;
        assert_eq!(results.iter().filter(|(hit, _)| !*hit).count(), 1);
        assert!(results.iter().all(|(_, ok)| *ok));
        assert_eq!(builds.load(Ordering::Relaxed), 1);

        let (next_generation, hit) = Linker::instance_pre_cell(&pre_cache, &name, 2);
        assert!(!hit);
        assert!(next_generation.get().is_none());
        let cache = pre_cache.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&(name, 2)));
    }

    #[tokio::test]
    async fn failed_instance_pre_build_remains_retryable() {
        let engine = test_engine();
        let policy = test_policy();
        let base_cache = Arc::new(Mutex::new(HashMap::new()));
        let base = Linker::base_linker(&engine, &policy, &base_cache, LinkerVariant::Plain)
            .await
            .unwrap();
        let component = Component::new(&engine, "(component)").unwrap();
        let pre_cache = Arc::new(Mutex::new(HashMap::new()));
        let name = ProgramName::parse("retryable-pre@1.0.0").unwrap();
        let (cell, hit) = Linker::instance_pre_cell(&pre_cache, &name, 1);
        assert!(!hit);

        let failed = cell
            .get_or_try_init(|| async {
                Err::<InstancePre<ProcessCtx>, _>(anyhow!("expected pre-link failure"))
            })
            .await;
        assert!(failed.is_err());
        assert!(cell.get().is_none());

        let _pre = cell
            .get_or_try_init(|| async {
                base.instantiate_pre(&component)
                    .map_err(anyhow::Error::from)
            })
            .await
            .unwrap();
        assert!(cell.get().is_some());
    }
}
