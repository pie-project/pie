//! Linker service: singleton service owning the pre-configured wasmtime Engine, three immutable base-linker variants, and generation-keyed `InstancePre`s.
//! Dynamic dependencies clone the appropriate base before adding store-bound definitions; no-dependency programs share a single-flight `InstancePre`.
//! Instantiations run concurrently: the actor is only the spawn point, each Instantiate message spawns an independent task over the shared Engine. Base-linker and `InstancePre` cells are shared behind short-held map mutexes; expensive construction is single-flight and happens outside those mutexes.

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
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        process_id,
        username,
        program_name: program_name.clone(),
        output,
        response: tx,
    })?;
    rx.await?
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

        // wasm32-wasip3 std still imports the rc-versioned insecure-seed name.
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

        // full_modules/stripped_modules are process-global, compiled once from the runtime directory at bootstrap.
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

    #[allow(
        clippy::too_many_arguments,
        reason = "one instantiation's whole context: the engine and policy to build \
                  under, the two caches it must hit rather than rebuild, and the \
                  process identity it is being built FOR (id, user, program, output \
                  mode). Folding them into a struct would re-list the same eight \
                  fields and add a lifetime to the two cache handles"
    )]
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
        let main = program::get_wasm_component(program_name)
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        let (dependency_components, python_runtime, any_snapshotted) =
            Self::resolve_dependencies_and_runtime(program_name, &main).await?;
        let generation = main.generation;
        let component = main.component;
        let cacheable_instance_pre = dependency_components.is_empty();

        // stripped shared modules when snapshotted (their data/start sections are baked into the image), full modules otherwise so CPython can initialize normally.
        let linker_variant = LinkerVariant::for_program(python_runtime.is_some(), any_snapshotted);
        let py_runtime_dir_for_ctx = python_runtime.is_some().then(py_runtime::dir).flatten();

        let process_ctx = ProcessCtx::new(
            process_id,
            username,
            output,
            &policy,
            py_runtime_dir_for_ctx,
        )
        .await?;
        let mut store = Store::new(&engine, process_ctx);

        // lazy, single-flight per variant.
        let base_linker =
            Self::base_linker(&engine, &policy, &base_linker_cache, linker_variant).await?;

        let dynamic_linker = if dependency_components.is_empty() {
            None
        } else {
            let mut linker = base_linker.as_ref().clone();
            dynamic::instantiate_libraries(&engine, &mut linker, &mut store, dependency_components)
                .await?;
            Some(linker)
        };

        let instance = if cacheable_instance_pre {
            let (cell, _cache_hit) =
                Self::instance_pre_cell(&instance_pre_cache, program_name, generation);
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
        Ok((store, instance))
    }

    /// Resolve dependency components and reconcile the python-runtime version declared across the main program and its dependencies; also tracks whether any is snapshotted. Errs on conflicting python-runtime declarations.
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
                // spawn, don't await: the actor loop stays a dispatch point.
                let engine = self.engine.clone();
                let policy = self.policy.clone();
                let base_cache = Arc::clone(&self.base_linker_cache);
                let pre_cache = Arc::clone(&self.instance_pre_cache);
                tokio::task::spawn(async move {
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

