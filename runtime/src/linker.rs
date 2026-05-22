//! Linker service.
//!
//! Singleton service that owns the pre-configured wasmtime Engine.
//! Creates per-instance linkers with WASI, WASI HTTP, Pie API host bindings,
//! and dynamically linked library dependencies.

pub(crate) mod dynamic_linking;

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::component::{Component, Instance as WasmInstance, Linker as WasmLinker};
use wasmtime::{Engine, Store};

use crate::api;
use crate::instance::InstanceState;
use crate::policy::{FsPolicy, NetworkPolicy};
use crate::process::ProcessId;
use crate::program::python::runtime as py_runtime;
use crate::program::{self, InstalledComponent, ProgramName};
use crate::service::{Service, ServiceHandler};

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

fn launch_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_PROFILE_LAUNCH").is_some())
}

fn launch_profile(process_id: ProcessId, stage: &str, elapsed: std::time::Duration) {
    if launch_profile_enabled() {
        println!(
            "[launch-profile] pid={process_id} stage={stage} elapsed_us={}",
            elapsed.as_micros()
        );
    }
}

/// Per-instance security policies. Compiled once at bootstrap and
/// shared by reference into every spawned inferlet.
#[derive(Clone)]
pub struct InstancePolicy {
    pub fs: FsPolicy,
    pub network: NetworkPolicy,
}

impl InstancePolicy {
    /// Maximally restrictive policy: no filesystem, no network. Used
    /// by code paths that instantiate a component for inspection only
    /// (e.g. python-runtime snapshotting), not user execution.
    pub fn deny_all() -> Self {
        InstancePolicy {
            fs: FsPolicy {
                allow: false,
                // Never used because allow=false blocks scratch creation;
                // an empty path is fine here.
                base_dir: std::path::PathBuf::new(),
            },
            network: NetworkPolicy::parse(false, &[]).expect("deny-all parse"),
        }
    }
}

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
    capture_outputs: bool,
    _token_budget: Option<usize>,
) -> Result<(Store<InstanceState>, WasmInstance)> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        process_id,
        username,
        program_name: program_name.clone(),
        capture_outputs,
        response: tx,
    })?;
    rx.await?
}

// ---- State ------------------------------------------------------------------

struct Linker {
    engine: Engine,
    policy: InstancePolicy,
    base_linkers: HashMap<BaseLinkerKey, WasmLinker<InstanceState>>,
    prepared_programs: HashMap<ProgramName, PreparedProgram>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BaseLinkerKey {
    python_runtime: bool,
    any_snapshotted: bool,
}

struct InstantiatePlan {
    engine: Engine,
    policy: InstancePolicy,
    component: Component,
    dependency_components: Vec<Component>,
    base_linker: WasmLinker<InstanceState>,
    py_runtime_dir_for_state: Option<&'static Path>,
}

#[derive(Clone)]
struct PreparedProgram {
    generation: u64,
    component: Component,
    dependency_components: Vec<Component>,
    base_linker_key: BaseLinkerKey,
    py_runtime_dir_for_state: Option<&'static Path>,
}

impl Linker {
    fn new(engine: &Engine, policy: InstancePolicy) -> Self {
        Linker {
            engine: engine.clone(),
            policy,
            base_linkers: HashMap::new(),
            prepared_programs: HashMap::new(),
        }
    }

    async fn prepare(&mut self, program_name: &ProgramName) -> Result<InstantiatePlan> {
        // 1. Get the main component (with snapshot status + python-runtime decl)
        let main = program::get_wasm_component(program_name)
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;
        if let Some(prepared) = self.prepared_programs.get(program_name).cloned() {
            if prepared.generation == main.generation {
                return self.instantiate_plan(prepared);
            }
        }

        // 2. Resolve dependencies and reconcile the python-runtime requirement
        //    across the main program and its direct dependencies. Also tracks
        //    whether any Python component in the graph was snapshotted — that
        //    determines which shared-module variant we use for instantiation.
        let (dependency_components, python_runtime, any_snapshotted) = self
            .resolve_dependencies_and_runtime(program_name, &main)
            .await?;
        let component = main.component;

        // 3. Gate shared Python runtime loading by whether anything in the graph
        //    declared a python-runtime requirement. Non-Python inferlets pay no
        //    cost for the py-runtime env vars or preopens.
        //
        //    For Python inferlets, pick stripped shared modules when any
        //    component in the graph is snapshotted — their data segments and
        //    start sections have been baked into the snapshot image, so
        //    running them again would clobber it. Use full modules otherwise
        //    so CPython can initialize normally.
        let key = if python_runtime.is_some() {
            BaseLinkerKey {
                python_runtime: true,
                any_snapshotted,
            }
        } else {
            BaseLinkerKey {
                python_runtime: false,
                any_snapshotted: false,
            }
        };
        let py_runtime_dir_for_state = python_runtime.as_ref().and(py_runtime::dir());

        let prepared = PreparedProgram {
            generation: main.generation,
            component,
            dependency_components,
            base_linker_key: key,
            py_runtime_dir_for_state,
        };
        self.prepared_programs
            .insert(program_name.clone(), prepared.clone());
        self.instantiate_plan(prepared)
    }

    fn instantiate_plan(&mut self, prepared: PreparedProgram) -> Result<InstantiatePlan> {
        let base_linker = self.base_linker(prepared.base_linker_key)?;
        Ok(InstantiatePlan {
            engine: self.engine.clone(),
            policy: self.policy.clone(),
            component: prepared.component,
            dependency_components: prepared.dependency_components,
            base_linker,
            py_runtime_dir_for_state: prepared.py_runtime_dir_for_state,
        })
    }

    fn base_linker(&mut self, key: BaseLinkerKey) -> Result<WasmLinker<InstanceState>> {
        if let Some(linker) = self.base_linkers.get(&key) {
            return Ok(linker.clone());
        }

        let mut linker = WasmLinker::<InstanceState>::new(&self.engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker).expect("Failed to link WASI");

        // wasi:http operates above wasi:sockets and bypasses the per-socket
        // policy hook (it uses the host's hyper stack with its own DNS).
        // Drop the binding entirely when the network is disabled so the
        // policy is honored end-to-end. With allow_network=true the hook
        // is wired; the allowlist still applies to wasi:sockets but
        // wasi:http is unrestricted (pre-DNS hostname allowlisting would
        // require a DNS shim — not in v1).
        if self.policy.network.allow {
            wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
                .expect("Failed to link WASI HTTP");
        }

        api::add_to_linker(&mut linker)?;

        // Register shared core modules (e.g. CPython interpreter) so Python
        // inferlets can dynamically import the runtime instead of bundling it.
        let shared_modules_for_linker = if key.python_runtime {
            if key.any_snapshotted {
                py_runtime::stripped_modules()
            } else {
                py_runtime::full_modules()
            }
        } else {
            &[][..]
        };
        for (name, module) in shared_modules_for_linker.iter() {
            linker
                .root()
                .module(name, module)
                .unwrap_or_else(|e| panic!("Failed to register shared module '{name}': {e}"));
        }

        self.base_linkers.insert(key, linker.clone());
        Ok(linker)
    }

    async fn instantiate_prepared(
        plan: InstantiatePlan,
        process_id: ProcessId,
        username: String,
        capture_outputs: bool,
    ) -> Result<(Store<InstanceState>, WasmInstance)> {
        let InstantiatePlan {
            engine,
            policy,
            component,
            dependency_components,
            mut base_linker,
            py_runtime_dir_for_state,
        } = plan;

        // 1. Create instance state and store
        let stage_start = Instant::now();
        let inst_state = InstanceState::new(
            process_id,
            username,
            capture_outputs,
            &policy,
            py_runtime_dir_for_state,
        )
        .await?;
        let mut store = Store::new(&engine, inst_state);
        launch_profile(process_id, "instance_state_store", stage_start.elapsed());

        // 2. Instantiate library dependencies (dynamic linking)
        if !dependency_components.is_empty() {
            let stage_start = Instant::now();
            dynamic_linking::instantiate_libraries(
                &engine,
                &mut base_linker,
                &mut store,
                dependency_components,
            )
            .await?;
            launch_profile(process_id, "dependency_instantiate", stage_start.elapsed());
        }

        // 3. Instantiate the main component
        let stage_start = Instant::now();
        let instance = base_linker
            .instantiate_async(&mut store, &component)
            .await
            .map_err(|e| anyhow!("Instantiation error: {e}"))?;
        launch_profile(process_id, "component_instantiate", stage_start.elapsed());

        Ok((store, instance))
    }

    /// Resolve dependency components for a program and reconcile the
    /// python-runtime version declared across the main program and its direct
    /// dependencies. Also tracks whether any Python component in the graph
    /// has been snapshotted — callers use this to pick stripped vs full
    /// shared modules at instantiate time. Returns an error if multiple
    /// python-runtime declarations conflict.
    async fn resolve_dependencies_and_runtime(
        &self,
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
        capture_outputs: bool,
        response: oneshot::Sender<Result<(Store<InstanceState>, WasmInstance)>>,
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
                capture_outputs,
                response,
            } => {
                let stage_start = Instant::now();
                let prepared = self.prepare(&program_name).await;
                launch_profile(process_id, "linker_prepare", stage_start.elapsed());
                match prepared {
                    Ok(plan) => {
                        tokio::spawn(async move {
                            let result = Linker::instantiate_prepared(
                                plan,
                                process_id,
                                username,
                                capture_outputs,
                            )
                            .await;
                            let _ = response.send(result);
                        });
                    }
                    Err(e) => {
                        let _ = response.send(Err(e));
                    }
                }
            }
        }
    }
}
