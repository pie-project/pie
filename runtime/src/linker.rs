//! Linker service.
//!
//! Singleton service that owns the pre-configured wasmtime Engine.
//! Creates per-instance linkers with WASI, WASI HTTP, Pie API host bindings,
//! and dynamically linked library dependencies.

pub(crate) mod dynamic_linking;

use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::component::{Component, Instance as WasmInstance, InstancePre, Linker as WasmLinker};
use wasmtime::{Engine, Store};

use crate::api;
use crate::instance::{InstanceState, OutputMode};
use crate::policy::{FsPolicy, NetworkPolicy};
use crate::process::ProcessId;
use crate::program::python::runtime as py_runtime;
use crate::program::{self, InstalledComponent, ProgramName};
use crate::service::{Service, ServiceHandler};

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

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
    output: OutputMode,
    token_budget: Option<usize>,
) -> Result<(Store<InstanceState>, WasmInstance)> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        process_id,
        username,
        program_name: program_name.clone(),
        output,
        token_budget,
        response: tx,
    })?;
    rx.await?
}

// ---- State ------------------------------------------------------------------

struct Linker {
    engine: Engine,
    policy: InstancePolicy,
    instance_pre_cache: HashMap<(ProgramName, u64), InstancePre<InstanceState>>,
}

impl Linker {
    fn new(engine: &Engine, policy: InstancePolicy) -> Self {
        Linker {
            engine: engine.clone(),
            policy,
            instance_pre_cache: HashMap::new(),
        }
    }

    async fn instantiate(
        &mut self,
        process_id: ProcessId,
        username: String,
        program_name: &ProgramName,
        output: OutputMode,
        token_budget: Option<usize>,
    ) -> Result<(Store<InstanceState>, WasmInstance)> {
        // 1. Get the main component (with snapshot status + python-runtime decl)
        let main = program::get_wasm_component(program_name)
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        // 2. Resolve dependencies and reconcile the python-runtime requirement
        //    across the main program and its direct dependencies. Also tracks
        //    whether any Python component in the graph was snapshotted — that
        //    determines which shared-module variant we use for instantiation.
        let (dependency_components, python_runtime, any_snapshotted) = self
            .resolve_dependencies_and_runtime(program_name, &main)
            .await?;
        let component = main.component;
        let cacheable_instance_pre = dependency_components.is_empty()
            && python_runtime.is_none()
            && linker_preinstantiate_enabled();

        // 3. Gate shared Python runtime loading by whether anything in the graph
        //    declared a python-runtime requirement. Non-Python inferlets pay no
        //    cost for the py-runtime env vars or preopens.
        //
        //    For Python inferlets, pick stripped shared modules when any
        //    component in the graph is snapshotted — their data segments and
        //    start sections have been baked into the snapshot image, so
        //    running them again would clobber it. Use full modules otherwise
        //    so CPython can initialize normally.
        let (shared_modules_for_linker, py_runtime_dir_for_state) = if python_runtime.is_some() {
            let modules = if any_snapshotted {
                py_runtime::stripped_modules()
            } else {
                py_runtime::full_modules()
            };
            (modules, py_runtime::dir())
        } else {
            (&[][..], None)
        };

        // 4. Create instance state and store
        let inst_state = InstanceState::new(
            process_id,
            username,
            output,
            &self.policy,
            py_runtime_dir_for_state,
        )
        .await?;
        let mut store = Store::new(&self.engine, inst_state);

        // 5. Create and configure linker
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
        for (name, module) in shared_modules_for_linker.iter() {
            linker
                .root()
                .module(name, module)
                .unwrap_or_else(|e| panic!("Failed to register shared module '{name}': {e}"));
        }

        // 6. Instantiate library dependencies (dynamic linking)
        if !dependency_components.is_empty() {
            dynamic_linking::instantiate_libraries(
                &self.engine,
                &mut linker,
                &mut store,
                dependency_components,
            )
            .await?;
        }

        // 7. Instantiate the main component
        let instance = if cacheable_instance_pre {
            let cache_key = (program_name.clone(), main.generation);
            let pre = match self.instance_pre_cache.get(&cache_key) {
                Some(pre) => pre.clone(),
                None => {
                    let pre = linker
                        .instantiate_pre(&component)
                        .map_err(|e| anyhow!("Instantiation pre-link error: {e}"))?;
                    self.instance_pre_cache.insert(cache_key, pre.clone());
                    pre
                }
            };
            pre.instantiate_async(&mut store)
                .await
                .map_err(|e| anyhow!("Instantiation error: {e}"))?
        } else {
            linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| anyhow!("Instantiation error: {e}"))?
        };

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

fn linker_preinstantiate_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("PIE_LINKER_PREINSTANTIATE")
            .map(|value| {
                let value = value.trim();
                value == "1"
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("on")
            })
            .unwrap_or(false)
    });
    *ENABLED
}

// ---- Messages ---------------------------------------------------------------

enum Message {
    Instantiate {
        process_id: ProcessId,
        username: String,
        program_name: ProgramName,
        output: OutputMode,
        token_budget: Option<usize>,
        response: oneshot::Sender<Result<(Store<InstanceState>, WasmInstance)>>,
    },
}

impl ServiceHandler for Linker {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Instantiate { process_id, username, program_name, output, token_budget, response } => {
                let _ = response.send(
                    self.instantiate(process_id, username, &program_name, output, token_budget).await
                );
            }
        }
    }
}
