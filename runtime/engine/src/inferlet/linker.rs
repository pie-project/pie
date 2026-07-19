//! Linker service.
//!
//! Singleton service that owns the pre-configured wasmtime Engine.
//! Creates per-instance linkers with WASI, WASI HTTP, Pie API host bindings,
//! and dynamically linked library dependencies.
//!
//! Instantiations run CONCURRENTLY (W3): the actor is only the spawn
//! point — each Instantiate message spawns an independent task over the
//! shared Engine (Send+Sync). The strictly sequential service loop used
//! to serialize ~2-3 ms of per-instance link work across a 256-process
//! fleet entry (~0.5-0.75 s of ramp with the GPU near-idle). Instances
//! are mutually independent: each builds its own Store/ProcessCtx, and
//! the component/dependency lookups are shared-read. Only the
//! InstancePre cache is shared state, behind a Mutex.

pub(super) mod dynamic;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
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

// ---- State ------------------------------------------------------------------

type InstancePreCache = Arc<Mutex<HashMap<(ProgramName, u64), InstancePre<ProcessCtx>>>>;

struct Linker {
    engine: Engine,
    policy: InstancePolicy,
    instance_pre_cache: InstancePreCache,
}

impl Linker {
    fn new(engine: &Engine, policy: InstancePolicy) -> Self {
        Linker {
            engine: engine.clone(),
            policy,
            instance_pre_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    async fn instantiate(
        engine: Engine,
        policy: InstancePolicy,
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
        let (shared_modules_for_linker, py_runtime_dir_for_ctx) = if python_runtime.is_some() {
            let modules = if any_snapshotted {
                py_runtime::stripped_modules()
            } else {
                py_runtime::full_modules()
            };
            (modules, py_runtime::dir())
        } else {
            (&[][..], None)
        };

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

        // 5. Create and configure linker
        let mut linker = WasmLinker::<ProcessCtx>::new(&engine);

        wasmtime_wasi::p2::add_to_linker_async(&mut linker).expect("Failed to link WASI");

        // wasi 0.3 (p3) surfaces the pie:inferlet world imports directly:
        // clocks (native async wait-for), filesystem (snapshot I/O), and http
        // (outbound client.send). Linked alongside p2 — different package
        // versions, so the two sets never collide (the supported wasmtime
        // p2+p3 side-by-side config). p3 http is always linked; its network
        // policy is enforced host-side by the ProcessCtx hooks
        // (`is_supported_scheme`), so a network-denied inferlet still
        // instantiates but every request fails.
        wasmtime_wasi::p3::add_to_linker(&mut linker).expect("Failed to link WASI p3");
        wasmtime_wasi_http::p3::add_to_linker(&mut linker).expect("Failed to link WASI HTTP p3");

        // The wasm32-wasip3 rustc target's std imports the RC-versioned
        // `wasi:random/insecure-seed` (std HashMap seeding, function
        // `get-insecure-seed`) while wasmtime links the finalized
        // `wasi:random@0.3.0` (`insecure-seed`) — without this shim any
        // guest that touches a std HashMap fails to instantiate. A fixed
        // seed is fine: hash-DoS resistance is irrelevant for sandboxed
        // inferlets, and determinism helps reproducibility.
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

        // wasi:http operates above wasi:sockets and bypasses the per-socket
        // policy hook (it uses the host's hyper stack with its own DNS).
        // Drop the binding entirely when the network is disabled so the
        // policy is honored end-to-end. With allow_network=true the hook
        // is wired; the allowlist still applies to wasi:sockets but
        // wasi:http is unrestricted (pre-DNS hostname allowlisting would
        // require a DNS shim — not in v1). This is the p2 link, kept for
        // StarlingMonkey JS guests that still import wasi:http@0.2.
        if policy.network.allow {
            wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)
                .expect("Failed to link WASI HTTP");
        }

        host::add_to_linker(&mut linker)?;

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
            dynamic::instantiate_libraries(
                &engine,
                &mut linker,
                &mut store,
                dependency_components,
            )
            .await?;
        }

        // 7. Instantiate the main component
        let mut pre_hit = false;
        let instance = if cacheable_instance_pre {
            let cache_key = (program_name.clone(), main.generation);
            let cached = instance_pre_cache.lock().unwrap().get(&cache_key).cloned();
            let pre = match cached {
                Some(pre) => {
                    pre_hit = true;
                    pre
                }
                None => {
                    // Built OUTSIDE the lock (pre-linking is CPU work);
                    // concurrent misses build twice and last-insert wins —
                    // both values are equivalent, so no double-checked
                    // ceremony is needed.
                    let pre = linker
                        .instantiate_pre(&component)
                        .map_err(|e| anyhow!("Instantiation pre-link error: {e}"))?;
                    instance_pre_cache
                        .lock()
                        .unwrap()
                        .insert(cache_key, pre.clone());
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
        response: oneshot::Sender<Result<(Store<ProcessCtx>, Instance)>>,
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
                // Spawn, do not await (W3): the actor loop must never
                // serialize link work — a fleet entry queues hundreds of
                // instantiations at once and each is 2-3 ms of CPU. The
                // spawned tasks share the Engine and the InstancePre
                // cache; everything else they build is their own.
                let engine = self.engine.clone();
                let policy = self.policy.clone();
                let cache = Arc::clone(&self.instance_pre_cache);
                tokio::task::spawn(async move {
                    let link_started_us = crate::scheduler::fire_timing_enabled()
                        .then(crate::scheduler::fire_timing_now_us);
                    let result = Linker::instantiate(
                        engine,
                        policy,
                        cache,
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
        }
    }
}
