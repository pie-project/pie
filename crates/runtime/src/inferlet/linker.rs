use std::collections::{HashMap, hash_map::Entry};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Result, anyhow};
use tokio::sync::{OnceCell, oneshot};
use wasmtime::component::{Component, Instance, InstancePre, Linker as WasmLinker};
use wasmtime::{Engine, Store};

use crate::inferlet::host;
use crate::service::{Service, ServiceHandler};

use super::process::{OutputMode, ProcessCtx, ProcessId};
use super::program::{self, ProgramName};
use super::sandbox::{FsPolicy, InstancePolicy, NetworkPolicy};

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

pub fn spawn(engine: &Engine, fs: FsPolicy, network: NetworkPolicy) {
    let policy = InstancePolicy { fs, network };
    SERVICE
        .spawn(|| Linker::new(engine, policy))
        .expect("linker already spawned");
}

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

type InstancePreKey = (ProgramName, u64);
type InstancePreCell = Arc<OnceCell<InstancePre<ProcessCtx>>>;
type InstancePreCache = Arc<Mutex<HashMap<InstancePreKey, InstancePreCell>>>;
type BaseLinkerCache = Arc<OnceCell<Arc<WasmLinker<ProcessCtx>>>>;

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
            base_linker_cache: Arc::new(OnceCell::new()),
            instance_pre_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn build_base_linker(
        engine: &Engine,
        policy: &InstancePolicy,
    ) -> Result<WasmLinker<ProcessCtx>> {
        let mut linker = WasmLinker::<ProcessCtx>::new(engine);
        #[cfg(target_arch = "wasm32")]
        let _ = policy;

        #[cfg(target_arch = "wasm32")]
        wasmtime_wasi::add_to_linker(&mut linker).expect("Failed to link WASI");

        #[cfg(not(target_arch = "wasm32"))]
        wasmtime_wasi::p2::add_to_linker_async(&mut linker).expect("Failed to link WASI");
        #[cfg(not(target_arch = "wasm32"))]
        wasmtime_wasi::p3::add_to_linker(&mut linker).expect("Failed to link WASI p3");
        #[cfg(not(target_arch = "wasm32"))]
        wasmtime_wasi_http::p3::add_to_linker(&mut linker).expect("Failed to link WASI HTTP p3");

        #[cfg(not(target_arch = "wasm32"))]
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

        #[cfg(not(target_arch = "wasm32"))]
        if policy.network.allow {
            wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)
                .expect("Failed to link WASI HTTP");
        }

        host::add_to_linker(&mut linker)?;

        Ok(linker)
    }

    async fn base_linker(
        engine: &Engine,
        policy: &InstancePolicy,
        cache: &BaseLinkerCache,
    ) -> Result<Arc<WasmLinker<ProcessCtx>>> {
        let linker = cache
            .get_or_try_init(|| async { Self::build_base_linker(engine, policy).map(Arc::new) })
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

        let generation = main.generation;
        let component = main.component;

        let process_ctx =
            ProcessCtx::new(process_id, username, output, &policy, main.script).await?;
        let mut store = Store::new(&engine, process_ctx);

        let base_linker = Self::base_linker(&engine, &policy, &base_linker_cache).await?;

        let (cell, _cache_hit) =
            Self::instance_pre_cell(&instance_pre_cache, program_name, generation);
        let pre = cell
            .get_or_try_init(|| async {
                Self::linker_for(&engine, &base_linker, &component)?
                    .instantiate_pre(&component)
                    .map_err(|error| anyhow!("Instantiation pre-link error: {error}"))
            })
            .await?;
        let instance = pre
            .instantiate_async(&mut store)
            .await
            .map_err(|e| anyhow!("Instantiation error: {e}"))?;
        Ok((store, instance))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn linker_for<'a>(
        _engine: &Engine,
        linker: &'a WasmLinker<ProcessCtx>,
        _component: &Component,
    ) -> Result<std::borrow::Cow<'a, WasmLinker<ProcessCtx>>> {
        Ok(std::borrow::Cow::Borrowed(linker))
    }

    #[cfg(target_arch = "wasm32")]
    fn linker_for<'a>(
        engine: &Engine,
        linker: &'a WasmLinker<ProcessCtx>,
        component: &Component,
    ) -> Result<std::borrow::Cow<'a, WasmLinker<ProcessCtx>>> {
        let mut linker = linker.clone();
        wasmtime_wasi::stub_unhosted(&mut linker, engine, component)?;
        Ok(std::borrow::Cow::Owned(linker))
    }
}

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
                let engine = self.engine.clone();
                let policy = self.policy.clone();
                let base_cache = Arc::clone(&self.base_linker_cache);
                let pre_cache = Arc::clone(&self.instance_pre_cache);
                crate::rt::spawn(async move {
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
