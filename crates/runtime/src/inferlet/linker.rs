use std::collections::{HashMap, hash_map::Entry};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Result, anyhow};
use tokio::sync::{OnceCell, oneshot};
use wasmtime::component::{Component, Instance, InstancePre, Linker as WasmLinker};
use wasmtime::{Engine, Store};

use crate::inferlet::host;
use crate::service::{Service, ServiceHandler};

use super::process::{OutputMode, ProcessCtx, ProcessId};
use super::program::{self, Program, ProgramHash};
use super::sandbox::{FsPolicy, InstancePolicy, NetworkPolicy};

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

pub fn spawn(engine: &Engine, fs: FsPolicy, network: NetworkPolicy, memory_limit: usize) {
    let policy = InstancePolicy {
        fs,
        network,
        memory_limit,
    };
    SERVICE
        .spawn(|| Linker::new(engine, policy))
        .expect("linker already spawned");
}

pub async fn instantiate(
    process_id: ProcessId,
    program: &Program,
    output: OutputMode,
) -> Result<(Store<ProcessCtx>, Instance)> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        process_id,
        program: program.clone(),
        output,
        response: tx,
    })?;
    rx.await?
}

/// Drop what is linked of these builds: nothing runs them by name any more.
pub(crate) fn forget(hashes: Vec<ProgramHash>) {
    if !hashes.is_empty() {
        let _ = SERVICE.send(Message::Forget { hashes });
    }
}

type InstancePreCell = Arc<OnceCell<InstancePre<ProcessCtx>>>;
/// By the hash of the program linked: its bytes never change under it.
type InstancePreCache = Arc<Mutex<HashMap<ProgramHash, InstancePreCell>>>;
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

    /// WASI 0.3 and pie's own interfaces: what every inferlet imports.
    fn build_base_linker(engine: &Engine) -> Result<WasmLinker<ProcessCtx>> {
        let mut linker = WasmLinker::<ProcessCtx>::new(engine);
        wasmtime_wasi::p3::add_to_linker(&mut linker).expect("Failed to link WASI");
        #[cfg(not(target_arch = "wasm32"))]
        wasmtime_wasi_http::p3::add_to_linker(&mut linker).expect("Failed to link WASI HTTP");
        host::add_to_linker(&mut linker)?;
        Ok(linker)
    }

    async fn base_linker(
        engine: &Engine,
        cache: &BaseLinkerCache,
    ) -> Result<Arc<WasmLinker<ProcessCtx>>> {
        let linker = cache
            .get_or_try_init(|| async { Self::build_base_linker(engine).map(Arc::new) })
            .await?;
        Ok(Arc::clone(linker))
    }

    /// componentize-js 0.23 and componentize-py 0.25 still import WASI 0.2
    /// (cli, io and random; JavaScript also clocks, filesystem and http): a
    /// language component gets it beside 0.3. Delete with them.
    fn with_wasi_0_2(
        linker: &WasmLinker<ProcessCtx>,
        policy: &InstancePolicy,
    ) -> Result<WasmLinker<ProcessCtx>> {
        let mut linker = linker.clone();
        #[cfg(not(target_arch = "wasm32"))]
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        #[cfg(target_arch = "wasm32")]
        wasmtime_wasi::p2::add_to_linker(&mut linker)?;
        #[cfg(not(target_arch = "wasm32"))]
        if policy.network.allow {
            wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)?;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = policy;
        Ok(linker)
    }

    fn instance_pre_cell(cache: &InstancePreCache, hash: &ProgramHash) -> InstancePreCell {
        let mut cache = cache.lock().unwrap();
        match cache.entry(hash.clone()) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => Arc::clone(entry.insert(Arc::new(OnceCell::new()))),
        }
    }

    async fn instantiate(
        engine: Engine,
        policy: InstancePolicy,
        base_linker_cache: BaseLinkerCache,
        instance_pre_cache: InstancePreCache,
        process_id: ProcessId,
        program: &Program,
        output: OutputMode,
    ) -> Result<(Store<ProcessCtx>, Instance)> {
        let main = program::get_wasm_component(program)
            .await
            .ok_or_else(|| anyhow!("{program} is not loaded"))?;

        let component = main.component;
        let language = main.script.is_some();

        let process_ctx =
            ProcessCtx::new(process_id, output, &policy, main.script).await?;
        let mut store = Store::new(&engine, process_ctx);
        store.limiter(|ctx| ctx.limits());

        let base_linker = Self::base_linker(&engine, &base_linker_cache).await?;

        let cell = Self::instance_pre_cell(&instance_pre_cache, &program.hash);
        let pre = cell
            .get_or_try_init(|| async {
                Self::linker_for(&engine, &base_linker, &component, language, &policy)?
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
        language: bool,
        policy: &InstancePolicy,
    ) -> Result<std::borrow::Cow<'a, WasmLinker<ProcessCtx>>> {
        Ok(if language {
            std::borrow::Cow::Owned(Self::with_wasi_0_2(linker, policy)?)
        } else {
            std::borrow::Cow::Borrowed(linker)
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn linker_for<'a>(
        engine: &Engine,
        linker: &'a WasmLinker<ProcessCtx>,
        component: &Component,
        language: bool,
        policy: &InstancePolicy,
    ) -> Result<std::borrow::Cow<'a, WasmLinker<ProcessCtx>>> {
        let mut linker = if language {
            Self::with_wasi_0_2(linker, policy)?
        } else {
            linker.clone()
        };
        wasmtime_wasi::stub_unhosted(&mut linker, engine, component)?;
        Ok(std::borrow::Cow::Owned(linker))
    }
}

enum Message {
    Instantiate {
        process_id: ProcessId,
        program: Program,
        output: OutputMode,
        response: oneshot::Sender<Result<(Store<ProcessCtx>, Instance)>>,
    },
    Forget {
        hashes: Vec<ProgramHash>,
    },
}

impl ServiceHandler for Linker {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Instantiate {
                process_id,
                program,
                output,
                response,
            } => {
                let engine = self.engine.clone();
                let policy = self.policy.clone();
                let base_cache = Arc::clone(&self.base_linker_cache);
                let pre_cache = Arc::clone(&self.instance_pre_cache);
                crate::rt::spawn(async move {
                    let result = Linker::instantiate(
                        engine, policy, base_cache, pre_cache, process_id, &program,
                        output,
                    )
                    .await;
                    let _ = response.send(result);
                });
            }
            Message::Forget { hashes } => {
                let mut cache = self.instance_pre_cache.lock().unwrap();
                for hash in &hashes {
                    cache.remove(hash);
                }
            }
        }
    }
}
