//! [`Server`]: the runtime on a tokio runtime of its own, behind blocking
//! calls that are safe from any thread. What a native app's bindings wrap.

use std::path::Path;
use std::sync::RwLock;

use anyhow::{Result, anyhow, bail};
use tokio::sync::watch;

use super::{BootConfig, BootSummary, Embedded, Host};
use crate::bootstrap::BuiltinProgram;
use crate::engine::EngineBox;
use crate::inferlet::program;
use crate::server::ClientId;

pub struct Server {
    summary: BootSummary,
    live: RwLock<Option<Live>>,
    stop: watch::Sender<bool>,
}

struct Live {
    embedded: Embedded,
    runtime: tokio::runtime::Runtime,
}

impl Server {
    /// Boots `artifact` on this platform's engine (Metal on Apple, else
    /// Vulkan). `config` is a [`BootConfig`] as TOML or JSON. One per process.
    pub fn start(
        artifact: &Path,
        config: Option<&str>,
        host: Host,
        builtins: Vec<BuiltinProgram>,
    ) -> Result<Server> {
        let config = config.map_or_else(|| Ok(BootConfig::default()), BootConfig::parse)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(host.worker_threads)
            .thread_stack_size(8 << 20)
            .enable_all()
            .build()?;
        let model_name = artifact
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let engine = if config.engine {
            Some(open_engine(&config, &host.home)?)
        } else {
            None
        };
        let loaded = super::load(&config, artifact, &model_name, engine)?;
        let embedded = runtime.block_on(super::start(
            &config,
            &host,
            artifact.to_path_buf(),
            loaded,
            builtins,
        ))?;
        Ok(Server {
            summary: embedded.summary.clone(),
            live: RwLock::new(Some(Live { embedded, runtime })),
            stop: watch::channel(false).0,
        })
    }

    pub fn summary(&self) -> &BootSummary {
        &self.summary
    }

    /// Installs a program (`file` names it: `x.wasm`, `x.py`, ...); returns
    /// its `name@version`.
    pub fn install(&self, program: Vec<u8>, file: &str, version: Option<&str>) -> Result<String> {
        self.with(|live| {
            let name = live
                .runtime
                .block_on(program::add(program, file, version, true))?;
            Ok(name.to_string())
        })
    }

    /// Installs the component that runs `language` (`python`, `javascript`).
    pub fn install_language(&self, language: &str, component: Vec<u8>) -> Result<()> {
        let language = program::Language::parse(language)?;
        self.with(|live| {
            live.runtime
                .block_on(program::add_language(language, component))
        })
    }

    pub fn open_session(&self) -> Result<ClientId> {
        self.with(|live| {
            let _entered = live.runtime.enter();
            crate::server::open_session()
        })
    }

    pub fn close_session(&self, session: ClientId) {
        let _ = self.with(|live| {
            let _entered = live.runtime.enter();
            crate::server::close_session(session);
            Ok(())
        });
    }

    pub fn send_frame(&self, session: ClientId, frame: &[u8]) -> Result<()> {
        self.with(|live| {
            let _entered = live.runtime.enter();
            super::send_frame(session, frame)
        })
    }

    /// Up to `max` server frames, waiting at most `max_wait_ms` for the
    /// first, or less once the session closes or the server shuts down.
    pub fn recv_frames(
        &self,
        session: ClientId,
        max_wait_ms: u64,
        max: usize,
    ) -> Result<Vec<Vec<u8>>> {
        let mut stop = self.stop.subscribe();
        self.with(|live| {
            live.runtime.block_on(async {
                tokio::select! {
                    frames = super::recv_frames(session, max_wait_ms, max.max(1)) => frames,
                    _ = stop.wait_for(|stopping| *stopping) => Ok(Vec::new()),
                }
            })
        })
    }

    /// Wakes every waiting receive, then stops the runtime once no call is
    /// using it. Idempotent; every other call fails from then on.
    pub fn shutdown(&self) {
        self.stop.send_replace(true);
        let live = self.live.write().unwrap().take();
        if let Some(Live { embedded, runtime }) = live
            && let Err(error) = runtime.block_on(embedded.shutdown())
        {
            tracing::warn!("shutdown: {error:#}");
        }
    }

    fn with<T>(&self, f: impl FnOnce(&Live) -> Result<T>) -> Result<T> {
        let live = self.live.read().unwrap();
        f(live
            .as_ref()
            .ok_or_else(|| anyhow!("the server is shut down"))?)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[allow(unreachable_code, unused_variables)]
fn open_engine(config: &BootConfig, home: &Path) -> Result<(EngineBox, poem_ir::Platform)> {
    let mut engine = toml::Table::new();
    engine.insert(
        "gpu_mem_utilization".into(),
        f64::from(config.gpu_mem_utilization).into(),
    );
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        let doc = toml::Table::from_iter([("metal".to_string(), engine.into())]).to_string();
        return Ok((
            crate::engine::backend::open::metal(doc.as_bytes())?,
            poem_ir::Platform::Metal,
        ));
    }
    #[cfg(feature = "vulkan")]
    {
        let cache = home.join("vulkan-pipelines.bin");
        engine.insert(
            "pipeline_cache".into(),
            cache.to_string_lossy().into_owned().into(),
        );
        let doc = toml::Table::from_iter([("vulkan".to_string(), engine.into())]).to_string();
        return Ok((
            crate::engine::backend::open::vulkan(doc.as_bytes())?,
            poem_ir::Platform::Vulkan,
        ));
    }
    bail!("this build has no native engine (the `metal` or `vulkan` feature)")
}
