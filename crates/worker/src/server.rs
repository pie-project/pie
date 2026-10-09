//! [`Server`]: the worker in a native host's process (an app, a Node or
//! Python process), on a tokio runtime of its own, behind blocking calls that
//! are safe from any thread. With `listen` the gateway serves it too, the way
//! `pie serve` does.

use std::net::SocketAddr;
use std::sync::RwLock;

use anyhow::{Result, anyhow};
use tokio::sync::watch;

use crate::config::Config;
use crate::embedded::{Embedded, Engine, Summary};

/// How a [`Server`] boots.
pub struct Options {
    pub engine: Engine,
    /// Registers the built-in inferlets (`compat-openai`, ...).
    pub builtins: bool,
    /// Also serves the gateway (its WebSocket and HTTP routes) here.
    pub listen: Option<SocketAddr>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            engine: Engine::Configured,
            builtins: true,
            listen: None,
        }
    }
}

pub struct Server {
    summary: Summary,
    listen_addr: Option<SocketAddr>,
    live: RwLock<Option<Live>>,
    stop: watch::Sender<bool>,
}

struct Live {
    serving: Serving,
    runtime: tokio::runtime::Runtime,
}

enum Serving {
    Embedded(Embedded),
    #[cfg(feature = "standalone")]
    Standalone(Box<crate::standalone::StandaloneHandle>),
}

impl Server {
    /// Boots `config`. One per process.
    pub fn start(mut config: Config, options: Options) -> Result<Server> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.server.worker_threads)
            .thread_stack_size(8 << 20)
            .enable_all()
            .build()?;
        let (serving, summary, listen_addr) = match options.listen {
            None => {
                let builtins = if options.builtins {
                    crate::translate::builtins()
                } else {
                    Vec::new()
                };
                let embedded =
                    runtime.block_on(Embedded::start(&config, options.engine, builtins))?;
                let summary = embedded.summary.clone();
                (Serving::Embedded(embedded), summary, None)
            }
            #[cfg(feature = "standalone")]
            Some(listen) => {
                anyhow::ensure!(
                    matches!(options.engine, Engine::Configured) && options.builtins,
                    "a gateway serves the configured engine, with the built-in inferlets"
                );
                config.server.host = listen.ip().to_string();
                config.server.port = listen.port();
                let controller = controller::Config::parse("")?;
                let gateway = gateway::Config::parse("")?;
                let handle = runtime.block_on(crate::standalone::run_standalone(
                    controller, gateway, config,
                ))?;
                let summary = handle.summary().clone();
                let addr = handle.listen_addr;
                (Serving::Standalone(Box::new(handle)), summary, Some(addr))
            }
            #[cfg(not(feature = "standalone"))]
            Some(_) => {
                let _ = &mut config;
                anyhow::bail!("this build serves no gateway (the worker's `standalone` feature)")
            }
        };
        Ok(Server {
            summary,
            listen_addr,
            live: RwLock::new(Some(Live { serving, runtime })),
            stop: watch::channel(false).0,
        })
    }

    /// Boots `config` the way `pie serve` does: the gateway at its
    /// `[server] host:port`.
    #[cfg(feature = "standalone")]
    pub fn serve(config: Config) -> Result<Server> {
        let host: std::net::IpAddr = config.server.host.parse().map_err(|_| {
            anyhow!(
                "[server] host {:?} is not an IP address",
                config.server.host
            )
        })?;
        let listen = SocketAddr::new(host, config.server.port);
        Server::start(
            config,
            Options {
                listen: Some(listen),
                ..Options::default()
            },
        )
    }

    /// True until [`Server::shutdown`].
    pub fn running(&self) -> bool {
        self.live.read().unwrap().is_some()
    }

    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    /// Where the gateway listens, with `listen`; port 0 resolves to the OS's.
    pub fn listen_addr(&self) -> Option<SocketAddr> {
        self.listen_addr
    }

    /// Installs a program (`file` names it: `x.wasm`, `x.py`, ...); returns
    /// its `name@version`.
    pub fn install(&self, program: Vec<u8>, file: &str, version: Option<&str>) -> Result<String> {
        self.with(|live| {
            let name = live.runtime.block_on(runtime::inferlet::program::add(
                program, file, version, true,
            ))?;
            Ok(name.to_string())
        })
    }

    /// Installs the component that runs `language` (`python`, `javascript`).
    pub fn install_language(&self, language: &str, component: Vec<u8>) -> Result<()> {
        let language = runtime::inferlet::program::Language::parse(language)?;
        self.with(|live| {
            live.runtime
                .block_on(runtime::inferlet::program::add_language(
                    language, component,
                ))
        })
    }

    pub fn open_session(&self) -> Result<u32> {
        self.with(|live| {
            let _entered = live.runtime.enter();
            runtime::server::open_session()
        })
    }

    pub fn close_session(&self, session: u32) {
        let _ = self.with(|live| {
            let _entered = live.runtime.enter();
            runtime::server::close_session(session);
            Ok(())
        });
    }

    pub fn send_frame(&self, session: u32, frame: &[u8]) -> Result<()> {
        self.with(|live| {
            let _entered = live.runtime.enter();
            crate::embedded::send_frame(session, frame)
        })
    }

    /// Up to `max` server frames, waiting at most `max_wait_ms` for the
    /// first, or less once the session closes or the server shuts down.
    pub fn recv_frames(&self, session: u32, max_wait_ms: u64, max: usize) -> Result<Vec<Vec<u8>>> {
        let mut stop = self.stop.subscribe();
        self.with(|live| {
            live.runtime.block_on(async {
                tokio::select! {
                    frames = crate::embedded::recv_frames(session, max_wait_ms, max.max(1)) => frames,
                    _ = stop.wait_for(|stopping| *stopping) => Ok(Vec::new()),
                }
            })
        })
    }

    /// Wakes every waiting receive, then stops the runtime once no call is
    /// using it. Idempotent; every other call fails from then on.
    pub fn shutdown(&self) {
        self.stop.send_replace(true);
        let Some(Live { serving, runtime }) = self.live.write().unwrap().take() else {
            return;
        };
        runtime.block_on(async {
            match serving {
                Serving::Embedded(embedded) => {
                    if let Err(error) = embedded.shutdown().await {
                        tracing::warn!("shutdown: {error:#}");
                    }
                }
                #[cfg(feature = "standalone")]
                Serving::Standalone(handle) => handle.shutdown().await,
            }
        });
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
