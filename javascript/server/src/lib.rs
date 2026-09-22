//! `@pie-project/server`: what `pie serve` boots, inside a Node process.
//!
//! [`pie::run_standalone`] brings up the controller, the gateway with its
//! HTTP routes and the worker from the same config file the CLI reads. The
//! boot and the shutdown block, so both run as napi async tasks on the
//! libuv thread pool and reach JavaScript as promises. A handle
//! garbage-collected without `shutdown()` still tears the engine down, on a
//! thread of its own so the collector is not held; what that thread has not
//! released by process exit, the OS reclaims.

use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use pie::StandaloneHandle;

/// The runtime outlives the handle's shutdown: every engine is joined on it.
type Live = (StandaloneHandle, tokio::runtime::Runtime);

fn stop((handle, runtime): Live) {
    runtime.block_on(handle.shutdown());
    drop(runtime);
}

#[napi]
pub struct Server {
    url: String,
    live: Mutex<Option<Live>>,
}

#[napi]
impl Server {
    /// `ws://host:port` the gateway listens on; its HTTP routes share the
    /// port. With `server.port = 0` this is the port the OS handed out.
    #[napi(getter)]
    pub fn url(&self) -> String {
        self.url.clone()
    }

    /// True until `shutdown()` resolves.
    #[napi(getter)]
    pub fn running(&self) -> bool {
        self.live.lock().unwrap().is_some()
    }

    /// Stop every engine, join them, and release the runtime. Idempotent.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn shutdown(&self) -> AsyncTask<Shutdown> {
        AsyncTask::new(Shutdown {
            live: self.live.lock().unwrap().take(),
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(live) = self.live.lock().unwrap().take() {
            std::thread::spawn(move || stop(live));
        }
    }
}

pub struct Shutdown {
    live: Option<Live>,
}

#[napi]
impl Task for Shutdown {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        if let Some(live) = self.live.take() {
            stop(live);
        }
        Ok(())
    }

    fn resolve(&mut self, _env: Env, _output: ()) -> Result<()> {
        Ok(())
    }
}

pub struct Boot {
    toml: String,
}

#[napi]
impl Task for Boot {
    type Output = Server;
    type JsValue = Server;

    fn compute(&mut self) -> Result<Server> {
        boot(&self.toml)
    }

    fn resolve(&mut self, _env: Env, server: Server) -> Result<Server> {
        Ok(server)
    }
}

/// Level from `RUST_LOG` (default `warn`), stderr as the writer; a host
/// that already installed a subscriber keeps it.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn invalid(what: &str, error: impl std::fmt::Display) -> Error {
    Error::new(Status::InvalidArg, format!("{what}: {error}"))
}

fn failed(what: &str, error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("{what}: {error}"))
}

fn boot(toml_str: &str) -> Result<Server> {
    init_tracing();
    let (controller, gateway, worker) = pie::derive::derive_standalone(toml_str)
        .map_err(|e| invalid("config", format!("{e:#}")))?;
    let runtime = worker::serve::build_runtime(&worker)
        .map_err(|e| failed("build tokio runtime", format!("{e:#}")))?;
    let handle = runtime
        .block_on(pie::run_standalone(controller, gateway, worker))
        .map_err(|e| failed("start", format!("{e:#}")))?;
    Ok(Server {
        url: format!("ws://{}", handle.listen_addr),
        live: Mutex::new(Some((handle, runtime))),
    })
}

/// Boot from a config in the shape of `pie serve`'s config file
/// (`{server: {port: 0}, model: {...}}`). Resolves once engines are up,
/// weights are loaded and the listener is bound.
#[napi(ts_return_type = "Promise<Server>")]
pub fn start(config: serde_json::Value) -> Result<AsyncTask<Boot>> {
    if !config.is_object() {
        return Err(invalid("config", "must be an object (or use startToml)"));
    }
    let toml = toml::to_string(&config).map_err(|e| invalid("config", e))?;
    Ok(AsyncTask::new(Boot { toml }))
}

/// Boot from the TOML text `pie serve --config` reads.
#[napi(ts_return_type = "Promise<Server>")]
pub fn start_toml(config: String) -> AsyncTask<Boot> {
    AsyncTask::new(Boot { toml: config })
}
