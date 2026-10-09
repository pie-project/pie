//! `@pie-project/server`: what `pie serve` boots, inside a Node process.
//!
//! [`worker::Server::serve`] brings up the worker and, beside it, the
//! controller and the gateway with its HTTP routes, from the same config file
//! the CLI reads. Its calls block, so each runs as a napi async task on the
//! libuv thread pool and reaches JavaScript as a promise. A handle
//! garbage-collected without `shutdown()` still tears the engine down, on a
//! thread of its own so the collector is not held.

use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::*;
use napi_derive::napi;

type Live = Arc<worker::Server>;

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

    /// Hand the runtime a language component (`python`, `javascript`) from
    /// bytes; a script inferlet in that language can run once this resolves.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn install_language(
        &self,
        language: String,
        component: Buffer,
    ) -> Result<AsyncTask<InstallLanguage>> {
        Ok(AsyncTask::new(InstallLanguage {
            server: self.live("install language")?,
            language,
            component: component.to_vec(),
        }))
    }

    #[napi(ts_return_type = "Promise<string>")]
    pub fn install(
        &self,
        bytes: Buffer,
        file: String,
        version: Option<String>,
    ) -> Result<AsyncTask<Install>> {
        Ok(AsyncTask::new(Install {
            server: self.live("install")?,
            bytes: bytes.to_vec(),
            file,
            version,
        }))
    }

    /// Stop every engine, join them, and release the runtime. Idempotent.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn shutdown(&self) -> AsyncTask<Shutdown> {
        AsyncTask::new(Shutdown {
            server: self.live.lock().unwrap().take(),
        })
    }

    fn live(&self, what: &str) -> Result<Live> {
        self.live
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| failed(what, "the server is shut down"))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(server) = self.live.lock().unwrap().take() {
            std::thread::spawn(move || server.shutdown());
        }
    }
}

pub struct InstallLanguage {
    server: Live,
    language: String,
    component: Vec<u8>,
}

#[napi]
impl Task for InstallLanguage {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        let name = self
            .server
            .install_language(&self.language, std::mem::take(&mut self.component))
            .map_err(|e| failed("install language", format!("{e:#}")))?;
        Ok(name.to_string())
    }

    fn resolve(&mut self, _env: Env, name: String) -> Result<String> {
        Ok(name)
    }
}

pub struct Install {
    server: Live,
    bytes: Vec<u8>,
    file: String,
    version: Option<String>,
}

#[napi]
impl Task for Install {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        self.server
            .install(
                std::mem::take(&mut self.bytes),
                &self.file,
                self.version.as_deref(),
            )
            .map_err(|e| failed("install", format!("{e:#}")))
    }

    fn resolve(&mut self, _env: Env, name: String) -> Result<String> {
        Ok(name)
    }
}

pub struct Shutdown {
    server: Option<Live>,
}

#[napi]
impl Task for Shutdown {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        if let Some(server) = self.server.take() {
            server.shutdown();
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
    let config =
        worker::Config::parse(toml_str).map_err(|e| invalid("config", format!("{e:#}")))?;
    let server = worker::Server::serve(config).map_err(|e| failed("start", format!("{e:#}")))?;
    let addr = server.listen_addr().expect("a served worker listens");
    Ok(Server {
        url: format!("ws://{addr}"),
        live: Mutex::new(Some(Arc::new(server))),
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
